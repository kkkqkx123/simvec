//! Collection compaction plus the file-rewriting helpers it relies on.
//!
//! Compaction rewrites every collection file into temp files, fsyncs them,
//! then atomically renames each over the live file (the `replace_from`
//! path), which is invisible to readers holding old mmap snapshots.
//!
//! The heavy rewrite phase runs under the `compact_mutex` only — the
//! store write lock is *not* held, so searches and upserts proceed on the
//! old files. A mutation counter guards the commit: the swap happens under
//! the write lock only if no transaction raced the rewrite; otherwise the
//! attempt is retried on a fresh snapshot. Under sustained write pressure
//! the compaction falls back to rewriting while holding the write lock, so
//! progress is always guaranteed. In both paths the `compact_mutex`
//! excludes index builds/drains, keeping slot numbers stable for the index.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::Arc;

use memmap2::Mmap;

use super::directory::KEY_REC_SIZE;
use super::keys::Keys;
use super::tombstones::TombstoneBits;
use super::{CollectionStore, Meta, WalRecord, WalTxn};
use crate::error::{Result, VectorSearchError};
use crate::types::PointId;

const HEADER_LEN: usize = 24;
const FILE_VERSION: u32 = 1;

/// How often the lock-free rewrite is retried when a write races it before
/// falling back to the whole-rewrite critical section.
const COMMIT_ATTEMPTS: usize = 4;

/// Everything one compaction attempt needs, captured from an immutable view.
struct CompactionPlan {
    dim: usize,
    segment_slots: u32,
    /// Slot high-water mark the plan covers.
    next_slot: u64,
    /// Tombstone count at planning time (commit validates it is unchanged).
    tombstone_count: u64,
    new_capacity: u64,
    /// `map[old_slot] = new_slot` for live slots; `u32::MAX` = dropped.
    map: Vec<u32>,
    live_count: u64,
}

/// Compute the plan from a metadata snapshot and the current tombstone bits.
fn build_plan(meta: &Meta, tombstones: &TombstoneBits) -> CompactionPlan {
    let (new_capacity, map) =
        plan_slots(|s| !tombstones.bit(s), meta.next_slot, meta.segment_slots);
    let live_count = map.iter().filter(|s| **s != u32::MAX).count() as u64;
    CompactionPlan {
        dim: meta.vector_size,
        segment_slots: meta.segment_slots,
        next_slot: meta.next_slot,
        tombstone_count: meta.tombstone_count,
        new_capacity,
        map,
        live_count,
    }
}

/// One entry of a rebuilt directory file: blob for `slot` with `flags`.
pub(crate) struct DirEntry {
    pub slot: u32,
    pub blob: Vec<u8>,
    pub flags: u8,
}

/// Compute the old-slot -> new-slot mapping for all live slots.
///
/// `is_live(slot)` reports whether the slot survives compaction. Returns the
/// new capacity (segment-aligned, at least one segment) and the mapping array:
/// `map[old_slot] = new_slot` for live slots.
pub(crate) fn plan_slots(
    mut is_live: impl FnMut(usize) -> bool,
    next_slot: u64,
    segment_slots: u32,
) -> (u64, Vec<u32>) {
    let mut map = vec![u32::MAX; next_slot as usize];
    let mut live = 0u32;
    for (slot, entry) in map.iter_mut().enumerate() {
        if is_live(slot) {
            *entry = live;
            live += 1;
        }
    }
    let new_capacity = (live as u64).max(1).div_ceil(segment_slots as u64) * segment_slots as u64;
    (new_capacity, map)
}

/// Write a complete directory file (header + record array + blob area) to
/// `path`, then fsync.
pub(crate) fn write_dir_file(
    path: &Path,
    magic: [u8; 4],
    rec_size: usize,
    rec_capacity: u64,
    entries: &[DirEntry],
) -> Result<()> {
    let blob_len: u64 = entries.iter().map(|e| e.blob.len() as u64).sum();

    let mut file = File::create(path)?;
    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(&magic);
    header.extend_from_slice(&FILE_VERSION.to_le_bytes());
    header.extend_from_slice(&rec_capacity.to_le_bytes());
    header.extend_from_slice(&blob_len.to_le_bytes());
    file.write_all(&header)?;

    let rec_array_len = rec_capacity as usize * rec_size;
    file.write_all(&vec![0u8; rec_array_len])?;

    let mut blob_offset = 0u32;
    for e in entries {
        let rec_start = HEADER_LEN + e.slot as usize * rec_size;
        let mut rec = Vec::with_capacity(rec_size);
        rec.extend_from_slice(&blob_offset.to_le_bytes());
        rec.extend_from_slice(&(e.blob.len() as u32).to_le_bytes());
        if rec_size >= 12 {
            rec.push(e.flags);
            rec.extend_from_slice(&[0u8; 3]);
        }
        rec.truncate(rec_size);
        write_at(&mut file, &rec, rec_start as u64)?;
        if !e.blob.is_empty() {
            let blob_start = HEADER_LEN + rec_array_len + blob_offset as usize;
            write_at(&mut file, &e.blob, blob_start as u64)?;
        }
        blob_offset += e.blob.len() as u32;
    }
    file.sync_all()?;
    Ok(())
}

/// Build a fresh dense `vectors.bin` at `path` (temp file) containing the
/// vectors of the live slots in new-slot order.
pub(crate) fn write_vectors_file(
    path: &Path,
    dim: usize,
    segment_slots: u32,
    new_capacity: u64,
    old_vectors: &[Arc<Mmap>],
    map: &[u32],
) -> Result<()> {
    let total = new_capacity as usize * dim * 4;
    let mut file = File::create(path)?;
    file.set_len(total as u64)?;
    file.sync_all()?;

    for (old_slot, new_slot) in map.iter().enumerate() {
        if *new_slot == u32::MAX {
            continue;
        }
        let seg_idx = old_slot / segment_slots as usize;
        let in_seg = old_slot % segment_slots as usize;
        let seg = old_vectors.get(seg_idx).ok_or_else(|| {
            VectorSearchError::CorruptData(format!("slot {old_slot} out of vectors.bin range"))
        })?;
        let offset = in_seg * dim * 4;
        let end = offset + dim * 4;
        if end > seg.len() {
            return Err(VectorSearchError::CorruptData(format!(
                "slot {old_slot} out of vectors.bin range"
            )));
        }
        let bytes = &seg[offset..end];
        let dst = *new_slot as usize * dim * 4;
        if dst + bytes.len() > total {
            return Err(VectorSearchError::Internal(
                "compaction vector write out of bounds".to_string(),
            ));
        }
        write_at(&mut file, bytes, dst as u64)?;
    }
    file.sync_all()?;
    Ok(())
}

#[cfg(unix)]
fn write_at(file: &mut File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(not(unix))]
fn write_at(file: &mut File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(buf)
}

impl CollectionStore {
    /// Physically remove tombstoned slots and rebuild all files with compacted
    /// slot numbering `0..live_count`.
    ///
    /// Holds the `compact_mutex` throughout, so an index build cannot run
    /// concurrently and observe torn slot numbers. The temp-file rewrite runs
    /// without the store write lock; the commit (rename swap + in-memory
    /// rebuild + meta rewrite + WAL checkpoint) is the only section holding
    /// it, and only when no transaction raced the rewrite. Under sustained
    /// write pressure a bounded fallback rewrites while holding the write
    /// lock, so compaction always makes progress.
    ///
    /// Returns the number of live points after compaction.
    pub fn compact(&self) -> Result<u64> {
        let _compact_guard = self.compact_mutex.lock();
        // Hold build_mutex jointly with compact_mutex (order: compact -> build)
        // so quantization rebuilds and index builds cannot interleave slot
        // renumbering. Dropped between attempts to avoid stalling builds longer
        // than one rewrite phase when a race is detected.
        let started = std::time::Instant::now();

        // Lock-free attempts: hold build_mutex together with compact_mutex
        // (order: compact -> build) so index/quant builds cannot interleave
        // slot renumbering. The build guard is scoped per attempt so a race
        // retry does not stall builds for the whole compaction.
        for _ in 0..COMMIT_ATTEMPTS {
            let _build_guard = self.build_mutex.lock();
            let plan = {
                let inner = self.inner.read();
                if inner.meta.tombstone_count == 0 || inner.meta.next_slot == 0 {
                    return Ok(inner.meta.live_count);
                }
                let tombstones = self.tombstones.load();
                build_plan(&inner.meta, &tombstones)
            };
            let version = self.mutations.load(AtomicOrdering::Relaxed);
            self.write_temp_files(&plan)?;
            {
                let mut inner = self.inner.write();
                let raced = self.mutations.load(AtomicOrdering::Relaxed) != version
                    || inner.meta.next_slot != plan.next_slot
                    || inner.meta.tombstone_count != plan.tombstone_count;
                if !raced {
                    let live = self.commit_compaction(&mut inner, &plan)?;
                    self.metrics.record_compaction(started.elapsed());
                    return Ok(live);
                }
            }
            self.metrics.record_compaction_race_retry();
            tracing::debug!(
                collection = %self.inner.read().meta.collection,
                "compaction raced concurrent writes; retrying"
            );
        }

        // Contended fallback: the whole rewrite inside the store write lock,
        // still holding both compact+build guards so no build races it.
        self.metrics.record_compaction_contended();
        let _build_guard = self.build_mutex.lock();
        let mut inner = self.inner.write();
        if inner.meta.tombstone_count == 0 || inner.meta.next_slot == 0 {
            return Ok(inner.meta.live_count);
        }
        let tombstones = self.tombstones.load();
        let plan = build_plan(&inner.meta, &tombstones);
        drop(tombstones);
        self.write_temp_files(&plan)?;
        let live = self.commit_compaction(&mut inner, &plan);
        self.metrics.record_compaction(started.elapsed());
        live
    }

    /// Write the three compacted temp files from immutable snapshots.
    ///
    /// Requires no store locks: every input is either an `ArcSwap` snapshot
    /// or captured in `plan`. Callers must validate via the mutation counter
    /// that no write raced the reads before committing these files.
    fn write_temp_files(&self, plan: &CompactionPlan) -> Result<()> {
        // 1. vectors.bin
        let tmp_vectors = self.dir.join("vectors_tmp.bin");
        {
            let vsnap = self.vectors.snapshot();
            write_vectors_file(
                &tmp_vectors,
                plan.dim,
                plan.segment_slots,
                plan.new_capacity,
                &vsnap,
                &plan.map,
            )?;
        }

        // 1b. quant.bin (derived, remapped exactly like vectors)
        if let Some(q) = self.quant.read().as_ref() {
            let tmp_quant = self.dir.join("quant_tmp.bin");
            let vsnap = self.vectors.snapshot();
            q.write_compacted_file(&tmp_quant, plan.new_capacity, &vsnap, &plan.map)?;
        }

        // 2. keys.bin
        let tmp_keys = self.dir.join("keys_tmp.bin");
        {
            let keys_view = self.keys.snapshot();
            let mut entries = Vec::with_capacity(plan.live_count as usize);
            for (old_slot, new_slot) in plan.map.iter().enumerate() {
                if *new_slot == u32::MAX {
                    continue;
                }
                let key = Keys::read_key(&keys_view, old_slot)?.ok_or_else(|| {
                    VectorSearchError::CorruptData(format!("live slot {old_slot} has no key"))
                })?;
                entries.push(DirEntry {
                    slot: *new_slot,
                    blob: key.into_bytes(),
                    flags: 0,
                });
            }
            write_dir_file(
                &tmp_keys,
                *b"VKEY",
                KEY_REC_SIZE,
                plan.new_capacity,
                &entries,
            )?;
        }

        let tmp_payload_store = self.dir.join("payloads_store_tmp");
        if tmp_payload_store.exists() {
            std::fs::remove_dir_all(&tmp_payload_store)?;
        }
        let live_slots: Vec<(u32, u32)> = plan
            .map
            .iter()
            .enumerate()
            .filter(|(_, new_slot)| **new_slot != u32::MAX)
            .map(|(old, &new)| (old as u32, new))
            .collect();
        let ps_guard = self.payload_store.load();
        let old_store = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        let new_store = old_store.compact_to(&tmp_payload_store, &live_slots)?;
        let _ = old_store;
        drop(ps_guard);
        let tracker_guard = new_store.tracker().write();
        let config = new_store.config().clone();
        tracker_guard.save(&tmp_payload_store, &config)?;
        Ok(())
    }

    /// Commit a validated plan: swap the files, rebuild the in-memory state,
    /// rewrite meta, invalidate the published ANN index and checkpoint the
    /// WAL. The caller must hold the store write lock and guarantee (mutation
    /// counter / lock ownership) that the plan still describes reality.
    fn commit_compaction(
        &self,
        inner: &mut super::StoreInner,
        plan: &CompactionPlan,
    ) -> Result<u64> {
        // Swap the files over their live names; readers keep viewing old
        // snapshots until each ArcSwap store below.
        self.vectors
            .replace_from(&self.dir.join("vectors_tmp.bin"))?;
        if let Some(q) = self.quant.read().as_ref() {
            // quant_tmp.bin only exists when quantization is active; its
            // remapping mirrors the vectors slot map so the quant slot numbers
            // stay consistent with the live vectors after compaction.
            let tmp_quant = self.dir.join("quant_tmp.bin");
            if tmp_quant.exists() {
                q.replace_from(&tmp_quant, plan.new_capacity, &plan.map)?;
            }
        }
        self.keys.replace_from(&self.dir.join("keys_tmp.bin"))?;

        {
            let live_store = self.dir.join("payloads_store");
            let tmp_store = self.dir.join("payloads_store_tmp");
            if tmp_store.exists() {
                if live_store.exists() {
                    std::fs::remove_dir_all(&live_store)?;
                }
                std::fs::rename(&tmp_store, &live_store)?;
                let new_ps = super::PayloadStore::open(&live_store)?;
                self.payload_store.store(Arc::new(Some(new_ps)));
            }
        }

        // In-memory rebuild against the new files.
        self.tombstones
            .store(Arc::new(TombstoneBits::new(plan.new_capacity as usize)));
        inner.reverse.clear();
        {
            let keys_view = self.keys.snapshot();
            for slot in 0..plan.live_count as usize {
                let key = Keys::read_key(&keys_view, slot)?.ok_or_else(|| {
                    VectorSearchError::CorruptData(format!("live slot {slot} has no key"))
                })?;
                inner.reverse.insert(PointId::from(key), slot as u32);
            }
        }
        inner.meta.slot_capacity = plan.new_capacity;
        inner.meta.next_slot = plan.live_count;
        inner.meta.live_count = plan.live_count;
        inner.meta.tombstone_count = 0;
        {
            // Pre-read all live payloads once for the filter bitmap and
            // payload index rebuilds (slot numbering changed wholesale).
            let mut live_payloads: Vec<Option<super::Payload>> =
                Vec::with_capacity(plan.live_count as usize);
            for slot in 0..plan.live_count {
                live_payloads.push(self.read_payload_at(slot as u32).ok().flatten());
            }
            inner.filter_bitmap.rebuild(
                plan.new_capacity as usize,
                |slot| live_payloads.get(slot as usize).cloned().flatten(),
                0..plan.live_count as u32,
            );
            if !inner.payload_indexes.is_empty() {
                inner.payload_indexes.rebuild(
                    plan.new_capacity as usize,
                    |slot| live_payloads.get(slot as usize).cloned().flatten(),
                    0..plan.live_count as u32,
                );
            }
        }
        inner.meta.save(&self.dir)?;

        // Invalidate the published ANN index: slot numbering changed
        // wholesale. A rebuild is scheduled by the maintenance worker.
        let had_index = self.index.load().is_some();
        self.index.store(Arc::new(None));
        self.pending.store(Arc::new(Vec::new()));
        self.building.store(false, AtomicOrdering::Relaxed);
        if had_index {
            self.needs_rebuild.store(true, AtomicOrdering::Relaxed);
        }
        self.discard_index_files();

        // WAL checkpoint + truncate.
        self.wal.append(&WalTxn {
            txn_id: inner.meta.last_applied_txn,
            ops: vec![WalRecord::Compact],
        })?;
        self.wal.truncate()?;

        Ok(plan.live_count)
    }
}
