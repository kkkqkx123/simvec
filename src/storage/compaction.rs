//! Collection compaction plus the file-rewriting helpers it relies on.
//!
//! Compaction builds a complete new generation in a staging directory,
//! fsyncs it, then publishes a generation manifest. No serving file is
//! overwritten in place: a crash before publication leaves an ignored
//! staging directory and the previous generation untouched.
//!
//! The heavy rewrite phase runs under the `compact_mutex` only — the store
//! write lock is *not* held, so searches and upserts proceed on the old
//! files. A mutation counter guards publication: the swap happens under the
//! write lock only if no transaction raced the rewrite; otherwise the attempt
//! is retried on a fresh snapshot. Under sustained write pressure the
//! compaction falls back to staging under the queue gate with only a read
//! snapshot, so progress is always guaranteed without holding the write lock
//! for the whole rewrite. In both paths `compact_mutex` excludes
//! index builds/drains, keeping slot numbers stable for the index.
//!
//! The commit also truncates the WAL, so it gates the commit queue for its
//! duration: a transaction that is queued or halfway through the queue
//! worker would otherwise be cut out of the log it was just fsynced into.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use memmap2::Mmap;

use super::dense_snapshot::{DenseSnapshot, SnapshotPoint};
use super::directory::KEY_REC_SIZE;
use super::generation::StagedGeneration;
use super::keys::Keys;
use super::segment_hnsw::{self, SegmentHnsw};
use super::segment_ivf::{self, SegmentIvf};
use super::segments;
use super::tombstones::TombstoneBits;
use super::vectors::Vectors;
use super::{CollectionStore, DataFiles, Meta};
use crate::error::{Result, VectorSearchError};
use crate::index::{persist, HnswIndex, IndexBuildParams, IvfIndex};
use crate::types::{HnswConfig, IndexConfig, IvfConfig, PointId};

const HEADER_LEN: usize = 32;
const FILE_VERSION: u32 = 2;

/// How often the lock-free rewrite is retried when a write races it before
/// falling back to the whole-rewrite critical section.
const COMMIT_ATTEMPTS: usize = 4;

/// Everything one compaction attempt needs, captured from an immutable view.
struct CompactionPlan {
    /// Metadata snapshot the staged generation is built from.
    meta: Meta,
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
    /// Id directory and per-id water marks captured under the store lock.
    points: Vec<SnapshotPoint>,
}

/// Compute the plan from a metadata snapshot and the current tombstone bits.
fn build_plan(
    meta: &Meta,
    tombstones: &TombstoneBits,
    inner: &super::StoreInner,
) -> CompactionPlan {
    let (new_capacity, map) =
        plan_slots(|s| !tombstones.bit(s), meta.next_slot, meta.segment_slots);
    let live_count = map.iter().filter(|s| **s != u32::MAX).count() as u64;
    CompactionPlan {
        meta: meta.clone(),
        dim: meta.vector_size,
        segment_slots: meta.segment_slots,
        next_slot: meta.next_slot,
        tombstone_count: meta.tombstone_count,
        new_capacity,
        map,
        live_count,
        points: DenseSnapshot::capture_points(&inner.reverse, &inner.versions),
    }
}

/// ANN artifacts built into the staged generation.
struct StagedIndex {
    ivf: Vec<SegmentIvf>,
    hnsw: Vec<SegmentHnsw>,
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
    // The record stride is part of the file, so a reader never has to agree
    // with the writer's compile-time constant.
    header.extend_from_slice(&(rec_size as u32).to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes());
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

/// Build one HNSW file per completed segment of the staged generation.
fn stage_hnsw_segments(
    store: &CollectionStore,
    dir: &Path,
    meta: &Meta,
    config: &HnswConfig,
) -> Result<Vec<SegmentHnsw>> {
    let boundary = segments::immutable_end(meta.next_slot, meta.segment_slots);
    if boundary == 0 {
        return Ok(Vec::new());
    }
    let vectors = Vectors::open(
        &dir.join("vectors.bin"),
        meta.vector_size,
        meta.segment_slots,
        meta.slot_capacity,
    )?;
    let snapshot = vectors.snapshot();
    let step = meta.segment_slots.max(1);
    let name = meta.collection.as_str();
    let mut graphs = Vec::new();
    for start in (0..boundary).step_by(step as usize) {
        let end = (start + step).min(boundary);
        let local: Vec<u32> = (0..end - start).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new(
                name,
                meta.vector_size,
                meta.distance,
                &local,
                &snapshot,
                meta.segment_slots,
            )
            .with_slot_base(start),
            config,
        )?;
        persist::save_hnsw_at(&segment_hnsw::file_path(dir, start), &index.to_persisted())?;
        let quant = {
            let data = store.data_arc();
            let guard = data.quant.read();
            match guard.as_ref() {
                Some(q) if q.is_ready() => q.write_segment_file(
                    &crate::storage::quant::segment_file_path(dir, start),
                    &snapshot,
                    meta.segment_slots,
                    start,
                    end - start,
                )?,
                _ => false,
            }
        };
        let quant = if quant {
            let data = store.data_arc();
            let guard = data.quant.read();
            guard
                .as_ref()
                .map(|q| {
                    crate::storage::quant::QuantStore::load_segment_file(
                        &crate::storage::quant::segment_file_path(dir, start),
                        meta.segment_slots,
                        q.bytes_per_vector(),
                    )
                })
                .transpose()?
                .flatten()
        } else {
            None
        };
        graphs.push(SegmentHnsw {
            start,
            end,
            index: Arc::new(index),
            quant,
        });
    }
    Ok(graphs)
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

/// Size of `path` in bytes, or 0 when it does not exist.
fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Recursive size of `dir` in bytes, or 0 when it does not exist.
fn dir_len(dir: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        total += match std::fs::metadata(&path) {
            Ok(m) if m.is_dir() => dir_len(&path),
            Ok(m) => m.len(),
            Err(_) => 0,
        };
    }
    total
}

/// Bytes available on the volume holding `path`.
#[cfg(unix)]
fn available_bytes(path: &Path) -> Option<u64> {
    let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    if rc != 0 {
        return None;
    }
    Some(stat.f_bavail as u64 * stat.f_frsize as u64)
}

#[cfg(not(unix))]
fn available_bytes(_path: &Path) -> Option<u64> {
    None
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
    /// write pressure a bounded fallback stages under the queue gate from a
    /// read snapshot, so compaction always makes progress without holding the
    /// write lock across the rewrite.
    ///
    /// Returns the number of live points after compaction. Refuses to start
    /// when the target volume cannot hold the temp files, so a full disk
    /// surfaces as [`VectorSearchError::InsufficientSpace`] instead of a
    /// half-rewritten collection.
    pub fn compact(&self) -> Result<u64> {
        self.ensure_not_poisoned()?;
        let _compact_guard = self.compact_mutex.lock();
        self.ensure_compaction_space()?;
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
                let tombstones = self.data_arc().tombstones.load();
                build_plan(&inner.meta, &tombstones, &inner)
            };
            let version = self.mutations.load(AtomicOrdering::Relaxed);
            let staged = self.generation.read().stage()?;
            let remapped = match self.stage_generation(&plan, &staged) {
                Ok(remapped) => remapped,
                Err(e) => {
                    staged.discard();
                    return Err(e);
                }
            };
            {
                // The commit publishes the manifest and truncates the WAL, so
                // the queue is gated for its duration: no transaction may be
                // queued or halfway through the worker while the log is cut.
                let _gate = self.queue.quiesce();
                let mut inner = self.inner.write();
                let raced = self.mutations.load(AtomicOrdering::Relaxed) != version
                    || inner.meta.next_slot != plan.next_slot
                    || inner.meta.tombstone_count != plan.tombstone_count;
                if !raced {
                    let live =
                        self.commit_generation(&mut inner, &plan, staged, remapped)?;
                    self.metrics.record_compaction(started.elapsed());
                    return Ok(live);
                }
            }
            staged.discard();
            self.metrics.record_compaction_race_retry();
            tracing::debug!(
                collection = %self.inner.read().meta.collection,
                "compaction raced concurrent writes; retrying"
            );
        }

        // Contended fallback: close the queue gate first, then stage from a
        // read snapshot. The gate already prevents writers from changing the
        // plan, so the expensive rewrite never needs the store write lock;
        // only the final generation swap takes it.
        self.metrics.record_compaction_contended();
        let _build_guard = self.build_mutex.lock();
        let _gate = self.queue.quiesce();
        let plan = {
            let inner = self.inner.read();
            if inner.meta.tombstone_count == 0 || inner.meta.next_slot == 0 {
                return Ok(inner.meta.live_count);
            }
            let tombstones = self.data_arc().tombstones.load();
            build_plan(&inner.meta, &tombstones, &inner)
        };
        let staged = self.generation.read().stage()?;
        let remapped = match self.stage_generation(&plan, &staged) {
            Ok(remapped) => remapped,
            Err(e) => {
                staged.discard();
                return Err(e);
            }
        };
        let mut inner = self.inner.write();
        let live = self.commit_generation(&mut inner, &plan, staged, remapped);
        self.metrics.record_compaction(started.elapsed());
        live
    }

    /// Upper bound on the temp-file footprint one compaction needs.
    ///
    /// The compacted files can never be larger than the live ones (capacity
    /// only shrinks, payload data only moves), so the current on-disk size of
    /// every file that gets rewritten bounds the peak usage of the rewrite
    /// phase, which briefly doubles that space.
    fn compaction_footprint(&self) -> u64 {
        let dir = self.data_dir();
        let mut total = file_len(&dir.join("vectors.bin"));
        total += file_len(&dir.join("keys.bin"));
        total += file_len(&dir.join("quant.bin"));
        total += dir_len(&dir.join("payloads_store"));
        total
    }

    /// Reject a compaction whose temp files cannot fit on the target volume.
    ///
    /// A half-rewritten collection is unrecoverable without a reindex, so the
    /// space question is asked up front and the failure is recorded as a
    /// metric: a growing counter is the signal to provision before the next
    /// compaction attempt.
    fn ensure_compaction_space(&self) -> Result<()> {
        let needed = self.compaction_footprint();
        let available = available_bytes(&self.dir).unwrap_or(u64::MAX);
        if needed > available {
            self.metrics.record_compaction_space_rejection();
            tracing::warn!(
                collection = %self.inner.read().meta.collection,
                needed,
                available,
                "compaction refused: target volume cannot hold the temp files"
            );
            return Err(VectorSearchError::InsufficientSpace {
                operation: format!("compaction of {}", self.dir.display()),
                needed,
                available,
            });
        }
        Ok(())
    }

    /// Build every file of a new generation from immutable snapshots.
    ///
    /// Requires no store locks: every input is either an `ArcSwap` snapshot or
    /// captured in `plan`. Callers validate via the mutation counter that no
    /// write raced the reads before publishing this generation. The returned
    /// ANN artifacts are already persisted into the staging dir.
    fn stage_generation(
        &self,
        plan: &CompactionPlan,
        staged: &StagedGeneration,
    ) -> Result<StagedIndex> {
        let dir = staged.path();
        let files = self.data_arc();
        let mut meta = plan.meta.clone();
        meta.slot_capacity = plan.new_capacity;
        meta.next_slot = plan.live_count;
        meta.live_count = plan.live_count;
        meta.tombstone_count = 0;

        write_vectors_file(
            &dir.join("vectors.bin"),
            plan.dim,
            plan.segment_slots,
            plan.new_capacity,
            &files.vectors.snapshot(),
            &plan.map,
        )?;

        if let Some(q) = files.quant.read().as_ref() {
            q.write_compacted_file(
                &dir.join("quant.bin"),
                plan.new_capacity,
                &files.vectors.snapshot(),
                &plan.map,
            )?;
            q.write_compacted_meta(dir, plan.new_capacity)?;
        }

        {
            let keys_view = files.keys.snapshot();
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
                &dir.join("keys.bin"),
                *b"VKEY",
                KEY_REC_SIZE,
                plan.new_capacity,
                &entries,
            )?;
        }

        let live_slots: Vec<(u32, u32)> = plan
            .map
            .iter()
            .enumerate()
            .filter(|(_, new_slot)| **new_slot != u32::MAX)
            .map(|(old, &new)| (old as u32, new))
            .collect();
        let ps_guard = files.payload_store.load();
        let old_store = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        let new_store = old_store.compact_to(&dir.join("payloads_store"), &live_slots)?;
        drop(new_store);
        drop(ps_guard);

        TombstoneBits::new(plan.new_capacity as usize).save(dir, plan.new_capacity)?;
        let defs = files.dir().join("payload_indexes.json");
        if defs.exists() {
            std::fs::copy(defs, dir.join("payload_indexes.json"))?;
        }
        let points = plan
            .points
            .iter()
            .map(|point| {
                let mut mapped = point.clone();
                if !mapped.deleted {
                    mapped.slot = plan.map[point.slot as usize];
                }
                mapped
            })
            .collect();
        let snapshot = DenseSnapshot {
            generation: staged.id(),
            lsn: meta.last_applied_txn,
            next_slot: meta.next_slot,
            slot_capacity: meta.slot_capacity,
            live_count: meta.live_count,
            tombstone_count: 0,
            points,
            tombstones: Vec::new(),
        };
        DenseSnapshot::write(dir, &snapshot)?;
        segments::write(dir, meta.segment_slots, meta.next_slot)?;
        meta.save(dir)?;

        let hnsw = match &meta.index {
            IndexConfig::Hnsw(config) => stage_hnsw_segments(self, dir, &meta, config)?,
            _ => Vec::new(),
        };
        let ivf = match &meta.index {
            IndexConfig::Ivf(Some(config)) => self.stage_ivf_segments(dir, &meta, config)?,
            _ => Vec::new(),
        };
        Ok(StagedIndex { ivf, hnsw })
    }

    /// Publish a validated generation and switch every in-memory file handle
    /// to it. The caller holds the store write lock and the queue gate, and
    /// must have verified that no write raced the staging phase.
    fn commit_generation(
        &self,
        inner: &mut super::StoreInner,
        plan: &CompactionPlan,
        staged: StagedGeneration,
        staged_index: StagedIndex,
    ) -> Result<u64> {
        let mut meta = plan.meta.clone();
        meta.slot_capacity = plan.new_capacity;
        meta.next_slot = plan.live_count;
        meta.live_count = plan.live_count;
        meta.tombstone_count = 0;
        let watermark = meta.last_applied_txn;

        // Validate the staged files before publication: a failure here leaves
        // the old generation serving and the staging directory unpublished.
        let staged_files =
            DataFiles::open(staged.path(), &meta, &self.storage, meta.last_applied_txn)?;
        drop(staged_files);
        let old_files_view = self.data_arc();
        let had_index = !old_files_view.hnsw_segments.load().is_empty()
            || !old_files_view.ivf_segments.load().is_empty();
        drop(old_files_view);
        let old_generation = self.generation.read().clone();
        let next_generation = old_generation.clone().publish(staged, watermark)?;
        // From here on the disk manifest already names the new generation.
        // Any failure before the in-memory switch completes must stop writes
        // instead of letting this process keep mutating files the manifest
        // no longer describes.
        self.generation_poisoned
            .store(true, AtomicOrdering::Release);
        // The ack published by the wipe below belongs to the new
        // generation, not to the one the log was opened with.
        self.wal.bind_generation(next_generation.id());
        // Reopen from the published path so every handle's path stays valid
        // for later growth and checkpoint writes.
        let new_files = DataFiles::open(&next_generation.path(), &meta, &self.storage, watermark)?;
        new_files.ivf_segments.store(Arc::new(staged_index.ivf));
        new_files.hnsw_segments.store(Arc::new(staged_index.hnsw));
        let has_new_index =
            !new_files.hnsw_segments.load().is_empty() || !new_files.ivf_segments.load().is_empty();
        let old_files = self.files.swap(Arc::new(new_files));
        *self.generation.write() = next_generation;

        // Rebuild the in-memory directory against the new slot numbering.
        // Tombstones are already all-clear in the staged table.
        inner.meta = meta;
        inner.reverse.clear();
        {
            let files = self.data_arc();
            let keys_view = files.keys.snapshot();
            for slot in 0..plan.live_count as usize {
                let key = Keys::read_key(&keys_view, slot)?.ok_or_else(|| {
                    VectorSearchError::CorruptData(format!("live slot {slot} has no key"))
                })?;
                inner.reverse.insert(&PointId::from(key), slot as u32);
            }
        }
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
        // `pending` holds old slot numbers as well.
        {
            let mut pending = self.pending.lock();
            let carried: Vec<u32> = pending
                .drain(..)
                .filter_map(|slot| match plan.map.get(slot as usize) {
                    Some(&m) if m != u32::MAX => Some(m),
                    _ => None,
                })
                .collect();
            *pending = carried;
        }
        self.building.store(false, AtomicOrdering::Relaxed);
        self.needs_rebuild
            .store(had_index && !has_new_index, AtomicOrdering::Relaxed);

        // The manifest is published; recovery replays from its lsn even if the
        // old WAL segments survive a crash here. The gate is drained, so no
        // transaction is newer than the watermark: truncating through it
        // empties the log. No marker is written — the surviving generation
        // lsn plus the published ack pin the replay floor, and the next
        // allocated LSN cannot collide.
        self.wal.truncate_through(watermark)?;

        // Old mmaps stay valid for readers that loaded them before the swap.
        // The directory is removed only after the last guard is gone.
        thread::spawn(move || {
            while Arc::strong_count(&old_files) > 1 {
                thread::sleep(Duration::from_millis(10));
            }
            if let Err(e) = old_generation.retire() {
                tracing::warn!(error = %e, "failed to remove superseded generation");
            }
        });
        self.generation_poisoned
            .store(false, AtomicOrdering::Release);
        Ok(plan.live_count)
    }

    /// Build one IVF file per completed segment of the staged generation.
    /// Centroids and assignments are recalculated for the compacted slots, so
    /// no cross-generation remap is needed.
    fn stage_ivf_segments(
        &self,
        dir: &Path,
        meta: &Meta,
        config: &IvfConfig,
    ) -> Result<Vec<SegmentIvf>> {
        let boundary = segments::immutable_end(meta.next_slot, meta.segment_slots);
        if boundary == 0 {
            return Ok(Vec::new());
        }
        let vectors = Vectors::open(
            &dir.join("vectors.bin"),
            meta.vector_size,
            meta.segment_slots,
            meta.slot_capacity,
        )?;
        let snapshot = vectors.snapshot();
        let step = meta.segment_slots.max(1);
        let mut indexes = Vec::new();
        for start in (0..boundary).step_by(step as usize) {
            let end = (start + step).min(boundary);
            let local: Vec<u32> = (0..end - start).collect();
            let index = IvfIndex::build(
                &IndexBuildParams::new(
                    meta.collection.as_str(),
                    meta.vector_size,
                    meta.distance,
                    &local,
                    &snapshot,
                    meta.segment_slots,
                )
                .with_slot_base(start),
                config,
            )?;
            persist::save_at(&segment_ivf::file_path(dir, start), &index.to_persisted())?;
            indexes.push(SegmentIvf {
                start,
                end,
                index: Arc::new(index),
            });
        }
        Ok(indexes)
    }
}
