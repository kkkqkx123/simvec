use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::Arc;

use rayon::prelude::*;

use super::quant::QuantStore;
use super::WalRecord;
use super::{CollectionStore, StoreInner};

use crate::error::{Result, VectorSearchError};
use crate::types::{IndexConfig, QuantizationConfig};

use super::*;

/// Deletes to accumulate before `after_mutation` runs a whole-store HNSW
/// repair. The sweep takes `build_mutex` and counts tombstones across every
/// segment; each delete already heals the graph that held the deleted slot.
const HNSW_REPAIR_TOMBSTONE_BATCH: u64 = 256;

/// Segments whose applied frontier is ahead of their durable one.
///
/// A segment with no entry in either table is treated as dirty: the marks are
/// grown lazily, so a missing persisted mark means "never flushed", not "clean".
fn raced_segments(applied: &[u64], persisted: &[u64], count: usize, floor: u64) -> Vec<usize> {
    (0..count)
        .filter(|index| {
            let applied = applied.get(*index).copied().unwrap_or(floor);
            let persisted = persisted.get(*index).copied().unwrap_or(0);
            applied > persisted
        })
        .collect()
}

impl CollectionStore {
    /// Whether the tombstone ratio has crossed the compaction threshold.
    ///
    /// The engine polls this after mutations and schedules a background
    /// compaction; visibility of data never depends on compaction having run.
    pub fn needs_compaction(&self) -> bool {
        !self
            .segments_needing_compaction(self.storage.compaction_threshold)
            .is_empty()
    }

    /// Whether the WAL has grown past the checkpoint threshold.
    ///
    /// Tombstone-driven compaction never fires on insert-heavy workloads, so
    /// this size-based trigger is what bounds replay time and log growth.
    pub fn needs_checkpoint(&self) -> bool {
        self.wal.appended_bytes() >= self.storage.wal_checkpoint_bytes
    }

    /// Checkpoint: make every derived file durable, persist the payload
    /// store and truncate the WAL.
    ///
    /// The bulk flush runs without the store lock and without the queue gate:
    /// it fsyncs the payload pages, the payload tracker/bitmask, `keys.bin`
    /// and `vectors.bin`, then persists the tombstone table, while appendable
    /// segments keep receiving writes. Whatever a writer changes meanwhile
    /// stays in the WAL. Readers keep taking the read lock throughout, so a
    /// multi-second fsync storm does not freeze search or scroll.
    ///
    /// The queue gate is closed only for the part that must not race a
    /// transaction: re-syncing the segments a writer touched, publishing the
    /// id trackers, `meta.bin` (written from the *live* state, so a concurrent
    /// `set_quantization_config` or `set_hnsw_config` is not overwritten by a
    /// stale copy), the WAL checkpoint marker (its own append fsyncs
    /// everything preceding it) and the truncate.
    ///
    /// Ordering: data files, then tombstones, then `meta.bin`, then the
    /// marker, then the truncate. A crash at any point leaves the log intact,
    /// so recovery replays forward from a state that is already consistent.
    ///
    /// The `compact_mutex` is held for the whole call: it is the collection's
    /// maintenance-exclusive lock, so compaction (which rewrites the files
    /// this pass flushes) cannot swap them under the flush.
    ///
    /// A published ANN index is not rebuilt here: it is a derived structure
    /// refreshed by the maintenance worker on its own schedule, exactly like
    /// after ordinary mutations.
    pub fn checkpoint(&self) -> Result<u64> {
        self.checkpoint_inner(|| {})
    }

    fn checkpoint_inner(&self, on_flush: impl FnOnce()) -> Result<u64> {
        self.ensure_not_poisoned()?;
        let started = std::time::Instant::now();
        // Compaction and index builds rewrite or replace the files flushed
        // below, so they must not run concurrently with this pass.
        let _maintenance = self.compact_mutex.lock();
        // The bulk flush runs before the gate: appendable segments keep
        // receiving writes while the expensive fsyncs happen, and whatever
        // lands meanwhile stays in the WAL until the gated re-sync below.
        // The live count is reported from the publish point, after the gate
        // drained the queue, so a checkpoint racing queued writes still
        // reports every transaction it waited for.
        let bulk_capacity = self.inner.read().meta.slot_capacity;

        on_flush();
        self.flush_data_files(bulk_capacity)?;

        // Close the queue gate, re-sync the segments a writer touched since
        // the flush, then publish. The log is truncated below, so no
        // transaction may be queued — or halfway through the worker — while
        // the marks are published.
        let _gate = self.queue.quiesce();
        let gated_capacity = self.inner.read().meta.slot_capacity;
        self.flush_raced_segments(gated_capacity)?;
        let mut inner = self.inner.write();
        let data_dir = self.data_dir();
        // A visible tracker must always name bytes that are already durable,
        // so the staged files are published only now that the gate is closed
        // and every dirty byte has been re-synced.
        let generation_id = self.generation.read().id();
        segments::publish_id_trackers(
            &data_dir,
            inner.meta.segment_slots,
            inner.segment_persisted.len(),
        )?;
        inner.meta.save(&data_dir)?;
        {
            let tombstones = self.data_arc().tombstones.load();
            let watermark = inner.meta.last_applied_txn;
            let applied = &inner.segment_applied;
            let persisted = &inner.segment_persisted;
            let entries = segments::derive_entries(
                inner.meta.next_slot,
                inner.meta.segment_slots,
                |s| tombstones.bit(s),
                |index| {
                    (
                        applied.get(index).copied().unwrap_or(watermark),
                        persisted.get(index).copied().unwrap_or(watermark),
                    )
                },
            );
            segments::write_with_entries(
                &data_dir,
                generation_id,
                inner.meta.segment_slots,
                inner.meta.next_slot,
                &entries,
            )?;
        }
        // The published trackers are the authoritative id directory for this
        // generation and were renamed in above, before the manifest advances.
        // A crash can therefore only leave them ahead of the manifest, which
        // open tolerates: it caps each point's water mark at the tracker's own
        // frontier and replay re-applies anything newer.
        // The field posting file is derived too, but keeping it at the same
        // generation/lsn as the checkpoint avoids a full rebuild at open. A
        // crash before the manifest advance leaves it ahead of the manifest
        // and open discards it.
        inner
            .payload_indexes
            .save(&data_dir, generation_id, inner.meta.last_applied_txn)?;
        // Publish the generation's lsn before the WAL watermark: recovery
        // never replays past what this generation's files contain.
        self.generation
            .write()
            .set_lsn(inner.meta.last_applied_txn)?;
        self.txns_since_last_save.store(0, AtomicOrdering::Relaxed);
        // Marker first (this append itself fsyncs the log), then truncate
        // through the water mark clamped by live pins. A snapshot pin holds
        // the truncate back so its copy window keeps the segments it needs;
        // the marker segment itself always survives because its LSN is above
        // every truncate water mark.
        //
        // The marker takes a freshly allocated LSN, never the watermark:
        // reusing the watermark would duplicate the previous tail's LSN and
        // break segment-name continuity on a rollover. The generation lsn
        // stays at the watermark (the files contain exactly that); the marker
        // is a no-op whose segment survives the truncate below.
        let watermark = inner.meta.last_applied_txn;
        // Each segment's persisted lsn was advanced independently by
        // flush_segment to its own applied frontier. Do not flatten them to a
        // single watermark: the ACK must be the minimum over real per-segment
        // durable points so a segment that lagged cannot be cut out of the log.
        let marker_lsn = inner.next_lsn;
        inner.next_lsn = marker_lsn + 1;
        if let Err(e) = self.wal.append(&WalTxn {
            txn_id: marker_lsn,
            ops: vec![WalRecord::Compact],
        }) {
            // The gate is closed and `inner` is held: no other LSN can be
            // allocated concurrently, so the number is safely returned.
            inner.next_lsn = marker_lsn;
            return Err(e);
        }
        // The in-memory water mark stays at the watermark (the files contain
        // exactly that); the marker is a no-op whose LSN was consumed from
        // `next_lsn`, so a back-to-back checkpoint naturally allocates a new
        // one. Replay converges on the same numbering.
        let effective = self.wal_pins.clamp(inner.min_persisted(watermark));
        let live_now = inner.meta.live_count;
        let collection = inner.meta.collection.clone();
        drop(inner);
        if effective != 0 || watermark == 0 {
            self.wal.truncate_through(effective)?;
        }
        self.clear_maintenance_error();
        self.metrics.record_checkpoint(started.elapsed());
        tracing::debug!(
            collection = %collection,
            live = live_now,
            "wal checkpoint committed"
        );
        Ok(live_now)
    }

    /// Flush every data file touched since the last checkpoint.
    ///
    /// Takes no store lock, so readers are unaffected and appendable segments
    /// keep receiving writes. Whatever this pass misses stays in the WAL until
    /// the gated re-sync that precedes the water mark publish.
    fn flush_data_files(&self, slot_capacity: u64) -> Result<()> {
        let segment_slots = self.inner.read().meta.segment_slots;
        if segment_slots == 0 {
            return Ok(());
        }
        let count = slot_capacity.div_ceil(segment_slots as u64) as usize;
        // Flush independent physical segments in parallel. Each segment owns
        // disjoint files and only takes short inner read/write locks, so no
        // segment's fsync blocks another's. A failure in any segment is
        // recorded once and aborts the pass; already-durable segments keep
        // their advanced persisted marks.
        let result: Result<()> = (0..count)
            .into_par_iter()
            .try_for_each(|index| self.flush_segment(index, segment_slots, slot_capacity));
        if let Err(err) = result {
            self.record_maintenance_error(&err.to_string());
            return Err(err);
        }
        Ok(())
    }

    /// Flush the segments whose applied frontier is ahead of their durable one.
    ///
    /// Runs with the queue gate closed. A segment that is already durable has
    /// nothing to sync, and syncing it anyway would lengthen the window for
    /// every segment of the collection.
    fn flush_raced_segments(&self, slot_capacity: u64) -> Result<()> {
        let inner = self.inner.read();
        let segment_slots = inner.meta.segment_slots;
        if segment_slots == 0 {
            return Ok(());
        }
        let count = slot_capacity.div_ceil(segment_slots as u64) as usize;
        let raced = raced_segments(
            &inner.segment_applied,
            &inner.segment_persisted,
            count,
            inner.meta.last_applied_txn,
        );
        drop(inner);
        if raced.is_empty() {
            return Ok(());
        }
        let result: Result<()> = raced
            .into_par_iter()
            .try_for_each(|index| self.flush_segment(index, segment_slots, slot_capacity));
        if let Err(err) = result {
            self.record_maintenance_error(&err.to_string());
            return Err(err);
        }
        Ok(())
    }

    /// Flush one segment's dirty files and stage its id tracker.
    ///
    /// The frontier is captured in a short metadata read; fsyncs run outside
    /// the store lock so writers in other segments keep going. A failure
    /// leaves this segment's persisted mark untouched, so the WAL ack cannot
    /// advance past bytes this segment has not written.
    ///
    /// The tracker is **staged only**: while the queue gate is open a
    /// concurrent write could enter it before its bytes are fsynced.
    /// `publish_id_trackers` renames it in under the gate.
    fn flush_segment(&self, index: usize, segment_slots: u32, slot_capacity: u64) -> Result<()> {
        #[cfg(test)]
        self.segment_flushes.fetch_add(1, AtomicOrdering::Relaxed);
        let (frontier, start, end) = {
            let inner = self.inner.read();
            (
                inner
                    .segment_applied
                    .get(index)
                    .copied()
                    .unwrap_or(inner.meta.last_applied_txn),
                index as u32 * segment_slots,
                ((index as u32 * segment_slots) as u64 + segment_slots as u64).min(slot_capacity)
                    as u32,
            )
        };
        let data = self.data_arc();
        if let Some(ps) = data.payload_store.load().as_ref().as_ref() {
            ps.checkpoint_segment(index)?;
        }
        data.keys.sync_segment(index)?;
        data.vectors.sync_segment(index)?;
        data.tombstones
            .load()
            .save_segment(data.dir(), start, segment_slots, slot_capacity)?;
        let generation = self.generation.read().id();
        segments::write_id_tracker(&data, start, end, &self.inner.read(), generation, frontier)?;
        let mut inner = self.inner.write();
        inner.ensure_segment_len(index + 1);
        inner.segment_persisted[index] = inner.segment_persisted[index].max(frontier);
        Ok(())
    }

    /// Test-only: run a checkpoint with a probe invoked at the start of the
    /// unlocked flush, so a test can assert which locks that phase holds.
    #[cfg(test)]
    pub(crate) fn checkpoint_with_probe(&self, probe: impl FnOnce()) -> Result<u64> {
        self.checkpoint_inner(probe)
    }

    /// Test-only: whether the store write lock is currently free.
    #[cfg(test)]
    pub(crate) fn write_lock_free(&self) -> bool {
        match self.inner.try_write() {
            Some(guard) => {
                drop(guard);
                true
            }
            None => false,
        }
    }

    /// Test-only: whether the store read lock is currently free.
    #[cfg(test)]
    pub(crate) fn read_lock_free(&self) -> bool {
        match self.inner.try_read() {
            Some(guard) => {
                drop(guard);
                true
            }
            None => false,
        }
    }

    /// Test-only: how many segment flush passes have run, counted per segment.
    #[cfg(test)]
    pub(crate) fn segment_flushes(&self) -> u64 {
        self.segment_flushes.load(AtomicOrdering::Relaxed)
    }

    /// Estimated resident index footprint in bytes.
    ///
    /// Covers the dense vector area (allocated capacity), the payload data
    /// pages, the payload pointer table, the directory files, the published
    /// HNSW graph (node table plus adjacency estimates) and the current WAL.
    /// It is an admission estimate, not an accounting figure: callers use it
    /// to judge whether a collection fits the host before writing.
    pub fn memory_estimate(&self) -> u64 {
        let inner = self.inner.read();
        let meta = &inner.meta;
        let dim = meta.vector_size;
        let mut total = meta.slot_capacity * dim as u64 * 4;
        // Payload pointer table: 12 bytes per slot.
        total += meta.next_slot * 12;
        // Payload data area: one page per `blocks_per_page` slots, sized by
        // the configured payload geometry rather than the historical default.
        let blocks_per_page =
            (self.storage.payload_page_size / self.storage.payload_block_size).max(1) as u64;
        let pages = meta.next_slot.div_ceil(blocks_per_page).max(1);
        total += pages * self.storage.payload_page_size as u64;
        // Directory file: keys (8 B/slot) plus the tombstone bitmap
        // (1 bit/slot, rounded up to whole bytes).
        total += meta.slot_capacity * 8 + meta.slot_capacity.div_ceil(8);
        // HNSW graph: Arc node per slot plus per-layer adjacency up to 2*m.
        if let IndexConfig::Hnsw(hnsw) = &meta.index {
            let per_node = 64 + (2 * hnsw.m as u64 + 1) * 4;
            total += meta.next_slot * per_node;
        }
        // Quantized codes: derived file plus in-memory codebook for Product.
        if let Some(qc) = meta.quantization_config.as_ref() {
            total += meta.slot_capacity * qc.quant_bytes_per_vector(meta.vector_size) as u64;
            let data = self.data_arc();
            if let Some(q) = data.quant.read().as_ref() {
                let snap = q.meta_snapshot();
                if let Some(cb) = snap.codebook.as_ref() {
                    total += cb.len() as u64 * 4;
                }
            };
        }
        total += self.wal.appended_bytes();
        total
    }

    /// Incrementally repair the HNSW graph by removing references to
    /// tombstoned slots from adjacency lists.
    ///
    /// This is an alternative to full rebuild after compaction: instead of
    /// invalidating the entire index, only nodes with stale references are
    /// touched. The repair is idempotent and can be called repeatedly as
    /// tombstones accumulate.
    ///
    /// Returns the number of nodes whose adjacency lists were modified,
    /// or 0 if no index is published or no repair was needed.
    pub fn repair_hnsw(&self) -> Result<usize> {
        let data = self.data_arc();
        let segments = data.hnsw_segments.load();
        if segments.is_empty() {
            // No published graph: nothing can be stale, so the backlog that
            // triggers the next sweep stays empty as well.
            self.hnsw_tombstones_pending.store(0, AtomicOrdering::Relaxed);
            return Ok(0);
        }
        let (dim, metric, segment_slots, name, config) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.segment_slots,
                inner.meta.collection.clone(),
                match &inner.meta.index {
                    IndexConfig::Hnsw(cfg) => cfg.clone(),
                    _ => crate::types::HnswConfig::default(),
                },
            )
        };
        let tombstones = data.tombstones.load();
        let vectors = data.vectors.snapshot();
        let _guard = self.build_mutex.lock();
        let mut repaired = 0usize;
        let mut graphs = Vec::with_capacity(segments.len());
        for segment in segments.iter() {
            let len = (segment.end - segment.start).max(1) as usize;
            let dead = (segment.start..segment.end)
                .filter(|slot| tombstones.bit(*slot as usize))
                .count();
            let drift = dead as f64 / len as f64;
            // A few tombstones are repaired through their complete inbound
            // neighborhoods; a dense stale graph is cheaper to rebuild.
            if !segment.index.covers_all(len) || drift > 0.25 || segment.index.stale_ratio() > 0.25
            {
                if let Some(rebuilt) = self.rebuild_segment_hnsw(
                    segment.start,
                    segment.end,
                    &name,
                    dim,
                    metric,
                    segment_slots,
                    &config,
                )? {
                    graphs.push(rebuilt);
                    repaired += len;
                    continue;
                }
            } else if dead > 0 {
                for slot in segment.start..segment.end {
                    if tombstones.bit(slot as usize) {
                        segment.index.heal_deleted(
                            slot - segment.start,
                            &vectors,
                            segment_slots,
                            segment.start,
                            |global| tombstones.bit(global as usize),
                        );
                    }
                }
                repaired += dead;
            }
            graphs.push(segment.clone());
        }
        data.hnsw_segments.store(Arc::new(graphs));
        // The sweep consumed every tombstone that existed when it started.
        // Deletes applied while it ran re-increment the counter, so the next
        // batch is still detected.
        self.hnsw_tombstones_pending.store(0, AtomicOrdering::Relaxed);
        Ok(repaired)
    }

    /// Whether enough deletes accumulated to justify a whole-store graph
    /// repair. Cheap: one relaxed load, no locks.
    pub(crate) fn hnsw_repair_due(&self) -> bool {
        self.hnsw_tombstones_pending.load(AtomicOrdering::Relaxed)
            >= HNSW_REPAIR_TOMBSTONE_BATCH
    }

    /// Grow storage to accommodate `needed_slots` slots (0-indexed high water).
    ///
    /// Refuses to grow past the collection's optional `max_points` cap: the
    /// cap is the caller's admission contract, so an over-limit upsert fails
    /// loudly instead of silently allocating index memory the host may not
    /// have.
    pub(super) fn ensure_capacity(&self, inner: &mut StoreInner, needed_slots: u64) -> Result<()> {
        let meta = &mut inner.meta;
        if needed_slots <= meta.slot_capacity {
            return Ok(());
        }
        if let Some(cap) = meta.max_points {
            if needed_slots > cap {
                return Err(VectorSearchError::CapacityExceeded {
                    collection: meta.collection.clone(),
                    max_points: cap,
                    requested: needed_slots,
                });
            }
        }
        let data = self.data_arc();
        data.vectors.grow_to(needed_slots)?;
        let new_capacity = data.vectors.slot_capacity();
        data.keys.grow_to(new_capacity)?;
        if let Some(q) = data.quant.read().as_ref() {
            // Quant file growth mirrors vectors capacity; zero-fill new range.
            if let Err(e) = q.grow_to(new_capacity) {
                tracing::warn!(error = %e, "quant grow_to failed; rebuild will recreate the file");
            }
        }

        let next = data.tombstones.load().resized(new_capacity as usize);
        data.tombstones.store(Arc::new(next));

        meta.slot_capacity = new_capacity;
        inner.filter_bitmap.resize(new_capacity as usize);
        inner.payload_indexes.resize(new_capacity as usize);
        // Capacity change is a rare, natural checkpoint for the payload store's
        // derived files: it costs O(new slots) once per segment growth instead
        // of once per point. Failures are non-fatal (the WAL covers replay).
        let ps_guard = data.payload_store.load();
        if let Some(ps) = ps_guard.as_ref().as_ref() {
            let _ = ps.ensure_segments(new_capacity);
            let _ = ps.checkpoint();
        }
        Ok(())
    }

    /// Build or rebuild quantization from the current live set.
    ///
    /// Heavy work (collecting vectors and training codebooks) runs without
    /// holding the store write lock, under the same `build_mutex` that
    /// serializes index builds and compaction. The product codebook requires
    /// k-means; scalar/binary need only a range scan.
    pub fn build_quantization(&self) -> Result<bool> {
        self.ensure_not_poisoned()?;
        let qc = {
            let inner = self.inner.read();
            match &inner.meta.quantization_config {
                Some(qc) => qc.clone(),
                _ => return Ok(false),
            }
        };
        // Exclusive with compaction and index builds (order: compact -> build)
        // so slot numbers stay stable and quant files are not rewritten
        // concurrently by compaction.
        let _compact_guard = self.compact_mutex.lock();
        let _guard = self.build_mutex.lock();
        let (dim, distance, segment_slots, next_slot) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.segment_slots,
                inner.meta.next_slot,
            )
        };
        let data = self.data_arc();
        // If store has no quant yet, create one now.
        {
            let mut qguard = data.quant.write();
            if qguard.is_none() {
                let capacity = self.inner.read().meta.slot_capacity;
                *qguard = Some(QuantStore::create(
                    data.dir(),
                    dim,
                    distance,
                    &qc,
                    segment_slots,
                    capacity,
                )?);
            }
        }
        let live: Vec<u32> = {
            let tomb = data.tombstones.load();
            (0..next_slot as u32)
                .filter(|&s| !tomb.bit(s as usize))
                .collect()
        };
        if live.is_empty() {
            return Ok(false);
        }
        let vsnap = data.vectors.snapshot();
        // Delegate training to QuantStore. No external cancellation source is
        // wired yet; rebuild checks the flag cooperatively.
        let stopped = std::sync::atomic::AtomicBool::new(false);
        let quant_store = data.quant.read();
        let qs = quant_store.as_ref().expect("quant store exists");
        qs.rebuild(&vsnap, segment_slots, &live, dim, &stopped)?;
        Ok(true)
    }

    /// Whether quantization is active and ready for search.
    pub fn has_quantization(&self) -> bool {
        self.data_arc()
            .quant
            .read()
            .as_ref()
            .is_some_and(|q| q.is_ready())
    }

    /// Quantization config in effect, if any.
    pub fn quantization_config(&self) -> Option<crate::types::QuantizationConfig> {
        self.inner.read().meta.quantization_config.clone()
    }

    /// Replace quantization config at runtime (engine setting). Persists to
    /// `meta.bin` and creates or replaces the quant files accordingly.
    ///
    /// Crash-safe order: quant files are created/synced first, then `meta.bin`
    /// is swapped. If the process crashes before `meta.bin` is durable the
    /// collection reopens without quantization (fallback to exact); orphan
    /// quant files are discarded on the next open.
    pub fn set_quantization_config(&self, config: QuantizationConfig) -> Result<()> {
        self.ensure_not_poisoned()?;
        config.validate(self.inner.read().meta.vector_size)?;
        // Create quant files before persisting meta, so a crash
        // cannot leave meta claiming quantization while files are missing.
        let new_quant = {
            // Pre-create the store while not holding the inner write lock.
            let (dim, distance, segment_slots, slot_capacity) = {
                let inner = self.inner.read();
                (
                    inner.meta.vector_size,
                    inner.meta.distance,
                    inner.meta.segment_slots,
                    inner.meta.slot_capacity,
                )
            };
            let data = self.data_arc();
            let mut qguard = data.quant.write();
            if qguard.is_none() {
                *qguard = Some(QuantStore::create(
                    data.dir(),
                    dim,
                    distance,
                    &config,
                    segment_slots,
                    slot_capacity,
                )?);
            } else if qguard.as_ref().is_some_and(|q| q.config() != &config) {
                // Config type changed: recreate files.
                *qguard = None;
                let _ = std::fs::remove_file(data.dir().join("quant.bin"));
                let _ = std::fs::remove_file(data.dir().join("quant_meta.bin"));
                *qguard = Some(QuantStore::create(
                    data.dir(),
                    dim,
                    distance,
                    &config,
                    segment_slots,
                    slot_capacity,
                )?);
            }
            // Hold the quant guard briefly to ensure files are durable before
            // meta claims them; QuantStore::create already fsyncs.
            drop(qguard);
            Some(config.clone())
        };
        let mut inner = self.inner.write();
        inner.meta.quantization_config = new_quant;
        inner.meta.save(&self.data_dir())?;
        // Route the marker through the commit queue so it carries a freshly
        // allocated LSN in queue order: reusing `last_applied_txn` here could
        // duplicate a concurrent transaction's LSN and break segment-name
        // continuity. Replay for `Quantize` is a no-op because quant files
        // are derived and rebuildable.
        // Best-effort: WAL failure after meta save is still recoverable
        // (open discards corrupt quant files and falls back to exact).
        drop(inner);
        let _ = self.apply_ops(&[WalRecord::Quantize]);
        Ok(())
    }
}
