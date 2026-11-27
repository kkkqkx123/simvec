use std::sync::Arc;

use super::dense_snapshot::DenseSnapshot;
use super::index_lifecycle::PublishedIndex;
use super::quant::QuantStore;
use super::WalRecord;
use super::{CollectionStore, StoreInner};

use crate::error::{Result, VectorSearchError};
use crate::types::{IndexConfig, QuantizationConfig};

use super::store_open::threshold_met;
use super::*;

impl CollectionStore {
    /// Whether the tombstone ratio has crossed the compaction threshold.
    ///
    /// The engine polls this after mutations and schedules a background
    /// compaction; visibility of data never depends on compaction having run.
    pub fn needs_compaction(&self) -> bool {
        let inner = self.inner.read();
        threshold_met(&inner.meta, self.storage.compaction_threshold)
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
    /// Two phases, and the split is the point:
    ///
    /// 1. **Flush without the store lock.** Inside the queue gate (so no
    ///    transaction can be applied) but *not* holding `inner`, this fsyncs
    ///    the payload pages, the payload tracker/bitmask, `keys.bin` and
    ///    `vectors.bin`, then persists the tombstone table. The gate is what
    ///    makes the unlocked flush safe: with it closed nothing mutates the
    ///    data files, so the bytes being flushed are stable. Readers keep
    ///    taking the read lock the whole time, so a multi-second fsync storm
    ///    no longer freezes search or scroll.
    /// 2. **Durable water marks under the write lock** (microseconds, not
    ///    seconds): `meta.bin` is written from the *live* state rather than
    ///    from a phase-1 snapshot, so a concurrent `set_quantization_config`
    ///    or `set_hnsw_config` cannot be overwritten by a stale copy, then a
    ///    WAL checkpoint marker (its own append fsyncs everything preceding
    ///    it) and finally the truncate.
    ///
    /// Ordering still matters: data files, then tombstones, then `meta.bin`,
    /// then the marker, then the truncate. A crash at any point leaves the
    /// log intact, so recovery replays forward from a state that is already
    /// consistent.
    ///
    /// The `compact_mutex` is held for the whole call: it is the collection's
    /// maintenance-exclusive lock, so compaction (which rewrites the very
    /// files phase 1 reads) cannot swap them under the flush.
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
        // The log is truncated below, so no transaction may be queued — or
        // halfway through the worker — while this runs.
        let _gate = self.queue.quiesce();
        let (capacity, live) = {
            let inner = self.inner.read();
            (inner.meta.slot_capacity, inner.meta.live_count)
        };

        // Phase 1: the expensive part, without the store lock.
        on_flush();
        self.flush_data_files(capacity)?;

        // Phase 2: durable water marks.
        let inner = self.inner.write();
        let data_dir = self.data_dir();
        inner.meta.save(&data_dir)?;
        segments::write(&data_dir, inner.meta.segment_slots, inner.meta.next_slot)?;
        // The snapshot and the manifest describe one generation. Writing the
        // snapshot first means a crash can only leave it ahead of the
        // manifest, which open detects and discards; the WAL is still intact
        // at that point.
        let generation = self.generation.read().clone();
        let data = self.data_arc();
        let tombstones = data.tombstones.load();
        let snapshot = DenseSnapshot::capture(
            generation.id(),
            inner.meta.last_applied_txn,
            &inner.meta,
            &inner.reverse,
            &inner.versions,
            &tombstones,
        );
        DenseSnapshot::write(&data_dir, &snapshot)?;
        // Publish the generation's lsn before the WAL watermark: recovery
        // never replays past what this generation's files contain.
        self.generation.write().set_lsn(inner.meta.last_applied_txn)?;
        self.txns_since_last_save.store(0, AtomicOrdering::Relaxed);
        // Marker first (this append itself fsyncs the log), then truncate
        // through the water mark: every segment covered by `meta.bin` is
        // unlinked, and only transactions newer than it would survive (there
        // are none — the queue gate is closed for the whole call).
        let watermark = inner.meta.last_applied_txn;
        self.wal.append(&WalTxn {
            txn_id: watermark,
            ops: vec![WalRecord::Compact],
        })?;
        self.wal.truncate_through(watermark)?;
        drop(inner);
        self.metrics.record_checkpoint(started.elapsed());
        tracing::debug!(
            collection = %self.inner.read().meta.collection,
            live,
            "wal checkpoint committed"
        );
        Ok(live)
    }

    /// Flush every data file touched since the last checkpoint.
    ///
    /// Requires the queue gate to be closed (no transaction can be applied)
    /// but takes no store lock, so readers are unaffected. `slot_capacity` is
    /// captured under the gate, where no growth can race it.
    fn flush_data_files(&self, slot_capacity: u64) -> Result<()> {
        let data = self.data_arc();
        // Payload store: dirty pages, block bitmask, slot→pointer table.
        let ps_guard = data.payload_store.load();
        if let Some(ps) = ps_guard.as_ref().as_ref() {
            ps.checkpoint()?;
        }
        drop(ps_guard);
        // Directory and vector files touched since the last checkpoint.
        data.keys.sync()?;
        data.vectors.sync()?;
        // Tombstones before meta and before the truncate: they are the only
        // surviving record of the deletes once the log is cleared.
        data.tombstones.load().save(data.dir(), slot_capacity)?;
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
        let index = data.index.load();
        let Some(published) = index.as_ref() else {
            return Ok(0);
        };
        let PublishedIndex::Hnsw(hnsw) = published else {
            return Ok(0);
        };

        let tombstones = data.tombstones.load();
        let vsnap = data.vectors.snapshot();
        let segment_slots = self.inner.read().meta.segment_slots;

        Ok(hnsw.repair(&tombstones, &vsnap, segment_slots))
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
        let quant = data
            .quant
            .read()
            .as_ref()
            .expect("quant present")
            .meta_snapshot();
        let _ = quant;
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
        // Append a WAL checkpoint so replay knows to re-validate quant files.
        // No in-memory mutation is needed; replay for `Quantize` is a no-op
        // because quant files are derived and rebuildable.
        // Best-effort: WAL append failure after meta save is still recoverable
        // (open discards corrupt quant files and falls back to exact).
        let _ = self.wal.append(&WalTxn {
            txn_id: inner.meta.last_applied_txn,
            ops: vec![WalRecord::Quantize],
        });
        Ok(())
    }
}
