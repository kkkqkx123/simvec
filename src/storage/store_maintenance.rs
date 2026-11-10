use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use arc_swap::ArcSwap;
use bitvec::prelude::*;
use parking_lot::{Mutex, RwLock};

use super::{StoreInner, CollectionStore};
use super::{set_applying, is_commit_worker, set_commit_worker, assert_not_applying};
use super::filter_bitmap::FilterBitmap;
use super::index_lifecycle::PublishedIndex;
use super::keys::Keys;
use super::payloads::Payloads;
use super::quant::QuantStore;
use super::queue::{CommitQueue, APPLY_WAIT_TIMEOUT};
use super::vectors::Vectors;
use super::{Wal, WalRecord};
use super::meta::Meta;
use super::payload_index::PayloadIndexManager;
use super::payload_store::PayloadStore;
use super::tombstones::TombstoneBits;

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::types::{CollectionConfig, DistanceMetric, IndexConfig, IvfConfig, Payload, PointId, QuantizationConfig, VectorPoint};

use super::store_open::{count_ops, threshold_met};
use super::*;

impl CollectionStore {
    /// Whether the tombstone ratio has crossed the compaction threshold.
    ///
    /// The engine polls this after mutations and schedules a background
    /// compaction; visibility of data never depends on compaction having run.
    pub fn needs_compaction(&self) -> bool {
        let inner = self.inner.read();
        threshold_met(&inner.meta)
    }

    /// Whether the WAL has grown past the checkpoint threshold.
    ///
    /// Tombstone-driven compaction never fires on insert-heavy workloads, so
    /// this size-based trigger is what bounds replay time and log growth.
    pub fn needs_checkpoint(&self) -> bool {
        self.wal.appended_bytes() >= WAL_CHECKPOINT_BYTES
    }

    /// Checkpoint: make every derived file durable, persist the payload
    /// store and truncate the WAL.
    ///
    /// Runs on the background maintenance thread under the store write lock,
    /// so no transaction can interleave with the flush. Ordering matters:
    /// derived files are synced first, then `meta.bin`, then a WAL checkpoint
    /// marker (its own append fsyncs everything preceding it) and finally the
    /// truncate. A crash at any point leaves the log intact, so recovery
    /// replays forward from a state that is already consistent.
    ///
    /// A published ANN index is not rebuilt here: it is a derived structure
    /// refreshed by the maintenance worker on its own schedule, exactly like
    /// after ordinary mutations.
    pub fn checkpoint(&self) -> Result<u64> {
        let started = std::time::Instant::now();
        // The log is truncated below, so no transaction may be queued — or
        // halfway through the worker — while this runs.
        let _gate = self.queue.quiesce();
        let inner = self.inner.write();
        // Payload store: dirty pages, block bitmask, slot→pointer table.
        let ps_guard = self.payload_store.load();
        if let Some(ps) = ps_guard.as_ref().as_ref() {
            ps.checkpoint()?;
        }
        drop(ps_guard);
        // Directory and vector files touched since the last checkpoint.
        self.keys.sync();
        self.payloads.sync();
        self.vectors.sync();
        // Durable metadata water marks.
        inner.meta.save(&self.dir)?;
        self.txns_since_last_save.store(0, AtomicOrdering::Relaxed);
        // Marker first (this append itself fsyncs the log), then truncate.
        self.wal.append(&WalTxn {
            txn_id: inner.meta.last_applied_txn,
            ops: vec![WalRecord::Compact],
        })?;
        self.wal.truncate()?;
        let live = inner.meta.live_count;
        drop(inner);
        self.metrics.record_checkpoint(started.elapsed());
        tracing::debug!(
            collection = %self.inner.read().meta.collection,
            live,
            "wal checkpoint committed"
        );
        Ok(live)
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
        // Payload data area: one 32 MiB page per 256K blocks of 128 B.
        let pages = meta
            .next_slot
            .div_ceil(
                payload_store::DEFAULT_PAGE_SIZE as u64 / payload_store::DEFAULT_BLOCK_SIZE as u64,
            )
            .max(1);
        total += pages * payload_store::DEFAULT_PAGE_SIZE as u64;
        // Directory files: keys (8 B/slot) + payloads (12 B/slot) records.
        total += meta.slot_capacity * 20;
        // HNSW graph: Arc node per slot plus per-layer adjacency up to 2*m.
        if let IndexConfig::Hnsw(hnsw) = &meta.index {
            let per_node = 64 + (2 * hnsw.m as u64 + 1) * 4;
            total += meta.next_slot * per_node;
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
        let index = self.index.load();
        let Some(published) = index.as_ref() else {
            return Ok(0);
        };
        let PublishedIndex::Hnsw(hnsw) = published else {
            return Ok(0);
        };

        let tombstones = self.tombstones.load();
        let vsnap = self.vectors.snapshot();
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
        self.vectors.grow_to(needed_slots)?;
        let new_capacity = self.vectors.slot_capacity();
        self.keys.grow_to(new_capacity)?;
        self.payloads.grow_to(new_capacity)?;
        if let Some(q) = self.quant.read().as_ref() {
            // Quant file growth mirrors vectors capacity; zero-fill new range.
            let _ = q.grow_to(new_capacity);
        }

        let next = self.tombstones.load().resized(new_capacity as usize);
        self.tombstones.store(Arc::new(next));

        meta.slot_capacity = new_capacity;
        inner.filter_bitmap.resize(new_capacity as usize);
        inner.payload_indexes.resize(new_capacity as usize);
        // Capacity change is a rare, natural checkpoint for the payload store's
        // derived files: it costs O(new slots) once per segment growth instead
        // of once per point. Failures are non-fatal (the WAL covers replay).
        let ps_guard = self.payload_store.load();
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
        // If store has no quant yet, create one now.
        {
            let mut qguard = self.quant.write();
            if qguard.is_none() {
                let capacity = self.inner.read().meta.slot_capacity;
                *qguard = Some(QuantStore::create(
                    &self.dir,
                    dim,
                    distance,
                    &qc,
                    segment_slots,
                    capacity,
                )?);
            }
        }
        let live: Vec<u32> = {
            let tomb = self.tombstones.load();
            (0..next_slot as u32)
                .filter(|&s| !tomb.bit(s as usize))
                .collect()
        };
        if live.is_empty() {
            return Ok(false);
        }
        let vsnap = self.vectors.snapshot();
        let quant = self
            .quant
            .read()
            .as_ref()
            .expect("quant present")
            .meta_snapshot();
        let _ = quant;
        // Delegate training to QuantStore.
        let quant_store = self.quant.read();
        let qs = quant_store.as_ref().expect("quant store exists");
        qs.rebuild(&vsnap, segment_slots, &live, dim)?;
        Ok(true)
    }

    /// Whether quantization is active and ready for search.
    pub fn has_quantization(&self) -> bool {
        self.quant.read().as_ref().is_some_and(|q| q.is_ready())
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
            let mut qguard = self.quant.write();
            if qguard.is_none() {
                *qguard = Some(QuantStore::create(
                    &self.dir,
                    dim,
                    distance,
                    &config,
                    segment_slots,
                    slot_capacity,
                )?);
            } else if qguard.as_ref().is_some_and(|q| q.config() != &config) {
                // Config type changed: recreate files.
                *qguard = None;
                let _ = std::fs::remove_file(self.dir.join("quant.bin"));
                let _ = std::fs::remove_file(self.dir.join("quant_meta.bin"));
                *qguard = Some(QuantStore::create(
                    &self.dir,
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
        inner.meta.save(&self.dir)?;
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
