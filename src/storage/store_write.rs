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

use super::store_open::{count_ops, validate_admission_locked, validate_point};
use super::*;

impl CollectionStore {
    /// Apply a WAL-backed transaction.
    ///
    /// Four steps, of which only two hold the store write lock:
    ///
    /// 1. **reserve** — validate every op (no invalid data may be logged) and
    ///    resolve the slot for every upsert, in one short `inner.write()`
    ///    critical section that also assigns the lsn and the queue position;
    /// 2. **flush** — the commit-queue worker drains the whole backlog into
    ///    one WAL batch, so concurrent writers share a single fsync;
    /// 3. **apply** — the same worker applies the batch in lsn order under a
    ///    short `inner.write()`;
    /// 4. **complete** — the worker settles the ticket and wakes this caller.
    ///
    /// Nothing is applied before its record is durable, and the call returns
    /// only once the transaction is applied, so `get`/`search` see exactly
    /// what a successful return promises.
    pub fn apply_txn(&self, ops: &[WalRecord]) -> Result<()> {
        assert_not_applying();
        // Validation happens before anything is logged: a logged record that
        // fails while being applied would sit in the WAL forever and make
        // every later open fail.
        {
            let inner = self.inner.read();
            for op in ops {
                if let WalRecord::Upsert { point } = op {
                    validate_point(&inner.meta, &point.to_point()?)?;
                }
            }
        }
        // Pass the queue gate before taking the store write lock: a
        // checkpoint or compaction owns the WAL while it truncates the log,
        // and a writer that blocked on the gate while holding the write lock
        // could never let that pass finish.
        let permit = self.queue.begin_write();
        let enqueued = std::time::Instant::now();
        let completion = {
            let mut inner = self.inner.write();
            validate_admission_locked(&inner, ops)?;
            self.reserve_slots_locked(&mut inner, ops);
            let lsn = inner.next_lsn;
            inner.next_lsn = lsn + 1;
            // The lsn, the slot reservations and the queue position are fixed
            // in this one critical section: together they are the log order,
            // and the worker applies in exactly that order.
            CommitQueue::enqueue(&self.queue, &self.self_weak, lsn, ops.to_vec())?
        };
        drop(permit);
        let outcome = completion.wait(APPLY_WAIT_TIMEOUT);
        self.metrics.record_apply_wait(enqueued.elapsed());
        outcome
    }

    /// Apply one queued transaction. Runs on the commit-queue worker, in lsn
    /// order, after the transaction's WAL batch is durable.
    pub(crate) fn apply_queued_txn(&self, lsn: u64, ops: &[WalRecord]) -> Result<()> {
        let started = std::time::Instant::now();
        let prev_capacity = self.inner.read().meta.slot_capacity;
        {
            let mut inner = self.inner.write();
            self.apply_records_locked(&mut inner, ops)?;
            // The lsn comes from the queue's monotonic counter, so this only
            // ever advances; replay applies the same records in the same order
            // and converges on the same water mark.
            inner.meta.last_applied_txn = inner.meta.last_applied_txn.max(lsn);
            // Only a capacity change must be persisted immediately:
            // `Vectors::open` rejects a file whose length disagrees with the
            // capacity in meta.bin, and that check runs before WAL replay can
            // reconcile anything. The slot water marks are all replayed from
            // the log, so saving them per transaction buys nothing but two
            // fsyncs per batch.
            let capacity_changed = inner.meta.slot_capacity != prev_capacity;
            let count = self
                .txns_since_last_save
                .fetch_add(1, AtomicOrdering::Relaxed)
                + 1;
            if capacity_changed || count >= META_SAVE_INTERVAL {
                inner.meta.save(&self.dir)?;
                self.txns_since_last_save.store(0, AtomicOrdering::Relaxed);
            }
        }
        let (upserts, deletes) = count_ops(ops);
        self.metrics
            .record_apply_txn(upserts, deletes, started.elapsed());
        Ok(())
    }

    /// Reserve a slot for every upsert of a not-yet-applied id.
    ///
    /// Reservations are pure in-memory bookkeeping: nothing durable is
    /// written here, so a reservation that never becomes a committed
    /// transaction leaves no trace on disk. They exist so that two concurrent
    /// transactions upserting the same new id agree on one slot, and so that
    /// `max_points` admission stays exact before anything is logged.
    pub(super) fn reserve_slots_locked(&self, inner: &mut StoreInner, ops: &[WalRecord]) {
        for op in ops {
            let WalRecord::Upsert { point } = op else {
                continue;
            };
            let id = PointId::from(point.id.as_str());
            if inner.reverse.contains_key(&id) || inner.reserved.contains_key(&id) {
                continue;
            }
            // Slots above the applied water mark are taken one per in-flight
            // reservation, so the reserved numbers are exactly the ones replay
            // assigns when it walks the same log order.
            let slot = (inner.meta.next_slot + inner.reserved.len() as u64) as u32;
            inner.reserved.insert(id, slot);
        }
    }

    /// Apply a WAL-backed batch of records.
    ///
    /// This is the crash-safe path for single-collection operations exposed by
    /// the engine; the log sequence number is assigned by the commit queue.
    pub fn apply_ops(&self, ops: &[WalRecord]) -> Result<()> {
        self.apply_txn(ops)
    }

    pub(super) fn apply_records_locked(&self, inner: &mut StoreInner, ops: &[WalRecord]) -> Result<()> {
        // One bump per batch: compaction compares this counter across its
        // temp-file rewrite phase to prove no write raced it.
        self.mutations.fetch_add(1, AtomicOrdering::Relaxed);
        for op in ops {
            match op {
                WalRecord::Upsert { point } => {
                    let point = point.to_point()?;
                    self.apply_upsert_locked(inner, &point)?;
                }
                WalRecord::Delete { point_id } => {
                    self.apply_delete_locked(inner, &PointId::from(point_id.as_str()))?
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for id in point_ids {
                        self.apply_delete_locked(inner, &PointId::from(id.as_str()))?;
                    }
                }
                WalRecord::SetPayload { slot, payload } => {
                    let parsed = match payload {
                        Some(s) => Some(serde_json::from_str::<Payload>(s)?),
                        None => None,
                    };
                    self.write_payload_at(*slot, parsed.as_ref())?;
                    self.refresh_filter_bitmap_locked(inner, *slot);
                }
                WalRecord::DeletePayloadKeys { slot, keys } => {
                    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
                    self.delete_keys_at(*slot, &key_refs)?;
                    self.refresh_filter_bitmap_locked(inner, *slot);
                }
                WalRecord::SetPayloadField { slot, key, value } => {
                    let parsed: serde_json::Value = serde_json::from_str(value)?;
                    let partial = Payload::from([(key.clone(), parsed)]);
                    self.merge_payload_at(*slot, &partial)?;
                    self.refresh_filter_bitmap_locked(inner, *slot);
                }
                // Checkpoint / quantization markers carry no separate mutation.
                WalRecord::Compact | WalRecord::Quantize | WalRecord::DropCollection => {}
            }
        }
        Ok(())
    }

    pub(super) fn apply_upsert_locked(&self, inner: &mut StoreInner, point: &VectorPoint) -> Result<()> {
        validate_point(&inner.meta, point)?;
        let slot = self.resolve_upsert_slot_locked(inner, &point.id)?;

        self.vectors.write_slot(slot, &point.vector)?;
        // Mirror the vector into quantized storage when ready.
        if let Some(q) = self.quant.read().as_ref() {
            if q.is_ready() {
                let _ = q.write_slot(slot, &point.vector);
            } else if matches!(
                q.config().quant_type,
                crate::types::QuantizationType::Scalar { .. }
                    | crate::types::QuantizationType::Binary { .. }
            ) {
                // Scalar/Binary are always ready after create/rebuild; attempt to
                // quantize even when `ready` flag races the build.
                let _ = q.write_slot(slot, &point.vector);
            }
        }
        self.write_payload_at(slot as u32, point.payload.as_ref())?;
        self.register_slot(slot as u32, &point.vector, inner.meta.segment_slots);
        inner
            .filter_bitmap
            .register_slot(slot as u32, point.payload.as_ref());
        inner
            .payload_indexes
            .register_slot(slot as u32, point.payload.as_ref());
        Ok(())
    }

    /// Resolve the slot an upsert writes to.
    ///
    /// A slot reserved by the commit queue's reserve step is filled in here:
    /// the reservation fixed the number — so replay rebuilds the same one —
    /// and this step writes the directory entry that makes the point visible.
    /// Reads never observe an id before this point, because only `reverse`
    /// publishes it.
    pub(super) fn resolve_upsert_slot_locked(&self, inner: &mut StoreInner, id: &PointId) -> Result<u64> {
        if let Some(slot) = inner.reserved.remove(id) {
            self.init_slot_locked(inner, id, slot)?;
            return Ok(slot as u64);
        }
        if let Some(slot) = inner.reverse.get(id).copied() {
            return Ok(slot as u64);
        }
        let slot = inner.meta.next_slot as u32;
        self.init_slot_locked(inner, id, slot)?;
        Ok(slot as u64)
    }

    /// Publish one new slot: grow the files if needed, write the id into
    /// `keys.bin`, and record the mapping and the counters.
    pub(super) fn init_slot_locked(&self, inner: &mut StoreInner, id: &PointId, slot: u32) -> Result<()> {
        self.ensure_capacity(inner, slot as u64 + 1)?;
        self.keys.append_key(slot as usize, &id.to_string())?;
        inner.reverse.insert(id.clone(), slot);
        // A reserved slot can sit beyond the applied high-water mark; the mark
        // only ever moves forward.
        inner.meta.next_slot = inner.meta.next_slot.max(slot as u64 + 1);
        inner.meta.live_count += 1;
        Ok(())
    }

    pub(super) fn apply_delete_locked(&self, inner: &mut StoreInner, id: &PointId) -> Result<()> {
        if let Some(slot) = inner.reverse.remove(id) {
            self.apply_delete_slot_locked(inner, slot)?;
        }
        Ok(())
    }

    pub(super) fn apply_delete_slot_locked(&self, inner: &mut StoreInner, slot: u32) -> Result<()> {
        self.payloads.set_tombstone(slot as usize, true)?;
        self.update_tombstone_bit(slot as usize, true);
        inner.filter_bitmap.unregister_slot(slot);
        inner.payload_indexes.unregister_slot(slot);
        inner.meta.live_count = inner.meta.live_count.saturating_sub(1);
        inner.meta.tombstone_count += 1;
        Ok(())
    }

    pub(super) fn update_tombstone_bit(&self, slot: usize, value: bool) {
        let next = self.tombstones.load().with_slot(slot, value);
        self.tombstones.store(Arc::new(next));
    }
}
