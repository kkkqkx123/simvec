use std::sync::Arc;

use super::assert_not_applying;
use super::queue::{CommitQueue, APPLY_WAIT_TIMEOUT};
use super::WalRecord;
use super::{CollectionStore, StoreInner};

use crate::error::{Result, VectorSearchError};
use crate::types::{Payload, PointId, VectorPoint};

use super::store_open::{count_ops, validate_admission_locked, validate_point};
use super::*;

/// Whether this id already has a water mark at or above `lsn`.
fn stale_for_id(inner: &StoreInner, id: &PointId, lsn: u64) -> bool {
    inner
        .versions
        .get_value(id)
        .is_some_and(|version| version >= lsn)
}

impl CollectionStore {
    /// Apply a WAL-backed transaction.
    ///
    /// Four steps, of which only the reserve and apply hold the store write
    /// lock (never across an fsync):
    ///
    /// 1. **reserve** — validate every op (no invalid data may be logged) and
    ///    record new ids in the reservation set, in one short `inner.write()`
    ///    critical section that also fixes the queue position;
    /// 2. **flush** — the commit-queue worker assigns contiguous LSNs in
    ///    queue order and drains the whole backlog into one WAL batch, so
    ///    concurrent writers share a single fsync;
    /// 3. **apply** — the same worker applies the batch in lsn order under a
    ///    short `inner.write()`, assigning slot numbers in that same order;
    /// 4. **complete** — the worker settles the ticket and wakes this caller.
    ///
    /// Nothing is applied before its record is durable, and the call returns
    /// only once the transaction is applied, so `get`/`search` see exactly
    /// what a successful return promises.
    pub fn apply_txn(&self, ops: &[WalRecord]) -> Result<()> {
        self.ensure_not_poisoned()?;
        assert_not_applying();
        // Validation happens before anything is logged: a logged record that
        // fails while being applied would sit in the WAL forever and make
        // every later open fail.
        {
            let inner = self.inner.read();
            for op in ops {
                match op {
                    WalRecord::Upsert { point } => {
                        let point = point.to_point()?;
                        validate_point(&inner.meta, &point)?;
                        self.validate_payload_size(point.payload.as_ref())?;
                    }
                    WalRecord::SetPayload {
                        payload: Some(json),
                        ..
                    } => {
                        self.validate_json_bytes(json.as_bytes())?;
                    }
                    WalRecord::SetPayloadField { value, .. } => {
                        // The merged shape is validated for the whole batch
                        // before the WAL append; here only the JSON encoding
                        // of the incoming value has to be rejected early.
                        self.validate_json_bytes(value.as_bytes())?;
                    }
                    _ => {}
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
            // The id reservations and the queue position are fixed in this
            // one critical section: together they are the log order. The
            // worker assigns LSNs and slot numbers later, in that same order.
            CommitQueue::enqueue(&self.queue, &self.self_weak, ops.to_vec())?
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
        let data = self.data_arc();
        let prev_capacity = self.inner.read().meta.slot_capacity;
        {
            let mut inner = self.inner.write();
            self.apply_records_locked(&mut inner, lsn, ops)?;
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
            if capacity_changed || count >= self.storage.meta_save_interval {
                inner.meta.save(data.dir())?;
                self.txns_since_last_save.store(0, AtomicOrdering::Relaxed);
            }
            let segments =
                super::segments::SegmentSet::new(inner.meta.next_slot, inner.meta.segment_slots);
            if data.segments.load().immutable_end != segments.immutable_end {
                data.segments.store(Arc::new(segments));
            }
        }
        let (upserts, deletes) = count_ops(ops);
        self.metrics
            .record_apply_txn(upserts, deletes, started.elapsed());
        Ok(())
    }

    /// Reserve every upsert of a not-yet-applied id.
    ///
    /// Reservations are pure in-memory presence bookkeeping: nothing durable
    /// is written here, so a reservation that never becomes a committed
    /// transaction leaves no trace on disk. They exist so that two concurrent
    /// transactions upserting the same new id agree on one slot, and so that
    /// `max_points` admission stays exact before anything is logged. Slot
    /// numbers are deliberately not assigned here: they are assigned at apply
    /// time in log order, which is the only numbering live and replay can
    /// both derive.
    pub(super) fn reserve_slots_locked(&self, inner: &mut StoreInner, ops: &[WalRecord]) {
        for op in ops {
            let WalRecord::Upsert { point } = op else {
                continue;
            };
            let id = PointId::from(point.id.as_str());
            if inner.reverse.contains_key(&id) || inner.reserved.contains_id(&id) {
                continue;
            }
            inner.reserved.insert_value(&id, ());
        }
    }

    /// Assign contiguous LSNs for one flushed batch, in queue order. The only
    /// bulk assigner is the commit-queue worker, so the assignment is dense
    /// by construction.
    pub(super) fn assign_batch_lsns(&self, count: u64) -> u64 {
        let mut inner = self.inner.write();
        let base = inner.next_lsn;
        inner.next_lsn += count;
        base
    }

    /// Roll back a batch whose WAL append failed: return its LSNs and drop
    /// its reservations. Safe because the worker is the only bulk assigner
    /// and no marker can be allocated concurrently with a flushed batch.
    pub(super) fn abort_unlogged_batch(&self, base: u64, batch: &[super::queue::Ticket]) {
        let mut inner = self.inner.write();
        if inner.next_lsn == base + batch.len() as u64 {
            inner.next_lsn = base;
        } else {
            debug_assert!(
                false,
                "wal batch abort raced an lsn assignment; holes are not allowed"
            );
        }
        for ticket in batch {
            for op in &ticket.ops {
                if let WalRecord::Upsert { point } = op {
                    inner
                        .reserved
                        .remove_value(&PointId::from(point.id.as_str()));
                }
            }
        }
    }

    /// Apply a WAL-backed batch of records.
    ///
    /// This is the crash-safe path for single-collection operations exposed by
    /// the engine; the log sequence number is assigned by the commit queue.
    pub fn apply_ops(&self, ops: &[WalRecord]) -> Result<()> {
        self.apply_txn(ops)
    }

    pub(super) fn apply_records_locked(
        &self,
        inner: &mut StoreInner,
        lsn: u64,
        ops: &[WalRecord],
    ) -> Result<()> {
        // Second line of defence after the generation water mark: a segment
        // put back by hand can carry lsns the store has already applied.
        if lsn <= inner.meta.last_applied_txn {
            return Ok(());
        }
        // One bump per batch: compaction compares this counter across its
        // temp-file rewrite phase to prove no write raced it.
        self.mutations.fetch_add(1, AtomicOrdering::Relaxed);
        // Versions advance once the batch is applied, so several operations
        // on the same id inside one transaction all take effect. A replay of
        // that transaction later still sees the version and is skipped.
        let mut touched: Vec<PointId> = Vec::new();
        let mut applied_slots: Vec<u32> = Vec::new();
        for op in ops {
            match op {
                WalRecord::Upsert { point } => {
                    let id = PointId::from(point.id.as_str());
                    if stale_for_id(inner, &id, lsn) {
                        continue;
                    }
                    let point = point.to_point()?;
                    // An overwrite moves the id to a new slot and tombstones
                    // the old one, so both segments have dirty bytes to flush.
                    let previous = inner.reverse.get(&id);
                    self.apply_upsert_locked(inner, &point, lsn)?;
                    if let Some(old) = previous {
                        applied_slots.push(old);
                    }
                    if let Some(slot) = inner.reverse.get(&id) {
                        applied_slots.push(slot);
                    }
                    touched.push(id);
                }
                WalRecord::Delete { point_id } => {
                    let id = PointId::from(point_id.as_str());
                    if stale_for_id(inner, &id, lsn) {
                        continue;
                    }
                    if let Some(slot) = inner.reverse.get(&id) {
                        applied_slots.push(slot);
                    }
                    self.apply_delete_locked(inner, &id)?;
                    touched.push(id);
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for id in point_ids {
                        let id = PointId::from(id.as_str());
                        if stale_for_id(inner, &id, lsn) {
                            continue;
                        }
                        if let Some(slot) = inner.reverse.get(&id) {
                            applied_slots.push(slot);
                        }
                        self.apply_delete_locked(inner, &id)?;
                        touched.push(id);
                    }
                }
                WalRecord::SetPayload { point_id, payload } => {
                    // Payload ops resolve the id exactly like the reserve-time
                    // validation did; a missing or tombstoned id is a no-op so
                    // replay and live can never disagree on a slot.
                    let Some(slot) = self.payload_slot_of_locked(inner, point_id) else {
                        continue;
                    };
                    let parsed = match payload {
                        Some(s) => Some(serde_json::from_str::<Payload>(s)?),
                        None => None,
                    };
                    self.write_payload_at(slot, parsed.as_ref(), lsn)?;
                    self.refresh_filter_bitmap_locked(inner, slot);
                    applied_slots.push(slot);
                }
                WalRecord::DeletePayloadKeys { point_id, keys } => {
                    let Some(slot) = self.payload_slot_of_locked(inner, point_id) else {
                        continue;
                    };
                    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
                    self.delete_keys_at(slot, &key_refs, lsn)?;
                    self.refresh_filter_bitmap_locked(inner, slot);
                    applied_slots.push(slot);
                }
                WalRecord::SetPayloadField {
                    point_id,
                    key,
                    value,
                } => {
                    let Some(slot) = self.payload_slot_of_locked(inner, point_id) else {
                        continue;
                    };
                    let parsed: serde_json::Value = serde_json::from_str(value)?;
                    let partial = Payload::from([(key.clone(), parsed)]);
                    self.merge_payload_at(slot, &partial, lsn)?;
                    self.refresh_filter_bitmap_locked(inner, slot);
                    applied_slots.push(slot);
                }
                // Checkpoint / quantization markers carry no separate mutation.
                WalRecord::Compact | WalRecord::Quantize => {}
                // Kind and named payloads never enter a dense collection's log.
                WalRecord::UpsertVectors { .. } | WalRecord::PatchNamedVectors { .. } => {
                    return Err(VectorSearchError::CorruptData(
                        "non-dense upsert record replayed in a dense collection".to_string(),
                    ));
                }
            }
        }
        for id in touched {
            inner.versions.insert_value(&id, lsn);
        }
        if !applied_slots.is_empty() && inner.meta.segment_slots != 0 {
            let segment_slots = inner.meta.segment_slots;
            let needed = applied_slots
                .iter()
                .map(|s| super::segments::segment_index(*s as u64, segment_slots) + 1)
                .max()
                .unwrap_or(0);
            inner.ensure_segment_len(needed);
            for slot in applied_slots {
                let index = super::segments::segment_index(slot as u64, segment_slots);
                if let Some(cell) = inner.segment_applied.get_mut(index) {
                    *cell = (*cell).max(lsn);
                }
            }
        }
        Ok(())
    }

    /// Apply one upsert.
    ///
    /// An upsert of an id that already lives somewhere never rewrites that
    /// slot: the new bytes are appended at a fresh slot (tail, or a reclaimed
    /// dead slot of the appendable tail) and the previous slot is tombstoned.
    /// A published segment therefore only ever receives appends, which is what
    /// lets a later generation reuse its files instead of rewriting them.
    ///
    /// Replay is the exception and writes back into the slot the id already
    /// owns: it reconstructs a state the same log produced, so re-deriving the
    /// slot keeps the log's numbering stable instead of moving every
    /// re-applied id forward. A re-applied record is exactly a record whose
    /// bytes may be missing from the files, so rewriting that slot is also
    /// what makes replay idempotent.
    pub(super) fn apply_upsert_locked(
        &self,
        inner: &mut StoreInner,
        point: &VectorPoint,
        lsn: u64,
    ) -> Result<()> {
        validate_point(&inner.meta, point)?;
        let data = self.data_arc();
        let id = point.id.clone();
        // The id keeps its water mark but not its slot. `reserved` only marks
        // presence for new ids, so a consumed reservation is cleared here.
        let previous = inner.reverse.get(&id);
        inner.reserved.remove_value(&id);
        let slot = match previous {
            Some(old) if inner.replaying => old,
            _ => self.allocate_slot_locked(inner)?,
        };
        match previous {
            Some(old) if old == slot => {
                // Replay path: the slot and the live count are unchanged.
                data.keys.append_key(slot as usize, &id.to_string())?;
            }
            Some(old) => {
                // Retiring the old slot also unregisters its filter/index
                // entries and heals the segment graph that held it.
                self.apply_delete_slot_locked(inner, old)?;
                self.publish_slot_locked(inner, &id, slot)?;
            }
            None => self.publish_slot_locked(inner, &id, slot)?,
        }

        data.vectors.write_slot(slot as u64, point.vector())?;
        // Mirror the vector into quantized storage when ready.
        if let Some(q) = data.quant.read().as_ref() {
            let attempt = q.is_ready()
                || matches!(
                    q.config().quant_type,
                    crate::types::QuantizationType::Scalar { .. }
                        | crate::types::QuantizationType::Binary { .. }
                );
            if attempt {
                if let Err(e) = q.write_slot(slot as u64, point.vector()) {
                    // A stale code can drop the true nearest neighbour from
                    // the candidate set even when rescore is on. Disable the
                    // quantized path persistently instead of only warning.
                    tracing::warn!(slot, error = %e, "quant write_slot failed; disabling quantized search until rebuild");
                    if let Err(mark) = q.mark_stale() {
                        tracing::error!(error = %mark, "failed to persist quant not-ready state");
                    }
                }
            }
        }
        self.write_payload_at(slot, point.payload.as_ref(), lsn)?;
        self.register_slot(slot, point.vector(), inner.meta.segment_slots);
        inner.filter_bitmap.register_slot(slot, point.payload.as_ref());
        inner.payload_indexes.register_slot(slot, point.payload.as_ref());
        Ok(())
    }

    /// Resolve a slot for the next upsert.
    ///
    /// A dead slot of the **appendable tail** is reclaimed first; otherwise the
    /// slot address space grows by one. Frozen segments are never reclaimed
    /// from: they are published, may be shared with a later generation
    /// byte-for-byte, and stay read-only.
    ///
    /// The scan is a pure function of the tombstone table plus `free_cursor`,
    /// and the cursor only advances through applied (logged) upserts, so live
    /// application and WAL replay pick the identical slot for the identical
    /// record sequence.
    pub(super) fn allocate_slot_locked(&self, inner: &mut StoreInner) -> Result<u32> {
        if inner.meta.tombstone_count > 0 {
            let segment_slots = inner.meta.segment_slots;
            let tail_start =
                super::segments::tail_base(inner.meta.next_slot, segment_slots) as usize;
            let end = inner.meta.next_slot as usize;
            if end > tail_start {
                let cursor = (inner.free_cursor as usize).clamp(tail_start, end);
                let tombstones = self.data_arc().tombstones.load();
                let found = tombstones
                    .first_free_in(cursor, end)
                    .or_else(|| tombstones.first_free_in(tail_start, cursor));
                if let Some(slot) = found {
                    inner.free_cursor = slot as u32 + 1;
                    self.reclaim_slot_locked(inner, slot as u32);
                    return Ok(slot as u32);
                }
            }
        }
        let slot = inner.meta.next_slot as u32;
        self.ensure_capacity(inner, slot as u64 + 1)?;
        inner.meta.next_slot = slot as u64 + 1;
        Ok(slot)
    }

    /// Take a dead slot back into service: clear its tombstone and undo the
    /// delete accounting. Its key, vector and payload are overwritten by the
    /// caller.
    fn reclaim_slot_locked(&self, inner: &mut StoreInner, slot: u32) {
        self.update_tombstone_bit(slot as usize, false);
        inner.meta.tombstone_count = inner.meta.tombstone_count.saturating_sub(1);
    }

    /// Publish one slot for `id`: write the key and record the mapping.
    ///
    /// Capacity and the slot frontier are the allocator's business, so this
    /// only handles the directory side of a slot that already exists.
    pub(super) fn publish_slot_locked(
        &self,
        inner: &mut StoreInner,
        id: &PointId,
        slot: u32,
    ) -> Result<()> {
        self.data_arc()
            .keys
            .append_key(slot as usize, &id.to_string())?;
        inner.reverse.insert(id, slot);
        inner.meta.live_count += 1;
        Ok(())
    }

    /// Resolve the id a payload mutation targets. Shared by the pre-logging
    /// validation and the log application, so both agree on the slot. A
    /// missing or tombstoned id resolves to nothing and the mutation is a
    /// no-op — exactly what replay must do with the same record.
    fn payload_slot_of_locked(&self, inner: &StoreInner, point_id: &str) -> Option<u32> {
        let id = PointId::from(point_id);
        let slot = inner.reverse.get(&id)?;
        if self.data_arc().tombstones.load().bit(slot as usize) {
            return None;
        }
        Some(slot)
    }

    pub(super) fn apply_delete_locked(&self, inner: &mut StoreInner, id: &PointId) -> Result<()> {
        if let Some(slot) = inner.reverse.remove(id) {
            self.apply_delete_slot_locked(inner, slot)?;
        }
        Ok(())
    }

    pub(super) fn apply_delete_slot_locked(&self, inner: &mut StoreInner, slot: u32) -> Result<()> {
        self.update_tombstone_bit(slot as usize, true);
        self.heal_hnsw_after_delete(inner, slot);
        inner.filter_bitmap.unregister_slot(slot);
        inner.payload_indexes.unregister_slot(slot);
        inner.meta.live_count = inner.meta.live_count.saturating_sub(1);
        inner.meta.tombstone_count += 1;
        Ok(())
    }

    /// Repair the one segment graph that contains a deleted slot. Replay does
    /// not see a published graph at open, so this is a live-only optimization;
    /// the persisted graph keeps the node and searches still filter it out.
    fn heal_hnsw_after_delete(&self, inner: &StoreInner, slot: u32) {
        if inner.meta.segment_slots == 0 {
            return;
        }
        let data = self.data_arc();
        let graphs = data.hnsw_segments.load();
        let Some(target) = graphs
            .iter()
            .find(|segment| slot >= segment.start && slot < segment.end)
        else {
            return;
        };
        let tombstones = data.tombstones.load();
        let vectors = data.vectors.snapshot();
        target.index.heal_deleted(
            slot - target.start,
            &vectors,
            inner.meta.segment_slots,
            target.start,
            |global| tombstones.bit(global as usize),
        );
    }

    pub(super) fn update_tombstone_bit(&self, slot: usize, value: bool) {
        let data = self.data_arc();
        let next = data.tombstones.load().with_slot(slot, value);
        data.tombstones.store(Arc::new(next));
        if value {
            // Counted here, not in the delete paths: replay and undo also flip
            // bits, and only a real accumulate-then-repair cycle should wake
            // the whole-store graph repair.
            self.hnsw_tombstones_pending
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}
