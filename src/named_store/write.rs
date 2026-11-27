use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use crate::error::{Result, VectorSearchError};
use crate::filter;
use crate::storage::{WalRecord, WalTxn};
use crate::types::{Payload, PointId, VectorData, VectorFilter, VectorPoint};

use super::{Inner, NamedStore};

/// Decode a serialized payload to confirm it is valid before it reaches the
/// WAL. Applying already-validated records then only decodes trusted bytes.
fn validate_payload(payload: Option<&str>) -> Result<()> {
    if let Some(payload) = payload {
        serde_json::from_str::<Payload>(payload)?;
    }
    Ok(())
}

impl NamedStore {
    /// WAL-backed upsert of a batch of named points. Each point must carry
    /// the full declared space set; the upsert replaces the point's whole
    /// named map.
    pub fn upsert_batch(&self, points: &[VectorPoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let mut records = Vec::with_capacity(points.len());
        for point in points {
            records.push(named_upsert_record(point)?);
        }
        self.commit(records)
    }

    /// WAL-backed delete by id. Deleting a missing id is a no-op.
    pub fn delete(&self, point_id: &PointId) -> Result<()> {
        self.delete_batch(std::slice::from_ref(point_id))
    }

    /// WAL-backed delete of a batch of ids.
    pub fn delete_batch(&self, point_ids: &[PointId]) -> Result<()> {
        if point_ids.is_empty() {
            return Ok(());
        }
        let records = point_ids
            .iter()
            .map(|id| WalRecord::Delete {
                point_id: id.to_string(),
            })
            .collect();
        self.commit(records)
    }

    /// WAL-backed delete of every point matching `filter`.
    pub fn delete_by_filter(&self, filter: &VectorFilter) -> Result<u64> {
        let mut records = Vec::new();
        {
            let inner = self.inner.lock();
            for point in inner.points.iter().flatten() {
                if filter::matches(filter, &point.id, point.payload.as_ref())? {
                    records.push(WalRecord::Delete {
                        point_id: point.id.to_string(),
                    });
                }
            }
        }
        let deleted = records.len() as u64;
        if deleted > 0 {
            self.commit(records)?;
        }
        Ok(deleted)
    }

    /// Replace selected named spaces of an existing point, preserving the
    /// point id, payload, slot and every undeclared space.
    pub fn patch_named(
        &self,
        point_id: &PointId,
        vectors: BTreeMap<String, VectorData>,
    ) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        let mut normalized = vectors;
        for data in normalized.values_mut() {
            *data = data.clone().normalized();
        }
        self.commit(vec![WalRecord::PatchNamedVectors {
            id: point_id.to_string(),
            vectors: normalized,
        }])
    }

    /// Apply a coordinated transaction. Records are pre-split per collection
    /// but are still appended to this collection's WAL before being applied.
    pub fn apply_records(&self, records: &[WalRecord]) -> Result<()> {
        self.commit(records.to_vec())
    }

    /// Commit `records` to the WAL and apply them in order.
    fn commit(&self, records: Vec<WalRecord>) -> Result<()> {
        self.commit_records(records, Instant::now())
    }

    fn commit_records(&self, records: Vec<WalRecord>, started: Instant) -> Result<()> {
        let _gate = self.gate.lock();
        let mut inner = self.inner.lock();
        self.commit_records_locked(&mut inner, records, started)
    }

    fn commit_for_point(
        &self,
        point_id: &PointId,
        build: impl FnOnce(String) -> Result<Vec<WalRecord>>,
    ) -> Result<()> {
        let started = Instant::now();
        let _gate = self.gate.lock();
        let mut inner = self.inner.lock();
        if !inner.reverse.contains_key(point_id) {
            return Err(VectorSearchError::InvalidConfig(format!(
                "point {point_id} not found"
            )));
        }
        let records = build(point_id.to_string())?;
        self.commit_records_locked(&mut inner, records, started)
    }

    fn commit_records_locked(
        &self,
        inner: &mut Inner,
        records: Vec<WalRecord>,
        started: Instant,
    ) -> Result<()> {
        self.ensure_not_poisoned()?;
        // Validate the entire batch before the WAL append so an illegal
        // upsert/patch can never be persisted (and then fail on replay).
        self.validate_records(inner, &records)?;
        let upserts = records
            .iter()
            .filter(|record| matches!(record, WalRecord::UpsertVectors { .. }))
            .count() as u64;
        let deletes = records
            .iter()
            .filter(|record| {
                matches!(
                    record,
                    WalRecord::Delete { .. } | WalRecord::DeleteBatch { .. }
                )
            })
            .count() as u64;
        // The LSN is consumed only by a durable append: bumping it before the
        // append would leave a hole on failure and break segment-name
        // continuity. The gate serializes commits, so no race is possible.
        let txn_id = inner.next_lsn;
        self.wal.append(&WalTxn {
            txn_id,
            ops: records.clone(),
        })?;
        inner.next_lsn += 1;
        for record in records {
            match record {
                WalRecord::UpsertVectors {
                    id,
                    vectors,
                    payload,
                } => {
                    self.apply_upsert(inner, id, vectors, payload)?;
                }
                WalRecord::PatchNamedVectors { id, vectors } => {
                    self.apply_patch(inner, id, vectors)?;
                }
                WalRecord::Delete { point_id } => {
                    if let Some(slot) = inner.reverse.get_str(&point_id) {
                        inner.tombstone_slot(slot);
                    }
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for point_id in point_ids {
                        if let Some(slot) = inner.reverse.get_str(&point_id) {
                            inner.tombstone_slot(slot);
                        }
                    }
                }
                WalRecord::SetPayload { point_id, payload } => {
                    let Some(slot) = inner.reverse.get_str(&point_id) else {
                        continue;
                    };
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        let payload = payload
                            .as_ref()
                            .map(|p| serde_json::from_str(p))
                            .transpose()?;
                        point.replace_payload(payload);
                    }
                }
                WalRecord::DeletePayloadKeys { point_id, keys } => {
                    let Some(slot) = inner.reverse.get_str(&point_id) else {
                        continue;
                    };
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        if let Some(payload) = &mut point.payload {
                            for key in keys {
                                payload.remove(&key);
                            }
                        }
                        point.touch_payload();
                    }
                }
                WalRecord::SetPayloadField {
                    point_id,
                    key,
                    value,
                } => {
                    let Some(slot) = inner.reverse.get_str(&point_id) else {
                        continue;
                    };
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        let payload = point.payload.get_or_insert_with(Default::default);
                        let value: serde_json::Value = serde_json::from_str(&value)?;
                        payload.insert(key, value);
                        point.touch_payload();
                    }
                }
                WalRecord::Upsert { .. }
                | WalRecord::Compact
                | WalRecord::Quantize
                | WalRecord::DropCollection => {
                    return Err(VectorSearchError::Internal(
                        "named store commit received an unsupported record".to_string(),
                    ));
                }
            }
        }
        self.metrics()
            .record_apply_txn(upserts, deletes, started.elapsed());
        if self.wal.appended_bytes() >= self.storage.wal_checkpoint_bytes {
            if let Err(error) = self.checkpoint_from_locked(inner) {
                tracing::warn!(error = %error, "named store checkpoint failed");
            }
        }
        Ok(())
    }

    /// Validate a whole record batch before any byte reaches the WAL.
    ///
    /// Space names, kinds, dimensions, capacity and payload JSON are all
    /// checked here against a simulated live set. Replay of an appended batch
    /// skips these checks: only validated records are ever persisted.
    fn validate_records(&self, inner: &Inner, records: &[WalRecord]) -> Result<()> {
        let mut present: HashSet<String> = HashSet::new();
        let mut absent: HashSet<String> = HashSet::new();
        let mut projected_live = inner.live_count;
        let is_live = |id: &String, present: &HashSet<String>, absent: &HashSet<String>| {
            if present.contains(id) {
                true
            } else if absent.contains(id) {
                false
            } else {
                inner.reverse.contains_str(id)
            }
        };
        for record in records {
            match record {
                WalRecord::UpsertVectors {
                    id,
                    vectors,
                    payload,
                } => {
                    self.validate_vectors(vectors)?;
                    validate_payload(payload.as_deref())?;
                    if !is_live(id, &present, &absent) {
                        if let Some(cap) = self.max_points {
                            if projected_live >= cap {
                                return Err(VectorSearchError::CapacityExceeded {
                                    collection: self.name.clone(),
                                    max_points: cap,
                                    requested: projected_live.saturating_add(1),
                                });
                            }
                        }
                        present.insert(id.clone());
                        absent.remove(id);
                        projected_live = projected_live.saturating_add(1);
                    }
                }
                WalRecord::PatchNamedVectors { id, vectors } => {
                    self.validate_patch(vectors)?;
                    if !is_live(id, &present, &absent) {
                        return Err(VectorSearchError::InvalidConfig(format!(
                            "patch targets missing point {id}"
                        )));
                    }
                }
                WalRecord::Delete { point_id } => {
                    if is_live(point_id, &present, &absent) {
                        present.remove(point_id);
                        absent.insert(point_id.clone());
                        projected_live = projected_live.saturating_sub(1);
                    }
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for point_id in point_ids {
                        if is_live(point_id, &present, &absent) {
                            present.remove(point_id);
                            absent.insert(point_id.clone());
                            projected_live = projected_live.saturating_sub(1);
                        }
                    }
                }
                WalRecord::SetPayload { payload, .. } => {
                    validate_payload(payload.as_deref())?;
                }
                WalRecord::SetPayloadField { value, .. } => {
                    serde_json::from_str::<serde_json::Value>(value)?;
                }
                WalRecord::DeletePayloadKeys { .. } => {}
                WalRecord::Upsert { .. }
                | WalRecord::Compact
                | WalRecord::Quantize
                | WalRecord::DropCollection => {
                    return Err(VectorSearchError::Internal(
                        "named store commit received an unsupported record".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl NamedStore {
    /// Replace the payload for a single point.
    pub fn set_payload(&self, point_id: &PointId, payload: Payload) -> Result<()> {
        let payload = Some(serde_json::to_string(&payload)?);
        self.commit_for_point(point_id, |id| {
            Ok(vec![WalRecord::SetPayload {
                point_id: id,
                payload,
            }])
        })
    }

    /// Remove specific keys from a point's payload.
    pub fn delete_payload_keys(&self, point_id: &PointId, keys: &[String]) -> Result<()> {
        self.commit_for_point(point_id, |id| {
            Ok(vec![WalRecord::DeletePayloadKeys {
                point_id: id,
                keys: keys.to_vec(),
            }])
        })
    }

    /// Merge the given fields into a point's payload.
    pub fn set_payload_fields(&self, point_id: &PointId, fields: Payload) -> Result<()> {
        let encoded: Result<Vec<_>> = fields
            .into_iter()
            .map(|(key, value)| Ok((key, serde_json::to_string(&value)?)))
            .collect();
        let encoded = encoded?;
        self.commit_for_point(point_id, |id| {
            Ok(encoded
                .into_iter()
                .map(|(key, value)| WalRecord::SetPayloadField {
                    point_id: id.clone(),
                    key,
                    value,
                })
                .collect())
        })
    }
}

/// Build the WAL upsert record for a named point.
pub fn named_upsert_record(point: &VectorPoint) -> Result<WalRecord> {
    if point.vectors.is_empty() {
        return Err(VectorSearchError::InvalidConfig(
            "named collection upserts must carry named-space vectors".to_string(),
        ));
    }
    if point.vectors.len() == 1 && point.vectors.contains_key(crate::types::DEFAULT_VECTOR_NAME) {
        return Err(VectorSearchError::InvalidConfig(
            "named collection upserts must target declared named spaces".to_string(),
        ));
    }
    let mut vectors = point.vectors.clone();
    for data in vectors.values_mut() {
        *data = data.clone().normalized();
    }
    Ok(WalRecord::UpsertVectors {
        id: point.id.to_string(),
        vectors,
        payload: point
            .payload
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?,
    })
}
