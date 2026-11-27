use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use crate::error::{Result, VectorSearchError};
use crate::filter;
use crate::storage::{WalRecord, WalTxn};
use crate::types::{Payload, PointId, VectorData, VectorFilter, VectorPoint};

use super::{Inner, KindStore};

/// Decode a serialized payload to confirm it is valid before it reaches the
/// WAL. Applying already-validated records then only decodes trusted bytes.
fn validate_payload(payload: Option<&str>) -> Result<()> {
    if let Some(payload) = payload {
        serde_json::from_str::<Payload>(payload)?;
    }
    Ok(())
}

impl KindStore {
    /// WAL-backed upsert of a batch of points.
    pub fn upsert_batch(&self, points: &[VectorPoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let mut records = Vec::with_capacity(points.len());
        for point in points {
            records.push(kind_upsert_record(point)?);
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

    /// Apply a coordinated transaction. The records are split per collection
    /// by the engine, but durability still belongs to this store: they are
    /// appended to the collection WAL before being applied in memory.
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

    /// Resolve the target slot and log/apply the payload edit in one gate. A
    /// concurrent overwrite cannot retire the slot between lookup and commit.
    fn commit_for_point(
        &self,
        point_id: &PointId,
        build: impl FnOnce(u32) -> Result<Vec<WalRecord>>,
    ) -> Result<()> {
        let started = Instant::now();
        let _gate = self.gate.lock();
        let mut inner = self.inner.lock();
        let slot = inner.reverse.get(point_id).ok_or_else(|| {
            VectorSearchError::Internal(format!("point {point_id} not found"))
        })?;
        let records = build(slot)?;
        self.commit_records_locked(&mut inner, records, started)
    }

    fn commit_records_locked(
        &self,
        inner: &mut Inner,
        records: Vec<WalRecord>,
        started: Instant,
    ) -> Result<()> {
        self.ensure_not_poisoned()?;
        // Validate everything that can reject a request (shape, kind,
        // capacity, payload JSON) before the WAL append. Once appended,
        // apply is a pure state transition that cannot fail on bad input.
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
        let txn_id = inner.next_lsn;
        inner.next_lsn += 1;
        self.wal.append(&WalTxn {
            txn_id,
            ops: records.clone(),
        })?;
        for record in records {
            match record {
                WalRecord::UpsertVectors {
                    id,
                    vectors,
                    payload,
                } => {
                    let data = kind_data(&vectors)?;
                    self.apply_upsert(inner, id, data, payload)?;
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
                WalRecord::SetPayload { slot, payload } => {
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        let payload = payload
                            .as_ref()
                            .map(|p| serde_json::from_str(p))
                            .transpose()?;
                        point.replace_payload(payload);
                    }
                }
                WalRecord::DeletePayloadKeys { slot, keys } => {
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        if let Some(payload) = &mut point.payload {
                            for key in keys {
                                payload.remove(&key);
                            }
                        }
                        point.touch_payload();
                    }
                }
                WalRecord::SetPayloadField { slot, key, value } => {
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        let payload = point.payload.get_or_insert_with(Default::default);
                        let value: serde_json::Value = serde_json::from_str(&value)?;
                        payload.insert(key, value);
                        point.touch_payload();
                    }
                }
                WalRecord::Upsert { .. }
                | WalRecord::PatchNamedVectors { .. }
                | WalRecord::Compact
                | WalRecord::Quantize
                | WalRecord::DropCollection => {
                    return Err(VectorSearchError::Internal(
                        "kind store commit received an unsupported record".to_string(),
                    ));
                }
            }
        }
        self.metrics()
            .record_apply_txn(upserts, deletes, started.elapsed());
        if self.wal.appended_bytes() >= self.storage.wal_checkpoint_bytes {
            if let Err(error) = self.checkpoint_from_locked(inner) {
                tracing::warn!(error = %error, "kind store checkpoint failed");
            }
        }
        Ok(())
    }

    /// Validate a whole record batch before any byte reaches the WAL.
    ///
    /// Capacity is checked against a simulated live set so a batch that adds
    /// more points than `max_points` allows is rejected as a unit. Replay of
    /// an appended batch never re-runs these checks: only validated records
    /// can be persisted, and applying them is a pure state transition.
    fn validate_records(&self, inner: &Inner, records: &[WalRecord]) -> Result<()> {
        // Only the batch's own effect is tracked here: the live directory is
        // consulted per record, so a commit no longer clones every point id.
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
                    let data = kind_data(vectors)?;
                    self.validate_data(&data)?;
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
                | WalRecord::PatchNamedVectors { .. }
                | WalRecord::Compact
                | WalRecord::Quantize
                | WalRecord::DropCollection => {
                    return Err(VectorSearchError::Internal(
                        "kind store commit received an unsupported record".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl KindStore {
    /// Replace the payload for a single point.
    pub fn set_payload(&self, point_id: &PointId, payload: Payload) -> Result<()> {
        let payload = Some(serde_json::to_string(&payload)?);
        self.commit_for_point(point_id, |slot| {
            Ok(vec![WalRecord::SetPayload { slot, payload }])
        })
    }

    /// Remove specific keys from a point's payload.
    pub fn delete_payload_keys(&self, point_id: &PointId, keys: &[String]) -> Result<()> {
        self.commit_for_point(point_id, |slot| {
            Ok(vec![WalRecord::DeletePayloadKeys {
                slot,
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
        self.commit_for_point(point_id, |slot| {
            Ok(encoded
                .into_iter()
                .map(|(key, value)| WalRecord::SetPayloadField { slot, key, value })
                .collect())
        })
    }
}

/// Extract the single default-space payload expected by a kind collection.
pub(super) fn kind_data(vectors: &BTreeMap<String, VectorData>) -> Result<VectorData> {
    match vectors.len() {
        1 => vectors
            .get(crate::types::DEFAULT_VECTOR_NAME)
            .cloned()
            .ok_or_else(|| {
                VectorSearchError::InvalidConfig(
                    "kind collections require the default vector space".to_string(),
                )
            }),
        _ => Err(VectorSearchError::InvalidConfig(
            "kind collections accept exactly one vector space".to_string(),
        )),
    }
}

/// Build the WAL upsert record for a non-dense point.
pub fn kind_upsert_record(point: &VectorPoint) -> Result<WalRecord> {
    let data = kind_data(&point.vectors)?;
    let data = data.normalized();
    Ok(WalRecord::UpsertVectors {
        id: point.id.to_string(),
        vectors: BTreeMap::from([(crate::types::DEFAULT_VECTOR_NAME.to_string(), data)]),
        payload: point
            .payload
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?,
    })
}
