use std::collections::BTreeMap;
use std::time::Instant;

use crate::error::{Result, VectorSearchError};
use crate::filter;
use crate::storage::{WalRecord, WalTxn};
use crate::types::{Payload, PointId, VectorData, VectorFilter, VectorPoint};

use super::{Inner, KindStore};

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
        let slot = inner
            .reverse
            .get(&point_id.to_string())
            .copied()
            .ok_or_else(|| {
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
                    if let Some(slot) = inner.reverse.get(&point_id).copied() {
                        inner.tombstone_slot(slot);
                    }
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for point_id in point_ids {
                        if let Some(slot) = inner.reverse.get(&point_id).copied() {
                            inner.tombstone_slot(slot);
                        }
                    }
                }
                WalRecord::SetPayload { slot, payload } => {
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        point.payload = payload
                            .as_ref()
                            .map(|p| serde_json::from_str(p))
                            .transpose()?;
                    }
                }
                WalRecord::DeletePayloadKeys { slot, keys } => {
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        if let Some(payload) = &mut point.payload {
                            for key in keys {
                                payload.remove(&key);
                            }
                        }
                    }
                }
                WalRecord::SetPayloadField { slot, key, value } => {
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        let payload = point.payload.get_or_insert_with(Default::default);
                        let value: serde_json::Value = serde_json::from_str(&value)?;
                        payload.insert(key, value);
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
