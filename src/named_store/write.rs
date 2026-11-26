use crate::error::{Result, VectorSearchError};
use crate::filter;
use crate::storage::{WalRecord, WalTxn};
use crate::types::{Payload, PointId, VectorFilter, VectorPoint};

use super::NamedStore;

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

    /// Apply already-encoded WAL records without re-logging them. Used by
    /// the engine's coordinated-transaction path.
    pub fn apply_records(&self, records: &[WalRecord]) -> Result<()> {
        let mut inner = self.inner.lock();
        for record in records {
            match record {
                WalRecord::UpsertNamed {
                    id,
                    vectors,
                    payload,
                } => {
                    self.apply_upsert(&mut inner, id.clone(), vectors.clone(), payload.clone())?;
                }
                WalRecord::Delete { point_id } => {
                    if let Some(slot) = inner.reverse.get(point_id).copied() {
                        inner.tombstone_slot(slot, &self.sparse_space_names());
                    }
                }
                _ => {
                    return Err(VectorSearchError::Internal(
                        "named store transaction received an unsupported record".to_string(),
                    ));
                }
            }
        }
        inner.next_lsn += 1;
        Ok(())
    }

    /// Commit `records` to the WAL and apply them in order.
    fn commit(&self, records: Vec<WalRecord>) -> Result<()> {
        let started = std::time::Instant::now();
        let upserts = records
            .iter()
            .filter(|record| matches!(record, WalRecord::UpsertNamed { .. }))
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
        let mut inner = self.inner.lock();
        let txn_id = inner.next_lsn;
        inner.next_lsn += 1;
        self.wal.append(&WalTxn {
            txn_id,
            ops: records.clone(),
        })?;
        for record in records {
            match record {
                WalRecord::UpsertNamed {
                    id,
                    vectors,
                    payload,
                } => {
                    self.apply_upsert(&mut inner, id, vectors, payload)?;
                }
                WalRecord::Delete { point_id } => {
                    if let Some(slot) = inner.reverse.get(&point_id).copied() {
                        inner.tombstone_slot(slot, &self.sparse_space_names());
                    }
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for point_id in point_ids {
                        if let Some(slot) = inner.reverse.get(&point_id).copied() {
                            inner.tombstone_slot(slot, &self.sparse_space_names());
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
                | WalRecord::UpsertKind { .. }
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
        let checkpoint = self.wal.appended_bytes() >= self.storage.wal_checkpoint_bytes;
        drop(inner);
        if checkpoint {
            if let Err(error) = self.checkpoint() {
                tracing::warn!(error = %error, "named store checkpoint failed");
            }
        }
        Ok(())
    }
}

impl NamedStore {
    /// Replace the payload for a single point.
    pub fn set_payload(&self, point_id: &PointId, payload: Payload) -> Result<()> {
        let slot = self.require_slot(point_id)?;
        self.commit(vec![WalRecord::SetPayload {
            slot,
            payload: Some(serde_json::to_string(&payload)?),
        }])
    }

    /// Remove specific keys from a point's payload.
    pub fn delete_payload_keys(&self, point_id: &PointId, keys: &[String]) -> Result<()> {
        let slot = self.require_slot(point_id)?;
        self.commit(vec![WalRecord::DeletePayloadKeys {
            slot,
            keys: keys.to_vec(),
        }])
    }

    /// Merge the given fields into a point's payload.
    pub fn set_payload_fields(&self, point_id: &PointId, fields: Payload) -> Result<()> {
        let slot = self.require_slot(point_id)?;
        let mut records = Vec::with_capacity(fields.len());
        for (key, value) in fields {
            records.push(WalRecord::SetPayloadField {
                slot,
                key,
                value: serde_json::to_string(&value)?,
            });
        }
        self.commit(records)
    }

    fn require_slot(&self, point_id: &PointId) -> Result<u32> {
        let inner = self.inner.lock();
        inner
            .reverse
            .get(&point_id.to_string())
            .copied()
            .ok_or_else(|| VectorSearchError::InvalidConfig(format!("point {point_id} not found")))
    }
}

/// Build the WAL upsert record for a named point.
pub fn named_upsert_record(point: &VectorPoint) -> Result<WalRecord> {
    if point.named.is_empty() {
        return Err(VectorSearchError::InvalidConfig(
            "named collection upserts must carry named-space vectors".to_string(),
        ));
    }
    if !point.vector.is_empty() || point.data.is_some() {
        return Err(VectorSearchError::InvalidConfig(
            "named collection upserts must not carry single-space payloads".to_string(),
        ));
    }
    Ok(WalRecord::UpsertNamed {
        id: point.id.to_string(),
        vectors: point.named.clone(),
        payload: point
            .payload
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?,
    })
}
