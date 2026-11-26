use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::storage::{Wal, WalRecord, WalTxn};
use crate::types::{Payload, PointId, VectorData};

use super::{Inner, KindMeta, KindPoint, KindSnapshot, KindSnapshotPoint, KindStore};

impl KindStore {
    /// Create a new kind collection directory (`kind.bin` marker + WAL).
    pub fn create(
        dir: &Path,
        name: &str,
        config: &crate::types::CollectionConfig,
        storage: &crate::types::StorageConfig,
    ) -> Result<Arc<Self>> {
        config.validate_kind()?;
        std::fs::create_dir_all(dir)?;
        let meta = KindMeta {
            collection: name.to_string(),
            kind: config.kind,
            vector_size: config.vector_size,
            distance: config.distance,
            max_vectors_per_point: config.multi_config.map_or(0, |c| c.max_vectors_per_point),
            max_points: config.max_points,
        };
        let bytes = postcard::to_stdvec(&meta)?;
        std::fs::write(dir.join("kind.bin"), &bytes)?;
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        Ok(Arc::new(Self {
            dir: dir.to_path_buf(),
            name: name.to_string(),
            kind: config.kind,
            dim: config.vector_size,
            max_vectors_per_point: meta.max_vectors_per_point,
            distance: config.distance,
            max_points: meta.max_points,
            inner: Mutex::new(Inner::default()),
            wal,
            storage: storage.clone(),
            gate: Mutex::new(()),
            metrics: Metrics::default(),
            snapshot: dir.join("kind_snapshot.bin"),
        }))
    }

    /// Open an existing kind collection, rebuilding the in-memory state by
    /// replaying the WAL.
    pub fn open(dir: &Path, storage: &crate::types::StorageConfig) -> Result<Arc<Self>> {
        let bytes = std::fs::read(dir.join("kind.bin"))?;
        let meta: KindMeta = postcard::from_bytes(&bytes)?;
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        let snapshot_path = dir.join("kind_snapshot.bin");
        let snapshot = crate::storage::snapshot::read::<KindSnapshot>(&snapshot_path)?;
        let store = Arc::new(Self {
            dir: dir.to_path_buf(),
            name: meta.collection.clone(),
            kind: meta.kind,
            dim: meta.vector_size,
            max_vectors_per_point: meta.max_vectors_per_point,
            distance: meta.distance,
            max_points: meta.max_points,
            inner: Mutex::new(Inner::default()),
            wal,
            storage: storage.clone(),
            gate: Mutex::new(()),
            metrics: Metrics::default(),
            snapshot: snapshot_path,
        });
        let after = snapshot.as_ref().map(|snapshot| snapshot.lsn);
        if let Some(snapshot) = snapshot {
            store.restore_snapshot(snapshot)?;
        }
        let next_lsn = store.wal.replay_after(after, |txn| store.replay_txn(txn))?;
        if next_lsn > after.unwrap_or(0) {
            store.inner.lock().next_lsn = next_lsn.saturating_add(1);
        }
        Ok(store)
    }

    fn restore_snapshot(&self, snapshot: KindSnapshot) -> Result<()> {
        let mut inner = self.inner.lock();
        let capacity = snapshot.points.iter().map(|p| p.slot).max().unwrap_or(0) + 1;
        inner.points.resize(capacity as usize, None);
        for point in snapshot.points {
            let payload: Option<Payload> = point
                .payload
                .as_ref()
                .map(|p| serde_json::from_str(p))
                .transpose()?;
            self.validate_data(&point.data)?;
            if let VectorData::Sparse { indices, values } = &point.data {
                inner.index_sparse(point.slot, indices, values);
            }
            inner.reverse.insert(point.id.clone(), point.slot);
            inner.points[point.slot as usize] = Some(KindPoint {
                id: PointId::from(point.id),
                data: point.data,
                payload,
            });
            inner.live_count += 1;
        }
        inner.next_lsn = snapshot.lsn.saturating_add(1);
        Ok(())
    }

    /// Publish a snapshot of the live state and truncate the covered WAL.
    ///
    /// The gate blocks writers for the whole snapshot-and-truncate window: a
    /// commit appended between the two would otherwise exist only in the log
    /// that is about to be discarded.
    pub fn checkpoint(&self) -> Result<()> {
        let _gate = self.gate.lock();
        let inner = self.inner.lock();
        self.checkpoint_from_locked(&inner)
    }

    pub(super) fn checkpoint_from_locked(&self, inner: &Inner) -> Result<()> {
        let snapshot = KindSnapshot {
            lsn: inner.next_lsn.saturating_sub(1),
            points: inner
                .points
                .iter()
                .enumerate()
                .filter_map(|(slot, point)| {
                    point.as_ref().map(|point| KindSnapshotPoint {
                        slot: slot as u32,
                        id: point.id.to_string(),
                        data: point.data.clone(),
                        payload: point
                            .payload
                            .as_ref()
                            .and_then(|p| serde_json::to_string(p).ok()),
                    })
                })
                .collect(),
        };
        crate::storage::snapshot::write(&self.snapshot, &snapshot)?;
        self.wal.truncate()?;
        Ok(())
    }

    /// Reclaim tombstoned slots and publish the compacted state. Slot ids are
    /// renumbered in the snapshot; the old WAL is truncated in the same gate.
    pub fn compact(&self) -> Result<u64> {
        let _gate = self.gate.lock();
        let mut inner = self.inner.lock();
        let reclaimed = (inner.points.len() as u64).saturating_sub(inner.live_count);
        if reclaimed == 0 {
            return Ok(0);
        }
        let mut points = Vec::with_capacity(inner.live_count as usize);
        let mut reverse = HashMap::with_capacity(inner.live_count as usize);
        let mut inverted = HashMap::new();
        let mut norms = HashMap::new();
        for point in inner.points.drain(..).flatten() {
            let slot = points.len() as u32;
            if let VectorData::Sparse { indices, values } = &point.data {
                for (index, value) in indices.iter().zip(values.iter()) {
                    inverted.entry(*index).or_insert_with(Vec::new).push((slot, *value));
                }
                let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
                norms.insert(slot, norm);
            }
            reverse.insert(point.id.to_string(), slot);
            points.push(Some(point));
        }
        inner.points = points;
        inner.reverse = reverse;
        inner.inverted = inverted;
        inner.norms = norms;
        self.checkpoint_from_locked(&inner)?;
        Ok(reclaimed)
    }

    fn replay_txn(&self, txn: &WalTxn) -> Result<()> {
        let mut inner = self.inner.lock();
        for record in &txn.ops {
            match record {
                WalRecord::UpsertVectors {
                    id,
                    vectors,
                    payload,
                } => {
                    let data = super::write::kind_data(vectors)?;
                    self.apply_upsert(&mut inner, id.clone(), data, payload.clone())?;
                }
                WalRecord::Delete { point_id } => {
                    if let Some(slot) = inner.reverse.get(point_id).copied() {
                        inner.tombstone_slot(slot);
                    }
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for point_id in point_ids {
                        if let Some(slot) = inner.reverse.get(point_id).copied() {
                            inner.tombstone_slot(slot);
                        }
                    }
                }
                WalRecord::SetPayload { slot, payload } => {
                    if let Some(Some(point)) = inner.points.get_mut(*slot as usize) {
                        point.payload = payload
                            .as_ref()
                            .map(|p| serde_json::from_str(p))
                            .transpose()?;
                    }
                }
                WalRecord::DeletePayloadKeys { slot, keys } => {
                    if let Some(Some(point)) = inner.points.get_mut(*slot as usize) {
                        if let Some(payload) = &mut point.payload {
                            for key in keys {
                                payload.remove(key);
                            }
                        }
                    }
                }
                WalRecord::SetPayloadField { slot, key, value } => {
                    if let Some(Some(point)) = inner.points.get_mut(*slot as usize) {
                        let payload = point.payload.get_or_insert_with(Default::default);
                        let value: serde_json::Value = serde_json::from_str(value)?;
                        payload.insert(key.clone(), value);
                    }
                }
                WalRecord::Compact => {}
                WalRecord::Upsert { .. }
                | WalRecord::PatchNamedVectors { .. }
                | WalRecord::Quantize
                | WalRecord::DropCollection => {
                    return Err(VectorSearchError::CorruptData(format!(
                        "unexpected WAL record in kind collection {}",
                        self.dir.display()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Apply an already-validated upsert as a pure state transition.
    ///
    /// Shape, kind, capacity and payload JSON were checked in
    /// `write::validate_records` before the record was appended to the WAL.
    /// Replay therefore never re-validates and cannot reject a persisted
    /// transaction; decoding trusted bytes is the only fallible step left.
    pub(super) fn apply_upsert(
        &self,
        inner: &mut Inner,
        id: String,
        data: VectorData,
        payload: Option<String>,
    ) -> Result<()> {
        let payload: Option<Payload> = payload
            .as_ref()
            .map(|p| serde_json::from_str(p))
            .transpose()?;
        // Idempotent overwrite: a replayed or repeated upsert of the same id
        // replaces the stored point in place (the old slot is tombstoned and
        // its postings dropped).
        if let Some(&slot) = inner.reverse.get(&id) {
            inner.tombstone_slot(slot);
        }
        let slot = inner.points.len() as u32;
        if let VectorData::Sparse { indices, values } = &data {
            inner.index_sparse(slot, indices, values);
        }
        inner.reverse.insert(id.clone(), slot);
        inner.points.push(Some(KindPoint {
            id: PointId::from(id),
            data,
            payload,
        }));
        inner.live_count += 1;
        Ok(())
    }

    pub(super) fn validate_data(&self, data: &VectorData) -> Result<()> {
        if data.kind() != self.kind {
            return Err(VectorSearchError::InvalidConfig(format!(
                "collection kind is {}, payload kind is {}",
                self.kind.as_name(),
                data.kind().as_name()
            )));
        }
        data.validate(self.dim, self.max_vectors_per_point.max(1))
    }
}
