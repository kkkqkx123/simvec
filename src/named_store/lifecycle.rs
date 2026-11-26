use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::storage::{Wal, WalRecord, WalTxn};
use crate::types::{CollectionConfig, Payload, PointId, VectorData, VectorKind, VectorSpaceConfig};

use super::{
    Inner, NamedMeta, NamedPoint, NamedSnapshot, NamedSnapshotPoint, NamedSpaceMeta, NamedStore,
};

impl NamedStore {
    /// Create a new named collection directory (`named.bin` marker + WAL).
    pub fn create(
        dir: &Path,
        name: &str,
        spaces: &BTreeMap<String, VectorSpaceConfig>,
        max_points: Option<u64>,
        storage: &crate::types::StorageConfig,
    ) -> Result<Arc<Self>> {
        let mut config = CollectionConfig::with_named(spaces.clone());
        config.max_points = max_points;
        config.validate_kind()?;
        std::fs::create_dir_all(dir)?;
        let meta = NamedMeta {
            collection: name.to_string(),
            spaces: spaces
                .iter()
                .map(|(name, config)| {
                    (
                        name.clone(),
                        NamedSpaceMeta {
                            kind: config.kind(),
                            vector_size: config.vector_size(),
                            distance: config.distance(),
                            multi_config: config.multi_config(),
                        },
                    )
                })
                .collect(),
            max_points,
        };
        let bytes = postcard::to_stdvec(&meta)?;
        std::fs::write(dir.join("named.bin"), &bytes)?;
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        Ok(Arc::new(Self {
            dir: dir.to_path_buf(),
            name: name.to_string(),
            spaces: meta.spaces,
            max_points,
            inner: Mutex::new(Inner::default()),
            wal,
            storage: storage.clone(),
            gate: Mutex::new(()),
            metrics: Metrics::default(),
            snapshot: dir.join("named_snapshot.bin"),
        }))
    }

    /// Open an existing named collection, rebuilding the in-memory state by
    /// replaying the WAL.
    pub fn open(dir: &Path, storage: &crate::types::StorageConfig) -> Result<Arc<Self>> {
        let bytes = std::fs::read(dir.join("named.bin"))?;
        let meta: NamedMeta = postcard::from_bytes(&bytes)?;
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        let snapshot_path = dir.join("named_snapshot.bin");
        let snapshot = crate::storage::snapshot::read::<NamedSnapshot>(&snapshot_path)?;
        let store = Arc::new(Self {
            dir: dir.to_path_buf(),
            name: meta.collection.clone(),
            spaces: meta.spaces,
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

    fn restore_snapshot(&self, snapshot: NamedSnapshot) -> Result<()> {
        let mut inner = self.inner.lock();
        let capacity = snapshot.points.iter().map(|p| p.slot).max().unwrap_or(0) + 1;
        inner.points.resize(capacity as usize, None);
        for point in snapshot.points {
            let payload: Option<Payload> = point
                .payload
                .as_ref()
                .map(|p| serde_json::from_str(p))
                .transpose()?;
            for (space, data) in &point.vectors {
                let Some(config) = self.spaces.get(space) else {
                    return Err(VectorSearchError::CorruptData(format!(
                        "snapshot references unknown space {space}"
                    )));
                };
                if data.kind() != config.kind {
                    return Err(VectorSearchError::CorruptData(format!(
                        "snapshot space {space} kind mismatch"
                    )));
                }
                let max_vectors = config
                    .multi_config
                    .map_or(1, |multi| multi.max_vectors_per_point);
                data.validate(config.vector_size, max_vectors)?;
                if let VectorData::Sparse { indices, values } = data {
                    inner.index_sparse(space, point.slot, indices, values);
                }
            }
            inner.reverse.insert(point.id.clone(), point.slot);
            inner.points[point.slot as usize] = Some(NamedPoint {
                id: PointId::from(point.id),
                vectors: point.vectors,
                payload,
            });
            inner.live_count += 1;
        }
        inner.next_lsn = snapshot.lsn.saturating_add(1);
        Ok(())
    }

    /// Publish a snapshot of the live state and truncate the covered WAL.
    ///
    /// The gate covers both operations so a concurrent commit cannot land in
    /// the log segment that is about to be discarded.
    pub fn checkpoint(&self) -> Result<()> {
        let _gate = self.gate.lock();
        let inner = self.inner.lock();
        self.checkpoint_from_locked(&inner)
    }

    pub(super) fn checkpoint_from_locked(&self, inner: &Inner) -> Result<()> {
        let snapshot = NamedSnapshot {
            lsn: inner.next_lsn.saturating_sub(1),
            points: inner
                .points
                .iter()
                .enumerate()
                .filter_map(|(slot, point)| {
                    point.as_ref().map(|point| NamedSnapshotPoint {
                        slot: slot as u32,
                        id: point.id.to_string(),
                        vectors: point.vectors.clone(),
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

    /// Reclaim tombstoned slots, rebuilding every sparse posting index with
    /// the compacted slot ids before publishing a new snapshot.
    pub fn compact(&self) -> Result<u64> {
        let _gate = self.gate.lock();
        let mut inner = self.inner.lock();
        let reclaimed = (inner.points.len() as u64).saturating_sub(inner.live_count);
        if reclaimed == 0 {
            return Ok(0);
        }
        let mut points = Vec::with_capacity(inner.live_count as usize);
        let mut reverse = HashMap::with_capacity(inner.live_count as usize);
        let mut inverted: HashMap<String, HashMap<u32, Vec<(u32, f32)>>> = HashMap::new();
        let mut norms: HashMap<String, HashMap<u32, f32>> = HashMap::new();
        for point in inner.points.drain(..).flatten() {
            let slot = points.len() as u32;
            for (space, data) in &point.vectors {
                if let VectorData::Sparse { indices, values } = data {
                    let space_index = inverted.entry(space.clone()).or_default();
                    for (index, value) in indices.iter().zip(values.iter()) {
                        space_index.entry(*index).or_default().push((slot, *value));
                    }
                    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
                    norms.entry(space.clone()).or_default().insert(slot, norm);
                }
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
                    self.apply_upsert(&mut inner, id.clone(), vectors.clone(), payload.clone())?;
                }
                WalRecord::PatchNamedVectors { id, vectors } => {
                    self.apply_patch(&mut inner, id.clone(), vectors.clone())?;
                }
                WalRecord::Delete { point_id } => {
                    if let Some(slot) = inner.reverse.get(point_id).copied() {
                        inner.tombstone_slot(slot, &self.sparse_space_names());
                    }
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for point_id in point_ids {
                        if let Some(slot) = inner.reverse.get(point_id).copied() {
                            inner.tombstone_slot(slot, &self.sparse_space_names());
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
                | WalRecord::Quantize
                | WalRecord::DropCollection => {
                    return Err(VectorSearchError::CorruptData(format!(
                        "unexpected WAL record in named collection {}",
                        self.dir.display()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Apply an already-validated upsert as a pure state transition.
    ///
    /// Space names, kinds, dimensions, capacity and payload JSON were checked
    /// in `write::validate_records` before the WAL append, so replay never
    /// re-validates a persisted transaction.
    pub(super) fn apply_upsert(
        &self,
        inner: &mut Inner,
        id: String,
        vectors: BTreeMap<String, VectorData>,
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
            inner.tombstone_slot(slot, &self.sparse_space_names());
        }
        let slot = inner.points.len() as u32;
        for (space, data) in &vectors {
            if let VectorData::Sparse { indices, values } = data {
                inner.index_sparse(space, slot, indices, values);
            }
        }
        inner.reverse.insert(id.clone(), slot);
        inner.points.push(Some(NamedPoint {
            id: PointId::from(id),
            vectors,
            payload,
        }));
        inner.live_count += 1;
        Ok(())
    }

    /// Apply an already-validated named-space patch as a pure state transition.
    ///
    /// Validation ran in `write::validate_records` before the WAL append.
    /// The slot lookup is a state read, not a request validation: it is a
    /// no-op-safe guard for a point removed by an earlier record in the batch.
    pub(super) fn apply_patch(
        &self,
        inner: &mut Inner,
        id: String,
        vectors: BTreeMap<String, VectorData>,
    ) -> Result<()> {
        let Some(slot) = inner.reverse.get(&id).copied() else {
            return Ok(());
        };
        let mut point = inner.points[slot as usize]
            .take()
            .ok_or_else(|| VectorSearchError::CorruptData(format!("point {id} has no slot")))?;
        for (space, _data) in &vectors {
            if let Some(old) = point.vectors.get(space) {
                inner.remove_sparse(space, slot, old);
            }
        }
        for (space, data) in vectors {
            if let VectorData::Sparse { indices, values } = &data {
                inner.index_sparse(&space, slot, indices, values);
            }
            point.vectors.insert(space, data);
        }
        inner.points[slot as usize] = Some(point);
        Ok(())
    }

    pub(super) fn validate_patch(&self, vectors: &BTreeMap<String, VectorData>) -> Result<()> {
        for (name, data) in vectors {
            let Some(space) = self.spaces.get(name) else {
                return Err(VectorSearchError::InvalidConfig(format!(
                    "unknown named-vector space {name}"
                )));
            };
            if data.kind() != space.kind {
                return Err(VectorSearchError::InvalidConfig(format!(
                    "space {name} kind is {}, payload kind is {}",
                    space.kind.as_name(),
                    data.kind().as_name()
                )));
            }
            let max_vectors = space
                .multi_config
                .map_or(1, |multi| multi.max_vectors_per_point);
            data.validate(space.vector_size, max_vectors)?;
        }
        Ok(())
    }

    pub(super) fn validate_vectors(&self, vectors: &BTreeMap<String, VectorData>) -> Result<()> {
        if vectors.len() != self.spaces.len() {
            return Err(VectorSearchError::InvalidConfig(format!(
                "named point carries {} spaces but the collection declares {}",
                vectors.len(),
                self.spaces.len()
            )));
        }
        for (name, data) in vectors {
            let Some(space) = self.spaces.get(name) else {
                return Err(VectorSearchError::InvalidConfig(format!(
                    "unknown named-vector space {name}"
                )));
            };
            if data.kind() != space.kind {
                return Err(VectorSearchError::InvalidConfig(format!(
                    "space {name} kind is {}, payload kind is {}",
                    space.kind.as_name(),
                    data.kind().as_name()
                )));
            }
            let max_vectors = space
                .multi_config
                .map_or(1, |multi| multi.max_vectors_per_point);
            data.validate(space.vector_size, max_vectors)?;
        }
        Ok(())
    }

    pub(super) fn sparse_space_names(&self) -> Vec<&str> {
        self.spaces
            .iter()
            .filter(|(_, space)| space.kind == VectorKind::Sparse)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}
