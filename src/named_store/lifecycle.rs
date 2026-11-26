use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Result, VectorSearchError};
use crate::storage::{Wal, WalRecord, WalTxn};
use crate::types::{CollectionConfig, Payload, PointId, VectorData, VectorKind, VectorSpaceConfig};

use super::{Inner, NamedMeta, NamedPoint, NamedSpaceMeta, NamedStore};

impl NamedStore {
    /// Create a new named collection directory (`named.bin` marker + WAL).
    pub fn create(
        dir: &Path,
        name: &str,
        spaces: &BTreeMap<String, VectorSpaceConfig>,
        storage: &crate::types::StorageConfig,
    ) -> Result<Arc<Self>> {
        CollectionConfig::with_named(spaces.clone()).validate_kind()?;
        std::fs::create_dir_all(dir)?;
        let meta = NamedMeta {
            collection: name.to_string(),
            spaces: spaces
                .iter()
                .map(|(name, config)| {
                    (
                        name.clone(),
                        NamedSpaceMeta {
                            kind: config.kind,
                            vector_size: config.vector_size,
                            distance: config.distance,
                        },
                    )
                })
                .collect(),
        };
        let bytes = postcard::to_stdvec(&meta)?;
        std::fs::write(dir.join("named.bin"), &bytes)?;
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        Ok(Arc::new(Self {
            dir: dir.to_path_buf(),
            spaces: meta.spaces,
            inner: Mutex::new(Inner::default()),
            wal,
            storage: storage.clone(),
        }))
    }

    /// Open an existing named collection, rebuilding the in-memory state by
    /// replaying the WAL.
    pub fn open(dir: &Path, storage: &crate::types::StorageConfig) -> Result<Arc<Self>> {
        let bytes = std::fs::read(dir.join("named.bin"))?;
        let meta: NamedMeta = postcard::from_bytes(&bytes)?;
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        let store = Arc::new(Self {
            dir: dir.to_path_buf(),
            spaces: meta.spaces,
            inner: Mutex::new(Inner::default()),
            wal,
            storage: storage.clone(),
        });
        let next_lsn = store.wal.replay(|txn| store.replay_txn(txn))?;
        store.inner.lock().next_lsn = next_lsn.saturating_add(1);
        Ok(store)
    }

    fn replay_txn(&self, txn: &WalTxn) -> Result<()> {
        let mut inner = self.inner.lock();
        for record in &txn.ops {
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
                | WalRecord::UpsertKind { .. }
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
        self.validate_vectors(&vectors)?;
        // Idempotent overwrite: a replayed or repeated upsert of the same id
        // replaces the stored point in place (the old slot is tombstoned and
        // its postings dropped).
        if let Some(&slot) = inner.reverse.get(&id) {
            inner.tombstone_slot(slot, &self.sparse_space_names());
        }
        let slot = inner.points.len() as u32;
        for (space, data) in &vectors {
            if let VectorData::Sparse { indices, values } = data {
                let space_index = inner.inverted.entry(space.clone()).or_default();
                for (index, value) in indices.iter().zip(values.iter()) {
                    space_index.entry(*index).or_default().push((slot, *value));
                }
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
            data.validate(space.vector_size, 1)?;
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
