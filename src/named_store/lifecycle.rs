use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::storage::point_blobs::{encode_named, encode_vector, rewrite_named, PointBlobs};
use crate::storage::postings::Postings;
use crate::storage::{IdMap, Wal, WalRecord, WalTxn};
use crate::types::{CollectionConfig, Payload, PointId, VectorData, VectorKind, VectorSpaceConfig};

use super::{
    Inner, NamedMeta, NamedPoint, NamedSnapshot, NamedSnapshotPoint, NamedSpaceMeta, NamedStore,
};

/// File holding the append-only point vectors.
const BLOBS_FILE: &str = "points.bin";
/// Directory holding the per-space postings.
const POSTINGS_DIR: &str = "postings";

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
        let wal = Wal::open_or_create(&dir.join("wal"), storage.wal_segment_bytes)?;
        let blobs = PointBlobs::create(&dir.join(BLOBS_FILE))?;
        let space_order = space_order(&meta.spaces);
        let postings = open_postings(dir, &meta.spaces, &space_order, None)?;
        Ok(Arc::new(Self {
            dir: dir.to_path_buf(),
            name: name.to_string(),
            spaces: meta.spaces,
            space_order,
            max_points,
            inner: Mutex::new(Inner::default()),
            wal,
            storage: storage.clone(),
            gate: Mutex::new(()),
            metrics: Metrics::default(),
            snapshot: dir.join("named_snapshot.bin"),
            blobs,
            postings,
        }))
    }

    /// Open an existing named collection: map the vector file, load the
    /// per-space dictionaries, then replay what the snapshot does not cover.
    pub fn open(dir: &Path, storage: &crate::types::StorageConfig) -> Result<Arc<Self>> {
        let bytes = std::fs::read(dir.join("named.bin"))?;
        let meta: NamedMeta = postcard::from_bytes(&bytes)?;
        let wal = Wal::open_or_create(&dir.join("wal"), storage.wal_segment_bytes)?;
        let snapshot_path = dir.join("named_snapshot.bin");
        let snapshot = crate::storage::snapshot::read::<NamedSnapshot>(&snapshot_path)?;
        let snapshot_lsn = snapshot.as_ref().map(|snapshot| snapshot.lsn);
        let blobs = PointBlobs::open(
            &dir.join(BLOBS_FILE),
            snapshot.as_ref().map(|snapshot| snapshot.blobs_len),
        )?;
        let space_order = space_order(&meta.spaces);
        let postings = open_postings(dir, &meta.spaces, &space_order, snapshot_lsn)?;
        let rebuild_postings = postings.values().any(|space| !space.was_restored());
        let store = Arc::new(Self {
            dir: dir.to_path_buf(),
            name: meta.collection.clone(),
            spaces: meta.spaces,
            space_order,
            max_points: meta.max_points,
            inner: Mutex::new(Inner::default()),
            wal,
            storage: storage.clone(),
            gate: Mutex::new(()),
            metrics: Metrics::default(),
            snapshot: snapshot_path,
            blobs,
            postings,
        });
        if let Some(snapshot) = snapshot {
            store.restore_snapshot(snapshot, rebuild_postings)?;
        }
        let next_lsn = store
            .wal
            .replay_after(snapshot_lsn, |txn| store.replay_txn(txn))?;
        if next_lsn > snapshot_lsn.unwrap_or(0) {
            store.inner.lock().next_lsn = next_lsn.saturating_add(1);
        }
        Ok(store)
    }

    fn restore_snapshot(&self, snapshot: NamedSnapshot, rebuild_postings: bool) -> Result<()> {
        let mut inner = self.inner.lock();
        let capacity = snapshot.points.iter().map(|p| p.slot).max().unwrap_or(0) + 1;
        inner.points.resize(capacity as usize, None);
        for point in snapshot.points {
            let payload: Option<Payload> = point
                .payload
                .as_ref()
                .map(|p| serde_json::from_str(p))
                .transpose()?;
            // Structural check per space; contents were validated before the
            // point was logged.
            {
                let reader = self.blobs.reader();
                let bytes = reader.bytes(point.blob).ok_or_else(|| {
                    VectorSearchError::CorruptData(format!(
                        "snapshot slot {} references a missing vector blob",
                        point.slot
                    ))
                })?;
                for (index, entry) in crate::storage::point_blobs::named_entries(bytes)
                    .ok_or_else(|| {
                        VectorSearchError::CorruptData(format!(
                            "snapshot slot {} holds an unreadable vector blob",
                            point.slot
                        ))
                    })?
                {
                    let space = self.space_at(index).ok_or_else(|| {
                        VectorSearchError::CorruptData(format!(
                            "snapshot slot {} references unknown space index {index}",
                            point.slot
                        ))
                    })?;
                    let view = crate::storage::point_blobs::decode_vector(entry).ok_or_else(|| {
                        VectorSearchError::CorruptData(format!(
                            "snapshot slot {} holds an undecodable {space} vector",
                            point.slot
                        ))
                    })?;
                    self.check_view(space, &view)?;
                    if rebuild_postings {
                        if let Some(postings) = self.postings_of(space) {
                            inner.index_view(postings, space, point.slot, view)?;
                        }
                    }
                }
            }
            inner.reverse.insert_str(&point.id, point.slot);
            for (index, norm) in point.norms {
                if let Some(space) = self.space_at(index) {
                    inner
                        .norms
                        .entry(space.to_string())
                        .or_default()
                        .insert(point.slot, norm);
                }
            }
            inner.points[point.slot as usize] = Some(NamedPoint {
                id: PointId::from(point.id),
                blob: point.blob,
                payload,
            });
            inner.live_count += 1;
        }
        inner.next_lsn = snapshot.lsn.saturating_add(1);
        Ok(())
    }

    /// Publish a snapshot of the live state and drop the WAL it covers.
    ///
    /// Same ordering contract as `KindStore`: data files, then the posting
    /// manifests, then the snapshot that references them, then the log.
    pub fn checkpoint(&self) -> Result<()> {
        let _gate = self.gate.lock();
        let inner = self.inner.lock();
        self.checkpoint_from_locked(&inner)
    }

    pub(super) fn checkpoint_from_locked(&self, inner: &Inner) -> Result<()> {
        let lsn = inner.next_lsn.saturating_sub(1);
        self.blobs.sync();
        for postings in self.postings.values() {
            postings.flush(Some(lsn))?;
        }
        let snapshot = NamedSnapshot {
            lsn,
            blobs_len: self.blobs.len(),
            points: inner
                .points
                .iter()
                .enumerate()
                .filter_map(|(slot, point)| {
                    point.as_ref().map(|point| NamedSnapshotPoint {
                        slot: slot as u32,
                        id: point.id.to_string(),
                        blob: point.blob,
                        payload: point
                            .payload
                            .as_ref()
                            .and_then(|p| serde_json::to_string(p).ok()),
                        norms: self
                            .sparse_spaces()
                            .filter_map(|(space, index)| {
                                inner
                                    .norms
                                    .get(space)
                                    .and_then(|norms| norms.get(&(slot as u32)))
                                    .map(|norm| (index, *norm))
                            })
                            .collect(),
                    })
                })
                .collect(),
        };
        crate::storage::snapshot::write(&self.snapshot, &snapshot)?;
        self.wal.truncate_through(lsn)?;
        Ok(())
    }

    /// Reclaim tombstoned slots and publish the compacted state. Both derived
    /// files are rewritten because slots are renumbered.
    pub fn compact(&self) -> Result<u64> {
        let _gate = self.gate.lock();
        let mut inner = self.inner.lock();
        let reclaimed = (inner.points.len() as u64).saturating_sub(inner.live_count);
        if reclaimed == 0 {
            return Ok(0);
        }

        let staging = self.dir.join("points.compact");
        let _ = std::fs::remove_file(&staging);
        let staging_blobs = PointBlobs::create(&staging)?;
        let old_norms = std::mem::take(&mut inner.norms);
        let mut points: Vec<Option<NamedPoint>> = Vec::with_capacity(inner.live_count as usize);
        let mut reverse = IdMap::with_capacity(inner.live_count as usize);
        let mut norms: HashMap<String, HashMap<u32, f32>> = HashMap::new();
        {
            let reader = self.blobs.reader();
            for (old_slot, point) in std::mem::take(&mut inner.points).into_iter().enumerate() {
                let Some(point) = point else { continue };
                let slot = points.len() as u32;
                let bytes = reader.bytes(point.blob).ok_or_else(|| {
                    VectorSearchError::CorruptData(format!(
                        "point {} references a missing vector blob during compaction",
                        point.id
                    ))
                })?;
                for (space, space_norms) in &old_norms {
                    if let Some(norm) = space_norms.get(&(old_slot as u32)) {
                        norms.entry(space.clone()).or_default().insert(slot, *norm);
                    }
                }
                let blob = staging_blobs.append(bytes)?;
                reverse.insert(&point.id, slot);
                points.push(Some(NamedPoint {
                    id: point.id,
                    blob,
                    payload: point.payload,
                }));
            }
        }
        staging_blobs.sync();
        self.blobs.replace_from(&staging, staging_blobs.len())?;

        for postings in self.postings.values() {
            postings.reset()?;
        }
        {
            let reader = self.blobs.reader();
            for (slot, point) in points.iter().enumerate() {
                let Some(point) = point else { continue };
                let Some(bytes) = reader.bytes(point.blob) else {
                    continue;
                };
                for (index, entry) in
                    crate::storage::point_blobs::named_entries(bytes).unwrap_or_default()
                {
                    let Some(space) = self.space_at(index) else {
                        continue;
                    };
                    let Some(postings) = self.postings_of(space) else {
                        continue;
                    };
                    if let Some(view) = crate::storage::point_blobs::decode_vector(entry) {
                        inner.index_view(postings, space, slot as u32, view)?;
                    }
                }
            }
        }

        inner.points = points;
        inner.reverse = reverse;
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
                    if let Some(slot) = inner.reverse.get_str(point_id) {
                        inner.tombstone_slot(slot);
                    }
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for point_id in point_ids {
                        if let Some(slot) = inner.reverse.get_str(point_id) {
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
                WalRecord::Upsert { .. } | WalRecord::Quantize | WalRecord::DropCollection => {
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
        // replaces the stored point (old slot tombstoned, bytes and postings
        // left for compaction).
        if let Some(slot) = inner.reverse.get_str(&id) {
            inner.tombstone_slot(slot);
        }
        let slot = inner.points.len() as u32;
        let blob = self.blobs.append(&encode_named(&vectors, &self.space_order))?;
        for (space, data) in &vectors {
            if let (VectorData::Sparse { indices, values }, Some(postings)) =
                (data, self.postings_of(space))
            {
                inner.index_sparse(postings, space, slot, indices, values)?;
            }
        }
        inner.reverse.insert_str(&id, slot);
        inner.points.push(Some(NamedPoint {
            id: PointId::from(id),
            blob,
            payload,
        }));
        inner.live_count += 1;
        Ok(())
    }

    /// Apply an already-validated named-space patch.
    ///
    /// The point is re-slotted instead of being rewritten in place: postings
    /// are keyed by `(slot, term)`, so keeping the slot would leave the old
    /// terms mixed with the new ones, and a stale posting on a live slot is
    /// indistinguishable from a real one at query time. The new slot then
    /// carries the merged blob, with the untouched spaces copied byte for byte
    /// and re-inverted on the new slot.
    ///
    /// Validation ran in `write::validate_records` before the WAL append.
    pub(super) fn apply_patch(
        &self,
        inner: &mut Inner,
        id: String,
        vectors: BTreeMap<String, VectorData>,
    ) -> Result<()> {
        let Some(slot) = inner.reverse.get_str(&id) else {
            return Ok(());
        };
        let Some(existing) = inner.points[slot as usize].clone() else {
            return Err(VectorSearchError::CorruptData(format!(
                "point {id} has no slot"
            )));
        };
        let mut replacements: BTreeMap<u16, Vec<u8>> = BTreeMap::new();
        for (space, data) in &vectors {
            let index = self.order_of(space).ok_or_else(|| {
                VectorSearchError::CorruptData(format!("patch targets unknown space {space}"))
            })?;
            replacements.insert(index, encode_vector(data));
        }
        let merged = {
            let reader = self.blobs.reader();
            let bytes = reader.bytes(existing.blob).ok_or_else(|| {
                VectorSearchError::CorruptData(format!("point {id} references a missing blob"))
            })?;
            rewrite_named(bytes, &replacements).ok_or_else(|| {
                VectorSearchError::CorruptData(format!("point {id} holds an unreadable blob"))
            })?
        };

        inner.tombstone_slot(slot);
        let new_slot = inner.points.len() as u32;
        let blob = self.blobs.append(&merged)?;
        // Every sparse space of the merged blob is inverted on the new slot,
        // including the ones the patch did not touch: their postings still
        // point at the retired slot.
        for (index, entry) in
            crate::storage::point_blobs::named_entries(&merged).unwrap_or_default()
        {
            let Some(space) = self.space_at(index) else {
                continue;
            };
            let Some(postings) = self.postings_of(space) else {
                continue;
            };
            if let Some(view) = crate::storage::point_blobs::decode_vector(entry) {
                inner.index_view(postings, space, new_slot, view)?;
            }
        }
        inner.reverse.insert_str(&id, new_slot);
        inner.points.push(Some(NamedPoint {
            id: existing.id,
            blob,
            payload: existing.payload,
        }));
        inner.live_count += 1;
        Ok(())
    }

    /// Validate a patch against the declared spaces before it reaches the WAL.
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

    /// Validate a whole point against the declared space set.
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

    /// Structural validation of a stored space vector (snapshot restore).
    fn check_view(&self, space: &str, data: &crate::types::VectorView<'_>) -> Result<()> {
        use crate::types::VectorView;
        let Some(meta) = self.spaces.get(space) else {
            return Err(VectorSearchError::CorruptData(format!(
                "stored payload references unknown space {space}"
            )));
        };
        if data.kind() != meta.kind {
            return Err(VectorSearchError::CorruptData(format!(
                "space {space} kind is {}, stored payload kind is {}",
                meta.kind.as_name(),
                data.kind().as_name()
            )));
        }
        match data {
            VectorView::Dense(values) => {
                if values.len() != meta.vector_size {
                    return Err(VectorSearchError::CorruptData(format!(
                        "space {space} holds {} values, expected {}",
                        values.len(),
                        meta.vector_size
                    )));
                }
            }
            VectorView::Sparse { indices, values } => {
                if indices.len() != values.len() {
                    return Err(VectorSearchError::CorruptData(format!(
                        "space {space} holds a ragged sparse vector"
                    )));
                }
                if let Some(&last) = indices.last() {
                    if last as usize >= meta.vector_size {
                        return Err(VectorSearchError::CorruptData(format!(
                            "space {space} term {last} exceeds the vocabulary bound {}",
                            meta.vector_size
                        )));
                    }
                }
            }
            VectorView::Multi { flat: _, dim } => {
                if *dim != meta.vector_size {
                    return Err(VectorSearchError::CorruptData(format!(
                        "space {space} multi dimension {dim} differs from the declared {}",
                        meta.vector_size
                    )));
                }
            }
        }
        Ok(())
    }

}

/// Assign each declared space a stable index. `BTreeMap` iterates in name
/// order, so the mapping survives a restart without being persisted.
fn space_order(spaces: &BTreeMap<String, NamedSpaceMeta>) -> BTreeMap<String, u16> {
    spaces
        .keys()
        .enumerate()
        .map(|(index, name)| (name.clone(), index as u16))
        .collect()
}

/// Open one postings store per sparse space, under `postings/<index>`.
fn open_postings(
    dir: &Path,
    spaces: &BTreeMap<String, NamedSpaceMeta>,
    order: &BTreeMap<String, u16>,
    snapshot_lsn: Option<u64>,
) -> Result<HashMap<String, Postings>> {
    let mut postings = HashMap::new();
    for (name, meta) in spaces {
        if meta.kind != VectorKind::Sparse {
            continue;
        }
        let index = order.get(name).copied().unwrap_or(0);
        let space_dir = dir.join(POSTINGS_DIR).join(format!("{index:03}"));
        postings.insert(name.clone(), Postings::open(&space_dir, snapshot_lsn)?);
    }
    Ok(postings)
}
