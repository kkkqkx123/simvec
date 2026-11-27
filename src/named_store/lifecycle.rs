use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::storage::point_blobs::{
    encode_named, encode_vector, rewrite_named, BlobRef, PointBlobs,
};
use crate::storage::postings::Postings;
use crate::storage::{snapshot, CollectionLock, Generation, IdMap, Wal, WalRecord, WalTxn};
use crate::types::{CollectionConfig, Payload, PointId, VectorData, VectorKind, VectorSpaceConfig};

use super::{
    Inner, NamedMeta, NamedPoint, NamedSnapshot, NamedSnapshotPoint, NamedSpaceMeta, NamedStore,
};

/// File holding the append-only point vectors.
const BLOBS_FILE: &str = "points.bin";
/// File holding the append-only JSON payloads.
const PAYLOADS_FILE: &str = "payloads.bin";
/// Directory holding the per-space postings.
const POSTINGS_DIR: &str = "postings";

/// State staged for a compacted generation.
struct StagedNamedState {
    points: Vec<Option<NamedPoint>>,
    reverse: IdMap,
    norms: HashMap<String, HashMap<u32, f32>>,
    blobs_len: u64,
    payloads_len: u64,
}

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
        let dir_lock = CollectionLock::acquire(dir)?;
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
        let generation = Generation::create(dir)?;
        let wal = Wal::open_or_create(&dir.join("wal"), storage.wal_segment_bytes)?;
        wal.bind_generation(generation.id());
        let data_dir = generation.path();
        let blobs = PointBlobs::create(&data_dir.join(BLOBS_FILE))?;
        let payloads = PointBlobs::create(&data_dir.join(PAYLOADS_FILE))?;
        let space_order = space_order(&meta.spaces);
        let postings = open_postings(&data_dir, &meta.spaces, &space_order, None)?;
        Ok(Arc::new(Self {
            dir: dir.to_path_buf(),
            _dir_lock: dir_lock,
            generation_poisoned: AtomicBool::new(false),
            name: name.to_string(),
            spaces: meta.spaces,
            space_order,
            max_points,
            inner: Mutex::new(Inner::default()),
            wal,
            storage: storage.clone(),
            gate: Mutex::new(()),
            metrics: Metrics::default(),
            generation: parking_lot::RwLock::new(generation),
            blobs: arc_swap::ArcSwap::from(Arc::new(blobs)),
            payloads: arc_swap::ArcSwap::from(Arc::new(payloads)),
            postings: arc_swap::ArcSwap::from(Arc::new(postings)),
        }))
    }

    /// Open an existing named collection: map the vector file, load the
    /// per-space dictionaries, then replay what the snapshot does not cover.
    pub fn open(dir: &Path, storage: &crate::types::StorageConfig) -> Result<Arc<Self>> {
        let dir_lock = CollectionLock::acquire(dir)?;
        let bytes = std::fs::read(dir.join("named.bin"))?;
        let meta: NamedMeta = postcard::from_bytes(&bytes)?;
        let generation = Generation::open(dir)?;
        let wal = Wal::open_or_create(&dir.join("wal"), storage.wal_segment_bytes)?;
        wal.bind_generation(generation.id());
        let data_dir = generation.path();
        let snapshot_path = data_dir.join("named_snapshot.bin");
        let snapshot = snapshot::read::<NamedSnapshot>(&snapshot_path)?;
        if generation.lsn() == 0 && snapshot.is_none() {
            // A newly created collection has no checkpoint yet.
        } else {
            let lsn = snapshot
                .as_ref()
                .ok_or_else(|| {
                    VectorSearchError::CorruptData(format!(
                        "published generation {} has no snapshot",
                        generation.lsn()
                    ))
                })?
                .lsn;
            if lsn != generation.lsn() {
                return Err(VectorSearchError::CorruptData(format!(
                    "snapshot lsn {lsn} differs from generation lsn {}",
                    generation.lsn()
                )));
            }
        }
        let snapshot_lsn = snapshot.as_ref().map(|snapshot| snapshot.lsn);
        let blobs = PointBlobs::open(
            &data_dir.join(BLOBS_FILE),
            snapshot.as_ref().map(|snapshot| snapshot.blobs_len),
        )?;
        // No published snapshot means no reachable payload handle; the file
        // is entirely an uncommitted tail.
        let payloads = PointBlobs::open(
            &data_dir.join(PAYLOADS_FILE),
            Some(
                snapshot
                    .as_ref()
                    .map_or(0, |snapshot| snapshot.payloads_len),
            ),
        )?;
        let space_order = space_order(&meta.spaces);
        let postings = open_postings(&data_dir, &meta.spaces, &space_order, snapshot_lsn)?;
        let rebuild_postings = postings.values().any(|space| !space.was_restored());
        let store = Arc::new(Self {
            dir: dir.to_path_buf(),
            _dir_lock: dir_lock,
            generation_poisoned: AtomicBool::new(false),
            name: meta.collection.clone(),
            spaces: meta.spaces,
            space_order,
            max_points: meta.max_points,
            inner: Mutex::new(Inner::default()),
            wal,
            storage: storage.clone(),
            gate: Mutex::new(()),
            metrics: Metrics::default(),
            generation: parking_lot::RwLock::new(generation),
            blobs: arc_swap::ArcSwap::from(Arc::new(blobs)),
            payloads: arc_swap::ArcSwap::from(Arc::new(payloads)),
            postings: arc_swap::ArcSwap::from(Arc::new(postings)),
        });
        if let Some(snapshot) = snapshot {
            store.restore_snapshot(snapshot, rebuild_postings)?;
        }
        let floor = store.generation.read().lsn();
        let (txns, last) = store.wal.read_all_from(floor)?;
        for txn in &txns {
            store.replay_txn(txn)?;
        }
        if last > floor {
            store.inner.lock().next_lsn = last.saturating_add(1);
        }
        Ok(store)
    }

    fn restore_snapshot(&self, snapshot: NamedSnapshot, rebuild_postings: bool) -> Result<()> {
        let mut inner = self.inner.lock();
        let capacity = snapshot.points.iter().map(|p| p.slot).max().unwrap_or(0) + 1;
        inner.points.resize(capacity as usize, None);
        for point in snapshot.points {
            let payload = self.load_payload(point.payload)?;
            // Structural check per space; contents were validated before the
            // point was logged.
            {
                let blobs_guard = self.blobs.load();
                let reader = blobs_guard.reader();
                let bytes = reader.bytes(point.blob).ok_or_else(|| {
                    VectorSearchError::CorruptData(format!(
                        "snapshot slot {} references a missing vector blob",
                        point.slot
                    ))
                })?;
                for (index, entry) in
                    crate::storage::point_blobs::named_entries(bytes).ok_or_else(|| {
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
                    let view =
                        crate::storage::point_blobs::decode_vector(entry).ok_or_else(|| {
                            VectorSearchError::CorruptData(format!(
                                "snapshot slot {} holds an undecodable {space} vector",
                                point.slot
                            ))
                        })?;
                    self.check_view(space, &view)?;
                    if rebuild_postings {
                        if let Some(postings) = self.postings_of(space) {
                            inner.index_view(&postings, space, point.slot, view)?;
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
                payload_blob: point.payload,
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
        self.ensure_not_poisoned()?;
        let _gate = self.gate.lock();
        let mut inner = self.inner.lock();
        self.checkpoint_from_locked(&mut inner)
    }

    pub(super) fn checkpoint_from_locked(&self, inner: &mut Inner) -> Result<()> {
        let lsn = inner.next_lsn.saturating_sub(1);
        self.blobs.load().sync()?;
        for postings in self.postings.load().values() {
            postings.flush(Some(lsn))?;
        }
        self.checkpoint_payloads(inner)?;
        let snapshot = NamedSnapshot {
            lsn,
            blobs_len: self.blobs.load().len(),
            payloads_len: self.payloads.load().len(),
            points: inner
                .points
                .iter()
                .enumerate()
                .filter_map(|(slot, point)| {
                    point.as_ref().map(|point| NamedSnapshotPoint {
                        slot: slot as u32,
                        id: point.id.to_string(),
                        blob: point.blob,
                        payload: point.payload_blob,
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
        let snapshot_path = self.generation.read().path().join("named_snapshot.bin");
        snapshot::write(&snapshot_path, &snapshot)?;
        self.generation.write().set_lsn(lsn)?;
        self.wal.truncate_through(lsn)?;
        Ok(())
    }

    /// Reclaim tombstoned slots by building and publishing a complete new
    /// generation. A failure leaves the serving generation and its WAL intact.
    pub fn compact(&self) -> Result<u64> {
        self.ensure_not_poisoned()?;
        let _gate = self.gate.lock();
        let mut inner = self.inner.lock();
        let reclaimed = (inner.points.len() as u64).saturating_sub(inner.live_count);
        if reclaimed == 0 {
            return Ok(0);
        }
        let lsn = inner.next_lsn.saturating_sub(1);
        let staged = self.generation.read().stage()?;
        let staged_state = match self.stage_compaction(&inner, &staged, lsn) {
            Ok(state) => state,
            Err(e) => {
                staged.discard();
                return Err(e);
            }
        };

        let blobs = PointBlobs::open(
            &staged.path().join(BLOBS_FILE),
            Some(staged_state.blobs_len),
        )?;
        let payloads = PointBlobs::open(
            &staged.path().join(PAYLOADS_FILE),
            Some(staged_state.payloads_len),
        )?;
        let postings = open_postings(staged.path(), &self.spaces, &self.space_order, Some(lsn))?;
        drop((blobs, payloads, postings));
        let old_generation = self.generation.read().clone();
        let next_generation = old_generation.clone().publish(staged, lsn)?;
        self.generation_poisoned
            .store(true, std::sync::atomic::Ordering::Release);
        self.wal.bind_generation(next_generation.id());

        let data_dir = next_generation.path();
        self.blobs.store(Arc::new(PointBlobs::open(
            &data_dir.join(BLOBS_FILE),
            Some(staged_state.blobs_len),
        )?));
        self.payloads.store(Arc::new(PointBlobs::open(
            &data_dir.join(PAYLOADS_FILE),
            Some(staged_state.payloads_len),
        )?));
        self.postings.store(Arc::new(open_postings(
            &data_dir,
            &self.spaces,
            &self.space_order,
            Some(lsn),
        )?));
        *self.generation.write() = next_generation;
        inner.points = staged_state.points;
        inner.reverse = staged_state.reverse;
        inner.norms = staged_state.norms;

        self.wal.truncate_through(lsn)?;
        old_generation.retire()?;
        self.generation_poisoned
            .store(false, std::sync::atomic::Ordering::Release);
        Ok(reclaimed)
    }

    /// Build the data files, postings and snapshot of a compacted generation
    /// without touching the serving state.
    fn stage_compaction(
        &self,
        inner: &Inner,
        staged: &crate::storage::generation::StagedGeneration,
        lsn: u64,
    ) -> Result<StagedNamedState> {
        let dir = staged.path();
        let staging_blobs = PointBlobs::create(&dir.join(BLOBS_FILE))?;
        let staging_payloads = PointBlobs::create(&dir.join(PAYLOADS_FILE))?;
        let postings = open_postings(dir, &self.spaces, &self.space_order, None)?;
        let mut points = Vec::with_capacity(inner.live_count as usize);
        let mut reverse = IdMap::with_capacity(inner.live_count as usize);
        let mut norms: HashMap<String, HashMap<u32, f32>> = HashMap::new();

        {
            let old_blobs = self.blobs.load();
            let old_reader = old_blobs.reader();
            for (old_slot, point) in inner.points.iter().enumerate() {
                let Some(point) = point else { continue };
                let slot = points.len() as u32;
                let bytes = old_reader.bytes(point.blob).ok_or_else(|| {
                    VectorSearchError::CorruptData(format!(
                        "point {} references a missing vector blob during compaction",
                        point.id
                    ))
                })?;
                for (space, space_norms) in &inner.norms {
                    if let Some(norm) = space_norms.get(&(old_slot as u32)) {
                        norms.entry(space.clone()).or_default().insert(slot, *norm);
                    }
                }
                let blob = staging_blobs.append(bytes)?;
                let payload_blob = match &point.payload {
                    Some(payload) => Some(staging_payloads.append(&serde_json::to_vec(payload)?)?),
                    None => None,
                };
                {
                    let new_reader = staging_blobs.reader();
                    if let Some(bytes) = new_reader.bytes(blob) {
                        for (index, entry) in
                            crate::storage::point_blobs::named_entries(bytes).unwrap_or_default()
                        {
                            let Some(space) = self.space_at(index) else {
                                continue;
                            };
                            let Some(space_postings) = postings.get(space) else {
                                continue;
                            };
                            if let Some(crate::types::VectorView::Sparse { indices, values }) =
                                crate::storage::point_blobs::decode_vector(entry)
                            {
                                space_postings.index(slot, indices, values)?;
                            }
                        }
                    }
                }
                reverse.insert(&point.id, slot);
                points.push(Some(NamedPoint {
                    id: point.id.clone(),
                    blob,
                    payload_blob,
                    payload: point.payload.clone(),
                }));
            }
        }
        staging_blobs.sync()?;
        staging_payloads.sync()?;
        for space_postings in postings.values() {
            space_postings.flush(Some(lsn))?;
        }

        let snapshot = NamedSnapshot {
            lsn,
            blobs_len: staging_blobs.len(),
            payloads_len: staging_payloads.len(),
            points: points
                .iter()
                .enumerate()
                .filter_map(|(slot, point)| {
                    point.as_ref().map(|point| NamedSnapshotPoint {
                        slot: slot as u32,
                        id: point.id.to_string(),
                        blob: point.blob,
                        payload: point.payload_blob,
                        norms: self
                            .sparse_spaces()
                            .filter_map(|(space, index)| {
                                norms
                                    .get(space)
                                    .and_then(|space_norms| space_norms.get(&(slot as u32)))
                                    .map(|norm| (index, *norm))
                            })
                            .collect(),
                    })
                })
                .collect(),
        };
        snapshot::write(&dir.join("named_snapshot.bin"), &snapshot)?;
        Ok(StagedNamedState {
            points,
            reverse,
            norms,
            blobs_len: snapshot.blobs_len,
            payloads_len: snapshot.payloads_len,
        })
    }

    /// Materialize one payload from its committed blob.
    fn load_payload(&self, blob: Option<BlobRef>) -> Result<Option<Payload>> {
        let Some(blob) = blob else {
            return Ok(None);
        };
        let payloads_guard = self.payloads.load();
        let reader = payloads_guard.reader();
        let bytes = reader.bytes(blob).ok_or_else(|| {
            VectorSearchError::CorruptData("snapshot references a missing payload blob".to_string())
        })?;
        Ok(Some(serde_json::from_slice(bytes)?))
    }

    /// Append JSON for changed payloads only; unchanged handles are reused so
    /// a checkpoint that follows a vector-only write does not grow the file.
    fn checkpoint_payloads(&self, inner: &mut Inner) -> Result<()> {
        for point in inner.points.iter_mut().flatten() {
            if point.payload.is_some() && point.payload_blob.is_none() {
                let bytes = serde_json::to_vec(point.payload.as_ref().expect("checked above"))?;
                point.payload_blob = Some(self.payloads.load().append(&bytes)?);
            }
        }
        self.payloads.load().sync()?;
        Ok(())
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
                WalRecord::SetPayload { point_id, payload } => {
                    let Some(slot) = inner.reverse.get_str(point_id) else {
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
                    let Some(slot) = inner.reverse.get_str(point_id) else {
                        continue;
                    };
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        if let Some(payload) = &mut point.payload {
                            for key in keys {
                                payload.remove(key);
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
                    let Some(slot) = inner.reverse.get_str(point_id) else {
                        continue;
                    };
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        let payload = point.payload.get_or_insert_with(Default::default);
                        let value: serde_json::Value = serde_json::from_str(value)?;
                        payload.insert(key.clone(), value);
                        point.touch_payload();
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
        let blob = self
            .blobs
            .load()
            .append(&encode_named(&vectors, &self.space_order))?;
        for (space, data) in &vectors {
            if let (VectorData::Sparse { indices, values }, Some(postings)) =
                (data, self.postings_of(space))
            {
                inner.index_sparse(&postings, space, slot, indices, values)?;
            }
        }
        inner.reverse.insert_str(&id, slot);
        inner.points.push(Some(NamedPoint {
            id: PointId::from(id),
            blob,
            payload,
            payload_blob: None,
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
            let blobs_guard = self.blobs.load();
            let reader = blobs_guard.reader();
            let bytes = reader.bytes(existing.blob).ok_or_else(|| {
                VectorSearchError::CorruptData(format!("point {id} references a missing blob"))
            })?;
            rewrite_named(bytes, &replacements).ok_or_else(|| {
                VectorSearchError::CorruptData(format!("point {id} holds an unreadable blob"))
            })?
        };

        inner.tombstone_slot(slot);
        let new_slot = inner.points.len() as u32;
        let blob = self.blobs.load().append(&merged)?;
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
                inner.index_view(&postings, space, new_slot, view)?;
            }
        }
        inner.reverse.insert_str(&id, new_slot);
        inner.points.push(Some(NamedPoint {
            id: existing.id,
            blob,
            payload_blob: existing.payload_blob,
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
) -> Result<HashMap<String, Arc<Postings>>> {
    let mut postings = HashMap::new();
    for (name, meta) in spaces {
        if meta.kind != VectorKind::Sparse {
            continue;
        }
        let index = order.get(name).copied().unwrap_or(0);
        let space_dir = dir.join(POSTINGS_DIR).join(format!("{index:03}"));
        postings.insert(
            name.clone(),
            Arc::new(Postings::open(&space_dir, snapshot_lsn)?),
        );
    }
    Ok(postings)
}
