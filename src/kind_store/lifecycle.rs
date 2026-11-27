use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::storage::point_blobs::{encode_vector, BlobRef, PointBlobs};
use crate::storage::postings::Postings;
use crate::storage::{snapshot, Generation, IdMap, Wal, WalRecord, WalTxn};
use crate::types::{Payload, PointId, VectorData};

use super::{Inner, KindMeta, KindPoint, KindSnapshot, KindSnapshotPoint, KindStore};

/// File holding the append-only point vectors.
const BLOBS_FILE: &str = "points.bin";
/// File holding the append-only JSON payloads.
const PAYLOADS_FILE: &str = "payloads.bin";
/// Directory holding the persisted sparse postings.
const POSTINGS_DIR: &str = "postings";

/// State staged for a compacted generation, kept out of the serving store
/// until the manifest is published.
struct StagedKindState {
    points: Vec<Option<KindPoint>>,
    reverse: IdMap,
    norms: HashMap<u32, f32>,
    blobs_len: u64,
    payloads_len: u64,
}

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
        let wal = Wal::open_or_create(&dir.join("wal"), storage.wal_segment_bytes)?;
        let generation = Generation::create(dir)?;
        let data_dir = generation.path();
        let blobs = PointBlobs::create(&data_dir.join(BLOBS_FILE))?;
        let payloads = PointBlobs::create(&data_dir.join(PAYLOADS_FILE))?;
        let postings = Postings::open(&data_dir.join(POSTINGS_DIR), None)?;
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
            generation: parking_lot::RwLock::new(generation),
            blobs: arc_swap::ArcSwap::from(Arc::new(blobs)),
            payloads: arc_swap::ArcSwap::from(Arc::new(payloads)),
            postings: arc_swap::ArcSwap::from(Arc::new(postings)),
        }))
    }

    /// Open an existing kind collection: map the vector file, load the posting
    /// dictionary, then replay whatever the snapshot does not cover.
    pub fn open(dir: &Path, storage: &crate::types::StorageConfig) -> Result<Arc<Self>> {
        let bytes = std::fs::read(dir.join("kind.bin"))?;
        let meta: KindMeta = postcard::from_bytes(&bytes)?;
        let wal = Wal::open_or_create(&dir.join("wal"), storage.wal_segment_bytes)?;
        let generation = Generation::open(dir)?;
        let data_dir = generation.path();
        let snapshot_path = data_dir.join("kind_snapshot.bin");
        let snapshot = snapshot::read::<KindSnapshot>(&snapshot_path)?;
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
        // Without a published snapshot no payload reference is reachable, so
        // the whole file is an uncommitted tail and can be discarded.
        let payloads = PointBlobs::open(
            &data_dir.join(PAYLOADS_FILE),
            Some(
                snapshot
                    .as_ref()
                    .map_or(0, |snapshot| snapshot.payloads_len),
            ),
        )?;
        let postings = Postings::open(&data_dir.join(POSTINGS_DIR), snapshot_lsn)?;
        // The dictionary is only usable when the same checkpoint wrote the
        // snapshot; otherwise every live point has to be re-inverted from its
        // blob (a crash between the two files, or a first-ever open).
        let rebuild_postings = !postings.was_restored();
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

    fn restore_snapshot(&self, snapshot: KindSnapshot, rebuild_postings: bool) -> Result<()> {
        let mut inner = self.inner.lock();
        let capacity = snapshot.points.iter().map(|p| p.slot).max().unwrap_or(0) + 1;
        inner.points.resize(capacity as usize, None);
        for point in snapshot.points {
            let payload = self.load_payload(point.payload)?;
            // Structural check only: the bytes were validated before they were
            // logged, and re-validating them fully would decode every vector
            // this snapshot exists to avoid decoding.
            {
                let blobs_guard = self.blobs.load();
        let reader = blobs_guard.reader();
                let view = reader.view(point.blob).ok_or_else(|| {
                    VectorSearchError::CorruptData(format!(
                        "snapshot slot {} references a missing vector blob",
                        point.slot
                    ))
                })?;
                self.check_view(&view)?;
                if rebuild_postings {
                    inner.index_view(&self.postings.load(), point.slot, view)?;
                }
            }
            inner.reverse.insert_str(&point.id, point.slot);
            if let Some(norm) = point.norm {
                inner.norms.insert(point.slot, norm);
            }
            inner.points[point.slot as usize] = Some(KindPoint {
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
    /// The gate blocks writers for the whole snapshot-and-truncate window: a
    /// commit appended between the two would otherwise exist only in the
    /// segments that are about to be unlinked.
    ///
    /// Order is data first, metadata last: `points.bin` and the posting
    /// manifest become durable before the snapshot that references them, and
    /// the WAL is cut back only once the snapshot is published. A crash in the
    /// middle costs the posting index (its manifest lsn no longer matches the
    /// snapshot, so it is rebuilt) but never a committed point.
    pub fn checkpoint(&self) -> Result<()> {
        let _gate = self.gate.lock();
        let mut inner = self.inner.lock();
        self.checkpoint_from_locked(&mut inner)
    }

    pub(super) fn checkpoint_from_locked(&self, inner: &mut Inner) -> Result<()> {
        let lsn = inner.next_lsn.saturating_sub(1);
        self.blobs.load().sync();
        // The manifest carries `lsn`: it is the promise that these postings
        // describe exactly the snapshot written below.
        self.postings.load().flush(Some(lsn))?;
        self.checkpoint_payloads(inner)?;
        let snapshot = KindSnapshot {
            lsn,
            blobs_len: self.blobs.load().len(),
            payloads_len: self.payloads.load().len(),
            points: inner
                .points
                .iter()
                .enumerate()
                .filter_map(|(slot, point)| {
                    point.as_ref().map(|point| KindSnapshotPoint {
                        slot: slot as u32,
                        id: point.id.to_string(),
                        blob: point.blob,
                        payload: point.payload_blob,
                        norm: inner.norms.get(&(slot as u32)).copied(),
                    })
                })
                .collect(),
        };
        let snapshot_path = self.generation.read().path().join("kind_snapshot.bin");
        snapshot::write(&snapshot_path, &snapshot)?;
        // Bind the WAL watermark to this generation before it can be advanced.
        self.generation.write().set_lsn(lsn)?;
        // Everything up to `lsn` is durable in the snapshot, so whole segments
        // covered by it can be unlinked instead of rewritten.
        self.wal.truncate_through(lsn)?;
        Ok(())
    }

    /// Reclaim tombstoned slots by building and publishing a complete new
    /// generation. Nothing in the serving generation is overwritten, so a
    /// failure leaves both the files and the WAL exactly as they were.
    pub fn compact(&self) -> Result<u64> {
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

        // Open the staged files before publication: a failure here must not
        // leave a manifest pointing at a generation the process cannot serve.
        let blobs = PointBlobs::open(
            &staged.path().join(BLOBS_FILE),
            Some(staged_state.blobs_len),
        )?;
        let payloads = PointBlobs::open(
            &staged.path().join(PAYLOADS_FILE),
            Some(staged_state.payloads_len),
        )?;
        let postings = Postings::open(&staged.path().join(POSTINGS_DIR), Some(lsn))?;
        drop((blobs, payloads, postings));
        let old_generation = self.generation.read().clone();
        let next_generation = old_generation.clone().publish(staged, lsn)?;

        let data_dir = next_generation.path();
        self.blobs.store(Arc::new(PointBlobs::open(
            &data_dir.join(BLOBS_FILE),
            Some(staged_state.blobs_len),
        )?));
        self.payloads.store(Arc::new(PointBlobs::open(
            &data_dir.join(PAYLOADS_FILE),
            Some(staged_state.payloads_len),
        )?));
        self.postings.store(Arc::new(Postings::open(
            &data_dir.join(POSTINGS_DIR),
            Some(lsn),
        )?));
        *self.generation.write() = next_generation;
        inner.points = staged_state.points;
        inner.reverse = staged_state.reverse;
        inner.norms = staged_state.norms;

        // The published generation owns its snapshot; replay from its lsn even
        // if the checkpoint crash window leaves old WAL segments behind.
        self.wal.truncate_through(lsn)?;
        old_generation.retire()?;
        Ok(reclaimed)
    }

    /// Build the data files, postings and snapshot of a compacted generation
    /// without touching the serving state.
    fn stage_compaction(
        &self,
        inner: &Inner,
        staged: &crate::storage::generation::StagedGeneration,
        lsn: u64,
    ) -> Result<StagedKindState> {
        let dir = staged.path();
        let staging_blobs = PointBlobs::create(&dir.join(BLOBS_FILE))?;
        let staging_payloads = PointBlobs::create(&dir.join(PAYLOADS_FILE))?;
        let postings = Postings::open(&dir.join(POSTINGS_DIR), None)?;
        let mut points = Vec::with_capacity(inner.live_count as usize);
        let mut reverse = IdMap::with_capacity(inner.live_count as usize);
        let mut norms = HashMap::with_capacity(inner.norms.len());

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
                if let Some(norm) = inner.norms.get(&(old_slot as u32)) {
                    norms.insert(slot, *norm);
                }
                let blob = staging_blobs.append(bytes)?;
                let payload_blob = match &point.payload {
                    Some(payload) => Some(staging_payloads.append(&serde_json::to_vec(payload)?)?),
                    None => None,
                };
                {
                    let new_reader = staging_blobs.reader();
                    if let Some(view) = new_reader.view(blob) {
                        if let crate::types::VectorView::Sparse { indices, values } = view {
                            postings.index(slot, indices, values)?;
                        }
                    }
                }
                reverse.insert(&point.id, slot);
                points.push(Some(KindPoint {
                    id: point.id.clone(),
                    blob,
                    payload_blob,
                    payload: point.payload.clone(),
                }));
            }
        }
        staging_blobs.sync();
        staging_payloads.sync();
        postings.flush(Some(lsn))?;

        let snapshot = KindSnapshot {
            lsn,
            blobs_len: staging_blobs.len(),
            payloads_len: staging_payloads.len(),
            points: points
                .iter()
                .enumerate()
                .filter_map(|(slot, point)| {
                    point.as_ref().map(|point| KindSnapshotPoint {
                        slot: slot as u32,
                        id: point.id.to_string(),
                        blob: point.blob,
                        payload: point.payload_blob,
                        norm: norms.get(&(slot as u32)).copied(),
                    })
                })
                .collect(),
        };
        snapshot::write(&dir.join("kind_snapshot.bin"), &snapshot)?;
        Ok(StagedKindState {
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

    /// Append JSON for every payload changed since the last checkpoint and
    /// publish its handle in the point directory. Unchanged payloads keep
    /// their existing handle, so repeated checkpoints do not grow the file.
    fn checkpoint_payloads(&self, inner: &mut Inner) -> Result<()> {
        for point in inner.points.iter_mut().flatten() {
            if point.payload.is_some() && point.payload_blob.is_none() {
                let bytes = serde_json::to_vec(point.payload.as_ref().expect("checked above"))?;
                point.payload_blob = Some(self.payloads.load().append(&bytes)?);
            }
        }
        self.payloads.load().sync();
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
                    let data = super::write::kind_data(vectors)?;
                    self.apply_upsert(&mut inner, id.clone(), data, payload.clone())?;
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
                        let payload = payload
                            .as_ref()
                            .map(|p| serde_json::from_str(p))
                            .transpose()?;
                        point.replace_payload(payload);
                    }
                }
                WalRecord::DeletePayloadKeys { slot, keys } => {
                    if let Some(Some(point)) = inner.points.get_mut(*slot as usize) {
                        if let Some(payload) = &mut point.payload {
                            for key in keys {
                                payload.remove(key);
                            }
                        }
                        point.touch_payload();
                    }
                }
                WalRecord::SetPayloadField { slot, key, value } => {
                    if let Some(Some(point)) = inner.points.get_mut(*slot as usize) {
                        let payload = point.payload.get_or_insert_with(Default::default);
                        let value: serde_json::Value = serde_json::from_str(value)?;
                        payload.insert(key.clone(), value);
                        point.touch_payload();
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
    /// transaction; appending the blob is the only fallible step left.
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
        // replaces the stored point (the old slot is tombstoned, leaving its
        // bytes and postings for compaction to reclaim).
        if let Some(slot) = inner.reverse.get_str(&id) {
            inner.tombstone_slot(slot);
        }
        let slot = inner.points.len() as u32;
        let blob = self.blobs.load().append(&encode_vector(&data))?;
        if let VectorData::Sparse { indices, values } = &data {
            inner.index_sparse(&self.postings.load(), slot, indices, values)?;
        }
        inner.reverse.insert_str(&id, slot);
        inner.points.push(Some(KindPoint {
            id: PointId::from(id),
            blob,
            payload,
            payload_blob: None,
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

    /// Structural validation of a decoded view (snapshot restore). Everything
    /// that could make a query wrong — kind mismatch, ragged arrays, terms
    /// beyond the declared bound — is checked here; the array contents were
    /// validated when the point was written.
    fn check_view(&self, data: &crate::types::VectorView<'_>) -> Result<()> {
        use crate::types::VectorView;
        if data.kind() != self.kind {
            return Err(VectorSearchError::CorruptData(format!(
                "collection kind is {}, stored payload kind is {}",
                self.kind.as_name(),
                data.kind().as_name()
            )));
        }
        match data {
            VectorView::Dense(values) => {
                if values.len() != self.dim {
                    return Err(VectorSearchError::CorruptData(format!(
                        "stored dense vector has {} values, expected {}",
                        values.len(),
                        self.dim
                    )));
                }
            }
            VectorView::Sparse { indices, values } => {
                if indices.len() != values.len() {
                    return Err(VectorSearchError::CorruptData(
                        "stored sparse vector is ragged".to_string(),
                    ));
                }
                if let Some(&last) = indices.last() {
                    if last as usize >= self.dim {
                        return Err(VectorSearchError::CorruptData(format!(
                            "stored sparse term {last} exceeds the vocabulary bound {}",
                            self.dim
                        )));
                    }
                }
            }
            VectorView::Multi { flat, dim } => {
                if *dim != self.dim {
                    return Err(VectorSearchError::CorruptData(format!(
                        "stored multi dimension {dim} differs from the declared {}",
                        self.dim
                    )));
                }
                let count = flat.len() / (*dim).max(1);
                let cap = self.max_vectors_per_point.max(1);
                if count == 0 || count > cap {
                    return Err(VectorSearchError::CorruptData(format!(
                        "stored multi vector has {count} sub-vectors, limit {cap}"
                    )));
                }
            }
        }
        Ok(())
    }
}
