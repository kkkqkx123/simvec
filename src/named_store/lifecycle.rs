use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::storage::point_blobs::{rewrite_named, BlobRef};
use crate::storage::postings::Postings;
use crate::storage::segmented_blobs::{SegmentedBlobs, DEFAULT_SEGMENT_BYTES};
use crate::storage::spaces_meta::{
    self, DtypeParams, SpaceMeta, SpacesManifest, DEFAULT_SEGMENT_SLOTS,
};
use crate::storage::{snapshot, CollectionLock, Generation, IdMap, Wal, WalRecord, WalTxn};
use crate::types::{CollectionConfig, Payload, PointId, VectorData, VectorKind, VectorSpaceConfig};

use super::{
    Inner, NamedMeta, NamedPoint, NamedSnapshot, NamedSnapshotPoint, NamedSpaceMeta, NamedStore,
};

/// File holding the append-only point vectors inside each segment directory.
const BLOBS_FILE: &str = "points.bin";
/// File holding the append-only JSON payloads inside each segment directory.
const PAYLOADS_FILE: &str = "payloads.bin";
/// Directory holding the per-space postings.
const POSTINGS_DIR: &str = "postings";
/// Space directory hosting the whole-point blob stream (named points encode
/// every declared space in one blob, so vectors share one segmented stream).
const VECTORS_SPACE: &str = "vectors";

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
        index: &crate::types::IndexConfig,
        storage: &crate::types::StorageConfig,
    ) -> Result<Arc<Self>> {
        let mut config = CollectionConfig::with_named(spaces.clone());
        config.max_points = max_points;
        config.index = index.clone();
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
        // spaces.json is the authoritative layout: all declared spaces, their
        // index/quantization selection and segment width are recorded here.
        let dtype_map = build_dtype_map(spaces);
        let spaces_meta_map = build_space_meta(spaces, index);
        spaces_meta::write(
            &data_dir,
            &SpacesManifest::new(name, generation.id(), generation.lsn(), spaces_meta_map, max_points),
        )?;
        let seg_root = spaces_meta::segment_dir(&data_dir, VECTORS_SPACE, 0)
            .parent()
            .expect("segment dir has a parent")
            .to_path_buf();
        let blobs = SegmentedBlobs::create(&seg_root, BLOBS_FILE, DEFAULT_SEGMENT_BYTES)?;
        let payloads = SegmentedBlobs::create(&seg_root, PAYLOADS_FILE, DEFAULT_SEGMENT_BYTES)?;
        let space_order = space_order(&meta.spaces);
        let postings = open_postings(&data_dir, &meta.spaces, None)?;
        let ann = build_ann_map(&data_dir, &meta.spaces, index, true)?;
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
            ann: arc_swap::ArcSwap::from_pointee(ann),
            index: index.clone(),
            dtype_map,
        }))
    }

    /// Open an existing named collection: map the vector file, load the
    /// per-space dictionaries, then replay what the snapshot does not cover.
    pub fn open(dir: &Path, storage: &crate::types::StorageConfig) -> Result<Arc<Self>> {
        let dir_lock = CollectionLock::acquire(dir)?;
        let generation = Generation::open(dir)?;
        let wal = Wal::open_or_create(&dir.join("wal"), storage.wal_segment_bytes)?;
        wal.bind_generation(generation.id());
        let data_dir = generation.path();
        // Rebuild the space declarations from the authoritative manifest.
        let manifest = spaces_meta::read(&data_dir, generation.id(), generation.lsn())?;
        let spaces: BTreeMap<String, NamedSpaceMeta> = manifest
            .spaces
            .iter()
            .map(|(name, space)| {
                (
                    name.clone(),
                    NamedSpaceMeta {
                        kind: space.kind,
                        vector_size: space.dim,
                        distance: space.distance,
                        multi_config: space.multi_config,
                    },
                )
            })
            .collect();
        let meta = NamedMeta {
            collection: manifest.collection.clone(),
            spaces: spaces.clone(),
            max_points: manifest.max_points,
        };
        // All spaces share the collection-level index tier; read it from the
        // authoritative manifest (any declared space carries it).
        let index = manifest
            .spaces
            .values()
            .next()
            .map(|space| space.index.clone())
            .unwrap_or(crate::types::IndexConfig::Flat);
        let dtype_map: BTreeMap<
            String,
            (
                crate::types::VectorDataType,
                DtypeParams,
            ),
        > = manifest
            .spaces
            .iter()
            .map(|(name, space)| {
                (
                    name.clone(),
                    (space.dtype, space.dtype_params),
                )
            })
            .collect();
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
        let slot_capacity = snapshot
            .as_ref()
            .map(|snapshot| {
                snapshot
                    .points
                    .iter()
                    .map(|point| point.slot as u64 + 1)
                    .max()
                    .unwrap_or(1)
            })
            .unwrap_or(1);
        let seg_root = spaces_meta::segment_dir(&data_dir, VECTORS_SPACE, 0)
            .parent()
            .expect("segment dir has a parent")
            .to_path_buf();
        let blobs = SegmentedBlobs::open(
            &seg_root,
            BLOBS_FILE,
            DEFAULT_SEGMENT_BYTES,
            snapshot.as_ref().map_or(0, |snapshot| snapshot.blobs_len),
        )?;
        // No published snapshot means no reachable payload handle; the file
        // is entirely an uncommitted tail.
        let payloads = SegmentedBlobs::open(
            &seg_root,
            PAYLOADS_FILE,
            DEFAULT_SEGMENT_BYTES,
            snapshot.as_ref().map_or(0, |snapshot| snapshot.payloads_len),
        )?;
        let space_order = space_order(&meta.spaces);
        let postings = open_postings(&data_dir, &meta.spaces, snapshot_lsn)?;
        let rebuild_postings = postings.values().any(|space| !space.was_restored());
        let ann = build_ann_map_open(&data_dir, &meta.spaces, &index, slot_capacity)?;
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
            ann: arc_swap::ArcSwap::from_pointee(ann),
            index,
            dtype_map,
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
                    let mut scratch = Vec::new();
                    let view = self.view_space(space, entry, &mut scratch).ok_or_else(|| {
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

    /// Rebuild each dense space's ANN from the live points. Reads vectors out
    /// of the committed blob stream through one reader pass; a space whose
    /// points never carry that space is simply left untrained.
    fn rebuild_ann(&self, inner: &Inner) -> Result<()> {
        if self.ann.load().is_empty() {
            return Ok(());
        }
        let blobs = self.blobs.load();
        let reader = blobs.reader();
        let ann = self.ann.load();
        for (name, index) in ann.iter() {
            let Some(space_idx) = self.order_of(name) else { continue };
            let (dtype, dtype_params) = self
                .dtype_map
                .get(name)
                .copied()
                .unwrap_or((crate::types::VectorDataType::F32, DtypeParams::default()));
            let qparams = crate::distance::dtype::QuantParams {
                scale: dtype_params.scale,
                zero_point: dtype_params.zero_point,
            };
            let typed = dtype != crate::types::VectorDataType::F32;
            // For typed spaces decode into an owned arena first (Box keeps each
            // slice's address stable), then build borrow rows in a second pass
            // so growing the arena never invalidates an outstanding borrow.
            let mut arena: Vec<Box<[f32]>> = Vec::new();
            let mut slots: Vec<u32> = Vec::with_capacity(inner.live_count as usize);
            for (slot, point) in inner.points.iter().enumerate() {
                let Some(point) = point else { continue };
                let Some(bytes) = reader.bytes(point.blob) else { continue };
                let entry = crate::storage::point_blobs::named_entries(bytes)
                    .and_then(|entries| entries.into_iter().find(|(idx, _)| *idx == space_idx))
                    .map(|(_, bytes)| bytes);
                let Some(entry) = entry else { continue };
                if typed {
                    if let Ok(crate::types::VectorData::Dense(values)) =
                        crate::storage::point_blobs::decode_vector_typed(entry, qparams)
                    {
                        arena.push(values.into_boxed_slice());
                        slots.push(slot as u32);
                    }
                } else if matches!(
                    crate::storage::point_blobs::decode_vector(entry),
                    Some(crate::types::VectorView::Dense(_))
                ) {
                    slots.push(slot as u32);
                }
            }
            if typed {
                let rows: Vec<(u32, &[f32])> = slots
                    .into_iter()
                    .zip(arena.iter())
                    .map(|(slot, values)| (slot, values.as_ref()))
                    .collect();
                index.rebuild(&rows)?;
            } else {
                // Zero-copy pass: borrow straight from the mmap.
                let mut rows: Vec<(u32, &[f32])> = Vec::with_capacity(slots.len());
                for slot in slots {
                    let point = inner.points[slot as usize].as_ref().expect("live slot");
                    let bytes = reader.bytes(point.blob).expect("blob present");
                    let entry = crate::storage::point_blobs::named_entries(bytes)
                        .and_then(|entries| entries.into_iter().find(|(idx, _)| *idx == space_idx))
                        .map(|(_, bytes)| bytes)
                        .expect("entry present");
                    if let Some(crate::types::VectorView::Dense(values)) =
                        crate::storage::point_blobs::decode_vector(entry)
                    {
                        rows.push((slot, values));
                    }
                }
                index.rebuild(&rows)?;
            }
        }
        Ok(())
    }

    /// Publish the authoritative space manifest for a generation/lsn pair.
    fn write_spaces_manifest(&self, gen_dir: &Path, generation: u64, lsn: u64) -> Result<()> {
        let spaces = self
            .spaces
            .iter()
            .map(|(name, space)| {
                (
                    name.clone(),
                    SpaceMeta {
                        kind: space.kind,
                        distance: space.distance,
                        dim: space.vector_size,
                        multi_config: space.multi_config,
                        index: if space.kind == VectorKind::Dense {
                            self.index.clone()
                        } else {
                            crate::types::IndexConfig::Flat
                        },
                        quantization: None,
                        dtype: self
                            .dtype_map
                            .get(name)
                            .map_or(crate::types::VectorDataType::F32, |(dtype, _)| *dtype),
                        dtype_params: self
                            .dtype_map
                            .get(name)
                            .map_or(DtypeParams::default(), |(_, params)| *params),
                        segment_slots: DEFAULT_SEGMENT_SLOTS,
                    },
                )
            })
            .collect();
        spaces_meta::write(
            gen_dir,
            &SpacesManifest::new(&self.name, generation, lsn, spaces, self.max_points),
        )
    }

    pub(super) fn checkpoint_from_locked(&self, inner: &mut Inner) -> Result<()> {
        let lsn = inner.next_lsn.saturating_sub(1);
        self.blobs.load().sync()?;
        // Rebuild every dense space's ANN from the live points. Any failure
        // here leaves the previous graph intact and aborts the checkpoint
        // before the generation advances (the graph is derived but the
        // checkpoint must not publish vectors a graph does not describe).
        self.rebuild_ann(inner)?;
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
        let gen_dir = self.generation.read().path();
        snapshot::write(&gen_dir.join("named_snapshot.bin"), &snapshot)?;
        // The authoritative manifest is written before the generation commit
        // point advances so a published generation always has a matching one.
        let generation_id = self.generation.read().id();
        self.write_spaces_manifest(&gen_dir, generation_id, lsn)?;
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

        let staged_seg_root = spaces_meta::segment_dir(&staged.path(), VECTORS_SPACE, 0)
            .parent()
            .expect("segment dir has a parent")
            .to_path_buf();
        let blobs = SegmentedBlobs::open(
            &staged_seg_root, BLOBS_FILE, DEFAULT_SEGMENT_BYTES, staged_state.blobs_len,
        )?;
        let payloads = SegmentedBlobs::open(
            &staged_seg_root, PAYLOADS_FILE, DEFAULT_SEGMENT_BYTES, staged_state.payloads_len,
        )?;
        let postings = open_postings(staged.path(), &self.spaces, Some(lsn))?;
        drop((blobs, payloads, postings));
        let old_generation = self.generation.read().clone();
        let next_generation = old_generation.clone().publish(staged, lsn)?;
        self.generation_poisoned
            .store(true, std::sync::atomic::Ordering::Release);
        self.wal.bind_generation(next_generation.id());

        let data_dir = next_generation.path();
        let seg_root = spaces_meta::segment_dir(&data_dir, VECTORS_SPACE, 0)
            .parent()
            .expect("segment dir has a parent")
            .to_path_buf();
        self.blobs.store(Arc::new(SegmentedBlobs::open(
            &seg_root, BLOBS_FILE, DEFAULT_SEGMENT_BYTES, staged_state.blobs_len,
        )?));
        self.payloads.store(Arc::new(SegmentedBlobs::open(
            &seg_root, PAYLOADS_FILE, DEFAULT_SEGMENT_BYTES, staged_state.payloads_len,
        )?));
        self.postings.store(Arc::new(open_postings(
            &data_dir, &self.spaces, Some(lsn),
        )?));
        *self.generation.write() = next_generation;
        inner.points = staged_state.points;
        inner.reverse = staged_state.reverse;
        inner.norms = staged_state.norms;
        // Rebuild the per-space ANN against the new generation's vector files.
        let ann_capacity = inner.points.len() as u64;
        let new_ann = build_ann_map_open(&data_dir, &self.spaces, &self.index, ann_capacity)?;
        self.ann.store(Arc::new(new_ann));
        self.rebuild_ann(&inner)?;

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
        let staging_seg_root = spaces_meta::segment_dir(&dir, VECTORS_SPACE, 0)
            .parent()
            .expect("segment dir has a parent")
            .to_path_buf();
        let staging_blobs = SegmentedBlobs::create(&staging_seg_root, BLOBS_FILE, DEFAULT_SEGMENT_BYTES)?;
        let staging_payloads = SegmentedBlobs::create(&staging_seg_root, PAYLOADS_FILE, DEFAULT_SEGMENT_BYTES)?;
        let postings = open_postings(dir, &self.spaces, None)?;
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
        self.write_spaces_manifest(&dir, staged.id(), lsn)?;
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
                WalRecord::Upsert { .. } | WalRecord::Quantize => {
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
    /// Decode one space's encoded vector into an owned payload (fetch paths).
    pub(super) fn decode_space_data(&self, space: &str, bytes: &[u8]) -> Option<crate::types::VectorData> {
        let (dtype, params) = self
            .dtype_map
            .get(space)
            .copied()
            .unwrap_or((crate::types::VectorDataType::F32, DtypeParams::default()));
        if dtype == crate::types::VectorDataType::F32 {
            return crate::storage::point_blobs::decode_vector(bytes).map(|v| v.to_data());
        }
        let qparams = crate::distance::dtype::QuantParams {
            scale: params.scale,
            zero_point: params.zero_point,
        };
        crate::storage::point_blobs::decode_vector_typed(bytes, qparams).ok()
    }

    /// Decode one space's encoded vector into a borrowable view. The default
    /// f32 dtype is zero-copy (borrowed from `bytes`); typed encodings are
    /// decoded into `scratch`.
    fn view_space<'a>(
        &self,
        space: &str,
        bytes: &'a [u8],
        scratch: &'a mut Vec<f32>,
    ) -> Option<crate::types::VectorView<'a>> {
        let (dtype, params) = self
            .dtype_map
            .get(space)
            .copied()
            .unwrap_or((crate::types::VectorDataType::F32, DtypeParams::default()));
        let qparams = crate::distance::dtype::QuantParams {
            scale: params.scale,
            zero_point: params.zero_point,
        };
        crate::storage::point_blobs::view_for_score(bytes, dtype, qparams, scratch)
    }

    /// Encode one space's vector with that space's element dtype.
    fn encode_space(&self, space: &str, data: &VectorData) -> Result<Vec<u8>> {
        let (dtype, params) = self
            .dtype_map
            .get(space)
            .copied()
            .unwrap_or((crate::types::VectorDataType::F32, DtypeParams::default()));
        let qparams = crate::distance::dtype::QuantParams {
            scale: params.scale,
            zero_point: params.zero_point,
        };
        Ok(crate::storage::point_blobs::encode_vector_typed(data, dtype, qparams)?.0)
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
        // Idempotent overwrite: a replayed or repeated upsert of the same id
        // replaces the stored point (old slot tombstoned, bytes and postings
        // left for compaction).
        if let Some(slot) = inner.reverse.get_str(&id) {
            inner.tombstone_slot(slot);
        }
        let slot = inner.points.len() as u32;
        let mut entries = Vec::with_capacity(vectors.len());
        for (space, data) in &vectors {
            let index = self.order_of(space).unwrap_or(u16::MAX);
            entries.push((index, self.encode_space(space, data)?));
        }
        let blob = self
            .blobs
            .load()
            .append(&crate::storage::point_blobs::pack_named(entries))?;
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
            replacements.insert(index, self.encode_space(space, data)?);
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

/// Build the authoritative per-space metadata recorded in `spaces.json`.
fn build_space_meta(
    spaces: &BTreeMap<String, crate::types::VectorSpaceConfig>,
    index: &crate::types::IndexConfig,
) -> BTreeMap<String, SpaceMeta> {
    spaces
        .iter()
        .map(|(name, config)| {
            (
                name.clone(),
                SpaceMeta {
                    kind: config.kind(),
                    distance: config.distance(),
                    dim: config.vector_size(),
                    multi_config: config.multi_config(),
                    // The collection-level tier applies only to dense spaces;
                    // sparse/multi metadata records Flat (validated at create).
                    index: if config.kind() == VectorKind::Dense {
                        index.clone()
                    } else {
                        crate::types::IndexConfig::Flat
                    },
                    quantization: None,
                    dtype: config.dtype(),
                    dtype_params: DtypeParams::default(),
                    segment_slots: DEFAULT_SEGMENT_SLOTS,
                },
            )
        })
        .collect()
}

/// Derive the per-space element dtype and affine params from the declared
/// space configs. Sparse/multi spaces always report F32.
fn build_dtype_map(
    spaces: &BTreeMap<String, crate::types::VectorSpaceConfig>,
) -> BTreeMap<String, (crate::types::VectorDataType, DtypeParams)> {
    spaces
        .iter()
        .map(|(name, config)| (name.clone(), (config.dtype(), DtypeParams::default())))
        .collect()
}

/// Build the ANN index map for dense spaces declaring an HNSW tier (create
/// path: empty index, no persisted graph yet).
fn build_ann_map(
    gen_dir: &Path,
    spaces: &BTreeMap<String, NamedSpaceMeta>,
    index: &crate::types::IndexConfig,
    _create: bool,
) -> Result<BTreeMap<String, Arc<crate::storage::space_ann::SpaceAnn>>> {
    let mut map = BTreeMap::new();
    if let Some(config) = crate::storage::space_ann::SpaceAnn::hnsw_config(index) {
        for (name, meta) in spaces {
            if meta.kind != VectorKind::Dense {
                continue;
            }
            map.insert(
                name.clone(),
                Arc::new(crate::storage::space_ann::SpaceAnn::create(
                    gen_dir,
                    name,
                    meta.vector_size,
                    meta.distance,
                    config.clone(),
                )?),
            );
        }
    }
    Ok(map)
}

/// Build the ANN index map on open, loading persisted graphs. Any bad graph
/// degrades to exact scan (handled inside `SpaceAnn::open`).
fn build_ann_map_open(
    gen_dir: &Path,
    spaces: &BTreeMap<String, NamedSpaceMeta>,
    index: &crate::types::IndexConfig,
    slot_capacity: u64,
) -> Result<BTreeMap<String, Arc<crate::storage::space_ann::SpaceAnn>>> {
    let mut map = BTreeMap::new();
    if let Some(config) = crate::storage::space_ann::SpaceAnn::hnsw_config(index) {
        for (name, meta) in spaces {
            if meta.kind != VectorKind::Dense {
                continue;
            }
            map.insert(
                name.clone(),
                Arc::new(crate::storage::space_ann::SpaceAnn::open(
                    gen_dir,
                    name,
                    meta.vector_size,
                    meta.distance,
                    config.clone(),
                    slot_capacity,
                )?),
            );
        }
    }
    Ok(map)
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

/// Open one postings store per sparse space, under `spaces/<space>/postings/`.
fn open_postings(
    dir: &Path,
    spaces: &BTreeMap<String, NamedSpaceMeta>,
    snapshot_lsn: Option<u64>,
) -> Result<HashMap<String, Arc<Postings>>> {
    let mut postings = HashMap::new();
    for (name, meta) in spaces {
        if meta.kind != VectorKind::Sparse {
            continue;
        }
        let space_dir = dir.join("spaces").join(name).join(POSTINGS_DIR);
        postings.insert(
            name.clone(),
            Arc::new(Postings::open(&space_dir, snapshot_lsn)?),
        );
    }
    Ok(postings)
}
