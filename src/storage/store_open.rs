use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::{Mutex, RwLock};

use super::dense_snapshot::DenseSnapshot;
use super::filter_bitmap::FilterBitmap;
use super::generation::Generation;
use super::keys::Keys;
use super::meta::Meta;
use super::payload_index::PayloadIndexManager;
use super::payload_store::PayloadStore;
use super::quant::QuantStore;
use super::queue::CommitQueue;
use super::tombstones::TombstoneBits;
use super::vectors::Vectors;
use super::{CollectionStore, DataFiles, StoreInner};
use super::{Wal, WalRecord};

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::types::{
    CollectionConfig, DistanceMetric, IndexConfig, PointId, StorageConfig, VectorPoint,
};

use super::*;

impl CollectionStore {
    /// Create a new collection directory and open it.
    pub fn create(
        dir: impl AsRef<Path>,
        collection: &str,
        config: &CollectionConfig,
    ) -> Result<Arc<Self>> {
        let slots = meta::default_segment_slots(config.vector_size);
        Self::create_with_segment_slots(dir, collection, config, slots)
    }

    /// Like [`CollectionStore::create`] but with an explicit segment slot count.
    #[doc(hidden)]
    pub fn create_with_segment_slots(
        dir: impl AsRef<Path>,
        collection: &str,
        config: &CollectionConfig,
        segment_slots: u32,
    ) -> Result<Arc<Self>> {
        Self::create_with_storage_and_slots(
            dir,
            collection,
            config,
            segment_slots,
            &StorageConfig::default(),
        )
    }

    /// Create a collection with explicit engine-level storage tuning.
    ///
    /// The tuning only affects this process: collection directories stay
    /// self-describing because the payload store records its own geometry in
    /// `payloads_store/config.json`.
    pub fn create_with_storage(
        dir: impl AsRef<Path>,
        collection: &str,
        config: &CollectionConfig,
        storage: &StorageConfig,
    ) -> Result<Arc<Self>> {
        let slots = meta::default_segment_slots(config.vector_size);
        Self::create_with_storage_and_slots(dir, collection, config, slots, storage)
    }

    /// Like [`CollectionStore::create_with_storage`] with an explicit segment
    /// slot count.
    #[doc(hidden)]
    pub fn create_with_storage_and_slots(
        dir: impl AsRef<Path>,
        collection: &str,
        config: &CollectionConfig,
        segment_slots: u32,
        storage: &StorageConfig,
    ) -> Result<Arc<Self>> {
        storage.validate()?;
        validate_collection_name(collection)?;
        if config.vector_size == 0 {
            return Err(VectorSearchError::InvalidVectorDimension {
                expected: 1,
                actual: config.vector_size,
            });
        }
        if !matches!(
            config.distance,
            DistanceMetric::Cosine
                | DistanceMetric::Euclid
                | DistanceMetric::Dot
                | DistanceMetric::Manhattan
        ) {
            return Err(VectorSearchError::UnsupportedMetric(config.distance));
        }
        match &config.index {
            IndexConfig::Hnsw(hnsw) => hnsw.validate()?,
            IndexConfig::Ivf(Some(ivf)) => ivf.validate()?,
            IndexConfig::Ivf(None) | IndexConfig::Flat => {}
        }
        if let Some(qc) = &config.quantization_config {
            qc.validate(config.vector_size)?;
        }
        if config.max_points == Some(0) {
            return Err(VectorSearchError::InvalidConfig(
                "max_points must be >= 1 when set".to_string(),
            ));
        }

        let dir = dir.as_ref();
        if dir.exists() {
            return Err(VectorSearchError::CollectionAlreadyExists(
                collection.to_string(),
            ));
        }
        std::fs::create_dir_all(dir)?;
        let dir_lock = CollectionLock::acquire(dir)?;

        let mut meta = Meta::new_with_segment_slots(
            collection,
            config.vector_size,
            config.distance,
            segment_slots,
        );
        // The tier enum makes cross-tier residue unrepresentable.
        meta.index = config.index.clone();
        // Persist quantization config alongside other tier settings.
        meta.quantization_config = config.quantization_config.clone();
        // Persist the admission cap so it survives reopen.
        meta.max_points = config.max_points;
        // Remote-only layout flags round-trip through meta.bin so collection
        // info and re-created requests stay byte-identical with the original.
        meta.vectors_on_disk = config.vectors_on_disk;
        meta.storage_hints = config.storage_hints.clone();
        // The generation manifest is published last, so an interrupted create
        // leaves no partially described serving generation behind.
        let generation = Generation::create(dir)?;
        let data_dir = generation.path();
        meta.save(&data_dir)?;
        super::segments::write(&data_dir, meta.segment_slots, 0)?;

        let vectors = Vectors::create(
            &data_dir.join("vectors.bin"),
            config.vector_size,
            meta.segment_slots,
        )?;
        let keys = Keys::create(&data_dir.join("keys.bin"), meta.slot_capacity)?;
        // Create quantized storage if requested. Scalar/Binary are ready immediately;
        // Product needs a codebook build before it becomes usable.
        let quant = if let Some(qc) = &config.quantization_config {
            Some(QuantStore::create(
                &data_dir,
                config.vector_size,
                config.distance,
                qc,
                meta.segment_slots,
                meta.slot_capacity,
            )?)
        } else {
            None
        };

        let initial_tombstones = TombstoneBits::new(meta.slot_capacity as usize);
        // A fresh collection owns a valid, all-zero table so `open` never has
        // to guess whether a missing file means "nothing deleted".
        initial_tombstones.save(&data_dir, meta.slot_capacity)?;
        let slot_capacity = meta.slot_capacity;
        let wal = Wal::open_or_create(&dir.join("wal"), storage.wal_segment_bytes)?;
        wal.bind_generation(generation.id());
        // New collections always use the Gridstore-style PayloadStore.
        let payload_store =
            PayloadStore::create(&data_dir.join("payloads_store"), storage.payload_store_config())?;
        let files = ArcSwap::from(Arc::new(DataFiles {
            dir: data_dir,
            vectors,
            keys,
            tombstones: ArcSwap::from(Arc::new(initial_tombstones)),
            payload_store: ArcSwap::from(Arc::new(Some(payload_store))),
            quant: parking_lot::RwLock::new(quant),
            index: ArcSwap::from(Arc::new(None)),
            hnsw_segments: ArcSwap::from(Arc::new(Vec::new())),
            ivf_segments: ArcSwap::from(Arc::new(Vec::new())),
            segments: ArcSwap::from(Arc::new(super::segments::SegmentSet::new(
                0,
                meta.segment_slots,
            ))),
        }));
        Ok(Arc::new_cyclic(|weak| Self {
            dir: dir.to_path_buf(),
            _dir_lock: dir_lock,
            generation_poisoned: AtomicBool::new(false),
            generation: RwLock::new(generation),
            files,
            storage: storage.clone(),
            self_weak: weak.clone(),
            inner: RwLock::new(StoreInner {
                meta,
                reverse: IdMap::with_capacity(slot_capacity as usize),
                reserved: IdMap::default(),
                versions: IdMap::default(),
                next_lsn: 1,
                ivf_config: match &config.index {
                    IndexConfig::Ivf(opt) => opt.clone(),
                    _ => None,
                },
                last_drift_ratio: None,
                filter_bitmap: FilterBitmap::with_capacity(slot_capacity as usize),
                payload_indexes: PayloadIndexManager::new(),
            }),
            wal,
            queue: Arc::new(CommitQueue::new()),
            pending: Mutex::new(Vec::new()),
            build_mutex: Mutex::new(()),
            compact_mutex: Mutex::new(()),
            building: AtomicBool::new(false),
            needs_rebuild: AtomicBool::new(false),
            mutations: AtomicU64::new(0),
            txns_since_last_save: AtomicU64::new(0),
            build_inserted: AtomicU64::new(0),
            build_points_total: AtomicU64::new(0),
            metrics: Arc::new(Metrics::default()),
        }))
    }

    /// Open an existing collection directory, rebuilding the id->slot map and
    /// replaying the WAL (idempotent).
    pub fn open(dir: impl AsRef<Path>) -> Result<Arc<Self>> {
        Self::open_with_storage(dir, &StorageConfig::default())
    }

    /// Open an existing collection with explicit engine-level storage tuning.
    pub fn open_with_storage(dir: impl AsRef<Path>, storage: &StorageConfig) -> Result<Arc<Self>> {
        storage.validate()?;
        let dir = dir.as_ref();
        let dir_lock = CollectionLock::acquire(dir)?;
        let generation = Generation::open(dir)?;
        let data_dir = generation.path();
        for name in &["meta.bin", "vectors.bin", "keys.bin"] {
            let path = data_dir.join(name);
            if !path.exists() {
                return Err(VectorSearchError::CollectionIncomplete {
                    dir: data_dir.clone(),
                    file: name.to_string(),
                });
            }
        }
        let mut meta = Meta::load(&data_dir)?;
        // `meta.bin` is written before the generation manifest. If a crash
        // separates them, the manifest is the authority: reapplying the tail
        // is idempotent, while skipping it would lose committed writes.
        meta.last_applied_txn = meta.last_applied_txn.min(generation.lsn());
        let files = Arc::new(DataFiles::open(
            &data_dir,
            &meta,
            storage,
            generation.lsn(),
        )?);

        let mut reverse = IdMap::with_capacity(meta.slot_capacity as usize);
        let mut versions: IdMap<u64> = IdMap::default();
        let mut filter_bitmap = FilterBitmap::with_capacity(meta.slot_capacity as usize);
        let mut payload_indexes = PayloadIndexManager::new();
        for def in PayloadIndexManager::load_defs(&data_dir) {
            if payload_indexes
                .declare(&def.field, def.schema, meta.slot_capacity as usize)
                .is_err()
            {
                continue;
            }
        }
        let tombstone_count = files.tombstones.load().count_ones();
        // The snapshot owns the id directory, not the payload bytes: a payload
        // written after the checkpoint advances the tracker but leaves the
        // directory intact, and WAL replay from the generation lsn re-applies
        // that write. Requiring equal tracker versions would discard the
        // directory after any payload-only mutation and force a full scan.
        let snapshot = DenseSnapshot::read(&data_dir)?
            .filter(|snapshot| snapshot.matches(&generation, &meta, tombstone_count));
        {
            let ps_guard = files.payload_store.load();
            let payload_store = ps_guard.as_ref().as_ref().expect("payload store is present");
            if let Some(snapshot) = &snapshot {
                for point in &snapshot.points {
                    let id = DenseSnapshot::point_id(point);
                    versions.insert_value(&id, point.lsn);
                    if point.deleted {
                        continue;
                    }
                    reverse.insert(&id, point.slot);
                    if let Ok(Some(p)) = payload_store.get(point.slot) {
                        filter_bitmap.register_slot(point.slot, Some(&p));
                        payload_indexes.register_slot(point.slot, Some(&p));
                    }
                }
                meta.last_applied_txn = meta.last_applied_txn.max(snapshot.lsn);
            } else {
                let keys_view = files.keys.snapshot();
                let tombstone_view = files.tombstones.load();
                for slot in 0..meta.next_slot as usize {
                    if let Some(key) = Keys::read_key(&keys_view, slot)? {
                        // Tombstoned slots keep their key in `keys.bin` until
                        // compaction reclaims them, but they must not re-enter the
                        // live id map: a later upsert of the same id has to take a
                        // fresh slot instead of silently writing into a slot that
                        // stays invisible.
                        if tombstone_view.bit(slot) {
                            continue;
                        }
                        let id = PointId::from(key);
                        reverse.insert(&id, slot as u32);
                        versions.insert_value(&id, meta.last_applied_txn);
                        if let Ok(Some(p)) = payload_store.get(slot as u32) {
                            filter_bitmap.register_slot(slot as u32, Some(&p));
                            payload_indexes.register_slot(slot as u32, Some(&p));
                        }
                    }
                }
            }
        }

        let wal = Wal::open_or_create(&dir.join("wal"), storage.wal_segment_bytes)?;
        wal.bind_generation(generation.id());

        let store = Arc::new_cyclic(|weak| Self {
            dir: dir.to_path_buf(),
            _dir_lock: dir_lock,
            generation_poisoned: AtomicBool::new(false),
            generation: RwLock::new(generation),
            files: ArcSwap::from(files),
            storage: storage.clone(),
            self_weak: weak.clone(),
            inner: RwLock::new(StoreInner {
                ivf_config: match &meta.index {
                    IndexConfig::Ivf(opt) => opt.clone(),
                    _ => None,
                },
                meta,
                reverse,
                reserved: IdMap::default(),
                versions,
                next_lsn: 1,
                last_drift_ratio: None,
                filter_bitmap,
                payload_indexes,
            }),
            wal,
            queue: Arc::new(CommitQueue::new()),
            pending: Mutex::new(Vec::new()),
            build_mutex: Mutex::new(()),
            compact_mutex: Mutex::new(()),
            building: AtomicBool::new(false),
            needs_rebuild: AtomicBool::new(false),
            mutations: AtomicU64::new(0),
            txns_since_last_save: AtomicU64::new(0),
            build_inserted: AtomicU64::new(0),
            build_points_total: AtomicU64::new(0),
            metrics: Arc::new(Metrics::default()),
        });
        store.replay_wal()?;
        store.load_index()?;
        Ok(store)
    }

    /// Replay the WAL into memory. Idempotent: upserts overwrite by point id,
    /// deletes of missing points are no-ops, `Compact` only advances the
    /// water mark. Counts are reconciled against the actual slot state, which
    /// covers the crash window where the WAL was fsync'ed but `meta.bin` was
    /// not yet written.
    pub(super) fn replay_wal(&self) -> Result<()> {
        // The WAL watermark alone is not an authority: only the published
        // generation says which operations are already in its data files.
        // Checkpoints write the generation manifest before the WAL watermark,
        // so a larger watermark means a manifest from another generation.
        let generation_lsn = self.generation.read().lsn();
        let ack = self.wal.acked_lsn();
        if ack > generation_lsn {
            return Err(VectorSearchError::CorruptData(format!(
                "WAL ack {ack} is newer than generation lsn {generation_lsn}"
            )));
        }
        let floor = generation_lsn.max(ack);
        let (txns, last) = self.wal.read_all_from(floor)?;
        let mut inner = self.inner.write();
        for txn in &txns {
            self.apply_records_locked(&mut inner, txn.txn_id, &txn.ops)?;
        }
        // Even an empty WAL carries information: the floor tells us how far
        // the data files have been checkpointed. Starting new lsns at 1 after
        // that point would collide with records that were truncated.
        let water_changed = inner.meta.last_applied_txn != last;
        inner.meta.last_applied_txn = inner.meta.last_applied_txn.max(last);
        inner.next_lsn = inner.meta.last_applied_txn + 1;
        let live = inner.reverse.len() as u64;
        let tomb = self.data_arc().tombstones.load().count_ones();
        let changed = water_changed
            || inner.meta.live_count != live
            || inner.meta.tombstone_count != tomb;
        inner.meta.live_count = live;
        inner.meta.tombstone_count = tomb;
        if changed {
            inner.meta.save(&self.data_dir())?;
        }
        Ok(())
    }
}

pub(super) fn validate_point(meta: &Meta, point: &VectorPoint) -> Result<()> {
    if point.vector().len() != meta.vector_size {
        return Err(VectorSearchError::InvalidVectorDimension {
            expected: meta.vector_size,
            actual: point.vector().len(),
        });
    }
    for (i, v) in point.vector().iter().enumerate() {
        if !v.is_finite() {
            return Err(VectorSearchError::NonFiniteElement(i));
        }
    }
    Ok(())
}

/// Reject, before the log append, upserts that would grow the live set past
/// the collection's optional [`max_points`](crate::types::CollectionConfig::max_points)
/// cap.
///
/// Admission has to close before the queue takes the transaction: the log is
/// authoritative, so a record that is guaranteed to fail while being applied
/// would sit in the WAL forever and make every later `open` fail. Ids already
/// present keep their slot, an id reserved by an in-flight transaction keeps
/// the slot that reservation took, and a batch listing the same new id twice
/// must only be counted once.
pub(super) fn validate_admission_locked(inner: &StoreInner, ops: &[WalRecord]) -> Result<()> {
    let Some(cap) = inner.meta.max_points else {
        return Ok(());
    };
    let mut new_ids: HashSet<&str> = HashSet::new();
    for op in ops {
        let WalRecord::Upsert { point } = op else {
            continue;
        };
        let id = PointId::from(point.id.as_str());
        if inner.reverse.contains_key(&id) || inner.reserved.contains_id(&id) {
            continue;
        }
        if !new_ids.insert(point.id.as_str()) {
            continue;
        }
        let needed = inner.meta.next_slot + inner.reserved.len() as u64 + new_ids.len() as u64;
        if needed > cap {
            return Err(VectorSearchError::CapacityExceeded {
                collection: inner.meta.collection.clone(),
                max_points: cap,
                requested: needed,
            });
        }
    }
    Ok(())
}

pub(super) fn threshold_met(meta: &Meta, threshold: f64) -> bool {
    meta.next_slot > 0 && meta.tombstone_count as f64 / meta.next_slot as f64 > threshold
}

/// Number of upserted and deleted points in a WAL operation batch.
pub(super) fn count_ops(ops: &[WalRecord]) -> (u64, u64) {
    let mut upserts = 0u64;
    let mut deletes = 0u64;
    for op in ops {
        match op {
            WalRecord::Upsert { .. } => upserts += 1,
            WalRecord::Delete { .. } => deletes += 1,
            WalRecord::DeleteBatch { point_ids } => deletes += point_ids.len() as u64,
            WalRecord::Compact
            | WalRecord::Quantize
            | WalRecord::DropCollection
            | WalRecord::UpsertVectors { .. }
            | WalRecord::PatchNamedVectors { .. }
            | WalRecord::SetPayload { .. }
            | WalRecord::DeletePayloadKeys { .. }
            | WalRecord::SetPayloadField { .. } => {}
        }
    }
    (upserts, deletes)
}

pub(crate) fn validate_collection_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(VectorSearchError::InvalidCollectionName(name.to_string()));
    }
    if name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.contains('\0')
    {
        return Err(VectorSearchError::InvalidCollectionName(name.to_string()));
    }
    Ok(())
}
