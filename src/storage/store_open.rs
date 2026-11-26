use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::{Mutex, RwLock};

use super::filter_bitmap::FilterBitmap;
use super::keys::Keys;
use super::meta::Meta;
use super::payload_index::PayloadIndexManager;
use super::payload_store::PayloadStore;
use super::quant::QuantStore;
use super::queue::CommitQueue;
use super::tombstones::TombstoneBits;
use super::vectors::Vectors;
use super::{CollectionStore, StoreInner};
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
        meta.save(dir)?;

        let vectors = Vectors::create(
            &dir.join("vectors.bin"),
            config.vector_size,
            meta.segment_slots,
        )?;
        let keys = Keys::create(&dir.join("keys.bin"), meta.slot_capacity)?;
        // Create quantized storage if requested. Scalar/Binary are ready immediately;
        // Product needs a codebook build before it becomes usable.
        let quant = if let Some(qc) = &config.quantization_config {
            Some(QuantStore::create(
                dir,
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
        initial_tombstones.save(dir, meta.slot_capacity)?;
        let tombstones = ArcSwap::from(Arc::new(initial_tombstones));
        let slot_capacity = meta.slot_capacity;
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        // New collections always use the Gridstore-style PayloadStore.
        let payload_store =
            PayloadStore::create(&dir.join("payloads_store"), storage.payload_store_config())?;
        Ok(Arc::new_cyclic(|weak| Self {
            dir: dir.to_path_buf(),
            storage: storage.clone(),
            self_weak: weak.clone(),
            inner: RwLock::new(StoreInner {
                meta,
                reverse: IdMap::with_capacity(slot_capacity as usize),
                reserved: IdMap::default(),
                next_lsn: 1,
                ivf_config: match &config.index {
                    IndexConfig::Ivf(opt) => opt.clone(),
                    _ => None,
                },
                last_drift_ratio: None,
                filter_bitmap: FilterBitmap::with_capacity(slot_capacity as usize),
                payload_indexes: PayloadIndexManager::new(),
            }),
            tombstones,
            vectors,
            keys,
            payload_store: ArcSwap::from(Arc::new(Some(payload_store))),
            quant: parking_lot::RwLock::new(quant),
            wal,
            queue: Arc::new(CommitQueue::new()),
            index: ArcSwap::from(Arc::new(None)),
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
        for name in &["meta.bin", "vectors.bin", "keys.bin"] {
            let path = dir.join(name);
            if !path.exists() {
                return Err(VectorSearchError::CollectionIncomplete {
                    dir: dir.to_path_buf(),
                    file: name.to_string(),
                });
            }
        }
        let meta = Meta::load(dir)?;

        let vectors = Vectors::open(
            &dir.join("vectors.bin"),
            meta.vector_size,
            meta.segment_slots,
            meta.slot_capacity,
        )?;
        let keys = Keys::open(&dir.join("keys.bin"))?;
        // The tombstone table outlives the WAL: once a checkpoint truncates the
        // log, this file is the only record of the deletes. Load it before any
        // scan so a deleted slot is never reported live.
        let tombstone_bits =
            TombstoneBits::load(dir, meta.slot_capacity as usize, meta.tombstone_count)?
                .unwrap_or_else(|| TombstoneBits::new(meta.slot_capacity as usize));

        let payload_store_path = dir.join("payloads_store");
        // Opened once and reused for the rebuild pass below: opening twice
        // re-read the tracker and bitmask for nothing.
        let payload_store = PayloadStore::open(&payload_store_path)?;

        let mut reverse = IdMap::with_capacity(meta.slot_capacity as usize);
        let mut filter_bitmap = FilterBitmap::with_capacity(meta.slot_capacity as usize);
        let mut payload_indexes = PayloadIndexManager::new();
        for def in PayloadIndexManager::load_defs(dir) {
            if payload_indexes
                .declare(&def.field, def.schema, meta.slot_capacity as usize)
                .is_err()
            {
                continue;
            }
        }
        {
            let keys_view = keys.snapshot();
            let tombstone_view = &tombstone_bits;
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
                    if let Ok(Some(p)) = payload_store.get(slot as u32) {
                        filter_bitmap.register_slot(slot as u32, Some(&p));
                        payload_indexes.register_slot(slot as u32, Some(&p));
                    }
                }
            }
        }

        // Load quantized storage if configured. Missing or corrupt quant files are
        // treated as absent (derived structure) so open still succeeds; the
        // collection will run exact until `build_quantization` recreates them.
        let quant = if meta.quantization_config.is_some() {
            QuantStore::open(
                dir,
                meta.vector_size,
                meta.distance,
                meta.segment_slots,
                meta.slot_capacity,
            )?
        } else {
            None
        };

        let payload_store = Some(payload_store);
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        let tombstones = ArcSwap::from(Arc::new(tombstone_bits));

        let store = Arc::new_cyclic(|weak| Self {
            dir: dir.to_path_buf(),
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
                next_lsn: 1,
                last_drift_ratio: None,
                filter_bitmap,
                payload_indexes,
            }),
            tombstones,
            vectors,
            keys,
            payload_store: ArcSwap::from(Arc::new(payload_store)),
            quant: parking_lot::RwLock::new(quant),
            wal,
            queue: Arc::new(CommitQueue::new()),
            index: ArcSwap::from(Arc::new(None)),
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
        let (txns, last) = self.wal.read_all()?;
        if !txns.is_empty() {
            let mut inner = self.inner.write();
            for txn in &txns {
                self.apply_records_locked(&mut inner, &txn.ops)?;
            }
            inner.meta.last_applied_txn = inner.meta.last_applied_txn.max(last);
            // Hand out lsns above the replayed water mark so a record that
            // just replayed can never collide with a new one.
            inner.next_lsn = inner.meta.last_applied_txn + 1;
            let live = inner.reverse.len() as u64;
            let tomb = self.tombstones.load().count_ones();
            let changed = inner.meta.live_count != live || inner.meta.tombstone_count != tomb;
            inner.meta.live_count = live;
            inner.meta.tombstone_count = tomb;
            if changed {
                inner.meta.save(&self.dir)?;
            }
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
        if inner.reverse.contains_key(&id) || inner.reserved.contains_key(&id) {
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
