//! Per-collection storage assembly.
//!
//! A collection directory contains:
//! - `meta.bin`     postcard `Meta`
//! - `vectors.bin`  dense row-major f32, segmented mmap
//! - `keys.bin`     slot -> point id directory
//! - `payloads.bin` slot -> payload blob directory + tombstone flags
//! - `wal.bin`      append-only transaction log (crash recovery)
//!
//! Writes are serialized through the store's `RwLock`; readers snapshot the
//! mmap-backed `ArcSwap` views and can scan without the lock afterwards.
//! Mutations go through [`CollectionStore::apply_txn`]: the WAL is appended
//! and fsync'ed before the transaction is applied to memory, so a crash at
//! any point recovers by idempotent replay.
//!
//! This module hosts the store type itself plus its lifecycle and
//! mutation/read paths; orthogonal concerns live in sibling modules:
//! - [`tombstones`]   lock-free tombstone table
//! - [`compaction`]   physical tombstone reclamation
//! - [`index_lifecycle`]  ANN index load/build/publish/drop
//! - [`search`]       exact scan and ANN search execution

mod compaction;
mod directory;
mod filter_bitmap;
mod index_lifecycle;
mod keys;
mod meta;
mod payload_index;
mod payload_key;
mod payload_store;
mod payloads;
pub(crate) mod quant;
mod search;
mod tombstones;
pub(crate) mod vectors;
mod wal;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use bitvec::prelude::*;
use parking_lot::{Mutex, RwLock};

pub(crate) use meta::Meta;
pub(crate) use payload_index::PayloadIndexManager;
pub(crate) use payload_store::PayloadStore;
pub(crate) use tombstones::TombstoneBits;
pub use wal::{Wal, WalPoint, WalRecord, WalTxn};

use self::filter_bitmap::FilterBitmap;
use self::index_lifecycle::PublishedIndex;
use self::keys::Keys;
use self::payloads::Payloads;
use self::quant::QuantStore;
use self::vectors::Vectors;

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::types::{
    CollectionConfig, DistanceMetric, IndexType, IvfConfig, Payload, PointId, QuantizationConfig,
    VectorPoint,
};

/// Tombstone ratio above which deletes trigger compaction.
const COMPACTION_THRESHOLD: f64 = 0.20;

/// How many WAL transactions between meta.bin saves. The WAL is always
/// fsynced on every transaction, so crash safety is preserved; meta.bin
/// is an optimization that reduces replay time on restart.
const META_SAVE_INTERVAL: u64 = 64;

/// In-memory mutable state, guarded by the store's `RwLock`.
struct StoreInner {
    meta: Meta,
    reverse: HashMap<PointId, u32>,
    /// IVF configuration for `IndexType::IVF` collections. Persisted in
    /// `meta.bin`; the engine may still override it at runtime via
    /// [`CollectionStore::set_ivf_config`].
    ivf_config: Option<IvfConfig>,
    /// Last measured drift ratio, exposed via `CollectionInfo`.
    last_drift_ratio: Option<f64>,
    /// Slot-indexed pre-filter bitmap for equality payload conditions.
    filter_bitmap: FilterBitmap,
    /// Declared per-field payload indexes (MapIndex / NumericIndex).
    payload_indexes: PayloadIndexManager,
}

/// A single opened collection.
pub struct CollectionStore {
    dir: PathBuf,
    inner: RwLock<StoreInner>,
    /// Tombstone mirror (slot -> deleted) for lock-free scans.
    tombstones: ArcSwap<TombstoneBits>,
    vectors: Vectors,
    keys: Keys,
    payloads: Payloads,
    /// Gridstore-style payload storage for all payload operations.
    /// Wrapped in `ArcSwap` for lock-free snapshot reads during search.
    payload_store: ArcSwap<Option<PayloadStore>>,
    /// Optional quantized storage (Scalar/Binary/Product). `None` when
    /// quantization is disabled or not yet built.
    quant: parking_lot::RwLock<Option<QuantStore>>,
    wal: Wal,
    /// Published ANN index; `None` = exact scan. Swapped atomically.
    index: ArcSwap<Option<PublishedIndex>>,
    /// Slots inserted while no index was published and a build was in
    /// flight; drained into the index on publish so probe search never
    /// misses them.
    pending: ArcSwap<Vec<u32>>,
    /// Serializes index build/drain vs compaction. Lock order:
    /// build_mutex or compact_mutex (never both held simultaneously).
    build_mutex: Mutex<()>,
    compact_mutex: Mutex<()>,
    /// Set while an index build is in flight (routes inserts to `pending`).
    building: AtomicBool,
    /// Compaction invalidated a published index and a rebuild should be
    /// scheduled by the engine maintenance worker.
    needs_rebuild: AtomicBool,
    /// Monotonic counter bumped on every applied mutation batch. Compaction
    /// uses it to detect that no write raced its temp-file rewrite phase, so
    /// the commit can swap the files without re-validating every slot.
    mutations: AtomicU64,
    /// Transactions since the last meta.bin save. Used to amortize fsync
    /// cost; the WAL guarantees crash safety independently.
    txns_since_last_save: AtomicU64,
    /// In-flight index build progress: slots incorporated so far / total
    /// slots targeted. Reset at build start, finalized at completion;
    /// surfaced through [`crate::types::IndexInfo`].
    build_inserted: AtomicU64,
    build_points_total: AtomicU64,
    /// Operational metrics (counters and latency histograms), wait-free.
    metrics: Arc<Metrics>,
}

impl CollectionStore {
    /// Create a new collection directory and open it.
    pub fn create(
        dir: impl AsRef<Path>,
        collection: &str,
        config: &CollectionConfig,
    ) -> Result<Self> {
        Self::create_with_segment_slots(dir, collection, config, meta::SEGMENT_SLOTS_DEFAULT)
    }

    /// Like [`CollectionStore::create`] but with an explicit segment slot count.
    #[doc(hidden)]
    pub fn create_with_segment_slots(
        dir: impl AsRef<Path>,
        collection: &str,
        config: &CollectionConfig,
        segment_slots: u32,
    ) -> Result<Self> {
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
        if matches!(
            config.index_type.unwrap_or(IndexType::HNSW),
            IndexType::HNSW
        ) {
            if let Some(hnsw) = &config.hnsw_config {
                hnsw.validate()?;
            }
        }
        if config.index_type == Some(IndexType::IVF) {
            if let Some(ivf) = &config.ivf_config {
                ivf.validate()?;
            }
        }
        if let Some(qc) = &config.quantization_config {
            qc.validate(config.vector_size)?;
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
        // HNSW is the default ANN tier; FLAT keeps exact scan only.
        meta.index_type = config.index_type.unwrap_or(IndexType::HNSW);
        match meta.index_type {
            IndexType::HNSW => {
                meta.hnsw_config = Some(config.hnsw_config.clone().unwrap_or_default());
                meta.ivf_config = None;
            }
            IndexType::IVF => {
                meta.ivf_config = config.ivf_config.clone();
                meta.hnsw_config = None;
            }
            IndexType::FLAT => {
                meta.ivf_config = None;
                meta.hnsw_config = None;
            }
        }
        // Persist quantization config alongside other tier settings.
        if let Some(qc) = &config.quantization_config {
            if qc.enabled {
                meta.quantization_config = Some(qc.clone());
            }
        }
        meta.save(dir)?;

        let vectors = Vectors::create(
            &dir.join("vectors.bin"),
            config.vector_size,
            meta.segment_slots,
        )?;
        let keys = Keys::create(&dir.join("keys.bin"), meta.slot_capacity)?;
        let payloads = Payloads::create(&dir.join("payloads.bin"), meta.slot_capacity)?;
        // Create quantized storage if requested. Scalar/Binary are ready immediately;
        // Product needs a codebook build before it becomes usable.
        let quant = if let Some(qc) = &config.quantization_config {
            if qc.enabled && qc.quant_type.is_some() {
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
            }
        } else {
            None
        };

        let tombstones = ArcSwap::from(Arc::new(TombstoneBits::new(meta.slot_capacity as usize)));
        let slot_capacity = meta.slot_capacity;
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        // New collections always use the Gridstore-style PayloadStore.
        let payload_store = PayloadStore::create(
            &dir.join("payloads_store"),
            payload_store::StoreConfig::default(),
        )?;
        Ok(Self {
            dir: dir.to_path_buf(),
            inner: RwLock::new(StoreInner {
                meta,
                reverse: HashMap::new(),
                ivf_config: config.ivf_config.clone(),
                last_drift_ratio: None,
                filter_bitmap: FilterBitmap::with_capacity(slot_capacity as usize),
                payload_indexes: PayloadIndexManager::new(),
            }),
            tombstones,
            vectors,
            keys,
            payloads,
            payload_store: ArcSwap::from(Arc::new(Some(payload_store))),
            quant: parking_lot::RwLock::new(quant),
            wal,
            index: ArcSwap::from(Arc::new(None)),
            pending: ArcSwap::from(Arc::new(Vec::new())),
            build_mutex: Mutex::new(()),
            compact_mutex: Mutex::new(()),
            building: AtomicBool::new(false),
            needs_rebuild: AtomicBool::new(false),
            mutations: AtomicU64::new(0),
            txns_since_last_save: AtomicU64::new(0),
            build_inserted: AtomicU64::new(0),
            build_points_total: AtomicU64::new(0),
            metrics: Arc::new(Metrics::default()),
        })
    }

    /// Open an existing collection directory, rebuilding the id->slot map and
    /// replaying the WAL (idempotent).
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        for name in &["meta.bin", "vectors.bin", "keys.bin", "payloads.bin"] {
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
        let payloads = Payloads::open(&dir.join("payloads.bin"))?;

        let payload_store_path = dir.join("payloads_store");

        let mut reverse = HashMap::new();
        let tombstones = bitvec![0; meta.slot_capacity as usize];
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
            let ps = PayloadStore::open(&payload_store_path)?;
            for slot in 0..meta.next_slot as usize {
                if let Some(key) = Keys::read_key(&keys_view, slot)? {
                    let id = PointId::from(key);
                    reverse.insert(id, slot as u32);
                    if let Ok(Some(p)) = ps.get(slot as u32) {
                        filter_bitmap.register_slot(slot as u32, Some(&p));
                        payload_indexes.register_slot(slot as u32, Some(&p));
                    }
                }
            }
        }

        // Load quantized storage if configured. Missing or corrupt quant files are
        // treated as absent (derived structure) so open still succeeds; the
        // collection will run exact until `build_quantization` recreates them.
        let quant = if let Some(qc) = &meta.quantization_config {
            if qc.enabled && qc.quant_type.is_some() {
                QuantStore::open(
                    dir,
                    meta.vector_size,
                    meta.distance,
                    meta.segment_slots,
                    meta.slot_capacity,
                )?
            } else {
                None
            }
        } else {
            None
        };

        let payload_store = Some(PayloadStore::open(&payload_store_path)?);

        let store = Self {
            dir: dir.to_path_buf(),
            inner: RwLock::new(StoreInner {
                ivf_config: meta.ivf_config.clone(),
                meta,
                reverse,
                last_drift_ratio: None,
                filter_bitmap,
                payload_indexes,
            }),
            tombstones: ArcSwap::from(Arc::new(TombstoneBits::from_bits(tombstones))),
            vectors,
            keys,
            payloads,
            payload_store: ArcSwap::from(Arc::new(payload_store)),
            quant: parking_lot::RwLock::new(quant),
            wal: Wal::open_or_create(&dir.join("wal.bin"))?,
            index: ArcSwap::from(Arc::new(None)),
            pending: ArcSwap::from(Arc::new(Vec::new())),
            build_mutex: Mutex::new(()),
            compact_mutex: Mutex::new(()),
            building: AtomicBool::new(false),
            needs_rebuild: AtomicBool::new(false),
            mutations: AtomicU64::new(0),
            txns_since_last_save: AtomicU64::new(0),
            build_inserted: AtomicU64::new(0),
            build_points_total: AtomicU64::new(0),
            metrics: Arc::new(Metrics::default()),
        };
        store.replay_wal()?;
        store.load_index()?;
        Ok(store)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    // ── Payload routing helpers ────────────────────────────────────────

    /// Read the payload for a slot from PayloadStore.
    fn read_payload_at(&self, slot: u32) -> Result<Option<Payload>> {
        let ps_guard = self.payload_store.load();
        let ps = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        ps.get(slot)
    }

    /// Write a payload for a slot to PayloadStore.
    fn write_payload_at(&self, slot: u32, payload: Option<&Payload>) -> Result<()> {
        let ps_guard = self.payload_store.load();
        let ps = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        ps.put(slot, payload)
    }

    /// Delete specific keys from a slot's payload in PayloadStore.
    fn delete_keys_at(&self, slot: u32, keys: &[&str]) -> Result<()> {
        let ps_guard = self.payload_store.load();
        let ps = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        ps.delete_keys(slot, keys)
    }

    /// Merge the given fields into a slot's payload in PayloadStore.
    fn merge_payload_at(&self, slot: u32, partial: &Payload) -> Result<()> {
        let ps_guard = self.payload_store.load();
        let ps = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        ps.merge(slot, partial.iter().map(|(k, v)| (k.clone(), v.clone())))
    }

    /// Re-index a slot's pre-filter structures after a non-upsert payload
    /// mutation (`SetPayload`/`SetPayloadField`/`DeletePayloadKeys`). The
    /// bitmap is conservative but must reflect current values: `build_mask`
    /// ANDs condition masks together, so an unregistered value would
    /// produce an all-zero candidate mask and wrongly exclude matches.
    fn refresh_filter_bitmap_locked(&self, inner: &mut StoreInner, slot: u32) {
        let payload = self.read_payload_at(slot).ok().flatten();
        inner.filter_bitmap.register_slot(slot, payload.as_ref());
        inner.payload_indexes.register_slot(slot, payload.as_ref());
    }

    /// Snapshot of the metadata (dimension, metric, counts).
    pub fn meta(&self) -> Meta {
        self.inner.read().meta.clone()
    }

    /// Operational metrics recorder for this collection.
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Metrics sink shared with the published ANN indexes for lock
    /// contention and version-reload diagnostics.
    pub(crate) fn metrics_arc(&self) -> Arc<Metrics> {
        Arc::clone(&self.metrics)
    }

    /// In-flight index build progress: (building, inserted, total).
    pub(crate) fn build_progress(&self) -> (bool, u64, u64) {
        (
            self.building.load(AtomicOrdering::Relaxed),
            self.build_inserted.load(AtomicOrdering::Relaxed),
            self.build_points_total.load(AtomicOrdering::Relaxed),
        )
    }

    /// Upsert a point. Existing ids reuse their slot (overwrite); new ids get
    /// the next free slot (growing capacity when needed).
    ///
    /// Raw apply without WAL: engine paths must use [`CollectionStore::apply_txn`]
    /// so that mutations are crash-safe and recoverable by replay.
    pub fn upsert(&self, point: &VectorPoint) -> Result<()> {
        let mut inner = self.inner.write();
        self.apply_upsert_locked(&mut inner, point)?;
        inner.meta.save(&self.dir)?;
        Ok(())
    }

    /// Delete a point by id (tombstone; physical removal happens on
    /// compaction). Returns whether the point existed.
    pub fn delete(&self, id: &PointId) -> Result<bool> {
        let mut inner = self.inner.write();
        let slot = match inner.reverse.remove(id) {
            Some(slot) => slot,
            None => return Ok(false),
        };
        self.apply_delete_slot_locked(&mut inner, slot)?;
        // Physical reclamation is scheduled by the engine's maintenance
        // worker once the tombstone ratio crosses the compaction threshold;
        // reads already filter tombstoned slots.
        Ok(true)
    }

    /// Apply a WAL-backed transaction:
    ///
    /// 1. validate every op (no invalid data may be logged);
    /// 2. append the whole `WalTxn` to `wal.bin` + fsync;
    /// 3. apply to memory and the mmap files;
    /// 4. persist `meta.bin` (with `last_applied_txn` advanced).
    ///
    /// On success the caller's buffer may be drained; on failure nothing was
    /// applied in memory and the call may be retried (replay is idempotent).
    pub fn apply_txn(&self, txn: &WalTxn) -> Result<()> {
        let started = std::time::Instant::now();
        {
            let mut inner = self.inner.write();
            for op in &txn.ops {
                if let WalRecord::Upsert { point } = op {
                    let point = point.to_point()?;
                    validate_point(&inner.meta, &point)?;
                }
            }
            let prev_next_slot = inner.meta.next_slot;
            let prev_capacity = inner.meta.slot_capacity;
            self.wal.append(txn)?;
            self.apply_records_locked(&mut inner, &txn.ops)?;
            // Monotonic water mark: late/duplicated txn ids must not regress
            // the last applied id (replay is idempotent, so this is safe).
            inner.meta.last_applied_txn = inner.meta.last_applied_txn.max(txn.txn_id);
            let slot_changed =
                inner.meta.next_slot != prev_next_slot || inner.meta.slot_capacity != prev_capacity;
            let count = self
                .txns_since_last_save
                .fetch_add(1, AtomicOrdering::Relaxed)
                + 1;
            if slot_changed || count >= META_SAVE_INTERVAL {
                inner.meta.save(&self.dir)?;
                self.txns_since_last_save.store(0, AtomicOrdering::Relaxed);
            }
        }
        let (upserts, deletes) = count_ops(&txn.ops);
        self.metrics
            .record_apply_txn(upserts, deletes, started.elapsed());
        Ok(())
    }

    /// Apply a WAL-backed batch of records with an auto-assigned txn id.
    ///
    /// This is the crash-safe path for single-collection operations exposed by
    /// the engine; unlike [`CollectionStore::apply_txn`] the caller does not
    /// coordinate a graph transaction id.
    pub fn apply_ops(&self, ops: &[WalRecord]) -> Result<()> {
        let txn_id = self.inner.read().meta.last_applied_txn + 1;
        self.apply_txn(&WalTxn {
            txn_id,
            ops: ops.to_vec(),
        })
    }

    /// Delete every live point matching `filter`. Returns the number of
    /// deleted points. Runs as a single WAL-backed batch so recovery replays
    /// the filter match deterministically.
    pub fn delete_by_filter(&self, filter: &crate::types::VectorFilter) -> Result<u64> {
        let point_ids: Vec<String> = {
            let inner = self.inner.read();
            let tombstones = self.tombstones.load();
            let keysnap = self.keys.snapshot();
            let mut point_ids = Vec::new();
            for slot in 0..inner.meta.next_slot as usize {
                if tombstones.bit(slot) {
                    continue;
                }
                let key = Keys::read_key(&keysnap, slot)?.ok_or_else(|| {
                    VectorSearchError::CorruptData(format!("slot {slot} has no key"))
                })?;
                let id = PointId::from(key);
                let payload = self.read_payload_at(slot as u32)?;
                if crate::filter::matches(filter, &id, payload.as_ref())? {
                    point_ids.push(id.to_string());
                }
            }
            point_ids
        };
        if point_ids.is_empty() {
            return Ok(0);
        }
        let deleted = point_ids.len() as u64;
        self.apply_ops(&[WalRecord::DeleteBatch { point_ids }])?;
        Ok(deleted)
    }

    /// Replay the WAL into memory. Idempotent: upserts overwrite by point id,
    /// deletes of missing points are no-ops, `Compact` only advances the
    /// water mark. Counts are reconciled against the actual slot state, which
    /// covers the crash window where the WAL was fsync'ed but `meta.bin` was
    /// not yet written.
    fn replay_wal(&self) -> Result<()> {
        let (txns, last) = self.wal.read_all()?;
        if !txns.is_empty() {
            let mut inner = self.inner.write();
            for txn in &txns {
                self.apply_records_locked(&mut inner, &txn.ops)?;
            }
            inner.meta.last_applied_txn = inner.meta.last_applied_txn.max(last);
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

    /// WAL accessor for the engine (per-collection transactions) and tests.
    pub fn wal(&self) -> &Wal {
        &self.wal
    }

    fn apply_records_locked(&self, inner: &mut StoreInner, ops: &[WalRecord]) -> Result<()> {
        // One bump per batch: compaction compares this counter across its
        // temp-file rewrite phase to prove no write raced it.
        self.mutations.fetch_add(1, AtomicOrdering::Relaxed);
        for op in ops {
            match op {
                WalRecord::Upsert { point } => {
                    let point = point.to_point()?;
                    self.apply_upsert_locked(inner, &point)?;
                }
                WalRecord::Delete { point_id } => self.apply_delete_locked(inner, point_id)?,
                WalRecord::DeleteBatch { point_ids } => {
                    for id in point_ids {
                        self.apply_delete_locked(inner, id)?;
                    }
                }
                WalRecord::SetPayload { slot, payload } => {
                    let parsed = match payload {
                        Some(s) => Some(serde_json::from_str::<Payload>(s)?),
                        None => None,
                    };
                    self.write_payload_at(*slot, parsed.as_ref())?;
                    self.refresh_filter_bitmap_locked(inner, *slot);
                }
                WalRecord::DeletePayloadKeys { slot, keys } => {
                    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
                    self.delete_keys_at(*slot, &key_refs)?;
                    self.refresh_filter_bitmap_locked(inner, *slot);
                }
                WalRecord::SetPayloadField { slot, key, value } => {
                    let parsed: serde_json::Value = serde_json::from_str(value)?;
                    let partial = Payload::from([(key.clone(), parsed)]);
                    self.merge_payload_at(*slot, &partial)?;
                    self.refresh_filter_bitmap_locked(inner, *slot);
                }
                // Checkpoint / quantization markers carry no separate mutation.
                WalRecord::Compact | WalRecord::Quantize | WalRecord::DropCollection => {}
            }
        }
        Ok(())
    }

    fn apply_upsert_locked(&self, inner: &mut StoreInner, point: &VectorPoint) -> Result<()> {
        validate_point(&inner.meta, point)?;

        let slot = if let Some(slot) = inner.reverse.get(&point.id) {
            *slot as u64
        } else {
            let slot = inner.meta.next_slot;
            self.ensure_capacity(inner, slot + 1)?;
            self.keys.append_key(slot as usize, &point.id.to_string())?;
            inner.reverse.insert(point.id.clone(), slot as u32);
            inner.meta.next_slot += 1;
            inner.meta.live_count += 1;
            slot
        };

        self.vectors.write_slot(slot, &point.vector)?;
        // Mirror the vector into quantized storage when ready.
        if let Some(q) = self.quant.read().as_ref() {
            if q.is_ready() {
                let _ = q.write_slot(slot, &point.vector);
            } else if matches!(
                q.config().quant_type,
                Some(crate::types::QuantizationType::Scalar { .. })
                    | Some(crate::types::QuantizationType::Binary { .. })
            ) {
                // Scalar/Binary are always ready after create/rebuild; attempt to
                // quantize even when `ready` flag races the build.
                let _ = q.write_slot(slot, &point.vector);
            }
        }
        self.write_payload_at(slot as u32, point.payload.as_ref())?;
        self.register_slot(slot as u32, &point.vector, inner.meta.segment_slots);
        inner
            .filter_bitmap
            .register_slot(slot as u32, point.payload.as_ref());
        inner
            .payload_indexes
            .register_slot(slot as u32, point.payload.as_ref());
        Ok(())
    }

    fn apply_delete_locked(&self, inner: &mut StoreInner, point_id: &str) -> Result<()> {
        let id = PointId::from(point_id.to_string());
        if let Some(slot) = inner.reverse.remove(&id) {
            self.apply_delete_slot_locked(inner, slot)?;
        }
        Ok(())
    }

    fn apply_delete_slot_locked(&self, inner: &mut StoreInner, slot: u32) -> Result<()> {
        self.payloads.set_tombstone(slot as usize, true)?;
        self.update_tombstone_bit(slot as usize, true);
        inner.filter_bitmap.unregister_slot(slot);
        inner.payload_indexes.unregister_slot(slot);
        inner.meta.live_count = inner.meta.live_count.saturating_sub(1);
        inner.meta.tombstone_count += 1;
        Ok(())
    }

    /// Fetch a point by id.
    pub fn get(&self, id: &PointId) -> Result<Option<VectorPoint>> {
        let inner = self.inner.read();
        let slot = match inner.reverse.get(id) {
            Some(slot) => *slot as u64,
            None => return Ok(None),
        };
        if self.tombstones.load().bit(slot as usize) {
            return Ok(None);
        }
        let dim = inner.meta.vector_size;
        let vsnap = self.vectors.snapshot();
        let vector = Vectors::read_slot(&vsnap, slot, inner.meta.segment_slots, dim)
            .ok_or_else(|| {
                VectorSearchError::CorruptData(format!("slot {slot} out of vectors.bin range"))
            })?
            .to_vec();
        let payload = self.read_payload_at(slot as u32)?;
        Ok(Some(VectorPoint {
            id: id.clone(),
            vector,
            payload,
        }))
    }

    /// Number of live points.
    pub fn count(&self) -> u64 {
        self.inner.read().meta.live_count
    }

    /// Replace the payload for a single point. The point must exist and not
    /// be tombstoned. The entire payload map is replaced atomically via a
    /// WAL-backed transaction.
    pub fn set_payload(&self, id: &PointId, payload: Payload) -> Result<()> {
        let slot = self.live_slot_of(id)?;
        let json = serde_json::to_string(&payload)?;
        self.apply_ops(&[WalRecord::SetPayload {
            slot,
            payload: Some(json),
        }])
    }

    /// Set a single field on a point's payload (merge semantics). A missing
    /// payload is created containing just this field; all other keys are
    /// preserved. Applied atomically via a WAL-backed transaction.
    pub fn set_payload_field(
        &self,
        id: &PointId,
        key: String,
        value: serde_json::Value,
    ) -> Result<()> {
        self.set_payload_fields(id, Payload::from([(key, value)]))
    }

    /// Merge the given fields into a point's payload within one WAL-backed
    /// transaction: keys in `fields` overwrite their previous values while
    /// all other keys are preserved. A missing payload is created. The point
    /// must exist and not be tombstoned.
    pub fn set_payload_fields(&self, id: &PointId, fields: Payload) -> Result<()> {
        if fields.is_empty() {
            return Ok(());
        }
        let slot = self.live_slot_of(id)?;
        let mut ops = Vec::with_capacity(fields.len());
        for (key, value) in fields {
            let json = serde_json::to_string(&value)?;
            ops.push(WalRecord::SetPayloadField {
                slot,
                key,
                value: json,
            });
        }
        self.apply_ops(&ops)
    }

    /// Remove specific keys from a point's payload. The remaining keys are
    /// preserved. If the point has no payload this is a no-op.
    pub fn delete_payload_keys(&self, id: &PointId, keys: Vec<String>) -> Result<()> {
        let slot = self.live_slot_of(id)?;
        self.apply_ops(&[WalRecord::DeletePayloadKeys { slot, keys }])
    }

    /// Resolve a live point's slot or fail with `InvalidPointId`.
    fn live_slot_of(&self, id: &PointId) -> Result<u32> {
        let slot = self
            .inner
            .read()
            .reverse
            .get(id)
            .copied()
            .ok_or_else(|| VectorSearchError::InvalidPointId(id.to_string()))?;
        if self.tombstones.load().bit(slot as usize) {
            return Err(VectorSearchError::InvalidPointId(format!(
                "point {} is tombstoned",
                id
            )));
        }
        Ok(slot)
    }

    // ── Payload field indexes ──────────────────────────────────────────

    /// Create a payload field index. The index is populated from the
    /// current live set synchronously (under the store write lock) and its
    /// definition persisted to `payload_indexes.json`, so a restart or
    /// concurrent search either sees the complete index or none.
    pub fn create_payload_index(
        &self,
        field: &str,
        schema: crate::types::PayloadSchemaType,
    ) -> Result<()> {
        let mut inner = self.inner.write();
        let capacity = inner.meta.slot_capacity as usize;
        inner.payload_indexes.declare(field, schema, capacity)?;
        // Populate from the current live payloads; registration for slots
        // without an indexable value on `field` is a no-op.
        let tombstones = self.tombstones.load();
        for slot in 0..inner.meta.next_slot as u32 {
            if tombstones.bit(slot as usize) {
                continue;
            }
            let payload = self.read_payload_at(slot).ok().flatten();
            inner.payload_indexes.register_slot(slot, payload.as_ref());
        }
        inner.payload_indexes.save_defs(&self.dir)?;
        Ok(())
    }

    /// Drop the payload field index on `field`. Returns whether it existed.
    /// The persisted definitions are updated only after successful removal.
    pub fn delete_payload_index(&self, field: &str) -> Result<bool> {
        let mut inner = self.inner.write();
        if !inner.payload_indexes.delete(field) {
            return Ok(false);
        }
        inner.payload_indexes.save_defs(&self.dir)?;
        Ok(true)
    }

    /// All declared payload indexes as `(field, schema_type)` pairs.
    pub fn list_payload_indexes(&self) -> Vec<(String, crate::types::PayloadSchemaType)> {
        self.inner
            .read()
            .payload_indexes
            .defs()
            .into_iter()
            .map(|d| (d.field, d.schema))
            .collect()
    }

    /// Paginated scan over live points in slot order.
    ///
    /// Returns up to `limit` points starting after `offset` (the last
    /// point_id from the previous page). When `with_payload` is false the
    /// payload field is omitted; when `with_vector` is false the vector
    /// field is omitted. The returned `Option<String>` is the id of the
    /// last point in the page (use as the next `offset`), or `None` when
    /// there are no more pages.
    pub fn scroll(
        &self,
        limit: usize,
        offset: Option<&str>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<String>)> {
        let inner = self.inner.read();
        let tombstones = self.tombstones.load();
        let keysnap = self.keys.snapshot();
        let vsnap = self.vectors.snapshot();
        let dim = inner.meta.vector_size;
        let include_payload = with_payload.unwrap_or(true);
        let include_vector = with_vector.unwrap_or(false);

        let mut skip = offset.is_some();
        let mut results = Vec::with_capacity(limit);
        let mut last_id: Option<String> = None;

        for slot in 0..inner.meta.next_slot as usize {
            if tombstones.bit(slot) {
                continue;
            }
            let key = Keys::read_key(&keysnap, slot)?
                .ok_or_else(|| VectorSearchError::CorruptData(format!("slot {slot} has no key")))?;
            if skip {
                if key == offset.unwrap_or("") {
                    skip = false;
                }
                continue;
            }
            let id = PointId::from(key);
            let payload = if include_payload {
                self.read_payload_at(slot as u32)?
            } else {
                None
            };
            let vector = if include_vector {
                Some(
                    Vectors::read_slot(&vsnap, slot as u64, inner.meta.segment_slots, dim)
                        .ok_or_else(|| {
                            VectorSearchError::CorruptData(format!(
                                "slot {slot} out of vectors.bin range"
                            ))
                        })?
                        .to_vec(),
                )
            } else {
                None
            };
            last_id = Some(id.to_string());
            results.push(VectorPoint {
                id,
                vector: vector.unwrap_or_default(),
                payload,
            });
            if results.len() >= limit {
                break;
            }
        }
        Ok((results, last_id))
    }

    /// Whether the tombstone ratio has crossed the compaction threshold.
    ///
    /// The engine polls this after mutations and schedules a background
    /// compaction; visibility of data never depends on compaction having run.
    pub fn needs_compaction(&self) -> bool {
        let inner = self.inner.read();
        threshold_met(&inner.meta)
    }

    /// Incrementally repair the HNSW graph by removing references to
    /// tombstoned slots from adjacency lists.
    ///
    /// This is an alternative to full rebuild after compaction: instead of
    /// invalidating the entire index, only nodes with stale references are
    /// touched. The repair is idempotent and can be called repeatedly as
    /// tombstones accumulate.
    ///
    /// Returns the number of nodes whose adjacency lists were modified,
    /// or 0 if no index is published or no repair was needed.
    pub fn repair_hnsw(&self) -> Result<usize> {
        let index = self.index.load();
        let Some(published) = index.as_ref() else {
            return Ok(0);
        };
        let PublishedIndex::Hnsw(hnsw) = published else {
            return Ok(0);
        };

        let tombstones = self.tombstones.load();
        let vsnap = self.vectors.snapshot();
        let segment_slots = self.inner.read().meta.segment_slots;

        Ok(hnsw.repair(&tombstones, &vsnap, segment_slots))
    }

    /// Grow storage to accommodate `needed_slots` slots (0-indexed high water).
    fn ensure_capacity(&self, inner: &mut StoreInner, needed_slots: u64) -> Result<()> {
        let meta = &mut inner.meta;
        if needed_slots <= meta.slot_capacity {
            return Ok(());
        }
        self.vectors.grow_to(needed_slots)?;
        let new_capacity = self.vectors.slot_capacity();
        self.keys.grow_to(new_capacity)?;
        self.payloads.grow_to(new_capacity)?;
        if let Some(q) = self.quant.read().as_ref() {
            // Quant file growth mirrors vectors capacity; zero-fill new range.
            let _ = q.grow_to(new_capacity);
        }

        let next = self.tombstones.load().resized(new_capacity as usize);
        self.tombstones.store(Arc::new(next));

        meta.slot_capacity = new_capacity;
        inner.filter_bitmap.resize(new_capacity as usize);
        inner.payload_indexes.resize(new_capacity as usize);
        Ok(())
    }

    /// Build or rebuild quantization from the current live set.
    ///
    /// Heavy work (collecting vectors and training codebooks) runs without
    /// holding the store write lock, under the same `build_mutex` that
    /// serializes index builds and compaction. The product codebook requires
    /// k-means; scalar/binary need only a range scan.
    pub fn build_quantization(&self) -> Result<bool> {
        let qc = {
            let inner = self.inner.read();
            match &inner.meta.quantization_config {
                Some(qc) if qc.enabled && qc.quant_type.is_some() => qc.clone(),
                _ => return Ok(false),
            }
        };
        // Exclusive with compaction and index builds (order: compact -> build)
        // so slot numbers stay stable and quant files are not rewritten
        // concurrently by compaction.
        let _compact_guard = self.compact_mutex.lock();
        let _guard = self.build_mutex.lock();
        let (dim, distance, segment_slots, next_slot) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.segment_slots,
                inner.meta.next_slot,
            )
        };
        // If store has no quant yet, create one now.
        {
            let mut qguard = self.quant.write();
            if qguard.is_none() {
                let capacity = self.inner.read().meta.slot_capacity;
                *qguard = Some(QuantStore::create(
                    &self.dir,
                    dim,
                    distance,
                    &qc,
                    segment_slots,
                    capacity,
                )?);
            }
        }
        let live: Vec<u32> = {
            let tomb = self.tombstones.load();
            (0..next_slot as u32)
                .filter(|&s| !tomb.bit(s as usize))
                .collect()
        };
        if live.is_empty() {
            return Ok(false);
        }
        let vsnap = self.vectors.snapshot();
        let quant = self
            .quant
            .read()
            .as_ref()
            .expect("quant present")
            .meta_snapshot();
        let _ = quant;
        // Delegate training to QuantStore.
        let quant_store = self.quant.read();
        let qs = quant_store.as_ref().expect("quant store exists");
        qs.rebuild(&vsnap, segment_slots, &live, dim)?;
        Ok(true)
    }

    /// Whether quantization is active and ready for search.
    pub fn has_quantization(&self) -> bool {
        self.quant.read().as_ref().is_some_and(|q| q.is_ready())
    }

    /// Quantization config in effect, if any.
    pub fn quantization_config(&self) -> Option<crate::types::QuantizationConfig> {
        self.inner.read().meta.quantization_config.clone()
    }

    /// Replace quantization config at runtime (engine setting). Persists to
    /// `meta.bin` and creates or drops the quant files accordingly.
    ///
    /// Crash-safe order: quant files are created/synced first, then `meta.bin`
    /// is swapped. If the process crashes before `meta.bin` is durable the
    /// collection reopens without quantization (fallback to exact); orphan
    /// quant files are discarded on the next open.
    pub fn set_quantization_config(&self, config: QuantizationConfig) -> Result<()> {
        config.validate(self.inner.read().meta.vector_size)?;
        // Create or drop quant files before persisting meta, so a crash
        // cannot leave meta claiming quantization while files are missing.
        let new_quant = if config.enabled && config.quant_type.is_some() {
            // Pre-create the store while not holding the inner write lock.
            let (dim, distance, segment_slots, slot_capacity) = {
                let inner = self.inner.read();
                (
                    inner.meta.vector_size,
                    inner.meta.distance,
                    inner.meta.segment_slots,
                    inner.meta.slot_capacity,
                )
            };
            let mut qguard = self.quant.write();
            if qguard.is_none() {
                *qguard = Some(QuantStore::create(
                    &self.dir,
                    dim,
                    distance,
                    &config,
                    segment_slots,
                    slot_capacity,
                )?);
            } else if qguard.as_ref().is_some_and(|q| q.config() != &config) {
                // Config type changed: recreate files.
                *qguard = None;
                let _ = std::fs::remove_file(self.dir.join("quant.bin"));
                let _ = std::fs::remove_file(self.dir.join("quant_meta.bin"));
                *qguard = Some(QuantStore::create(
                    &self.dir,
                    dim,
                    distance,
                    &config,
                    segment_slots,
                    slot_capacity,
                )?);
            }
            // Hold the quant guard briefly to ensure files are durable before
            // meta claims them; QuantStore::create already fsyncs.
            drop(qguard);
            Some(config.clone())
        } else {
            // Disable: drop in-memory store and remove files first.
            {
                let mut qguard = self.quant.write();
                *qguard = None;
            }
            let _ = std::fs::remove_file(self.dir.join("quant.bin"));
            let _ = std::fs::remove_file(self.dir.join("quant_meta.bin"));
            // Still persist the disabled config so callers can distinguish
            // "never configured" vs "explicitly disabled".
            Some(config.clone())
        };
        let mut inner = self.inner.write();
        inner.meta.quantization_config = new_quant;
        inner.meta.save(&self.dir)?;
        // Append a WAL checkpoint so replay knows to re-validate quant files.
        // No in-memory mutation is needed; replay for `Quantize` is a no-op
        // because quant files are derived and rebuildable.
        // Best-effort: WAL append failure after meta save is still recoverable
        // (open discards corrupt quant files and falls back to exact).
        let _ = self.wal.append(&WalTxn {
            txn_id: inner.meta.last_applied_txn,
            ops: vec![WalRecord::Quantize],
        });
        Ok(())
    }

    fn update_tombstone_bit(&self, slot: usize, value: bool) {
        let next = self.tombstones.load().with_slot(slot, value);
        self.tombstones.store(Arc::new(next));
    }
}

fn validate_point(meta: &Meta, point: &VectorPoint) -> Result<()> {
    if point.vector.len() != meta.vector_size {
        return Err(VectorSearchError::InvalidVectorDimension {
            expected: meta.vector_size,
            actual: point.vector.len(),
        });
    }
    for (i, v) in point.vector.iter().enumerate() {
        if !v.is_finite() {
            return Err(VectorSearchError::NonFiniteElement(i));
        }
    }
    Ok(())
}

fn threshold_met(meta: &Meta) -> bool {
    meta.next_slot > 0 && meta.tombstone_count as f64 / meta.next_slot as f64 > COMPACTION_THRESHOLD
}

/// Number of upserted and deleted points in a WAL operation batch.
fn count_ops(ops: &[WalRecord]) -> (u64, u64) {
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

impl std::fmt::Debug for CollectionStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        f.debug_struct("CollectionStore")
            .field("dir", &self.dir)
            .field("collection", &inner.meta.collection)
            .field("vector_size", &inner.meta.vector_size)
            .field("distance", &inner.meta.distance)
            .field("next_slot", &inner.meta.next_slot)
            .field("live_count", &inner.meta.live_count)
            .field("tombstone_count", &inner.meta.tombstone_count)
            .field("slot_capacity", &inner.meta.slot_capacity)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CollectionConfig, DistanceMetric, VectorPoint};

    fn config(dim: usize) -> CollectionConfig {
        CollectionConfig::new(dim, DistanceMetric::Cosine)
    }

    fn point(id: u64, dim: usize) -> VectorPoint {
        VectorPoint::new(
            id,
            (0..dim)
                .map(|i| ((id as usize * 31 + i) % 100) as f32 / 100.0)
                .collect(),
        )
    }

    fn point_with_payload(id: u64, dim: usize, payload: crate::types::Payload) -> VectorPoint {
        VectorPoint::new(id, (0..dim).map(|_| 0.5).collect()).with_payload(payload)
    }

    #[test]
    fn test_create_open_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("col_a");
        let store = CollectionStore::create(&store_dir, "col_a", &config(4)).unwrap();
        store.upsert(&point(1, 4)).unwrap();
        store.upsert(&point(2, 4)).unwrap();
        assert_eq!(store.count(), 2);
        drop(store);

        let reopened = CollectionStore::open(&store_dir).unwrap();
        assert_eq!(reopened.count(), 2);
        let p1 = reopened.get(&PointId::Num(1)).unwrap().unwrap();
        assert_eq!(p1.vector.len(), 4);
        assert_eq!(p1.vector, point(1, 4).vector);
        let p2 = reopened.get(&PointId::Num(2)).unwrap().unwrap();
        assert_eq!(p2.vector, point(2, 4).vector);
        assert!(reopened.get(&PointId::Num(99)).unwrap().is_none());
    }

    #[test]
    fn test_payload_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("col_p");
        let store = CollectionStore::create(&store_dir, "col_p", &config(2)).unwrap();
        let mut payload = HashMap::new();
        payload.insert("color".to_string(), serde_json::json!("red"));
        payload.insert("size".to_string(), serde_json::json!(42));
        store
            .upsert(&point_with_payload(1, 2, payload.clone()))
            .unwrap();

        let got = store.get(&PointId::Num(1)).unwrap().unwrap();
        assert_eq!(got.payload, Some(payload.clone()));

        drop(store);
        let reopened = CollectionStore::open(&store_dir).unwrap();
        let got = reopened.get(&PointId::Num(1)).unwrap().unwrap();
        assert_eq!(got.payload, Some(payload));
    }

    #[test]
    fn test_set_payload_fields_merges_and_reindexes_bitmap() {
        use crate::filter;
        use crate::types::{ConditionType, FilterCondition, VectorFilter};

        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("col_pf");
        let store = CollectionStore::create(&store_dir, "col_pf", &config(2)).unwrap();
        let mut payload = HashMap::new();
        payload.insert("color".to_string(), serde_json::json!("red"));
        store.upsert(&point_with_payload(1, 2, payload)).unwrap();

        // Partial merge overwrites the given keys and preserves the rest.
        let fields: Payload = [
            ("color", serde_json::json!("blue")),
            ("size", serde_json::json!(7)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        store.set_payload_fields(&PointId::Num(1), fields).unwrap();
        let got = store.get(&PointId::Num(1)).unwrap().unwrap();
        let got = got.payload.expect("payload present");
        assert_eq!(got.get("color"), Some(&serde_json::json!("blue")));
        assert_eq!(got.get("size"), Some(&serde_json::json!(7)));

        // The pre-filter bitmap must reflect the merged values so an
        // equality filter on the new value finds the slot and on the old
        // value no longer claims it.
        let make_filter = |value: &str| {
            VectorFilter::new().must(FilterCondition {
                field: "color".to_string(),
                condition: ConditionType::Match {
                    value: value.to_string(),
                },
            })
        };
        let inner = store.inner.read();
        let blue = inner
            .filter_bitmap
            .build_mask(&make_filter("blue"))
            .expect("merged value is indexed");
        assert!(!blue.not_any());
        match inner.filter_bitmap.build_mask(&make_filter("red")) {
            None => {}
            Some(m) => assert!(!m.any()),
        }
        // Full filter evaluation agrees with the stored payload.
        let id = PointId::Num(1);
        assert!(filter::matches(&make_filter("blue"), &id, Some(&got)).unwrap());
        assert!(!filter::matches(&make_filter("red"), &id, Some(&got)).unwrap());
    }

    #[test]
    fn test_set_payload_field_wal_replay_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("col_r");
        let store = CollectionStore::create(&store_dir, "col_r", &config(2)).unwrap();
        store.upsert(&point(1, 2)).unwrap();
        store
            .set_payload_field(
                &PointId::Num(1),
                "env".to_string(),
                serde_json::json!("prod"),
            )
            .unwrap();
        drop(store);

        let reopened = CollectionStore::open(&store_dir).unwrap();
        let payload = reopened
            .get(&PointId::Num(1))
            .unwrap()
            .unwrap()
            .payload
            .expect("payload recreated by partial update");
        assert_eq!(payload.get("env"), Some(&serde_json::json!("prod")));
    }

    #[test]
    fn test_numeric_index_range_accelerated_search() {
        use crate::types::{
            ConditionType, FilterCondition, PayloadSchemaType, RangeCondition, SearchQuery,
            VectorFilter,
        };

        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("col_num");
        let store = CollectionStore::create(&store_dir, "col_num", &config(2)).unwrap();
        for (id, price) in [(1u64, 10.0f64), (2, 50.0), (3, 120.0)] {
            let mut p = HashMap::new();
            p.insert("price".to_string(), serde_json::json!(price));
            store.upsert(&point_with_payload(id, 2, p)).unwrap();
        }
        assert!(store.list_payload_indexes().is_empty());
        store
            .create_payload_index("price", PayloadSchemaType::Float)
            .unwrap();
        assert_eq!(
            store.list_payload_indexes(),
            vec![("price".to_string(), PayloadSchemaType::Float)]
        );

        // Range filter via the numeric index mask; identical results must
        // hold with and without the index because the post-filter still
        // re-evaluates the full filter. All vectors score identically, so
        // ties break by slot ascending.
        let make_query = || {
            SearchQuery::new(vec![0.5, 0.5], 10).with_filter(VectorFilter::new().must(
                FilterCondition {
                    field: "price".to_string(),
                    condition: ConditionType::Range(RangeCondition::new().gt(20.0).lte(130.0)),
                },
            ))
        };
        let hits = store
            .search(&make_query())
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect::<Vec<_>>();
        assert_eq!(hits, vec![PointId::Num(2), PointId::Num(3)]);

        // Definitions survive reopen and contents rebuild from payloads.
        drop(store);
        let reopened = CollectionStore::open(&store_dir).unwrap();
        let hits = reopened
            .search(&make_query())
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect::<Vec<_>>();
        assert_eq!(hits, vec![PointId::Num(2), PointId::Num(3)]);
        assert!(reopened.delete_payload_index("price").unwrap());
        assert!(!reopened.delete_payload_index("price").unwrap());
        assert!(reopened.list_payload_indexes().is_empty());
    }

    #[test]
    fn test_search_payload_selector_projection() {
        use crate::types::{PayloadSelector, SearchQuery};

        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("col_sel");
        let store = CollectionStore::create(&store_dir, "col_sel", &config(2)).unwrap();
        let mut p = HashMap::new();
        p.insert("keep".to_string(), serde_json::json!("yes"));
        p.insert("drop".to_string(), serde_json::json!("gone"));
        store.upsert(&point_with_payload(1, 2, p.clone())).unwrap();

        let query = SearchQuery::new(vec![0.5, 0.5], 5)
            .with_payload_selector(PayloadSelector::include(vec!["keep".to_string()]));
        let results = store.search(&query).unwrap();
        let payload = results[0].payload.as_ref().expect("payload present");
        assert_eq!(payload.len(), 1);
        assert_eq!(payload.get("keep"), Some(&serde_json::json!("yes")));

        // Selector building helpers merge into one selector.
        let mut sel = PayloadSelector::all();
        sel.include = Some(vec!["a".to_string()]);
        assert_eq!(sel.apply(&p).len(), 0, "missing field yields empty map");
    }

    #[test]
    fn test_upsert_overwrite_reuses_slot() {
        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("col_o");
        let store = CollectionStore::create(&store_dir, "col_o", &config(4)).unwrap();
        store.upsert(&point(1, 4)).unwrap();
        store.upsert(&point(2, 4)).unwrap();

        let overwritten = VectorPoint::new(1u64, vec![9.0; 4]);
        store.upsert(&overwritten).unwrap();

        assert_eq!(store.count(), 2, "overwrite must not grow live count");
        let got = store.get(&PointId::Num(1)).unwrap().unwrap();
        assert_eq!(got.vector, vec![9.0; 4]);
        let got2 = store.get(&PointId::Num(2)).unwrap().unwrap();
        assert_eq!(got2.vector, point(2, 4).vector);
    }

    #[test]
    fn test_grow_across_segments() {
        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("col_g");
        let store = CollectionStore::create_with_segment_slots(&store_dir, "col_g", &config(4), 16)
            .unwrap();

        let total = 100u64;
        for i in 0..total {
            store.upsert(&point(i, 4)).unwrap();
        }
        assert_eq!(store.count(), total);
        let meta = store.meta();
        assert!(
            meta.slot_capacity >= total,
            "capacity grown to {}",
            meta.slot_capacity
        );
        assert_eq!(meta.slot_capacity % meta.segment_slots as u64, 0);

        for i in (0..total).step_by(7) {
            let got = store.get(&PointId::Num(i)).unwrap().unwrap();
            assert_eq!(got.vector, point(i, 4).vector);
        }

        drop(store);
        let reopened = CollectionStore::open(&store_dir).unwrap();
        assert_eq!(reopened.count(), total);
        for i in (0..total).step_by(11) {
            let got = reopened.get(&PointId::Num(i)).unwrap().unwrap();
            assert_eq!(got.vector, point(i, 4).vector);
        }
    }

    #[test]
    fn test_delete_tombstone_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("col_d");
        let store = CollectionStore::create(&store_dir, "col_d", &config(4)).unwrap();
        store.upsert(&point(1, 4)).unwrap();
        store.upsert(&point(2, 4)).unwrap();
        store.upsert(&point(3, 4)).unwrap();

        assert!(store.delete(&PointId::Num(2)).unwrap());
        assert!(
            !store.delete(&PointId::Num(2)).unwrap(),
            "double delete is a no-op"
        );
        assert!(store.get(&PointId::Num(2)).unwrap().is_none());
        assert_eq!(store.count(), 2);
        // The tombstone stays pending until compaction runs (the engine
        // schedules it in the background); visibility never depends on it.
        let meta = store.meta();
        assert_eq!(meta.tombstone_count, 1);
        store.compact().unwrap();
        let meta = store.meta();
        assert_eq!(meta.tombstone_count, 0);
        assert_eq!(meta.next_slot, 2);

        drop(store);
        let reopened = CollectionStore::open(&store_dir).unwrap();
        assert_eq!(reopened.count(), 2);
        assert!(reopened.get(&PointId::Num(2)).unwrap().is_none());
        assert_eq!(
            reopened.get(&PointId::Num(1)).unwrap().unwrap().vector,
            point(1, 4).vector
        );
        assert_eq!(
            reopened.get(&PointId::Num(3)).unwrap().unwrap().vector,
            point(3, 4).vector
        );
    }

    #[test]
    fn test_upsert_then_delete_then_reinsert() {
        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("col_r");
        let store = CollectionStore::create(&store_dir, "col_r", &config(4)).unwrap();
        store.upsert(&point(1, 4)).unwrap();
        assert!(store.delete(&PointId::Num(1)).unwrap());

        // Without an intervening compaction the re-insert takes a fresh slot:
        // the stale tombstone stays pending and flags the collection for
        // background compaction, while visibility is already correct.
        store.upsert(&point(1, 4)).unwrap();
        assert_eq!(store.count(), 1);
        let meta = store.meta();
        assert_eq!(meta.next_slot, 2);
        assert_eq!(meta.tombstone_count, 1);
        assert!(store.needs_compaction());
        store.compact().unwrap();
        assert!(!store.needs_compaction());
        assert!(store.get(&PointId::Num(1)).unwrap().is_some());
    }

    #[test]
    fn test_rejects_invalid_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let store = CollectionStore::create(dir.path().join("col_v"), "col_v", &config(4)).unwrap();
        let bad = VectorPoint::new(1u64, vec![1.0, 2.0]);
        let err = store.upsert(&bad).unwrap_err();
        assert!(matches!(
            err,
            VectorSearchError::InvalidVectorDimension {
                expected: 4,
                actual: 2
            }
        ));
    }

    #[test]
    fn test_rejects_non_finite_elements() {
        let dir = tempfile::tempdir().unwrap();
        let store = CollectionStore::create(dir.path().join("col_n"), "col_n", &config(2)).unwrap();
        let bad = VectorPoint::new(1u64, vec![f32::NAN, 1.0]);
        let err = store.upsert(&bad).unwrap_err();
        assert!(matches!(err, VectorSearchError::NonFiniteElement(0)));

        let bad_inf = VectorPoint::new(1u64, vec![1.0, f32::INFINITY]);
        let err = store.upsert(&bad_inf).unwrap_err();
        assert!(matches!(err, VectorSearchError::NonFiniteElement(1)));
    }

    #[test]
    fn test_create_validates_inputs() {
        let dir = tempfile::tempdir().unwrap();

        let err = CollectionStore::create(dir.path().join("bad"), "", &config(4)).unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidCollectionName(_)));

        let err = CollectionStore::create(dir.path().join("bad"), "a/b", &config(4)).unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidCollectionName(_)));

        let mut cfg = config(0);
        cfg.vector_size = 0;
        let err = CollectionStore::create(dir.path().join("bad"), "col", &cfg).unwrap_err();
        assert!(matches!(
            err,
            VectorSearchError::InvalidVectorDimension {
                expected: 1,
                actual: 0
            }
        ));

        let mut cfg = config(4);
        cfg.distance = DistanceMetric::Manhattan;
        let manhattan_dir = dir.path().join("manhattan_ok");
        let store = CollectionStore::create(&manhattan_dir, "col", &cfg).unwrap();
        assert_eq!(store.meta().distance, DistanceMetric::Manhattan);
    }

    #[test]
    fn test_create_existing_dir_conflicts() {
        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("col_x");
        let _store = CollectionStore::create(&store_dir, "col_x", &config(4)).unwrap();
        let err = CollectionStore::create(&store_dir, "col_x", &config(4)).unwrap_err();
        assert!(matches!(err, VectorSearchError::CollectionAlreadyExists(_)));
    }

    #[test]
    fn test_corrupt_vectors_length_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let store_dir = dir.path().join("col_c");
        let store = CollectionStore::create(&store_dir, "col_c", &config(4)).unwrap();
        store.upsert(&point(1, 4)).unwrap();
        drop(store);

        // Corrupt vectors.bin by truncating it.
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(store_dir.join("vectors.bin"))
            .unwrap();
        file.set_len(16).unwrap();
        drop(file);

        let err = CollectionStore::open(&store_dir).unwrap_err();
        assert!(matches!(err, VectorSearchError::CorruptData(_)));
    }
}
