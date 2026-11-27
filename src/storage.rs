//! Per-collection storage assembly.
//!
//! A collection root contains `generation.bin`, the shared `wal/`, and a
//! type marker. The published `generations/g<id>/` directory holds:
//! - `meta.bin`     postcard `Meta`
//! - `vectors.bin`  dense row-major f32, segmented mmap
//! - `keys.bin`     slot -> point id directory
//! - `tombstones.bin` slot -> deleted flag (survives WAL truncation)
//! - `payloads_store/` block-based payload storage (tracker + pages)
//!
//! The generation manifest is the commit point; compaction stages a whole
//! new generation and publishes it by replacing that manifest.
//!
//! Readers snapshot the mmap-backed `ArcSwap` views and scan without the
//! store lock. Mutations go through the commit queue
//! ([`CollectionStore::apply_txn`]): one short write-lock critical section
//! assigns the lsn and the slots, the queue worker flushes the whole backlog
//! with a single WAL fsync and then applies it in lsn order, and the caller
//! returns only once its transaction is applied. A crash at any point
//! recovers by idempotent replay in log order — which rebuilds exactly the
//! slot numbering the live store used, because reserve and replay walk the
//! log in the same order.
//!
//! This module hosts the store type itself plus its lifecycle and
//! mutation/read paths; orthogonal concerns live in sibling modules:
//! - [`queue`]       commit queue: group commit plus the single applier
//! - [`tombstones`]   lock-free tombstone table
//! - [`compaction`]   physical tombstone reclamation
//! - [`index_lifecycle`]  ANN index load/build/publish/drop
//! - [`search`]       exact scan and ANN search execution

mod compaction;
mod directory;
mod filter_bitmap;
pub(crate) mod generation;
mod index_lifecycle;
mod keys;
mod meta;
mod payload_index;
mod payload_key;
pub(crate) mod point_blobs;
pub(crate) mod postings;
pub(crate) mod payload_store;
pub(crate) mod quant;
mod queue;
mod search;
pub(crate) mod snapshot;
mod tombstones;
pub(crate) mod vectors;
mod wal;

mod store_maintenance;
mod store_open;
mod store_payload;
mod store_read;
mod store_write;

use std::cell::Cell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Weak};

use arc_swap::ArcSwap;
use parking_lot::{Mutex, RwLock};

pub(crate) use generation::Generation;
pub(crate) use meta::Meta;
pub(crate) use payload_index::PayloadIndexManager;
pub(crate) use payload_store::PayloadStore;
pub(crate) use tombstones::TombstoneBits;
pub use wal::{Wal, WalPoint, WalRecord, WalTxn, DEFAULT_SEGMENT_BYTES};

use self::filter_bitmap::FilterBitmap;
use self::index_lifecycle::PublishedIndex;
use self::keys::Keys;
use self::quant::QuantStore;
use self::queue::CommitQueue;
use self::vectors::Vectors;

use crate::error::Result;
use crate::metrics::Metrics;
use crate::types::{IvfConfig, Payload, PointId, StorageConfig};

// Storage tuning (WAL checkpoint size, compaction threshold, meta save
// interval, payload page geometry) lives in `types::StorageConfig` and is
// supplied by the engine when a collection is opened or created.

/// Point id -> slot map, split by id kind.
///
/// A `HashMap<PointId, u32>` pays for the widest variant: `PointId` is 32
/// bytes (the `String` variant plus the tag), so numeric ids — the common
/// case — carry a fat key and a discriminant that is constant for the whole
/// map. Splitting the two kinds keeps numeric ids at `u64` keys (roughly 20
/// bytes per entry including hashbrown's control bytes and load factor,
/// against 48-56 for the enum) and confines the owned `String` allocation to
/// ids that really are uuids.
#[derive(Debug, Default)]
pub(crate) struct IdMap {
    numeric: HashMap<u64, u32>,
    uuid: HashMap<String, u32>,
}

impl IdMap {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            numeric: HashMap::with_capacity(capacity),
            uuid: HashMap::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.numeric.len() + self.uuid.len()
    }

    /// Slot holding a point id that arrived as a string (WAL records and
    /// snapshots carry ids as text). Numeric ids are parsed, never allocated.
    pub(crate) fn get_str(&self, id: &str) -> Option<u32> {
        match id.parse::<u64>() {
            Ok(number) => self.numeric.get(&number).copied(),
            Err(_) => self.uuid.get(id).copied(),
        }
    }

    pub(crate) fn insert_str(&mut self, id: &str, slot: u32) -> Option<u32> {
        match id.parse::<u64>() {
            Ok(number) => self.numeric.insert(number, slot),
            Err(_) => self.uuid.insert(id.to_string(), slot),
        }
    }

    pub(crate) fn contains_str(&self, id: &str) -> bool {
        self.get_str(id).is_some()
    }

    /// Slot holding `id`.
    ///
    /// Takes the id by reference: the enum-keyed map had to hash the whole
    /// 32-byte key (padding included) on every lookup, while this hashes a
    /// `u64` for the numeric case and reuses the caller's string otherwise.
    pub(crate) fn get(&self, id: &PointId) -> Option<u32> {
        match id {
            PointId::Num(n) => self.numeric.get(n).copied(),
            PointId::Uuid(u) => self.uuid.get(u.as_str()).copied(),
        }
    }

    pub(crate) fn contains_key(&self, id: &PointId) -> bool {
        self.get(id).is_some()
    }

    /// Record `id -> slot`, returning the slot it replaced.
    pub(crate) fn insert(&mut self, id: &PointId, slot: u32) -> Option<u32> {
        match id {
            PointId::Num(n) => self.numeric.insert(*n, slot),
            PointId::Uuid(u) => self.uuid.insert(u.clone(), slot),
        }
    }

    pub(crate) fn remove(&mut self, id: &PointId) -> Option<u32> {
        match id {
            PointId::Num(n) => self.numeric.remove(n),
            PointId::Uuid(u) => self.uuid.remove(u.as_str()),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.numeric.clear();
        self.uuid.clear();
    }
}

/// In-memory mutable state, guarded by the store's `RwLock`.
struct StoreInner {
    meta: Meta,
    reverse: IdMap,
    /// Slots handed out by the commit queue's reserve step for transactions
    /// that are not applied yet. An id stays here from the moment its slot is
    /// reserved until the applier writes the directory entry, so a concurrent
    /// transaction upserting the same id reuses the slot instead of getting a
    /// second one. The slot numbers are exactly the ones replay assigns,
    /// because both walk the log in the same order.
    reserved: IdMap,
    /// Next log sequence number to hand out. Initialized above the water mark
    /// replayed at open, so a replayed record can never collide with a new
    /// one.
    next_lsn: u64,
    /// IVF configuration for `Ivf` collections. Mirrors the persisted
    /// `meta.index`; the engine may still override it at runtime via
    /// [`CollectionStore::set_ivf_config`].
    ivf_config: Option<IvfConfig>,
    /// Last measured drift ratio, exposed via `CollectionInfo`.
    last_drift_ratio: Option<f64>,
    /// Slot-indexed pre-filter bitmap for equality payload conditions.
    filter_bitmap: FilterBitmap,
    /// Declared per-field payload indexes (MapIndex / NumericIndex).
    payload_indexes: PayloadIndexManager,
}

thread_local! {
    /// True on the commit-queue worker thread: joining itself in `Drop` would
    /// deadlock, so shutdown skips the join there.
    static COMMIT_WORKER: Cell<bool> = const { Cell::new(false) };
    /// True while the worker applies a batch. `apply_txn` must never run on
    /// that thread: the worker is the only applier, so waiting for its own
    /// batch would deadlock.
    static APPLYING: Cell<bool> = const { Cell::new(false) };
}

fn set_applying(value: bool) {
    APPLYING.with(|applying| applying.set(value));
}

fn is_commit_worker() -> bool {
    COMMIT_WORKER.with(|worker| worker.get())
}

fn set_commit_worker(value: bool) {
    COMMIT_WORKER.with(|worker| worker.set(value));
}

/// Guard the invariant that the apply path never enqueues a transaction.
fn assert_not_applying() {
    debug_assert!(
        !APPLYING.with(|applying| applying.get()),
        "the commit-queue apply path must not enqueue transactions"
    );
}

/// Every file-backed handle of one published generation.
///
/// A reader loads the outer `ArcSwap` once and keeps that guard for the whole
/// pass, so a compaction swapping generations cannot pair a new key table with
/// an old vector mapping. Mutations hold the store write lock, which excludes
/// the swap, and therefore use `load_full()` safely.
pub(super) struct DataFiles {
    /// Published generation directory holding all of the files below.
    dir: PathBuf,
    vectors: Vectors,
    keys: Keys,
    tombstones: ArcSwap<TombstoneBits>,
    payload_store: ArcSwap<Option<PayloadStore>>,
    quant: RwLock<Option<QuantStore>>,
    index: ArcSwap<Option<PublishedIndex>>,
}

impl DataFiles {
    /// Open every serving file of one generation.
    pub(super) fn open(dir: &Path, meta: &Meta, _storage: &StorageConfig) -> Result<Self> {
        let dir = dir.to_path_buf();
        let vectors = Vectors::open(
            &dir.join("vectors.bin"),
            meta.vector_size,
            meta.segment_slots,
            meta.slot_capacity,
        )?;
        let keys = Keys::open(&dir.join("keys.bin"))?;
        let tombstones = TombstoneBits::load(
            &dir,
            meta.slot_capacity as usize,
            meta.tombstone_count,
        )?
        .unwrap_or_else(|| TombstoneBits::new(meta.slot_capacity as usize));
        let payload_store = PayloadStore::open(&dir.join("payloads_store"))?;
        let quant = if meta.quantization_config.is_some() {
            QuantStore::open(
                &dir,
                meta.vector_size,
                meta.distance,
                meta.segment_slots,
                meta.slot_capacity,
            )?
        } else {
            None
        };
        Ok(Self {
            dir,
            vectors,
            keys,
            tombstones: ArcSwap::from(Arc::new(tombstones)),
            payload_store: ArcSwap::from(Arc::new(Some(payload_store))),
            quant: RwLock::new(quant),
            index: ArcSwap::from(Arc::new(None)),
        })
    }

    pub(super) fn dir(&self) -> &Path {
        &self.dir
    }
}

/// A single opened collection.
pub struct CollectionStore {
    dir: PathBuf,
    /// Published generation. Its lsn bounds every WAL replay.
    generation: RwLock<Generation>,
    /// File-backed handles of the published generation, swapped as a unit.
    files: ArcSwap<DataFiles>,
    /// Engine-level storage tuning this collection was opened with. Sourced
    /// from [`StorageConfig`]; never persisted per collection.
    storage: StorageConfig,
    inner: RwLock<StoreInner>,
    wal: Wal,
    /// Commit queue: group commit for the WAL plus the single applier thread.
    /// Shared with that thread, which observes the store through a weak
    /// reference so a dropped collection is never kept alive by it.
    queue: Arc<CommitQueue>,
    /// Weak self-reference, so the commit-queue worker can observe the store
    /// without keeping it alive past the last user.
    self_weak: Weak<CollectionStore>,
    /// Slots inserted while no index was published and a build was in
    /// flight; drained into the index on publish so probe search never
    /// misses them. A mutex-guarded queue: `register_slot` runs under the
    /// store write lock per point, and cloning an `ArcSwap` snapshot on every
    /// push made bulk inserts quadratic in the pending backlog.
    pending: Mutex<Vec<u32>>,
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
    /// Load the published file set as one guard. Read passes must keep the
    /// guard alive until every derived view has been captured.
    fn data(&self) -> arc_swap::Guard<Arc<DataFiles>> {
        self.files.load()
    }

    /// Owned handle for mutations, which run under the store write lock and
    /// therefore cannot race a generation swap.
    fn data_arc(&self) -> Arc<DataFiles> {
        self.files.load_full()
    }

    /// Directory holding the serving generation's files.
    /// Directory holding the currently published generation's data files.
    pub fn data_dir(&self) -> PathBuf {
        self.data_arc().dir.clone()
    }

    // ── Payload routing helpers ────────────────────────────────────────

    // ── Payload field indexes ──────────────────────────────────────────
}

impl Drop for CollectionStore {
    fn drop(&mut self) {
        self.queue.shutdown();
    }
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
mod tests;
