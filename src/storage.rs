//! Per-collection storage assembly.
//!
//! A collection directory contains:
//! - `meta.bin`     postcard `Meta`
//! - `vectors.bin`  dense row-major f32, segmented mmap
//! - `keys.bin`     slot -> point id directory
//! - `payloads.bin` slot -> payload blob directory + tombstone flags
//! - `wal.bin`      append-only transaction log (crash recovery)
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
mod index_lifecycle;
mod keys;
mod meta;
mod payload_index;
mod payload_key;
mod payload_store;
mod payloads;
pub(crate) mod quant;
mod queue;
mod search;
mod tombstones;
pub(crate) mod vectors;
mod wal;

mod store_maintenance;
mod store_open;
mod store_payload;
mod store_read;
mod store_write;

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Weak};

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
use self::queue::{CommitQueue, APPLY_WAIT_TIMEOUT};
use self::vectors::Vectors;

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::types::{
    CollectionConfig, DistanceMetric, IndexConfig, IvfConfig, Payload, PointId, QuantizationConfig,
    VectorPoint,
};

/// Tombstone ratio above which deletes trigger compaction.
const COMPACTION_THRESHOLD: f64 = 0.20;

/// WAL size at which a background checkpoint runs: flush every derived file,
/// persist the payload store and truncate the log. Tombstone-ratio compaction
/// never fires on pure-insert workloads, so without this bound the log would
/// grow (and replay time with it) without limit.
const WAL_CHECKPOINT_BYTES: u64 = 64 * 1024 * 1024;

/// How many WAL transactions between meta.bin saves. The WAL is always
/// fsynced on every transaction, so crash safety is preserved; meta.bin is an
/// optimization that shortens replay, not a durability boundary. Slot water
/// marks and counts are replayed from the log, so this interval — not the
/// mutation rate — is what paces meta.bin writes.
const META_SAVE_INTERVAL: u64 = 64;

/// In-memory mutable state, guarded by the store's `RwLock`.
struct StoreInner {
    meta: Meta,
    reverse: HashMap<PointId, u32>,
    /// Slots handed out by the commit queue's reserve step for transactions
    /// that are not applied yet. An id stays here from the moment its slot is
    /// reserved until the applier writes the directory entry, so a concurrent
    /// transaction upserting the same id reuses the slot instead of getting a
    /// second one. The slot numbers are exactly the ones replay assigns,
    /// because both walk the log in the same order.
    reserved: HashMap<PointId, u32>,
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
    /// Commit queue: group commit for the WAL plus the single applier thread.
    /// Shared with that thread, which observes the store through a weak
    /// reference so a dropped collection is never kept alive by it.
    queue: Arc<CommitQueue>,
    /// Weak self-reference, so the commit-queue worker can observe the store
    /// without keeping it alive past the last user.
    self_weak: Weak<CollectionStore>,
    /// Published ANN index; `None` = exact scan. Swapped atomically.
    index: ArcSwap<Option<PublishedIndex>>,
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
