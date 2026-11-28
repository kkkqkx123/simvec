//! Per-collection storage assembly.
//!
//! A collection root contains `generation.bin`, the shared `wal/`, and a
//! type marker. The published `generations/g<id>/` directory holds `meta.bin`
//! plus one physical segment directory per slot range:
//! - `segments/<start>/vectors.bin`   dense row-major f32 for this segment
//! - `segments/<start>/keys.bin`      local offset -> point id directory
//! - `segments/<start>/tombstones.bin` deleted flags (survive WAL truncation)
//! - `segments/<start>/id_tracker.bin` authoritative id -> offset directory + versions
//! - `segments/<start>/payloads_store/` block-based payload storage (tracker + pages)
//! - `segments.json`                  segment boundaries, counts and persisted lsns
//!
//! The id directory lives in the per-segment trackers; there is no
//! whole-generation snapshot file. The generation manifest is the commit
//! point; compaction stages a whole
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
pub(crate) mod fault;
mod filter_bitmap;
pub(crate) mod generation;
mod index_lifecycle;
mod keys;
mod meta;
mod payload_index;
mod payload_key;
pub(crate) mod payload_store;
pub(crate) mod point_blobs;
pub(crate) mod postings;
pub(crate) mod quant;
mod queue;
mod search;
mod segment_hnsw;
mod segment_ivf;
pub(crate) mod segment_payloads;
pub(crate) mod segments;
pub(crate) mod segmented_blobs;
pub(crate) mod snapshot;
pub(crate) mod spaces_meta;
pub(crate) mod space_ann;
mod tombstones;
pub(crate) mod vectors;
mod wal;
mod wal_ack;
mod wal_blobs;

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

pub(crate) use generation::{CollectionLock, Generation};
pub(crate) use meta::Meta;
pub(crate) use payload_index::PayloadIndexManager;
pub(crate) use segment_payloads::SegmentPayloadStore;
pub use snapshot::{verify_snapshot, SnapshotInfo};
pub(crate) use tombstones::TombstoneBits;
pub use wal::{Wal, WalPoint, WalRecord, WalTxn, DEFAULT_SEGMENT_BYTES};
pub(crate) use wal_ack::{WalAckPinGuard, WalAckPins};

use self::filter_bitmap::FilterBitmap;
use self::keys::Keys;
use self::quant::QuantStore;
use self::queue::CommitQueue;
use self::vectors::Vectors;

use crate::error::{Result, VectorSearchError};
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
pub(crate) struct IdMap<V = u32> {
    numeric: HashMap<u64, V>,
    uuid: HashMap<String, V>,
}

impl<V: Copy> IdMap<V> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            numeric: HashMap::with_capacity(capacity),
            uuid: HashMap::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.numeric.len() + self.uuid.len()
    }

    pub(crate) fn get_value(&self, id: &PointId) -> Option<V> {
        match id {
            PointId::Num(n) => self.numeric.get(n).copied(),
            PointId::Uuid(u) => self.uuid.get(u.as_str()).copied(),
        }
    }

    pub(crate) fn insert_value(&mut self, id: &PointId, value: V) -> Option<V> {
        match id {
            PointId::Num(n) => self.numeric.insert(*n, value),
            PointId::Uuid(u) => self.uuid.insert(u.clone(), value),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.numeric.clear();
        self.uuid.clear();
    }

    /// Whether `id` is present, regardless of its value.
    pub(crate) fn contains_id(&self, id: &PointId) -> bool {
        match id {
            PointId::Num(n) => self.numeric.contains_key(n),
            PointId::Uuid(u) => self.uuid.contains_key(u.as_str()),
        }
    }

    /// Remove `id`, returning its value if present.
    pub(crate) fn remove_value(&mut self, id: &PointId) -> Option<V> {
        match id {
            PointId::Num(n) => self.numeric.remove(n),
            PointId::Uuid(u) => self.uuid.remove(u.as_str()),
        }
    }

    /// All entries as typed ids and values.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (PointId, V)> + '_ {
        self.numeric
            .iter()
            .map(|(&number, &value)| (PointId::Num(number), value))
            .chain(
                self.uuid
                    .iter()
                    .map(|(id, &value)| (PointId::Uuid(id.clone()), value)),
            )
    }
}

impl IdMap<u32> {
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
}

/// In-memory mutable state, guarded by the store's `RwLock`.
struct StoreInner {
    meta: Meta,
    reverse: IdMap,
    /// Ids reserved by the commit queue's reserve step for transactions that
    /// are not applied yet. Presence only — slot numbers are assigned at
    /// apply time in log order, so live and replay derive identical numbers
    /// by construction. An id stays here from reserve until the applier
    /// publishes it, letting a concurrent transaction upserting the same id
    /// share the forthcoming slot instead of counting a second one.
    reserved: IdMap<()>,
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
    /// Last applied lsn for every known id, including deleted ids. A
    /// replayed record older than its id's water mark is dropped instead of
    /// overwriting newer state with a resurrected WAL segment.
    versions: IdMap<u64>,
    /// Slot-indexed pre-filter bitmap for equality payload conditions.
    filter_bitmap: FilterBitmap,
    /// Declared per-field payload indexes (MapIndex / NumericIndex).
    payload_indexes: PayloadIndexManager,
    /// Highest applied LSN per dense segment index. Advanced on every
    /// applied batch for the segments its slots touch; drives the flush
    /// water mark without a global stop-the-world pass.
    segment_applied: Vec<u64>,
    /// Highest durable LSN per dense segment index. Advanced by checkpoint
    /// once the segment files covering it are synced; the WAL ack is the
    /// minimum over all segments, clamped by live pins.
    segment_persisted: Vec<u64>,
    /// Scan hint for reclaiming dead slots of the appendable tail. Not
    /// persisted: it is derived state, rebuilt identically by WAL replay, and
    /// reset at every durable boundary (open, checkpoint, compaction) so live
    /// and replay never diverge on a slot number.
    free_cursor: u32,
    /// True while [`CollectionStore::replay_wal`] runs. Replay rebuilds the
    /// state the log already produced, so an upsert of a known id writes back
    /// into its own slot instead of appending a new one.
    replaying: bool,
}

impl StoreInner {
    /// Ensure per-segment water marks cover `segment_count` segments.
    pub(super) fn ensure_segment_len(&mut self, segment_count: usize) {
        if self.segment_applied.len() < segment_count {
            self.segment_applied.resize(segment_count, 0);
        }
        if self.segment_persisted.len() < segment_count {
            self.segment_persisted.resize(segment_count, 0);
        }
    }

    /// Minimum durable LSN over all tracked segments, or the generation
    /// water mark when no segment has been tracked yet.
    pub(super) fn min_persisted(&self, fallback: u64) -> u64 {
        if self.segment_persisted.is_empty() {
            return fallback;
        }
        self.segment_persisted
            .iter()
            .copied()
            .min()
            .unwrap_or(fallback)
    }
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
    payload_store: ArcSwap<Option<SegmentPayloadStore>>,
    quant: RwLock<Option<QuantStore>>,
    /// One HNSW graph per completed dense segment.
    hnsw_segments: ArcSwap<Vec<segment_hnsw::SegmentHnsw>>,
    /// One IVF index per completed dense segment.
    ivf_segments: ArcSwap<Vec<segment_ivf::SegmentIvf>>,
    /// Frozen/appendable boundary of this generation's segment set.
    segments: ArcSwap<segments::SegmentSet>,
}

impl DataFiles {
    /// Open every serving file of one generation.
    ///
    /// `generation` must be the manifest id the caller loaded: the segment
    /// accounting in `segments.json` is bound to it, so a manifest left behind
    /// by another generation is discarded instead of trusted.
    pub(super) fn open(
        dir: &Path,
        meta: &Meta,
        storage: &StorageConfig,
        replay_floor: u64,
        generation: u64,
    ) -> Result<Self> {
        let dir = dir.to_path_buf();
        let vectors = Vectors::open(
            &dir,
            meta.vector_size,
            meta.segment_slots,
            meta.slot_capacity,
        )?;
        let keys = Keys::open(&dir, meta.segment_slots, meta.slot_capacity)?;
        let segments =
            segments::read_or_derive(&dir, generation, meta.segment_slots, meta.next_slot)?;
        let tombstones = TombstoneBits::load_segments(
            &dir,
            meta.segment_slots,
            meta.slot_capacity as usize,
            meta.tombstone_count,
        )?
        .unwrap_or_else(|| TombstoneBits::new(meta.slot_capacity as usize));
        let payload_store = SegmentPayloadStore::open_with_repair(
            &dir,
            meta.segment_slots,
            meta.slot_capacity,
            replay_floor,
            storage.payload_store_config(),
        )?;
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
            hnsw_segments: ArcSwap::from(Arc::new(Vec::new())),
            ivf_segments: ArcSwap::from(Arc::new(Vec::new())),
            segments: ArcSwap::from(Arc::new(segments)),
        })
    }

    pub(super) fn dir(&self) -> &Path {
        &self.dir
    }
}

/// A single opened collection.
pub struct CollectionStore {
    dir: PathBuf,
    /// Cross-process single-writer lock on `dir`, shared by in-process handles.
    _dir_lock: CollectionLock,
    /// Set when a generation manifest was published but the in-memory
    /// switch did not finish. The collection stays readable but refuses
    /// further mutations until it is reopened from the published manifest.
    generation_poisoned: AtomicBool,
    /// Published generation. Its lsn bounds every WAL replay.
    generation: RwLock<Generation>,
    /// File-backed handles of the published generation, swapped as a unit.
    files: ArcSwap<DataFiles>,
    /// Engine-level storage tuning this collection was opened with. Sourced
    /// from [`StorageConfig`]; never persisted per collection.
    storage: StorageConfig,
    inner: RwLock<StoreInner>,
    wal: Wal,
    /// Live WAL acknowledge pins. Snapshots and copies pin the log before
    /// capturing files so a concurrent checkpoint cannot unlink the segments
    /// they still need. Truncation always clamps to the lowest pin.
    wal_pins: WalAckPins,
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
    /// Tombstones added since the last whole-store HNSW repair.
    hnsw_tombstones_pending: AtomicU64,
    /// Segment flushes since open, counted per segment.
    #[cfg(test)]
    segment_flushes: AtomicU64,
    /// Last failed maintenance flush. Reads and writes continue; the WAL
    /// watermark stays put until a later flush succeeds.
    maintenance_error: Mutex<Option<String>>,
    /// Transactions since the last meta.bin save. Used to amortize fsync
    /// cost; the WAL guarantees crash safety independently.
    txns_since_last_save: AtomicU64,
    /// Microseconds the last checkpoint held the queue gate. The gated
    /// section is publication only: tracker renames plus small metadata
    /// writes (`meta.bin`, indexes, generation lsn, the log marker). Those
    /// metadata writes still fsync, so this is bounded by the in-flight
    /// drain and a handful of small fsyncs — milliseconds, not microseconds
    /// — and `truncate_through` runs after the gate reopens. A jump into the
    /// seconds range means a data-segment flush was moved back inside the
    /// gate, which is the regression this counter exists to catch.
    last_gate_wait_us: AtomicU64,
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

    /// Directory holding the currently published generation's data files.
    pub fn data_dir(&self) -> PathBuf {
        self.data_arc().dir.clone()
    }

    /// Copy the serving generation and every surviving WAL segment into
    /// `dest`, with a checksummed manifest.
    ///
    /// The maintenance lock and the queue gate are held for the whole copy: no
    /// checkpoint can truncate the log, no compaction can swap the generation
    /// and no config write can replace `meta.bin` while files are being read.
    /// The store read lock is not taken, because the copy reads the file tree
    /// only. A WAL acknowledge pin covers the same window, so no segment being
    /// copied can be unlinked. Because the copied WAL is complete, the
    /// manifest's replay floor pins a self-consistent restore point.
    pub fn snapshot_to(&self, dest: impl AsRef<Path>) -> Result<SnapshotInfo> {
        self.snapshot_capture(None, dest)
    }

    /// Take a snapshot that reuses unchanged files from a verified base
    /// snapshot. Reuse is a hard link: the destination remains a complete,
    /// self-contained directory tree, but only changed files are copied.
    /// Every candidate is checked by generation, length and CRC before it is
    /// linked, so a rewritten file can never be mistaken for an old one.
    pub fn snapshot_to_incremental(
        &self,
        base: impl AsRef<Path>,
        dest: impl AsRef<Path>,
    ) -> Result<SnapshotInfo> {
        let base = base.as_ref();
        let base_info = snapshot::verify_snapshot(base)?;
        self.snapshot_capture(Some((base, base_info)), dest)
    }

    fn snapshot_capture(
        &self,
        base: Option<(&Path, SnapshotInfo)>,
        dest: impl AsRef<Path>,
    ) -> Result<SnapshotInfo> {
        self.snapshot_capture_inner(base, dest.as_ref(), |_| {})
    }

    fn snapshot_capture_inner(
        &self,
        base: Option<(&Path, SnapshotInfo)>,
        dest: &Path,
        on_copy: impl FnOnce(&Self),
    ) -> Result<SnapshotInfo> {
        if dest.exists() {
            return Err(VectorSearchError::InvalidConfig(format!(
                "snapshot destination {} already exists",
                dest.display()
            )));
        }
        self.ensure_not_poisoned()?;
        let _maintenance = self.compact_mutex.lock();
        let _gate = self.queue.quiesce();
        // The copy reads the file tree only. Compaction, checkpoint and index
        // builds all take `compact_mutex` first and the gate has stopped the
        // applier, so the tree stays stable without the store read lock.
        let generation = self.generation.read().clone();
        let pin = self.pin_wal_ack(generation.lsn());
        pin.set(generation.lsn());
        debug_assert_eq!(pin.version(), generation.lsn());
        let _pin = pin;

        std::fs::create_dir_all(dest)?;
        let mut files = Vec::new();
        let mut reused_files = 0;
        let mut copy = |src: &Path, rel: &str| -> Result<()> {
            copy_snapshot_file(
                src,
                dest,
                rel,
                generation.id(),
                base.as_ref(),
                &mut files,
                &mut reused_files,
                None,
            )
        };
        copy(&self.dir.join("generation.bin"), "generation.bin")?;
        on_copy(self);
        let generation_rel = format!("{}/g{:020}", generation::GENERATIONS_DIR, generation.id());
        copy(&generation.path(), &generation_rel)?;
        drop(copy);
        // WAL segments are bound to their first lsn, not to the current
        // generation id: recording the name as `version` lets an incremental
        // snapshot reuse an unchanged segment across generation advances.
        let wal_dir = self.dir.join("wal");
        let wal_copy = WalCopy {
            root: &wal_dir,
            tail: self.wal.tail_valid_len(),
        };
        copy_snapshot_file(
            &wal_dir,
            dest,
            "wal",
            generation.id(),
            base.as_ref(),
            &mut files,
            &mut reused_files,
            Some(&wal_copy),
        )?;

        let info = SnapshotInfo {
            version: snapshot::SNAPSHOT_FORMAT_VERSION,
            generation: generation.id(),
            lsn: generation.lsn(),
            reused_files,
            files,
        };
        snapshot::write_info(&dest.join(snapshot::SNAPSHOT_MANIFEST_FILE), &info)?;
        generation::sync_tree(dest)?;
        Ok(info)
    }

    /// Test-only: `snapshot_to` with `probe` invoked inside the copy window.
    #[cfg(test)]
    pub(crate) fn snapshot_with_probe(
        &self,
        dest: impl AsRef<Path>,
        probe: impl FnOnce(&Self),
    ) -> Result<SnapshotInfo> {
        self.snapshot_capture_inner(None, dest.as_ref(), probe)
    }

    /// Whether a generation publish left the in-memory switch unfinished.
    /// Such a collection stays readable but must be reopened before more
    /// writes are accepted.
    pub fn is_poisoned(&self) -> bool {
        self.generation_poisoned.load(AtomicOrdering::Acquire)
    }

    fn ensure_not_poisoned(&self) -> Result<()> {
        if self.is_poisoned() {
            return Err(VectorSearchError::Internal(format!(
                "collection {} must be reopened after an interrupted generation publish",
                self.dir.display()
            )));
        }
        Ok(())
    }

    /// Last maintenance failure, used by the public maintenance status.
    pub fn maintenance_error(&self) -> Option<String> {
        self.maintenance_error.lock().clone()
    }

    pub(crate) fn record_maintenance_error(&self, error: &str) {
        *self.maintenance_error.lock() = Some(error.to_string());
        tracing::warn!(error, collection_dir = %self.dir.display(), "segment flush failed");
    }

    pub(crate) fn clear_maintenance_error(&self) {
        *self.maintenance_error.lock() = None;
    }

    /// Durable segment water marks and the last maintenance failure.
    pub fn maintenance_status(&self) -> crate::types::MaintenanceStatus {
        let inner = self.inner.read();
        let error = self.maintenance_error();
        let segment_slots = inner.meta.segment_slots;
        let segments = if segment_slots == 0 {
            Vec::new()
        } else {
            (0..inner.segment_persisted.len())
                .map(|index| crate::types::SegmentDurability {
                    start: index as u32 * segment_slots,
                    applied_lsn: inner
                        .segment_applied
                        .get(index)
                        .copied()
                        .unwrap_or(inner.meta.last_applied_txn),
                    persisted_lsn: inner.segment_persisted[index],
                })
                .collect()
        };
        crate::types::MaintenanceStatus {
            healthy: error.is_none(),
            persisted_lsn: inner.min_persisted(inner.meta.last_applied_txn),
            wal_blob_bytes: self.wal.blob_bytes(),
            gate_wait_ms: self.last_gate_wait_us.load(AtomicOrdering::Relaxed) / 1000,
            error,
            segments,
        }
    }

    /// Pin the WAL acknowledge at `version` for the lifetime of the guard.
    /// Callers capture files first under the pin, then release it.
    pub(crate) fn pin_wal_ack(&self, version: u64) -> WalAckPinGuard {
        self.wal_pins.pin(version)
    }

    /// Test-only: lowest live pin.
    #[cfg(test)]
    pub(crate) fn lowest_wal_pin(&self) -> Option<u64> {
        self.wal_pins.lowest()
    }

    /// Truncate the WAL through `watermark`, clamped by live pins.
    ///
    /// A live pin at or below the watermark holds the truncate back to just
    /// below that pin, so a snapshot or copy window keeps the segments it
    /// still reads. A clamped value of zero truncates nothing.
    pub(crate) fn truncate_wal_through(&self, watermark: u64) -> Result<()> {
        let effective = self.wal_pins.clamp(watermark);
        if effective == 0 && watermark != 0 {
            return Ok(());
        }
        self.wal.truncate_through(effective)
    }

    /// Snapshot-facing view of the flush water mark: the minimum durable
    /// LSN over all segments, clamped by live pins.
    pub fn durable_wal_floor(&self) -> u64 {
        let inner = self.inner.read();
        let floor = inner.min_persisted(inner.meta.last_applied_txn);
        self.wal_pins.clamp(floor)
    }

    /// Per-segment live/tombstone accounting for the current generation.
    pub(crate) fn segment_entries(&self) -> Vec<segments::SegmentEntry> {
        let inner = self.inner.read();
        let tombstones = self.data_arc().tombstones.load();
        let fallback = inner.meta.last_applied_txn;
        let next_slot = inner.meta.next_slot;
        let segment_slots = inner.meta.segment_slots;
        segments::derive_entries(
            next_slot,
            segment_slots,
            |s| tombstones.bit(s),
            |index| {
                (
                    inner
                        .segment_applied
                        .get(index)
                        .copied()
                        .unwrap_or(fallback),
                    inner
                        .segment_persisted
                        .get(index)
                        .copied()
                        .unwrap_or(fallback),
                )
            },
        )
    }

    /// Starts of frozen segments whose garbage ratio meets `threshold`.
    pub(crate) fn segments_needing_compaction(&self, threshold: f64) -> Vec<u32> {
        segments::select_for_compaction(&self.segment_entries(), threshold)
    }
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

/// Restore a verified collection snapshot into a new directory.
///
/// The destination must not exist: restoring into a live collection would
/// mix two generations and two WALs under one lock file. Opening the restored
/// collection before returning exercises the same replay path a real restart
/// uses, so a snapshot with a mismatched manifest fails here rather than on
/// first use.
pub fn restore_snapshot(src: impl AsRef<Path>, dest: impl AsRef<Path>) -> Result<()> {
    let src = src.as_ref();
    let dest = dest.as_ref();
    let info = snapshot::verify_snapshot(src)?;
    if dest.exists() {
        return Err(VectorSearchError::InvalidConfig(format!(
            "restore destination {} already exists",
            dest.display()
        )));
    }
    std::fs::create_dir_all(dest)?;
    for file in &info.files {
        let target = snapshot::checked_path(dest, &file.path)?;
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        copy_snapshot_bytes(&snapshot::checked_path(src, &file.path)?, &target)?;
    }
    generation::sync_tree(dest)?;
    let restored = CollectionStore::open(dest)?;
    drop(restored);
    Ok(())
}

/// How the WAL subtree is copied into a snapshot.
struct WalCopy<'a> {
    /// Directory being walked. Only its direct children are treated as
    /// segments; everything below stays generation-bound.
    root: &'a Path,
    /// Tail segment name and its valid byte length.
    tail: Option<(u64, u64)>,
}

/// Preallocation scratch inside `wal/`: a full-capacity hole consumed by the
/// next rollover and rebuilt on demand. `.spare.tmp` is its half-written
/// precursor.
fn is_wal_scratch(name: &str) -> bool {
    name == ".spare" || name == ".spare.tmp"
}

fn copy_snapshot_file(
    src: &Path,
    dest_root: &Path,
    rel: &str,
    version: u64,
    base: Option<&(&Path, SnapshotInfo)>,
    files: &mut Vec<snapshot::SnapshotFile>,
    reused: &mut usize,
    wal: Option<&WalCopy<'_>>,
) -> Result<()> {
    if src.is_dir() {
        std::fs::create_dir_all(dest_root.join(rel))?;
        let skip_scratch = wal.is_some_and(|wal| src == wal.root);
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if skip_scratch && is_wal_scratch(&name) {
                continue;
            }
            let child = if rel.is_empty() {
                name
            } else {
                format!("{rel}/{name}")
            };
            copy_snapshot_file(
                &entry.path(),
                dest_root,
                &child,
                version,
                base,
                files,
                reused,
                wal,
            )?;
        }
        return Ok(());
    }
    // A WAL segment directly under `wal/` is named by its first lsn. Bind its
    // manifest version to that lsn instead of the generation id so unchanged
    // segments stay reusable across generation advances. Other wal files
    // (`acked-lsn`, `blobs.bin`) stay bound to the generation.
    let segment = wal
        .filter(|wal| src.parent() == Some(wal.root))
        .and_then(|_| {
            src.file_name()?
                .to_str()?
                .strip_suffix(".bin")?
                .parse::<u64>()
                .ok()
        });
    let version = segment.unwrap_or(version);
    // The tail segment is copied to its valid prefix: the rest is
    // preallocated slack, which replay treats as the end of the log.
    // `CollectionStore::open` extends a short tail back to capacity.
    let limit = segment.and_then(|segment| {
        wal.and_then(|wal| wal.tail)
            .filter(|(tail, _)| *tail == segment)
            .map(|(_, valid)| valid)
    });
    let target = dest_root.join(rel);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let (len, crc32) = snapshot::file_checksum_prefix(src, limit)?;
    let reusable = base.and_then(|(base_dir, info)| {
        info.files.iter().find(|file| {
            file.path == rel && file.version == version && file.len == len && file.crc32 == crc32
        })?;
        snapshot::checked_path(base_dir, rel).ok()
    });
    let mut linked = false;
    if let Some(from) = reusable {
        if from.is_file() && std::fs::hard_link(&from, &target).is_ok() {
            linked = true;
        }
    }
    if !linked {
        match limit {
            Some(limit) => copy_prefix(src, &target, limit)?,
            None => copy_snapshot_bytes(src, &target)?,
        }
    } else {
        *reused += 1;
    }
    files.push(snapshot::SnapshotFile {
        path: rel.to_string(),
        version,
        len,
        crc32,
    });
    Ok(())
}

/// Copy the first `limit` bytes of `src` into a new file at `target`.
///
/// Used for WAL segments, whose trailing preallocated slack is not part of the
/// log. The destination is a fresh file, so it ends exactly at `limit`.
fn copy_prefix(src: &Path, target: &Path, limit: u64) -> Result<()> {
    use std::io::{Read, Write};

    let mut input = std::fs::File::open(src)?;
    let mut output = std::fs::File::create(target)?;
    let mut buffer = [0u8; 64 * 1024];
    let mut copied = 0u64;
    while copied < limit {
        let want = (limit - copied).min(buffer.len() as u64) as usize;
        let read = input.read(&mut buffer[..want])?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        copied += read as u64;
    }
    Ok(())
}

/// Copy a whole file into a fresh destination, preferring a copy-on-write
/// clone: a reflink shares the source's extents instead of reading them, and
/// `std::fs::copy` covers filesystems that cannot clone.
fn copy_snapshot_bytes(src: &Path, target: &Path) -> Result<()> {
    #[cfg(target_os = "linux")]
    if clone_file(src, target).is_ok() {
        return Ok(());
    }
    std::fs::copy(src, target)?;
    Ok(())
}

/// Ask the filesystem to clone `src` into `target` (`FICLONE`).
#[cfg(target_os = "linux")]
fn clone_file(src: &Path, target: &Path) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;

    // `FICLONE` from <linux/fs.h>: `_IOW(0x94, 9, int)`.
    const FICLONE: libc::c_ulong = 0x4004_9409;
    let source = std::fs::File::open(src)?;
    let dest = std::fs::File::create(target)?;
    let rc = unsafe { libc::ioctl(dest.as_raw_fd(), FICLONE, source.as_raw_fd()) };
    if rc == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    drop(dest);
    let _ = std::fs::remove_file(target);
    Err(err)
}

#[cfg(test)]
mod tests;
