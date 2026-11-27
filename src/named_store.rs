//! `NamedStore` — named-vectors collections.
//!
//! A named collection declares a map of vector spaces (dense or sparse, each
//! with its own dimension and distance metric). Points carry one vector per
//! declared space and every query targets exactly one space by name.
//!
//! The layout mirrors [`crate::kind_store::KindStore`]: the slot table and the
//! id directory stay in memory, point vectors live in an append-only mmap blob
//! (`points.bin`, one entry per declared space) and sparse postings live in
//! per-space bucket files (`postings/<space index>/`), and JSON payloads in
//! `payloads.bin`. Durability is the WAL plus a snapshot that references blobs
//! instead of inlining them. No ANN indexes, no quantization.

#[cfg(test)]
#[path = "named_store_tests.rs"]
mod tests;

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::{Mutex, RwLock};

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::storage::point_blobs::BlobRef;
use crate::storage::postings::Postings;
use crate::storage::segmented_blobs::SegmentedBlobs;
use crate::storage::space_ann::SpaceAnn;
use crate::storage::{CollectionLock, Generation, IdMap, Wal};
use crate::types::{Payload, PointId, StorageConfig, VectorKind};

mod lifecycle;
mod read;
mod score;
mod state;
mod write;

pub(crate) use self::score::score_pair;
pub(crate) use self::score::sparse_scores;
pub use self::write::named_upsert_record;
pub(crate) use crate::kind_store::truncate_topk;

/// One live point of a named collection: a handle to its blob plus payload.
/// The vectors are decoded out of the blob on demand, per space.
#[derive(Debug, Clone)]
struct NamedPoint {
    id: PointId,
    blob: BlobRef,
    payload: Option<Payload>,
    /// Handle to the committed JSON bytes in `payloads.bin`; `None` until the
    /// current payload has been checkpointed.
    payload_blob: Option<BlobRef>,
}

impl NamedPoint {
    fn replace_payload(&mut self, payload: Option<Payload>) {
        self.payload = payload;
        self.payload_blob = None;
    }

    fn touch_payload(&mut self) {
        self.payload_blob = None;
    }
}

/// In-memory mutable state of a named collection.
#[derive(Debug, Default)]
struct Inner {
    /// Slot-indexed live points; `None` marks a tombstoned slot.
    points: Vec<Option<NamedPoint>>,
    /// Point id -> live slot directory.
    reverse: IdMap,
    /// Cached L2 norms of sparse vectors: space -> slot -> norm.
    norms: HashMap<String, HashMap<u32, f32>>,
    next_lsn: u64,
    live_count: u64,
}

/// Persisted per-collection metadata (`named.bin`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct NamedMeta {
    collection: String,
    /// Declared spaces keyed by name, in stable (BTreeMap) order.
    spaces: BTreeMap<String, NamedSpaceMeta>,
    /// Admission cap mirrored from the collection configuration.
    #[serde(default)]
    max_points: Option<u64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct NamedSpaceMeta {
    kind: VectorKind,
    vector_size: usize,
    distance: crate::types::DistanceMetric,
    #[serde(default)]
    multi_config: Option<crate::types::MultiConfig>,
}

/// One checkpointed point. The vector map is referenced through `blob`: the
/// snapshot is a directory, not a copy of the data.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct NamedSnapshotPoint {
    slot: u32,
    id: String,
    blob: BlobRef,
    /// Handle into `payloads.bin`, not an inlined JSON string.
    payload: Option<BlobRef>,
    /// Sparse norms by space index, so scoring needs no blob reads at open.
    #[serde(default)]
    norms: Vec<(u16, f32)>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct NamedSnapshot {
    lsn: u64,
    /// Length of `points.bin` this snapshot describes.
    blobs_len: u64,
    /// Length of `payloads.bin` this snapshot describes.
    payloads_len: u64,
    points: Vec<NamedSnapshotPoint>,
}

/// A single opened named-vectors collection.
pub struct NamedStore {
    dir: PathBuf,
    _dir_lock: CollectionLock,
    generation_poisoned: AtomicBool,
    name: String,
    spaces: BTreeMap<String, NamedSpaceMeta>,
    /// Stable space index, the key a point blob uses for each space. Derived
    /// from the declaration order, which `BTreeMap` keeps across restarts.
    space_order: BTreeMap<String, u16>,
    inner: Mutex<Inner>,
    wal: Wal,
    /// Engine-level storage tuning this collection was opened with.
    storage: StorageConfig,
    /// Admission cap enforced before any mutation is logged.
    max_points: Option<u64>,
    /// Serializes WAL append, in-memory apply and checkpoint/truncate.
    gate: Mutex<()>,
    /// Operational counters for the exact-scan path.
    metrics: Metrics,
    /// Published data generation. Its lsn bounds WAL replay.
    generation: RwLock<Generation>,
    /// Segmented append-only vector bytes.
    blobs: ArcSwap<SegmentedBlobs>,
    /// Segmented append-only JSON payload bytes.
    payloads: ArcSwap<SegmentedBlobs>,
    /// Sparse postings per space, keyed by space name. Dense and multi spaces
    /// have no entry.
    postings: ArcSwap<HashMap<String, Arc<Postings>>>,
    /// Per-space ANN indexes for dense spaces that declare an ANN tier. Spaces
    /// on Flat or non-dense kinds have no entry and use exact scan. Swapped as
    /// a whole on compaction so readers never see a half-rebuilt set.
    ann: ArcSwap<BTreeMap<String, Arc<SpaceAnn>>>,
    /// Collection-level index tier applied to dense spaces (named layouts
    /// share one tier across their dense spaces; sparse/multi stay exact).
    index: crate::types::IndexConfig,
    /// Per-space element storage type and its affine parameters. Dense spaces
    /// may use f16/uint8/binary; sparse/multi stay f32.
    dtype_map: BTreeMap<
        String,
        (
            crate::types::VectorDataType,
            crate::storage::spaces_meta::DtypeParams,
        ),
    >,
}

impl NamedStore {
    /// Whether a generation publish left the in-memory switch unfinished.
    pub fn is_poisoned(&self) -> bool {
        self.generation_poisoned
            .load(std::sync::atomic::Ordering::Acquire)
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
}

impl std::fmt::Debug for NamedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedStore")
            .field("dir", &self.dir)
            .field("spaces", &self.spaces.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl NamedStore {
    /// Space index used inside point blobs and as the postings directory name.
    fn order_of(&self, space: &str) -> Option<u16> {
        self.space_order.get(space).copied()
    }

    /// Name of a space index stored in a blob.
    fn space_at(&self, index: u16) -> Option<&str> {
        self.space_order
            .iter()
            .find_map(|(name, slot)| (*slot == index).then_some(name.as_str()))
    }

    /// Sparse spaces in declaration order, with their indices.
    fn sparse_spaces(&self) -> impl Iterator<Item = (&str, u16)> {
        self.spaces.iter().filter_map(|(name, meta)| {
            (meta.kind == VectorKind::Sparse)
                .then(|| (name.as_str(), self.order_of(name).unwrap_or(0)))
        })
    }

    /// Postings of one space, if it is sparse.
    fn postings_of(&self, space: &str) -> Option<Arc<Postings>> {
        self.postings.load_full().get(space).cloned()
    }
}
