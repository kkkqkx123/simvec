//! `KindStore` — sparse and multi-vector collections.
//!
//! Dense collections run through [`crate::storage::CollectionStore`] with its
//! fixed-slot `vectors.bin` and the HNSW/IVF/quantization pipeline. Sparse
//! and Multi (late-interaction) payloads do not fit that pipeline: their data
//! is variable length and neither kind uses ANN indexes or quantization.
//! This module hosts a self-contained exact-scan engine for those kinds:
//!
//! - slot table with an id -> slot directory ([`IdMap`], split by id type);
//! - point vectors and JSON payloads in append-only mmap blob files
//!   ([`PointBlobs`]), so a reopen maps the data instead of deserializing a
//!   full snapshot;
//! - sparse postings in persisted bucket files ([`Postings`]), so a reopen
//!   loads a term dictionary instead of rebuilding the inverted index;
//! - weighted dot / Euclid / Manhattan scoring for Sparse and exact MaxSim
//!   for Multi, straight out of the mappings;
//! - WAL-backed durability by replaying the shared `wal/` record format
//!   (`WalRecord::UpsertVectors` carries the named payload map).
//!
//! What stays on the heap is the *directory* — slot -> (blob handle, payload)
//! plus the sparse term dictionary and norm cache — not the vectors. Only the
//! postings written since the last checkpoint are staged in memory.
//!
//! Nothing in the read path may assume a blob or a posting is unique to a live
//! slot: deleting or overwriting a point only unlinks the bytes, and
//! compaction is what rewrites the files. Queries filter tombstoned slots,
//! which is also what makes stale postings harmless.

#[cfg(test)]
#[path = "kind_store_tests.rs"]
mod tests;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use arc_swap::ArcSwap;
use parking_lot::{Mutex, RwLock};

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::storage::point_blobs::BlobRef;
use crate::storage::postings::Postings;
use crate::storage::segmented_blobs::SegmentedBlobs;
use crate::storage::{CollectionLock, Generation, IdMap, Wal};
use crate::types::{Payload, PointId, StorageConfig, VectorKind};

mod lifecycle;
mod read;
mod score;
mod state;
mod write;

pub(crate) use self::score::prefer_posting_scan;
pub(crate) use self::score::score_view;
pub(crate) use self::score::sparse_candidate_scores;
pub(crate) use self::score::truncate_topk;
pub use self::write::kind_upsert_record;

/// One live point of a Sparse/Multi collection.
#[derive(Debug, Clone)]
struct KindPoint {
    id: PointId,
    /// Handle to the vector bytes in `points.bin`.
    blob: BlobRef,
    payload: Option<Payload>,
    /// Handle to the committed JSON bytes in `payloads.bin`. `None` means the
    /// in-memory payload has changed since the last checkpoint.
    payload_blob: Option<BlobRef>,
}

impl KindPoint {
    fn replace_payload(&mut self, payload: Option<Payload>) {
        self.payload = payload;
        self.payload_blob = None;
    }

    fn touch_payload(&mut self) {
        self.payload_blob = None;
    }
}

/// In-memory mutable state of a kind collection.
#[derive(Debug, Default)]
struct Inner {
    /// Slot-indexed live points; `None` marks a tombstoned slot.
    points: Vec<Option<KindPoint>>,
    /// Point id -> live slot directory.
    reverse: IdMap,
    /// Cached L2 norm of each live sparse vector, indexed by slot. One `f32`
    /// per sparse slot, restored from the snapshot so scoring never has to
    /// read a blob at open.
    norms: HashMap<u32, f32>,
    next_lsn: u64,
    live_count: u64,
}

/// A single opened Sparse or Multi collection.
pub struct KindStore {
    dir: PathBuf,
    _dir_lock: CollectionLock,
    generation_poisoned: AtomicBool,
    name: String,
    kind: VectorKind,
    /// Dense dimension / sparse vocabulary bound / multi sub-vector dimension.
    dim: usize,
    /// Per-point sub-vector cap for Multi collections; `0` for Sparse.
    max_vectors_per_point: usize,
    distance: crate::types::DistanceMetric,
    /// Element storage type of dense vectors and its affine parameters.
    dtype: crate::types::VectorDataType,
    dtype_params: crate::storage::spaces_meta::DtypeParams,
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
    /// Segmented append-only vector bytes (`spaces/<space>/segments/<start>/points.bin`).
    blobs: ArcSwap<SegmentedBlobs>,
    /// Segmented append-only JSON payload bytes.
    payloads: ArcSwap<SegmentedBlobs>,
    /// Persisted sparse postings (`postings/`).
    postings: ArcSwap<Postings>,
}

impl KindStore {
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

impl std::fmt::Debug for KindStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KindStore")
            .field("dir", &self.dir)
            .field("kind", &self.kind)
            .field("vector_size", &self.dim)
            .finish()
    }
}

/// Checkpointed live point. Slot ids stay stable across snapshots so the
/// slot-addressed payload records in the WAL remain valid; the vector itself
/// is referenced (`blob`), never inlined.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct KindSnapshotPoint {
    slot: u32,
    id: String,
    blob: BlobRef,
    /// Handle into `payloads.bin`, not an inlined JSON string.
    payload: Option<BlobRef>,
    /// Sparse L2 norm, so a reopen needs no blob reads to score.
    #[serde(default)]
    norm: Option<f32>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct KindSnapshot {
    lsn: u64,
    /// Length of `points.bin` this snapshot describes; `PointBlobs::open`
    /// drops anything beyond it.
    blobs_len: u64,
    /// Length of `payloads.bin` this snapshot describes.
    payloads_len: u64,
    points: Vec<KindSnapshotPoint>,
}

/// Persisted per-collection metadata (`kind.bin`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct KindMeta {
    collection: String,
    kind: VectorKind,
    vector_size: usize,
    distance: crate::types::DistanceMetric,
    /// Sub-vector cap for Multi collections; `0` for Sparse.
    max_vectors_per_point: usize,
    /// Admission cap mirrored from the collection configuration.
    #[serde(default)]
    max_points: Option<u64>,
}
