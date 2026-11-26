//! `KindStore` — sparse and multi-vector collections.
//!
//! Dense collections run through [`crate::storage::CollectionStore`] with its
//! fixed-slot `vectors.bin` and the HNSW/IVF/quantization pipeline. Sparse
//! and Multi (late-interaction) payloads do not fit that pipeline: their data
//! is variable length and neither kind uses ANN indexes or quantization.
//! This module hosts a self-contained exact-scan engine for those kinds:
//!
//! - in-memory slot table with id -> slot directory;
//! - inverted postings per sparse term for candidate collection;
//! - weighted dot / Euclid / Manhattan scoring for Sparse and exact MaxSim
//!   for Multi;
//! - WAL-backed durability by replaying the shared `wal.bin` record format
//!   (`WalRecord::UpsertVectors` carries the named payload map).
//!
//! The store is a reference implementation on purpose: it keeps every live
//! point in memory and rebuilds by full WAL replay, mirroring the memory
//! backend's role in `vector-contracts`.

#[cfg(test)]
#[path = "kind_store_tests.rs"]
mod tests;

use std::collections::HashMap;
use std::path::PathBuf;

use parking_lot::Mutex;

use crate::metrics::Metrics;
use crate::storage::Wal;
use crate::types::{Payload, PointId, StorageConfig, VectorData, VectorKind};

mod lifecycle;
mod read;
mod score;
mod state;
mod write;

pub(crate) use self::score::score_pair;
pub(crate) use self::score::sparse_candidate_scores;
pub(crate) use self::score::truncate_topk;
pub use self::write::kind_upsert_record;

/// One live point of a Sparse/Multi collection.
#[derive(Debug, Clone)]
struct KindPoint {
    id: PointId,
    data: VectorData,
    payload: Option<Payload>,
}

/// In-memory mutable state of a kind collection.
#[derive(Debug, Default)]
struct Inner {
    /// Slot-indexed live points; `None` marks a tombstoned slot.
    points: Vec<Option<KindPoint>>,
    /// Point id -> live slot directory.
    reverse: HashMap<String, u32>,
    /// Sparse-only inverted index: term -> (slot, weight) postings. Empty for
    /// Multi collections.
    inverted: HashMap<u32, Vec<(u32, f32)>>,
    /// Cached L2 norm of each live sparse vector, indexed by slot.
    norms: HashMap<u32, f32>,
    next_lsn: u64,
    live_count: u64,
}

/// A single opened Sparse or Multi collection.
pub struct KindStore {
    dir: PathBuf,
    name: String,
    kind: VectorKind,
    /// Dense dimension / sparse vocabulary bound / multi sub-vector dimension.
    dim: usize,
    /// Per-point sub-vector cap for Multi collections; `0` for Sparse.
    max_vectors_per_point: usize,
    distance: crate::types::DistanceMetric,
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
    /// Snapshot path; the WAL is replayed on top of this derived state.
    snapshot: PathBuf,
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
/// slot-addressed payload records in the WAL remain valid.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct KindSnapshotPoint {
    slot: u32,
    id: String,
    data: VectorData,
    payload: Option<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct KindSnapshot {
    lsn: u64,
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
