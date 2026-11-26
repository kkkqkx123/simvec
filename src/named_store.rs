//! `NamedStore` — named-vectors collections.
//!
//! A named collection declares a map of vector spaces (dense or sparse, each
//! with its own dimension and distance metric). Points carry one vector per
//! declared space and every query targets exactly one space by name. This
//! module mirrors [`crate::kind_store::KindStore`]: an in-memory exact-scan
//! engine with variable-length payloads and WAL-backed durability, rebuilt
//! by full replay on open. No ANN indexes, no quantization.

#[cfg(test)]
#[path = "named_store_tests.rs"]
mod tests;

use std::collections::{BTreeMap, HashMap};
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
pub(crate) use self::score::sparse_postings;
pub use self::write::named_upsert_record;

/// One live point of a named collection: a full vector per declared space.
#[derive(Debug, Clone)]
struct NamedPoint {
    id: PointId,
    vectors: BTreeMap<String, VectorData>,
    payload: Option<Payload>,
}

/// In-memory mutable state of a named collection.
#[derive(Debug, Default)]
struct Inner {
    /// Slot-indexed live points; `None` marks a tombstoned slot.
    points: Vec<Option<NamedPoint>>,
    /// Point id -> live slot directory.
    reverse: HashMap<String, u32>,
    /// Per-space sparse inverted index: space -> term -> (slot, weight).
    /// Dense spaces have no entry.
    inverted: HashMap<String, HashMap<u32, Vec<(u32, f32)>>>,
    next_lsn: u64,
    live_count: u64,
}

/// Persisted per-collection metadata (`named.bin`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct NamedMeta {
    collection: String,
    /// Declared spaces keyed by name, in stable (BTreeMap) order.
    spaces: BTreeMap<String, NamedSpaceMeta>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct NamedSpaceMeta {
    kind: VectorKind,
    vector_size: usize,
    distance: crate::types::DistanceMetric,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct NamedSnapshotPoint {
    slot: u32,
    id: String,
    vectors: BTreeMap<String, VectorData>,
    payload: Option<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct NamedSnapshot {
    lsn: u64,
    points: Vec<NamedSnapshotPoint>,
}

/// A single opened named-vectors collection.
pub struct NamedStore {
    dir: PathBuf,
    spaces: BTreeMap<String, NamedSpaceMeta>,
    inner: Mutex<Inner>,
    wal: Wal,
    /// Engine-level storage tuning this collection was opened with.
    storage: StorageConfig,
    /// Operational counters for the exact-scan path.
    metrics: Metrics,
    /// Snapshot path; the WAL is replayed on top of this derived state.
    snapshot: PathBuf,
}

impl std::fmt::Debug for NamedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedStore")
            .field("dir", &self.dir)
            .field("spaces", &self.spaces.keys().collect::<Vec<_>>())
            .finish()
    }
}
