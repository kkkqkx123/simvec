//! HNSW graph index.
//!
//! Ported from the bundled pgvector 0.8.x reference implementation
//! (`ref/pgvector/src/hnsw*.c`) and adapted to this engine's slot/mmap
//! storage model:
//!
//! - Nodes are slots; vector values stay in `vectors.bin` and are read
//!   through mmap snapshots exactly like the IVF and exact-scan paths.
//! - The graph is a *derived* structure: it can be rebuilt from the live set
//!   at any time, so persistence is a pure open-latency optimization and any
//!   validation failure falls back to exact scan.
//! - Tombstoned slots remain navigable but are never returned, mirroring
//!   pgvector's treatment of dead tuples between DELETE and VACUUM; physical
//!   reclamation happens on compaction followed by a wholesale rebuild.
//! - Inserts are incremental (paper Algorithm 1): no global retraining and
//!   no drift maintenance. Overwriting an existing slot keeps its position
//!   in the graph; distances are always computed against the live vector in
//!   `vectors.bin`, so a stale position only costs recall until the next
//!   rebuild fixes the topology. Overwrites are counted
//!   ([`HnswIndex::overwrites_since_build`]) so the optional ratio-based
//!   rebuild policy (`HnswConfig::stale_rebuild_ratio`) can observe and
//!   react to staleness instead of waiting for a compaction.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::error::{Result, VectorSearchError};
use crate::index::persist::{PersistedHnsw, PersistedNodeRecord};
use crate::index::IndexBuildParams;
use crate::metrics::{timed_write_lock, Metrics};
use crate::storage::vectors::Vectors;
use crate::storage::TombstoneBits;
use crate::types::{DistanceMetric, HnswConfig};

use bitvec::prelude::*;

mod build;
mod maintain;
mod persist;
mod search;

/// Default cap on iterative-scan expansion rounds when a filtered search
/// comes up short and `HnswConfig::iterative_max_rounds` is unset.
pub(crate) const DEFAULT_ITERATIVE_MAX_ROUNDS: usize = 3;

pub(crate) struct HnswSearchContext<'a> {
    pub tombstones: Option<&'a TombstoneBits>,
    pub vectors: &'a [Arc<memmap2::Mmap>],
    pub segment_slots: u32,
    pub filter_mask: Option<&'a BitVec>,
}

/// Per-thread reusable working sets for the search hot path.
///
/// The visited set is a generation array sized to the node table: marking
/// compares the slot generation instead of clearing, so successive queries
/// pay no reset pass. Heaps, the discard buffer, and the neighbor copy
/// buffer are cleared but keep their capacity across queries on the same
/// thread.
#[derive(Default)]
struct HnswQueryScratch {
    visited_gen: Vec<u32>,
    current_gen: u32,
    candidates: std::collections::BinaryHeap<std::cmp::Reverse<Cand>>,
    results: std::collections::BinaryHeap<Cand>,
    discarded: Vec<Cand>,
    neighbors: Vec<u32>,
}

impl HnswQueryScratch {
    fn prepare(&mut self, node_count: usize) {
        if self.visited_gen.len() < node_count {
            self.visited_gen.resize(node_count, 0);
        }
        self.current_gen = self.current_gen.wrapping_add(1);
        if self.current_gen == 0 {
            self.visited_gen.fill(0);
            self.current_gen = 1;
        }
        self.candidates.clear();
        self.results.clear();
        self.discarded.clear();
    }

    #[inline]
    fn mark_visited(&mut self, slot: u32) -> bool {
        let idx = slot as usize;
        if idx >= self.visited_gen.len() {
            self.visited_gen.resize(idx + 1, 0);
        }
        if self.visited_gen[idx] == self.current_gen {
            return false;
        }
        self.visited_gen[idx] = self.current_gen;
        true
    }
}

thread_local! {
    static QUERY_SCRATCH: std::cell::RefCell<HnswQueryScratch> =
        std::cell::RefCell::new(HnswQueryScratch::default());
}

/// Per-thread reusable buffers for neighbor selection.
///
/// The sorted pool and the pruned spill keep capacity across inserts on the
/// same thread; only the returned selection (at most one layer cap) is
/// freshly allocated.
#[derive(Default)]
struct HnswSelectScratch {
    sorted: Vec<Cand>,
    pruned: Vec<Cand>,
}

thread_local! {
    static SELECT_SCRATCH: std::cell::RefCell<HnswSelectScratch> =
        std::cell::RefCell::new(HnswSelectScratch::default());
}

/// Per-thread buffers for the reverse-link overflow path. The pool and the
/// selection output are both bounded by the layer cap plus one, so these two
/// buffers cover every call on this thread. They stay separate from
/// `SELECT_SCRATCH` because the pool is still borrowed while the selection
/// runs, which borrows the select scratch in turn.
#[derive(Default)]
struct HnswLinkScratch {
    pool: Vec<Cand>,
    selected: Vec<Cand>,
}

thread_local! {
    static LINK_SCRATCH: std::cell::RefCell<HnswLinkScratch> =
        std::cell::RefCell::new(HnswLinkScratch::default());
}

/// Hard cap for generated levels. `ml = 1/ln(m)` makes higher levels
/// exponentially unlikely; the cap only guards pathological RNG draws
/// (pgvector caps by page geometry, which does not apply here).
pub(crate) const MAX_LEVEL: u8 = 32;

/// Minimum connections per layer (> 0), mirroring pgvector `HNSW_MIN_M`.
const MIN_M: usize = 2;

/// Deterministic splitmix64; seeded from the collection name and live count
/// so builds are reproducible across restarts (same property as the IVF
/// k-means seeding).
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform sample from (0, 1): 53 mantissa bits, nudged off zero so the
    /// logarithm in the level draw stays finite.
    fn next_open01(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        if u == 0.0 {
            f64::EPSILON
        } else {
            u
        }
    }
}

/// Entry point of the layered graph.
#[derive(Debug, Clone, Copy)]
struct EntryPoint {
    slot: u32,
    level: u8,
}

/// One graph node. Immutable after creation except for the adjacency lists,
/// which sit behind per-layer locks so searches never block on the store
/// lock (same concurrency shape as the IVF per-list locks).
///
/// Mirrors pgvector's `HnswElementData.version` field: a 4-bit counter
/// (cycling 1–15) incremented on every adjacency mutation. Concurrent readers
/// can snapshot the version before and after loading a neighborhood; if the
/// version changed, the loaded adjacency may be stale and the read should be
/// retried or the search widened. This detects the same class of anomalies
/// that pgvector guards against during vacuum-initiated neighbor rewrites
/// under concurrent iterative scans.
struct Node {
    level: u8,
    /// Adjacency per layer; one locked list per layer `0..=level`.
    neighbors: Vec<RwLock<Vec<u32>>>,
    /// Monotonic counter (wrapping, 4-bit effective range 1–15) incremented
    /// every time any layer's adjacency list is mutated. Readers snapshot this
    /// before and after loading neighbors to detect concurrent modifications.
    version: std::sync::atomic::AtomicU8,
}

impl Node {
    fn new(level: u8) -> Self {
        Self {
            level,
            neighbors: (0..=level).map(|_| RwLock::new(Vec::new())).collect(),
            version: std::sync::atomic::AtomicU8::new(1),
        }
    }

    /// Adjacency of `lc`. Callers only reach layer `lc` through links created
    /// at that layer, so `lc <= level` holds by construction; the clamp keeps
    /// a corrupted link from panicking.
    fn layer(&self, lc: u8) -> &RwLock<Vec<u32>> {
        &self.neighbors[(lc as usize).min(self.neighbors.len() - 1)]
    }

    /// Snapshot the current version. Callers comparing before/after versions
    /// detect any adjacency mutation that occurred between the two reads.
    fn version(&self) -> u8 {
        self.version.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Bump the version after an adjacency mutation. The wrapping u8 naturally
    /// cycles through 1–15 then wraps to 0 and is immediately bumped back to 1,
    /// matching pgvector's 4-bit cycling semantics.
    fn bump_version(&self) {
        use std::sync::atomic::Ordering;
        let old = self.version.load(Ordering::Relaxed);
        // Ensure version is never 0 (pgvector convention: 0 means invalid).
        let new = if old == 0 || old == u8::MAX {
            1
        } else {
            old.wrapping_add(1)
        };
        self.version.store(new, Ordering::Release);
    }
}

/// Internal (distance, slot) pair ordered by distance for the heaps below.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cand {
    dist: f32,
    slot: u32,
}

/// Result of a single search_layer pass, containing both the live top-ef
/// candidates and the discarded candidates that were evicted from the live
/// set. Discarded candidates can be used to resume search in iterative scan.
struct SearchLayerResult {
    /// Top-ef live candidates, closest first.
    live: Vec<Cand>,
    /// Discarded candidates that were evicted from the live set during the
    /// search. May contain duplicates across iterative calls.
    discarded: Vec<Cand>,
    /// Distinct nodes touched by this pass (including tombstoned and
    /// non-matching ones); feeds the iterative-scan tuple budget.
    visited: usize,
}

impl Eq for Cand {}

impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then(self.slot.cmp(&other.slot))
    }
}

impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// In-memory HNSW index for one collection.
///
/// Incremental mutation paths run serialized under the store write lock.
/// Builds may instead run several workers over disjoint slot ranges against
/// one shared instance; that path relies on the per-layer/per-node locks,
/// the mutex-guarded RNG, and [`Self::promote_lock`] serializing entry-point
/// publication (see `insert`). Search paths take short locks and an atomic
/// entry-point load, so concurrent searches never block mutations beyond
/// individual list updates.
pub(crate) struct HnswIndex {
    dim: usize,
    metric: DistanceMetric,
    /// Maximum connections per layer above the ground layer.
    m: usize,
    /// Ground-layer cap: `2 * m`, mirroring pgvector `HnswGetLayerM`.
    max_neighbors_0: usize,
    /// Build-time candidate list size.
    ef_construct: usize,
    /// Default layer-0 search width when a query carries none.
    ef_search: usize,
    /// Level assignment factor `1 / ln(m)` from the paper.
    ml: f64,
    /// Slot-indexed node table; `None` marks slots absent from the graph.
    nodes: RwLock<Vec<Option<Arc<Node>>>>,
    entry: RwLock<Option<EntryPoint>>,
    rng: Mutex<SplitMix64>,
    /// Serializes entry-point publication during concurrent builds (the
    /// counterpart of pgvector's `entryLock`). Unlike pgvector no second
    /// reader-facing lock is needed: the new entry is fully linked before
    /// its pointer becomes visible, and readers either observe the previous
    /// entry (still valid) or the new one with complete adjacency lists.
    promote_lock: Mutex<()>,
    built_at_live_count: AtomicU64,
    /// Overwrite upserts observed since this instance was created (build or
    /// reload). Each overwrite keeps the node's old graph position, so this
    /// is a staleness proxy: combined with `built_at_live_count` it feeds
    /// `HnswConfig::stale_rebuild_ratio`. It resets when a persisted graph
    /// is reloaded, i.e. the baseline restarts with the process.
    overwrites_since_build: AtomicU64,
    /// Sum of absolute distance changes from overwrite upserts since build,
    /// in fixed-point (× 1e6). Combined with `overwrites_since_build` this
    /// gives a more precise staleness signal than raw counts alone.
    overwrites_distance_delta: AtomicU64,
    /// Iterative-scan expansion round cap, resolved from
    /// `HnswConfig::iterative_max_rounds` at construction (baked in like the
    /// other graph parameters; changes take effect on the next rebuild).
    iterative_max_rounds: usize,
    /// Cumulative visited-node cap for iterative scans, from
    /// `HnswConfig::max_scan_tuples`. `None` = unbounded.
    max_scan_tuples: Option<u64>,
    /// Per-collection metrics sink for lock-contention and version-reload
    /// diagnostics. Attached by the store after construction; `None` in
    /// standalone builds (benches, tests), which then record nothing.
    lock_metrics: Option<Arc<Metrics>>,
}

impl HnswIndex {
    fn new(dim: usize, metric: DistanceMetric, config: &HnswConfig, seed: u64) -> Self {
        let m = config.m.max(MIN_M);
        Self {
            dim,
            metric,
            m,
            max_neighbors_0: m * 2,
            ef_construct: config.ef_construct.max(1),
            ef_search: config.ef_search.max(1),
            ml: 1.0 / (m as f64).ln(),
            nodes: RwLock::new(Vec::new()),
            entry: RwLock::new(None),
            rng: Mutex::new(SplitMix64::new(seed)),
            promote_lock: Mutex::new(()),
            built_at_live_count: AtomicU64::new(0),
            overwrites_since_build: AtomicU64::new(0),
            overwrites_distance_delta: AtomicU64::new(0),
            iterative_max_rounds: config
                .iterative_max_rounds
                .unwrap_or(DEFAULT_ITERATIVE_MAX_ROUNDS)
                .max(1),
            max_scan_tuples: config.max_scan_tuples,
            lock_metrics: None,
        }
    }

    /// Attach the owning collection's metrics sink. Called by the store
    /// right after construction, before the index is published.
    pub(crate) fn set_metrics(&mut self, metrics: Arc<Metrics>) {
        self.lock_metrics = Some(metrics);
    }

    /// Cap on iterative-scan expansion rounds for filtered searches.
    pub(crate) fn iterative_rounds(&self) -> usize {
        self.iterative_max_rounds
    }

    /// Cumulative visited-node cap for iterative scans.
    pub(crate) fn scan_tuple_budget(&self) -> Option<u64> {
        self.max_scan_tuples
    }

    pub(crate) fn node_count(&self) -> usize {
        self.nodes.read().iter().flatten().count()
    }

    pub(crate) fn built_at_live_count(&self) -> u64 {
        self.built_at_live_count.load(Ordering::Relaxed)
    }

    /// Overwrite upserts observed since this instance was built or reloaded.
    pub(crate) fn overwrites_since_build(&self) -> u64 {
        self.overwrites_since_build.load(Ordering::Relaxed)
    }

    pub(crate) fn default_ef(&self) -> usize {
        self.ef_search
    }

    /// Maximum connections per layer above the ground layer.
    pub(crate) fn m(&self) -> usize {
        self.m
    }

    /// Build-time candidate list size.
    pub(crate) fn ef_construct(&self) -> usize {
        self.ef_construct
    }

    fn layer_cap(&self, lc: u8) -> usize {
        if lc == 0 {
            self.max_neighbors_0
        } else {
            self.m
        }
    }

    fn node(&self, slot: u32) -> Option<Arc<Node>> {
        self.nodes.read().get(slot as usize).cloned().flatten()
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        crate::distance::distance(self.metric, a, b)
    }
}

#[cfg(test)]
#[path = "hnsw/tests.rs"]
mod tests;
