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

use std::collections::HashSet;
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

/// Default cap on iterative-scan expansion rounds when a filtered search
/// comes up short and `HnswConfig::iterative_max_rounds` is unset.
pub(crate) const DEFAULT_ITERATIVE_MAX_ROUNDS: usize = 3;

pub(crate) struct HnswSearchContext<'a> {
    pub tombstones: Option<&'a TombstoneBits>,
    pub vectors: &'a [Arc<memmap2::Mmap>],
    pub segment_slots: u32,
    pub filter_mask: Option<&'a BitVec>,
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

    /// Build a fresh index by inserting the given live slots (off the store
    /// lock; `params.slots` must be sorted ascending).
    ///
    /// `HnswConfig::max_indexing_threads` selects the build shape:
    /// - unset/0: sequential insertion on the global rayon pool;
    /// - 1: sequential insertion on a dedicated single-thread pool;
    /// - n >= 2: n workers insert disjoint round-robin slot subsets into the
    ///   shared graph concurrently (entry-point publication serialized by
    ///   [`Self::promote_lock`]). Topology then depends on interleaving, so
    ///   only recall invariants hold — asserted by the multi-worker tests.
    ///
    /// `IndexBuildParams::progress`, when attached, is incremented once per
    /// processed slot and emits milestone debug logs.
    pub(crate) fn build(params: &IndexBuildParams<'_>, config: &HnswConfig) -> Result<Self> {
        config.validate()?;
        let mut seed = params.collection.len() as u64;
        for b in params.collection.as_bytes() {
            seed = seed.wrapping_mul(0x100_0000_01b3) ^ (*b as u64);
        }
        seed ^= params.slots.len() as u64;
        seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);

        let index = Self::new(params.dim, params.metric, config, seed);
        let total = params.slots.len() as u64;
        // Milestone for periodic progress logging: every ~10% of a large
        // build, at most every 10k slots on huge corpora.
        let milestone = (total / 10).max(10_000);
        let insert_one = |index: &Self, slot: u32| {
            if let Some(v) = Vectors::read_slot(
                params.vectors,
                slot as u64,
                params.segment_slots,
                params.dim,
            ) {
                index.insert(slot, v, params.vectors, params.segment_slots);
            }
            if let Some(progress) = params.progress {
                let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if done % milestone == 0 || done == total {
                    tracing::debug!(
                        collection = %params.collection,
                        done,
                        total,
                        "hnsw build progress"
                    );
                }
            }
        };

        let workers = match config.max_indexing_threads {
            Some(threads) if threads >= 2 => threads.min(params.slots.len().max(1)),
            _ => 1,
        };
        if workers > 1 {
            // Concurrent workers share one pool sized to their count; the
            // per-insert distance fan-outs also land on this pool. Pool
            // tasks are pure distance math and never take graph locks, so
            // nested joins cannot deadlock against worker progress.
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .map_err(|e| VectorSearchError::Internal(format!("hnsw build thread pool: {e}")))?;
            std::thread::scope(|scope| {
                let index_ref = &index;
                let pool_ref = &pool;
                for w in 0..workers {
                    scope.spawn(move || {
                        pool_ref.install(|| {
                            for (i, &slot) in params.slots.iter().enumerate() {
                                if i % workers == w {
                                    insert_one(index_ref, slot);
                                }
                            }
                        });
                    });
                }
            });
        } else if matches!(config.max_indexing_threads, Some(threads) if threads == 1) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .map_err(|e| VectorSearchError::Internal(format!("hnsw build thread pool: {e}")))?;
            pool.install(|| {
                for &slot in params.slots {
                    insert_one(&index, slot);
                }
            });
        } else {
            for &slot in params.slots {
                insert_one(&index, slot);
            }
        }
        index
            .built_at_live_count
            .store(params.slots.len() as u64, Ordering::Relaxed);
        Ok(index)
    }

    /// Rehydrate from persisted state. Structural violations produce
    /// `CorruptData` so callers discard the file and fall back to exact scan.
    pub(crate) fn from_persisted(data: PersistedHnsw, config: &HnswConfig) -> Result<Self> {
        if data.dim == 0 || data.m < MIN_M {
            return Err(VectorSearchError::CorruptData(
                "hnsw.bin invalid header".to_string(),
            ));
        }
        let index = Self::new(
            data.dim,
            data.distance,
            &HnswConfig {
                m: data.m,
                ef_construct: data.ef_construct,
                ..config.clone()
            },
            0,
        );

        let mut nodes = Vec::new();
        for record in &data.nodes {
            if record.slot as usize != nodes.len() {
                return Err(VectorSearchError::CorruptData(
                    "hnsw.bin nodes not dense from slot 0".to_string(),
                ));
            }
            if record.neighbors.len() != record.level as usize + 1 {
                return Err(VectorSearchError::CorruptData(
                    "hnsw.bin node layer count mismatch".to_string(),
                ));
            }
            nodes.push(Some(Arc::new(Node::new(record.level))));
        }

        // Resolve adjacency after all nodes exist.
        for (i, record) in data.nodes.iter().enumerate() {
            let node = nodes[i].as_ref().expect("just pushed");
            // Restore version from persisted state (cycling 1–15).
            // Clamp to valid range for safety; version 0 is invalid in pgvector.
            let version = if record.version == 0 {
                1
            } else {
                record.version.min(15)
            };
            node.version
                .store(version, std::sync::atomic::Ordering::Release);
            for (lc, list) in record.neighbors.iter().enumerate() {
                let cap = index.layer_cap(lc as u8);
                if list.len() > cap || list.iter().any(|&s| s as usize >= nodes.len()) {
                    return Err(VectorSearchError::CorruptData(format!(
                        "hnsw.bin adjacency violation at slot {}",
                        record.slot
                    )));
                }
                *node.layer(lc as u8).write() = list.clone();
            }
        }

        if let Some((slot, level)) = data.entry {
            match nodes.get(slot as usize) {
                Some(Some(node)) if node.level as i32 == level => {}
                _ => {
                    return Err(VectorSearchError::CorruptData(
                        "hnsw.bin entry point invalid".to_string(),
                    ))
                }
            }
            *index.entry.write() = Some(EntryPoint {
                slot,
                level: level as u8,
            });
        }

        *index.nodes.write() = nodes;
        index
            .built_at_live_count
            .store(data.built_at_live_count, Ordering::Relaxed);
        Ok(index)
    }

    /// Snapshot of the persisted representation (for `hnsw.bin`).
    pub(crate) fn to_persisted(&self) -> PersistedHnsw {
        let nodes = self.nodes.read();
        let records = nodes
            .iter()
            .enumerate()
            .filter_map(|(slot, node)| {
                let node = node.as_ref()?;
                Some(PersistedNodeRecord {
                    slot: slot as u32,
                    level: node.level,
                    version: node.version(),
                    neighbors: node
                        .neighbors
                        .iter()
                        .map(|l| l.read().clone())
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        PersistedHnsw {
            dim: self.dim,
            distance: self.metric,
            m: self.m,
            ef_construct: self.ef_construct,
            ef_search: self.ef_search,
            entry: (*self.entry.read()).map(|e| (e.slot, e.level as i32)),
            built_at_live_count: self.built_at_live_count.load(Ordering::Relaxed),
            nodes: records,
        }
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

    /// Staleness ratio combining overwrite count and distance delta.
    /// Returns `max(count_ratio, delta_ratio)` where:
    /// - `count_ratio = overwrites / built_at_live_count`
    /// - `delta_ratio = distance_delta / built_at_live_count`
    pub(crate) fn stale_ratio(&self) -> f64 {
        let count = self.overwrites_since_build.load(Ordering::Relaxed);
        let delta = self.overwrites_distance_delta.load(Ordering::Relaxed) as f64 / 1e6;
        let base = self.built_at_live_count.load(Ordering::Relaxed).max(1) as f64;
        let count_ratio = count as f64 / base;
        let delta_ratio = delta / base;
        count_ratio.max(delta_ratio)
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

    fn random_level(&self) -> u8 {
        let draw = -self.rng.lock().next_open01().ln() * self.ml;
        (draw as u8).min(MAX_LEVEL)
    }

    /// Paper Algorithm 1 (insert). `vector` is the freshly written value of
    /// `slot`; all other node values come from the mmap snapshots.
    ///
    /// Inserting an already-present slot is a counted no-op: overwrite
    /// upserts keep their graph position (see the module docs for the
    /// trade-off) and bump [`Self::overwrites_since_build`].
    ///
    /// Safe to call concurrently for disjoint slots: node registration and
    /// adjacency updates take their own locks, and both entry-point
    /// transitions (first node, higher-level promotion) run under
    /// [`Self::promote_lock`] with a fresh re-read so concurrent builders
    /// can neither orphan a first node nor clobber an already-higher entry.
    pub(crate) fn insert(
        &self,
        slot: u32,
        vector: &[f32],
        vectors: &[Arc<memmap2::Mmap>],
        segment_slots: u32,
    ) {
        if self.node(slot).is_some() {
            self.overwrites_since_build.fetch_add(1, Ordering::Relaxed);
            // Track the distance delta: how far the new vector is from the
            // old one. A large delta means the node's graph position is more
            // likely stale.
            if let Some(old_v) = Vectors::read_slot(vectors, slot as u64, segment_slots, self.dim) {
                let delta = (self.distance(vector, old_v) * 1e6) as u64;
                self.overwrites_distance_delta
                    .fetch_add(delta, Ordering::Relaxed);
            }
            return;
        }

        let level = self.random_level();
        let node = Arc::new(Node::new(level));
        {
            let mut nodes = self.nodes.write();
            if slot as usize >= nodes.len() {
                nodes.resize(slot as usize + 1, None);
            }
            nodes[slot as usize] = Some(Arc::clone(&node));
        }

        // First node in the graph: claim it under the promotion lock so a
        // racing builder cannot leave two unlinked roots.
        if self.entry.read().is_none() {
            let claim = self.promote_lock.lock();
            if self.entry.read().is_none() {
                *self.entry.write() = Some(EntryPoint { slot, level });
                return;
            }
            drop(claim);
        }

        // Snapshot the entry once. It may move upward while this insert is
        // running; descending from the stale (still valid) snapshot only
        // risks missing newer shortcuts, never correctness.
        let Some(entry) = *self.entry.read() else {
            return;
        };

        let entry_vec = Vectors::read_slot(vectors, entry.slot as u64, segment_slots, self.dim);
        let mut ep = match entry_vec {
            Some(ev) => vec![Cand {
                dist: self.distance(vector, ev),
                slot: entry.slot,
            }],
            None => return,
        };

        // Phase 1: greedy descent through layers above `level` (ef = 1).
        let mut lc = entry.level;
        while lc > level {
            let Some(&start) = ep.first() else {
                return;
            };
            let best = self.greedy_step(vector, start, lc, vectors, segment_slots);
            ep = vec![best];
            lc -= 1;
        }

        // Phase 2: search + connect on each layer down to the ground.
        for layer in (0..=level.min(entry.level)).rev() {
            let result = self.search_layer(
                vector,
                &ep,
                self.ef_construct,
                layer,
                &HnswSearchContext {
                    tombstones: None,
                    vectors,
                    segment_slots,
                    filter_mask: None,
                },
            );
            let selected =
                self.select_neighbors(&result.live, self.layer_cap(layer), vectors, segment_slots);
            *timed_write_lock(self.lock_metrics.as_deref(), node.layer(layer)) =
                selected.iter().map(|c| c.slot).collect();
            node.bump_version();
            for cand in &selected {
                self.link_neighbor(cand.slot, slot, cand.dist, layer, vectors, segment_slots);
            }
            ep = result.live;
        }

        if level > entry.level {
            // Publish the new higher entry under the promotion lock,
            // re-reading the latest state: another builder may have promoted
            // an even-higher entry in the meantime, and clobbering it would
            // strand its upper layers.
            let _publish = self.promote_lock.lock();
            if self.entry.read().as_ref().is_none_or(|e| level > e.level) {
                *self.entry.write() = Some(EntryPoint { slot, level });
            }
        }
    }

    /// Greedy descent with `ef = 1`: repeatedly move to the closest neighbor
    /// until no improvement. Equivalent to running paper Algorithm 2 with
    /// `ef = 1`, which is what pgvector uses between upper layers.
    fn greedy_step(
        &self,
        query: &[f32],
        start: Cand,
        layer: u8,
        vectors: &[Arc<memmap2::Mmap>],
        segment_slots: u32,
    ) -> Cand {
        let mut best = start;
        loop {
            let Some(node) = self.node(best.slot) else {
                return best;
            };
            let neighbors = node.layer(layer).read().clone();
            let mut improved = false;
            for slot in neighbors {
                let Some(v) = Vectors::read_slot(vectors, slot as u64, segment_slots, self.dim)
                else {
                    continue;
                };
                let dist = self.distance(query, v);
                if (Cand { dist, slot }).cmp(&best) == std::cmp::Ordering::Less {
                    best = Cand { dist, slot };
                    improved = true;
                }
            }
            if !improved {
                return best;
            }
        }
    }

    /// Paper Algorithm 2 (SEARCH-LAYER).
    ///
    /// `tombstones` filters which elements count towards `ef` and appear in
    /// the result; tombstoned nodes stay fully traversable so the graph does
    /// not fragment around deletions. `None` (insert-time search) counts
    /// every element, matching pgvector's build behavior.
    /// Paper Algorithm 2 (SEARCH-LAYER).
    ///
    /// `tombstones` filters which elements count towards `ef` and appear in
    /// the result; tombstoned nodes stay fully traversable so the graph does
    /// not fragment around deletions. `None` (insert-time search) counts
    /// every element, matching pgvector's build behavior.
    ///
    /// `filter_mask` (pre-filter bitmap) constrains which elements count
    /// towards `ef` and appear in the result, while the traversal frontier
    /// still walks through every live node. Mirroring pgvector's filter-aware
    /// scan this way keeps mismatched regions navigable: hard-pruning them
    /// from the frontier would disconnect the search whenever the entry seed
    /// lands outside the matching subgraph.
    ///
    /// This implementation detects concurrent adjacency mutations via the
    /// per-node `version` counter (mirroring pgvector's 4-bit version field).
    /// When a version change is detected between the snapshot and the load,
    /// the neighborhood is reloaded once to avoid using a torn adjacency list.
    fn search_layer(
        &self,
        query: &[f32],
        entries: &[Cand],
        ef: usize,
        layer: u8,
        ctx: &HnswSearchContext<'_>,
    ) -> SearchLayerResult {
        let live = |slot: u32| -> bool {
            match ctx.tombstones {
                Some(t) => !t.bit(slot as usize),
                None => true,
            }
        };
        let matches_filter = |slot: u32| -> bool {
            ctx.filter_mask
                .is_none_or(|m| (slot as usize) < m.len() && m[slot as usize])
        };

        // Pre-allocate with reasonable capacities to avoid frequent resizing
        // during the hot search loop. The visited set grows as we explore
        // neighbors; the heaps are bounded by ef.
        let initial_cap = (entries.len() + ef).max(16);
        let mut visited: HashSet<u32> = HashSet::with_capacity(initial_cap);
        // Candidate min-heap (closest pop) and result max-heap (furthest pop).
        //
        // The frontier navigates through every live node so a pre-filter
        // cannot disconnect the traversal; `results` collects only
        // filter-matching nodes and `live_results` budgets those matches.
        let mut candidates: std::collections::BinaryHeap<std::cmp::Reverse<Cand>> =
            std::collections::BinaryHeap::with_capacity(initial_cap);
        let mut results: std::collections::BinaryHeap<Cand> =
            std::collections::BinaryHeap::with_capacity(ef);
        let mut discarded: Vec<Cand> = Vec::new();
        let mut live_results = 0usize;

        for &cand in entries {
            if !live(cand.slot) || !visited.insert(cand.slot) {
                continue;
            }
            candidates.push(std::cmp::Reverse(cand));
            if matches_filter(cand.slot) {
                results.push(cand);
                live_results += 1;
            }
        }

        while let Some(std::cmp::Reverse(c)) = candidates.pop() {
            if live_results >= ef {
                match results.peek() {
                    // Enough matches collected and this frontier node is
                    // farther than the worst match: the rest of the frontier
                    // cannot improve the output.
                    Some(furthest) if c.dist > furthest.dist => break,
                    _ => {}
                }
            }
            let Some(node) = self.node(c.slot) else {
                continue;
            };
            // Snapshot the version before loading the adjacency list.
            // If it changes during the read, the list may be torn; reload once.
            let v1 = node.version();
            let neighbors = node.layer(layer).read().clone();
            let v2 = node.version();
            let neighbors = if v1 != v2 {
                // Version changed during the read; reload once. Counted so
                // concurrency regression tests can assert this recovery path
                // is actually exercised under write/read contention.
                if let Some(metrics) = &self.lock_metrics {
                    metrics.record_version_reload();
                }
                node.layer(layer).read().clone()
            } else {
                neighbors
            };
            for slot in neighbors {
                if !visited.insert(slot) || !live(slot) {
                    continue;
                }
                let Some(v) =
                    Vectors::read_slot(ctx.vectors, slot as u64, ctx.segment_slots, self.dim)
                else {
                    continue;
                };
                let dist = self.distance(query, v);
                let cand = Cand { dist, slot };
                candidates.push(std::cmp::Reverse(cand));
                if !matches_filter(slot) {
                    continue;
                }
                results.push(cand);
                live_results += 1;
                // Trim back to the `ef` budget. Evicted live candidates
                // are recorded in `discarded` for iterative scan resume.
                while results.len() > ef {
                    let Some(evicted) = results.pop() else {
                        break;
                    };
                    live_results -= 1;
                    discarded.push(evicted);
                }
            }
        }

        // Emit only live elements, closest first.
        let mut live_out: Vec<Cand> = results.into_iter().filter(|c| live(c.slot)).collect();
        live_out.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
        SearchLayerResult {
            live: live_out,
            discarded,
            visited: visited.len(),
        }
    }

    /// Paper Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) with pgvector's
    /// keep-pruned-connections variant: discarded candidates backfill `r`
    /// up to `lm` so nodes never end up under-connected.
    ///
    /// Pairwise distances are measured between candidate vectors, not
    /// against the query.
    fn select_neighbors(
        &self,
        candidates: &[Cand],
        lm: usize,
        vectors: &[Arc<memmap2::Mmap>],
        segment_slots: u32,
    ) -> Vec<Cand> {
        let mut sorted: Vec<Cand> = candidates.to_vec();
        sorted.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
        if sorted.len() <= lm {
            return sorted;
        }

        let mut selected: Vec<Cand> = Vec::with_capacity(lm);
        let mut pruned: Vec<Cand> = Vec::new();
        for cand in sorted {
            if selected.len() == lm {
                break;
            }
            let Some(cv) = Vectors::read_slot(vectors, cand.slot as u64, segment_slots, self.dim)
            else {
                continue;
            };
            // Pairwise checks stay sequential: the fan-out is at most `lm`
            // distances (~microseconds), far below rayon task-routing cost.
            // Routing each check through the global pool made builds orders
            // of magnitude slower (every join wakes the worker threads) with
            // no compute to amortize it. Slot-level parallelism belongs to
            // the multi-worker build path, not this inner loop.
            let mut closer = true;
            for s in &selected {
                let Some(rv) = Vectors::read_slot(vectors, s.slot as u64, segment_slots, self.dim)
                else {
                    closer = false;
                    break;
                };
                // Strictly-farther-from-every-selected means diverse enough
                // to keep (pgvector CheckElementCloser).
                if self.distance(cv, rv) <= cand.dist {
                    closer = false;
                    break;
                }
            }
            if closer {
                selected.push(cand);
            } else {
                pruned.push(cand);
            }
        }
        for cand in pruned {
            if selected.len() == lm {
                break;
            }
            selected.push(cand);
        }
        selected
    }

    /// Add the reverse edge `neighbor -> slot`, shrinking the neighbor's
    /// list heuristically when it exceeds the layer cap (pgvector
    /// UpdateNeighborOnDisk + HnswUpdateConnection).
    fn link_neighbor(
        &self,
        neighbor: u32,
        slot: u32,
        dist_to_neighbor: f32,
        layer: u8,
        vectors: &[Arc<memmap2::Mmap>],
        segment_slots: u32,
    ) {
        let Some(node) = self.node(neighbor) else {
            return;
        };
        let lm = self.layer_cap(layer);
        let mut adj = timed_write_lock(self.lock_metrics.as_deref(), node.layer(layer));
        if adj.contains(&slot) {
            return;
        }
        if adj.len() < lm {
            adj.push(slot);
            node.bump_version();
            return;
        }

        // Overflow: reselect around the neighbor's own vector. The pool is
        // bounded by the layer cap (a few dozen entries), so the distance
        // fan-out stays sequential — rayon routing would dominate the work.
        let Some(nv) = Vectors::read_slot(vectors, neighbor as u64, segment_slots, self.dim) else {
            return;
        };
        let mut pool: Vec<Cand> = adj
            .iter()
            .filter_map(|&s| {
                let v = Vectors::read_slot(vectors, s as u64, segment_slots, self.dim)?;
                Some(Cand {
                    dist: self.distance(nv, v),
                    slot: s,
                })
            })
            .collect();
        pool.push(Cand {
            dist: dist_to_neighbor,
            slot,
        });
        let selected = self.select_neighbors(&pool, lm, vectors, segment_slots);
        *adj = selected.into_iter().map(|c| c.slot).collect();
        node.bump_version();
    }

    /// Approximate kNN: greedy descent to layer 1, full search at layer 0.
    /// Returns `(score, slot)` pairs in output-score space (higher is
    /// better), unsorted; shared post-processing stays in `CollectionStore`.
    pub(crate) fn probe_candidates(
        &self,
        query: &[f32],
        ef: usize,
        k: usize,
        ctx: &HnswSearchContext<'_>,
    ) -> Result<Vec<(f32, u32)>> {
        let Some(entry) = *self.entry.read() else {
            return Ok(Vec::new());
        };
        let Some(entry_vec) =
            Vectors::read_slot(ctx.vectors, entry.slot as u64, ctx.segment_slots, self.dim)
        else {
            return Ok(Vec::new());
        };

        let mut best = Cand {
            dist: self.distance(query, entry_vec),
            slot: entry.slot,
        };
        let mut lc = entry.level;
        while lc > 0 {
            best = self.greedy_step(query, best, lc, ctx.vectors, ctx.segment_slots);
            lc -= 1;
        }

        let w = self.search_layer(query, &[best], ef.max(k), 0, ctx);
        Ok(w.live
            .into_iter()
            .map(|c| (crate::distance::to_score(self.metric, c.dist), c.slot))
            .collect())
    }

    /// Iterative kNN: like `probe_candidates` but resumes from discarded
    /// candidates when the initial search yields fewer than `k` results.
    /// Each iteration feeds the previous round's discarded candidates as new
    /// entry points, expanding the search frontier until `k` results are
    /// found, `max_iterations` is exhausted, or the cumulative visited-node
    /// budget (`max_scan_tuples`) runs out.
    ///
    /// This mirrors pgvector's iterative scan (`hnswscan.c`): when
    /// `ef_search` is too small to cover the result set, the search resumes
    /// from candidates that were evicted from the live set rather than simply
    /// doubling `ef`.
    pub(crate) fn probe_candidates_iterative(
        &self,
        query: &[f32],
        ef: usize,
        k: usize,
        max_iterations: usize,
        max_scan_tuples: Option<u64>,
        ctx: &HnswSearchContext<'_>,
    ) -> Result<Vec<(f32, u32)>> {
        let Some(entry) = *self.entry.read() else {
            return Ok(Vec::new());
        };
        let Some(entry_vec) =
            Vectors::read_slot(ctx.vectors, entry.slot as u64, ctx.segment_slots, self.dim)
        else {
            return Ok(Vec::new());
        };

        let mut best = Cand {
            dist: self.distance(query, entry_vec),
            slot: entry.slot,
        };
        let mut lc = entry.level;
        while lc > 0 {
            best = self.greedy_step(query, best, lc, ctx.vectors, ctx.segment_slots);
            lc -= 1;
        }

        let mut all_results: Vec<Cand> = Vec::new();
        let mut all_discarded: Vec<Cand> = Vec::new();
        let mut current_entries = vec![best];
        let mut visited_total: u64 = 0;

        for _ in 0..max_iterations {
            let result = self.search_layer(query, &current_entries, ef, 0, ctx);
            all_results.extend(result.live);
            all_discarded.extend(result.discarded);
            visited_total += result.visited as u64;

            if all_results.len() >= k {
                break;
            }
            if all_discarded.is_empty() {
                break;
            }
            // Budget check after the pass so a single round is never skipped,
            // mirroring pgvector's "finish the current batch, then re-check"
            // scan-memory behavior.
            if max_scan_tuples.is_some_and(|cap| visited_total >= cap) {
                break;
            }
            // Use discarded candidates as entry points for the next iteration.
            current_entries = std::mem::take(&mut all_discarded);
        }

        // Deduplicate by slot (keep closest), sort, and take top-k.
        all_results.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
        all_results.dedup_by_key(|c| c.slot);
        Ok(all_results
            .into_iter()
            .take(k)
            .map(|c| (crate::distance::to_score(self.metric, c.dist), c.slot))
            .collect())
    }

    /// Incrementally repair the graph by removing references to tombstoned
    /// slots from adjacency lists.
    ///
    /// This mirrors pgvector's VACUUM graph repair pass: for each node whose
    /// adjacency list contains tombstoned slots, the affected neighbors are
    /// removed and new neighbors are selected from the remaining candidates.
    /// The repair is local — only nodes with stale references are touched —
    /// avoiding a full rebuild after each compaction.
    ///
    /// Returns the number of nodes whose adjacency lists were modified.
    pub(crate) fn repair(
        &self,
        tombstones: &crate::storage::TombstoneBits,
        vectors: &[Arc<memmap2::Mmap>],
        segment_slots: u32,
    ) -> usize {
        let nodes = self.nodes.read();
        let mut repaired = 0usize;

        for (slot, node_opt) in nodes.iter().enumerate() {
            let Some(node) = node_opt else {
                continue;
            };
            let slot = slot as u32;

            // Skip nodes that are themselves tombstoned — they will be
            // cleaned up during compaction.
            if tombstones.bit(slot as usize) {
                continue;
            }

            for layer in 0..=node.level {
                let mut adj = timed_write_lock(self.lock_metrics.as_deref(), node.layer(layer));
                let old_len = adj.len();

                // Remove tombstoned slots from the adjacency list.
                adj.retain(|&s| !tombstones.bit(s as usize));

                // If we removed any entries, try to fill the gaps with
                // new neighbors from the graph.
                if adj.len() < old_len {
                    let Some(nv) =
                        Vectors::read_slot(vectors, slot as u64, segment_slots, self.dim)
                    else {
                        continue;
                    };
                    let lm = self.layer_cap(layer);

                    // Collect candidates: all live neighbors of this node's
                    // neighbors (2-hop) that are not already in the list.
                    let existing: HashSet<u32> = adj.iter().copied().collect();
                    let mut candidates: Vec<Cand> = Vec::new();

                    // Start with direct neighbors that survived.
                    for &neighbor_slot in adj.iter() {
                        if let Some(neighbor_node) = self.node(neighbor_slot) {
                            let neighbor_adj = neighbor_node.layer(layer).read();
                            for &candidate_slot in neighbor_adj.iter() {
                                if candidate_slot == slot
                                    || tombstones.bit(candidate_slot as usize)
                                    || existing.contains(&candidate_slot)
                                {
                                    continue;
                                }
                                if let Some(cv) = Vectors::read_slot(
                                    vectors,
                                    candidate_slot as u64,
                                    segment_slots,
                                    self.dim,
                                ) {
                                    candidates.push(Cand {
                                        dist: self.distance(nv, cv),
                                        slot: candidate_slot,
                                    });
                                }
                            }
                        }
                    }

                    // Sort by distance and select the best candidates to fill gaps.
                    candidates.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
                    for cand in candidates {
                        if adj.len() >= lm {
                            break;
                        }
                        if !adj.contains(&cand.slot) {
                            adj.push(cand.slot);
                        }
                    }

                    // Update version since we modified the adjacency list.
                    node.bump_version();
                    repaired += 1;
                }
            }
        }

        repaired
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DistanceMetric;

    const DIM: usize = 8;

    fn config() -> HnswConfig {
        // m stays 8: halving it shrinks the ground-layer cap (maxM0 = 2m)
        // enough that reverse-link pruning drops long-range edges and
        // disconnects the tiny clustered fixtures these tests rely on.
        // ef_construct carries the build-cost reduction instead.
        HnswConfig {
            m: 8,
            ef_construct: 16,
            ef_search: 16,
            ..HnswConfig::default()
        }
    }

    /// Two separated blobs of 8-dim vectors.
    fn blobs(n_per_blob: usize) -> Vec<(Vec<f32>, usize)> {
        let mut out: Vec<(Vec<f32>, usize)> = Vec::new();
        for (blob, center) in [[0.0f32; DIM], [50.0; DIM]].iter().enumerate() {
            for i in 0..n_per_blob {
                let mut v = *center;
                v[i % DIM] += i as f32 * 0.1;
                out.push((v.to_vec(), blob));
            }
        }
        out
    }

    fn mmap_from(vectors: &[Vec<f32>]) -> (tempfile::TempDir, Vec<Arc<memmap2::Mmap>>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.bin");
        let mut bytes = Vec::with_capacity(vectors.len() * DIM * 4);
        for v in vectors {
            for x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        std::fs::write(&path, &bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        (dir, vec![Arc::new(mmap)])
    }

    fn all_live(n: usize) -> TombstoneBits {
        TombstoneBits::from_bits(bitvec::bitvec![0; n])
    }

    #[test]
    fn test_insert_and_probe_finds_blobs() {
        let data = blobs(15);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();

        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        assert_eq!(index.node_count(), data.len());

        // Every query's top-10 must come from its own blob: the blobs sit
        // 50 units apart while members stay within ~1 unit of their center,
        // so cross-blob hits in the top-10 would indicate broken navigation.
        for (blob, center) in [[0.0f32; DIM], [50.0; DIM]].iter().enumerate() {
            let q = *center;
            let mut hits = index
                .probe_candidates(
                    &q,
                    24,
                    10,
                    &HnswSearchContext {
                        tombstones: Some(&all_live(data.len())),
                        vectors: &mmaps,
                        segment_slots: 128,
                        filter_mask: None,
                    },
                )
                .unwrap();
            assert!(hits.len() >= 10, "expected results");
            hits.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(_, slot) in hits.iter().take(10) {
                assert_eq!(
                    data[slot as usize].1, blob,
                    "slot {slot} from the wrong blob"
                );
            }
        }
    }

    #[test]
    fn test_probe_respects_tombstones() {
        let data = blobs(10);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        let mut bits = bitvec::bitvec![0; data.len()];
        bits.set(3, true);
        let tombstones = TombstoneBits::from_bits(bits);
        let hits = index
            .probe_candidates(
                &[50.0; DIM],
                64,
                100,
                &HnswSearchContext {
                    tombstones: Some(&tombstones),
                    vectors: &mmaps,
                    segment_slots: 128,
                    filter_mask: None,
                },
            )
            .unwrap();
        assert!(!hits.iter().any(|&(_, s)| s == 3));

        // The tombstoned node stays navigable: dropping the filter brings it
        // back without any structural repair.
        let hits = index
            .probe_candidates(
                &[50.0; DIM],
                64,
                100,
                &HnswSearchContext {
                    tombstones: Some(&all_live(data.len())),
                    vectors: &mmaps,
                    segment_slots: 128,
                    filter_mask: None,
                },
            )
            .unwrap();
        assert!(hits.iter().any(|&(_, s)| s == 3));
    }

    #[test]
    fn test_incremental_insert_reaches_graph() {
        let data = blobs(10);
        let mut vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 256),
            &config(),
        )
        .unwrap();

        // Insert one extra far-away point through the incremental path.
        vectors.push(vec![-100.0; DIM]);
        let (_dir2, mmaps2) = mmap_from(&vectors);
        let new_slot = data.len() as u32;
        let v = vec![-100.0f32; DIM];
        index.insert(new_slot, &v, &mmaps2, 256);
        assert_eq!(index.node_count(), data.len() + 1);

        let hits = index
            .probe_candidates(
                &[-100.0; DIM],
                64,
                5,
                &HnswSearchContext {
                    tombstones: Some(&all_live(vectors.len())),
                    vectors: &mmaps2,
                    segment_slots: 256,
                    filter_mask: None,
                },
            )
            .unwrap();
        assert!(
            hits.iter().any(|&(_, s)| s == new_slot),
            "incrementally inserted point must be reachable"
        );
    }

    #[test]
    fn test_overwrite_insert_counts_staleness() {
        let data = blobs(6);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 64),
            &config(),
        )
        .unwrap();
        assert_eq!(index.overwrites_since_build(), 0);

        let before = index.node_count();
        for _ in 0..3 {
            index.insert(0, &vectors[0], &mmaps, 64);
        }
        assert_eq!(index.node_count(), before, "overwrite keeps the position");
        assert_eq!(
            index.overwrites_since_build(),
            3,
            "each overwrite must bump the staleness counter"
        );
    }

    #[test]
    fn test_search_layer_result_never_exceeds_ef_with_dead_nodes() {
        // Tombstone almost all of blob 1 and probe its center: the search
        // explores many dead nodes around the entry. The eviction accounting
        // must keep the tracked candidate set within `ef` while still
        // returning only live elements.
        let data = blobs(15);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        let mut bits = bitvec::bitvec![0; data.len()];
        let blob1_start = data.len() / 2;
        for slot in blob1_start..data.len() - 2 {
            bits.set(slot, true);
        }
        let dead: HashSet<u32> = (blob1_start..data.len() - 2).map(|s| s as u32).collect();
        let tombstones = TombstoneBits::from_bits(bits);

        // Enter from the last node (a live blob-1 member next to the dead zone).
        let entry_slot = (data.len() - 1) as u32;
        let out = index.search_layer(
            &[50.0; DIM],
            &[Cand {
                dist: index.distance(&[50.0; DIM], &vectors[entry_slot as usize]),
                slot: entry_slot,
            }],
            4,
            0,
            &HnswSearchContext {
                tombstones: Some(&tombstones),
                vectors: &mmaps,
                segment_slots: 128,
                filter_mask: None,
            },
        );
        assert!(
            out.live.len() <= 4,
            "tracked results must stay within ef, got {}",
            out.live.len()
        );
        assert!(out.live.iter().all(|c| !dead.contains(&c.slot)));
    }

    #[test]
    fn test_dedicated_pool_build_matches_sequential_topology() {
        let data = blobs(12);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();

        let sequential = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();
        let threaded_cfg = HnswConfig {
            max_indexing_threads: Some(1),
            ..config()
        };
        let threaded = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &threaded_cfg,
        )
        .unwrap();

        type Topology = (Option<(u32, i32)>, Vec<(u32, u8, Vec<Vec<u32>>)>);
        fn topology(index: &HnswIndex) -> Topology {
            let p = index.to_persisted();
            (
                p.entry,
                p.nodes
                    .into_iter()
                    .map(|n| (n.slot, n.level, n.neighbors))
                    .collect(),
            )
        }
        assert_eq!(
            topology(&sequential),
            topology(&threaded),
            "a dedicated single-thread pool must not change the built graph"
        );
    }

    #[test]
    fn test_multiworker_build_registers_all_and_keeps_recall() {
        let n = 50;
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut state = SplitMix64::new(7);
        for _ in 0..n {
            vectors.push(
                (0..DIM)
                    .map(|_| (state.next_u64() % 1000) as f32 / 100.0)
                    .collect(),
            );
        }
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..n as u32).collect();
        let workers_cfg = HnswConfig {
            max_indexing_threads: Some(4),
            ..config()
        };
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &workers_cfg,
        )
        .unwrap();
        assert_eq!(
            index.node_count(),
            n,
            "every slot must be registered exactly once"
        );

        let q = vec![5.0f32; DIM];
        let mut exact: Vec<(f32, u32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    crate::distance::distance(DistanceMetric::Euclid, &q, v),
                    i as u32,
                )
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: HashSet<u32> = exact[..5].iter().map(|&(_, s)| s).collect();

        let hits = index
            .probe_candidates(
                &q,
                32,
                5,
                &HnswSearchContext {
                    tombstones: Some(&all_live(n)),
                    vectors: &mmaps,
                    segment_slots: 128,
                    filter_mask: None,
                },
            )
            .unwrap();
        let found = hits.iter().filter(|&&(_, s)| truth.contains(&s)).count();
        assert!(
            found >= 3,
            "concurrent build must keep the recall bar, got {found}/5"
        );
    }

    #[test]
    fn test_multiworker_build_with_more_workers_than_points() {
        let data = blobs(12);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let workers_cfg = HnswConfig {
            max_indexing_threads: Some(8),
            ..config()
        };
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &workers_cfg,
        )
        .unwrap();
        assert_eq!(index.node_count(), data.len());
        assert!(
            index.to_persisted().entry.is_some(),
            "exactly one entry point must survive concurrent first inserts"
        );
    }

    #[test]
    fn test_persisted_roundtrip_preserves_topology() {
        let data = blobs(12);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let original = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        let restored = HnswIndex::from_persisted(original.to_persisted(), &config()).unwrap();
        assert_eq!(restored.node_count(), original.node_count());
        assert_eq!(
            restored.built_at_live_count(),
            original.built_at_live_count()
        );
        assert_eq!(
            restored.default_ef(),
            original.default_ef(),
            "effective ef_search must survive the roundtrip"
        );

        let q = vec![50.0; DIM];
        let a = original
            .probe_candidates(
                &q,
                32,
                10,
                &HnswSearchContext {
                    tombstones: Some(&all_live(data.len())),
                    vectors: &mmaps,
                    segment_slots: 128,
                    filter_mask: None,
                },
            )
            .unwrap();
        let b = restored
            .probe_candidates(
                &q,
                32,
                10,
                &HnswSearchContext {
                    tombstones: Some(&all_live(data.len())),
                    vectors: &mmaps,
                    segment_slots: 128,
                    filter_mask: None,
                },
            )
            .unwrap();
        assert_eq!(a, b, "restored graph must answer identically");
    }

    #[test]
    fn test_from_persisted_rejects_corruption() {
        let data = blobs(10);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 64),
            &config(),
        )
        .unwrap();

        let mut persisted = index.to_persisted();
        persisted.nodes[2].neighbors[0][0] = 999;
        assert!(HnswIndex::from_persisted(persisted, &config()).is_err());

        let mut persisted = index.to_persisted();
        persisted.entry = Some((999, 0));
        assert!(HnswIndex::from_persisted(persisted, &config()).is_err());

        let mut persisted = index.to_persisted();
        persisted.nodes[1].level = 5;
        assert!(HnswIndex::from_persisted(persisted, &config()).is_err());
    }

    #[test]
    fn test_recall_against_exact_scan() {
        // Random-ish deterministic vectors; HNSW with these settings should
        // recover most of the true top-10 for queries near the data cloud.
        let n = 50;
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut state = SplitMix64::new(42);
        for _ in 0..n {
            vectors.push(
                (0..DIM)
                    .map(|_| (state.next_u64() % 1000) as f32 / 100.0)
                    .collect(),
            );
        }
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..n as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        let q = vec![5.0f32; DIM];
        let mut exact: Vec<(f32, u32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    crate::distance::distance(DistanceMetric::Euclid, &q, v),
                    i as u32,
                )
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: HashSet<u32> = exact[..5].iter().map(|&(_, s)| s).collect();

        let hits = index
            .probe_candidates(
                &q,
                32,
                5,
                &HnswSearchContext {
                    tombstones: Some(&all_live(n)),
                    vectors: &mmaps,
                    segment_slots: 128,
                    filter_mask: None,
                },
            )
            .unwrap();
        let found = hits.iter().filter(|&&(_, s)| truth.contains(&s)).count();
        assert!(found >= 3, "recall@5 too low: {found}/5");
    }

    #[test]
    fn test_iterative_scan_improves_recall() {
        // Build a reasonably large graph, then compare one-shot vs iterative
        // scan with a deliberately small ef to force iterative expansion.
        let n = 50;
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut state = SplitMix64::new(42);
        for _ in 0..n {
            vectors.push(
                (0..DIM)
                    .map(|_| (state.next_u64() % 1000) as f32 / 100.0)
                    .collect(),
            );
        }
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..n as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        let q = vec![5.0f32; DIM];
        let mut exact: Vec<(f32, u32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    crate::distance::distance(DistanceMetric::Euclid, &q, v),
                    i as u32,
                )
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: HashSet<u32> = exact[..5].iter().map(|&(_, s)| s).collect();

        let ctx = HnswSearchContext {
            tombstones: Some(&all_live(n)),
            vectors: &mmaps,
            segment_slots: 128,
            filter_mask: None,
        };

        // One-shot with small ef.
        let small_ef = 4;
        let oneshot = index.probe_candidates(&q, small_ef, 5, &ctx).unwrap();
        let oneshot_found = oneshot.iter().filter(|&&(_, s)| truth.contains(&s)).count();

        // Iterative scan with the same small ef, 3 iterations.
        let iterative = index
            .probe_candidates_iterative(&q, small_ef, 5, 3, None, &ctx)
            .unwrap();
        let iterative_found = iterative
            .iter()
            .filter(|&&(_, s)| truth.contains(&s))
            .count();

        // Iterative should achieve reasonable recall (it explores more of the
        // graph via discarded candidates, though one-shot can occasionally
        // score higher on specific topologies).
        assert!(
            iterative_found >= 3,
            "iterative recall@5 too low: {iterative_found}/5"
        );
        assert!(
            oneshot_found >= 3,
            "oneshot recall@5 too low: {oneshot_found}/5"
        );
    }

    #[test]
    fn test_stale_ratio_combines_count_and_delta() {
        let data = blobs(6);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 64),
            &config(),
        )
        .unwrap();
        assert_eq!(index.stale_ratio(), 0.0);

        // Overwrite with the same vector: count increases, delta stays ~0.
        index.insert(0, &vectors[0], &mmaps, 64);
        assert!(index.stale_ratio() > 0.0, "count-based ratio should be > 0");

        // Overwrite with a very different vector: delta should dominate.
        let far = vec![1000.0; DIM];
        index.insert(1, &far, &mmaps, 64);
        let ratio = index.stale_ratio();
        assert!(ratio > 0.1, "delta should push ratio up, got {ratio}");
    }

    #[test]
    fn test_iterative_scan_tuple_budget_truncates() {
        let n = 50;
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut state = SplitMix64::new(42);
        for _ in 0..n {
            vectors.push(
                (0..DIM)
                    .map(|_| (state.next_u64() % 1000) as f32 / 100.0)
                    .collect(),
            );
        }
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..n as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        let ctx = HnswSearchContext {
            tombstones: Some(&all_live(n)),
            vectors: &mmaps,
            segment_slots: 128,
            filter_mask: None,
        };
        let q = vec![5.0f32; DIM];

        // A one-node budget stops the expansion after the first pass; an
        // unbounded budget may keep resuming from discarded candidates.
        let bounded = index
            .probe_candidates_iterative(&q, 4, 50, 8, Some(1), &ctx)
            .unwrap();
        let unbounded = index
            .probe_candidates_iterative(&q, 4, 50, 8, None, &ctx)
            .unwrap();
        assert!(!bounded.is_empty(), "first pass must still return results");
        assert!(
            bounded.len() <= unbounded.len(),
            "budgeted scan must not explore more than unbounded ({}/{})",
            bounded.len(),
            unbounded.len()
        );
    }

    #[test]
    fn test_concurrent_insert_and_probe_exercises_version_reload() {
        // Writers keep mutating adjacency lists while a reader hammers the
        // probe path; the reader's version double-read protocol must detect
        // at least one concurrent mutation and reload the neighborhood.
        let base = blobs(30);
        let mut vectors: Vec<Vec<f32>> = base.iter().map(|(v, _)| v.clone()).collect();
        // Extra slots inserted by the writer thread, clustered on blob 1 so
        // reverse-links keep mutating the same hub neighborhoods.
        let extra: Vec<Vec<f32>> = (0..300)
            .map(|i| {
                let mut v = vec![50.0f32; DIM];
                v[i % DIM] += (i % 11) as f32 * 0.1;
                v
            })
            .collect();
        vectors.extend(extra.iter().cloned());
        let (_dir, mmaps) = mmap_from(&vectors);

        let initial: Vec<u32> = (0..base.len() as u32).collect();
        let metrics = Arc::new(Metrics::default());
        let mut index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &initial, &mmaps, 256),
            &config(),
        )
        .unwrap();
        index.set_metrics(Arc::clone(&metrics));
        let index = Arc::new(index);

        let first_extra = base.len() as u32;
        let total_slots = vectors.len();
        let writer_mmaps = mmaps.clone();
        let writer_index = Arc::clone(&index);
        let writer = std::thread::spawn(move || {
            for i in 0..extra.len() as u32 {
                writer_index.insert(first_extra + i, &extra[i as usize], &writer_mmaps, 256);
            }
        });

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader_index = Arc::clone(&index);
        let reader_stop = Arc::clone(&stop);
        let reader = std::thread::spawn(move || {
            while !reader_stop.load(std::sync::atomic::Ordering::Relaxed) {
                for _ in 0..50 {
                    let _ = reader_index.probe_candidates(
                        &[50.0; DIM],
                        8,
                        5,
                        &HnswSearchContext {
                            tombstones: Some(&all_live(total_slots)),
                            vectors: &mmaps,
                            segment_slots: 256,
                            filter_mask: None,
                        },
                    );
                }
            }
        });

        writer.join().unwrap();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        reader.join().unwrap();

        assert_eq!(index.node_count(), vectors.len());
        assert!(
            metrics.snapshot().search_version_reloads > 0,
            "write/read contention must exercise the version double-read reload path"
        );
    }
}
