//! ANN index tiers for the local engine.
//!
//! [`HnswIndex`] is the default tier: incremental inserts, no retraining,
//! and the same algorithm/knob surface as Qdrant. [`IvfIndex`] is the bulk
//! alternative with cluster-based pruning. Both are derived structures over
//! `vectors.bin`: they can be dropped and rebuilt at any time without
//! affecting correctness.

pub(crate) mod hnsw;
pub(crate) mod kmeans;
pub(crate) mod persist;

pub(crate) use hnsw::HnswIndex;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use memmap2::Mmap;
use parking_lot::RwLock;
use rayon::prelude::*;

use crate::error::Result;
use crate::index::persist::PersistedIvf;
use crate::metrics::{timed_write_lock, Metrics};
use crate::storage::vectors::Vectors;
use crate::types::{DistanceMetric, IvfConfig};

use bitvec::prelude::*;

/// Sentinel list id for "not assigned".
pub(crate) const UNASSIGNED: u32 = u32::MAX;

/// Inputs shared by both ANN-tier build entry points.
///
/// Groups the corpus description and the observability sink so the build
/// signatures stay stable as knobs are added. `progress` is only wired by
/// the store's lifecycle paths; standalone builders (tests, tooling) leave
/// it unset via [`IndexBuildParams::new`].
pub(crate) struct IndexBuildParams<'a> {
    /// Target collection name; seeds the deterministic build RNG.
    pub collection: &'a str,
    /// Vector dimensionality; must match the stored rows.
    pub dim: usize,
    pub metric: DistanceMetric,
    /// Live slots to index (sorted ascending).
    pub slots: &'a [u32],
    /// Immutable mmap snapshot the slot rows are read from.
    pub vectors: &'a [Arc<Mmap>],
    /// Slots per `vectors.bin` segment.
    pub segment_slots: u32,
    /// Optional per-slot progress sink, incremented once per processed slot.
    pub progress: Option<&'a AtomicU64>,
}

impl<'a> IndexBuildParams<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        collection: &'a str,
        dim: usize,
        metric: DistanceMetric,
        slots: &'a [u32],
        vectors: &'a [Arc<Mmap>],
        segment_slots: u32,
    ) -> Self {
        Self {
            collection,
            dim,
            metric,
            slots,
            vectors,
            segment_slots,
            progress: None,
        }
    }

    /// Attach a per-slot progress sink for long builds.
    pub(crate) fn with_progress(mut self, progress: &'a AtomicU64) -> Self {
        self.progress = Some(progress);
        self
    }
}

/// Upper bound for reported drift ratios so stored/serialized values stay
/// finite even for degenerate baselines.
pub(crate) const DRIFT_RATIO_CAP: f64 = 1.0e6;

/// Mean distance from a sample of vectors to their nearest centroid; the
/// drift baseline recorded at build time. For Cosine/Dot metrics the
/// centroids are on the unit sphere, so samples are normalized first.
fn baseline_distance(metric: DistanceMetric, sample: &[&[f32]], centroids: &[Vec<f32>]) -> f32 {
    if sample.is_empty() || centroids.is_empty() {
        return 0.0;
    }
    let is_spherical = matches!(metric, DistanceMetric::Cosine | DistanceMetric::Dot);
    let total: f64 = sample
        .iter()
        .map(|v| {
            let q = if is_spherical {
                let mut nv = v.to_vec();
                kmeans::normalize_l2(&mut nv);
                nv
            } else {
                v.to_vec()
            };
            let nearest = kmeans::nearest_centroid(metric, &q, centroids);
            crate::distance::distance(metric, &q, &centroids[nearest]) as f64
        })
        .sum();
    (total / sample.len() as f64) as f32
}

pub(crate) struct IvfSearchContext<'a> {
    pub tombstones: &'a crate::storage::TombstoneBits,
    pub vectors: &'a [Arc<Mmap>],
    pub segment_slots: u32,
    pub filter_mask: Option<&'a BitVec>,
}

/// In-memory IVFFlat index for one collection.
pub(crate) struct IvfIndex {
    dim: usize,
    metric: DistanceMetric,
    /// Tier 1 configuration. Swappable at runtime: a rehydrated index starts
    /// with defaults and adopts the engine-supplied settings via
    /// [`IvfIndex::set_config`], keeping every reader consistent.
    config: ArcSwap<IvfConfig>,
    /// Immutable after construction.
    centroids: Vec<Vec<f32>>,
    /// Per-list slot membership.
    lists: Vec<RwLock<Vec<u32>>>,
    /// slot -> list; grows on demand; `UNASSIGNED` for unknown slots.
    slot_list: RwLock<Vec<u32>>,
    /// Mean distance from the training sample to its nearest centroid;
    /// the baseline against which drift is measured.
    baseline_mean_dist: f32,
    built_at_live_count: u64,
    upserts_since_check: AtomicU64,
    /// Per-collection metrics sink for lock-contention diagnostics,
    /// attached by the store after construction (see [`HnswIndex`]).
    lock_metrics: Option<Arc<Metrics>>,
}

impl IvfIndex {
    /// Build a fresh index from the given live slots (called off the store
    /// lock; `params.slots` must be sorted ascending). k-means training runs
    /// on rayon; `IndexBuildParams::progress`, when attached, is incremented
    /// once per assigned slot and emits milestone debug logs.
    pub(crate) fn build(params: &IndexBuildParams<'_>, config: &IvfConfig) -> Result<Self> {
        // Reservoir sampling: uniform random sample of up to `sample_limit`
        // slots, matching pgvector's sampling strategy. This is unbiased
        // regardless of slot distribution and uses O(sample_limit) memory.
        let sample_limit = config.sample_limit.max(1);
        let slots = params.slots;
        let collection = params.collection;
        let sample_slots: Vec<u32> = if slots.len() <= sample_limit {
            slots.to_vec()
        } else {
            // Deterministic reservoir sampling with a seeded PRNG.
            let mut seed = collection.len() as u64;
            for b in collection.as_bytes() {
                seed = seed.wrapping_mul(0x100_0000_01b3) ^ (*b as u64);
            }
            seed ^= slots.len() as u64;
            seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng = crate::index::kmeans::XorShift::new(seed);

            let mut reservoir: Vec<u32> = slots[..sample_limit].to_vec();
            for (i, &slot) in slots.iter().enumerate().skip(sample_limit) {
                // Reservoir sampling: replace element j with probability k/(i+1)
                let j = (rng.next_u64() as usize) % (i + 1);
                if j < sample_limit {
                    reservoir[j] = slot;
                }
            }
            reservoir
        };
        let sample: Vec<&[f32]> = sample_slots
            .iter()
            .filter_map(|&s| {
                Vectors::read_slot(params.vectors, s as u64, params.segment_slots, params.dim)
            })
            .collect();

        let mut seed = collection.len() as u64;
        for b in collection.as_bytes() {
            seed = seed.wrapping_mul(0x100_0000_01b3) ^ (*b as u64);
        }
        seed ^= slots.len() as u64;
        seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);

        let opts = kmeans::KmeansOptions {
            k: config.effective_lists(slots.len() as u64),
            max_iter: config.kmeans_max_iter.max(1),
            seed,
        };
        let trained = kmeans::train(params.metric, &sample, &opts)?;
        let baseline = baseline_distance(params.metric, &sample, &trained.centroids);

        let index = Self::from_centroids(
            config.clone(),
            params.dim,
            params.metric,
            trained.centroids,
            baseline,
            slots.len() as u64,
        );
        let total = slots.len() as u64;
        let milestone = (total / 10).max(10_000);
        for &slot in slots.iter() {
            if let Some(v) = Vectors::read_slot(
                params.vectors,
                slot as u64,
                params.segment_slots,
                params.dim,
            ) {
                index.assign_slot(slot, v);
            }
            if let Some(progress) = params.progress {
                let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if done % milestone == 0 || done == total {
                    tracing::debug!(collection = %collection, done, total, "ivf build progress");
                }
            }
        }
        Ok(index)
    }

    /// Rehydrate from persisted state. Callers validate dimension/metric
    /// coverage against collection metadata before calling this.
    pub(crate) fn from_persisted(data: PersistedIvf, config: IvfConfig) -> Self {
        let index = Self {
            dim: data.dim,
            metric: data.distance,
            lists: (0..data.centroids.len())
                .map(|_| RwLock::new(Vec::new()))
                .collect(),
            slot_list: RwLock::new(data.slot_list),
            baseline_mean_dist: data.baseline_mean_dist,
            built_at_live_count: data.built_at_live_count,
            upserts_since_check: AtomicU64::new(0),
            centroids: data.centroids,
            config: ArcSwap::from_pointee(config),
            lock_metrics: None,
        };
        {
            let slot_map = index.slot_list.read();
            for (slot, list) in slot_map.iter().enumerate() {
                if *list != UNASSIGNED {
                    index.lists[*list as usize].write().push(slot as u32);
                }
            }
        }
        index
    }

    fn from_centroids(
        config: IvfConfig,
        dim: usize,
        metric: DistanceMetric,
        centroids: Vec<Vec<f32>>,
        baseline_mean_dist: f32,
        built_at_live_count: u64,
    ) -> Self {
        let lists = centroids.len();
        Self {
            dim,
            metric,
            config: ArcSwap::from_pointee(config),
            centroids,
            lists: (0..lists).map(|_| RwLock::new(Vec::new())).collect(),
            slot_list: RwLock::new(Vec::new()),
            baseline_mean_dist,
            built_at_live_count,
            upserts_since_check: AtomicU64::new(0),
            lock_metrics: None,
        }
    }

    /// Attach the owning collection's metrics sink. Called by the store
    /// right after construction, before the index is published.
    pub(crate) fn set_metrics(&mut self, metrics: Arc<Metrics>) {
        self.lock_metrics = Some(metrics);
    }

    pub(crate) fn list_count(&self) -> usize {
        self.centroids.len()
    }

    /// Ceiling for multi-round probe widening during filtered searches:
    /// `IvfConfig::max_probes` clamped to the list count.
    pub(crate) fn max_probe_cap(&self) -> usize {
        self.config
            .load()
            .effective_max_probes(self.centroids.len())
    }

    /// Current configuration snapshot (all-scalar struct; cloning is cheap).
    pub(crate) fn config(&self) -> IvfConfig {
        self.config.load_full().as_ref().clone()
    }

    /// Adopt a new configuration at runtime. Readers pick it up on their
    /// next snapshot load, so search and drift maintenance immediately honor
    /// engine-supplied settings after restart.
    pub(crate) fn set_config(&self, config: IvfConfig) {
        self.config.store(Arc::new(config));
    }

    pub(crate) fn default_nprobe(&self) -> usize {
        let cfg = self.config.load();
        cfg.clamp_nprobe(None, self.list_count())
    }

    pub(crate) fn clamp_nprobe(&self, nprobe: Option<usize>) -> usize {
        let cfg = self.config.load();
        cfg.clamp_nprobe(nprobe, self.list_count())
    }

    pub(crate) fn built_at_live_count(&self) -> u64 {
        self.built_at_live_count
    }

    /// Snapshot of the persisted representation (for `index.bin`).
    pub(crate) fn to_persisted(&self) -> PersistedIvf {
        PersistedIvf {
            lists: self.centroids.len() as u32,
            dim: self.dim,
            distance: self.metric,
            built_at_live_count: self.built_at_live_count,
            baseline_mean_dist: self.baseline_mean_dist,
            centroids: self.centroids.clone(),
            slot_list: self.slot_list.read().clone(),
        }
    }

    /// Normalize a query vector for centroid comparison. For Cosine/Dot
    /// metrics the centroids live on the unit sphere (spherical k-means),
    /// so the query must be normalized before distance computation.
    fn normalize_query_for_probe(query: &[f32], metric: DistanceMetric) -> Vec<f32> {
        if matches!(metric, DistanceMetric::Cosine | DistanceMetric::Dot) {
            let mut q = query.to_vec();
            kmeans::normalize_l2(&mut q);
            q
        } else {
            query.to_vec()
        }
    }

    /// Assign (or reassign) one slot to its nearest centroid. Reassignments
    /// swap-remove the slot from its previous list.
    pub(crate) fn assign_slot(&self, slot: u32, vector: &[f32]) {
        let q = Self::normalize_query_for_probe(vector, self.metric);
        let new_list = kmeans::nearest_centroid(self.metric, &q, &self.centroids) as u32;
        let old_list = {
            let mut slot_map = self.slot_list.write();
            if slot as usize >= slot_map.len() {
                slot_map.resize(slot as usize + 1, UNASSIGNED);
            }
            std::mem::replace(&mut slot_map[slot as usize], new_list)
        };
        if old_list == new_list {
            return;
        }
        if old_list != UNASSIGNED {
            let mut members =
                timed_write_lock(self.lock_metrics.as_deref(), &self.lists[old_list as usize]);
            if let Some(pos) = members.iter().position(|&s| s == slot) {
                members.swap_remove(pos);
            }
        }
        timed_write_lock(self.lock_metrics.as_deref(), &self.lists[new_list as usize]).push(slot);
    }

    /// Assign every non-tombstoned slot in `start..end` (publish path: slots
    /// appended while this index was being built).
    pub(crate) fn adopt_range(
        &self,
        start: u32,
        end: u32,
        tombstones: &crate::storage::TombstoneBits,
        vectors: &[Arc<Mmap>],
        segment_slots: u32,
    ) {
        for slot in start..end {
            if tombstones.bit(slot as usize) {
                continue;
            }
            if let Some(v) = Vectors::read_slot(vectors, slot as u64, segment_slots, self.dim) {
                self.assign_slot(slot, v);
            }
        }
    }

    /// Probe search over the `nprobe` closest lists plus any unassigned
    /// slots. Returns `(score, slot)` candidates in output-score space;
    /// filtering, thresholding and top-K stay in `CollectionStore` so both
    /// search paths share identical semantics. `filter_mask` is an optional
    /// pre-filter bitmap that prunes non-matching candidates before they
    /// are scored.
    pub(crate) fn probe_candidates(
        &self,
        query: &[f32],
        nprobe: usize,
        extra_slots: &[u32],
        ctx: &IvfSearchContext<'_>,
    ) -> Result<Vec<(f32, u32)>> {
        // 1. Rank centroids, keep the closest nprobe lists.
        // Normalize query for Cosine/Dot since centroids are on unit sphere.
        let normalized = Self::normalize_query_for_probe(query, self.metric);
        let mut ranked: Vec<(f32, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (crate::distance::distance(self.metric, &normalized, c), i))
            .collect();
        ranked.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let probed = &ranked[..nprobe.min(ranked.len())];

        let matches_mask = |s: u32| {
            ctx.filter_mask
                .is_none_or(|m| (s as usize) < m.len() && m[s as usize])
        };

        // 2. Gather candidate slots (short lock hold per list).
        let mut candidates: Vec<u32> = Vec::new();
        for &(_, list) in probed {
            candidates.extend(self.lists[list].read().iter().copied());
        }
        candidates.extend_from_slice(extra_slots);

        // 3. Score live candidates in parallel.
        let scored: Vec<(f32, u32)> = candidates
            .par_iter()
            .copied()
            .filter(|&s| !ctx.tombstones.bit(s as usize) && matches_mask(s))
            .filter_map(|s| {
                let v = Vectors::read_slot(ctx.vectors, s as u64, ctx.segment_slots, self.dim)?;
                let dist = crate::distance::distance(self.metric, query, v);
                Some((crate::distance::to_score(self.metric, dist), s))
            })
            .collect();
        Ok(scored)
    }
    /// Relative growth of the mean distance from sampled live points to
    /// their assigned centroid, versus the training-time baseline. Points
    /// drifting away from stale centroids raise the current mean, so a ratio
    /// above the configured threshold means the clustering no longer reflects
    /// the data and a rebuild is warranted. Unassigned (pending) slots are
    /// ignored; they are merged into lists by the next publish.
    pub(crate) fn drift_ratio(
        &self,
        sample_slots: &[u32],
        tombstones: &crate::storage::TombstoneBits,
        vectors: &[Arc<Mmap>],
        segment_slots: u32,
    ) -> f64 {
        let is_spherical = matches!(self.metric, DistanceMetric::Cosine | DistanceMetric::Dot);
        let slot_map = self.slot_list.read();
        let mut total = 0f64;
        let mut counted = 0usize;
        for &slot in sample_slots {
            if tombstones.bit(slot as usize) {
                continue;
            }
            let Some(&list) = slot_map.get(slot as usize) else {
                continue;
            };
            if list == UNASSIGNED || list as usize >= self.centroids.len() {
                continue;
            }
            let Some(v) = Vectors::read_slot(vectors, slot as u64, segment_slots, self.dim) else {
                continue;
            };
            let q = if is_spherical {
                let mut nv = v.to_vec();
                kmeans::normalize_l2(&mut nv);
                nv
            } else {
                v.to_vec()
            };
            total +=
                crate::distance::distance(self.metric, &q, &self.centroids[list as usize]) as f64;
            counted += 1;
        }
        if counted == 0 {
            return 0.0;
        }
        let current = total / counted as f64;
        let baseline = self.baseline_mean_dist as f64;
        // Degenerate baselines (all points on centroids) cannot express
        // relative growth; cap instead of dividing by zero.
        if baseline <= 1e-9 {
            return if current <= 1e-9 {
                0.0
            } else {
                DRIFT_RATIO_CAP
            };
        }
        ((current - baseline) / baseline).clamp(0.0, DRIFT_RATIO_CAP)
    }

    /// Sample up to `sample_limit` live slots evenly across `next_slot`.
    pub(crate) fn sample_plan(sample_limit: usize, next_slot: u32) -> Vec<u32> {
        let total = next_slot as usize;
        if total == 0 {
            return Vec::new();
        }
        let stride = total.div_ceil(sample_limit.max(1));
        if stride <= 1 {
            return (0..next_slot).collect();
        }
        (0..next_slot).step_by(stride).collect()
    }

    pub(crate) fn note_upsert(&self) {
        self.upserts_since_check.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether enough upserts accumulated for another drift check; resets the
    /// counter when returning true.
    pub(crate) fn should_check_drift(&self) -> bool {
        let interval = self.config.load().drift_check_interval.max(1);
        self.upserts_since_check.swap(0, Ordering::Relaxed) >= interval
    }
}

#[cfg(test)]
#[path = "index_tests.rs"]
mod tests;
