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
mod tests {
    use super::*;
    use crate::types::DistanceMetric;

    fn config() -> IvfConfig {
        IvfConfig {
            lists: Some(4),
            min_build_points: 1,
            sample_limit: 1024,
            kmeans_max_iter: 5,
            drift_threshold: 0.10,
            drift_check_interval: 3,
            default_nprobe: 2,
            auto_promotion: true,
            max_probes: None,
        }
    }

    /// Two separated blobs of 8-dim vectors; returns (vectors, blob id).
    fn blobs(n_per_blob: usize) -> Vec<(Vec<f32>, usize)> {
        let mut out: Vec<(Vec<f32>, usize)> = Vec::new();
        for (blob, center) in [[0.0f32; 8], [50.0; 8]].iter().enumerate() {
            for i in 0..n_per_blob {
                let mut v = *center;
                v[i % 8] += i as f32 * 0.1;
                out.push((v.to_vec(), blob));
            }
        }
        out
    }

    fn mmap_from(vectors: &[Vec<f32>]) -> (tempfile::TempDir, Vec<Arc<Mmap>>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.bin");
        let mut bytes = Vec::with_capacity(vectors.len() * 8 * 4);
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

    #[test]
    fn test_assign_and_membership() {
        let data = blobs(4);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();

        let index = IvfIndex::build(
            &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &slots, &mmaps, 16),
            &config(),
        )
        .unwrap();

        assert_eq!(index.list_count(), 4);
        // Every slot is assigned exactly once across all lists.
        let total: usize = index.lists.iter().map(|l| l.read().len()).sum();
        assert_eq!(total, data.len());
        for slot in 0..data.len() as u32 {
            assert_ne!(index.slot_list.read()[slot as usize], UNASSIGNED);
        }
    }

    #[test]
    fn test_reassignment_moves_slot_between_lists() {
        let data = blobs(4);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = IvfIndex::build(
            &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &slots, &mmaps, 16),
            &config(),
        )
        .unwrap();

        // Reassigning the same slot with a vector from the other blob must
        // not duplicate it in two lists.
        let other_blob_vector = vec![50.0; 8];
        let before_total: usize = index.lists.iter().map(|l| l.read().len()).sum();
        index.assign_slot(0, &other_blob_vector);
        let after_total: usize = index.lists.iter().map(|l| l.read().len()).sum();
        assert_eq!(before_total, after_total);
        let occurrences: usize = index
            .lists
            .iter()
            .map(|l| l.read().iter().filter(|&&s| s == 0).count())
            .sum();
        assert_eq!(occurrences, 1);
    }

    #[test]
    fn test_probe_candidates_skips_tombstones_and_scores_all() {
        let data = blobs(4);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = IvfIndex::build(
            &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &slots, &mmaps, 16),
            &config(),
        )
        .unwrap();

        let mut bits = bitvec::bitvec![0; data.len()];
        bits.set(0, true);
        let tombstones = crate::storage::TombstoneBits::from_bits(bits);

        let query = vec![0.0f32; 8];
        let candidates = index
            .probe_candidates(
                &query,
                4,
                &[],
                &IvfSearchContext {
                    tombstones: &tombstones,
                    vectors: &mmaps,
                    segment_slots: 16,
                    filter_mask: None,
                },
            )
            .unwrap();
        assert_eq!(candidates.len(), data.len() - 1, "tombstoned slot skipped");
        assert!(!candidates.iter().any(|&(_, s)| s == 0));

        // nprobe clamped to lists; probing everything covers every live point.
        let exact = index
            .probe_candidates(
                &query,
                99,
                &[],
                &IvfSearchContext {
                    tombstones: &tombstones,
                    vectors: &mmaps,
                    segment_slots: 16,
                    filter_mask: None,
                },
            )
            .unwrap();
        assert_eq!(exact.len(), candidates.len());
    }

    #[test]
    fn test_pending_slots_are_probed() {
        let mut vectors: Vec<Vec<f32>> = blobs(2).into_iter().map(|(v, _)| v).collect();
        // One extra mapped slot that the index does not know about yet.
        vectors.push(vec![50.0; 8]);
        let (_dir, mmaps) = mmap_from(&vectors);
        let known: Vec<u32> = (0..4).collect();
        let index = IvfIndex::build(
            &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &known, &mmaps, 16),
            &config(),
        )
        .unwrap();

        // A slot unknown to the index but passed via `extra_slots` must show
        // up in probe results.
        let extra = vec![vectors.len() as u32 - 1];
        let tombstones =
            crate::storage::TombstoneBits::from_bits(bitvec::bitvec![0; vectors.len()]);
        let candidates = index
            .probe_candidates(
                &[50.0; 8],
                4,
                &extra,
                &IvfSearchContext {
                    tombstones: &tombstones,
                    vectors: &mmaps,
                    segment_slots: 16,
                    filter_mask: None,
                },
            )
            .unwrap();
        assert!(candidates
            .iter()
            .any(|&(_, s)| s == vectors.len() as u32 - 1));

        // Rebuild the tombstone table with the extra slot marked deleted.
        let mut bits = bitvec::bitvec![0; vectors.len()];
        bits.set(vectors.len() - 1, true);
        let tombstones = crate::storage::TombstoneBits::from_bits(bits);
        let candidates = index
            .probe_candidates(
                &[50.0; 8],
                4,
                &extra,
                &IvfSearchContext {
                    tombstones: &tombstones,
                    vectors: &mmaps,
                    segment_slots: 16,
                    filter_mask: None,
                },
            )
            .unwrap();
        assert!(!candidates
            .iter()
            .any(|&(_, s)| s == vectors.len() as u32 - 1));
    }

    #[test]
    fn test_adopt_range_covers_new_slots() {
        let vectors: Vec<Vec<f32>> = blobs(2).into_iter().map(|(v, _)| v).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let first_half: Vec<u32> = (0..4).collect();
        let index = IvfIndex::build(
            &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &first_half, &mmaps, 16),
            &config(),
        )
        .unwrap();

        let tombstones =
            crate::storage::TombstoneBits::from_bits(bitvec::bitvec![0; vectors.len()]);
        index.adopt_range(4, vectors.len() as u32, &tombstones, &mmaps, 16);
        let total: usize = index.lists.iter().map(|l| l.read().len()).sum();
        assert_eq!(total, vectors.len());
    }

    #[test]
    fn test_drift_ratio_detects_moved_points() {
        let vectors: Vec<Vec<f32>> = blobs(4).into_iter().map(|(v, _)| v).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..vectors.len() as u32).collect();
        let index = IvfIndex::build(
            &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &slots, &mmaps, 16),
            &config(),
        )
        .unwrap();

        let samples = IvfIndex::sample_plan(1024, vectors.len() as u32);
        let tombstones =
            crate::storage::TombstoneBits::from_bits(bitvec::bitvec![0; vectors.len()]);
        let clean = index.drift_ratio(&samples, &tombstones, &mmaps, 16);
        assert_eq!(clean, 0.0, "assignments straight after build have no drift");

        // Re-point half the slots as if their vectors had moved to a
        // brand-new region far outside the trained clusters: the label then
        // refers to a centroid that is nowhere near the stored vector, so the
        // measured mean distance explodes relative to the baseline.
        let far = [500.0f32; 8];
        for slot in 0..vectors.len() / 2 {
            index.assign_slot(slot as u32, &far);
        }
        let dirty = index.drift_ratio(&samples, &tombstones, &mmaps, 16);
        assert!(dirty > 1.0, "expected substantial drift, got {dirty}");
    }

    #[test]
    fn test_sample_plan_stride() {
        assert!(IvfIndex::sample_plan(100, 0).is_empty());
        assert_eq!(IvfIndex::sample_plan(100, 10).len(), 10);
        let sampled = IvfIndex::sample_plan(4, 100);
        assert!(sampled.len() <= 4);
        assert!(sampled.iter().all(|&s| s < 100));
    }

    #[test]
    fn test_should_check_drift_resets_counter() {
        let data = blobs(2);
        let vectors: Vec<Vec<f32>> = data.into_iter().map(|(v, _)| v).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..vectors.len() as u32).collect();
        let index = IvfIndex::build(
            &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &slots, &mmaps, 16),
            &config(),
        )
        .unwrap();

        assert!(!index.should_check_drift());
        for _ in 0..3 {
            index.note_upsert();
        }
        assert!(index.should_check_drift());
        // Counter was reset by the swap above.
        assert!(!index.should_check_drift());
    }

    #[test]
    fn test_persisted_roundtrip_preserves_membership() {
        let data = blobs(4);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let original = IvfIndex::build(
            &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &slots, &mmaps, 16),
            &config(),
        )
        .unwrap();

        let restored = IvfIndex::from_persisted(original.to_persisted(), config());
        assert_eq!(
            restored.baseline_mean_dist, original.baseline_mean_dist,
            "drift baseline must survive the roundtrip"
        );
        let mut a: Vec<(u32, u32)> = (0..data.len() as u32)
            .map(|s| (s, original.slot_list.read()[s as usize]))
            .collect();
        let mut b: Vec<(u32, u32)> = (0..data.len() as u32)
            .map(|s| (s, restored.slot_list.read()[s as usize]))
            .collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }
}
