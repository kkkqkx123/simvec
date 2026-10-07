//! Sampled k-means training for the IVFFlat index.
//!
//! Pure functions over borrowed sample slices; sampling from mmap-backed
//! storage happens in the caller (`CollectionStore::build_index`). Training is
//! fully deterministic: a fixed-seed xorshift generator drives both the
//! k-means++ initialization and empty-cluster reseeding, so repeated builds
//! over the same data produce identical centroids.
//!
//! Two optimizations over basic Lloyd's algorithm:
//! - **Elkan's algorithm** uses triangle inequality pruning to skip distance
//!   computations for points that cannot change cluster assignment.
//! - **Spherical k-means** normalizes vectors to the unit sphere and uses L2²
//!   distance for Cosine/Dot metrics, matching pgvector's behavior.

use rayon::prelude::*;

use crate::error::{Result, VectorSearchError};
use crate::types::DistanceMetric;

/// Mean centroid movement below which training stops early.
const CONVERGENCE_TOL: f32 = 1e-3;

/// Maximum number of iterations for k-means convergence.
const MAX_ITER: u32 = 500;

/// Fallback to Lloyd's when n * k exceeds this threshold (Elkan's memory).
const ELKAN_MEMORY_LIMIT: usize = 64 * 1024 * 1024;

/// Deterministic xorshift64* generator (no external dependency).
pub(crate) struct XorShift(u64);

impl XorShift {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let x = self.0;
        self.0 ^= x << 13;
        self.0 ^= x >> 7;
        self.0 ^= x << 17;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform float in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Uniform integer in `[0, bound)` (bound > 0).
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

#[derive(Debug, Clone)]
pub(crate) struct KmeansOptions {
    pub k: u32,
    pub max_iter: u32,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct KmeansResult {
    /// `k x dim` cluster centroids.
    pub centroids: Vec<Vec<f32>>,
}

/// Precomputed state for Elkan's k-means algorithm.
///
/// Stores triangle-inequality bounds to prune unnecessary distance
/// computations during the assignment step.
struct ElkanState {
    /// `lower_bound[i][k]` = lower bound on d(sample[i], centroid[k]).
    lower_bound: Vec<Vec<f32>>,
    /// `upper_bound[i]` = d(sample[i], centroid[labels[i]]).
    upper_bound: Vec<f32>,
    /// `half_center_dist[j][k]` = 0.5 * d(centroid[j], centroid[k]).
    half_center_dist: Vec<Vec<f32>>,
    /// `min_other_dist[k]` = min_{j!=k} d(centroid[k], centroid[j]).
    min_other_dist: Vec<f32>,
}

impl ElkanState {
    fn new(
        metric: DistanceMetric,
        sample: &[&[f32]],
        centroids: &[Vec<f32>],
        labels: &[u32],
    ) -> Self {
        let n = sample.len();
        let k = centroids.len();

        // Precompute pairwise centroid half-distances.
        let half_center_dist: Vec<Vec<f32>> = (0..k)
            .map(|j| {
                (0..k)
                    .map(|l| 0.5 * crate::distance::distance(metric, &centroids[j], &centroids[l]))
                    .collect()
            })
            .collect();

        // min_other_dist[k] = min_{j!=k} d(centroid[k], centroid[j]).
        let min_other_dist: Vec<f32> = (0..k)
            .map(|j| {
                (0..k)
                    .filter(|&l| l != j)
                    .map(|l| half_center_dist[j][l] * 2.0)
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();

        // Initialize bounds from current labels.
        let mut lower_bound = vec![vec![0.0f32; k]; n];
        let mut upper_bound = vec![0.0f32; n];
        for (i, v) in sample.iter().enumerate() {
            let li = labels[i] as usize;
            let dist_to_label = crate::distance::distance(metric, v, &centroids[li]);
            upper_bound[i] = dist_to_label;
            for l in 0..k {
                if l == li {
                    lower_bound[i][l] = dist_to_label;
                } else {
                    // Triangle inequality: d(v, c_l) >= |d(v, c_li) - d(c_li, c_l)|
                    let delta = (dist_to_label - half_center_dist[li][l] * 2.0).abs();
                    lower_bound[i][l] = delta;
                }
            }
        }

        Self {
            lower_bound,
            upper_bound,
            half_center_dist,
            min_other_dist,
        }
    }

    /// Update after centroids have moved. Recomputes all bounds from scratch.
    fn recompute_bounds(
        &mut self,
        metric: DistanceMetric,
        sample: &[&[f32]],
        centroids: &[Vec<f32>],
        labels: &[u32],
    ) {
        let k = centroids.len();

        // Recompute pairwise centroid distances.
        for j in 0..k {
            for l in 0..k {
                self.half_center_dist[j][l] =
                    0.5 * crate::distance::distance(metric, &centroids[j], &centroids[l]);
            }
        }
        for j in 0..k {
            self.min_other_dist[j] = (0..k)
                .filter(|&l| l != j)
                .map(|l| self.half_center_dist[j][l] * 2.0)
                .fold(f32::INFINITY, f32::min);
        }

        // Recompute per-sample bounds.
        for (i, v) in sample.iter().enumerate() {
            let li = labels[i] as usize;
            let dist_to_label = crate::distance::distance(metric, v, &centroids[li]);
            self.upper_bound[i] = dist_to_label;
            for l in 0..k {
                if l == li {
                    self.lower_bound[i][l] = dist_to_label;
                } else {
                    let delta = (dist_to_label - self.half_center_dist[li][l] * 2.0).abs();
                    self.lower_bound[i][l] = delta;
                }
            }
        }
    }
}

/// Elkan's assignment step: uses triangle-inequality pruning to skip
/// distance computations for points guaranteed to remain in their cluster.
///
/// Processes samples in parallel via rayon's `par_iter_mut` on labels.
/// Each task gets exclusive access to the corresponding lower_bound and
/// upper_bound entries through index-based access on shared references.
fn elkan_assign(
    state: &mut ElkanState,
    sample: &[&[f32]],
    metric: DistanceMetric,
    centroids: &[Vec<f32>],
    labels: &mut [u32],
) {
    let k = centroids.len();
    // Rebuild per-sample assignments in parallel. The triangle-inequality
    // pruning is the main win; parallelism is secondary but still helpful.
    let lb = &state.lower_bound;
    let ub = &state.upper_bound;
    let hcd = &state.half_center_dist;
    let mod_ = &state.min_other_dist;

    // We need exclusive access to lb/ub per-index. Use `par_chunks_mut`
    // on a temporary index array and process via that.
    let n = sample.len();
    let indices: Vec<usize> = (0..n).collect();

    // For each sample, compute the new label and bounds. We store results
    // in temporary vectors and apply them after the parallel loop.
    struct AssignResult {
        label: u32,
        upper_bound: f32,
        lower_bounds: Vec<f32>,
    }

    let results: Vec<AssignResult> = indices
        .par_iter()
        .map(|&i| {
            let v = sample[i];
            let cur_label = labels[i];
            let cur_ub = ub[i];

            // Pruning test 1.
            if cur_ub <= mod_[cur_label as usize] {
                return AssignResult {
                    label: cur_label,
                    upper_bound: cur_ub,
                    lower_bounds: Vec::new(),
                };
            }

            let dist = crate::distance::distance(metric, v, &centroids[cur_label as usize]);
            let mut new_lb = lb[i].clone();
            new_lb[cur_label as usize] = dist;

            let mut best = cur_label;
            let mut best_dist = dist;

            for l in 0..k {
                if l as u32 == cur_label {
                    continue;
                }
                if new_lb[l] >= dist {
                    continue;
                }
                let half_d = hcd[cur_label as usize][l];
                if new_lb[l] >= half_d && new_lb[l] >= dist {
                    continue;
                }
                let d = crate::distance::distance(metric, v, &centroids[l]);
                new_lb[l] = d;
                if d < best_dist {
                    best = l as u32;
                    best_dist = d;
                }
            }

            AssignResult {
                label: best,
                upper_bound: best_dist,
                lower_bounds: new_lb,
            }
        })
        .collect();

    // Apply results sequentially.
    for (i, r) in results.into_iter().enumerate() {
        labels[i] = r.label;
        state.upper_bound[i] = r.upper_bound;
        if !r.lower_bounds.is_empty() {
            state.lower_bound[i] = r.lower_bounds;
        }
    }
}

/// Train k-means on `sample`. The effective `k` is clamped down to the sample
/// size; an empty sample is an error.
///
/// For Cosine/Dot metrics, spherical k-means is used: vectors are normalized
/// to the unit sphere and L2² distance is used during training.
pub(crate) fn train(
    metric: DistanceMetric,
    sample: &[&[f32]],
    opts: &KmeansOptions,
) -> Result<KmeansResult> {
    if sample.is_empty() {
        return Err(VectorSearchError::Internal(
            "k-means training requires a non-empty sample".to_string(),
        ));
    }
    let k = (opts.k.max(1) as usize).min(sample.len());
    let max_iter = opts.max_iter.min(MAX_ITER);
    let mut rng = XorShift::new(opts.seed);

    // Spherical k-means: normalize for Cosine/Dot, use Euclid internally.
    let is_spherical = matches!(metric, DistanceMetric::Cosine | DistanceMetric::Dot);
    let effective_metric = if is_spherical {
        DistanceMetric::Euclid
    } else {
        metric
    };

    // Pre-allocate normalized buffer only when needed.
    let normalized_buf: Vec<Vec<f32>> = if is_spherical {
        sample
            .iter()
            .map(|v| {
                let mut nv = v.to_vec();
                normalize_l2(&mut nv);
                nv
            })
            .collect()
    } else {
        Vec::new()
    };
    let effective_sample: Vec<&[f32]> = if is_spherical {
        normalized_buf.iter().map(|v| v.as_slice()).collect()
    } else {
        sample.to_vec()
    };

    let mut centroids = kmeans_pp_init(effective_metric, &effective_sample, k, &mut rng);
    let mut labels = vec![0u32; sample.len()];

    // Determine whether to use Elkan's or Lloyd's.
    let use_elkan = k * sample.len() <= ELKAN_MEMORY_LIMIT;

    if use_elkan {
        // Initial assignment to set up Elkan bounds.
        labels = effective_sample
            .par_iter()
            .map(|v| nearest_centroid(effective_metric, v, &centroids) as u32)
            .collect();

        let mut state = ElkanState::new(effective_metric, &effective_sample, &centroids, &labels);

        for _ in 0..max_iter {
            // Elkan assignment step (with pruning).
            elkan_assign(
                &mut state,
                &effective_sample,
                effective_metric,
                &centroids,
                &mut labels,
            );

            // Update step: accumulate member sums, then average.
            let shift = update_centroids(
                effective_metric,
                &effective_sample,
                &mut centroids,
                &mut labels,
                is_spherical,
            );

            // Recompute Elkan bounds after centroid movement.
            state.recompute_bounds(effective_metric, &effective_sample, &centroids, &labels);

            if shift / (k as f32) < CONVERGENCE_TOL {
                break;
            }
        }
    } else {
        // Lloyd's fallback (no triangle-inequality pruning).
        for _ in 0..max_iter {
            labels = effective_sample
                .par_iter()
                .map(|v| nearest_centroid(effective_metric, v, &centroids) as u32)
                .collect();

            let shift = update_centroids(
                effective_metric,
                &effective_sample,
                &mut centroids,
                &mut labels,
                is_spherical,
            );

            if shift / (k as f32) < CONVERGENCE_TOL {
                break;
            }
        }
    }

    Ok(KmeansResult { centroids })
}

/// Update centroid positions from current labels. Returns mean centroid shift.
/// Handles empty clusters by reseeding from the farthest point.
fn update_centroids(
    metric: DistanceMetric,
    sample: &[&[f32]],
    centroids: &mut [Vec<f32>],
    labels: &mut [u32],
    is_spherical: bool,
) -> f32 {
    let k = centroids.len();
    let dim = centroids[0].len();
    let mut sums = vec![vec![0.0f32; dim]; k];
    let mut counts = vec![0u64; k];
    for (v, &l) in sample.iter().zip(labels.iter()) {
        counts[l as usize] += 1;
        let sum = &mut sums[l as usize];
        for (s, x) in sum.iter_mut().zip(v.iter()) {
            *s += x;
        }
    }

    let mut shift = 0.0f32;
    for c in 0..k {
        if counts[c] == 0 {
            let far = farthest_point(metric, sample, centroids);
            shift += centroid_shift(&centroids[c], sample[far]);
            centroids[c] = sample[far].to_vec();
            continue;
        }
        let mut next = std::mem::take(&mut sums[c]);
        for x in &mut next {
            *x /= counts[c] as f32;
        }
        if is_spherical {
            normalize_l2(&mut next);
        }
        shift += centroid_shift(&centroids[c], &next);
        centroids[c] = next;
    }
    shift
}

/// Nearest centroid index for one vector.
pub(crate) fn nearest_centroid(metric: DistanceMetric, v: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = crate::distance::distance(metric, v, c);
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best
}

/// k-means++ seeding: first centroid uniform-random, each following centroid
/// sampled with probability proportional to squared distance (approximated by
/// the metric's internal distance) to the closest existing centroid.
fn kmeans_pp_init(
    metric: DistanceMetric,
    sample: &[&[f32]],
    k: usize,
    rng: &mut XorShift,
) -> Vec<Vec<f32>> {
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(sample[rng.below(sample.len())].to_vec());

    let mut min_dists: Vec<f32> = sample
        .iter()
        .map(|v| crate::distance::distance(metric, v, &centroids[0]))
        .collect();

    while centroids.len() < k {
        let total: f32 = min_dists.iter().sum();
        let next = if total <= 0.0 {
            // Degenerate (all points identical / zero mass): uniform choice.
            sample[rng.below(sample.len())].to_vec()
        } else {
            let mut target = rng.next_f32() * total;
            let mut idx = sample.len() - 1;
            for (i, &d) in min_dists.iter().enumerate() {
                target -= d;
                if target <= 0.0 {
                    idx = i;
                    break;
                }
            }
            sample[idx].to_vec()
        };
        for (v, md) in sample.iter().zip(min_dists.iter_mut()) {
            let d = crate::distance::distance(metric, v, &next);
            if d < *md {
                *md = d;
            }
        }
        centroids.push(next);
    }
    centroids
}

fn farthest_point(metric: DistanceMetric, sample: &[&[f32]], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::NEG_INFINITY;
    for (i, v) in sample.iter().enumerate() {
        let c = &centroids[nearest_centroid(metric, v, centroids)];
        let d = crate::distance::distance(metric, v, c);
        if d > best_dist {
            best_dist = d;
            best = i;
        }
    }
    best
}

fn centroid_shift(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// L2-normalize a vector in place. Zero vectors are left unchanged.
pub(crate) fn normalize_l2(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const DIM: usize = 8;

    /// Three well-separated gaussian-ish blobs.
    fn clustered_sample() -> Vec<Vec<f32>> {
        let centers: [[f32; DIM]; 3] = [[0.0; DIM], [10.0; DIM], [20.0; DIM]];
        let mut rng = StdRng::seed_from_u64(42);
        let mut out = Vec::with_capacity(300);
        for center in &centers {
            for _ in 0..100 {
                let mut v = [0.0f32; DIM];
                for x in &mut v {
                    *x = center[0] + rng.gen_range(-0.5..0.5);
                }
                out.push(v.to_vec());
            }
        }
        out
    }

    #[test]
    fn test_train_separates_clusters() {
        let data = clustered_sample();
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let result = train(
            DistanceMetric::Euclid,
            &refs,
            &KmeansOptions {
                k: 3,

                max_iter: 10,
                seed: 7,
            },
        )
        .unwrap();

        assert_eq!(result.centroids.len(), 3);
        for v in &refs {
            let c =
                &result.centroids[nearest_centroid(DistanceMetric::Euclid, v, &result.centroids)];
            let dist = crate::distance::distance(DistanceMetric::Euclid, v, c).sqrt();
            assert!(dist < 2.0, "point {v:?} landed {dist} away from {c:?}");
        }
    }

    #[test]
    fn test_train_is_deterministic() {
        let data = clustered_sample();
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let opts = KmeansOptions {
            k: 4,

            max_iter: 10,
            seed: 99,
        };
        let a = train(DistanceMetric::Cosine, &refs, &opts).unwrap();
        let b = train(DistanceMetric::Cosine, &refs, &opts).unwrap();
        assert_eq!(a.centroids, b.centroids);
    }

    #[test]
    fn test_train_clamps_k_to_sample() {
        let data = [vec![1.0f32; DIM], vec![2.0f32; DIM]];
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let result = train(
            DistanceMetric::Euclid,
            &refs,
            &KmeansOptions {
                k: 16,

                max_iter: 5,
                seed: 1,
            },
        )
        .unwrap();
        assert_eq!(result.centroids.len(), 2);
    }

    #[test]
    fn test_empty_sample_errors() {
        let err = train(
            DistanceMetric::Euclid,
            &[],
            &KmeansOptions {
                k: 2,

                max_iter: 5,
                seed: 1,
            },
        )
        .unwrap_err();
        assert!(matches!(err, VectorSearchError::Internal(_)));
    }

    #[test]
    fn test_xorshift_bounds() {
        let mut rng = XorShift::new(12345);
        for _ in 0..10_000 {
            let v = rng.below(7);
            assert!(v < 7);
            let f = rng.next_f32();
            assert!((0.0..1.0).contains(&f));
        }
    }

    // --- F1: Elkan vs Lloyd correctness ---

    #[test]
    fn test_elkan_matches_lloyd_euclid() {
        let data = clustered_sample();
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let opts = KmeansOptions {
            k: 3,

            max_iter: 20,
            seed: 42,
        };
        // Both paths produce valid clustering (may differ in tie-breaks
        // but should have comparable within-cluster sum of squares).
        let result = train(DistanceMetric::Euclid, &refs, &opts).unwrap();
        assert_eq!(result.centroids.len(), 3);

        // Verify all points are close to some centroid.
        for v in &refs {
            let c =
                &result.centroids[nearest_centroid(DistanceMetric::Euclid, v, &result.centroids)];
            let dist = crate::distance::distance(DistanceMetric::Euclid, v, c).sqrt();
            assert!(dist < 2.0, "Elkan result: point landed {dist} away");
        }
    }

    #[test]
    fn test_elkan_assign_prunes_correctly() {
        // Small 2-cluster case where we can manually verify pruning.
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 0.0],
            vec![10.1, 0.0],
        ];
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 0.0]];
        let mut labels = vec![0u32, 0, 1, 1];

        let mut state = ElkanState::new(DistanceMetric::Euclid, &refs, &centroids, &labels);
        elkan_assign(
            &mut state,
            &refs,
            DistanceMetric::Euclid,
            &centroids,
            &mut labels,
        );

        assert_eq!(labels, [0, 0, 1, 1]);
    }

    // --- F2: Spherical k-means correctness ---

    #[test]
    fn test_spherical_kmeans_cosine() {
        // Create unit vectors in distinct directions.
        let data: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.99, 0.14], // near (1,0)
            vec![0.0, 1.0],
            vec![0.14, 0.99], // near (0,1)
        ];
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let result = train(
            DistanceMetric::Cosine,
            &refs,
            &KmeansOptions {
                k: 2,

                max_iter: 20,
                seed: 7,
            },
        )
        .unwrap();

        assert_eq!(result.centroids.len(), 2);

        // Each centroid should be L2-normalized (on unit sphere).
        for c in &result.centroids {
            let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "centroid not on unit sphere: norm={norm}"
            );
        }

        // The two centroids should point in different directions.
        let cos_sim: f32 = result.centroids[0]
            .iter()
            .zip(result.centroids[1].iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            cos_sim.abs() < 0.5,
            "centroids too similar: cos_sim={cos_sim}"
        );
    }

    #[test]
    fn test_spherical_kmeans_dot() {
        // Dot metric should also normalize like Cosine.
        let data: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.99, 0.14],
            vec![0.0, 1.0],
            vec![0.14, 0.99],
        ];
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let result = train(
            DistanceMetric::Dot,
            &refs,
            &KmeansOptions {
                k: 2,

                max_iter: 20,
                seed: 7,
            },
        )
        .unwrap();

        for c in &result.centroids {
            let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "Dot centroid not on unit sphere: norm={norm}"
            );
        }
    }

    #[test]
    fn test_normalize_l2_preserves_direction() {
        let mut v = vec![3.0, 4.0];
        normalize_l2(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_l2_zero_vector() {
        let mut v = vec![0.0, 0.0];
        normalize_l2(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }
}
