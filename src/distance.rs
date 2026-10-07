//! Distance kernels and score conversion.
//!
//! The local engine ranks by an internal *distance* (smaller is nearer) and
//! converts it to a *similarity* score for output. That score **is** the
//! crate-wide contract: higher is better, and `score_threshold` is a lower
//! bound on it. Remote Qdrant collections with the Euclid metric return raw
//! distances; the qdrant client normalizes them back to this contract at its
//! boundary (`engine/common/distance_utils.rs`).
//!
//! | metric    | internal distance          | output score        |
//! |-----------|----------------------------|---------------------|
//! | Euclid    | `Σ(a-b)²` (squared, no sqrt) | `1/(1+sqrt(d²))`  |
//! | Dot       | `-Σ(a·b)`                  | `Σ(a·b)`            |
//! | Cosine    | `1 - similarity` clamped to `[-1,1]` | `similarity` |
//! | Manhattan | `Σ\|a-b\|`                | `1/(1+sqrt(d))`    |
//!
//! Cosine deliberately computes norms on the fly instead of normalizing at
//! insert time. `vectors.bin` is the single copy of the data and
//! `get()`/`with_vector` must return the original bytes, so pgvector's
//! dual-copy layout (raw heap tuple + normalized index copy) does not apply
//! here; the cost is one norm per distance evaluation, traded against a
//! second storage copy and an indirection on every point read.
//!
//! The active kernel is selected once per process by [`kernel::selected`]
//! (best available SIMD implementation with a naive fallback); the naive
//! path serves as the correctness baseline and is always exercised by the
//! unit tests.
//!
//! Available kernels:
//! - `Naive` (always)
//! - `Avx2+FMA` (x86-64, 8 x f32 YMM)
//! - `Avx512`  (x86-64, 16 x f32 ZMM, `avx512f`)
//! - `Neon`    (aarch64, 4 x f32 Q, `neon` baseline)
//! - `Portable` (`std::simd` 8-wide, feature `simd_portable`, any arch)

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
pub mod kernel;
pub mod naive;
pub mod neon;
pub mod portable;

use crate::types::DistanceMetric;

/// Internal distance between two vectors (smaller = nearer).
#[inline]
pub fn distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    kernel::selected().distance(metric, a, b)
}

/// Convert an internal distance to the Qdrant-compatible output score.
pub fn to_score(metric: DistanceMetric, dist: f32) -> f32 {
    match metric {
        DistanceMetric::Euclid | DistanceMetric::Manhattan => 1.0 / (1.0 + dist.sqrt()),
        DistanceMetric::Dot => -dist,
        DistanceMetric::Cosine => 1.0 - dist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DistanceMetric;

    use rand::Rng;

    fn assert_close(a: f32, b: f32) {
        assert!(
            (a - b).abs() < 1e-5,
            "expected {a} close to {b} (diff {})",
            (a - b).abs()
        );
    }

    #[test]
    fn test_known_values() {
        // L2: a=(0,0) b=(3,4) -> squared distance 25 -> score 1/(1+5) = 1/6
        let d = naive::distance(DistanceMetric::Euclid, &[0.0, 0.0], &[3.0, 4.0]);
        assert_close(d, 25.0);
        assert_close(to_score(DistanceMetric::Euclid, d), 1.0 / 6.0);

        // Manhattan: |1-4|+|2-5|+|3-6|=9 -> score 1/(1+3)=0.25
        let d = naive::distance(
            DistanceMetric::Manhattan,
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
        );
        assert_close(d, 9.0);
        assert_close(to_score(DistanceMetric::Manhattan, d), 1.0 / 4.0);

        // Dot: a=(1,2,3) b=(4,5,6) -> 32
        let d = naive::distance(DistanceMetric::Dot, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert_close(d, -32.0);
        assert_close(to_score(DistanceMetric::Dot, d), 32.0);

        // Cosine: identical unit vectors -> similarity 1 -> distance 0
        let d = naive::distance(DistanceMetric::Cosine, &[1.0, 0.0], &[1.0, 0.0]);
        assert_close(d, 0.0);
        assert_close(to_score(DistanceMetric::Cosine, d), 1.0);

        // Cosine: orthogonal -> similarity 0 -> distance 1
        let d = naive::distance(DistanceMetric::Cosine, &[1.0, 0.0], &[0.0, 1.0]);
        assert_close(d, 1.0);

        // Cosine: opposite -> similarity -1 -> distance 2 (clamped)
        let d = naive::distance(DistanceMetric::Cosine, &[1.0, 0.0], &[-1.0, 0.0]);
        assert_close(d, 2.0);
    }

    #[test]
    fn test_zero_vector_cosine_boundary() {
        // Zero norm: distance 1, score 0 (zero-vector edge).
        let d = naive::distance(DistanceMetric::Cosine, &[0.0, 0.0], &[1.0, 2.0]);
        assert_close(d, 1.0);
        assert_close(to_score(DistanceMetric::Cosine, d), 0.0);
        let d = naive::distance(DistanceMetric::Cosine, &[0.0, 0.0], &[0.0, 0.0]);
        assert_close(d, 1.0);
    }

    #[test]
    fn test_to_score_monotonicity() {
        // Score must be non-increasing as the internal distance grows.
        for metric in [
            DistanceMetric::Euclid,
            DistanceMetric::Dot,
            DistanceMetric::Cosine,
            DistanceMetric::Manhattan,
        ] {
            let mut prev = f32::INFINITY;
            for i in 0..100 {
                let dist = (i as f32) * 0.05;
                let score = to_score(metric, dist);
                assert!(
                    score <= prev + 1e-6,
                    "{metric:?}: score {score} grew from {prev} at dist {dist}"
                );
                prev = score;
            }
        }
    }

    // Helper: compare two kernels with relative tolerance (FMA rounding).
    fn assert_kernels_close(metric: DistanceMetric, dim: usize, expected: f32, got: f32) {
        let tolerance = 1e-4 * expected.abs().max(1.0);
        assert!(
            (expected - got).abs() < tolerance,
            "{metric:?} dim={dim}: expected {expected} vs got {got} (tol {tolerance})"
        );
    }

    #[test]
    fn test_naive_vs_avx2_consistency() {
        #[cfg(target_arch = "x86_64")]
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            eprintln!("avx2 not available, skipping");
            return;
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            eprintln!("not x86_64, skipping avx2 test");
            return;
        }

        let mut rng = rand::thread_rng();
        for dim in [1usize, 2, 7, 8, 15, 16, 128, 1025] {
            for _ in 0..20 {
                let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                for metric in [
                    DistanceMetric::Euclid,
                    DistanceMetric::Dot,
                    DistanceMetric::Cosine,
                    DistanceMetric::Manhattan,
                ] {
                    let expected = naive::distance(metric, &a, &b);
                    let got = unsafe { crate::distance::avx2::distance(metric, &a, &b) };
                    assert_kernels_close(metric, dim, expected, got);
                }
            }
        }

        // Zero-norm boundary
        let zero: Vec<f32> = vec![0.0; 8];
        let ones: Vec<f32> = vec![1.0; 8];
        for (a, b) in [(&zero, &ones), (&ones, &zero), (&zero, &zero)] {
            for metric in [
                DistanceMetric::Euclid,
                DistanceMetric::Dot,
                DistanceMetric::Cosine,
                DistanceMetric::Manhattan,
            ] {
                let expected = naive::distance(metric, a, b);
                let got = unsafe { crate::distance::avx2::distance(metric, a, b) };
                assert_eq!(got, expected, "{metric:?} zero-norm avx2");
            }
        }
    }

    #[test]
    fn test_naive_vs_avx512_consistency() {
        #[cfg(target_arch = "x86_64")]
        if !std::arch::is_x86_feature_detected!("avx512f") {
            eprintln!("avx512f not available, skipping");
            return;
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            eprintln!("not x86_64, skipping avx512 test");
            return;
        }

        let mut rng = rand::thread_rng();
        for dim in [1usize, 7, 8, 15, 16, 31, 32, 128, 384, 768, 1025, 1536] {
            for _ in 0..20 {
                let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                for metric in [
                    DistanceMetric::Euclid,
                    DistanceMetric::Dot,
                    DistanceMetric::Cosine,
                    DistanceMetric::Manhattan,
                ] {
                    let expected = naive::distance(metric, &a, &b);
                    let got = unsafe { crate::distance::avx512::distance(metric, &a, &b) };
                    assert_kernels_close(metric, dim, expected, got);
                }
            }
        }

        let zero: Vec<f32> = vec![0.0; 16];
        let ones: Vec<f32> = vec![1.0; 16];
        for (a, b) in [(&zero, &ones), (&ones, &zero), (&zero, &zero)] {
            for metric in [
                DistanceMetric::Euclid,
                DistanceMetric::Dot,
                DistanceMetric::Cosine,
                DistanceMetric::Manhattan,
            ] {
                let expected = naive::distance(metric, a, b);
                let got = unsafe { crate::distance::avx512::distance(metric, a, b) };
                assert_eq!(got, expected, "{metric:?} zero-norm avx512");
            }
        }
    }

    #[test]
    fn test_naive_vs_neon_consistency() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("neon not available, skipping");
            return;
        }
        // On x86_64 the neon module is a naive wrapper, so we still verify
        // it matches naive exactly (fallback path).
        let mut rng = rand::thread_rng();
        for dim in [1usize, 7, 8, 15, 16, 128, 1025] {
            for _ in 0..20 {
                let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                for metric in [
                    DistanceMetric::Euclid,
                    DistanceMetric::Dot,
                    DistanceMetric::Cosine,
                    DistanceMetric::Manhattan,
                ] {
                    let expected = naive::distance(metric, &a, &b);
                    let got = unsafe { crate::distance::neon::distance(metric, &a, &b) };
                    assert_kernels_close(metric, dim, expected, got);
                }
            }
        }
        let zero: Vec<f32> = vec![0.0; 4];
        let ones: Vec<f32> = vec![1.0; 4];
        for (a, b) in [(&zero, &ones), (&ones, &zero), (&zero, &zero)] {
            for metric in [
                DistanceMetric::Euclid,
                DistanceMetric::Dot,
                DistanceMetric::Cosine,
                DistanceMetric::Manhattan,
            ] {
                let expected = naive::distance(metric, a, b);
                let got = unsafe { crate::distance::neon::distance(metric, a, b) };
                assert_eq!(got, expected, "{metric:?} zero-norm neon");
            }
        }
    }

    #[test]
    fn test_naive_vs_portable_consistency() {
        // Portable is feature-gated; without the feature it delegates to naive
        // and is trivially consistent. With the feature we verify 1e-4.
        let mut rng = rand::thread_rng();
        for dim in [1usize, 7, 8, 15, 16, 128, 384, 1025] {
            for _ in 0..20 {
                let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                for metric in [
                    DistanceMetric::Euclid,
                    DistanceMetric::Dot,
                    DistanceMetric::Cosine,
                    DistanceMetric::Manhattan,
                ] {
                    let expected = naive::distance(metric, &a, &b);
                    let got = crate::distance::portable::distance(metric, &a, &b);
                    assert_kernels_close(metric, dim, expected, got);
                }
            }
        }

        let zero: Vec<f32> = vec![0.0; 8];
        let ones: Vec<f32> = vec![1.0; 8];
        for (a, b) in [(&zero, &ones), (&ones, &zero), (&zero, &zero)] {
            for metric in [
                DistanceMetric::Euclid,
                DistanceMetric::Dot,
                DistanceMetric::Cosine,
                DistanceMetric::Manhattan,
            ] {
                let expected = naive::distance(metric, a, b);
                let got = crate::distance::portable::distance(metric, a, b);
                assert_eq!(got, expected, "{metric:?} zero-norm portable");
            }
        }
    }

    #[test]
    fn test_all_kernels_agree() {
        // Cross-compare every kernel exposed by `kernel::all_kernels` against
        // naive on a few representative dimensions. This verifies that
        // naive vs avx2 vs avx512 vs neon agree within 1e-4.
        let mut rng = rand::thread_rng();
        let kernels = crate::distance::kernel::all_kernels();
        // Filter to available kernels only for the comparison.
        let available: Vec<_> = kernels.into_iter().filter(|k| k.is_available()).collect();
        if available.is_empty() {
            eprintln!("no kernels available, skipping");
            return;
        }
        for dim in [8usize, 31, 128, 1536] {
            let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            for metric in [
                DistanceMetric::Euclid,
                DistanceMetric::Dot,
                DistanceMetric::Cosine,
                DistanceMetric::Manhattan,
            ] {
                let expected = naive::distance(metric, &a, &b);
                for k in &available {
                    let got = k.distance(metric, &a, &b);
                    assert_kernels_close(metric, dim, expected, got);
                }
            }
        }
    }

    #[test]
    fn test_selected_kernel() {
        use crate::distance::kernel::Kernel;

        // Determine expected best-available without relying on cached SELECTED.
        let mut expected = Kernel::Naive;
        for k in crate::distance::kernel::all_kernels() {
            if k.is_available() {
                expected = k;
                break;
            }
        }
        assert_eq!(kernel::selected(), expected);

        // Pinning the naive kernel must produce identical results through
        // the dispatch path.
        kernel::force_for_test(Kernel::Naive);
        assert_eq!(kernel::selected(), Kernel::Naive);
        let a = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let b = [0.8f32, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        for metric in [
            DistanceMetric::Euclid,
            DistanceMetric::Dot,
            DistanceMetric::Cosine,
            DistanceMetric::Manhattan,
        ] {
            assert_eq!(distance(metric, &a, &b), naive::distance(metric, &a, &b));
        }
        // Also verify portable pinning when feature is enabled.
        #[cfg(feature = "simd_portable")]
        {
            kernel::force_for_test(Kernel::Portable);
            assert_eq!(kernel::selected(), Kernel::Portable);
            for metric in [
                DistanceMetric::Euclid,
                DistanceMetric::Dot,
                DistanceMetric::Cosine,
                DistanceMetric::Manhattan,
            ] {
                let got = distance(metric, &a, &b);
                let expected = naive::distance(metric, &a, &b);
                let tolerance = 1e-4 * expected.abs().max(1.0);
                assert!(
                    (got - expected).abs() < tolerance,
                    "portable vs naive {metric:?}: {got} vs {expected}"
                );
            }
        }
    }
}
