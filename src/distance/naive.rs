//! Naive scalar distance kernels.
//!
//! Deliberately simple and readable: the correctness baseline that the SIMD
//! kernels are asserted against. No manual optimization.

use crate::types::DistanceMetric;

/// Naive distance for any metric.
pub fn distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Euclid => distance_l2(a, b),
        DistanceMetric::Dot => -inner_product(a, b),
        DistanceMetric::Cosine => distance_cosine(a, b),
        DistanceMetric::Manhattan => distance_l1(a, b),
    }
}

/// Squared Euclidean distance (no sqrt, aligned with pgvector
/// `VectorL2SquaredDistance`, `vector.c:560`).
pub fn distance_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Manhattan distance (rejected at collection creation; provided for
/// completeness only).
pub fn distance_l1(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// Dot product.
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Cosine distance, single loop accumulating the dot product and both norms
/// (aligned with pgvector `VectorCosineSimilarity`, `vector.c:650-662`).
///
/// Returns `1 - similarity` clamped to `[-1, 1]`; a zero-norm vector yields
/// distance 1 (score 0) per the design's zero-vector edge case.
pub fn distance_cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - (dot / denom).clamp(-1.0, 1.0)
}
