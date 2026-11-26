use std::collections::HashMap;

use crate::kind_store::sparse_score;
use crate::types::VectorData;

/// Union the postings of `indices`, deduplicated by slot.
pub(crate) fn sparse_postings(
    space_index: &HashMap<u32, Vec<(u32, f32)>>,
    indices: &[u32],
    _live: usize,
) -> Option<Vec<u32>> {
    let mut candidates = Vec::new();
    for index in indices {
        if let Some(postings) = space_index.get(index) {
            for (slot, _) in postings {
                candidates.push(*slot);
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    Some(candidates)
}

/// Score a (query, stored) payload pair within one named space.
///
/// Scores follow the crate-wide higher-is-better contract: dot product for
/// `Dot` and its norm-normalized form for `Cosine` in sparse spaces.
pub(crate) fn score_pair(
    query: &VectorData,
    stored: &VectorData,
    metric: crate::types::DistanceMetric,
) -> f32 {
    match (query, stored) {
        (
            VectorData::Sparse {
                indices: q_idx,
                values: q_val,
            },
            VectorData::Sparse {
                indices: d_idx,
                values: d_val,
            },
        ) => sparse_score(q_idx, q_val, d_idx, d_val, metric),
        (VectorData::Dense(q), VectorData::Dense(d)) => dense_similarity(q, d, metric),
        _ => 0.0,
    }
}

/// Dense similarity within a named space: the shared SIMD distance kernel
/// remapped onto the higher-is-better contract.
fn dense_similarity(query: &[f32], stored: &[f32], metric: crate::types::DistanceMetric) -> f32 {
    let distance = crate::distance::distance(metric, query, stored);
    match metric {
        crate::types::DistanceMetric::Cosine => 1.0 - distance,
        crate::types::DistanceMetric::Dot => -distance,
        crate::types::DistanceMetric::Euclid | crate::types::DistanceMetric::Manhattan => {
            1.0 / (1.0 + distance)
        }
    }
}
