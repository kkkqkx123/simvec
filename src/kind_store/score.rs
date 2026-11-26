use std::collections::HashMap;

use crate::types::{DistanceMetric, VectorData};

/// Score a (query, stored) payload pair under the collection metric.
///
/// Scores follow the crate-wide higher-is-better contract: dot product for
/// `Dot` and its norm-normalized form for `Cosine`.
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
        (VectorData::Multi(q), VectorData::Multi(d)) => max_sim(q, d, metric),
        (VectorData::Dense(q), VectorData::Dense(d)) => dense_similarity(q, d, metric),
        _ => 0.0,
    }
}

/// Accumulate dot contributions directly from posting lists.
///
/// This avoids reconstructing and merging every candidate's sparse vector.
/// Cosine divides by the cached stored norm and the query norm.
pub(crate) fn sparse_candidate_scores(
    inverted: &HashMap<u32, Vec<(u32, f32)>>,
    norms: &HashMap<u32, f32>,
    indices: &[u32],
    values: &[f32],
    metric: DistanceMetric,
) -> Vec<(u32, f32)> {
    let mut scores: HashMap<u32, f32> = HashMap::new();
    for (index, weight) in indices.iter().zip(values.iter()) {
        if let Some(postings) = inverted.get(index) {
            for (slot, stored_weight) in postings {
                *scores.entry(*slot).or_default() += *weight * *stored_weight;
            }
        }
    }
    if metric == DistanceMetric::Cosine {
        let query_norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            return Vec::new();
        }
        for (slot, score) in scores.iter_mut() {
            let stored_norm = norms.get(slot).copied().unwrap_or(0.0);
            *score = if stored_norm == 0.0 {
                0.0
            } else {
                *score / (query_norm * stored_norm)
            };
        }
    }
    scores.into_iter().collect()
}

/// Sparse similarity: merge two strictly-increasing index lists.
///
/// `Dot` returns the raw weighted overlap; `Cosine` divides it by both
/// vector norms. Other metrics are rejected by collection validation and
/// never reach this function.
pub(crate) fn sparse_score(
    q_idx: &[u32],
    q_val: &[f32],
    d_idx: &[u32],
    d_val: &[f32],
    metric: crate::types::DistanceMetric,
) -> f32 {
    use crate::types::DistanceMetric::{Cosine, Dot};
    let mut qi = 0usize;
    let mut di = 0usize;
    let mut dot = 0f32;
    while qi < q_idx.len() && di < d_idx.len() {
        match q_idx[qi].cmp(&d_idx[di]) {
            std::cmp::Ordering::Less => qi += 1,
            std::cmp::Ordering::Greater => di += 1,
            std::cmp::Ordering::Equal => {
                dot += q_val[qi] * d_val[di];
                qi += 1;
                di += 1;
            }
        }
    }
    match metric {
        Dot => dot,
        Cosine => {
            let q_norm = q_val.iter().map(|v| v * v).sum::<f32>().sqrt();
            let d_norm = d_val.iter().map(|v| v * v).sum::<f32>().sqrt();
            let denom = q_norm * d_norm;
            if denom == 0.0 {
                0.0
            } else {
                dot / denom
            }
        }
        crate::types::DistanceMetric::Euclid | crate::types::DistanceMetric::Manhattan => {
            unreachable!("sparse distance metric was validated at collection creation")
        }
    }
}

/// Dense similarity remapped onto the crate-wide higher-is-better contract.
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

/// Exact MaxSim: for every query sub-vector take the best-matching stored
/// sub-vector and sum the per-query bests. Inner dense operations reuse the
/// shared SIMD kernel dispatch.
fn max_sim(
    query: &crate::types::MultiVector,
    stored: &crate::types::MultiVector,
    metric: crate::types::DistanceMetric,
) -> f32 {
    let similarity = |q: &[f32], d: &[f32]| -> f32 {
        let distance = crate::distance::distance(metric, q, d);
        match metric {
            crate::types::DistanceMetric::Cosine => 1.0 - distance,
            crate::types::DistanceMetric::Dot => -distance,
            crate::types::DistanceMetric::Euclid | crate::types::DistanceMetric::Manhattan => {
                1.0 / (1.0 + distance)
            }
        }
    };
    let mut total = 0f32;
    for q in query.vectors() {
        let Some(best) = stored
            .vectors()
            .map(|d| similarity(q, d))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        else {
            return 0.0;
        };
        total += best;
    }
    total
}
