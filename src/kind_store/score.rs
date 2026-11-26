use crate::types::VectorData;

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
        _ => 0.0,
    }
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

/// Exact MaxSim: for every query sub-vector take the best-matching stored
/// sub-vector and sum the per-query bests. Inner dense operations reuse the
/// shared SIMD kernel dispatch.
fn max_sim(query: &[Vec<f32>], stored: &[Vec<f32>], metric: crate::types::DistanceMetric) -> f32 {
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
    for q in query {
        let Some(best) = stored
            .iter()
            .map(|d| similarity(q, d))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        else {
            return 0.0;
        };
        total += best;
    }
    total
}
