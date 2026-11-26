use crate::types::VectorData;

/// Score a (query, stored) payload pair under the collection metric.
///
/// Scores follow the crate-wide higher-is-better contract: raw dot products
/// for Cosine/Dot and `1/(1 + distance)` for Euclid/Manhattan.
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
/// Cosine/Dot accumulate the weighted dot product over shared terms;
/// Euclid/Manhattan accumulate squared/absolute gaps over the union and
/// normalize to the higher-is-better contract.
pub(crate) fn sparse_score(
    q_idx: &[u32],
    q_val: &[f32],
    d_idx: &[u32],
    d_val: &[f32],
    metric: crate::types::DistanceMetric,
) -> f32 {
    use crate::types::DistanceMetric::{Cosine, Dot, Euclid, Manhattan};
    let mut qi = 0usize;
    let mut di = 0usize;
    let mut acc = 0f32;
    while qi < q_idx.len() && di < d_idx.len() {
        match q_idx[qi].cmp(&d_idx[di]) {
            std::cmp::Ordering::Less => {
                if matches!(metric, Euclid | Manhattan) {
                    acc += match metric {
                        Euclid => q_val[qi] * q_val[qi],
                        _ => q_val[qi].abs(),
                    };
                }
                qi += 1;
            }
            std::cmp::Ordering::Greater => {
                if matches!(metric, Euclid | Manhattan) {
                    acc += match metric {
                        Euclid => d_val[di] * d_val[di],
                        _ => d_val[di].abs(),
                    };
                }
                di += 1;
            }
            std::cmp::Ordering::Equal => {
                let gap = q_val[qi] - d_val[di];
                acc += match metric {
                    Cosine | Dot => q_val[qi] * d_val[di],
                    Euclid => gap * gap,
                    Manhattan => gap.abs(),
                };
                qi += 1;
                di += 1;
            }
        }
    }
    for (i, value) in q_val.iter().enumerate().skip(qi) {
        if matches!(metric, Euclid | Manhattan) {
            acc += match metric {
                Euclid => value * value,
                _ => value.abs(),
            };
        }
        let _ = i;
    }
    for value in d_val.iter().skip(di) {
        if matches!(metric, Euclid | Manhattan) {
            acc += match metric {
                Euclid => value * value,
                _ => value.abs(),
            };
        }
    }
    match metric {
        Euclid | Manhattan => 1.0 / (1.0 + acc),
        Cosine | Dot => acc,
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
