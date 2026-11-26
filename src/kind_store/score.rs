use std::collections::HashMap;

use crate::error::Result;
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
/// `accept` is evaluated lazily for each posting hit that first appears and
/// its verdict cached per slot, so a filter never forces a full scan of live
/// slots — only slots that actually share a query term are tested. A `None`
/// accept means no filter and every posting hit is accumulated.
pub(crate) fn sparse_candidate_scores(
    inverted: &HashMap<u32, Vec<(u32, f32)>>,
    norms: &HashMap<u32, f32>,
    indices: &[u32],
    values: &[f32],
    metric: DistanceMetric,
    mut accept: Option<&mut dyn FnMut(u32) -> Result<bool>>,
) -> Result<Vec<(u32, f32)>> {
    let mut scores: HashMap<u32, f32> = HashMap::new();
    let mut verdicts: HashMap<u32, bool> = HashMap::new();
    for (index, weight) in indices.iter().zip(values.iter()) {
        if let Some(postings) = inverted.get(index) {
            for (slot, stored_weight) in postings {
                if let Some(accept) = accept.as_mut() {
                    let keep = match verdicts.entry(*slot) {
                        std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            let verdict = accept(*slot)?;
                            *entry.insert(verdict)
                        }
                    };
                    if !keep {
                        continue;
                    }
                }
                *scores.entry(*slot).or_default() += *weight * *stored_weight;
            }
        }
    }
    if metric == DistanceMetric::Cosine {
        let query_norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            return Ok(Vec::new());
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
    // Zero-contribution candidates are not results: an all-zero query must
    // not surface every point sharing a term with a score of zero.
    Ok(scores
        .into_iter()
        .filter(|(_, score)| *score != 0.0)
        .collect())
}

/// Estimate whether direct posting accumulation is cheaper than a full scan
/// of live points. Uses only posting length and the live-point count — both
/// local statistics — so no global term-frequency table is needed.
pub(crate) fn posting_hits(
    inverted: &HashMap<u32, Vec<(u32, f32)>>,
    indices: &[u32],
) -> usize {
    indices
        .iter()
        .map(|index| inverted.get(index).map_or(0, Vec::len))
        .sum()
}

/// Keep only the best `offset + limit` entries in score order, leaving the
/// tail unexamined. The surviving prefix is fully sorted so callers can
/// apply offset and limit directly. Ties break by ascending slot.
pub(crate) fn truncate_topk(scored: &mut Vec<(u32, f32)>, offset: usize, limit: usize) {
    let need = offset.saturating_add(limit);
    if need == 0 || scored.len() <= need {
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        return;
    }
    scored.select_nth_unstable_by(need, |a, b| {
        b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
    });
    scored.truncate(need);
    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
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
/// Uses the single shared conversion so exact-scan stores and multi spaces
/// report scores comparable with the dense ANN path under the same metric.
fn dense_similarity(query: &[f32], stored: &[f32], metric: crate::types::DistanceMetric) -> f32 {
    let distance = crate::distance::distance(metric, query, stored);
    crate::distance::to_score(metric, distance)
}

/// Exact MaxSim: for every query sub-vector take the best-matching stored
/// sub-vector and sum the per-query bests. The metric branch is resolved once
/// into a score-conversion function pointer rather than re-matched for every
/// sub-vector pair, and inner dense operations reuse the shared SIMD kernel.
fn max_sim(
    query: &crate::types::MultiVector,
    stored: &crate::types::MultiVector,
    metric: crate::types::DistanceMetric,
) -> f32 {
    use crate::types::DistanceMetric::{Cosine, Dot, Euclid, Manhattan};
    let to_score: fn(f32) -> f32 = match metric {
        Dot => |d| -d,
        Cosine => |d| 1.0 - d,
        Euclid | Manhattan => |d| 1.0 / (1.0 + d.sqrt()),
    };
    let mut total = 0f32;
    for q in query.vectors() {
        let Some(best) = stored
            .vectors()
            .map(|d| to_score(crate::distance::distance(metric, q, d)))
            .max_by(|a, b| a.total_cmp(b))
        else {
            return 0.0;
        };
        total += best;
    }
    total
}
