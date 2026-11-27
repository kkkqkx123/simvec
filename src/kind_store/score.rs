use std::collections::HashMap;

use crate::error::{Result, VectorSearchError};
use crate::storage::postings::Postings;
use crate::types::{DistanceMetric, VectorData, VectorView};

/// Score a query against a stored vector borrowed from the blob file.
pub(crate) fn score_view(
    query: &VectorData,
    stored: &VectorView<'_>,
    metric: crate::types::DistanceMetric,
) -> f32 {
    match (query, stored) {
        (
            VectorData::Sparse {
                indices: q_idx,
                values: q_val,
            },
            VectorView::Sparse {
                indices: d_idx,
                values: d_val,
            },
        ) => sparse_score(q_idx, q_val, d_idx, d_val, metric),
        (VectorData::Multi(q), VectorView::Multi { flat, dim }) => max_sim(q, flat, *dim, metric),
        (VectorData::Dense(q), VectorView::Dense(d)) => dense_similarity(q, d, metric),
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
///
/// Postings still hold tombstoned slots (removing one entry would rewrite the
/// record that contains it); those candidates are dropped by the caller, which
/// is where the live slot table is consulted anyway.
pub(crate) fn sparse_candidate_scores(
    postings: &Postings,
    norms: &HashMap<u32, f32>,
    indices: &[u32],
    values: &[f32],
    metric: DistanceMetric,
    mut accept: Option<&mut dyn FnMut(u32) -> Result<bool>>,
) -> Result<Vec<(u32, f32)>> {
    let mut scores: HashMap<u32, f32> = HashMap::new();
    let mut verdicts: HashMap<u32, bool> = HashMap::new();
    // `visit` cannot propagate an error out of its callback, so the first one
    // is parked here and re-raised before returning.
    let mut failure: Option<VectorSearchError> = None;
    for (index, weight) in indices.iter().zip(values.iter()) {
        postings.visit(*index, |slot, stored_weight| {
            if failure.is_some() {
                return;
            }
            if let Some(accept) = accept.as_mut() {
                let keep = match verdicts.entry(slot) {
                    std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                    std::collections::hash_map::Entry::Vacant(entry) => match accept(slot) {
                        Ok(verdict) => *entry.insert(verdict),
                        Err(error) => {
                            failure = Some(error);
                            return;
                        }
                    },
                };
                if !keep {
                    return;
                }
            }
            *scores.entry(slot).or_default() += *weight * stored_weight;
        });
    }
    if let Some(error) = failure {
        return Err(error);
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
/// of live points. The persisted document frequency gives the selectivity of
/// the rarest query term; a term present in most points makes posting
/// accumulation more expensive than one filtered pass over the live set.
/// The verdict only chooses an execution path — every path applies the full
/// filter to each candidate.
pub(crate) fn prefer_posting_scan(postings: &Postings, indices: &[u32], live: u64) -> bool {
    // An empty query has no posting hits at all; taking the posting path
    // yields an empty candidate set instead of a full scan of zero scores.
    if indices.is_empty() || live == 0 {
        return true;
    }
    postings.max_document_frequency(indices) as u64 <= live.saturating_mul(2)
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
    scored.select_nth_unstable_by(need, |a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
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
/// sub-vector and sum the per-query bests. The stored sub-vectors are walked
/// in place out of the blob mapping — one allocation-free pass per pair set.
/// The metric branch is resolved once into a score-conversion function pointer
/// rather than re-matched for every sub-vector pair, and inner dense
/// operations reuse the shared SIMD kernel.
fn max_sim(
    query: &crate::types::MultiVector,
    stored: &[f32],
    dim: usize,
    metric: crate::types::DistanceMetric,
) -> f32 {
    use crate::types::DistanceMetric::{Cosine, Dot, Euclid, Manhattan};
    let to_score: fn(f32) -> f32 = match metric {
        Dot => |d| -d,
        Cosine => |d| 1.0 - d,
        Euclid | Manhattan => |d| 1.0 / (1.0 + d.sqrt()),
    };
    let dim = dim.max(1);
    let mut total = 0f32;
    for q in query.vectors() {
        let Some(best) = stored
            .chunks_exact(dim)
            .map(|d| to_score(crate::distance::distance(metric, q, d)))
            .max_by(|a, b| a.total_cmp(b))
        else {
            return 0.0;
        };
        total += best;
    }
    total
}
