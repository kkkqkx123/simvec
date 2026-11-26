use std::collections::HashMap;

use crate::error::Result;
use crate::kind_store::{score_pair as kind_score_pair, sparse_candidate_scores};
use crate::types::DistanceMetric;

/// Score a (query, stored) payload pair within one named space.
///
/// Dense, sparse and multi spaces all share the same scoring implementation;
/// the crate-wide higher-is-better contract is preserved for every kind.
pub(crate) fn score_pair(
    query: &crate::types::VectorData,
    stored: &crate::types::VectorData,
    metric: DistanceMetric,
) -> f32 {
    kind_score_pair(query, stored, metric)
}

/// Accumulate sparse candidates directly from one space's posting lists.
pub(crate) fn sparse_scores(
    space_index: &HashMap<u32, Vec<(u32, f32)>>,
    norms: &HashMap<u32, f32>,
    indices: &[u32],
    values: &[f32],
    metric: DistanceMetric,
    accept: Option<&mut dyn FnMut(u32) -> Result<bool>>,
) -> Result<Vec<(u32, f32)>> {
    sparse_candidate_scores(space_index, norms, indices, values, metric, accept)
}
