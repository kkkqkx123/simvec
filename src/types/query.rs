use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::filter_cond::{PayloadSelector, PointId, VectorFilter};
use crate::types::config::DEFAULT_VECTOR_NAME;
use crate::types::point::{Payload, VectorData};

pub const DEFAULT_WITH_PAYLOAD: bool = true;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMode {
    TopK(usize),
    /// HNSW-style kNN search; `ef_search` is the only recall knob remote
    /// (Qdrant) backends honor — `nprobe` is ignored there.
    KNN {
        k: usize,
        ef_search: Option<usize>,
    },
    Range {
        radius: f32,
        max_results: Option<usize>,
    },
}

/// Per-search quantization controls (mirrors qdrant's
/// `QuantizationSearchParams`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantizationSearchParams {
    pub ignore: bool,
    pub exact: bool,
    pub rescore: Option<bool>,
    pub oversampling: Option<f64>,
}

impl QuantizationSearchParams {
    pub fn default_rescore(&self, quant_type: &crate::types::QuantizationType) -> bool {
        self.rescore.unwrap_or(matches!(
            quant_type,
            crate::types::QuantizationType::Binary { .. }
        ))
    }
}

/// One query vector and the space it targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedVectorQuery {
    pub space: String,
    pub data: VectorData,
}

pub const DEFAULT_RRF_K: usize = 2;

/// How multiple target rankings are fused into one result list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionConfig {
    /// Reciprocal rank fusion. Weights are per target; a missing weight
    /// defaults to 1.0. `k` defaults to [`DEFAULT_RRF_K`] when omitted.
    Rrf {
        #[serde(default)]
        k: Option<usize>,
        #[serde(default)]
        weights: Vec<f32>,
    },
    /// Distribution-based score fusion: each list is normalized to [0,1]
    /// using mean ± 3σ, then the (optionally weighted) scores are summed.
    Dbsf {
        #[serde(default)]
        weights: Vec<f32>,
    },
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self::Rrf {
            k: None,
            weights: Vec::new(),
        }
    }
}

fn fusion_weight(weights: &[f32], target: usize) -> f32 {
    weights.get(target).copied().unwrap_or(1.0)
}

/// Fuse per-target result lists into one ranked list. Each input list is
/// already ordered by its own score (best first). Vector maps of repeated ids
/// are merged so callers receive the union of all targeted spaces.
pub fn fuse_rankings(
    rankings: Vec<Vec<SearchResult>>,
    config: &FusionConfig,
    offset: usize,
    limit: usize,
) -> Vec<SearchResult> {
    use std::collections::HashMap;

    let mut fused: HashMap<PointId, (f32, SearchResult)> = HashMap::new();
    match config {
        FusionConfig::Rrf { k, weights } => {
            let k = k.unwrap_or(DEFAULT_RRF_K);
            for (target, list) in rankings.into_iter().enumerate() {
                let weight = fusion_weight(weights, target);
                if weight <= 0.0 {
                    continue;
                }
                for (rank, result) in list.into_iter().enumerate() {
                    let contribution = 1.0 / ((rank as f32 + 1.0) / weight + k as f32 - 1.0);
                    merge_fused(&mut fused, result, contribution);
                }
            }
        }
        FusionConfig::Dbsf { weights } => {
            let normalized: Vec<(f32, Vec<SearchResult>)> = rankings
                .into_iter()
                .enumerate()
                .map(|(target, list)| (fusion_weight(weights, target), normalize_distr(list)))
                .collect();
            for (weight, list) in normalized {
                for result in list {
                    merge_fused(&mut fused, result, weight);
                }
            }
        }
    }
    let mut ranked: Vec<(f32, SearchResult)> = fused.into_values().collect();
    ranked.sort_unstable_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.id.to_string().cmp(&b.1.id.to_string()))
    });
    ranked
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|(score, mut result)| {
            result.score = score;
            result
        })
        .collect()
}

fn merge_fused(
    fused: &mut std::collections::HashMap<PointId, (f32, SearchResult)>,
    result: SearchResult,
    contribution: f32,
) {
    match fused.get_mut(&result.id) {
        Some((score, best)) => {
            *score += contribution;
            for (space, data) in result.vectors {
                best.vectors.insert(space, data);
            }
        }
        None => {
            fused.insert(result.id.clone(), (contribution, result));
        }
    }
}

/// Normalize a scored list to [0,1] using mean ± 3σ (Welford one-pass).
fn normalize_distr(mut list: Vec<SearchResult>) -> Vec<SearchResult> {
    if list.len() < 2 {
        if let Some(one) = list.first_mut() {
            one.score = 0.5;
        }
        return list;
    }
    let n = list.len() as f32;
    let mean = list.iter().map(|r| r.score).sum::<f32>() / n;
    let variance = list
        .iter()
        .map(|r| (r.score - mean) * (r.score - mean))
        .sum::<f32>()
        / (n - 1.0);
    let std_dev = variance.sqrt();
    let min = mean - 3.0 * std_dev;
    let max = mean + 3.0 * std_dev;
    if min == max {
        for r in &mut list {
            r.score = 0.5;
        }
        return list;
    }
    for r in &mut list {
        r.score = (r.score - min) / (max - min);
    }
    list
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Query vectors, one per targeted space. Single-space collections carry
    /// exactly one entry for [`DEFAULT_VECTOR_NAME`].
    pub vectors: Vec<NamedVectorQuery>,
    pub limit: usize,
    pub offset: Option<usize>,
    pub score_threshold: Option<f32>,
    pub filter: Option<VectorFilter>,
    pub with_payload: Option<bool>,
    pub with_vector: Option<bool>,
    pub nprobe: Option<usize>,
    pub search_mode: Option<SearchMode>,
    pub payload_selector: Option<PayloadSelector>,
    pub quantization: Option<QuantizationSearchParams>,
    /// How multiple target rankings are combined. Defaults to weighted RRF
    /// when absent. Serde-defaulted to keep the JSON boundary compatible.
    #[serde(default)]
    pub fusion: Option<FusionConfig>,
    /// When `with_vector` is set, restrict the echoed spaces to this list.
    /// Empty/absent means all spaces.
    #[serde(default)]
    pub vector_spaces: Option<Vec<String>>,
}

impl SearchQuery {
    fn empty(vectors: Vec<NamedVectorQuery>, limit: usize) -> Self {
        Self {
            vectors,
            limit,
            offset: None,
            score_threshold: None,
            filter: None,
            with_payload: Some(true),
            with_vector: None,
            nprobe: None,
            search_mode: None,
            payload_selector: None,
            quantization: None,
            fusion: None,
            vector_spaces: None,
        }
    }

    pub fn new(vector: Vec<f32>, limit: usize) -> Self {
        Self::empty(
            vec![NamedVectorQuery {
                space: DEFAULT_VECTOR_NAME.to_string(),
                data: VectorData::Dense(vector),
            }],
            limit,
        )
    }

    /// Build a non-dense query for a single-space collection.
    pub fn with_data(data: VectorData, limit: usize) -> Self {
        Self::empty(
            vec![NamedVectorQuery {
                space: DEFAULT_VECTOR_NAME.to_string(),
                data,
            }],
            limit,
        )
    }

    /// Build a query targeting one named space.
    pub fn with_named(name: impl Into<String>, data: VectorData, limit: usize) -> Self {
        Self::empty(
            vec![NamedVectorQuery {
                space: name.into(),
                data,
            }],
            limit,
        )
    }

    /// Add another named target. Multi-target searches fuse their rankings.
    pub fn with_named_vector(mut self, name: impl Into<String>, data: VectorData) -> Self {
        self.vectors.push(NamedVectorQuery {
            space: name.into(),
            data,
        });
        self
    }

    /// Reject a query without any target before a dense path indexes `primary`.
    pub fn validate_nonempty(&self) -> Result<()> {
        if self.vectors.is_empty() {
            return Err(VectorSearchError::InvalidConfig(
                "query must declare at least one vector".to_string(),
            ));
        }
        Ok(())
    }

    /// The first query vector; dense paths always use this one.
    pub fn primary(&self) -> &VectorData {
        &self.vectors[0].data
    }

    /// Dense shortcut for the primary target.
    pub fn primary_vector(&self) -> &[f32] {
        match self.primary() {
            VectorData::Dense(vector) => vector,
            _ => &[],
        }
    }

    /// Exactly one target, used by exact-scan stores.
    pub fn single_target(&self) -> Result<(&str, &VectorData)> {
        match self.vectors.as_slice() {
            [query] => Ok((&query.space, &query.data)),
            _ => Err(VectorSearchError::InvalidConfig(
                "this collection accepts exactly one query vector".to_string(),
            )),
        }
    }

    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn with_score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }

    pub fn with_filter(mut self, filter: VectorFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_payload(mut self, with_payload: bool) -> Self {
        self.with_payload = Some(with_payload);
        self
    }

    pub fn with_vector(mut self, with_vector: bool) -> Self {
        self.with_vector = Some(with_vector);
        self
    }

    /// Restrict the echoed named spaces when `with_vector` is enabled.
    pub fn with_vector_spaces(mut self, spaces: Vec<String>) -> Self {
        self.vector_spaces = Some(spaces);
        self
    }

    /// Configure how multiple target rankings are fused.
    pub fn with_fusion(mut self, fusion: FusionConfig) -> Self {
        self.fusion = Some(fusion);
        self
    }

    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = Some(nprobe);
        self
    }

    pub fn with_quantization_params(mut self, params: QuantizationSearchParams) -> Self {
        self.quantization = Some(params);
        self
    }

    pub fn with_exact(mut self, exact: bool) -> Self {
        let params = self.quantization.get_or_insert_with(Default::default);
        params.exact = exact;
        self
    }

    pub fn with_payload_include(mut self, fields: Vec<String>) -> Self {
        let mut sel = self.payload_selector.take().unwrap_or_default();
        sel.include = Some(fields);
        self.payload_selector = Some(sel);
        self
    }

    pub fn with_payload_selector(mut self, selector: PayloadSelector) -> Self {
        self.payload_selector = Some(selector);
        self
    }

    pub fn with_payload_exclude(mut self, fields: Vec<String>) -> Self {
        let mut sel = self.payload_selector.take().unwrap_or_default();
        sel.exclude = Some(fields);
        self.payload_selector = Some(sel);
        self
    }

    pub fn effective_limit(&self) -> usize {
        match &self.search_mode {
            Some(SearchMode::Range {
                max_results: Some(max),
                ..
            }) => *max,
            Some(SearchMode::TopK(k)) => *k,
            Some(SearchMode::KNN { k, .. }) => *k,
            _ => self.limit,
        }
    }

    pub fn hnsw_ef(&self) -> Option<usize> {
        match &self.search_mode {
            Some(SearchMode::KNN { ef_search, .. }) => *ef_search,
            _ => None,
        }
    }

    pub fn score_threshold(&self) -> Option<f32> {
        match &self.search_mode {
            Some(SearchMode::Range { radius, .. }) => Some(*radius),
            _ => self.score_threshold,
        }
    }

    pub fn with_search_mode(mut self, mode: SearchMode) -> Self {
        self.search_mode = Some(mode);
        self
    }

    pub fn with_knn(mut self, k: usize, ef_search: Option<usize>) -> Self {
        self.search_mode = Some(SearchMode::KNN { k, ef_search });
        self.limit = k;
        self
    }

    pub fn with_range(mut self, radius: f32, max_results: Option<usize>) -> Self {
        self.search_mode = Some(SearchMode::Range {
            radius,
            max_results,
        });
        self.score_threshold = Some(radius);
        if let Some(max) = max_results {
            self.limit = max;
        }
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: PointId,
    pub score: f32,
    pub payload: Option<Payload>,
    /// Dense payload for FFI convenience. Named results also carry `vectors`.
    pub vector: Option<Vec<f32>>,
    #[serde(default)]
    pub data: Option<VectorData>,
    /// Full named vector map echoed back when `with_vector` is requested.
    #[serde(default)]
    pub vectors: BTreeMap<String, VectorData>,
}

impl SearchResult {
    pub fn new(id: impl Into<PointId>, score: f32) -> Self {
        Self {
            id: id.into(),
            score,
            payload: None,
            vector: None,
            data: None,
            vectors: BTreeMap::new(),
        }
    }

    pub fn with_payload(mut self, payload: Payload) -> Self {
        self.payload = Some(payload);
        self
    }

    pub fn with_vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(vector);
        self
    }

    pub fn with_data(mut self, data: VectorData) -> Self {
        self.data = Some(data);
        self
    }

    pub fn with_named_vector(mut self, space: impl Into<String>, data: VectorData) -> Self {
        self.vectors.insert(space.into(), data);
        self
    }
}

/// Reject dense-only search controls on exact-scan collections.
pub(crate) fn reject_exact_query_controls(query: &SearchQuery) -> crate::error::Result<()> {
    if query.search_mode.is_some() || query.nprobe.is_some() || query.quantization.is_some() {
        return Err(VectorSearchError::InvalidConfig(
            "search_mode, nprobe and quantization controls only apply to dense collections"
                .to_string(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    pub results: Vec<SearchResult>,
    pub total: Option<u64>,
}

impl SearchResults {
    pub fn new(results: Vec<SearchResult>) -> Self {
        let total = Some(results.len() as u64);
        Self { results, total }
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    pub fn len(&self) -> usize {
        self.results.len()
    }
}

impl From<Vec<SearchResult>> for SearchResults {
    fn from(results: Vec<SearchResult>) -> Self {
        Self::new(results)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSearchQuery {
    pub queries: Vec<SearchQuery>,
}

impl BatchSearchQuery {
    pub fn new(queries: Vec<SearchQuery>) -> Self {
        Self { queries }
    }
}

impl From<Vec<SearchQuery>> for BatchSearchQuery {
    fn from(queries: Vec<SearchQuery>) -> Self {
        Self::new(queries)
    }
}
