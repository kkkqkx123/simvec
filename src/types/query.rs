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
