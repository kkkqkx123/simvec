use serde::{Deserialize, Serialize};

use crate::filter_cond::{PayloadSelector, PointId, VectorFilter};
use crate::types::point::VectorData;

use super::point::Payload;

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
/// `QuantizationSearchParams`). The exact-scan quantization path consumes all
/// fields; HNSW graph traversal consumes `ignore`/`exact` plus `rescore` and
/// `oversampling` for candidate widening; the IVF path stays exact and
/// ignores these controls, as do remote backends.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantizationSearchParams {
    /// Skip the quantized scan and use the exact path.
    pub ignore: bool,
    /// Exact search: same as `ignore`, expressed the way qdrant's
    /// `SearchRequest.exact` does it. Takes precedence over `ignore`.
    pub exact: bool,
    /// Rescore quantized coarse candidates with original f32 vectors before
    /// returning. Default depends on the quant method: Binary defaults to
    /// true (Hamming ranking is only coarsely monotonic in the true metric),
    /// Scalar/Product default to false (their quantized distance is already
    /// in the true metric's units).
    pub rescore: Option<bool>,
    /// Candidate multiplier for the quantized coarse scan (>= 1.0). When
    /// unset, a per-method default is used (Scalar 2x, Binary 4x, Product 3x).
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub vector: Vec<f32>,
    /// Non-dense query payload for Sparse/Multi collections. Exactly one of
    /// `vector` and `data` must be set, and the shape must match the
    /// collection kind.
    #[serde(default)]
    pub data: Option<VectorData>,
    pub limit: usize,
    pub offset: Option<usize>,
    /// Similarity lower bound: results with `score >= score_threshold` are
    /// kept. Scores are "higher is better" on every backend.
    pub score_threshold: Option<f32>,
    pub filter: Option<VectorFilter>,
    pub with_payload: Option<bool>,
    pub with_vector: Option<bool>,
    /// Number of IVF cells probed per search. Consumed only when the local
    /// engine published an IVF index; HNSW and remote backends ignore it.
    /// For HNSW recall control use `SearchMode::KNN.ef_search` instead.
    pub nprobe: Option<usize>,
    pub search_mode: Option<SearchMode>,
    /// Returned payload field projection (include / exclude lists).
    /// `None` returns the whole payload. Applied after the payload read,
    /// so it only trims serialization cost, never filtering semantics.
    pub payload_selector: Option<PayloadSelector>,
    /// Quantized-scan controls (`ignore` / `exact` / `rescore` /
    /// `oversampling`). `None` uses the defaults of [`QuantizationSearchParams`].
    pub quantization: Option<QuantizationSearchParams>,
}

impl SearchQuery {
    pub fn new(vector: Vec<f32>, limit: usize) -> Self {
        Self {
            vector,
            data: None,
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

    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Build a non-dense query (Sparse or Multi collections).
    pub fn with_data(data: VectorData, limit: usize) -> Self {
        Self {
            vector: Vec::new(),
            data: Some(data),
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

    /// Set quantized-scan controls for this search.
    pub fn with_quantization_params(mut self, params: QuantizationSearchParams) -> Self {
        self.quantization = Some(params);
        self
    }

    /// Force exact (non-quantized) search for this query.
    pub fn with_exact(mut self, exact: bool) -> Self {
        let params = self.quantization.get_or_insert_with(Default::default);
        params.exact = exact;
        self
    }

    /// Restrict the returned payload to the given fields (include list).
    pub fn with_payload_include(mut self, fields: Vec<String>) -> Self {
        let mut sel = self.payload_selector.take().unwrap_or_default();
        sel.include = Some(fields);
        self.payload_selector = Some(sel);
        self
    }

    /// Set an explicit payload projection for the search results.
    pub fn with_payload_selector(mut self, selector: PayloadSelector) -> Self {
        self.payload_selector = Some(selector);
        self
    }

    /// Remove the given fields from the returned payload (exclude list).
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
            _ => None,
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
    pub vector: Option<Vec<f32>>,
}

impl SearchResult {
    pub fn new(id: impl Into<PointId>, score: f32) -> Self {
        Self {
            id: id.into(),
            score,
            payload: None,
            vector: None,
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
