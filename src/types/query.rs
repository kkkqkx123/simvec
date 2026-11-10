use serde::{Deserialize, Serialize};

use crate::filter_cond::{PayloadSelector, PointId, VectorFilter};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub vector: Vec<f32>,
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
}

impl SearchQuery {
    pub fn new(vector: Vec<f32>, limit: usize) -> Self {
        Self {
            vector,
            limit,
            offset: None,
            score_threshold: None,
            filter: None,
            with_payload: Some(true),
            with_vector: None,
            nprobe: None,
            search_mode: None,
            payload_selector: None,
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
