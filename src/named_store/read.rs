use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{Result, VectorSearchError};
use crate::filter;
use crate::types::query::reject_exact_query_controls;
use crate::types::{
    PointId, SearchQuery, SearchResult, VectorData, VectorFilter, VectorPoint, VectorSpaceConfig,
};

use super::{score_pair, sparse_postings, NamedStore};

impl NamedStore {
    /// Declared named-vector spaces.
    pub fn spaces(&self) -> BTreeMap<String, VectorSpaceConfig> {
        self.spaces
            .iter()
            .map(|(name, meta)| {
                (
                    name.clone(),
                    VectorSpaceConfig {
                        kind: meta.kind,
                        vector_size: meta.vector_size,
                        distance: meta.distance,
                    },
                )
            })
            .collect()
    }

    /// Directory backing this collection.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Engine-level storage tuning.
    pub fn storage_config(&self) -> &crate::types::StorageConfig {
        &self.storage
    }

    /// Number of live points.
    pub fn count(&self) -> u64 {
        self.inner.lock().live_count
    }

    /// Operational counters for this collection.
    pub fn metrics(&self) -> &crate::metrics::Metrics {
        &self.metrics
    }

    /// Approximate in-memory payload size of the live points.
    pub fn memory_bytes(&self) -> u64 {
        let inner = self.inner.lock();
        inner
            .points
            .iter()
            .flatten()
            .map(|point| point.vectors.values().map(vector_data_bytes).sum::<usize>() as u64)
            .sum()
    }

    /// Number of live points matching `filter`. Pure read.
    pub fn count_by_filter(&self, filter: &VectorFilter) -> Result<u64> {
        let inner = self.inner.lock();
        let mut count = 0u64;
        for (slot, point) in inner
            .points
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.is_some().then_some((i, p.as_ref().expect("live point"))))
        {
            let _ = slot;
            if filter::matches(filter, &point.id, point.payload.as_ref())? {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Exact-scan top-k search over one named space.
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let started = std::time::Instant::now();
        let target = query.target.as_deref().ok_or_else(|| {
            VectorSearchError::InvalidConfig(
                "named collection queries must declare a target space".to_string(),
            )
        })?;
        let data = query.named_vector.as_ref().ok_or_else(|| {
            VectorSearchError::InvalidConfig(
                "named collection queries must carry a named query vector".to_string(),
            )
        })?;
        if !query.vector.is_empty() || query.data.is_some() {
            return Err(VectorSearchError::InvalidConfig(
                "named collection queries must not carry single-space payloads".to_string(),
            ));
        }
        reject_exact_query_controls(query)?;
        let space = self.spaces.get(target).ok_or_else(|| {
            VectorSearchError::InvalidConfig(format!("unknown named-vector space {target}"))
        })?;
        if data.kind() != space.kind {
            return Err(VectorSearchError::InvalidConfig(format!(
                "space {target} kind is {}, query kind is {}",
                space.kind.as_name(),
                data.kind().as_name()
            )));
        }
        data.validate(space.vector_size, 1)?;

        let inner = self.inner.lock();
        let limit = query.effective_limit();
        let offset = query.offset.unwrap_or(0);
        if limit == 0 {
            return Ok(Vec::new());
        }

        // Candidate set: inverted postings for sparse spaces, every live
        // slot for dense spaces.
        let candidates: Vec<u32> = match data {
            VectorData::Sparse { indices, .. } => inner
                .inverted
                .get(target)
                .and_then(|space_index| sparse_postings(space_index, indices, inner.points.len()))
                .unwrap_or_default(),
            _ => (0..inner.points.len() as u32)
                .filter(|slot| inner.points[*slot as usize].is_some())
                .collect(),
        };

        let metric = space.distance;
        let mut scored: Vec<(u32, f32)> = Vec::new();
        for slot in candidates {
            let Some(point) = inner.points[slot as usize].as_ref() else {
                continue;
            };
            if let Some(filter) = &query.filter {
                if !filter::matches(filter, &point.id, point.payload.as_ref())? {
                    continue;
                }
            }
            let stored = point
                .vectors
                .get(target)
                .expect("validated point carries every space");
            let score = score_pair(data, stored, metric);
            if query
                .score_threshold()
                .is_some_and(|threshold| score < threshold)
            {
                continue;
            }
            scored.push((slot, score));
        }

        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        let with_payload = query
            .with_payload
            .unwrap_or(crate::types::DEFAULT_WITH_PAYLOAD);
        let with_vector = query.with_vector.unwrap_or(false);
        let results: Vec<SearchResult> = scored
            .into_iter()
            .skip(offset)
            .take(limit)
            .filter_map(|(slot, score)| {
                let point = inner.points[slot as usize].as_ref()?;
                let mut result = SearchResult::new(point.id.clone(), score);
                if with_payload {
                    if let Some(payload) = &point.payload {
                        let payload = match &query.payload_selector {
                            Some(selector) => selector.apply(payload),
                            None => payload.clone(),
                        };
                        result = result.with_payload(payload);
                    }
                }
                if with_vector {
                    let stored = point
                        .vectors
                        .get(target)
                        .expect("validated point carries every space");
                    result = result.with_data(stored.clone());
                }
                Some(result)
            })
            .collect();
        self.metrics.record_search(
            crate::metrics::SearchPath::Exact,
            query.filter.is_some(),
            started.elapsed(),
        );
        Ok(results)
    }

    /// Fetch a point by id.
    pub fn get(&self, point_id: &PointId) -> Result<Option<VectorPoint>> {
        let inner = self.inner.lock();
        let Some(&slot) = inner.reverse.get(&point_id.to_string()) else {
            return Ok(None);
        };
        Ok(inner.points[slot as usize]
            .as_ref()
            .map(|point| VectorPoint {
                id: point.id.clone(),
                vector: Vec::new(),
                data: None,
                named: point.vectors.clone(),
                payload: point.payload.clone(),
            }))
    }

    /// Paginated scan over live points in slot order.
    pub fn scroll(
        &self,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        self.scroll_impl(limit, offset, with_payload, with_vector, None)
    }

    /// Paginated scan over live points matching `filter`, in slot order.
    /// `limit` counts only matching points; the cursor is the last match.
    pub fn scroll_by_filter(
        &self,
        filter: &VectorFilter,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        self.scroll_impl(limit, offset, with_payload, with_vector, Some(filter))
    }

    fn scroll_impl(
        &self,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
        filter: Option<&VectorFilter>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        let inner = self.inner.lock();
        let start_slot = match offset {
            Some(cursor) => inner
                .reverse
                .get(&cursor.to_string())
                .map(|slot| slot + 1)
                .unwrap_or(inner.points.len() as u32),
            None => 0,
        };
        let with_payload = with_payload.unwrap_or(true);
        let with_vector = with_vector.unwrap_or(false);
        let mut points = Vec::new();
        let mut cursor = None;
        for slot in start_slot as usize..inner.points.len() {
            let Some(point) = inner.points[slot].as_ref() else {
                continue;
            };
            if let Some(filter) = filter {
                if !filter::matches(filter, &point.id, point.payload.as_ref())? {
                    continue;
                }
            }
            points.push(VectorPoint {
                id: point.id.clone(),
                vector: Vec::new(),
                data: None,
                named: if with_vector {
                    point.vectors.clone()
                } else {
                    Default::default()
                },
                payload: with_payload.then(|| point.payload.clone()).flatten(),
            });
            cursor = Some(point.id.clone());
            if points.len() >= limit {
                break;
            }
        }
        Ok((points, cursor))
    }
}

fn vector_data_bytes(data: &VectorData) -> usize {
    match data {
        VectorData::Dense(values) => values.len() * std::mem::size_of::<f32>(),
        VectorData::Sparse { indices, values } => {
            indices.len() * std::mem::size_of::<u32>() + values.len() * std::mem::size_of::<f32>()
        }
        VectorData::Multi(vectors) => {
            vectors.iter().map(|v| v.len()).sum::<usize>() * std::mem::size_of::<f32>()
        }
    }
}
