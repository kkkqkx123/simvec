use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::error::{Result, VectorSearchError};
use crate::filter;
use crate::types::query::reject_exact_query_controls;
use crate::types::{
    PointId, SearchQuery, SearchResult, VectorData, VectorFilter, VectorPoint, VectorSpaceConfig,
};

use super::{score_pair, sparse_scores, NamedStore};

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
                        multi_config: meta.multi_config,
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

    /// Admission cap mirrored from the collection configuration.
    pub fn max_points(&self) -> Option<u64> {
        self.max_points
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

    /// Exact-scan search. A query with multiple named targets fuses their
    /// per-space rankings with reciprocal rank fusion.
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        if query.vectors.len() > 1 {
            return self.search_fused(query);
        }
        let started = std::time::Instant::now();
        let (target, data) = query.single_target()?;
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
        let max_vectors = space
            .multi_config
            .map_or(1, |multi| multi.max_vectors_per_point);
        data.validate(space.vector_size, max_vectors)?;

        let inner = self.inner.lock();
        let limit = query.effective_limit();
        let offset = query.offset.unwrap_or(0);
        if limit == 0 {
            return Ok(Vec::new());
        }

        // Sparse scores accumulate from postings; other spaces scan all slots.
        let metric = space.distance;
        let candidates: Vec<(u32, f32)> = match data {
            VectorData::Sparse { indices, values } => {
                let space_index = inner.inverted.get(target);
                let norms = inner.norms.get(target);
                match (space_index, norms) {
                    (Some(space_index), Some(norms)) => {
                        sparse_scores(space_index, norms, indices, values, metric)
                    }
                    _ => Vec::new(),
                }
            }
            _ => (0..inner.points.len() as u32)
                .filter(|slot| inner.points[*slot as usize].is_some())
                .map(|slot| (slot, 0.0))
                .collect(),
        };

        let mut scored: Vec<(u32, f32)> = Vec::new();
        for (slot, sparse_score) in candidates {
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
            let score = match data {
                VectorData::Sparse { .. } => sparse_score,
                _ => score_pair(data, stored, metric),
            };
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
                    result = result.with_named_vector(target, stored.clone());
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

    /// Fuse per-space rankings with RRF. Each target shares the query's
    /// filters, paging and payload projection; only the vector target differs.
    fn search_fused(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let started = std::time::Instant::now();
        let mut fused: HashMap<PointId, (f32, SearchResult)> = HashMap::new();
        for target in &query.vectors {
            let mut single = query.clone();
            single.vectors = vec![target.clone()];
            for (rank, result) in self.search(&single)?.into_iter().enumerate() {
                let contribution = 1.0 / (60.0 + rank as f32 + 1.0);
                match fused.get_mut(&result.id) {
                    Some((score, best)) => {
                        *score += contribution;
                        for (space, data) in &result.vectors {
                            best.vectors.insert(space.clone(), data.clone());
                        }
                    }
                    None => {
                        fused.insert(result.id.clone(), (contribution, result));
                    }
                }
            }
        }
        let mut ranked: Vec<(f32, SearchResult)> = fused.into_values().collect();
        ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.id.to_string().cmp(&b.1.id.to_string())));
        let limit = query.effective_limit();
        let offset = query.offset.unwrap_or(0);
        let results = ranked
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(score, mut result)| {
                result.score = score;
                result
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
        Ok(inner.points[slot as usize].as_ref().map(|point| VectorPoint {
            id: point.id.clone(),
            vectors: point.vectors.clone(),
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
                vectors: if with_vector {
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
        VectorData::Multi(multi) => multi.flat.len() * std::mem::size_of::<f32>(),
    }
}
