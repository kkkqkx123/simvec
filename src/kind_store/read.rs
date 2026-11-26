use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use crate::error::{Result, VectorSearchError};
use crate::filter;
use crate::types::query::reject_exact_query_controls;
use crate::types::{
    PointId, SearchQuery, SearchResult, VectorData, VectorFilter, VectorKind, VectorPoint,
};

use super::{score_pair, sparse_candidate_scores, truncate_topk, KindStore};

impl KindStore {
    /// Collection kind served by this store.
    pub fn kind(&self) -> VectorKind {
        self.kind
    }

    /// Declared `vector_size` (vocabulary bound / sub-vector dimension).
    pub fn vector_size(&self) -> usize {
        self.dim
    }

    /// Declared distance metric.
    pub fn distance(&self) -> crate::types::DistanceMetric {
        self.distance
    }

    /// Per-point sub-vector cap for Multi collections (`0` for Sparse).
    pub fn max_vectors_per_point(&self) -> usize {
        self.max_vectors_per_point
    }

    /// Admission cap mirrored from the collection configuration.
    pub fn max_points(&self) -> Option<u64> {
        self.max_points
    }

    /// Directory backing this collection.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Engine-level storage tuning (unused today; kept for symmetry with the
    /// dense store's checkpoint schedule).
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
            .map(|point| vector_data_bytes(&point.data) as u64)
            .sum()
    }

    /// Number of live points matching `filter`. Pure read.
    pub fn count_by_filter(&self, filter: &VectorFilter) -> Result<u64> {
        let inner = self.inner.lock();
        let mut count = 0u64;
        for slot in inner
            .points
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.is_some().then_some(i))
        {
            let point = inner.points[slot].as_ref().expect("live point");
            if filter::matches(filter, &point.id, point.payload.as_ref())? {
                count += 1;
            }
        }
        Ok(count)
    }
}

impl KindStore {
    /// Exact-scan top-k search over the collection.
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let started = std::time::Instant::now();
        let (space, data) = query.single_target()?;
        if space != crate::types::DEFAULT_VECTOR_NAME {
            return Err(VectorSearchError::InvalidConfig(
                "kind collection queries must target the default space".to_string(),
            ));
        }
        reject_exact_query_controls(query)?;
        let owned = data.clone().normalized();
        self.validate_data(&owned)?;

        let inner = self.inner.lock();
        let limit = query.effective_limit();
        let offset = query.offset.unwrap_or(0);
        if limit == 0 {
            return Ok(Vec::new());
        }
        let allowed: Option<HashSet<u32>> = match &query.filter {
            Some(filter) => {
                let mut set = HashSet::new();
                for (slot, point) in inner.points.iter().enumerate() {
                    if let Some(point) = point {
                        if filter::matches(filter, &point.id, point.payload.as_ref())? {
                            set.insert(slot as u32);
                        }
                    }
                }
                Some(set)
            }
            None => None,
        };

        // Sparse scores accumulate directly from postings; multi scans every
        // live slot because there is no coarse structure to consult.
        let candidates: Vec<(u32, f32)> = match &owned {
            VectorData::Sparse { indices, values } => sparse_candidate_scores(
                &inner.inverted,
                &inner.norms,
                indices,
                values,
                self.distance,
                allowed.as_ref(),
            ),
            _ => (0..inner.points.len() as u32)
                .filter(|slot| inner.points[*slot as usize].is_some())
                .map(|slot| (slot, 0.0))
                .collect(),
        };

        let mut scored: Vec<(u32, f32)> = Vec::new();
        for (slot, sparse_score) in candidates {
            if allowed.is_some() && !allowed.as_ref().is_some_and(|set| set.contains(&slot)) {
                continue;
            }
            let Some(point) = inner.points[slot as usize].as_ref() else {
                continue;
            };
            if allowed.is_none() {
                if let Some(filter) = &query.filter {
                    if !filter::matches(filter, &point.id, point.payload.as_ref())? {
                        continue;
                    }
                }
            }
            let score = match &owned {
                VectorData::Sparse { .. } => sparse_score,
                _ => score_pair(&owned, &point.data, self.distance),
            };
            if query
                .score_threshold()
                .is_some_and(|threshold| score < threshold)
            {
                continue;
            }
            scored.push((slot, score));
        }

        truncate_topk(&mut scored, offset, limit);

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
                    result = result.with_data(point.data.clone());
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
        Ok(inner.points[slot as usize].as_ref().map(|point| VectorPoint {
            id: point.id.clone(),
            vectors: BTreeMap::from([(
                crate::types::DEFAULT_VECTOR_NAME.to_string(),
                point.data.clone(),
            )]),
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
            points.push(VectorPoint {
                id: point.id.clone(),
                vectors: if with_vector {
                    BTreeMap::from([(
                        crate::types::DEFAULT_VECTOR_NAME.to_string(),
                        point.data.clone(),
                    )])
                } else {
                    BTreeMap::new()
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
            if !filter::matches(filter, &point.id, point.payload.as_ref())? {
                continue;
            }
            points.push(VectorPoint {
                id: point.id.clone(),
                vectors: if with_vector {
                    BTreeMap::from([(
                        crate::types::DEFAULT_VECTOR_NAME.to_string(),
                        point.data.clone(),
                    )])
                } else {
                    BTreeMap::new()
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
