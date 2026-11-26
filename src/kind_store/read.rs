use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{Result, VectorSearchError};
use crate::filter;
use crate::types::{
    PointId, SearchQuery, SearchResult, VectorData, VectorFilter, VectorKind, VectorPoint,
};

use super::{score_pair, KindStore};

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
        let data = query.data.as_ref().ok_or_else(|| {
            VectorSearchError::InvalidConfig(
                "kind collection queries must carry a non-dense payload".to_string(),
            )
        })?;
        if !query.vector.is_empty() {
            return Err(VectorSearchError::InvalidConfig(
                "kind collection queries must not also carry a dense vector".to_string(),
            ));
        }
        if query.quantization.is_some() || query.nprobe.is_some() {
            return Err(VectorSearchError::InvalidConfig(
                "quantization and nprobe controls only apply to dense collections".to_string(),
            ));
        }
        self.validate_data(data)?;

        let inner = self.inner.lock();
        let limit = query.effective_limit();
        let offset = query.offset.unwrap_or(0);
        if limit == 0 {
            return Ok(Vec::new());
        }

        // Candidate set: inverted postings for sparse, every live slot for
        // multi (there is no coarse structure to consult in the latter case).
        let candidates: Vec<u32> = match data {
            VectorData::Sparse { indices, .. } => inner.sparse_candidates(indices),
            _ => (0..inner.points.len() as u32)
                .filter(|slot| inner.points[*slot as usize].is_some())
                .collect(),
        };

        let mut scored: Vec<(u32, f32)> = Vec::new();
        let mut skipped = 0usize;
        for slot in candidates {
            let Some(point) = inner.points[slot as usize].as_ref() else {
                continue;
            };
            if let Some(filter) = &query.filter {
                if !filter::matches(filter, &point.id, point.payload.as_ref())? {
                    continue;
                }
            }
            let score = score_pair(data, &point.data, self.distance);
            if let Some(threshold) = query.score_threshold {
                if score < threshold {
                    continue;
                }
            }
            if skipped < offset {
                skipped += 1;
                continue;
            }
            scored.push((slot, score));
        }

        scored.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(limit);

        let with_vector = query.with_vector.unwrap_or(false);
        Ok(scored
            .into_iter()
            .filter_map(|(slot, score)| {
                let point = inner.points[slot as usize].as_ref()?;
                Some(SearchResult::new(point.id.clone(), score).with_payload(
                    match (&point.payload, query.with_payload.unwrap_or(true)) {
                        (Some(payload), true) => payload.clone(),
                        _ => Default::default(),
                    },
                ))
            })
            .map(|result| {
                if with_vector {
                    // The stored data payload is returned through the point's
                    // `data` field; SearchResult carries dense vectors only,
                    // so non-dense results stay in the store.
                    result
                } else {
                    result
                }
            })
            .collect())
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
                data: Some(point.data.clone()),
                named: BTreeMap::new(),
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
            if with_vector {
                let data = point.data.clone();
                let _ = data;
            }
            points.push(VectorPoint {
                id: point.id.clone(),
                vector: Vec::new(),
                data: with_vector.then(|| point.data.clone()),
                named: BTreeMap::new(),
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
                vector: Vec::new(),
                data: with_vector.then(|| point.data.clone()),
                named: BTreeMap::new(),
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
