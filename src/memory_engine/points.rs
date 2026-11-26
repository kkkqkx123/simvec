use crate::distance::{distance, to_score};
use crate::error::{Result, VectorSearchError};
use crate::types::{PointId, SearchQuery, SearchResult, VectorFilter, VectorPoint};

use super::{project_payload, validate_point, MemoryEngine};

impl MemoryEngine {
    /// Upsert a single point.
    pub fn upsert(&self, collection: &str, point: VectorPoint) -> Result<()> {
        self.with_collection_mut(collection, |c| {
            validate_point(&c.config, &point)?;
            if let Some(cap) = c.config.max_points {
                let grows = !c.points.contains_key(&point.id);
                if grows && c.points.len() as u64 >= cap {
                    return Err(VectorSearchError::CapacityExceeded {
                        collection: collection.to_string(),
                        max_points: cap,
                        requested: c.points.len() as u64 + 1,
                    });
                }
            }
            c.points.insert(point.id.clone(), point);
            self.bump_revision();
            Ok(())
        })
    }

    /// Upsert a batch of points.
    pub fn upsert_batch(&self, collection: &str, points: &[VectorPoint]) -> Result<()> {
        for point in points {
            self.upsert(collection, point.clone())?;
        }
        Ok(())
    }

    /// Delete a point by id. Deleting a missing id is a no-op.
    pub fn delete(&self, collection: &str, point_id: &PointId) -> Result<()> {
        self.with_collection_mut(collection, |c| {
            if c.points.remove(point_id).is_some() {
                self.bump_revision();
            }
            Ok(())
        })
    }

    /// Delete a batch of points.
    pub fn delete_batch(&self, collection: &str, point_ids: &[PointId]) -> Result<()> {
        self.with_collection_mut(collection, |c| {
            let mut removed = false;
            for id in point_ids {
                if c.points.remove(id).is_some() {
                    removed = true;
                }
            }
            if removed {
                self.bump_revision();
            }
            Ok(())
        })
    }

    /// Delete every point matching the filter. Returns the number deleted.
    pub fn delete_by_filter(&self, collection: &str, filter: &VectorFilter) -> Result<u64> {
        self.with_collection_mut(collection, |c| {
            let mut doomed = Vec::new();
            for (id, point) in &c.points {
                if crate::filter::matches(filter, id, point.payload.as_ref())? {
                    doomed.push(id.clone());
                }
            }
            let count = doomed.len() as u64;
            for id in doomed {
                c.points.remove(&id);
            }
            if count > 0 {
                self.bump_revision();
            }
            Ok(count)
        })
    }

    /// Exact-scan search. Scores follow the crate-wide higher-is-better contract.
    /// Kernel dispatch degrades to the scalar baseline on wasm32.
    pub fn search(&self, collection: &str, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        query.validate_nonempty()?;
        self.with_collection(collection, |c| {
            if query.primary_vector().len() != c.config.vector_size {
                return Err(VectorSearchError::InvalidVectorDimension {
                    expected: c.config.vector_size,
                    actual: query.primary_vector().len(),
                });
            }
            for (idx, v) in query.primary_vector().iter().enumerate() {
                if !v.is_finite() {
                    return Err(VectorSearchError::NonFiniteElement(idx));
                }
            }
            let mut scored: Vec<SearchResult> = Vec::new();
            for point in c.points.values() {
                if let Some(filter) = &query.filter {
                    if !crate::filter::matches(filter, &point.id, point.payload.as_ref())? {
                        continue;
                    }
                }
                let dist = distance(c.config.distance, query.primary_vector(), point.vector());
                let score = to_score(c.config.distance, dist);
                if let Some(threshold) = query.score_threshold {
                    if score < threshold {
                        continue;
                    }
                }
                let mut result = SearchResult::new(point.id.clone(), score);
                if query.with_payload.unwrap_or(true) {
                    result.payload = project_payload(point.payload.clone(), query);
                }
                if query.with_vector.unwrap_or(false) {
                    result.vector = Some(point.vector().to_vec());
                }
                scored.push(result);
            }
            scored.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let offset = query.offset.unwrap_or(0);
            let limit = query.effective_limit();
            Ok(scored.into_iter().skip(offset).take(limit).collect())
        })
    }

    /// Fetch a point by id.
    pub fn get(&self, collection: &str, point_id: &PointId) -> Result<Option<VectorPoint>> {
        self.with_collection(collection, |c| Ok(c.points.get(point_id).cloned()))
    }

    /// Number of live points.
    pub fn count(&self, collection: &str) -> Result<u64> {
        self.with_collection(collection, |c| Ok(c.live_count()))
    }

    /// Number of live points matching `filter`. Pure read.
    pub fn count_by_filter(&self, collection: &str, filter: &VectorFilter) -> Result<u64> {
        self.with_collection(collection, |c| {
            let mut count = 0u64;
            for (id, point) in &c.points {
                if crate::filter::matches(filter, id, point.payload.as_ref())? {
                    count += 1;
                }
            }
            Ok(count)
        })
    }

    /// Paginated scan ordered by point id string.
    pub fn scroll(
        &self,
        collection: &str,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        self.with_collection(collection, |c| {
            let mut ids: Vec<&PointId> = c.points.keys().collect();
            ids.sort_by_key(|a| a.to_string());
            let mut skipping = offset.is_some();
            let mut results = Vec::new();
            let mut last: Option<PointId> = None;
            for id in ids {
                if skipping {
                    if Some(id) == offset {
                        skipping = false;
                    }
                    continue;
                }
                if let Some(point) = c.points.get(id) {
                    last = Some(id.clone());
                    let vectors = if with_vector.unwrap_or(false) {
                        point.vectors.clone()
                    } else {
                        Default::default()
                    };
                    results.push(VectorPoint {
                        id: id.clone(),
                        vectors,
                        payload: if with_payload.unwrap_or(true) {
                            point.payload.clone()
                        } else {
                            None
                        },
                    });
                    if results.len() >= limit {
                        break;
                    }
                }
            }
            Ok((results, last))
        })
    }

    /// Paginated scan over points matching `filter`, ordered by id string.
    /// `limit` counts only matching points; the cursor is the last match.
    /// A short page does not imply exhaustion: page until an empty page.
    pub fn scroll_by_filter(
        &self,
        collection: &str,
        filter: &VectorFilter,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        self.with_collection(collection, |c| {
            let mut ids: Vec<&PointId> = c.points.keys().collect();
            ids.sort_by_key(|a| a.to_string());
            let mut skipping = offset.is_some();
            let mut results = Vec::new();
            let mut last: Option<PointId> = None;
            for id in ids {
                if skipping {
                    if Some(id) == offset {
                        skipping = false;
                    }
                    continue;
                }
                if let Some(point) = c.points.get(id) {
                    if !crate::filter::matches(filter, id, point.payload.as_ref())? {
                        continue;
                    }
                    last = Some(id.clone());
                    let vectors = if with_vector.unwrap_or(false) {
                        point.vectors.clone()
                    } else {
                        Default::default()
                    };
                    results.push(VectorPoint {
                        id: id.clone(),
                        vectors,
                        payload: if with_payload.unwrap_or(true) {
                            point.payload.clone()
                        } else {
                            None
                        },
                    });
                    if results.len() >= limit {
                        break;
                    }
                }
            }
            Ok((results, last))
        })
    }
}
