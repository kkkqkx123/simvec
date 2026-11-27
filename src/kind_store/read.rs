use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::error::{Result, VectorSearchError};
use crate::filter;
use crate::types::query::reject_exact_query_controls;
use crate::types::{
    PointId, SearchQuery, SearchResult, VectorData, VectorFilter, VectorKind, VectorPoint,
};

use super::{prefer_posting_scan, score_view, sparse_candidate_scores, truncate_topk, KindStore};

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

    /// Directory holding the published generation's data files.
    pub fn data_dir(&self) -> std::path::PathBuf {
        self.generation.read().path()
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

    /// Whether the posting dictionary came from disk at open. `false` means it
    /// was rebuilt from the snapshot because the manifest did not match it —
    /// the state a crash between the manifest write and the snapshot leaves.
    pub fn postings_restored(&self) -> bool {
        self.postings.load().was_restored()
    }

    /// (bytes of the vector blob file, bytes of the posting buckets). Both are
    /// on disk and mapped, not resident.
    pub fn storage_bytes(&self) -> (u64, u64) {
        (self.blobs.load().len(), self.postings.load().disk_bytes())
    }

    /// Committed JSON bytes held in `payloads.bin`, outside the snapshot.
    pub fn payload_bytes(&self) -> u64 {
        self.payloads.load().len()
    }

    /// Approximate resident size of the live points: the directory, the term
    /// dictionary and the pending posting window. The vectors themselves live
    /// in the mapped blob file.
    pub fn memory_bytes(&self) -> u64 {
        let inner = self.inner.lock();
        let directory =
            inner.points.capacity() as u64 * std::mem::size_of::<Option<super::KindPoint>>() as u64;
        let norms = (inner.norms.capacity() as u64) * (std::mem::size_of::<(u32, f32)>() as u64);
        let postings = self.postings.load().memory_bytes();
        directory + norms + postings + self.blobs.load().mapped_bytes()
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
        // Filter verdicts are computed lazily and cached per slot: only slots
        // that a posting actually reaches (or that a full scan visits) are
        // tested, never the whole slot table up front.
        let mut verdicts: HashMap<u32, bool> = HashMap::new();
        let mut accept = |slot: u32| -> Result<bool> {
            if let Some(verdict) = verdicts.get(&slot) {
                return Ok(*verdict);
            }
            let verdict = match (&query.filter, inner.points[slot as usize].as_ref()) {
                (Some(filter), Some(point)) => {
                    filter::matches(filter, &point.id, point.payload.as_ref())?
                }
                (Some(_), None) => false,
                (None, _) => true,
            };
            verdicts.insert(slot, verdict);
            Ok(verdict)
        };

        // Cost choice: compare posting hits with the live point count. When
        // postings are denser than the live set, a pre-filtered full scan is
        // cheaper than accumulating a fat inverted map; otherwise accumulate
        // directly from postings. Both paths share the same lazy filter cache.
        let threshold = query.score_threshold();
        let mut candidates: Vec<(u32, f32)> = Vec::new();
        let mut full_scan = false;
        if let VectorData::Sparse { indices, values } = &owned {
            full_scan = !prefer_posting_scan(&self.postings.load(), indices, inner.live_count);
            if !full_scan {
                candidates = sparse_candidate_scores(
                    &self.postings.load(),
                    &inner.norms,
                    indices,
                    values,
                    self.distance,
                    Some(&mut accept),
                )?;
            }
        }
        if full_scan || !matches!(owned, VectorData::Sparse { .. }) {
            candidates = (0..inner.points.len() as u32)
                .filter(|slot| inner.points[*slot as usize].is_some())
                .map(|slot| (slot, 0.0))
                .collect();
        }

        // One borrow of the blob mapping for the whole scoring pass.
        let blobs_guard = self.blobs.load();
        let reader = blobs_guard.reader();
        let mut scored: Vec<(u32, f32)> = Vec::new();
        for (slot, sparse_score) in candidates {
            if !accept(slot)? {
                continue;
            }
            let Some(point) = inner.points[slot as usize].as_ref() else {
                continue;
            };
            let Some(view) = reader.view(point.blob) else {
                continue;
            };
            let score = match &owned {
                VectorData::Sparse { .. } if full_scan => score_view(&owned, &view, self.distance),
                VectorData::Sparse { .. } => sparse_score,
                _ => score_view(&owned, &view, self.distance),
            };
            // Online threshold pruning: a candidate below the bound never
            // enters the TopK selection, reducing work before the sort.
            if threshold.is_some_and(|bound| score < bound) {
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
                    let data = reader.to_data(point.blob)?;
                    result = result.with_data(data);
                }
                Some(result)
            })
            .collect();
        drop(reader);
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
        let Some(slot) = inner.reverse.get(point_id) else {
            return Ok(None);
        };
        let Some(point) = inner.points[slot as usize].as_ref() else {
            return Ok(None);
        };
        let blobs_guard = self.blobs.load();
        let reader = blobs_guard.reader();
        let Some(data) = reader.to_data(point.blob) else {
            return Ok(None);
        };
        drop(reader);
        Ok(Some(VectorPoint {
            id: point.id.clone(),
            vectors: BTreeMap::from([(crate::types::DEFAULT_VECTOR_NAME.to_string(), data)]),
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
        self.scroll_inner(None, limit, offset, with_payload, with_vector)
    }

    /// Fetch one page of points, optionally filtered.
    ///
    /// `filter` counts only matching points; the cursor is the last match.
    fn scroll_inner(
        &self,
        filter: Option<&VectorFilter>,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        let inner = self.inner.lock();
        let start_slot = match offset {
            Some(cursor) => inner
                .reverse
                .get(cursor)
                .map(|slot| slot + 1)
                .unwrap_or(inner.points.len() as u32),
            None => 0,
        };
        let with_payload = with_payload.unwrap_or(true);
        let with_vector = with_vector.unwrap_or(false);
        let blobs_guard = self.blobs.load();
        let reader = if with_vector {
            Some(blobs_guard.reader())
        } else {
            None
        };
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
            let vectors = match reader.as_ref() {
                Some(reader) => match reader.to_data(point.blob) {
                    Some(data) => {
                        BTreeMap::from([(crate::types::DEFAULT_VECTOR_NAME.to_string(), data)])
                    }
                    None => continue,
                },
                None => BTreeMap::new(),
            };
            points.push(VectorPoint {
                id: point.id.clone(),
                vectors,
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
    pub fn scroll_by_filter(
        &self,
        filter: &VectorFilter,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        self.scroll_inner(Some(filter), limit, offset, with_payload, with_vector)
    }
}
