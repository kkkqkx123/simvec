use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::error::{Result, VectorSearchError};
use crate::filter;
use crate::types::query::reject_exact_query_controls;
use crate::types::{
    PointId, SearchQuery, SearchResult, VectorData, VectorFilter, VectorPoint, VectorSpaceConfig,
};

use super::{score_pair, sparse_scores, truncate_topk, NamedStore};
use crate::storage::point_blobs::named_entries;

impl NamedStore {
    /// Declared named-vector spaces.
    pub fn spaces(&self) -> BTreeMap<String, VectorSpaceConfig> {
        self.spaces
            .iter()
            .map(|(name, meta)| {
                let space = match meta.kind {
                    crate::types::VectorKind::Sparse => VectorSpaceConfig::sparse(meta.distance),
                    crate::types::VectorKind::Multi => VectorSpaceConfig::multi(
                        meta.vector_size,
                        meta.distance,
                        meta.multi_config.unwrap_or_else(|| crate::types::MultiConfig::new(1)),
                    ),
                    crate::types::VectorKind::Dense => {
                        VectorSpaceConfig::dense(meta.vector_size, meta.distance)
                    }
                };
                (name.clone(), space)
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

    /// Whether every sparse space's dictionary came from disk at open. `false`
    /// means at least one was rebuilt from the snapshot, which is what a crash
    /// between the manifest write and the snapshot leaves behind.
    pub fn postings_restored(&self) -> bool {
        self.postings.values().all(|space| space.was_restored())
    }

    /// (bytes of the vector blob file, bytes of all posting buckets).
    pub fn storage_bytes(&self) -> (u64, u64) {
        (
            self.blobs.len(),
            self.postings.values().map(|space| space.disk_bytes()).sum(),
        )
    }

    /// Committed JSON bytes held in `payloads.bin`, outside the snapshot.
    pub fn payload_bytes(&self) -> u64 {
        self.payloads.len()
    }

    /// Approximate resident size: the point directory, the norm caches, the
    /// posting dictionaries and the pending posting window. The vectors
    /// themselves live in the mapped blob file.
    pub fn memory_bytes(&self) -> u64 {
        let inner = self.inner.lock();
        let directory = inner.points.capacity() as u64
            * std::mem::size_of::<Option<super::NamedPoint>>() as u64;
        let norms: u64 = inner
            .norms
            .values()
            .map(|space| space.capacity() as u64 * std::mem::size_of::<(u32, f32)>() as u64)
            .sum();
        let postings: u64 = self
            .postings
            .values()
            .map(|space| space.memory_bytes())
            .sum();
        directory + norms + postings + self.blobs.mapped_bytes()
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
        let owned = data.clone().normalized();
        owned.validate(space.vector_size, max_vectors)?;

        let inner = self.inner.lock();
        let limit = query.effective_limit();
        let offset = query.offset.unwrap_or(0);
        if limit == 0 {
            return Ok(Vec::new());
        }
        // Filter verdicts are computed lazily and cached per slot; only slots
        // a posting reaches (or a full scan visits) are ever tested.
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

        let metric = space.distance;
        let threshold = query.score_threshold();
        let mut candidates: Vec<(u32, f32)> = Vec::new();
        let mut full_scan = false;
        if let VectorData::Sparse { indices, values } = &owned {
            let space_index = self.postings_of(target);
            let norms = inner.norms.get(target);
            if let (Some(space_index), Some(norms)) = (space_index, norms) {
                let hits = crate::kind_store::posting_hits(space_index, indices);
                full_scan = hits > inner.live_count.saturating_mul(2) as usize;
                if !full_scan {
                    candidates = sparse_scores(
                        space_index,
                        norms,
                        indices,
                        values,
                        metric,
                        Some(&mut accept),
                    )?;
                }
            } else {
                candidates = Vec::new();
            }
        }
        if full_scan || !matches!(owned, VectorData::Sparse { .. }) {
            candidates = (0..inner.points.len() as u32)
                .filter(|slot| inner.points[*slot as usize].is_some())
                .map(|slot| (slot, 0.0))
                .collect();
        }

        // One borrow of the blob mapping for the whole scoring pass.
        let reader = self.blobs.reader();
        let space_index = self.order_of(target);
        let mut scored: Vec<(u32, f32)> = Vec::new();
        for (slot, sparse_score) in candidates {
            if !accept(slot)? {
                continue;
            }
            let Some(point) = inner.points[slot as usize].as_ref() else {
                continue;
            };
            // A point missing the target space is not a candidate for it.
            let Some(index) = space_index else {
                continue;
            };
            let Some(stored) = reader.space_view(point.blob, index) else {
                continue;
            };
            let score = match &owned {
                VectorData::Sparse { .. } if full_scan => score_pair(&owned, &stored, metric),
                VectorData::Sparse { .. } => sparse_score,
                _ => score_pair(&owned, &stored, metric),
            };
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
                    let bytes = reader.bytes(point.blob)?;
                    for (index, entry) in named_entries(bytes).unwrap_or_default() {
                        let Some(space) = self.space_at(index) else {
                            continue;
                        };
                        if query
                            .vector_spaces
                            .as_ref()
                            .is_some_and(|allowed| !allowed.iter().any(|name| name == space))
                        {
                            continue;
                        }
                        if let Some(view) = crate::storage::point_blobs::decode_vector(entry) {
                            result = result.with_named_vector(space, view.to_data());
                        }
                    }
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
        // Each target is searched independently with its own filter, threshold
        // and per-target candidate budget; only the final paging is applied by
        // the query-layer fusion of these per-target rankings.
        let mut rankings: Vec<Vec<SearchResult>> = Vec::with_capacity(query.vectors.len());
        for target in &query.vectors {
            let mut single = query.clone();
            single.vectors = vec![target.clone()];
            rankings.push(self.search(&single)?);
        }
        let config = query.fusion.clone().unwrap_or_default();
        let limit = query.effective_limit();
        let offset = query.offset.unwrap_or(0);
        let results = crate::types::query::fuse_rankings(rankings, &config, offset, limit);
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
        let reader = self.blobs.reader();
        let Some(bytes) = reader.bytes(point.blob) else {
            return Ok(None);
        };
        let mut vectors = BTreeMap::new();
        for (index, entry) in named_entries(bytes).unwrap_or_default() {
            let Some(space) = self.space_at(index) else {
                continue;
            };
            if let Some(view) = crate::storage::point_blobs::decode_vector(entry) {
                vectors.insert(space.to_string(), view.to_data());
            }
        }
        drop(reader);
        Ok(Some(VectorPoint {
            id: point.id.clone(),
            vectors,
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
                .get(cursor)
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
            let vectors = if with_vector {
                let reader = self.blobs.reader();
                let mut vectors = BTreeMap::new();
                if let Some(bytes) = reader.bytes(point.blob) {
                    for (index, entry) in named_entries(bytes).unwrap_or_default() {
                        let Some(space) = self.space_at(index) else {
                            continue;
                        };
                        if let Some(view) = crate::storage::point_blobs::decode_vector(entry) {
                            vectors.insert(space.to_string(), view.to_data());
                        }
                    }
                }
                vectors
            } else {
                BTreeMap::new()
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
}
