use std::sync::Arc;

use super::super::quant::QuantQuery;
use super::super::vectors::Vectors;
use super::super::CollectionStore;
use crate::error::Result;
use crate::index::hnsw::{HnswQuantCtx, HnswSearchContext};
use crate::index::IvfSearchContext;
use crate::index::{HnswIndex, IvfIndex};
use crate::metrics::{SearchPath, SearchRetry};
use crate::types::{SearchQuery, SearchResult};

use super::SearchSnapshot;

impl CollectionStore {
    /// IVF path: probe the `nprobe` closest lists (+ pending slots), then
    /// shared post-processing. With a payload filter that leaves fewer than
    /// `limit` results, the probe width keeps doubling (each widening
    /// recorded as a [`SearchRetry::NprobeDoubling`]) until the result count
    /// is met, every list has been probed, or `IvfConfig::max_probes` caps
    /// the widening — pgvector's multi-round iterative scan semantics.
    pub(super) fn search_ivf(
        &self,
        index: &IvfIndex,
        query: &SearchQuery,
        snap: &SearchSnapshot<'_>,
        pending: &[u32],
    ) -> Result<Vec<SearchResult>> {
        let mut nprobe = index.clamp_nprobe(query.nprobe);
        let cap = index.max_probe_cap();

        let ivf_ctx = IvfSearchContext {
            tombstones: snap.tombstones,
            vectors: snap.vsnap,
            segment_slots: snap.segment_slots,
            slot_base: 0,
            filter_mask: snap.filter_mask,
        };
        loop {
            let candidates = index.probe_candidates(query.primary_vector(), nprobe, pending, &ivf_ctx)?;
            let results = self.finish_candidates(candidates, query, snap)?;

            let short = query.filter.is_some() && results.len() < query.limit;
            if !short || nprobe >= cap {
                return Ok(results);
            }
            self.metrics
                .record_search_retry(SearchRetry::NprobeDoubling);
            nprobe = (nprobe * 2).min(cap);
        }
    }

    /// Probe every published IVF segment and merge the results before the
    /// shared filter / score / top-K tail. Probe width doubling is applied to
    /// all segments together so segment order cannot change the answer.
    pub(super) fn search_segmented_ivf(
        &self,
        query: &SearchQuery,
        snap: &SearchSnapshot<'_>,
        pending: &[u32],
    ) -> Result<Vec<SearchResult>> {
        let first = &snap.ivf_segments[0].index;
        let mut nprobe = first.clamp_nprobe(query.nprobe);
        let cap = first.max_probe_cap();
        loop {
            let mut candidates = Vec::new();
            for segment in snap.ivf_segments {
                let ctx = IvfSearchContext {
                    tombstones: snap.tombstones,
                    vectors: snap.vsnap,
                    segment_slots: snap.segment_slots,
                    slot_base: segment.start,
                    filter_mask: snap.filter_mask,
                };
                candidates.extend(segment.index.probe_candidates(
                    query.primary_vector(),
                    nprobe,
                    &[],
                    &ctx,
                )?);
            }
            let candidates = self.merge_pending(candidates, pending, query, snap);
            let results = self.finish_candidates(candidates, query, snap)?;
            if query.filter.is_none() || results.len() >= query.limit || nprobe >= cap {
                return Ok(results);
            }
            self.metrics
                .record_search_retry(SearchRetry::NprobeDoubling);
            nprobe = (nprobe * 2).min(cap);
        }
    }

    /// HNSW path: layered-graph search with `ef` from the query's
    /// `SearchMode::KNN.ef_search`, falling back to the index default. With a
    /// payload filter that leaves fewer than `limit` results, iterative scan
    /// expansion is attempted before falling back to doubling `ef`.
    ///
    /// Pending slots (not yet incorporated into the graph) are scored by
    /// brute force and merged with graph candidates, ensuring visibility
    /// matches the IVF path's `extra_slots` semantics.
    pub(super) fn search_hnsw(
        &self,
        index: &HnswIndex,
        query: &SearchQuery,
        snap: &SearchSnapshot<'_>,
        pending: &[u32],
    ) -> Result<Vec<SearchResult>> {
        let mut ef = query.hnsw_ef().unwrap_or_else(|| index.default_ef());
        let quant_params = query.quantization.clone().unwrap_or_default();
        // Quantized graph traversal: distances during the graph walk use
        // quantized codes (qdrant-style); survivors are rescored with exact
        // vectors below when the method's rescore default (or the query's
        // override) says so. `quantization.ignore/exact` opts out per query.
        // Quant setup runs under a short-lived read lock: preprocess the
        // query and clone the code snapshot, then release the lock before
        // the graph walk so concurrent rebuilds are never blocked.
        let mut quant_bytes = 0usize;
        let mut quant_snap_arc: Option<Arc<Vec<Arc<memmap2::Mmap>>>> = None;
        let mut held_qq: Option<QuantQuery<'_>> = None;
        let mut quant_rescore = false;
        let quant_enabled = self.has_quantization() && !quant_params.ignore && !quant_params.exact;
        if quant_enabled {
            let setup = {
                let data = self.data_arc();
                let quant_guard = data.quant.read();
                quant_guard.as_ref().filter(|q| q.is_ready()).and_then(|quant| {
                    let metric = self.inner.read().meta.distance;
                    match quant.encode_query(query.primary_vector(), metric) {
                        Err(e) => {
                            tracing::warn!(error = %e, "quant encode_query failed; HNSW traversal falls back to exact");
                            None
                        }
                        Ok(qq) => {
                            let bytes = quant.bytes_per_vector();
                            let snap: Arc<Vec<Arc<memmap2::Mmap>>> =
                                Arc::clone(&quant.snapshot());
                            let default_oversample = match &quant.config().quant_type {
                                crate::types::QuantizationType::Scalar { .. } => 2usize,
                                crate::types::QuantizationType::Binary { .. } => 4usize,
                                crate::types::QuantizationType::Product { .. } => 3usize,
                            };
                            let oversample = quant_params
                                .oversampling
                                .map(|o| (o.max(1.0).ceil() as usize).max(1))
                                .unwrap_or(default_oversample);
                            let rescore =
                                quant_params.default_rescore(&quant.config().quant_type);
                            Some((bytes, snap, qq, rescore, oversample))
                        }
                    }
                })
            };
            if let Some((bytes, snap, qq, rescore, oversample)) = setup {
                ef = (ef.saturating_mul(oversample)).max(ef);
                quant_rescore = rescore;
                quant_bytes = bytes;
                quant_snap_arc = Some(snap);
                held_qq = Some(qq);
            }
        }
        let hnsw_ctx = HnswSearchContext {
            tombstones: Some(snap.tombstones),
            vectors: snap.vsnap,
            segment_slots: snap.segment_slots,
            slot_base: 0,
            filter_mask: snap.filter_mask,
            quant: match (&quant_snap_arc, &held_qq) {
                (Some(snap), Some(qq)) => Some(HnswQuantCtx {
                    bytes_per_vector: quant_bytes,
                    slot_base: 0,
                    snapshot: snap,
                    query: qq,
                }),
                _ => None,
            },
        };
        let candidates =
            index.probe_candidates(query.primary_vector(), ef, query.effective_limit(), &hnsw_ctx)?;
        let candidates = self.merge_pending(candidates, pending, query, snap);
        let candidates = self.rescore_exact(candidates, quant_rescore, query, snap);

        let results = self.finish_candidates(candidates, query, snap)?;

        let short = query.filter.is_some() && results.len() < query.limit;
        if !short {
            return Ok(results);
        }
        // Iterative expansion: try resuming from discarded candidates before
        // doubling ef. This mirrors pgvector's iterative scan which recovers
        // evicted candidates rather than widening the search window.
        let cap = index.node_count().max(1);
        if ef >= cap {
            return Ok(results);
        }
        self.metrics
            .record_search_retry(SearchRetry::IterativeExpansion);
        let iterative_results = index.probe_candidates_iterative(
            query.primary_vector(),
            ef,
            query.effective_limit(),
            index.iterative_rounds(),
            index.scan_tuple_budget(),
            &hnsw_ctx,
        );
        if let Ok(iterative_candidates) = iterative_results {
            let candidates = self.merge_pending(iterative_candidates, pending, query, snap);
            let candidates = self.rescore_exact(candidates, quant_rescore, query, snap);
            return self.finish_candidates(candidates, query, snap);
        }

        // Fallback: double ef for a single controlled retry.
        self.metrics.record_search_retry(SearchRetry::EfDoubling);
        ef = (ef * 2).min(cap);
        let candidates =
            index.probe_candidates(query.primary_vector(), ef, query.effective_limit(), &hnsw_ctx)?;
        let candidates = self.merge_pending(candidates, pending, query, snap);
        let candidates = self.rescore_exact(candidates, quant_rescore, query, snap);
        self.finish_candidates(candidates, query, snap)
    }

    /// Probe every published segment graph with its own local-to-global slot
    /// base and merge the candidates before shared post-processing.
    pub(super) fn probe_segmented_hnsw(
        &self,
        query: &SearchQuery,
        ef: usize,
        snap: &SearchSnapshot<'_>,
        quant: Option<(&QuantQuery<'_>, usize)>,
    ) -> Result<Vec<(f32, u32)>> {
        let mut candidates = Vec::new();
        for segment in snap.hnsw_segments {
            let quant_ctx = match (quant, segment.quant.as_ref()) {
                (Some((encoded, bytes)), Some(snapshot)) => Some(HnswQuantCtx {
                    bytes_per_vector: bytes,
                    slot_base: segment.start,
                    snapshot,
                    query: encoded,
                }),
                _ => None,
            };
            let ctx = HnswSearchContext {
                tombstones: Some(snap.tombstones),
                vectors: snap.vsnap,
                segment_slots: snap.segment_slots,
                slot_base: segment.start,
                filter_mask: snap.filter_mask,
                quant: quant_ctx,
            };
            candidates.extend(segment.index.probe_candidates(
                query.primary_vector(),
                ef,
                query.effective_limit(),
                &ctx,
            )?);
        }
        Ok(candidates)
    }

    /// Segmented HNSW search: each completed segment contributes graph
    /// candidates and the appendable tail contributes exact-scanned pending
    /// slots. A short filtered result widens `ef` once, applied uniformly to
    /// every segment so segment order cannot change the answer.
    pub(super) fn search_segmented_hnsw(
        &self,
        query: &SearchQuery,
        snap: &SearchSnapshot<'_>,
        pending: &[u32],
        started: std::time::Instant,
    ) -> Result<Vec<SearchResult>> {
        let mut ef = query
            .hnsw_ef()
            .unwrap_or_else(|| snap.hnsw_segments[0].index.default_ef());
        let quant_params = query.quantization.clone().unwrap_or_default();
        let quant_enabled = self.has_quantization() && !quant_params.ignore && !quant_params.exact;
        let mut rescore = false;
        let mut encoded = None;
        let mut quant_bytes = 0usize;
        if quant_enabled {
            let data = self.data_arc();
            let guard = data.quant.read();
            if let Some(quant) = guard.as_ref().filter(|q| q.is_ready()) {
                match quant.encode_query(query.primary_vector(), self.inner.read().meta.distance) {
                    Ok(q) => {
                        quant_bytes = quant.bytes_per_vector();
                        let default_oversample = match &quant.config().quant_type {
                            crate::types::QuantizationType::Scalar { .. } => 2usize,
                            crate::types::QuantizationType::Binary { .. } => 4usize,
                            crate::types::QuantizationType::Product { .. } => 3usize,
                        };
                        let oversample = quant_params
                            .oversampling
                            .map(|o| (o.max(1.0).ceil() as usize).max(1))
                            .unwrap_or(default_oversample);
                        ef = ef.saturating_mul(oversample).max(ef);
                        rescore = quant_params.default_rescore(&quant.config().quant_type);
                        encoded = Some(q);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "quant encode_query failed; segment graphs use exact traversal")
                    }
                }
            }
        }
        let quant = encoded
            .as_ref()
            .filter(|_| snap.hnsw_segments.iter().any(|s| s.quant.is_some()))
            .map(|q| (q, quant_bytes));
        let do_rescore = quant.is_some() && rescore;
        let candidates = self.probe_segmented_hnsw(query, ef, snap, quant)?;
        let candidates = self.merge_pending(candidates, pending, query, snap);
        let candidates = self.rescore_exact(candidates, do_rescore, query, snap);
        let results = self.finish_candidates(candidates, query, snap)?;
        if query.filter.is_none()
            || results.len() >= query.limit
            || ef >= snap.hnsw_segments[0].index.node_count()
        {
            self.metrics
                .record_search(SearchPath::Hnsw, true, started.elapsed());
            return Ok(results);
        }
        ef = (ef * 2).min(snap.hnsw_segments[0].index.node_count().max(1));
        let candidates = self.probe_segmented_hnsw(query, ef, snap, quant)?;
        let candidates = self.merge_pending(candidates, pending, query, snap);
        let candidates = self.rescore_exact(candidates, do_rescore, query, snap);
        let results = self.finish_candidates(candidates, query, snap)?;
        self.metrics
            .record_search(SearchPath::Hnsw, true, started.elapsed());
        Ok(results)
    }

    /// Rescore ANN candidates with exact f32 vectors after quantized graph
    /// traversal, so returned scores and ordering are exact. No-op when the
    /// traversal was already exact (`do_rescore == false`).
    pub(super) fn rescore_exact(
        &self,
        candidates: Vec<(f32, u32)>,
        do_rescore: bool,
        query: &SearchQuery,
        snap: &SearchSnapshot<'_>,
    ) -> Vec<(f32, u32)> {
        if !do_rescore {
            return candidates;
        }
        let metric = self.inner.read().meta.distance;
        candidates
            .into_iter()
            .filter_map(|(_, slot)| {
                let v = Vectors::read_slot(snap.vsnap, slot as u64, snap.segment_slots, snap.dim)?;
                let dist = crate::distance::distance(metric, query.primary_vector(), v);
                Some((crate::distance::to_score(metric, dist), slot))
            })
            .collect()
    }
}
