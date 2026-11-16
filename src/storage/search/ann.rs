use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use bitvec::prelude::*;
use memmap2::Mmap;
use rayon::prelude::*;

use super::super::directory::DirView;
use super::super::index_lifecycle::PublishedIndex;
use super::super::keys::Keys;
use super::super::quant::{QuantQuery, QuantStore};
use super::super::vectors::Vectors;
use super::super::{CollectionStore, PayloadStore, TombstoneBits};
use crate::error::{Result, VectorSearchError};
use crate::index::hnsw::{HnswQuantCtx, HnswSearchContext};
use crate::index::IvfSearchContext;
use crate::index::{HnswIndex, IvfIndex};
use crate::metrics::{SearchPath, SearchRetry};
use crate::types::{PointId, SearchQuery, SearchResult};

use super::{ScoredSlot, SearchSnapshot};

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
            filter_mask: snap.filter_mask,
        };
        loop {
            let candidates = index.probe_candidates(&query.vector, nprobe, pending, &ivf_ctx)?;
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
        let quant_rescore;
        let held_guard: Option<parking_lot::RwLockReadGuard<'_, Option<QuantStore>>>;
        let held_snap: Option<arc_swap::Guard<Arc<Vec<Arc<memmap2::Mmap>>>>>;
        let held_qq: Option<QuantQuery<'_>>;
        let quant_enabled = self.has_quantization() && !quant_params.ignore && !quant_params.exact;
        if quant_enabled {
            let quant_guard = self.quant.read();
            if let Some(quant) = quant_guard.as_ref().filter(|q| q.is_ready()) {
                let qq = quant.encode_query(&query.vector, self.inner.read().meta.distance)?;
                let q_snap = quant.snapshot();
                // Widen ef by the oversampling factor so the cheaper quantized
                // traversal still yields the same candidate quality.
                let default_oversample = match &quant.config().quant_type {
                    crate::types::QuantizationType::Scalar { .. } => 2usize,
                    crate::types::QuantizationType::Binary { .. } => 4usize,
                    crate::types::QuantizationType::Product { .. } => 3usize,
                };
                let oversample = quant_params
                    .oversampling
                    .map(|o| (o.max(1.0).ceil() as usize).max(1))
                    .unwrap_or(default_oversample);
                ef = (ef.saturating_mul(oversample)).max(ef);
                quant_rescore = quant_params.default_rescore(&quant.config().quant_type);
                held_guard = Some(quant_guard);
                held_snap = Some(q_snap);
                held_qq = Some(qq);
            } else {
                // Quant exists but is not ready; traversal stays exact.
                quant_rescore = false;
                held_guard = Some(quant_guard);
                held_snap = None;
                held_qq = None;
            }
        } else {
            quant_rescore = false;
            held_guard = None;
            held_snap = None;
            held_qq = None;
        }
        let hnsw_ctx = HnswSearchContext {
            tombstones: Some(snap.tombstones),
            vectors: snap.vsnap,
            segment_slots: snap.segment_slots,
            filter_mask: snap.filter_mask,
            quant: match (&held_guard, &held_snap, &held_qq) {
                (Some(guard), Some(q_snap), Some(qq)) => Some(HnswQuantCtx {
                    store: guard.as_ref().expect("quant store held"),
                    snapshot: q_snap,
                    query: qq,
                }),
                _ => None,
            },
        };
        let candidates =
            index.probe_candidates(&query.vector, ef, query.effective_limit(), &hnsw_ctx)?;
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
            &query.vector,
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
            index.probe_candidates(&query.vector, ef, query.effective_limit(), &hnsw_ctx)?;
        let candidates = self.merge_pending(candidates, pending, query, snap);
        let candidates = self.rescore_exact(candidates, quant_rescore, query, snap);
        self.finish_candidates(candidates, query, snap)
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
                let dist = crate::distance::distance(metric, &query.vector, v);
                Some((crate::distance::to_score(metric, dist), slot))
            })
            .collect()
    }
}
