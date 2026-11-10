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
use super::super::quant::QuantStore;
use super::super::vectors::Vectors;
use super::super::{CollectionStore, PayloadStore, TombstoneBits};
use crate::error::{Result, VectorSearchError};
use crate::index::hnsw::HnswSearchContext;
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
        let hnsw_ctx = HnswSearchContext {
            tombstones: Some(snap.tombstones),
            vectors: snap.vsnap,
            segment_slots: snap.segment_slots,
            filter_mask: snap.filter_mask,
        };
        let candidates =
            index.probe_candidates(&query.vector, ef, query.effective_limit(), &hnsw_ctx)?;
        let candidates = self.merge_pending(candidates, pending, query, snap);

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
            return self.finish_candidates(candidates, query, snap);
        }

        // Fallback: double ef for a single controlled retry.
        self.metrics.record_search_retry(SearchRetry::EfDoubling);
        ef = (ef * 2).min(cap);
        let candidates =
            index.probe_candidates(&query.vector, ef, query.effective_limit(), &hnsw_ctx)?;
        let candidates = self.merge_pending(candidates, pending, query, snap);
        self.finish_candidates(candidates, query, snap)
    }
}
