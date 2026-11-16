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
use crate::index::hnsw::HnswSearchContext;
use crate::index::IvfSearchContext;
use crate::index::{HnswIndex, IvfIndex};
use crate::metrics::{SearchPath, SearchRetry};
use crate::types::{PointId, SearchQuery, SearchResult};

use super::{ScoredSlot, SearchSnapshot};

impl CollectionStore {
    pub(super) fn search_quantized(
        &self,
        query: &SearchQuery,
        snap: &SearchSnapshot<'_>,
        metric: crate::types::DistanceMetric,
        next_slot: u64,
        filtered: bool,
        started: Instant,
    ) -> Result<Vec<SearchResult>> {
        // Preprocess under a short-lived read lock; the parallel scan below
        // holds only an owned snapshot Arc, never the quant lock, so
        // concurrent rebuilds are never blocked by searches.
        let (bytes_per_vector, q_snap_arc, oversample, rescore, qq) = {
            let quant_guard = self.quant.read();
            let Some(quant) = quant_guard.as_ref() else {
                return Err(VectorSearchError::Internal(
                    "quantization not active".to_string(),
                ));
            };
            if !quant.is_ready() {
                return Err(VectorSearchError::Internal(
                    "quantization not ready".to_string(),
                ));
            }
            let bytes = quant.bytes_per_vector();
            if bytes == 0 {
                return Err(VectorSearchError::Internal(
                    "quant bytes per vector is zero".to_string(),
                ));
            }
            let default_oversample = match &quant.config().quant_type {
                crate::types::QuantizationType::Scalar { .. } => 2usize,
                crate::types::QuantizationType::Binary { .. } => 4usize,
                crate::types::QuantizationType::Product { .. } => 3usize,
            };
            let params = query.quantization.clone().unwrap_or_default();
            let oversample = params
                .oversampling
                .map(|o| (o.max(1.0).ceil() as usize).max(1))
                .unwrap_or(default_oversample);
            let rescore = params.default_rescore(&quant.config().quant_type);
            let snap: Arc<Vec<Arc<Mmap>>> = Arc::clone(&quant.snapshot());
            let qq = quant.encode_query(&query.vector, metric)?;
            (bytes, snap, oversample, rescore, qq)
        };
        let k = query.offset.unwrap_or(0).saturating_add(query.limit);
        let quant_k = (k * oversample).min(next_slot as usize).max(k);

        let heap: BinaryHeap<std::cmp::Reverse<ScoredSlot>> = (0..next_slot as u32)
            .into_par_iter()
            .filter(|s| !snap.tombstones.bit(*s as usize))
            .fold(
                || BinaryHeap::with_capacity(quant_k),
                |mut acc, s| {
                    self.quant_scan_step(
                        &mut acc,
                        s,
                        quant_k,
                        metric,
                        query,
                        &q_snap_arc,
                        bytes_per_vector,
                        snap,
                        &qq,
                    );
                    acc
                },
            )
            .reduce(
                || BinaryHeap::with_capacity(quant_k),
                |mut acc, other| {
                    for item in other.into_iter() {
                        if acc.len() < quant_k {
                            acc.push(item);
                        } else if let Some(std::cmp::Reverse(min)) = acc.peek() {
                            if item.0 > *min {
                                acc.pop();
                                acc.push(item);
                            }
                        }
                    }
                    acc
                },
            );

        let mut quant_candidates: Vec<(f32, u32)> = heap
            .into_iter()
            .map(|std::cmp::Reverse(s)| (s.score, s.slot))
            .collect();
        // Already sorted by quant score; now rerank the limited set with exact
        // vectors for guaranteed recall of the visible top-k.
        // If heap empty, fall back to exact (no candidates pass filter).
        if quant_candidates.is_empty() {
            self.metrics
                .record_search(SearchPath::Quantized, filtered, started.elapsed());
            return self.finish_scored(
                Vec::new(),
                query,
                &SearchSnapshot {
                    dim: snap.dim,
                    segment_slots: snap.segment_slots,
                    tombstones: snap.tombstones,
                    vsnap: snap.vsnap,
                    keysnap: snap.keysnap,
                    filter_mask: None,
                    payload_view: snap.payload_view,
                },
            );
        }
        quant_candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

        // Without rescoring, return quantized-score results directly
        // (truncated to k). Quantized scores are internally consistent with
        // the metric; Binary Hamming is only coarsely monotonic in the true
        // distance, which `QuantizationSearchParams::rescore` documents.
        let final_scored = if !rescore {
            quant_candidates.truncate(k);
            quant_candidates
        } else {
            // Rerank with exact distances for guaranteed recall of the
            // visible top-k.
            let mut reranked: Vec<(f32, u32)> = Vec::with_capacity(quant_candidates.len());
            for (_, slot) in quant_candidates {
                let Some(v) =
                    Vectors::read_slot(snap.vsnap, slot as u64, snap.segment_slots, snap.dim)
                else {
                    continue;
                };
                let dist = crate::distance::distance(metric, &query.vector, v);
                let score = crate::distance::to_score(metric, dist);
                reranked.push((score, slot));
            }
            reranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            reranked
        };

        self.metrics
            .record_search(SearchPath::Quantized, filtered, started.elapsed());
        // Candidates already passed the full payload filter inside
        // `quant_scan_step`; re-filtering here would repeat every
        // key/payload read.
        self.finish_scored(
            final_scored,
            query,
            &SearchSnapshot {
                dim: snap.dim,
                segment_slots: snap.segment_slots,
                tombstones: snap.tombstones,
                vsnap: snap.vsnap,
                keysnap: snap.keysnap,
                filter_mask: None,
                payload_view: snap.payload_view,
            },
        )
    }

    /// Hot inner-loop function; parameters are kept flat for zero-cost
    /// inlining in the parallel iterator closure.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn quant_scan_step(
        &self,
        heap: &mut BinaryHeap<std::cmp::Reverse<ScoredSlot>>,
        slot: u32,
        k: usize,
        _metric: crate::types::DistanceMetric,
        query: &SearchQuery,
        q_snap: &[Arc<Mmap>],
        bytes_per_vector: usize,
        snap: &SearchSnapshot<'_>,
        qq: &QuantQuery<'_>,
    ) {
        // Fast pre-filter via bitmap for equality payload conditions, when present.
        if let Some(mask) = snap.filter_mask {
            if (slot as usize) >= mask.len() || !mask[slot as usize] {
                return;
            }
        }
        // Full payload filter fallback when bitmap not built (range, compound, etc.).
        // When a bitmap is present it only covers equality predicates, so we
        // still need to verify the full filter against payload. For pure
        // equality filters this double-checks but keeps correctness simple; the
        // bitmap already eliminated most non-matching slots cheaply.
        if let Some(filter) = &query.filter {
            let Some(id) = Keys::read_key(snap.keysnap, slot as usize)
                .ok()
                .flatten()
                .map(PointId::from)
            else {
                return;
            };
            let payload = match snap.payload_view {
                Some(ps) => ps.get(slot).ok().flatten(),
                None => self.read_payload_at(slot).ok().flatten(),
            };
            let ok = crate::filter::matches(filter, &id, payload.as_ref()).unwrap_or(false);
            if !ok {
                return;
            }
        }
        let Some(code) =
            QuantStore::read_slot(q_snap, slot as u64, snap.segment_slots, bytes_per_vector)
        else {
            return;
        };
        // Hot path: preprocessed query + code bytes only, no locks.
        let score = QuantStore::score_quantized_for_query(qq, code);
        let item = ScoredSlot { score, slot };
        if heap.len() < k {
            heap.push(std::cmp::Reverse(item));
        } else if let Some(std::cmp::Reverse(min)) = heap.peek() {
            if item > *min {
                heap.pop();
                heap.push(std::cmp::Reverse(item));
            }
        }
    }
}
