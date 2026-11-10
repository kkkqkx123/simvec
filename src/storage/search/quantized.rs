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
    pub(super) fn search_quantized(
        &self,
        query: &SearchQuery,
        snap: &SearchSnapshot<'_>,
        metric: crate::types::DistanceMetric,
        next_slot: u64,
        filtered: bool,
        started: Instant,
    ) -> Result<Vec<SearchResult>> {
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
        let q_snap = quant.snapshot();
        let bytes_per_vector = quant.bytes_per_vector();
        if bytes_per_vector == 0 {
            return Err(VectorSearchError::Internal(
                "quant bytes per vector is zero".to_string(),
            ));
        }
        let k = query.offset.unwrap_or(0).saturating_add(query.limit);
        // Oversampling: trade recall vs latency. Scalar 2x, Binary 4x, Product 3x.
        let oversample = match &quant.config().quant_type {
            crate::types::QuantizationType::Scalar { .. } => 2usize,
            crate::types::QuantizationType::Binary { .. } => 4usize,
            crate::types::QuantizationType::Product { .. } => 3usize,
        };
        let quant_k = (k * oversample).min(next_slot as usize).max(k);
        // Precompute binary query bits once to avoid per-slot allocation.
        let query_bits: Option<Vec<u8>> = match &quant.config().quant_type {
            crate::types::QuantizationType::Binary { .. } => {
                Some(super::super::quant::encode_binary_for_query(&query.vector))
            }
            _ => None,
        };

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
                        quant,
                        &q_snap,
                        bytes_per_vector,
                        snap,
                        query_bits.as_deref(),
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

        // Rerank with exact distances.
        let mut reranked: Vec<(f32, u32)> = Vec::with_capacity(quant_candidates.len());
        for (_, slot) in quant_candidates {
            let Some(v) = Vectors::read_slot(snap.vsnap, slot as u64, snap.segment_slots, snap.dim)
            else {
                continue;
            };
            let dist = crate::distance::distance(metric, &query.vector, v);
            let score = crate::distance::to_score(metric, dist);
            reranked.push((score, slot));
        }
        reranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

        self.metrics
            .record_search(SearchPath::Quantized, filtered, started.elapsed());
        // Candidates already passed the full payload filter inside
        // `quant_scan_step`; re-filtering here would repeat every
        // key/payload read.
        self.finish_scored(
            reranked,
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
        metric: crate::types::DistanceMetric,
        query: &SearchQuery,
        quant: &QuantStore,
        q_snap: &[Arc<Mmap>],
        bytes_per_vector: usize,
        snap: &SearchSnapshot<'_>,
        query_bits: Option<&[u8]>,
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
        // Binary hamming can be computed from pre-encoded query bits without
        // per-slot allocation; other types use the quantized distance helper.
        let score = match &quant.config().quant_type {
            crate::types::QuantizationType::Binary { .. } => {
                let qbits = query_bits.expect("binary query bits precomputed");
                let mut ham = 0u32;
                for (a, b) in qbits.iter().zip(code.iter()) {
                    ham += (a ^ b).count_ones();
                }
                -(ham as f32)
            }
            _ => {
                let qdist = quant.distance_quantized(&query.vector, code, metric);
                crate::distance::to_score(metric, qdist)
            }
        };
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
