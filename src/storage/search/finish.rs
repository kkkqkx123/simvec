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
    /// Score pending slots by brute force against the live vectors and merge
    /// them into ANN candidates (deduplicating in favor of the first
    /// occurrence), so points inserted after a build stay visible.
    pub(super) fn merge_pending(
        &self,
        mut candidates: Vec<(f32, u32)>,
        pending: &[u32],
        query: &SearchQuery,
        snap: &SearchSnapshot<'_>,
    ) -> Vec<(f32, u32)> {
        if pending.is_empty() {
            return candidates;
        }
        let metric = self.inner.read().meta.distance;
        let scored: Vec<(f32, u32)> = pending
            .par_iter()
            .copied()
            .filter(|&s| !snap.tombstones.bit(s as usize))
            .filter(|&s| {
                snap.filter_mask
                    .is_none_or(|m| (s as usize) < m.len() && m[s as usize])
            })
            .filter_map(|s| {
                let v = Vectors::read_slot(snap.vsnap, s as u64, snap.segment_slots, snap.dim)?;
                let dist = crate::distance::distance(metric, &query.vector, v);
                Some((crate::distance::to_score(metric, dist), s))
            })
            .collect();
        let mut seen: HashSet<u32> = candidates.iter().map(|&(_, s)| s).collect();
        for scored @ (_, slot) in scored {
            if seen.insert(slot) {
                candidates.push(scored);
            }
        }
        candidates
    }

    /// Shared post-processing for the ANN paths: payload post-filter, score
    /// threshold, top-K heap selection and result assembly. Index probes
    /// only pre-filter (bitmap plus traversal), so the full payload check
    /// here is load-bearing. Exact and quantized scans filter inline during
    /// the scan and must use [`Self::finish_scored`] instead to avoid paying
    /// the key/payload reads a second time.
    pub(super) fn finish_candidates(
        &self,
        mut candidates: Vec<(f32, u32)>,
        query: &SearchQuery,
        snap: &SearchSnapshot,
    ) -> Result<Vec<SearchResult>> {
        let SearchSnapshot {
            keysnap,
            payload_view,
            ..
        } = snap;
        // 2. Post-filter on payload, against the snapshots taken by the
        // caller (one compaction generation).
        if let Some(filter) = &query.filter {
            let mut kept = Vec::with_capacity(candidates.len());
            for (score, slot) in candidates {
                let key = Keys::read_key(keysnap, slot as usize)?.ok_or_else(|| {
                    VectorSearchError::CorruptData(format!("slot {slot} has no key"))
                })?;
                let id = PointId::from(key);
                let payload = match payload_view {
                    Some(ps) => ps.get(slot)?,
                    None => self.read_payload_at(slot)?,
                };
                if crate::filter::matches(filter, &id, payload.as_ref())? {
                    kept.push((score, slot));
                }
            }
            candidates = kept;
        }

        self.finish_scored(candidates, query, snap)
    }

    /// Post-filter-free tail of [`Self::finish_candidates`]: score
    /// threshold, top-K heap selection and result assembly, for candidate
    /// sets whose payload filter already ran inline during the scan (exact
    /// and quantized paths).
    pub(super) fn finish_scored(
        &self,
        candidates: Vec<(f32, u32)>,
        query: &SearchQuery,
        snap: &SearchSnapshot,
    ) -> Result<Vec<SearchResult>> {
        let SearchSnapshot {
            dim,
            segment_slots,
            tombstones,
            vsnap,
            keysnap,
            payload_view,
            ..
        } = snap;
        let _ = tombstones;
        let mut candidates = candidates;
        // 3. Score threshold (lower bound on the output score).
        if let Some(threshold) = query.score_threshold {
            candidates.retain(|(score, _)| *score >= threshold);
        }

        // 4. Top-K by score (K = offset + limit), then sort descending.
        let k = query.offset.unwrap_or(0).saturating_add(query.limit);
        let top: Vec<(f32, u32)> = if candidates.len() <= k {
            candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            candidates
        } else {
            let mut heap: BinaryHeap<std::cmp::Reverse<ScoredSlot>> = BinaryHeap::new();
            for (score, slot) in candidates {
                let s = ScoredSlot { score, slot };
                if heap.len() < k {
                    heap.push(std::cmp::Reverse(s));
                } else if heap.peek().is_some_and(|top| s > top.0) {
                    heap.pop();
                    heap.push(std::cmp::Reverse(s));
                }
            }
            let mut v: Vec<(f32, u32)> = heap
                .into_iter()
                .map(|std::cmp::Reverse(s)| (s.score, s.slot))
                .collect();
            v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            v
        };

        // 5. Assemble results against the same snapshots the scan used.
        let with_payload = query
            .with_payload
            .unwrap_or(crate::types::DEFAULT_WITH_PAYLOAD);
        let with_vector = query.with_vector.unwrap_or(false);
        let mut results = Vec::new();
        for (score, slot) in top
            .into_iter()
            .skip(query.offset.unwrap_or(0))
            .take(query.limit)
        {
            let key = Keys::read_key(keysnap, slot as usize)?
                .ok_or_else(|| VectorSearchError::CorruptData(format!("slot {slot} has no key")))?;
            let mut result = SearchResult::new(PointId::from(key), score);
            if with_payload {
                let payload = match payload_view {
                    Some(ps) => ps.get(slot)?,
                    None => self.read_payload_at(slot)?,
                };
                if let Some(payload) = payload {
                    // Field projection trims the payload after the read; a
                    // selector that removes everything yields an empty map.
                    let payload = match &query.payload_selector {
                        Some(sel) => sel.apply(&payload),
                        None => payload,
                    };
                    result = result.with_payload(payload);
                }
            }
            if with_vector {
                let v = Vectors::read_slot(vsnap, slot as u64, *segment_slots, *dim).ok_or_else(
                    || {
                        VectorSearchError::CorruptData(format!(
                            "slot {slot} out of vectors.bin range"
                        ))
                    },
                )?;
                result = result.with_vector(v.to_vec());
            }
            results.push(result);
        }
        Ok(results)
    }
}
