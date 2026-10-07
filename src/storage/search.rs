//! Search execution for a collection: routing to the published ANN tier or
//! the exact parallel scan, plus shared post-processing.
//!
//! All paths take their metadata and per-file views inside one read-lock
//! acquisition (so a concurrent background compaction cannot produce a
//! mixed-generation view), run on immutable snapshots without locks, and
//! share identical post-processing through [`finish_candidates`]: payload
//! filter, score threshold, top-K, offset/limit. Scores and semantics are
//! therefore path-independent.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use bitvec::prelude::*;
use memmap2::Mmap;
use rayon::prelude::*;

use super::directory::DirView;
use super::index_lifecycle::PublishedIndex;
use super::keys::Keys;
use super::quant::QuantStore;
use super::vectors::Vectors;
use super::{CollectionStore, PayloadStore, TombstoneBits};
use crate::error::{Result, VectorSearchError};
use crate::index::hnsw::HnswSearchContext;
use crate::index::IvfSearchContext;
use crate::index::{HnswIndex, IvfIndex};
use crate::metrics::{SearchPath, SearchRetry};
use crate::types::{PointId, SearchQuery, SearchResult};

/// Snapshot of the immutable views used by a single search pass. All
/// fields are captured under one `inner.read()` acquisition so a concurrent
/// background compaction (which swaps every file under the write lock) cannot
/// produce a mixed-generation view. The payload store reference is captured
/// atomically via `ArcSwap::load()` to prevent compaction from swapping it
/// during the search.
struct SearchSnapshot<'a> {
    dim: usize,
    segment_slots: u32,
    tombstones: &'a TombstoneBits,
    vsnap: &'a [Arc<memmap2::Mmap>],
    keysnap: &'a DirView,
    filter_mask: Option<&'a BitVec>,
    /// Snapshot of the payload store. Held via a read lock for the search
    /// duration so a concurrent compaction cannot swap the store (and renumber
    /// slots) while we are reading payloads.
    payload_view: Option<&'a PayloadStore>,
}

impl CollectionStore {
    /// Search: route to the published ANN tier (HNSW by default) or the
    /// exact full scan. All paths share identical post-processing (payload
    /// filter, score threshold, top-K, offset/limit) through
    /// [`Self::finish_candidates`], so scores and semantics are
    /// path-independent.
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let started = Instant::now();
        // Take the metadata and all per-file views inside one read-lock
        // acquisition so a concurrent background compaction (which swaps
        // every file under the write lock) cannot produce a mixed-generation
        // view. Only the atomic loads happen under the lock; the actual scan
        // runs on immutable snapshots. `pending` must be captured here too:
        // a slot enters pending only after its key/payload/vector are
        // visible in the current file generations (both happen under the
        // store write lock), so a pending slot is always covered by these
        // snapshots. Loading it later could surface a slot whose key is
        // missing from the older key generation.
        //
        // The payload store is snapshotted atomically via ArcSwap so it
        // cannot be swapped by a concurrent compaction during the search.
        let (
            dim,
            metric,
            segment_slots,
            next_slot,
            tombstones,
            vsnap,
            keysnap,
            published,
            filter_mask,
            pending,
        ) = {
            let inner = self.inner.read();
            if query.vector.len() != inner.meta.vector_size {
                return Err(VectorSearchError::InvalidVectorDimension {
                    expected: inner.meta.vector_size,
                    actual: query.vector.len(),
                });
            }
            let mask = query.filter.as_ref().and_then(|f| {
                let capacity = inner.meta.slot_capacity as usize;
                // Prefer per-field index acceleration (numeric ranges plus
                // equality on declared fields); when no condition could be
                // accelerated fall back to the auto-populated equality
                // bitmap, and finally to no mask at all.
                match inner.payload_indexes.plan_filter(f, capacity) {
                    plan if plan.pre_mask.is_some() => plan.pre_mask,
                    _ => inner.filter_bitmap.build_mask(f),
                }
            });
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.segment_slots,
                inner.meta.next_slot,
                self.tombstones.load(),
                self.vectors.snapshot(),
                self.keys.snapshot(),
                self.index.load(),
                mask,
                self.pending.load_full(),
            )
        };
        let ps_guard = self.payload_store.load();
        let payload_view: Option<&PayloadStore> = (*ps_guard).as_ref().as_ref();
        let filtered = query.filter.is_some();
        let snap = SearchSnapshot {
            dim,
            segment_slots,
            tombstones: &tombstones,
            vsnap: &vsnap,
            keysnap: &keysnap,
            filter_mask: filter_mask.as_ref(),
            payload_view,
        };
        match published.as_ref() {
            Some(PublishedIndex::Ivf(index)) => {
                let results = self.search_ivf(index.as_ref(), query, &snap, &pending);
                self.metrics
                    .record_search(SearchPath::Ivf, filtered, started.elapsed());
                drop(published);
                return results;
            }
            Some(PublishedIndex::Hnsw(index)) => {
                let results = self.search_hnsw(index.as_ref(), query, &snap, &pending);
                self.metrics
                    .record_search(SearchPath::Hnsw, filtered, started.elapsed());
                drop(published);
                return results;
            }
            None => {}
        }
        drop(published);

        // Quantized path: when quantization is active and ready, do a coarse
        // quantized scan for `k * oversample` then rerank with exact vectors.
        // This is a derived storage optimization over the exact scan; recall is
        // traded for 4x-32x memory reduction.
        if self.has_quantization() {
            // Try the quantized path; if it fails (corrupt quant file) fall
            // back to exact scan transparently.
            let quant_result =
                self.search_quantized(query, &snap, metric, next_slot, filtered, started);
            if let Ok(results) = quant_result {
                return Ok(results);
            }
            // Fall through to exact on error.
        }

        // 1. Parallel exact scan with streaming top-K. The parallel heap
        //    avoids materializing the full (score, slot) set when the
        //    collection is large, cutting peak memory to O(k).
        let k = query.offset.unwrap_or(0).saturating_add(query.limit);
        let heap: BinaryHeap<std::cmp::Reverse<ScoredSlot>> = (0..next_slot as u32)
            .into_par_iter()
            .filter(|s| !tombstones.bit(*s as usize))
            .fold(
                || BinaryHeap::with_capacity(k),
                |mut acc, s| {
                    self.exact_scan_step(
                        &mut acc,
                        s,
                        k,
                        metric,
                        query,
                        &SearchSnapshot {
                            dim,
                            segment_slots,
                            tombstones: &tombstones,
                            vsnap: &vsnap,
                            keysnap: &keysnap,
                            filter_mask: None,
                            payload_view,
                        },
                    );
                    acc
                },
            )
            .reduce(
                || BinaryHeap::with_capacity(k),
                |mut acc, other| {
                    for item in other.into_iter() {
                        if acc.len() < k {
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

        let mut candidates: Vec<(f32, u32)> = heap
            .into_iter()
            .map(|std::cmp::Reverse(s)| (s.score, s.slot))
            .collect();
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

        self.metrics
            .record_search(SearchPath::Exact, filtered, started.elapsed());
        self.finish_candidates(
            candidates,
            query,
            &SearchSnapshot {
                dim,
                segment_slots,
                tombstones: &tombstones,
                vsnap: &vsnap,
                keysnap: &keysnap,
                filter_mask: None,
                payload_view,
            },
        )
    }

    fn search_quantized(
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
        let oversample = match quant.config().quant_type.as_ref() {
            Some(crate::types::QuantizationType::Scalar { .. }) => 2usize,
            Some(crate::types::QuantizationType::Binary { .. }) => 4usize,
            Some(crate::types::QuantizationType::Product { .. }) => 3usize,
            None => 2usize,
        };
        let quant_k = (k * oversample).min(next_slot as usize).max(k);
        // Precompute binary query bits once to avoid per-slot allocation.
        let query_bits: Option<Vec<u8>> = match quant.config().quant_type.as_ref() {
            Some(crate::types::QuantizationType::Binary { .. }) => {
                Some(super::quant::encode_binary_for_query(&query.vector))
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
            return self.finish_candidates(
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
        self.finish_candidates(
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
    fn quant_scan_step(
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
        let score = match quant.config().quant_type.as_ref() {
            Some(crate::types::QuantizationType::Binary { .. }) => {
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

    /// Single live-slot evaluation for the streaming exact scan.
    ///
    /// The payload filter is evaluated inline so the heap is only ever
    /// populated with candidates that pass the filter; stale data (missing
    /// key/payload) is silently skipped, mirroring the IVF index path's
    /// `filter_map` behaviour.
    fn exact_scan_step(
        &self,
        heap: &mut BinaryHeap<std::cmp::Reverse<ScoredSlot>>,
        slot: u32,
        k: usize,
        metric: crate::types::DistanceMetric,
        query: &SearchQuery,
        snap: &SearchSnapshot<'_>,
    ) {
        let SearchSnapshot {
            keysnap,
            payload_view,
            ..
        } = snap;
        if let Some(filter) = &query.filter {
            let Some(id) = Keys::read_key(keysnap, slot as usize)
                .ok()
                .flatten()
                .map(PointId::from)
            else {
                return;
            };
            let payload = match payload_view {
                Some(ps) => ps.get(slot).ok().flatten(),
                None => self.read_payload_at(slot).ok().flatten(),
            };
            let ok = crate::filter::matches(filter, &id, payload.as_ref()).unwrap_or(false);
            if !ok {
                return;
            }
        }
        let Some(v) = Vectors::read_slot(snap.vsnap, slot as u64, snap.segment_slots, snap.dim)
        else {
            return;
        };
        let dist = crate::distance::distance(metric, &query.vector, v);
        let score = crate::distance::to_score(metric, dist);
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

    /// IVF path: probe the `nprobe` closest lists (+ pending slots), then
    /// shared post-processing. With a payload filter that leaves fewer than
    /// `limit` results, the probe width keeps doubling (each widening
    /// recorded as a [`SearchRetry::NprobeDoubling`]) until the result count
    /// is met, every list has been probed, or `IvfConfig::max_probes` caps
    /// the widening — pgvector's multi-round iterative scan semantics.
    fn search_ivf(
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
    fn search_hnsw(
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

    /// Score pending slots by brute force against the live vectors and merge
    /// them into ANN candidates (deduplicating in favor of the first
    /// occurrence), so points inserted after a build stay visible.
    fn merge_pending(
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

    /// Shared post-processing for both search paths: payload post-filter,
    /// score threshold, top-K heap selection and result assembly.
    fn finish_candidates(
        &self,
        mut candidates: Vec<(f32, u32)>,
        query: &SearchQuery,
        snap: &SearchSnapshot,
    ) -> Result<Vec<SearchResult>> {
        let SearchSnapshot {
            dim,
            segment_slots,
            tombstones,
            vsnap,
            keysnap,
            filter_mask,
            payload_view,
        } = snap;
        let _ = tombstones;
        let _ = filter_mask;
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

/// A (score, slot) pair ordered by score for the top-K heap.
#[derive(Clone, Copy, Debug)]
struct ScoredSlot {
    score: f32,
    slot: u32,
}

impl PartialEq for ScoredSlot {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.slot == other.slot
    }
}

impl Eq for ScoredSlot {}

impl PartialOrd for ScoredSlot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredSlot {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then(self.slot.cmp(&other.slot))
    }
}
