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

mod ann;
mod exact;
mod finish;
mod quantized;


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
                self.pending.lock().clone(),
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
        //    collection is large, cutting peak memory to O(k). The heap only
        //    ever holds slots that already passed the full payload filter
        //    (bitmap pre-check plus `matches`), so the shared post-processing
        //    below must not filter a second time.
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
                            filter_mask: snap.filter_mask,
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
        self.finish_scored(
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
