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
use std::sync::Arc;
use std::time::Instant;

use bitvec::prelude::*;

use super::directory::DirView;
use super::index_lifecycle::PublishedIndex;
use super::{CollectionStore, PayloadStore, TombstoneBits};
use crate::error::{Result, VectorSearchError};
use crate::metrics::SearchPath;
use crate::types::{SearchQuery, SearchResult};

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
        query.validate_nonempty()?;
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
            if query.primary_vector().len() != inner.meta.vector_size {
                return Err(VectorSearchError::InvalidVectorDimension {
                    expected: inner.meta.vector_size,
                    actual: query.primary_vector().len(),
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
                // Highly selective filters match fewer slots than one layer-0
                // pass would score, so the exact scan underneath is cheaper
                // than the graph walk plus its retries and returns true top-k.
                let selective = snap.filter_mask.as_ref().is_some_and(|m| {
                    m.count_ones()
                        <= query.hnsw_ef().unwrap_or_else(|| index.default_ef())
                });
                if selective {
                    let results =
                        self.search_exact(query, &snap, metric, next_slot, filtered, started);
                    drop(published);
                    return results;
                }
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
        // traded for 4x-32x memory reduction. `quantization.ignore/exact`
        // opts out per query.
        let quant_params = query.quantization.clone().unwrap_or_default();
        if self.has_quantization() && !quant_params.ignore && !quant_params.exact {
            // Try the quantized path; if it fails (corrupt quant file) fall
            // back to exact scan transparently.
            let quant_result =
                self.search_quantized(query, &snap, metric, next_slot, filtered, started);
            if let Ok(results) = quant_result {
                return Ok(results);
            }
            // Fall through to exact on error.
        }

        self.search_exact(query, &snap, metric, next_slot, filtered, started)
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
