use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::error::{Result, VectorSearchError};
use crate::index::persist::{PersistedHnsw, PersistedNodeRecord};
use crate::index::IndexBuildParams;
use crate::metrics::{timed_write_lock, Metrics};
use crate::storage::vectors::Vectors;
use crate::storage::TombstoneBits;
use crate::types::{DistanceMetric, HnswConfig};

use bitvec::prelude::*;

use super::*;

impl HnswIndex {
    /// Greedy descent with `ef = 1`: repeatedly move to the closest neighbor
    /// until no improvement. Equivalent to running paper Algorithm 2 with
    /// `ef = 1`, which is what pgvector uses between upper layers.
    pub(super) fn greedy_step(
        &self,
        query: &[f32],
        start: Cand,
        layer: u8,
        vectors: &[Arc<memmap2::Mmap>],
        segment_slots: u32,
    ) -> Cand {
        let mut best = start;
        loop {
            let Some(node) = self.node(best.slot) else {
                return best;
            };
            let neighbors = node.layer(layer).read().clone();
            let mut improved = false;
            for slot in neighbors {
                let Some(v) = Vectors::read_slot(vectors, slot as u64, segment_slots, self.dim)
                else {
                    continue;
                };
                let dist = self.distance(query, v);
                if (Cand { dist, slot }).cmp(&best) == std::cmp::Ordering::Less {
                    best = Cand { dist, slot };
                    improved = true;
                }
            }
            if !improved {
                return best;
            }
        }
    }

    /// Paper Algorithm 2 (SEARCH-LAYER).
    ///
    /// `tombstones` filters which elements count towards `ef` and appear in
    /// the result; tombstoned nodes stay fully traversable so the graph does
    /// not fragment around deletions. `None` (insert-time search) counts
    /// every element, matching pgvector's build behavior.
    /// Paper Algorithm 2 (SEARCH-LAYER).
    ///
    /// `tombstones` filters which elements count towards `ef` and appear in
    /// the result; tombstoned nodes stay fully traversable so the graph does
    /// not fragment around deletions. `None` (insert-time search) counts
    /// every element, matching pgvector's build behavior.
    ///
    /// `filter_mask` (pre-filter bitmap) constrains which elements count
    /// towards `ef` and appear in the result, while the traversal frontier
    /// still walks through every live node. Mirroring pgvector's filter-aware
    /// scan this way keeps mismatched regions navigable: hard-pruning them
    /// from the frontier would disconnect the search whenever the entry seed
    /// lands outside the matching subgraph.
    ///
    /// This implementation detects concurrent adjacency mutations via the
    /// per-node `version` counter (mirroring pgvector's 4-bit version field).
    /// When a version change is detected between the snapshot and the load,
    /// the neighborhood is reloaded once to avoid using a torn adjacency list.
    pub(super) fn search_layer(
        &self,
        query: &[f32],
        entries: &[Cand],
        ef: usize,
        layer: u8,
        ctx: &HnswSearchContext<'_>,
    ) -> SearchLayerResult {
        QUERY_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            self.search_layer_with_scratch(query, entries, ef, layer, ctx, &mut scratch)
        })
    }

    pub(super) fn search_layer_with_scratch(
        &self,
        query: &[f32],
        entries: &[Cand],
        ef: usize,
        layer: u8,
        ctx: &HnswSearchContext<'_>,
        scratch: &mut HnswQueryScratch,
    ) -> SearchLayerResult {
        let live = |slot: u32| -> bool {
            match ctx.tombstones {
                Some(t) => !t.bit(slot as usize),
                None => true,
            }
        };
        let matches_filter = |slot: u32| -> bool {
            ctx.filter_mask
                .is_none_or(|m| (slot as usize) < m.len() && m[slot as usize])
        };

        // Working sets reuse the per-thread capacity: the generation array
        // compares generations instead of clearing, and the heaps keep
        // their allocations across queries on this thread.
        let node_count = self.nodes.read().len();
        scratch.prepare(node_count.max(entries.len() + ef));
        let initial_cap = (entries.len() + ef).max(16);
        if scratch.candidates.capacity() < initial_cap {
            scratch
                .candidates
                .reserve(initial_cap - scratch.candidates.capacity());
        }
        if scratch.results.capacity() < ef {
            scratch.results.reserve(ef - scratch.results.capacity());
        }
        let mut visited_count = 0usize;
        let mut live_results = 0usize;

        for &cand in entries {
            if !live(cand.slot) || !scratch.mark_visited(cand.slot) {
                continue;
            }
            visited_count += 1;
            scratch.candidates.push(std::cmp::Reverse(cand));
            if matches_filter(cand.slot) {
                scratch.results.push(cand);
                live_results += 1;
            }
        }

        while let Some(std::cmp::Reverse(c)) = scratch.candidates.pop() {
            if live_results >= ef {
                match scratch.results.peek() {
                    // Enough matches collected and this frontier node is
                    // farther than the worst match: the rest of the frontier
                    // cannot improve the output.
                    Some(furthest) if c.dist > furthest.dist => break,
                    _ => {}
                }
            }
            let Some(node) = self.node(c.slot) else {
                continue;
            };
            // Snapshot the version before loading the adjacency list. The
            // list is copied into the per-thread neighbor buffer so the hot
            // loop performs no per-node allocation; the read guard is dropped
            // before distance math so writers are never held across it. If
            // the version changed during the copy, the list may be torn, so
            // it is copied once more.
            let v1 = node.version();
            scratch.neighbors.clear();
            scratch
                .neighbors
                .extend_from_slice(&node.layer(layer).read());
            let v2 = node.version();
            if v1 != v2 {
                // Version changed during the read; copy once more. Counted so
                // concurrency regression tests can assert this recovery path
                // is actually exercised under write/read contention.
                if let Some(metrics) = &self.lock_metrics {
                    metrics.record_version_reload();
                }
                scratch.neighbors.clear();
                scratch
                    .neighbors
                    .extend_from_slice(&node.layer(layer).read());
            }
            for idx in 0..scratch.neighbors.len() {
                let slot = scratch.neighbors[idx];
                if !scratch.mark_visited(slot) || !live(slot) {
                    continue;
                }
                visited_count += 1;
                let Some(v) =
                    Vectors::read_slot(ctx.vectors, slot as u64, ctx.segment_slots, self.dim)
                else {
                    continue;
                };
                let dist = self.distance(query, v);
                let cand = Cand { dist, slot };
                scratch.candidates.push(std::cmp::Reverse(cand));
                if !matches_filter(slot) {
                    continue;
                }
                scratch.results.push(cand);
                live_results += 1;
                // Trim back to the `ef` budget. Evicted live candidates
                // are recorded in `discarded` for iterative scan resume.
                while scratch.results.len() > ef {
                    let Some(evicted) = scratch.results.pop() else {
                        break;
                    };
                    live_results -= 1;
                    scratch.discarded.push(evicted);
                }
            }
        }

        // Emit only live elements, closest first. The working heaps stay in
        // the scratch for the next query and the discard buffer is split off
        // so it keeps its capacity too; only the compact outputs allocate.
        let mut live_out: Vec<Cand> = scratch.results.drain().filter(|c| live(c.slot)).collect();
        live_out.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
        let discarded_out = scratch.discarded.split_off(0);
        SearchLayerResult {
            live: live_out,
            discarded: discarded_out,
            visited: visited_count,
        }
    }

    /// Approximate kNN: greedy descent to layer 1, full search at layer 0.
    /// Returns `(score, slot)` pairs in output-score space (higher is
    /// better), unsorted; shared post-processing stays in `CollectionStore`.
    pub(crate) fn probe_candidates(
        &self,
        query: &[f32],
        ef: usize,
        k: usize,
        ctx: &HnswSearchContext<'_>,
    ) -> Result<Vec<(f32, u32)>> {
        let Some(entry) = *self.entry.read() else {
            return Ok(Vec::new());
        };
        let Some(entry_vec) =
            Vectors::read_slot(ctx.vectors, entry.slot as u64, ctx.segment_slots, self.dim)
        else {
            return Ok(Vec::new());
        };

        let mut best = Cand {
            dist: self.distance(query, entry_vec),
            slot: entry.slot,
        };
        let mut lc = entry.level;
        while lc > 0 {
            best = self.greedy_step(query, best, lc, ctx.vectors, ctx.segment_slots);
            lc -= 1;
        }

        let w = self.search_layer(query, &[best], ef.max(k), 0, ctx);
        Ok(w.live
            .into_iter()
            .map(|c| (crate::distance::to_score(self.metric, c.dist), c.slot))
            .collect())
    }

    /// Iterative kNN: like `probe_candidates` but resumes from discarded
    /// candidates when the initial search yields fewer than `k` results.
    /// Each iteration feeds the previous round's discarded candidates as new
    /// entry points, expanding the search frontier until `k` results are
    /// found, `max_iterations` is exhausted, or the cumulative visited-node
    /// budget (`max_scan_tuples`) runs out.
    ///
    /// This mirrors pgvector's iterative scan (`hnswscan.c`): when
    /// `ef_search` is too small to cover the result set, the search resumes
    /// from candidates that were evicted from the live set rather than simply
    /// doubling `ef`.
    pub(crate) fn probe_candidates_iterative(
        &self,
        query: &[f32],
        ef: usize,
        k: usize,
        max_iterations: usize,
        max_scan_tuples: Option<u64>,
        ctx: &HnswSearchContext<'_>,
    ) -> Result<Vec<(f32, u32)>> {
        let Some(entry) = *self.entry.read() else {
            return Ok(Vec::new());
        };
        let Some(entry_vec) =
            Vectors::read_slot(ctx.vectors, entry.slot as u64, ctx.segment_slots, self.dim)
        else {
            return Ok(Vec::new());
        };

        let mut best = Cand {
            dist: self.distance(query, entry_vec),
            slot: entry.slot,
        };
        let mut lc = entry.level;
        while lc > 0 {
            best = self.greedy_step(query, best, lc, ctx.vectors, ctx.segment_slots);
            lc -= 1;
        }

        let mut all_results: Vec<Cand> = Vec::with_capacity(k + ef * max_iterations.max(1));
        let mut all_discarded: Vec<Cand> = Vec::with_capacity(ef * max_iterations.max(1));
        let mut current_entries = vec![best];
        let mut visited_total: u64 = 0;

        for _ in 0..max_iterations {
            let result = self.search_layer(query, &current_entries, ef, 0, ctx);
            all_results.extend(result.live);
            all_discarded.extend(result.discarded);
            visited_total += result.visited as u64;

            if all_results.len() >= k {
                break;
            }
            if all_discarded.is_empty() {
                break;
            }
            // Budget check after the pass so a single round is never skipped,
            // mirroring pgvector's "finish the current batch, then re-check"
            // scan-memory behavior.
            if max_scan_tuples.is_some_and(|cap| visited_total >= cap) {
                break;
            }
            // Use discarded candidates as entry points for the next iteration.
            // Swap the buffers instead of taking the accumulator so both keep
            // their capacity across rounds instead of reallocating each pass.
            current_entries.clear();
            std::mem::swap(&mut current_entries, &mut all_discarded);
        }

        // Deduplicate by slot (keep closest), sort, and take top-k.
        all_results.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
        all_results.dedup_by_key(|c| c.slot);
        Ok(all_results
            .into_iter()
            .take(k)
            .map(|c| (crate::distance::to_score(self.metric, c.dist), c.slot))
            .collect())
    }
}
