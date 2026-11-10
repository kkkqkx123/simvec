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
    /// Incrementally repair the graph by removing references to tombstoned
    /// slots from adjacency lists.
    ///
    /// This mirrors pgvector's VACUUM graph repair pass: for each node whose
    /// adjacency list contains tombstoned slots, the affected neighbors are
    /// removed and new neighbors are selected from the remaining candidates.
    /// The repair is local — only nodes with stale references are touched —
    /// avoiding a full rebuild after each compaction.
    ///
    /// Returns the number of nodes whose adjacency lists were modified.
    pub(crate) fn repair(
        &self,
        tombstones: &crate::storage::TombstoneBits,
        vectors: &[Arc<memmap2::Mmap>],
        segment_slots: u32,
    ) -> usize {
        // Cheap early exit: with no tombstones there is nothing to repair, and
        // the per-chunk popcount of the tombstone table costs far less than a
        // full node-table walk with per-layer write locks.
        if tombstones.count_ones() == 0 {
            return 0;
        }
        let nodes = self.nodes.read();
        let mut repaired = 0usize;

        for (slot, node_opt) in nodes.iter().enumerate() {
            let Some(node) = node_opt else {
                continue;
            };
            let slot = slot as u32;

            // Skip nodes that are themselves tombstoned — they will be
            // cleaned up during compaction.
            if tombstones.bit(slot as usize) {
                continue;
            }

            for layer in 0..=node.level {
                let mut adj = timed_write_lock(self.lock_metrics.as_deref(), node.layer(layer));
                let old_len = adj.len();

                // Remove tombstoned slots from the adjacency list.
                adj.retain(|&s| !tombstones.bit(s as usize));

                // If we removed any entries, try to fill the gaps with
                // new neighbors from the graph.
                if adj.len() < old_len {
                    let Some(nv) =
                        Vectors::read_slot(vectors, slot as u64, segment_slots, self.dim)
                    else {
                        continue;
                    };
                    let lm = self.layer_cap(layer);

                    // Collect candidates: all live neighbors of this node's
                    // neighbors (2-hop) that are not already in the list.
                    // The adjacency is at most one layer cap (a few dozen),
                    // so a linear scan avoids hashing for the membership
                    // check on this offline path.
                    let mut candidates: Vec<Cand> = Vec::with_capacity(adj.len().max(4));

                    // Start with direct neighbors that survived.
                    for &neighbor_slot in adj.iter() {
                        if let Some(neighbor_node) = self.node(neighbor_slot) {
                            let neighbor_adj = neighbor_node.layer(layer).read();
                            for &candidate_slot in neighbor_adj.iter() {
                                if candidate_slot == slot
                                    || tombstones.bit(candidate_slot as usize)
                                    || adj.contains(&candidate_slot)
                                {
                                    continue;
                                }
                                if let Some(cv) = Vectors::read_slot(
                                    vectors,
                                    candidate_slot as u64,
                                    segment_slots,
                                    self.dim,
                                ) {
                                    candidates.push(Cand {
                                        dist: self.distance(nv, cv),
                                        slot: candidate_slot,
                                    });
                                }
                            }
                        }
                    }

                    // Sort by distance and select the best candidates to fill gaps.
                    candidates.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
                    for cand in candidates {
                        if adj.len() >= lm {
                            break;
                        }
                        if !adj.contains(&cand.slot) {
                            adj.push(cand.slot);
                        }
                    }

                    // Update version since we modified the adjacency list.
                    node.bump_version();
                    repaired += 1;
                }
            }
        }

        repaired
    }

    /// Staleness ratio combining overwrite count and distance delta.
    /// Returns `max(count_ratio, delta_ratio)` where:
    /// - `count_ratio = overwrites / built_at_live_count`
    /// - `delta_ratio = distance_delta / built_at_live_count`
    pub(crate) fn stale_ratio(&self) -> f64 {
        let count = self.overwrites_since_build.load(Ordering::Relaxed);
        let delta = self.overwrites_distance_delta.load(Ordering::Relaxed) as f64 / 1e6;
        let base = self.built_at_live_count.load(Ordering::Relaxed).max(1) as f64;
        let count_ratio = count as f64 / base;
        let delta_ratio = delta / base;
        count_ratio.max(delta_ratio)
    }
}
