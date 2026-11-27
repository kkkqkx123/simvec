//! Deletion-local HNSW repair.
//!
//! Tombstoned nodes stay in the graph until compaction, but their neighbors
//! lose a useful shortcut when the node stops being a result. The inbound
//! lists maintained alongside forward adjacency let this pass reach every
//! node pointing at the removed slot, including links that are not reachable
//! through the removed node's own adjacency. Each touched node reselects its
//! neighborhood with the same diversity heuristic as insert, so no
//! segment-wide rebuild is needed for a single delete.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::*;
use crate::storage::vectors::Vectors;

impl HnswIndex {
    /// Repair the layers of `slot` after it becomes non-live.
    ///
    /// `slot` is graph-local; `slot_base` maps it back to the global slot
    /// table used by `deleted`. Called under the store write lock, while
    /// searches may still traverse the graph under short per-layer locks.
    pub(crate) fn heal_deleted(
        &self,
        slot: u32,
        vectors: &[Arc<memmap2::Mmap>],
        segment_slots: u32,
        slot_base: u32,
        deleted: impl Fn(u32) -> bool,
    ) {
        let Some(node) = self.node(slot) else {
            return;
        };
        self.healed_nodes.fetch_add(1, Ordering::Relaxed);

        for layer in 0..=node.level {
            // Inbound and outbound links together form the complete local
            // neighborhood. Deleted peers are still repaired: their forward
            // lists must not retain an edge to the removed slot either.
            let outgoing: Vec<u32> = node.layer(layer).read().iter().copied().collect();
            let mut peers = outgoing.clone();
            for &source in node.reverse(layer).read().iter() {
                if source != slot && !peers.contains(&source) {
                    peers.push(source);
                }
            }

            for &peer_slot in &peers {
                if peer_slot == slot {
                    continue;
                }
                let Some(peer) = self.node(peer_slot) else {
                    continue;
                };
                let previous = {
                    let mut adjacency = peer.layer(layer).write();
                    let previous = adjacency.clone();
                    adjacency.retain(|&s| s != slot && !deleted(s + slot_base));
                    previous
                };
                let Some(vector) = Vectors::read_slot(
                    vectors,
                    (peer_slot + slot_base) as u64,
                    segment_slots,
                    self.dim,
                ) else {
                    continue;
                };
                let cap = self.layer_cap(layer);
                let mut candidates = Vec::with_capacity(cap + peers.len());
                for &existing in &previous {
                    if existing == peer_slot || deleted(existing + slot_base) {
                        continue;
                    }
                    if let Some(other) = Vectors::read_slot(
                        vectors,
                        (existing + slot_base) as u64,
                        segment_slots,
                        self.dim,
                    ) {
                        candidates.push(Cand {
                            dist: self.distance(vector, other),
                            slot: existing,
                        });
                    }
                }
                for &other in &peers {
                    if other == peer_slot || other == slot || deleted(other + slot_base) {
                        continue;
                    }
                    if let Some(other_vector) = Vectors::read_slot(
                        vectors,
                        (other + slot_base) as u64,
                        segment_slots,
                        self.dim,
                    ) {
                        candidates.push(Cand {
                            dist: self.distance(vector, other_vector),
                            slot: other,
                        });
                    }
                }
                candidates.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
                candidates.dedup_by_key(|candidate| candidate.slot);
                let selected =
                    self.select_neighbors(&candidates, cap, vectors, segment_slots, slot_base);
                let replacement: Vec<u32> = selected.iter().map(|c| c.slot).collect();
                {
                    let mut adjacency = peer.layer(layer).write();
                    *adjacency = replacement.clone();
                }
                self.record_neighbor_change(peer_slot, layer, &previous, &replacement);
                if previous != replacement {
                    self.healed_edges
                        .fetch_add((previous.len() + replacement.len()) as u64, Ordering::Relaxed);
                }
            }

            // Drop every edge out of the deleted node and its inbound index;
            // the tombstone filters results until compaction rebuilds the
            // whole topology.
            let previous = {
                let mut adjacency = node.layer(layer).write();
                let previous = adjacency.clone();
                adjacency.clear();
                previous
            };
            self.record_neighbor_change(slot, layer, &previous, &[]);
            node.reverse(layer).write().clear();
        }
    }
}
