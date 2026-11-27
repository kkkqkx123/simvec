//! Deletion-local HNSW repair.
//!
//! Tombstoned nodes stay in the graph until compaction, but their former
//! neighbors lose a useful shortcut when the node stops being a result. This
//! pass reselects each former neighbor's adjacency from its existing links
//! plus the other nodes that shared the removed node. It touches only the
//! affected neighborhood and uses the same diversity heuristic as insert, so
//! no segment-wide rebuild is needed for a single delete.

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
            let former: Vec<u32> = node
                .layer(layer)
                .read()
                .iter()
                .copied()
                .filter(|&neighbor| neighbor != slot && !deleted(neighbor + slot_base))
                .collect();
            if former.is_empty() {
                continue;
            }

            // Remove the dead link from every former neighbor first, so the
            // re-selection below sees the adjacency it will actually publish.
            for &neighbor in &former {
                if let Some(peer) = self.node(neighbor) {
                    let mut adjacency = peer.layer(layer).write();
                    if let Some(position) = adjacency.iter().position(|&s| s == slot) {
                        adjacency.remove(position);
                        peer.bump_version();
                    }
                }
            }

            let cap = self.layer_cap(layer);
            for &neighbor in &former {
                let Some(peer) = self.node(neighbor) else {
                    continue;
                };
                let Some(vector) = Vectors::read_slot(
                    vectors,
                    (neighbor + slot_base) as u64,
                    segment_slots,
                    self.dim,
                ) else {
                    continue;
                };
                let mut candidates = Vec::with_capacity(cap + former.len());
                for &existing in peer.layer(layer).read().iter() {
                    if existing == neighbor || existing == slot || deleted(existing + slot_base) {
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
                // The other nodes that linked to the removed node are the
                // neighborhood most likely to lose connectivity.
                for &other in &former {
                    if other == neighbor {
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
                let mut adjacency = peer.layer(layer).write();
                if adjacency.as_slice() != replacement.as_slice() {
                    let changed = adjacency.len() + replacement.len();
                    adjacency.clear();
                    adjacency.extend(replacement);
                    peer.bump_version();
                    self.healed_edges
                        .fetch_add(changed as u64, Ordering::Relaxed);
                }
            }
        }
    }
}
