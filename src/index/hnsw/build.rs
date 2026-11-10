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
    /// Build a fresh index by inserting the given live slots (off the store
    /// lock; `params.slots` must be sorted ascending).
    ///
    /// `HnswConfig::max_indexing_threads` selects the build shape:
    /// - unset/0: sequential insertion on the global rayon pool;
    /// - 1: sequential insertion on a dedicated single-thread pool;
    /// - n >= 2: n workers insert disjoint round-robin slot subsets into the
    ///   shared graph concurrently (entry-point publication serialized by
    ///   [`Self::promote_lock`]). Topology then depends on interleaving, so
    ///   only recall invariants hold — asserted by the multi-worker tests.
    ///
    /// `IndexBuildParams::progress`, when attached, is incremented once per
    /// processed slot and emits milestone debug logs.
    pub(crate) fn build(params: &IndexBuildParams<'_>, config: &HnswConfig) -> Result<Self> {
        config.validate()?;
        let mut seed = params.collection.len() as u64;
        for b in params.collection.as_bytes() {
            seed = seed.wrapping_mul(0x100_0000_01b3) ^ (*b as u64);
        }
        seed ^= params.slots.len() as u64;
        seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);

        let index = Self::new(params.dim, params.metric, config, seed);
        let total = params.slots.len() as u64;
        // Milestone for periodic progress logging: every ~10% of a large
        // build, at most every 10k slots on huge corpora.
        let milestone = (total / 10).max(10_000);
        let insert_one = |index: &Self, slot: u32| {
            if let Some(v) = Vectors::read_slot(
                params.vectors,
                slot as u64,
                params.segment_slots,
                params.dim,
            ) {
                index.insert(slot, v, params.vectors, params.segment_slots);
            }
            if let Some(progress) = params.progress {
                let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if done % milestone == 0 || done == total {
                    tracing::debug!(
                        collection = %params.collection,
                        done,
                        total,
                        "hnsw build progress"
                    );
                }
            }
        };

        let workers = match config.max_indexing_threads {
            Some(threads) if threads >= 2 => threads.min(params.slots.len().max(1)),
            _ => 1,
        };
        if workers > 1 {
            // Concurrent workers share one pool sized to their count; the
            // per-insert distance fan-outs also land on this pool. Pool
            // tasks are pure distance math and never take graph locks, so
            // nested joins cannot deadlock against worker progress.
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .map_err(|e| VectorSearchError::Internal(format!("hnsw build thread pool: {e}")))?;
            std::thread::scope(|scope| {
                let index_ref = &index;
                let pool_ref = &pool;
                for w in 0..workers {
                    scope.spawn(move || {
                        pool_ref.install(|| {
                            for (i, &slot) in params.slots.iter().enumerate() {
                                if i % workers == w {
                                    insert_one(index_ref, slot);
                                }
                            }
                        });
                    });
                }
            });
        } else if matches!(config.max_indexing_threads, Some(threads) if threads == 1) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .map_err(|e| VectorSearchError::Internal(format!("hnsw build thread pool: {e}")))?;
            pool.install(|| {
                for &slot in params.slots {
                    insert_one(&index, slot);
                }
            });
        } else {
            for &slot in params.slots {
                insert_one(&index, slot);
            }
        }
        index
            .built_at_live_count
            .store(params.slots.len() as u64, Ordering::Relaxed);
        Ok(index)
    }

    /// Paper Algorithm 1 (insert). `vector` is the freshly written value of
    /// `slot`; all other node values come from the mmap snapshots.
    ///
    /// Inserting an already-present slot is a counted no-op: overwrite
    /// upserts keep their graph position (see the module docs for the
    /// trade-off) and bump [`Self::overwrites_since_build`].
    ///
    /// Safe to call concurrently for disjoint slots: node registration and
    /// adjacency updates take their own locks, and both entry-point
    /// transitions (first node, higher-level promotion) run under
    /// [`Self::promote_lock`] with a fresh re-read so concurrent builders
    /// can neither orphan a first node nor clobber an already-higher entry.
    pub(crate) fn insert(
        &self,
        slot: u32,
        vector: &[f32],
        vectors: &[Arc<memmap2::Mmap>],
        segment_slots: u32,
    ) {
        if self.node(slot).is_some() {
            self.overwrites_since_build.fetch_add(1, Ordering::Relaxed);
            // Track the distance delta: how far the new vector is from the
            // old one. A large delta means the node's graph position is more
            // likely stale.
            if let Some(old_v) = Vectors::read_slot(vectors, slot as u64, segment_slots, self.dim) {
                let delta = (self.distance(vector, old_v) * 1e6) as u64;
                self.overwrites_distance_delta
                    .fetch_add(delta, Ordering::Relaxed);
            }
            return;
        }

        let level = self.random_level();
        let node = Arc::new(Node::new(level));
        {
            let mut nodes = self.nodes.write();
            if slot as usize >= nodes.len() {
                nodes.resize(slot as usize + 1, None);
            }
            nodes[slot as usize] = Some(Arc::clone(&node));
        }

        // First node in the graph: claim it under the promotion lock so a
        // racing builder cannot leave two unlinked roots.
        if self.entry.read().is_none() {
            let claim = self.promote_lock.lock();
            if self.entry.read().is_none() {
                *self.entry.write() = Some(EntryPoint { slot, level });
                return;
            }
            drop(claim);
        }

        // Snapshot the entry once. It may move upward while this insert is
        // running; descending from the stale (still valid) snapshot only
        // risks missing newer shortcuts, never correctness.
        let Some(entry) = *self.entry.read() else {
            return;
        };

        let entry_vec = Vectors::read_slot(vectors, entry.slot as u64, segment_slots, self.dim);
        let mut ep = match entry_vec {
            Some(ev) => vec![Cand {
                dist: self.distance(vector, ev),
                slot: entry.slot,
            }],
            None => return,
        };

        // Phase 1: greedy descent through layers above `level` (ef = 1).
        let mut lc = entry.level;
        while lc > level {
            let Some(&start) = ep.first() else {
                return;
            };
            let best = self.greedy_step(vector, start, lc, vectors, segment_slots);
            ep = vec![best];
            lc -= 1;
        }

        // Phase 2: search + connect on each layer down to the ground.
        for layer in (0..=level.min(entry.level)).rev() {
            let result = self.search_layer(
                vector,
                &ep,
                self.ef_construct,
                layer,
                &HnswSearchContext {
                    tombstones: None,
                    vectors,
                    segment_slots,
                    filter_mask: None,
                },
            );
            let selected =
                self.select_neighbors(&result.live, self.layer_cap(layer), vectors, segment_slots);
            *timed_write_lock(self.lock_metrics.as_deref(), node.layer(layer)) =
                selected.iter().map(|c| c.slot).collect();
            node.bump_version();
            for cand in &selected {
                self.link_neighbor(cand.slot, slot, cand.dist, layer, vectors, segment_slots);
            }
            ep = result.live;
        }

        if level > entry.level {
            // Publish the new higher entry under the promotion lock,
            // re-reading the latest state: another builder may have promoted
            // an even-higher entry in the meantime, and clobbering it would
            // strand its upper layers.
            let _publish = self.promote_lock.lock();
            if self.entry.read().as_ref().is_none_or(|e| level > e.level) {
                *self.entry.write() = Some(EntryPoint { slot, level });
            }
        }
    }

    /// Paper Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) with pgvector's
    /// keep-pruned-connections variant: discarded candidates backfill `r`
    /// up to `lm` so nodes never end up under-connected.
    ///
    /// Pairwise distances are measured between candidate vectors, not
    /// against the query.
    pub(super) fn select_neighbors(
        &self,
        candidates: &[Cand],
        lm: usize,
        vectors: &[Arc<memmap2::Mmap>],
        segment_slots: u32,
    ) -> Vec<Cand> {
        let mut out = Vec::with_capacity(lm);
        self.select_neighbors_into(candidates, lm, vectors, segment_slots, &mut out);
        out
    }

    /// Heuristic selection writing into a caller-provided buffer so hot paths
    /// (reverse-link overflow) select without a per-call allocation. The
    /// sorted pool and pruned spill still reuse the per-thread capacity; only
    /// `out` grows, once per thread, to the layer cap.
    pub(super) fn select_neighbors_into(
        &self,
        candidates: &[Cand],
        lm: usize,
        vectors: &[Arc<memmap2::Mmap>],
        segment_slots: u32,
        out: &mut Vec<Cand>,
    ) {
        SELECT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let mut sorted = std::mem::take(&mut scratch.sorted);
            let mut pruned = std::mem::take(&mut scratch.pruned);
            sorted.clear();
            pruned.clear();
            sorted.extend_from_slice(candidates);
            sorted.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
            out.clear();
            if sorted.len() <= lm {
                out.extend_from_slice(&sorted);
                scratch.sorted = sorted;
                scratch.pruned = pruned;
                return;
            }

            for cand in sorted.drain(..) {
                if out.len() == lm {
                    break;
                }
                let Some(cv) =
                    Vectors::read_slot(vectors, cand.slot as u64, segment_slots, self.dim)
                else {
                    continue;
                };
                // Pairwise checks stay sequential: the fan-out is at most `lm`
                // distances (~microseconds), far below rayon task-routing cost.
                // Routing each check through the global pool made builds orders
                // of magnitude slower (every join wakes the worker threads) with
                // no compute to amortize it. Slot-level parallelism belongs to
                // the multi-worker build path, not this inner loop.
                let mut closer = true;
                for s in out.iter() {
                    let Some(rv) =
                        Vectors::read_slot(vectors, s.slot as u64, segment_slots, self.dim)
                    else {
                        closer = false;
                        break;
                    };
                    // Strictly-farther-from-every-selected means diverse enough
                    // to keep (pgvector CheckElementCloser).
                    if self.distance(cv, rv) <= cand.dist {
                        closer = false;
                        break;
                    }
                }
                if closer {
                    out.push(cand);
                } else {
                    pruned.push(cand);
                }
            }
            for cand in pruned.drain(..) {
                if out.len() == lm {
                    break;
                }
                out.push(cand);
            }
            scratch.sorted = sorted;
            scratch.pruned = pruned;
        })
    }

    /// Add the reverse edge `neighbor -> slot`, shrinking the neighbor's
    /// list heuristically when it exceeds the layer cap (pgvector
    /// UpdateNeighborOnDisk + HnswUpdateConnection).
    pub(super) fn link_neighbor(
        &self,
        neighbor: u32,
        slot: u32,
        dist_to_neighbor: f32,
        layer: u8,
        vectors: &[Arc<memmap2::Mmap>],
        segment_slots: u32,
    ) {
        let Some(node) = self.node(neighbor) else {
            return;
        };
        let lm = self.layer_cap(layer);
        let mut adj = timed_write_lock(self.lock_metrics.as_deref(), node.layer(layer));
        if adj.contains(&slot) {
            return;
        }
        if adj.len() < lm {
            adj.push(slot);
            node.bump_version();
            return;
        }

        // Overflow: reselect around the neighbor's own vector. The pool is
        // bounded by the layer cap (a few dozen entries), so the distance
        // fan-out stays sequential — rayon routing would dominate the work.
        // The pool and the selection output reuse the per-thread link buffers
        // and the adjacency list is rewritten in place (it was full, so the
        // reselected set fits), which removes every per-overflow allocation
        // from this hot path.
        let Some(nv) = Vectors::read_slot(vectors, neighbor as u64, segment_slots, self.dim) else {
            return;
        };
        LINK_SCRATCH.with(|link_cell| {
            let mut link = link_cell.borrow_mut();
            let HnswLinkScratch { pool, selected } = &mut *link;
            pool.clear();
            pool.extend(adj.iter().filter_map(|&s| {
                let v = Vectors::read_slot(vectors, s as u64, segment_slots, self.dim)?;
                Some(Cand {
                    dist: self.distance(nv, v),
                    slot: s,
                })
            }));
            pool.push(Cand {
                dist: dist_to_neighbor,
                slot,
            });
            self.select_neighbors_into(pool.as_slice(), lm, vectors, segment_slots, selected);
            adj.clear();
            adj.extend(selected.iter().map(|c| c.slot));
        });
        node.bump_version();
    }

    pub(super) fn random_level(&self) -> u8 {
        let draw = -self.rng.lock().next_open01().ln() * self.ml;
        (draw as u8).min(MAX_LEVEL)
    }
}
