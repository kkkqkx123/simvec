use std::collections::BinaryHeap;
use std::time::Instant;

use rayon::prelude::*;

use super::super::keys::Keys;
use super::super::vectors::Vectors;
use super::super::CollectionStore;
use crate::error::Result;
use crate::metrics::SearchPath;
use crate::types::{PointId, SearchQuery, SearchResult};

use super::{ScoredSlot, SearchSnapshot};

impl CollectionStore {
    /// Parallel exact scan with streaming top-K over every live slot.
    /// The parallel heap avoids materializing the full (score, slot) set,
    /// cutting peak memory to O(k). The heap only ever holds slots that
    /// already passed the full payload filter (bitmap pre-check plus
    /// `matches`), so the shared post-processing below must not filter a
    /// second time.
    pub(super) fn search_exact(
        &self,
        query: &SearchQuery,
        snap: &SearchSnapshot<'_>,
        metric: crate::types::DistanceMetric,
        next_slot: u64,
        filtered: bool,
        started: Instant,
    ) -> Result<Vec<SearchResult>> {
        let k = query.offset.unwrap_or(0).saturating_add(query.limit);
        let heap: BinaryHeap<std::cmp::Reverse<ScoredSlot>> = (0..next_slot as u32)
            .into_par_iter()
            .filter(|s| !snap.tombstones.bit(*s as usize))
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
                            dim: snap.dim,
                            segment_slots: snap.segment_slots,
                            tombstones: snap.tombstones,
                            vsnap: snap.vsnap,
                            keysnap: snap.keysnap,
                            filter_mask: snap.filter_mask,
                            payload_view: snap.payload_view,
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

    /// Single live-slot evaluation for the streaming exact scan.
    ///
    /// The payload filter is evaluated inline so the heap is only ever
    /// populated with candidates that pass the filter; stale data (missing
    /// key/payload) is silently skipped, mirroring the IVF index path's
    /// `filter_map` behaviour.
    pub(super) fn exact_scan_step(
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
            filter_mask,
            ..
        } = snap;
        // Fast bitmap pre-filter for equality payload conditions: skipped
        // slots never pay the key/payload reads below. Slots without filter
        // coverage fall through to the full check, so the mask can only
        // remove work, never results.
        if let Some(mask) = filter_mask {
            if (slot as usize) >= mask.len() || !mask[slot as usize] {
                return;
            }
        }
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
        let dist = crate::distance::distance(metric, query.primary_vector(), v);
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
}
