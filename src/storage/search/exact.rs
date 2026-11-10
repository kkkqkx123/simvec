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
}
