use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use arc_swap::ArcSwap;
use bitvec::prelude::*;
use parking_lot::{Mutex, RwLock};

use super::{StoreInner, CollectionStore};
use super::{set_applying, is_commit_worker, set_commit_worker, assert_not_applying};
use super::filter_bitmap::FilterBitmap;
use super::index_lifecycle::PublishedIndex;
use super::keys::Keys;
use super::payloads::Payloads;
use super::quant::QuantStore;
use super::queue::{CommitQueue, APPLY_WAIT_TIMEOUT};
use super::vectors::Vectors;
use super::{Wal, WalRecord};
use super::meta::Meta;
use super::payload_index::PayloadIndexManager;
use super::payload_store::PayloadStore;
use super::tombstones::TombstoneBits;

use crate::error::{Result, VectorSearchError};
use crate::metrics::Metrics;
use crate::types::{CollectionConfig, DistanceMetric, IndexConfig, IvfConfig, Payload, PointId, QuantizationConfig, VectorPoint};

use super::*;

impl CollectionStore {
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Snapshot of the metadata (dimension, metric, counts).
    pub fn meta(&self) -> Meta {
        self.inner.read().meta.clone()
    }

    /// Operational metrics recorder for this collection.
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Metrics sink shared with the published ANN indexes for lock
    /// contention and version-reload diagnostics.
    pub(crate) fn metrics_arc(&self) -> Arc<Metrics> {
        Arc::clone(&self.metrics)
    }

    /// In-flight index build progress: (building, inserted, total).
    pub(crate) fn build_progress(&self) -> (bool, u64, u64) {
        (
            self.building.load(AtomicOrdering::Relaxed),
            self.build_inserted.load(AtomicOrdering::Relaxed),
            self.build_points_total.load(AtomicOrdering::Relaxed),
        )
    }

    /// WAL accessor for the engine (per-collection transactions) and tests.
    pub fn wal(&self) -> &Wal {
        &self.wal
    }

    /// Slot-ordered ids of every live point matching `filter`.
    ///
    /// Shared match stage for filtered count, filtered scroll and filtered
    /// delete: single full scan over live slots, tombstones skipped, payload
    /// read followed by post-filter verdict. Pure read path, no log writes.
    pub(super) fn filter_match_ids(&self, filter: &crate::types::VectorFilter) -> Result<Vec<PointId>> {
        let inner = self.inner.read();
        let tombstones = self.tombstones.load();
        let keysnap = self.keys.snapshot();
        let mut point_ids = Vec::new();
        for slot in 0..inner.meta.next_slot as usize {
            if tombstones.bit(slot) {
                continue;
            }
            let key = Keys::read_key(&keysnap, slot)?.ok_or_else(|| {
                VectorSearchError::CorruptData(format!("slot {slot} has no key"))
            })?;
            let id = PointId::from(key);
            let payload = self.read_payload_at(slot as u32)?;
            if crate::filter::matches(filter, &id, payload.as_ref())? {
                point_ids.push(id);
            }
        }
        Ok(point_ids)
    }

    /// Number of live points matching `filter`. Pure read, no log writes
    /// and no maintenance side effects.
    pub fn count_by_filter(&self, filter: &crate::types::VectorFilter) -> Result<u64> {
        Ok(self.filter_match_ids(filter)?.len() as u64)
    }

    /// Delete every live point matching `filter`. Returns the number of
    /// deleted points. Runs as a single WAL-backed batch so recovery replays
    /// the filter match deterministically.
    pub fn delete_by_filter(&self, filter: &crate::types::VectorFilter) -> Result<u64> {
        let point_ids = self.filter_match_ids(filter)?;
        if point_ids.is_empty() {
            return Ok(0);
        }
        let deleted = point_ids.len() as u64;
        let encoded: Vec<String> = point_ids.iter().map(|id| id.to_string()).collect();
        self.apply_ops(&[WalRecord::DeleteBatch { point_ids: encoded }])?;
        Ok(deleted)
    }

    /// Fetch a point by id.
    pub fn get(&self, id: &PointId) -> Result<Option<VectorPoint>> {
        let inner = self.inner.read();
        let slot = match inner.reverse.get(id) {
            Some(slot) => *slot as u64,
            None => return Ok(None),
        };
        if self.tombstones.load().bit(slot as usize) {
            return Ok(None);
        }
        let dim = inner.meta.vector_size;
        let vsnap = self.vectors.snapshot();
        let vector = Vectors::read_slot(&vsnap, slot, inner.meta.segment_slots, dim)
            .ok_or_else(|| {
                VectorSearchError::CorruptData(format!("slot {slot} out of vectors.bin range"))
            })?
            .to_vec();
        let payload = self.read_payload_at(slot as u32)?;
        Ok(Some(VectorPoint {
            id: id.clone(),
            vector,
            payload,
        }))
    }

    /// Slot currently holding `id`, or `None` when it is absent or tombstoned.
    ///
    /// Exposed so tests (and operators) can verify that the live slot
    /// numbering matches the one WAL replay rebuilds, which is the invariant
    /// the commit queue exists to preserve.
    pub fn slot_of(&self, id: &PointId) -> Option<u32> {
        let slot = self.inner.read().reverse.get(id).copied()?;
        (!self.tombstones.load().bit(slot as usize)).then_some(slot)
    }

    /// Number of live points.
    pub fn count(&self) -> u64 {
        self.inner.read().meta.live_count
    }

    /// Paginated scan over live points in slot order.
    ///
    /// Returns up to `limit` points starting after `offset` (the last
    /// point_id from the previous page). When `with_payload` is false the
    /// payload field is omitted; when `with_vector` is false the vector
    /// field is omitted. The returned `Option<PointId>` is the id of the
    /// last point in the page (use as the next `offset`), or `None` when
    /// there are no more pages.
    pub fn scroll(
        &self,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        let inner = self.inner.read();
        let tombstones = self.tombstones.load();
        let keysnap = self.keys.snapshot();
        let vsnap = self.vectors.snapshot();
        let dim = inner.meta.vector_size;
        let include_payload = with_payload.unwrap_or(true);
        let include_vector = with_vector.unwrap_or(false);

        let mut skip = offset.is_some();
        let mut results = Vec::with_capacity(limit);
        let mut last_id: Option<PointId> = None;

        for slot in 0..inner.meta.next_slot as usize {
            if tombstones.bit(slot) {
                continue;
            }
            let key = Keys::read_key(&keysnap, slot)?
                .ok_or_else(|| VectorSearchError::CorruptData(format!("slot {slot} has no key")))?;
            let id = PointId::from(key);
            if skip {
                if Some(&id) == offset {
                    skip = false;
                }
                continue;
            }
            let payload = if include_payload {
                self.read_payload_at(slot as u32)?
            } else {
                None
            };
            let vector = if include_vector {
                Some(
                    Vectors::read_slot(&vsnap, slot as u64, inner.meta.segment_slots, dim)
                        .ok_or_else(|| {
                            VectorSearchError::CorruptData(format!(
                                "slot {slot} out of vectors.bin range"
                            ))
                        })?
                        .to_vec(),
                )
            } else {
                None
            };
            last_id = Some(id.clone());
            results.push(VectorPoint {
                id,
                vector: vector.unwrap_or_default(),
                payload,
            });
            if results.len() >= limit {
                break;
            }
        }
        Ok((results, last_id))
    }

    /// Paginated scan over live points matching `filter`, in slot order.
    ///
    /// Cursor semantics match [`CollectionStore::scroll`]: iteration starts
    /// after `offset`, `limit` counts only matching points, and the returned
    /// cursor is the last matching id (use as the next `offset`). A short
    /// page does not imply exhaustion when a filter is present: keep paging
    /// until an empty page appears. Unfiltered callers keep using `scroll`.
    pub fn scroll_by_filter(
        &self,
        filter: &crate::types::VectorFilter,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        let inner = self.inner.read();
        let tombstones = self.tombstones.load();
        let keysnap = self.keys.snapshot();
        let vsnap = self.vectors.snapshot();
        let dim = inner.meta.vector_size;
        let include_payload = with_payload.unwrap_or(true);
        let include_vector = with_vector.unwrap_or(false);

        let mut skip = offset.is_some();
        let mut results = Vec::with_capacity(limit);
        let mut last_id: Option<PointId> = None;

        for slot in 0..inner.meta.next_slot as usize {
            if tombstones.bit(slot) {
                continue;
            }
            let key = Keys::read_key(&keysnap, slot)?
                .ok_or_else(|| VectorSearchError::CorruptData(format!("slot {slot} has no key")))?;
            let id = PointId::from(key);
            if skip {
                if Some(&id) == offset {
                    skip = false;
                }
                continue;
            }
            let payload = self.read_payload_at(slot as u32)?;
            if !crate::filter::matches(filter, &id, payload.as_ref())? {
                continue;
            }
            let payload = if include_payload { payload } else { None };
            let vector = if include_vector {
                Some(
                    Vectors::read_slot(&vsnap, slot as u64, inner.meta.segment_slots, dim)
                        .ok_or_else(|| {
                            VectorSearchError::CorruptData(format!(
                                "slot {slot} out of vectors.bin range"
                            ))
                        })?
                        .to_vec(),
                )
            } else {
                None
            };
            last_id = Some(id.clone());
            results.push(VectorPoint {
                id,
                vector: vector.unwrap_or_default(),
                payload,
            });
            if results.len() >= limit {
                break;
            }
        }
        Ok((results, last_id))
    }
}
