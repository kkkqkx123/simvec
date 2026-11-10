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
    /// Read the payload for a slot from PayloadStore.
    pub(super) fn read_payload_at(&self, slot: u32) -> Result<Option<Payload>> {
        let ps_guard = self.payload_store.load();
        let ps = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        ps.get(slot)
    }

    /// Write a payload for a slot to PayloadStore.
    pub(super) fn write_payload_at(&self, slot: u32, payload: Option<&Payload>) -> Result<()> {
        let ps_guard = self.payload_store.load();
        let ps = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        ps.put(slot, payload)
    }

    /// Delete specific keys from a slot's payload in PayloadStore.
    pub(super) fn delete_keys_at(&self, slot: u32, keys: &[&str]) -> Result<()> {
        let ps_guard = self.payload_store.load();
        let ps = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        ps.delete_keys(slot, keys)
    }

    /// Merge the given fields into a slot's payload in PayloadStore.
    pub(super) fn merge_payload_at(&self, slot: u32, partial: &Payload) -> Result<()> {
        let ps_guard = self.payload_store.load();
        let ps = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        ps.merge(slot, partial.iter().map(|(k, v)| (k.clone(), v.clone())))
    }

    /// Re-index a slot's pre-filter structures after a non-upsert payload
    /// mutation (`SetPayload`/`SetPayloadField`/`DeletePayloadKeys`). The
    /// bitmap is conservative but must reflect current values: `build_mask`
    /// ANDs condition masks together, so an unregistered value would
    /// produce an all-zero candidate mask and wrongly exclude matches.
    pub(super) fn refresh_filter_bitmap_locked(&self, inner: &mut StoreInner, slot: u32) {
        let payload = self.read_payload_at(slot).ok().flatten();
        inner.filter_bitmap.register_slot(slot, payload.as_ref());
        inner.payload_indexes.register_slot(slot, payload.as_ref());
    }

    /// Replace the payload for a single point. The point must exist and not
    /// be tombstoned. The entire payload map is replaced atomically via a
    /// WAL-backed transaction.
    pub fn set_payload(&self, id: &PointId, payload: Payload) -> Result<()> {
        let slot = self.live_slot_of(id)?;
        let json = serde_json::to_string(&payload)?;
        self.apply_ops(&[WalRecord::SetPayload {
            slot,
            payload: Some(json),
        }])
    }

    /// Set a single field on a point's payload (merge semantics). A missing
    /// payload is created containing just this field; all other keys are
    /// preserved. Applied atomically via a WAL-backed transaction.
    pub fn set_payload_field(
        &self,
        id: &PointId,
        key: String,
        value: serde_json::Value,
    ) -> Result<()> {
        self.set_payload_fields(id, Payload::from([(key, value)]))
    }

    /// Merge the given fields into a point's payload within one WAL-backed
    /// transaction: keys in `fields` overwrite their previous values while
    /// all other keys are preserved. A missing payload is created. The point
    /// must exist and not be tombstoned.
    pub fn set_payload_fields(&self, id: &PointId, fields: Payload) -> Result<()> {
        if fields.is_empty() {
            return Ok(());
        }
        let slot = self.live_slot_of(id)?;
        let mut ops = Vec::with_capacity(fields.len());
        for (key, value) in fields {
            let json = serde_json::to_string(&value)?;
            ops.push(WalRecord::SetPayloadField {
                slot,
                key,
                value: json,
            });
        }
        self.apply_ops(&ops)
    }

    /// Remove specific keys from a point's payload. The remaining keys are
    /// preserved. If the point has no payload this is a no-op.
    pub fn delete_payload_keys(&self, id: &PointId, keys: Vec<String>) -> Result<()> {
        let slot = self.live_slot_of(id)?;
        self.apply_ops(&[WalRecord::DeletePayloadKeys { slot, keys }])
    }

    /// Resolve a live point's slot or fail with `InvalidPointId`.
    pub(super) fn live_slot_of(&self, id: &PointId) -> Result<u32> {
        let slot = self
            .inner
            .read()
            .reverse
            .get(id)
            .copied()
            .ok_or_else(|| VectorSearchError::InvalidPointId(id.to_string()))?;
        if self.tombstones.load().bit(slot as usize) {
            return Err(VectorSearchError::InvalidPointId(format!(
                "point {} is tombstoned",
                id
            )));
        }
        Ok(slot)
    }

    /// Create a payload field index. The index is populated from the
    /// current live set synchronously (under the store write lock) and its
    /// definition persisted to `payload_indexes.json`, so a restart or
    /// concurrent search either sees the complete index or none.
    pub fn create_payload_index(
        &self,
        field: &str,
        schema: crate::types::PayloadSchemaType,
    ) -> Result<()> {
        if field == "_id" {
            return Err(VectorSearchError::InvalidConfig(
                "payload index on reserved field '_id' is not allowed".to_string(),
            ));
        }
        let mut inner = self.inner.write();
        let capacity = inner.meta.slot_capacity as usize;
        inner.payload_indexes.declare(field, schema, capacity)?;
        // Populate from the current live payloads; registration for slots
        // without an indexable value on `field` is a no-op.
        let tombstones = self.tombstones.load();
        for slot in 0..inner.meta.next_slot as u32 {
            if tombstones.bit(slot as usize) {
                continue;
            }
            let payload = self.read_payload_at(slot).ok().flatten();
            inner.payload_indexes.register_slot(slot, payload.as_ref());
        }
        inner.payload_indexes.save_defs(&self.dir)?;
        Ok(())
    }

    /// Drop the payload field index on `field`. Returns whether it existed.
    /// The persisted definitions are updated only after successful removal.
    pub fn delete_payload_index(&self, field: &str) -> Result<bool> {
        let mut inner = self.inner.write();
        if !inner.payload_indexes.delete(field) {
            return Ok(false);
        }
        inner.payload_indexes.save_defs(&self.dir)?;
        Ok(true)
    }

    /// All declared payload indexes as `(field, schema_type)` pairs.
    pub fn list_payload_indexes(&self) -> Vec<(String, crate::types::PayloadSchemaType)> {
        self.inner
            .read()
            .payload_indexes
            .defs()
            .into_iter()
            .map(|d| (d.field, d.schema))
            .collect()
    }
}
