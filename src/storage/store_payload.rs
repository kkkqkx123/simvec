use super::WalRecord;
use super::{CollectionStore, StoreInner};

use crate::error::{Result, VectorSearchError};
use crate::types::{Payload, PointId};

impl CollectionStore {
    /// Read the payload for a slot from PayloadStore.
    pub(super) fn read_payload_at(&self, slot: u32) -> Result<Option<Payload>> {
        let data = self.data_arc();
        let ps_guard = data.payload_store.load();
        let ps = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        ps.get(slot)
    }

    /// Reject a payload that cannot fit in a single payload page.
    ///
    /// Called on the admission path (before the WAL append): a record that is
    /// logged but cannot be applied would stay in the log forever and make
    /// every later `open` fail while replaying it.
    pub(super) fn validate_payload_size(&self, payload: Option<&Payload>) -> Result<()> {
        let Some(payload) = payload else {
            return Ok(());
        };
        let len = serde_json::to_vec(payload)?.len();
        self.validate_payload_len(len)
    }

    pub(super) fn validate_payload_len(&self, len: usize) -> Result<()> {
        let data = self.data_arc();
        let ps_guard = data.payload_store.load();
        let Some(ps) = ps_guard.as_ref().as_ref() else {
            return Ok(());
        };
        let limit = ps.max_payload_bytes();
        if len > limit {
            return Err(VectorSearchError::InvalidConfig(format!(
                "payload of {len} bytes exceeds the {limit} byte payload page; \
                 raise StorageConfig::payload_page_size"
            )));
        }
        Ok(())
    }

    /// Write a payload for a slot to PayloadStore.
    pub(super) fn write_payload_at(
        &self,
        slot: u32,
        payload: Option<&Payload>,
        lsn: u64,
    ) -> Result<()> {
        let data = self.data_arc();
        let ps_guard = data.payload_store.load();
        let ps = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        ps.put_at(slot, payload, lsn)
    }

    /// Delete specific keys from a slot's payload in PayloadStore.
    pub(super) fn delete_keys_at(&self, slot: u32, keys: &[&str], lsn: u64) -> Result<()> {
        let data = self.data_arc();
        let ps_guard = data.payload_store.load();
        let ps = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        ps.delete_keys_at(slot, keys, lsn)
    }

    /// Merge the given fields into a slot's payload in PayloadStore.
    pub(super) fn merge_payload_at(&self, slot: u32, partial: &Payload, lsn: u64) -> Result<()> {
        let data = self.data_arc();
        let ps_guard = data.payload_store.load();
        let ps = ps_guard
            .as_ref()
            .as_ref()
            .expect("PayloadStore must be present");
        ps.merge_at(
            slot,
            partial.iter().map(|(k, v)| (k.clone(), v.clone())),
            lsn,
        )
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
        self.live_slot_of(id)?;
        let json = serde_json::to_string(&payload)?;
        self.apply_ops(&[WalRecord::SetPayload {
            point_id: id.to_string(),
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
        self.live_slot_of(id)?;
        let mut ops = Vec::with_capacity(fields.len());
        for (key, value) in fields {
            let json = serde_json::to_string(&value)?;
            ops.push(WalRecord::SetPayloadField {
                point_id: id.to_string(),
                key,
                value: json,
            });
        }
        self.apply_ops(&ops)
    }

    /// Remove specific keys from a point's payload. The remaining keys are
    /// preserved. If the point has no payload this is a no-op.
    pub fn delete_payload_keys(&self, id: &PointId, keys: Vec<String>) -> Result<()> {
        self.live_slot_of(id)?;
        self.apply_ops(&[WalRecord::DeletePayloadKeys {
            point_id: id.to_string(),
            keys,
        }])
    }

    /// Resolve a live point's slot or fail with `InvalidPointId`.
    pub(super) fn live_slot_of(&self, id: &PointId) -> Result<u32> {
        let data = self.data_arc();
        let slot = self
            .inner
            .read()
            .reverse
            .get(id)
            .ok_or_else(|| VectorSearchError::InvalidPointId(id.to_string()))?;
        if data.tombstones.load().bit(slot as usize) {
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
        let data = self.data_arc();
        let tombstones = data.tombstones.load();
        for slot in 0..inner.meta.next_slot as u32 {
            if tombstones.bit(slot as usize) {
                continue;
            }
            let payload = self.read_payload_at(slot).ok().flatten();
            inner.payload_indexes.register_slot(slot, payload.as_ref());
        }
        let data_dir = self.data_dir();
        let generation = self.generation.read().clone();
        inner.payload_indexes.save_defs(&data_dir)?;
        inner
            .payload_indexes
            .save(&data_dir, generation.id(), generation.lsn())?;
        Ok(())
    }

    /// Drop the payload field index on `field`. Returns whether it existed.
    /// The persisted definitions are updated only after successful removal.
    pub fn delete_payload_index(&self, field: &str) -> Result<bool> {
        let mut inner = self.inner.write();
        if !inner.payload_indexes.delete(field) {
            return Ok(false);
        }
        let data_dir = self.data_dir();
        inner.payload_indexes.save_defs(&data_dir)?;
        let generation = self.generation.read().clone();
        inner
            .payload_indexes
            .save(&data_dir, generation.id(), generation.lsn())?;
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
