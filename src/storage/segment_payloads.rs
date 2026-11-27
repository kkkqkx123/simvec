//! Payload pages scoped to one physical segment.
//!
//! Each segment owns an independent Gridstore-style `payloads_store/`. A
//! logical slot is resolved to its segment before touching a page, so a
//! corrupted page or a flush failure stays local to that segment.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::error::{Result, VectorSearchError};
use crate::types::Payload;

use super::payload_store::{PayloadStore, StoreConfig};
use super::segments::segment_dir;

/// Payload stores of one generation.
pub struct SegmentPayloadStore {
    base: PathBuf,
    segment_slots: u32,
    config: StoreConfig,
    stores: RwLock<Vec<Arc<PayloadStore>>>,
}

impl SegmentPayloadStore {
    /// Create the first segment's payload store.
    pub fn create(dir: &Path, segment_slots: u32, config: StoreConfig) -> Result<Self> {
        let path = segment_dir(dir, 0).join("payloads_store");
        let store = PayloadStore::create(&path, config.clone())?;
        Ok(Self {
            base: dir.to_path_buf(),
            segment_slots,
            config,
            stores: RwLock::new(vec![Arc::new(store)]),
        })
    }

    /// Open every serving segment's payload store. A store that fails
    /// validation before the first checkpoint is rebuilt by `PayloadStore`
    /// itself from the WAL.
    pub fn open_with_repair(
        dir: &Path,
        segment_slots: u32,
        slot_capacity: u64,
        replay_floor: u64,
        config: StoreConfig,
    ) -> Result<Self> {
        if segment_slots == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "segment_slots must be positive".to_string(),
            ));
        }
        let count = slot_capacity.div_ceil(segment_slots as u64).max(1) as usize;
        let mut stores = Vec::with_capacity(count);
        for index in 0..count {
            let start = index as u32 * segment_slots;
            let path = segment_dir(dir, start).join("payloads_store");
            stores.push(Arc::new(PayloadStore::open_with_repair(
                &path,
                replay_floor,
                config.clone(),
            )?));
        }
        Ok(Self {
            base: dir.to_path_buf(),
            segment_slots,
            config,
            stores: RwLock::new(stores),
        })
    }

    fn index(&self, slot: u32) -> Result<usize> {
        let index = slot as u64 / self.segment_slots as u64;
        if self.segment_slots == 0 || index > usize::MAX as u64 {
            return Err(VectorSearchError::CorruptData(format!(
                "payload slot {slot} has no segment"
            )));
        }
        Ok(index as usize)
    }

    fn with_store<T>(
        &self,
        slot: u32,
        action: impl FnOnce(&PayloadStore) -> Result<T>,
    ) -> Result<T> {
        let index = self.index(slot)?;
        let stores = self.stores.read();
        let Some(store) = stores.get(index) else {
            return Err(VectorSearchError::CorruptData(format!(
                "payload segment for slot {slot} is not open"
            )));
        };
        action(store)
    }

    /// Ensure every segment up to `capacity` has a payload store.
    pub fn ensure_segments(&self, capacity: u64) -> Result<()> {
        let needed = capacity.div_ceil(self.segment_slots as u64).max(1) as usize;
        let mut stores = self.stores.write();
        while stores.len() < needed {
            let start = stores.len() as u32 * self.segment_slots;
            let path = segment_dir(&self.base, start).join("payloads_store");
            stores.push(Arc::new(PayloadStore::create(
                &path,
                self.config.clone(),
            )?));
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn persist_tracker_for_test(&self) -> Result<()> {
        let stores = self.stores.read();
        let Some(store) = stores.first() else {
            return Ok(());
        };
        store.persist_tracker_for_test()
    }

    pub fn get(&self, slot: u32) -> Result<Option<Payload>> {
        self.with_store(slot, |store| store.get(slot))
    }

    pub fn validate_json_bytes(&self, json: &[u8]) -> Result<()> {
        // Every segment uses the same page geometry, so the first store is
        // enough for admission checks.
        let stores = self.stores.read();
        let Some(store) = stores.first() else {
            return Ok(());
        };
        store.validate_json_bytes(json)
    }

    pub fn put_at(&self, slot: u32, payload: Option<&Payload>, lsn: u64) -> Result<()> {
        self.with_store(slot, |store| store.put_at(slot, payload, lsn))
    }

    pub fn delete_keys_at(&self, slot: u32, keys: &[&str], lsn: u64) -> Result<()> {
        self.with_store(slot, |store| store.delete_keys_at(slot, keys, lsn))
    }

    pub fn merge_at<I>(&self, slot: u32, partial: I, lsn: u64) -> Result<()>
    where
        I: IntoIterator<Item = (String, serde_json::Value)>,
    {
        self.with_store(slot, |store| store.merge_at(slot, partial, lsn))
    }

    pub fn checkpoint_segment(&self, index: usize) -> Result<()> {
        let stores = self.stores.read();
        let Some(store) = stores.get(index) else {
            return Ok(());
        };
        store.checkpoint()
    }

    pub fn checkpoint(&self) -> Result<()> {
        for store in self.stores.read().iter() {
            store.checkpoint()?;
        }
        Ok(())
    }

    /// Rebuild a compacted generation's per-segment payload stores.
    pub fn compact_to(
        &self,
        dest: &Path,
        segment_slots: u32,
        new_capacity: u64,
        live_slots: &[(u32, u32)],
    ) -> Result<Self> {
        let count = new_capacity.div_ceil(segment_slots as u64).max(1) as usize;
        let sources = self.stores.read();
        let mut stores = Vec::with_capacity(count);
        for index in 0..count {
            let start = index as u32 * segment_slots;
            let end = start + segment_slots;
            let path = segment_dir(dest, start).join("payloads_store");
            let store = PayloadStore::create(&path, self.config.clone())?;
            for &(old_slot, new_slot) in live_slots {
                if new_slot < start || new_slot >= end {
                    continue;
                }
                let source = sources
                    .get(old_slot as usize / self.segment_slots as usize)
                    .ok_or_else(|| {
                        VectorSearchError::CorruptData(format!(
                            "source payload segment for slot {old_slot} is missing"
                        ))
                    })?;
                let payload = source.get(old_slot)?;
                store.put(new_slot, payload.as_ref())?;
            }
            store.checkpoint()?;
            stores.push(Arc::new(store));
        }
        Ok(Self {
            base: dest.to_path_buf(),
            segment_slots,
            config: self.config.clone(),
            stores: RwLock::new(stores),
        })
    }
}
