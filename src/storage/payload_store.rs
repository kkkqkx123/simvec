//! Gridstore-style payload storage engine.
//!
//! Replaces the append-only `payloads.bin` blob directory with a block-based
//! storage that supports in-place updates, eliminates tombstone-driven
//! compaction, and provides O(1) space allocation.
//!
//! # On-disk layout
//!
//! ```text
//! payloads_store/
//! ├── config.json     — StoreConfig (block_size, page_size)
//! ├── tracker.bin     — slot → ValuePointer mapping
//! ├── bitmask.bin     — block usage bitmap (1 = used, 0 = free)
//! ├── page_0000.bin   — 32 MB data page (256K × 128 B blocks)
//! ├── page_0001.bin
//! └── ...
//! ```
//!
//! Each payload is serialized as JSON bytes and stored across one or more
//! contiguous 128-byte blocks within a page. The tracker provides direct
//! O(1) slot → pointer lookup, and the bitmask enables fast free-space
//! allocation without scanning.
//!
//! All on-disk state is derived: the collection WAL carries every payload
//! write durably before it is applied, so replay can rebuild tracker, bitmask
//! and pages after any crash. `put` therefore updates memory and page files
//! without fsync, and [`PayloadStore::checkpoint`] batches the persistence of
//! the derived files.

mod bitmask;
mod pages;
mod tracker;

use std::path::{Path, PathBuf};

pub use bitmask::MmapBitmask;
pub use pages::Pages;
pub use tracker::{Tracker, ValuePointer};

use crate::error::{Result, VectorSearchError};
use crate::types::Payload;

/// Magic bytes for the tracker file.
const TRACKER_MAGIC: [u8; 4] = *b"VPTR";

/// Magic bytes for the bitmask file.
const BITMASK_MAGIC: [u8; 4] = *b"VBMP";

/// Default block size: 128 bytes.
pub const DEFAULT_BLOCK_SIZE: usize = 128;

/// Default page size: 32 MiB (256K blocks of 128 B each).
pub const DEFAULT_PAGE_SIZE: usize = 32 * 1024 * 1024;

/// Configuration for the payload store.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StoreConfig {
    /// Size of each block in bytes. Payloads are allocated in multiples of
    /// this granularity.
    pub block_size: usize,
    /// Size of each data page in bytes. Must be a multiple of `block_size`.
    pub page_size: usize,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            page_size: DEFAULT_PAGE_SIZE,
        }
    }
}

impl StoreConfig {
    /// Number of blocks per page.
    pub fn blocks_per_page(&self) -> usize {
        self.page_size / self.block_size
    }

    fn validate(&self) -> Result<()> {
        if self.block_size == 0 {
            return Err(VectorSearchError::CorruptData(
                "block_size must be > 0".into(),
            ));
        }
        if self.page_size == 0 {
            return Err(VectorSearchError::CorruptData(
                "page_size must be > 0".into(),
            ));
        }
        if !self.page_size.is_multiple_of(self.block_size) {
            return Err(VectorSearchError::CorruptData(format!(
                "page_size ({}) must be a multiple of block_size ({})",
                self.page_size, self.block_size,
            )));
        }
        Ok(())
    }
}

/// Gridstore-style payload storage.
///
/// Manages a set of mmap-backed data pages, a tracker for direct slot→pointer
/// lookup, and a bitmask for free-space management. All mutation goes through
/// `parking_lot::RwLock` on the inner components; readers use mmap snapshots.
///
/// Durability model: the WAL is the authoritative log and carries the full
/// payload of every write, so `put` touches memory and page files only.
/// [`PayloadStore::checkpoint`] persists the tracker and bitmask and flushes
/// the dirty pages; it runs at WAL checkpoints, capacity changes and
/// compaction, never per point. Reads always consult the in-memory tracker,
/// so a stale on-disk copy never affects correctness — only replay time.
pub struct PayloadStore {
    base_path: PathBuf,
    config: StoreConfig,
    tracker: parking_lot::RwLock<Tracker>,
    bitmask: parking_lot::RwLock<MmapBitmask>,
    pages: parking_lot::RwLock<Pages>,
}

impl PayloadStore {
    /// Create a new payload store at `path`.
    pub fn create(path: &Path, config: StoreConfig) -> Result<Self> {
        config.validate()?;
        std::fs::create_dir_all(path)?;

        let tracker = Tracker::create(path, &config)?;
        let bitmask = MmapBitmask::create(path, &config)?;
        let pages = Pages::create(path, &config)?;

        // Persist config.
        let config_path = path.join("config.json");
        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| VectorSearchError::CorruptData(format!("config serialize: {e}")))?;
        std::fs::write(&config_path, config_json)?;

        Ok(Self {
            base_path: path.to_path_buf(),
            config,
            tracker: parking_lot::RwLock::new(tracker),
            bitmask: parking_lot::RwLock::new(bitmask),
            pages: parking_lot::RwLock::new(pages),
        })
    }

    /// Open an existing payload store at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        let config_path = path.join("config.json");
        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|e| VectorSearchError::CorruptData(format!("cannot read config.json: {e}")))?;
        let config: StoreConfig = serde_json::from_str(&config_json).map_err(|e| {
            VectorSearchError::CorruptData(format!("cannot parse config.json: {e}"))
        })?;
        config.validate()?;

        let tracker = Tracker::open(path, &config)?;
        let bitmask = MmapBitmask::open(path, &config)?;
        let pages = Pages::open(path)?;

        Ok(Self {
            base_path: path.to_path_buf(),
            config,
            tracker: parking_lot::RwLock::new(tracker),
            bitmask: parking_lot::RwLock::new(bitmask),
            pages: parking_lot::RwLock::new(pages),
        })
    }

    /// Read the payload for a slot.
    pub fn get(&self, slot: u32) -> Result<Option<Payload>> {
        let pointer = {
            let tracker = self.tracker.read();
            match tracker.get(slot) {
                Some(ptr) => ptr,
                None => return Ok(None),
            }
        };
        if pointer.is_empty() {
            return Ok(None);
        }
        let pages = self.pages.read();
        let bytes = pages.read_value(&pointer, &self.config)?;
        if bytes.is_empty() {
            return Ok(None);
        }
        let payload: Payload = serde_json::from_slice(&bytes).map_err(|e| {
            VectorSearchError::CorruptData(format!("bad payload blob at slot {slot}: {e}"))
        })?;
        Ok(Some(payload))
    }

    /// Write or overwrite the payload for a slot (full replace).
    ///
    /// Allocates new blocks, writes the data, and updates the tracker and
    /// bitmask in memory. Nothing is fsynced here: the owning WAL transaction
    /// already carries this payload durably, and
    /// [`PayloadStore::checkpoint`] persists the derived state in batch.
    ///
    /// The page map is only ever taken for reading here: `ensure_page` and
    /// `write_value` serialize their own mutations through the inner page map,
    /// so a payload write never blocks a reader inspecting another page.
    pub fn put(&self, slot: u32, payload: Option<&Payload>) -> Result<()> {
        let bytes = match payload {
            Some(p) => serde_json::to_vec(p)?,
            None => Vec::new(),
        };

        let mut tracker = self.tracker.write();
        let bitmask = self.bitmask.write();

        // Free old blocks if the slot had data.
        if let Some(old_ptr) = tracker.get(slot) {
            if !old_ptr.is_empty() {
                bitmask.free_blocks(
                    old_ptr.page_id as usize,
                    old_ptr.block_offset as usize,
                    old_ptr.length,
                    &self.config,
                );
            }
        }

        if bytes.is_empty() {
            tracker.set(slot, ValuePointer::empty());
            return Ok(());
        }

        // Allocate contiguous blocks.
        let blocks_needed = self.blocks_needed(bytes.len());
        let (page_id, block_offset) = bitmask.allocate(blocks_needed, &self.config)?;

        let pages = self.pages.read();
        // Ensure the page exists.
        pages.ensure_page(page_id as usize, &self.config)?;

        // Write data to pages.
        let pointer = ValuePointer::new(page_id, block_offset, bytes.len() as u32);
        pages.write_value(&pointer, &bytes, &self.config)?;

        // In-memory tracker only; persisted at the next checkpoint.
        tracker.set(slot, pointer);

        Ok(())
    }

    /// Persist all derived state: the block bitmask, the slot→pointer tracker
    /// and every page written since the last checkpoint. Runs at WAL
    /// checkpoint time, collection close and compaction commit — never inside
    /// the per-point write path.
    pub fn checkpoint(&self) -> Result<()> {
        let pages_dirty = {
            let pages = self.pages.read();
            pages.sync()
        };
        {
            let bitmask = self.bitmask.read();
            bitmask.persist()?;
        }
        let tracker = self.tracker.read();
        tracker.save(&self.base_path, &self.config)?;
        drop(tracker);
        if pages_dirty > 0 {
            tracing::debug!(pages = pages_dirty, "payload store checkpoint");
        }
        Ok(())
    }

    /// Delete specific keys from a slot's payload.
    pub fn delete_keys(&self, slot: u32, keys: &[&str]) -> Result<()> {
        let mut current = match self.get(slot)? {
            Some(p) => p,
            None => return Ok(()),
        };
        for key in keys {
            current.remove(*key);
        }
        self.put(slot, Some(&current))
    }

    /// Merge a partial payload into a slot's payload (merge semantics):
    /// keys present in `partial` overwrite their previous values while all
    /// other keys are preserved. Creates the payload if the slot has none.
    ///
    /// A single-field write (`set_field`) is expressed as a one-entry merge;
    /// block allocation always follows the write-to-new model, so no
    /// compaction is needed to reclaim superseded blobs.
    pub fn merge<I>(&self, slot: u32, partial: I) -> Result<()>
    where
        I: IntoIterator<Item = (String, serde_json::Value)>,
    {
        let mut merged = self.get(slot)?.unwrap_or_default();
        for (key, value) in partial {
            merged.insert(key, value);
        }
        self.put(slot, Some(&merged))
    }

    /// Compact the store: rebuild it with only the live slots specified
    /// by `live_slots`. Returns a new compacted PayloadStore at `dest_path`.
    ///
    /// The rebuilt store is fully checkpointed before it is returned, so the
    /// caller can swap the directory in without extra persistence steps.
    pub fn compact_to(
        &self,
        dest_path: &Path,
        live_slots: &[(u32, u32)], // (old_slot, new_slot)
    ) -> Result<PayloadStore> {
        let new_store = PayloadStore::create(dest_path, self.config.clone())?;
        for &(old_slot, new_slot) in live_slots {
            let payload = self.get(old_slot)?;
            new_store.put(new_slot, payload.as_ref())?;
        }
        new_store.checkpoint()?;
        Ok(new_store)
    }

    /// Number of blocks needed for a given byte length.
    fn blocks_needed(&self, len: usize) -> u32 {
        len.div_ceil(self.config.block_size) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_store() -> (PayloadStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store =
            PayloadStore::create(dir.path(), StoreConfig::default()).expect("create payload store");
        (store, dir)
    }

    fn payload(kv: &[(&str, serde_json::Value)]) -> Payload {
        kv.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn test_put_and_get() {
        let (store, _dir) = temp_store();
        let p = payload(&[("color", json!("red")), ("size", json!(42))]);
        store.put(0, Some(&p)).unwrap();
        let got = store.get(0).unwrap().unwrap();
        assert_eq!(got.get("color"), Some(&json!("red")));
        assert_eq!(got.get("size"), Some(&json!(42)));
    }

    #[test]
    fn test_overwrite() {
        let (store, _dir) = temp_store();
        store.put(0, Some(&payload(&[("a", json!(1))]))).unwrap();
        store.put(0, Some(&payload(&[("b", json!(2))]))).unwrap();
        let got = store.get(0).unwrap().unwrap();
        assert!(!got.contains_key("a"));
        assert_eq!(got.get("b"), Some(&json!(2)));
    }

    #[test]
    fn test_delete_slot() {
        let (store, _dir) = temp_store();
        store.put(0, Some(&payload(&[("x", json!(1))]))).unwrap();
        store.put(0, None).unwrap();
        assert!(store.get(0).unwrap().is_none());
    }

    #[test]
    fn test_delete_keys() {
        let (store, _dir) = temp_store();
        store
            .put(
                0,
                Some(&payload(&[
                    ("a", json!(1)),
                    ("b", json!(2)),
                    ("c", json!(3)),
                ])),
            )
            .unwrap();
        store.delete_keys(0, &["b", "c"]).unwrap();
        let got = store.get(0).unwrap().unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got.get("a"), Some(&json!(1)));
    }

    #[test]
    fn test_multiple_slots() {
        let (store, _dir) = temp_store();
        for i in 0..100u32 {
            store.put(i, Some(&payload(&[("id", json!(i))]))).unwrap();
        }
        for i in 0..100u32 {
            let got = store.get(i).unwrap().unwrap();
            assert_eq!(got.get("id"), Some(&json!(i)));
        }
    }

    #[test]
    fn test_large_payload() {
        let (store, _dir) = temp_store();
        let big_value = "x".repeat(10_000);
        store
            .put(0, Some(&payload(&[("data", json!(big_value.clone()))])))
            .unwrap();
        let got = store.get(0).unwrap().unwrap();
        assert_eq!(got.get("data").unwrap().as_str().unwrap(), &big_value);
    }

    #[test]
    fn test_merge_creates_and_updates() {
        let (store, _dir) = temp_store();
        // A merge on an empty slot creates the payload.
        store.merge(0, [("a".to_string(), json!(1))]).unwrap();
        assert_eq!(store.get(0).unwrap().unwrap().get("a"), Some(&json!(1)));
        // Merging preserves the other keys and overwrites only given ones.
        store
            .merge(
                0,
                [
                    ("b".to_string(), json!("x")),
                    ("a".to_string(), json!(true)),
                ],
            )
            .unwrap();
        let got = store.get(0).unwrap().unwrap();
        assert_eq!(got.get("a"), Some(&json!(true)));
        assert_eq!(got.get("b"), Some(&json!("x")));
    }

    #[test]
    fn test_put_leaves_tracker_on_disk_stale_until_checkpoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let store = PayloadStore::create(dir.path(), StoreConfig::default()).expect("create");
            // A burst of writes must not rewrite the whole pointer table per
            // point (the quadratic behavior that used to stall the engine).
            for i in 0..200u32 {
                store.put(i, Some(&payload(&[("id", json!(i))]))).unwrap();
            }
            // On-disk tracker is untouched: replay from WAL is the recovery
            // path, so per-point persistence is unnecessary.
            let persisted =
                Tracker::open(dir.path(), &store.config).expect("open persisted tracker");
            assert!(persisted.get(199).is_none());

            store.checkpoint().unwrap();
            let persisted =
                Tracker::open(dir.path(), &store.config).expect("open persisted tracker");
            assert_eq!(
                persisted.get(199).expect("slot 199 persisted"),
                store.tracker.read().get(199).expect("slot 199 in memory"),
            );
        }
        {
            let store = PayloadStore::open(dir.path()).expect("open");
            for i in 0..200u32 {
                let got = store.get(i).unwrap().unwrap();
                assert_eq!(got.get("id"), Some(&json!(i)));
            }
        }
    }

    #[test]
    fn test_put_then_checkpoint_survives_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        const SLOTS: u32 = 64;
        {
            let store = PayloadStore::create(dir.path(), StoreConfig::default()).expect("create");
            for i in 0..SLOTS {
                store.put(i, Some(&payload(&[("id", json!(i))]))).unwrap();
            }
            store.checkpoint().unwrap();
        }
        {
            // The checkpoint persists the data pages and the pointer table, so
            // reopening the store on its own still yields every write.
            let store = PayloadStore::open(dir.path()).expect("open");
            for i in 0..SLOTS {
                let got = store
                    .get(i)
                    .unwrap()
                    .unwrap_or_else(|| panic!("slot {i} lost"));
                assert_eq!(got.get("id"), Some(&json!(i)));
            }
        }
    }

    #[test]
    fn test_checkpoint_persists_overwrite_and_delete() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let store = PayloadStore::create(dir.path(), StoreConfig::default()).expect("create");
            for i in 0..8u32 {
                store.put(i, Some(&payload(&[("id", json!(i))]))).unwrap();
            }
            store.checkpoint().unwrap();
            // Overwrite one slot, free another, then persist the new state.
            store
                .put(0, Some(&payload(&[("id", json!("zero"))])))
                .unwrap();
            store.put(3, None).unwrap();
            store.checkpoint().unwrap();
        }
        {
            let store = PayloadStore::open(dir.path()).expect("open");
            assert_eq!(
                store.get(0).unwrap().unwrap().get("id"),
                Some(&json!("zero"))
            );
            assert!(store.get(3).unwrap().is_none(), "freed slot stays free");
            // The bitmask persisted at the checkpoint made those blocks
            // reusable after reopen.
            store
                .put(3, Some(&payload(&[("id", json!("three"))])))
                .unwrap();
            assert_eq!(
                store.get(3).unwrap().unwrap().get("id"),
                Some(&json!("three"))
            );
        }
    }

    #[test]
    fn test_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let store = PayloadStore::create(dir.path(), StoreConfig::default()).expect("create");
            store.put(0, Some(&payload(&[("x", json!(1))]))).unwrap();
            store.put(1, Some(&payload(&[("y", json!(2))]))).unwrap();
            store.checkpoint().unwrap();
        }
        {
            let store = PayloadStore::open(dir.path()).expect("open");
            assert_eq!(store.get(0).unwrap().unwrap().get("x"), Some(&json!(1)));
            assert_eq!(store.get(1).unwrap().unwrap().get("y"), Some(&json!(2)));
        }
    }

    #[test]
    fn test_reuse_freed_blocks() {
        let (store, _dir) = temp_store();
        // Write to slot 0, delete, write to slot 1 — slot 1 should reuse freed blocks.
        store.put(0, Some(&payload(&[("a", json!(1))]))).unwrap();
        store.put(0, None).unwrap();
        store.put(1, Some(&payload(&[("b", json!(2))]))).unwrap();
        assert_eq!(store.get(1).unwrap().unwrap().get("b"), Some(&json!(2)));
    }
}
