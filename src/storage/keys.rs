//! `keys.bin` — slot-to-point-id directory with a utf8 blob area.

use std::path::Path;
use std::sync::Arc;

use crate::error::{Result, VectorSearchError};
use crate::storage::directory::{BlobDirectory, DirView, KEY_REC_SIZE};

const MAGIC: [u8; 4] = *b"VKEY";

/// Slot-to-point-id mapping backed by `keys.bin`.
pub struct Keys {
    dir: BlobDirectory,
}

impl Keys {
    pub fn create(path: &Path, initial_capacity: u64) -> Result<Self> {
        Ok(Self {
            dir: BlobDirectory::create(path, MAGIC, KEY_REC_SIZE, initial_capacity)?,
        })
    }

    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            dir: BlobDirectory::open(path, MAGIC, KEY_REC_SIZE)?,
        })
    }

    pub fn snapshot(&self) -> arc_swap::Guard<Arc<DirView>> {
        self.dir.snapshot()
    }

    pub fn grow_to(&self, target_capacity: u64) -> Result<()> {
        self.dir.grow_to(target_capacity)
    }

    /// Atomically replace the backing file with a compacted `tmp_path`.
    pub fn replace_from(&self, tmp_path: &Path) -> Result<()> {
        self.dir.replace_from(tmp_path)
    }

    /// Record the key for a new slot.
    pub fn append_key(&self, slot: usize, key: &str) -> Result<()> {
        self.dir.append_blob(slot, key.as_bytes(), 0)
    }

    /// Read the key for a slot from a snapshot.
    pub fn read_key(view: &DirView, slot: usize) -> Result<Option<String>> {
        match view.blob(KEY_REC_SIZE, slot) {
            Some(bytes) => {
                let key = std::str::from_utf8(bytes).map_err(|e| {
                    VectorSearchError::CorruptData(format!("invalid utf8 key at slot {slot}: {e}"))
                })?;
                Ok(Some(key.to_string()))
            }
            None => Ok(None),
        }
    }
}
