//! `payloads.bin` — per-slot payload records with a tombstone flag.

use std::path::Path;

use crate::error::Result;
use crate::storage::directory::{BlobDirectory, FLAG_TOMBSTONE, SLOT_REC_SIZE};

const MAGIC: [u8; 4] = *b"VPLD";

/// Per-slot payload blobs with tombstone flags, backed by `payloads.bin`.
pub struct Payloads {
    dir: BlobDirectory,
}

impl Payloads {
    pub fn create(path: &Path, initial_capacity: u64) -> Result<Self> {
        Ok(Self {
            dir: BlobDirectory::create(path, MAGIC, SLOT_REC_SIZE, initial_capacity)?,
        })
    }

    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            dir: BlobDirectory::open(path, MAGIC, SLOT_REC_SIZE)?,
        })
    }

    pub fn grow_to(&self, target_capacity: u64) -> Result<()> {
        self.dir.grow_to(target_capacity)
    }

    /// Mark a slot tombstoned (`true`) or alive (`false`).
    pub fn set_tombstone(&self, slot: usize, tombstoned: bool) -> Result<()> {
        let flags = if tombstoned { FLAG_TOMBSTONE } else { 0 };
        self.dir.set_flags(slot, flags)
    }
}
