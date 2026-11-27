use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};

/// Engine-level storage tuning for the local engine.
///
/// One instance governs every collection under the engine root: these knobs
/// describe the host's IO profile (log size, checkpoint cadence, payload
/// page geometry) rather than a single collection's layout, so they are not
/// part of [`CollectionConfig`]. `Default` reproduces the historical
/// compile-time constants exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StorageConfig {
    /// WAL bytes after which a background checkpoint runs.
    pub wal_checkpoint_bytes: u64,
    /// Bytes after which the WAL rolls over to a new segment file. Segments
    /// bound the amount of log a watermark truncation can drop at once and
    /// keep a single file from growing without bound between checkpoints.
    pub wal_segment_bytes: u64,
    /// Tombstone ratio that triggers physical reclamation.
    pub compaction_threshold: f64,
    /// Transactions between `meta.bin` saves.
    pub meta_save_interval: u64,
    /// Payload store block size in bytes.
    pub payload_block_size: usize,
    /// Payload store page size in bytes.
    pub payload_page_size: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            wal_checkpoint_bytes: 64 * 1024 * 1024,
            wal_segment_bytes: crate::storage::DEFAULT_SEGMENT_BYTES,
            compaction_threshold: 0.20,
            meta_save_interval: 64,
            payload_block_size: 128,
            payload_page_size: 32 * 1024 * 1024,
        }
    }
}

impl StorageConfig {
    /// Validate the tuning knobs. Called when the engine opens so a bad
    /// value fails startup instead of surfacing as a mid-run surprise.
    pub fn validate(&self) -> Result<()> {
        if self.wal_checkpoint_bytes == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "wal_checkpoint_bytes must be >= 1".to_string(),
            ));
        }
        if !(self.compaction_threshold > 0.0 && self.compaction_threshold < 1.0) {
            return Err(VectorSearchError::InvalidConfig(
                "compaction_threshold must be within (0, 1)".to_string(),
            ));
        }
        if self.meta_save_interval == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "meta_save_interval must be >= 1".to_string(),
            ));
        }
        if self.wal_segment_bytes == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "wal_segment_bytes must be >= 1".to_string(),
            ));
        }
        if self.payload_block_size == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "payload_block_size must be >= 1".to_string(),
            ));
        }
        if self.payload_page_size == 0
            || !self
                .payload_page_size
                .is_multiple_of(self.payload_block_size)
        {
            return Err(VectorSearchError::InvalidConfig(
                "payload_page_size must be a positive multiple of payload_block_size".to_string(),
            ));
        }
        Ok(())
    }

    /// Payload store geometry derived from the tuning knobs.
    pub fn payload_store_config(&self) -> crate::storage::payload_store::StoreConfig {
        crate::storage::payload_store::StoreConfig {
            block_size: self.payload_block_size,
            page_size: self.payload_page_size,
        }
    }
}
