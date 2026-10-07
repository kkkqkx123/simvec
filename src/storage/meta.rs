//! Collection metadata persisted to `meta.bin`.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::types::{DistanceMetric, HnswConfig, IndexType, IvfConfig, QuantizationConfig};

/// Current on-disk format version.
///
/// Version 2 adds `quantization_config` for Scalar/Binary/Product quantization
/// support. No migration is provided for version 1 files created before this
/// change — they will be rejected on open and must be recreated. The store is
/// still pre-stable, so breaking format changes are applied directly.
pub(crate) const FORMAT_VERSION: u32 = 2;

/// Default number of slots per `vectors.bin` segment.
pub(crate) const SEGMENT_SLOTS_DEFAULT: u32 = 8192;

/// Per-collection metadata. Written with postcard to `meta.bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub format_version: u32,
    pub collection: String,
    pub vector_size: usize,
    pub distance: DistanceMetric,
    /// ANN tier selection in force for this collection (`HNSW` by default;
    /// `FLAT` keeps the collection on exact scan permanently).
    pub index_type: IndexType,
    /// Effective HNSW configuration (engine defaults already applied) when
    /// `index_type == HNSW`.
    pub hnsw_config: Option<HnswConfig>,
    /// Effective IVF configuration when `index_type == IVF`. `None` = exact
    /// scan only.
    pub ivf_config: Option<IvfConfig>,
    /// Quantization configuration. `None` or `enabled=false` means exact f32
    /// storage only. Persisted atomically with the rest of the meta via
    /// `tmp+rename` so a crash cannot leave a half-written quant state.
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
    pub segment_slots: u32,
    /// Allocated slots (grows in segment steps).
    pub slot_capacity: u64,
    /// High water mark of allocated slots (includes tombstones).
    pub next_slot: u64,
    /// Live points.
    pub live_count: u64,
    pub tombstone_count: u64,
    /// WAL water mark; reserved for the WAL work item.
    pub last_applied_txn: u64,
    pub created_at: i64,
}

impl Meta {
    pub fn new(collection: &str, vector_size: usize, distance: DistanceMetric) -> Self {
        Self::new_with_segment_slots(collection, vector_size, distance, SEGMENT_SLOTS_DEFAULT)
    }

    pub fn new_with_segment_slots(
        collection: &str,
        vector_size: usize,
        distance: DistanceMetric,
        segment_slots: u32,
    ) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            collection: collection.to_string(),
            vector_size,
            distance,
            index_type: IndexType::HNSW,
            hnsw_config: None,
            ivf_config: None,
            quantization_config: None,
            segment_slots,
            slot_capacity: segment_slots as u64,
            next_slot: 0,
            live_count: 0,
            tombstone_count: 0,
            last_applied_txn: 0,
            created_at: Utc::now().timestamp(),
        }
    }

    pub fn is_initial_capacity(&self) -> bool {
        self.slot_capacity == self.segment_slots as u64
    }

    /// Validate invariants on open (aligned with pgvector `CheckDim`).
    pub fn validate(&self) -> Result<()> {
        if self.format_version != FORMAT_VERSION && self.format_version != 1 {
            return Err(VectorSearchError::CorruptData(format!(
                "unsupported format version {}",
                self.format_version
            )));
        }
        if self.vector_size == 0 {
            return Err(VectorSearchError::CorruptData(
                "vector_size must be >= 1".to_string(),
            ));
        }
        if self.segment_slots == 0 {
            return Err(VectorSearchError::CorruptData(
                "segment_slots must be >= 1".to_string(),
            ));
        }
        if self.slot_capacity < self.next_slot {
            return Err(VectorSearchError::CorruptData(format!(
                "slot_capacity {} < next_slot {}",
                self.slot_capacity, self.next_slot
            )));
        }
        Ok(())
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        let bytes = postcard::to_stdvec(self)?;
        let target = dir.join("meta.bin");
        let tmp = dir.join("meta.bin.tmp");
        let mut file = File::create(&tmp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, &target)?;
        sync_parent(dir)?;
        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("meta.bin");
        let bytes = std::fs::read(&path)?;
        // Try current layout first (version 2 with quantization).
        if let Ok(meta) = postcard::from_bytes::<Self>(&bytes) {
            meta.validate()?;
            return Ok(meta);
        }
        // Fallback: old layout version 1 without `quantization_config`.
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct MetaV1 {
            format_version: u32,
            collection: String,
            vector_size: usize,
            distance: DistanceMetric,
            index_type: IndexType,
            hnsw_config: Option<HnswConfig>,
            ivf_config: Option<IvfConfig>,
            segment_slots: u32,
            slot_capacity: u64,
            next_slot: u64,
            live_count: u64,
            tombstone_count: u64,
            last_applied_txn: u64,
            created_at: i64,
        }
        if let Ok(old) = postcard::from_bytes::<MetaV1>(&bytes) {
            let meta = Self {
                format_version: old.format_version,
                collection: old.collection,
                vector_size: old.vector_size,
                distance: old.distance,
                index_type: old.index_type,
                hnsw_config: old.hnsw_config,
                ivf_config: old.ivf_config,
                quantization_config: None,
                segment_slots: old.segment_slots,
                slot_capacity: old.slot_capacity,
                next_slot: old.next_slot,
                live_count: old.live_count,
                tombstone_count: old.tombstone_count,
                last_applied_txn: old.last_applied_txn,
                created_at: old.created_at,
            };
            meta.validate()?;
            return Ok(meta);
        }
        Err(VectorSearchError::CorruptData(
            "meta.bin corrupt or unsupported layout".to_string(),
        ))
    }
}

fn sync_parent(dir: &Path) -> Result<()> {
    let parent = dir.parent().unwrap_or(dir);
    let file = File::open(parent)?;
    file.sync_all()?;
    Ok(())
}
