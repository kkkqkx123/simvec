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
/// support. Version 3 adds `max_points`, the optional live-point admission
/// cap. The store is pre-stable, so breaking format changes are applied
/// directly and older files are rejected on open.
pub(crate) const FORMAT_VERSION: u32 = 3;

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
    /// Optional live-point admission cap (see `CollectionConfig::max_points`).
    #[serde(default)]
    pub max_points: Option<u64>,
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
            max_points: None,
            created_at: Utc::now().timestamp(),
        }
    }

    pub fn is_initial_capacity(&self) -> bool {
        self.slot_capacity == self.segment_slots as u64
    }

    /// Validate invariants on open (aligned with pgvector `CheckDim`).
    pub fn validate(&self) -> Result<()> {
        if self.format_version != FORMAT_VERSION {
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
        let meta = postcard::from_bytes::<Self>(&bytes)?;
        meta.validate()?;
        Ok(meta)
    }
}

fn sync_parent(dir: &Path) -> Result<()> {
    let parent = dir.parent().unwrap_or(dir);
    let file = File::open(parent)?;
    file.sync_all()?;
    Ok(())
}
