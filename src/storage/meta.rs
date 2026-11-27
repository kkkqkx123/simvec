//! Collection metadata persisted to `meta.bin`.
//!
//! ```text
//! [0..4)   magic            "VSM2"
//! [4..8)   envelope format  u32 = 1
//! [8..12)  crc32            over the postcard body
//! [12..)   postcard(Meta)
//! ```
//!
//! `meta.bin` describes the capacity of every other file, so it is the one
//! artifact that cannot be rebuilt from the data. A checksum turns a silent
//! bit flip (`slot_capacity` becoming nonsense and every length assertion
//! downstream failing) into a clean "this collection is corrupt" signal that
//! the engine can quarantine.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::types::{DistanceMetric, IndexConfig, QuantizationConfig, StorageHints};

/// Current on-disk format version.
///
/// Development stage: locked at 1.  `Meta::validate` rejects any file whose
/// version differs, so a stale directory is rebuilt instead of misread.
/// Version evolution starts with the first production release, following the
/// same policy as `INDEX_VERSION` / `HNSW_VERSION` in `index::persist`.
pub(crate) const FORMAT_VERSION: u32 = 1;

/// Envelope format for `meta.bin` itself (independent of `FORMAT_VERSION`,
/// which versions the `Meta` payload).
const META_MAGIC: [u8; 4] = *b"VSMC";
const META_ENVELOPE_VERSION: u32 = 1;
const META_HEADER_LEN: usize = 12;

/// Default number of slots per `vectors.bin` segment.
///
/// This is the *cap*, not a fixed size: [`default_segment_slots`] shrinks the
/// segment for wide vectors so one mmap stays a few megabytes.
pub(crate) const SEGMENT_SLOTS_DEFAULT: u32 = 8192;

/// Bytes one `vectors.bin` segment aims for.
///
/// Segments exist so a reader maps a bounded range and a growing collection
/// remaps one segment instead of the whole file, so the useful size is a
/// memory/remap trade-off that does not depend on the dimension. A fixed slot
/// count does depend on it: 8192 slots are 4 MiB at 128 dimensions and 25 MiB
/// at 768, and a one-point collection still allocates a whole segment.
pub(crate) const SEGMENT_TARGET_BYTES: u64 = 4 * 1024 * 1024;

/// Smallest segment, so a very wide vector still amortizes the mmap call.
pub(crate) const SEGMENT_SLOTS_MIN: u32 = 1024;

/// Segment size for `dim`-dimensional f32 vectors: about
/// [`SEGMENT_TARGET_BYTES`] per segment, clamped to
/// `[SEGMENT_SLOTS_MIN, SEGMENT_SLOTS_DEFAULT]`.
///
/// The clamp keeps every dimension at or below 128 exactly on the historical
/// 8192-slot segment (4 MiB at dim 128), so only wide vectors change layout.
pub(crate) fn default_segment_slots(dim: usize) -> u32 {
    let bytes_per_slot = (dim.max(1) as u64) * 4;
    let slots = SEGMENT_TARGET_BYTES / bytes_per_slot;
    (slots as u32).clamp(SEGMENT_SLOTS_MIN, SEGMENT_SLOTS_DEFAULT)
}

/// Per-collection metadata. Written with postcard to `meta.bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub format_version: u32,
    pub collection: String,
    pub vector_size: usize,
    pub distance: DistanceMetric,
    /// ANN tier selection in force for this collection. The three variants
    /// are mutually exclusive; `Ivf(None)` behaves as exact scan until trained.
    pub index: IndexConfig,
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
    /// Highest lsn applied to the serving files.
    pub last_applied_txn: u64,
    /// Optional live-point admission cap (see `CollectionConfig::max_points`).
    #[serde(default)]
    pub max_points: Option<u64>,
    /// Whether dense vectors live on disk (see
    /// `CollectionConfig::vectors_on_disk`). Remote-backend metadata only.
    #[serde(default)]
    pub vectors_on_disk: Option<bool>,
    /// Remote storage-engine hints (see `CollectionConfig::storage_hints`).
    #[serde(default)]
    pub storage_hints: Option<StorageHints>,
    pub created_at: i64,
}

impl Meta {
    pub fn new(collection: &str, vector_size: usize, distance: DistanceMetric) -> Self {
        Self::new_with_segment_slots(
            collection,
            vector_size,
            distance,
            default_segment_slots(vector_size),
        )
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
            index: IndexConfig::default(),
            quantization_config: None,
            segment_slots,
            slot_capacity: segment_slots as u64,
            next_slot: 0,
            live_count: 0,
            tombstone_count: 0,
            last_applied_txn: 0,
            max_points: None,
            vectors_on_disk: None,
            storage_hints: None,
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
        let body = postcard::to_stdvec(self)?;
        let mut bytes = Vec::with_capacity(META_HEADER_LEN + body.len());
        bytes.extend_from_slice(&META_MAGIC);
        bytes.extend_from_slice(&META_ENVELOPE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&crate::checksum::hash(&body).to_le_bytes());
        bytes.extend_from_slice(&body);

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
        if bytes.len() < META_HEADER_LEN {
            return Err(VectorSearchError::CorruptData(format!(
                "meta.bin truncated: {} bytes",
                bytes.len()
            )));
        }
        let magic = &bytes[0..4];
        if magic != META_MAGIC {
            return Err(VectorSearchError::CorruptData(format!(
                "meta.bin bad magic: {magic:?}"
            )));
        }
        let envelope = u32::from_le_bytes(bytes[4..8].try_into().expect("version slice"));
        if envelope != META_ENVELOPE_VERSION {
            return Err(VectorSearchError::CorruptData(format!(
                "meta.bin unsupported envelope version {envelope}"
            )));
        }
        let expected_crc = u32::from_le_bytes(bytes[8..12].try_into().expect("crc slice"));
        let body = &bytes[META_HEADER_LEN..];
        let actual_crc = crate::checksum::hash(body);
        if actual_crc != expected_crc {
            return Err(VectorSearchError::CorruptData(format!(
                "meta.bin checksum mismatch: stored {expected_crc:#x}, computed {actual_crc:#x}"
            )));
        }
        let meta = postcard::from_bytes::<Self>(body)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(dir: &Path) -> Meta {
        let meta = Meta::new_with_segment_slots("probe", 4, DistanceMetric::Dot, 8);
        meta.save(dir).expect("save");
        meta
    }

    #[test]
    fn roundtrip_preserves_fields() {
        let dir = tempfile::tempdir().expect("tempdir");
        let meta = sample(dir.path());
        let loaded = Meta::load(dir.path()).expect("load");
        assert_eq!(loaded.collection, meta.collection);
        assert_eq!(loaded.vector_size, meta.vector_size);
        assert_eq!(loaded.slot_capacity, meta.slot_capacity);
        assert_eq!(loaded.created_at, meta.created_at);
    }

    #[test]
    fn checksum_rejects_single_bit_flip() {
        let dir = tempfile::tempdir().expect("tempdir");
        sample(dir.path());
        let path = dir.path().join("meta.bin");
        let mut bytes = std::fs::read(&path).expect("read");
        // Flip a byte inside the payload, past the 12-byte envelope header.
        let last = bytes.len() - 1;
        bytes[last] ^= 0b0000_0001;
        std::fs::write(&path, &bytes).expect("write");

        let err = Meta::load(dir.path()).expect_err("must reject");
        assert!(err.to_string().contains("checksum"), "{err}");
    }

    /// Segment size follows the dimension instead of a fixed slot count: a
    /// 768-dimensional segment is ~4 MiB, not the 25 MiB an 8192-slot segment
    /// would be, and narrow vectors keep the historical layout.
    #[test]
    fn segment_slots_scale_with_the_dimension() {
        assert_eq!(default_segment_slots(4), SEGMENT_SLOTS_DEFAULT);
        assert_eq!(default_segment_slots(128), SEGMENT_SLOTS_DEFAULT);
        assert_eq!(default_segment_slots(768), 1365);
        // Wide vectors hit the floor rather than shrinking without bound.
        assert_eq!(default_segment_slots(1536), SEGMENT_SLOTS_MIN);
        assert_eq!(default_segment_slots(4096), SEGMENT_SLOTS_MIN);

        // Whatever the dimension, one segment stays within the target's
        // neighbourhood (the floor can exceed it for very wide vectors).
        for dim in [4usize, 64, 128, 256, 768, 1024] {
            let bytes = default_segment_slots(dim) as u64 * dim as u64 * 4;
            assert!(
                bytes <= 4 * SEGMENT_TARGET_BYTES,
                "dim {dim} segment is {bytes} bytes"
            );
        }
    }

    #[test]
    fn bad_magic_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        sample(dir.path());
        let path = dir.path().join("meta.bin");
        let mut bytes = std::fs::read(&path).expect("read");
        bytes[0] = b'X';
        std::fs::write(&path, &bytes).expect("write");
        let err = Meta::load(dir.path()).expect_err("must reject");
        assert!(err.to_string().contains("magic"), "{err}");
    }
}
