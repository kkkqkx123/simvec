//! `spaces.json` — authoritative per-space layout metadata for named/kind
//! collections.
//!
//! Root markers (`kind.bin` / `named.bin`) keep only the collection name and
//! a type tag. Every immutable space parameter — kind, distance, dimension,
//! multi config, index and quantization selection, segment width — lives in
//! this checksummed manifest, published atomically inside the generation
//! directory. A mismatch with `generation.bin` is `CorruptData`: the manifest
//! is a commit file, not derived state, so it is never silently rebuilt.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::types::{
    DistanceMetric, IndexConfig, MultiConfig, QuantizationConfig, VectorDataType, VectorKind,
};

/// Affine parameters for an `Uint8` element dtype: `value = (q - zero_point) * scale`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct DtypeParams {
    pub scale: f32,
    pub zero_point: f32,
}

impl Default for DtypeParams {
    fn default() -> Self {
        Self {
            scale: 1.0,
            zero_point: 0.0,
        }
    }
}

use super::snapshot;

/// File name of the space manifest inside a generation directory.
pub(crate) const SPACES_META_FILE: &str = "spaces.json";
const SPACES_FORMAT_VERSION: u32 = 1;

/// Slots covered by one physical segment of a space.
///
/// Exact-scan stores address points by a monotonic slot; `segment_slots`
/// partitions that space into independently checkpointed directories. It is a
/// stable layout parameter recorded in the manifest so reopen uses the same
/// partitioning that wrote the files.
pub(crate) const DEFAULT_SEGMENT_SLOTS: u32 = 4096;

/// Immutable parameters of one declared vector space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SpaceMeta {
    pub kind: VectorKind,
    pub distance: DistanceMetric,
    pub dim: usize,
    /// Sub-vector cap for Multi spaces; `None` for dense/sparse.
    #[serde(default)]
    pub multi_config: Option<MultiConfig>,
    /// Index tier selected for this space. Dense spaces may select HNSW/IVF;
    /// sparse and multi are validated to Flat at creation.
    pub index: IndexConfig,
    /// Quantization selection for this space (dense only).
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    /// Element storage type of dense vectors. `F32` is the default and keeps
    /// the historical byte layout; f16/uint8/binary compress the blob body.
    #[serde(default)]
    pub dtype: VectorDataType,
    /// Affine parameters for `Uint8` dtype; ignored for other types.
    #[serde(default)]
    pub dtype_params: DtypeParams,
    /// Slots per physical segment.
    pub segment_slots: u32,
}

/// Authoritative layout of every space in a collection, bound to one
/// generation and its lsn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SpacesManifest {
    pub version: u32,
    pub collection: String,
    pub generation: u64,
    pub lsn: u64,
    /// Declared spaces in stable BTreeMap order.
    pub spaces: BTreeMap<String, SpaceMeta>,
    /// Admission cap mirrored from the collection configuration.
    #[serde(default)]
    pub max_points: Option<u64>,
}

impl SpacesManifest {
    pub(crate) fn new(
        collection: &str,
        generation: u64,
        lsn: u64,
        spaces: BTreeMap<String, SpaceMeta>,
        max_points: Option<u64>,
    ) -> Self {
        Self {
            version: SPACES_FORMAT_VERSION,
            collection: collection.to_string(),
            generation,
            lsn,
            spaces,
            max_points,
        }
    }

    fn validate(&self, generation_id: u64, generation_lsn: u64) -> Result<()> {
        if self.version != SPACES_FORMAT_VERSION {
            return Err(VectorSearchError::CorruptData(format!(
                "unsupported spaces.json version {}",
                self.version
            )));
        }
        if self.generation != generation_id || self.lsn != generation_lsn {
            return Err(VectorSearchError::CorruptData(format!(
                "spaces.json describes generation {} at lsn {}, but generation.bin says {} at {}",
                self.generation, self.lsn, generation_id, generation_lsn
            )));
        }
        for (name, space) in &self.spaces {
            if space.segment_slots == 0 {
                return Err(VectorSearchError::CorruptData(format!(
                    "space {name:?} has zero segment_slots"
                )));
            }
        }
        Ok(())
    }
}

/// Write the space manifest into a generation directory. Uses the shared
/// checksummed envelope (temporary file + fsync + rename) so a crash can only
/// leave a whole old or whole new manifest.
pub(crate) fn write(dir: &Path, manifest: &SpacesManifest) -> Result<()> {
    snapshot::write(&dir.join(SPACES_META_FILE), manifest)
}

/// Read and validate the space manifest of a generation directory.
pub(crate) fn read(dir: &Path, generation_id: u64, generation_lsn: u64) -> Result<SpacesManifest> {
    let path = dir.join(SPACES_META_FILE);
    let manifest = snapshot::read::<SpacesManifest>(&path)?.ok_or_else(|| {
        VectorSearchError::CorruptData(format!(
            "published generation {} has no {SPACES_META_FILE}",
            generation_id
        ))
    })?;
    manifest.validate(generation_id, generation_lsn)?;
    Ok(manifest)
}

/// Path of one physical segment directory of one space.
pub(crate) fn segment_dir(gen_dir: &Path, space: &str, start: u32) -> PathBuf {
    gen_dir
        .join("spaces")
        .join(space)
        .join("segments")
        .join(format!("{start:010}"))
}
