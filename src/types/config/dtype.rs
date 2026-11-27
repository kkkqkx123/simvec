//! Element type of a dense vector space.
//!
//! The dtype is an immutable property of a space, recorded in `spaces.json`,
//! the segment vector header, blob headers and WAL blob references. It is
//! validated against the space's distance metric at collection creation:
//! unsupported combinations are rejected rather than silently degraded at
//! query time. `F32` is the default and keeps the historical byte layout.

use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};

use super::DistanceMetric;

/// On-disk element encoding of a vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VectorDataType {
    /// IEEE-754 binary32 (default, historical format).
    #[default]
    F32,
    /// IEEE-754 binary16 half precision.
    F16,
    /// Unsigned 8-bit quantized with a per-space scale and zero point.
    Uint8,
    /// Packed 1-bit values mapped to {-1, +1} for scoring.
    Binary,
    /// Turbo (experimental) encoding; rejected unless the metric mapping is
    /// explicit. Currently no metric qualifies, so it is always rejected.
    Turbo,
}

impl VectorDataType {
    pub fn as_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Uint8 => "uint8",
            Self::Binary => "binary",
            Self::Turbo => "turbo",
        }
    }

    /// Stable on-disk tag stored in blob/vector headers and WAL references.
    pub fn tag(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Uint8 => 2,
            Self::Binary => 3,
            Self::Turbo => 4,
        }
    }

    pub fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Uint8),
            3 => Ok(Self::Binary),
            4 => Ok(Self::Turbo),
            _ => Err(VectorSearchError::CorruptData(format!(
                "unknown vector data type tag {tag}"
            ))),
        }
    }

    /// Bytes per element for the raw encoding (binary is packed, hence a
    /// fractional value handled separately by the decoder).
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Uint8 => 1,
            Self::Binary => 0,
            Self::Turbo => 0,
        }
    }

    /// Validate that this dtype can be scored with `metric`. Quantized and
    /// binary encodings map cleanly onto dot/cosine/euclid; turbo has no
    /// defined mapping yet and is always rejected at creation.
    pub fn validate_metric(self, metric: DistanceMetric) -> Result<()> {
        match self {
            Self::F32 | Self::F16 => Ok(()),
            Self::Uint8 | Self::Binary => match metric {
                DistanceMetric::Dot
                | DistanceMetric::Cosine
                | DistanceMetric::Euclid
                | DistanceMetric::Manhattan => Ok(()),
            },
            Self::Turbo => Err(VectorSearchError::InvalidConfig(
                "turbo vectors have no defined distance mapping".to_string(),
            )),
        }
    }
}
