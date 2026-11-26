use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};

mod index;
mod quantization;
mod remote_hints;
mod storage;
mod vectors;

pub use self::index::*;
pub use self::quantization::*;
pub use self::remote_hints::*;
pub use self::storage::*;
pub use self::vectors::*;

pub type CollectionName = String;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub vector_size: usize,
    pub distance: DistanceMetric,
    /// Collection vector kind; defaults to `Dense` for configs created
    /// before the field existed.
    #[serde(default)]
    pub kind: VectorKind,
    /// Collection vector layout. `Single` (default) interprets the
    /// `kind`/`vector_size`/`distance` fields above; `Named` carries its own
    /// per-space parameters and ignores those fields.
    #[serde(default)]
    pub vectors: VectorsConfig,
    /// Parameters for `VectorKind::Multi` collections; must be set when
    /// (and only when) `kind` is `Multi`.
    #[serde(default)]
    pub multi_config: Option<MultiConfig>,
    pub index: IndexConfig,
    pub quantization_config: Option<QuantizationConfig>,
    pub replication_factor: Option<usize>,
    pub write_consistency_factor: Option<usize>,
    pub on_disk_payload: Option<bool>,
    pub shard_number: Option<usize>,
    /// Optional admission cap on live points. When set, an upsert that would
    /// grow the live set past this limit fails with
    /// [`VectorSearchError::CapacityExceeded`](crate::error::VectorSearchError::CapacityExceeded)
    /// instead of silently allocating unbounded index memory. `None` = no cap.
    #[serde(default)]
    pub max_points: Option<u64>,
    /// Whether the dense vectors themselves live on disk (Qdrant
    /// `vectors.on_disk`). The local engine always memory-maps its vector
    /// segment and therefore only carries this field so the remote backend
    /// can be described with the same type.
    #[serde(default)]
    pub vectors_on_disk: Option<bool>,
    /// Remote storage-engine hints (optimizers, WAL, strict mode). The local
    /// engine persists but never acts on them.
    #[serde(default)]
    pub storage_hints: Option<StorageHints>,
}

impl CollectionConfig {
    pub fn new(vector_size: usize, distance: DistanceMetric) -> Self {
        Self {
            vector_size,
            distance,
            kind: VectorKind::Dense,
            vectors: VectorsConfig::Single,
            multi_config: None,
            index: IndexConfig::default(),
            quantization_config: None,
            replication_factor: None,
            write_consistency_factor: None,
            on_disk_payload: None,
            shard_number: None,
            max_points: None,
            vectors_on_disk: None,
            storage_hints: None,
        }
    }

    /// Declare a `VectorKind::Sparse` collection. `vector_size` is the
    /// vocabulary bound: every sparse index must be smaller than it.
    pub fn with_sparse(vector_size: usize, distance: DistanceMetric) -> Self {
        Self {
            kind: VectorKind::Sparse,
            index: IndexConfig::Flat,
            ..Self::new(vector_size, distance)
        }
    }

    /// Declare a `VectorKind::Multi` collection.
    pub fn with_multi(vector_size: usize, distance: DistanceMetric, config: MultiConfig) -> Self {
        Self {
            kind: VectorKind::Multi,
            multi_config: Some(config),
            index: IndexConfig::Flat,
            ..Self::new(vector_size, distance)
        }
    }

    /// Validate the kind-specific parameter combinations. Dense collections
    /// must not carry kind parameters; Sparse/Multi collections may not
    /// select an ANN tier or quantization (both operate on fixed-length
    /// dense vectors only).
    pub fn validate_kind(&self) -> Result<()> {
        match self.kind {
            VectorKind::Dense => {
                if self.multi_config.is_some() {
                    return Err(VectorSearchError::InvalidConfig(
                        "dense collections must not declare multi parameters".to_string(),
                    ));
                }
            }
            VectorKind::Sparse => {
                if self.multi_config.is_some() {
                    return Err(VectorSearchError::InvalidConfig(
                        "sparse collections must not declare multi parameters".to_string(),
                    ));
                }
                if self.vector_size == 0 {
                    return Err(VectorSearchError::InvalidConfig(
                        "sparse vector_size must be >= 1".to_string(),
                    ));
                }
                validate_sparse_distance(self.distance)?;
            }
            VectorKind::Multi => {
                if self.multi_config.is_none() {
                    return Err(VectorSearchError::InvalidConfig(
                        "multi collections must declare multi parameters".to_string(),
                    ));
                }
                if self.vector_size == 0 {
                    return Err(VectorSearchError::InvalidConfig(
                        "multi sub-vector dimension must be >= 1".to_string(),
                    ));
                }
                if self
                    .multi_config
                    .is_some_and(|c| c.max_vectors_per_point == 0)
                {
                    return Err(VectorSearchError::InvalidConfig(
                        "multi max_vectors_per_point must be >= 1".to_string(),
                    ));
                }
            }
        }
        if self.kind != VectorKind::Dense {
            if !matches!(self.index, IndexConfig::Flat) {
                return Err(VectorSearchError::InvalidConfig(format!(
                    "{} collections do not support ANN indexes; use IndexConfig::Flat",
                    self.kind.as_name()
                )));
            }
            if self.quantization_config.is_some() {
                return Err(VectorSearchError::InvalidConfig(format!(
                    "{} collections do not support quantization",
                    self.kind.as_name()
                )));
            }
        }
        if matches!(self.vectors, VectorsConfig::Named(_)) {
            if self.kind != VectorKind::Dense || self.multi_config.is_some() {
                return Err(VectorSearchError::InvalidConfig(
                    "named-vectors collections must keep the default single-space kind fields"
                        .to_string(),
                ));
            }
            if !matches!(self.index, IndexConfig::Flat) {
                return Err(VectorSearchError::InvalidConfig(
                    "named-vectors collections do not support ANN indexes; use IndexConfig::Flat"
                        .to_string(),
                ));
            }
            if self.quantization_config.is_some() {
                return Err(VectorSearchError::InvalidConfig(
                    "named-vectors collections do not support quantization".to_string(),
                ));
            }
            if let VectorsConfig::Named(spaces) = &self.vectors {
                if spaces.is_empty() {
                    return Err(VectorSearchError::InvalidConfig(
                        "named-vectors collections must declare at least one space".to_string(),
                    ));
                }
                for (name, space) in spaces {
                    if name.is_empty() {
                        return Err(VectorSearchError::InvalidConfig(
                            "named-vector space names must not be empty".to_string(),
                        ));
                    }
                    if space.kind == VectorKind::Multi {
                        return Err(VectorSearchError::InvalidConfig(format!(
                            "named-vector space {name} must not use the multi kind"
                        )));
                    }
                    if space.vector_size == 0 {
                        return Err(VectorSearchError::InvalidConfig(format!(
                            "named-vector space {name} vector_size must be >= 1"
                        )));
                    }
                    if space.kind == VectorKind::Sparse {
                        validate_sparse_distance(space.distance).map_err(|e| {
                            VectorSearchError::InvalidConfig(format!(
                                "named-vector space {name}: {e}"
                            ))
                        })?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn with_ivf(mut self, ivf_config: IvfConfig) -> Self {
        self.index = IndexConfig::Ivf(Some(ivf_config));
        self
    }

    pub fn with_index_type(mut self, index_type: IndexType) -> Self {
        self.index = match index_type {
            IndexType::HNSW => IndexConfig::Hnsw(HnswConfig::default()),
            IndexType::IVF => IndexConfig::Ivf(None),
            IndexType::FLAT => IndexConfig::Flat,
        };
        self
    }

    pub fn with_hnsw(mut self, hnsw_config: HnswConfig) -> Self {
        self.index = IndexConfig::Hnsw(hnsw_config);
        self
    }

    pub fn index_type(&self) -> IndexType {
        self.index.index_type()
    }

    pub fn with_quantization(mut self, quantization_config: QuantizationConfig) -> Self {
        self.quantization_config = Some(quantization_config);
        self
    }

    pub fn with_shard_number(mut self, shard_number: usize) -> Self {
        self.shard_number = Some(shard_number);
        self
    }

    pub fn with_on_disk_payload(mut self, on_disk_payload: bool) -> Self {
        self.on_disk_payload = Some(on_disk_payload);
        self
    }

    /// Set the optional live-point admission cap.
    pub fn with_max_points(mut self, max_points: u64) -> Self {
        self.max_points = Some(max_points);
        self
    }

    /// Set whether the dense vectors live on disk (remote backend only).
    pub fn with_vectors_on_disk(mut self, vectors_on_disk: bool) -> Self {
        self.vectors_on_disk = Some(vectors_on_disk);
        self
    }

    /// Attach remote storage-engine hints.
    pub fn with_storage_hints(mut self, storage_hints: StorageHints) -> Self {
        self.storage_hints = Some(storage_hints);
        self
    }
}

/// Sparse scoring supports raw dot and normalized cosine only. Distance
/// metrics would require a different upper-bound scheme in the inverted
/// index and are rejected at configuration time.
pub fn validate_sparse_distance(distance: DistanceMetric) -> Result<()> {
    match distance {
        DistanceMetric::Dot | DistanceMetric::Cosine => Ok(()),
        DistanceMetric::Euclid | DistanceMetric::Manhattan => {
            Err(VectorSearchError::InvalidConfig(format!(
                "sparse collections do not support {distance:?} distance"
            )))
        }
    }
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self::new(1536, DistanceMetric::Cosine)
    }
}
