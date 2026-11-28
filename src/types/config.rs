use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};

mod dtype;
mod index;
mod quantization;
mod remote_hints;
mod storage;
mod vectors;

pub use self::dtype::*;
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
            vectors: VectorsConfig::from([(
                DEFAULT_VECTOR_NAME.to_string(),
                VectorSpaceConfig::dense(vector_size, distance),
            )]),
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

    /// Declare a `VectorKind::Sparse` collection. Sparse ids live in an open
    /// `u32` domain, so there is no vocabulary-bound parameter; the legacy
    /// two-argument form is kept only as a JSON/binding boundary adapter and
    /// its size argument is ignored.
    pub fn with_sparse(distance: DistanceMetric) -> Self {
        let mut config = Self::new(SPARSE_DIM_BOUND, distance);
        config.kind = VectorKind::Sparse;
        config.vectors = VectorsConfig::from([(
            DEFAULT_VECTOR_NAME.to_string(),
            VectorSpaceConfig::sparse(distance),
        )]);
        config.index = IndexConfig::Flat;
        config
    }

    /// Declare a `VectorKind::Multi` collection.
    pub fn with_multi(vector_size: usize, distance: DistanceMetric, config: MultiConfig) -> Self {
        let mut result = Self::new(vector_size, distance);
        result.kind = VectorKind::Multi;
        result.multi_config = Some(config);
        result.vectors = VectorsConfig::from([(
            DEFAULT_VECTOR_NAME.to_string(),
            VectorSpaceConfig::multi(vector_size, distance, config),
        )]);
        result.index = IndexConfig::Flat;
        result
    }

    /// Validate the element dtype of every declared dense space.
    ///
    /// Separate from [`Self::validate_kind`] because it is the subset the
    /// storage layer itself must reject before it creates files: an encoding
    /// with no distance mapping (`Turbo`) has no writer, so a collection
    /// created with it would fail on its first upsert instead of at creation.
    /// It deliberately skips the cross-field mirror checks, so a caller that
    /// only tunes the convenience fields is unaffected.
    pub fn validate_element_types(&self) -> Result<()> {
        for (name, space) in &self.vectors {
            if let VectorSpaceConfig::Dense { distance, dtype, .. } = space {
                dtype.validate_metric(*distance).map_err(|e| {
                    VectorSearchError::InvalidConfig(format!("space {name:?}: {e}"))
                })?;
            }
        }
        Ok(())
    }

    /// Validate every declared space. The single-space convenience fields
    /// are mirrors of the authoritative `vectors` map and are checked against
    /// it so a hand-written configuration cannot select two kinds at once.
    pub fn validate_kind(&self) -> Result<()> {
        if self.vectors.is_empty() {
            return Err(VectorSearchError::InvalidConfig(
                "collection must declare at least one vector space".to_string(),
            ));
        }
        for (name, space) in &self.vectors {
            match space {
                VectorSpaceConfig::Dense {
                    vector_size,
                    distance,
                    multi_config,
                    dtype,
                } => {
                    if *vector_size == 0 {
                        return Err(VectorSearchError::InvalidConfig(format!(
                            "space {name:?} vector_size must be >= 1"
                        )));
                    }
                    dtype.validate_metric(*distance).map_err(|e| {
                        VectorSearchError::InvalidConfig(format!("space {name:?}: {e}"))
                    })?;
                    if let Some(multi) = multi_config {
                        if multi.max_vectors_per_point == 0 {
                            return Err(VectorSearchError::InvalidConfig(format!(
                                "space {name:?} max_vectors_per_point must be >= 1"
                            )));
                        }
                    }
                }
                VectorSpaceConfig::Sparse { distance } => {
                    validate_sparse_distance(*distance).map_err(|e| {
                        VectorSearchError::InvalidConfig(format!("space {name:?}: {e}"))
                    })?;
                }
            }
        }
        // Index and quantization tiers are legal only for dense spaces:
        // sparse uses posting + exact and multi uses MaxSim, neither of which
        // has an ANN graph or a quantization codebook. Validate per space so a
        // named collection may mix an ANN dense space with an exact sparse one.
        let ann = !matches!(self.index, IndexConfig::Flat);
        let quantized = self.quantization_config.is_some();
        for (name, space) in &self.vectors {
            if space.kind() != VectorKind::Dense && ann {
                return Err(VectorSearchError::InvalidConfig(format!(
                    "space {name:?} is {} and does not support ANN indexes; use IndexConfig::Flat",
                    space.kind().as_name()
                )));
            }
            if space.kind() != VectorKind::Dense && quantized {
                return Err(VectorSearchError::InvalidConfig(format!(
                    "space {name:?} is {} and does not support quantization",
                    space.kind().as_name()
                )));
            }
        }
        if !self.is_named_layout() {
            if let Some(space) = self.default_space() {
                if self.vector_size != space.vector_size()
                    || self.distance != space.distance()
                    || self.kind != space.kind()
                    || self.multi_config != space.multi_config()
                {
                    return Err(VectorSearchError::InvalidConfig(
                        "single-space fields disagree with the declared default space".to_string(),
                    ));
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
