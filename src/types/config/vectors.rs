use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{CollectionConfig, DistanceMetric, IndexConfig};

/// Name of the single vector space in a non-named collection. Named layouts
/// use their own explicit names; the default name keeps the dense storage and
/// WAL paths independent of whether a collection declares one space or many.
pub const DEFAULT_VECTOR_NAME: &str = "";

/// Sparse dimensions are open `u32` ids. Configurations do not declare a
/// vocabulary bound; this constant only gives validation a finite upper bound.
pub const SPARSE_DIM_BOUND: usize = u32::MAX as usize + 1;

/// Payload kind of one vector space. Dense and multi spaces share the dense
/// storage pipeline; sparse spaces use the inverted index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VectorKind {
    #[default]
    Dense,
    Sparse,
    Multi,
}

impl VectorKind {
    /// Human-readable name used in diagnostics and unsupported errors.
    pub fn as_name(self) -> &'static str {
        match self {
            VectorKind::Dense => "dense",
            VectorKind::Sparse => "sparse",
            VectorKind::Multi => "multi",
        }
    }
}

/// Parameters of a multi-vector space. MaxSim is the only comparison rule,
/// so there is no comparator knob to configure; reserved capacity is a
/// resource guard rather than a scoring choice.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MultiConfig {
    /// Maximum number of sub-vectors stored per point.
    pub max_vectors_per_point: usize,
}

impl MultiConfig {
    pub fn new(max_vectors_per_point: usize) -> Self {
        Self {
            max_vectors_per_point,
        }
    }
}

/// Immutable parameters of one named vector space.
///
/// Dense and multi are the same storage kind: multi is a dense space carrying
/// a `multi_config`, not a separate payload kind. Sparse declares no dimension
/// at all — its ids are open `u32` — so there is no placeholder `vector_size`
/// to keep in sync. The internally-tagged JSON shape stays compatible with
/// the flat `{ kind, vector_size, distance, multi_config }` form accepted at
/// the FFI boundary; a legacy sparse `vector_size` is simply ignored on input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum VectorSpaceConfig {
    Dense {
        vector_size: usize,
        distance: DistanceMetric,
        #[serde(default)]
        multi_config: Option<MultiConfig>,
    },
    Sparse {
        distance: DistanceMetric,
    },
}

impl VectorSpaceConfig {
    pub fn dense(vector_size: usize, distance: DistanceMetric) -> Self {
        Self::Dense {
            vector_size,
            distance,
            multi_config: None,
        }
    }

    pub fn multi(vector_size: usize, distance: DistanceMetric, config: MultiConfig) -> Self {
        Self::Dense {
            vector_size,
            distance,
            multi_config: Some(config),
        }
    }

    pub fn sparse(distance: DistanceMetric) -> Self {
        Self::Sparse { distance }
    }

    /// Payload kind derived from the variant. A dense space with multi params
    /// reports [`VectorKind::Multi`].
    pub fn kind(&self) -> VectorKind {
        match self {
            Self::Dense {
                multi_config: Some(_),
                ..
            } => VectorKind::Multi,
            Self::Dense { .. } => VectorKind::Dense,
            Self::Sparse { .. } => VectorKind::Sparse,
        }
    }

    pub fn distance(&self) -> DistanceMetric {
        match self {
            Self::Dense { distance, .. } | Self::Sparse { distance } => *distance,
        }
    }

    pub fn multi_config(&self) -> Option<MultiConfig> {
        match self {
            Self::Dense { multi_config, .. } => *multi_config,
            Self::Sparse { .. } => None,
        }
    }

    /// Declared dense sub-vector dimension, or the open-domain validation
    /// bound for sparse spaces (which declare no dimension themselves).
    pub fn vector_size(&self) -> usize {
        match self {
            Self::Dense { vector_size, .. } => *vector_size,
            Self::Sparse { .. } => SPARSE_DIM_BOUND,
        }
    }

    /// Dimension used by payload validation.
    pub fn effective_dim(&self) -> usize {
        self.vector_size()
    }
}

/// All vector spaces declared by a collection, keyed by name. A single-space
/// collection contains only [`DEFAULT_VECTOR_NAME`].
pub type VectorsConfig = BTreeMap<String, VectorSpaceConfig>;

impl CollectionConfig {
    /// Declare a named-vectors collection.
    pub fn with_named(spaces: VectorsConfig) -> Self {
        let mut config = Self {
            vectors: spaces,
            index: IndexConfig::Flat,
            ..Self::new(0, DistanceMetric::Cosine)
        };
        config.sync_primary_fields();
        config
    }

    /// Fill an empty map from the single-space convenience fields, then make
    /// those fields mirror the default space. The `vectors` map is the sole
    /// authority for named / multi-space layouts. The flat convenience fields
    /// exist only as a JSON/struct-literal boundary adapter: for a single
    /// default-space collection they are consumed into the map so callers that
    /// build a config from a dimension and distance pair take effect. Queries
    /// (`collection_config` / `collection_info`) always read back from the map.
    pub fn normalize(&mut self) {
        if self.vectors.is_empty() {
            let space = self.space_from_primary_fields();
            self.vectors.insert(DEFAULT_VECTOR_NAME.to_string(), space);
        } else if !self.is_named_layout() {
            // Single-space struct-literal configs override the mirror fields
            // (e.g. `CollectionConfig { vector_size: 128, ..Default::default() }`).
            // The explicit single-space fields win for the default space; a
            // named layout never gets rewritten this way. This keeps the
            // convenience boundary adapter working while the map stays the
            // authority for reads and for multi-space layouts.
            let disagree = self
                .vectors
                .get(DEFAULT_VECTOR_NAME)
                .is_some_and(|space| {
                    self.vector_size != space.vector_size()
                        || self.distance != space.distance()
                        || self.kind != space.kind()
                        || self.multi_config != space.multi_config()
                });
            if disagree {
                let space = self.space_from_primary_fields();
                self.vectors
                    .insert(DEFAULT_VECTOR_NAME.to_string(), space);
            }
        }
        self.sync_primary_fields();
    }

    fn space_from_primary_fields(&self) -> VectorSpaceConfig {
        match self.kind {
            VectorKind::Multi => VectorSpaceConfig::multi(
                self.vector_size,
                self.distance,
                self.multi_config.unwrap_or_else(|| MultiConfig::new(1)),
            ),
            VectorKind::Sparse => VectorSpaceConfig::sparse(self.distance),
            VectorKind::Dense => VectorSpaceConfig::dense(self.vector_size, self.distance),
        }
    }

    fn sync_primary_fields(&mut self) {
        let primary = self
            .vectors
            .get(DEFAULT_VECTOR_NAME)
            .or_else(|| self.vectors.values().next());
        if let Some(space) = primary {
            self.vector_size = space.vector_size();
            self.distance = space.distance();
            self.kind = space.kind();
            self.multi_config = space.multi_config();
        }
    }

    /// Whether the collection declares more than one space or only named
    /// spaces other than the default one.
    pub fn is_named_layout(&self) -> bool {
        self.vectors.len() != 1 || !self.vectors.contains_key(DEFAULT_VECTOR_NAME)
    }

    /// The default space, when the collection has one.
    pub fn default_space(&self) -> Option<&VectorSpaceConfig> {
        self.vectors.get(DEFAULT_VECTOR_NAME)
    }
}
