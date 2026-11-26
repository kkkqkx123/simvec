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

/// How a multi-vector space compares two points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MultiComparator {
    #[default]
    MaxSim,
}

/// Parameters of a multi-vector space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MultiConfig {
    /// Maximum number of sub-vectors stored per point. Resource guard only;
    /// it does not change MaxSim semantics.
    pub max_vectors_per_point: usize,
    /// Comparison rule applied to two points in this space.
    #[serde(default)]
    pub comparator: MultiComparator,
}

impl MultiConfig {
    pub fn new(max_vectors_per_point: usize) -> Self {
        Self {
            max_vectors_per_point,
            comparator: MultiComparator::MaxSim,
        }
    }
}

/// Immutable parameters of one named vector space.
///
/// A space is dense, sparse or multi-dense; multi is a dense space with an
/// optional `multi_config` rather than a separate point payload kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorSpaceConfig {
    pub kind: VectorKind,
    /// Dense / multi sub-vector dimension. Sparse spaces use
    /// [`SPARSE_DIM_BOUND`] and do not declare a vocabulary bound.
    pub vector_size: usize,
    pub distance: DistanceMetric,
    /// Present exactly when `kind` is `Multi`.
    #[serde(default)]
    pub multi_config: Option<MultiConfig>,
}

impl VectorSpaceConfig {
    pub fn dense(vector_size: usize, distance: DistanceMetric) -> Self {
        Self {
            kind: VectorKind::Dense,
            vector_size,
            distance,
            multi_config: None,
        }
    }

    pub fn multi(vector_size: usize, distance: DistanceMetric, config: MultiConfig) -> Self {
        Self {
            kind: VectorKind::Multi,
            vector_size,
            distance,
            multi_config: Some(config),
        }
    }

    pub fn sparse(distance: DistanceMetric) -> Self {
        Self {
            kind: VectorKind::Sparse,
            vector_size: SPARSE_DIM_BOUND,
            distance,
            multi_config: None,
        }
    }

    /// Dimension used by payload validation.
    pub fn effective_dim(&self) -> usize {
        if self.kind == VectorKind::Sparse {
            SPARSE_DIM_BOUND
        } else {
            self.vector_size
        }
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
    /// those fields mirror the default space. The map is authoritative for
    /// every store; the mirrors exist for dense metadata and callers that
    /// construct configurations from a dimension and distance pair.
    pub fn normalize(&mut self) {
        if self.vectors.is_empty() {
            let space = self.space_from_primary_fields();
            self.vectors.insert(DEFAULT_VECTOR_NAME.to_string(), space);
        } else if !self.is_named_layout() {
            // Struct-literal configs can override the mirror fields without
            // rebuilding the map. The explicit single-space fields win for a
            // single-space collection; validate_kind still rejects conflicts
            // when called directly on an unnormalized config.
            let disagree = self
                .vectors
                .get(DEFAULT_VECTOR_NAME)
                .is_some_and(|space| {
                    self.vector_size != space.vector_size
                        || self.distance != space.distance
                        || self.kind != space.kind
                        || self.multi_config != space.multi_config
                });
            if disagree {
                let space = self.space_from_primary_fields();
                self.vectors.insert(DEFAULT_VECTOR_NAME.to_string(), space);
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
            self.vector_size = space.vector_size;
            self.distance = space.distance;
            self.kind = space.kind;
            self.multi_config = space.multi_config;
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
