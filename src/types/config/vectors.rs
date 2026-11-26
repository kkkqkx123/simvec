use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{CollectionConfig, DistanceMetric, IndexConfig};

/// Collection-level vector kind. `Dense` is the historical default; `Sparse`
/// and `Multi` collections are served by the exact-scan kind store
/// (`crate::kind_store`) instead of the dense HNSW/IVF pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VectorKind {
    /// Single dense embedding per point (default).
    #[default]
    Dense,
    /// Sparse (index, weight) pairs scored with a weighted dot product.
    Sparse,
    /// A bounded sequence of dense vectors per point scored with MaxSim.
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

/// Parameters for `VectorKind::Sparse` collections.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SparseConfig {
    /// Vocabulary size bound: a sparse index must satisfy `index < vocab_size`.
    pub vocab_size: usize,
}

/// Parameters for `VectorKind::Multi` collections.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MultiConfig {
    /// Maximum number of sub-vectors stored per point.
    pub max_vectors_per_point: usize,
}

/// Parameters of one named vector space inside a named-vectors collection.
///
/// A named space is either dense or sparse; `Multi` is not a valid named
/// space kind (see `VectorsConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorSpaceConfig {
    /// Space payload kind; must be `Dense` or `Sparse`.
    pub kind: VectorKind,
    /// Dense dimension for `Dense` spaces, vocabulary bound for `Sparse`.
    pub vector_size: usize,
    /// Distance metric scored within this space only.
    pub distance: DistanceMetric,
}

/// Collection vector layout: a single space (the historical shape, keyed by
/// `kind`/`vector_size`/`distance`) or a named map of spaces.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum VectorsConfig {
    /// Single-space layout; the fields on `CollectionConfig` apply.
    #[default]
    Single,
    /// Named layout: every point carries one vector per declared space and
    /// every query targets one space by name. Spaces may mix dense and
    /// sparse kinds; `Multi` is not allowed.
    Named(BTreeMap<String, VectorSpaceConfig>),
}

impl CollectionConfig {
    /// Declare a named-vectors collection. `vector_size`/`distance`/`kind`
    /// on the returned config keep their default single-space values; they
    /// are not interpreted for named layouts.
    pub fn with_named(spaces: BTreeMap<String, VectorSpaceConfig>) -> Self {
        Self {
            vectors: VectorsConfig::Named(spaces),
            index: IndexConfig::Flat,
            ..Self::new(0, DistanceMetric::Cosine)
        }
    }
}
