use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::filter_cond::PointId;
use crate::types::config::{DEFAULT_VECTOR_NAME, VectorKind};

pub type Payload = std::collections::HashMap<String, serde_json::Value>;

/// Flattened multi-vector payload. Keeping sub-vectors contiguous avoids one
/// allocation per token and lets scoring walk a single slice.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MultiVector {
    pub flat: Vec<f32>,
    pub dim: usize,
}

impl MultiVector {
    /// Fallible constructor. Rejects ragged sub-vectors, zero dimensions and
    /// an empty set so an invalid payload cannot enter the store or a query.
    pub fn try_new(vectors: Vec<Vec<f32>>) -> Result<Self> {
        let dim = vectors.first().map_or(0, Vec::len);
        if dim == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "multi vector requires at least one non-zero sub-vector".to_string(),
            ));
        }
        if vectors.iter().any(|v| v.len() != dim) {
            return Err(VectorSearchError::InvalidConfig(
                "multi vector sub-vectors must have equal length".to_string(),
            ));
        }
        Ok(Self {
            flat: vectors.into_iter().flatten().collect(),
            dim,
        })
    }

    /// Convenience constructor for trusted inputs; panics on malformed data.
    /// Use [`Self::try_new`] at any untrusted boundary.
    pub fn new(vectors: Vec<Vec<f32>>) -> Self {
        Self::try_new(vectors).expect("invalid multi vector")
    }

    pub fn vectors(&self) -> impl Iterator<Item = &[f32]> {
        self.flat.chunks_exact(self.dim)
    }

    pub fn count(&self) -> usize {
        self.flat.len() / self.dim
    }
}

/// Payload of one vector space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VectorData {
    Dense(Vec<f32>),
    Sparse { indices: Vec<u32>, values: Vec<f32> },
    Multi(MultiVector),
}

impl VectorData {
    pub fn kind(&self) -> VectorKind {
        match self {
            VectorData::Dense(_) => VectorKind::Dense,
            VectorData::Sparse { .. } => VectorKind::Sparse,
            VectorData::Multi(_) => VectorKind::Multi,
        }
    }

    pub fn dense(vector: Vec<f32>) -> Self {
        Self::Dense(vector)
    }

    pub fn sparse(indices: Vec<u32>, values: Vec<f32>) -> Self {
        Self::Sparse { indices, values }
    }

    /// Fallible multi-vector constructor for untrusted payloads.
    pub fn try_multi(vectors: Vec<Vec<f32>>) -> Result<Self> {
        Ok(Self::Multi(MultiVector::try_new(vectors)?))
    }

    pub fn multi(vectors: Vec<Vec<f32>>) -> Self {
        Self::Multi(MultiVector::new(vectors))
    }

    /// Sort sparse pairs by index so callers do not have to pre-sort input.
    /// Length, uniqueness and finiteness are still checked by [`Self::validate`].
    pub fn normalized(self) -> Self {
        match self {
            VectorData::Sparse { mut indices, mut values } => {
                let mut pairs: Vec<(u32, f32)> = indices.drain(..).zip(values.drain(..)).collect();
                pairs.sort_unstable_by_key(|(index, _)| *index);
                let (indices, values) = pairs.into_iter().unzip();
                Self::Sparse { indices, values }
            }
            other => other,
        }
    }

    /// Validate the payload shape against the space parameters.
    pub fn validate(&self, dim: usize, max_vectors: usize) -> Result<()> {
        match self {
            VectorData::Dense(vector) => {
                if vector.len() != dim {
                    return Err(VectorSearchError::InvalidVectorDimension {
                        expected: dim,
                        actual: vector.len(),
                    });
                }
                validate_finite(vector)?;
            }
            VectorData::Sparse { indices, values } => {
                if indices.len() != values.len() {
                    return Err(VectorSearchError::InvalidConfig(format!(
                        "sparse indices length {} must equal values length {}",
                        indices.len(),
                        values.len()
                    )));
                }
                // Empty sparse vectors are valid: they contribute no postings
                // and an empty sparse query returns an empty result set.
                let mut seen = std::collections::HashSet::with_capacity(indices.len());
                for index in indices.iter() {
                    if !seen.insert(*index) {
                        return Err(VectorSearchError::InvalidConfig(
                            "sparse indices must be unique".to_string(),
                        ));
                    }
                }
                if let Some(&last) = indices.last() {
                    if last as usize >= dim {
                        return Err(VectorSearchError::InvalidVectorDimension {
                            expected: dim,
                            actual: last as usize,
                        });
                    }
                }
                // Negative weights are allowed: there is no posting pruning,
                // so negative contributions cannot hide valid candidates.
                validate_finite(values)?;
            }
            VectorData::Multi(multi) => {
                if multi.dim == 0 {
                    return Err(VectorSearchError::InvalidConfig(
                        "multi vector requires a non-zero sub-vector dimension".to_string(),
                    ));
                }
                if multi.flat.is_empty() {
                    return Err(VectorSearchError::InvalidConfig(
                        "multi vector must contain at least one sub-vector".to_string(),
                    ));
                }
                if multi.flat.len() % multi.dim != 0 {
                    return Err(VectorSearchError::InvalidConfig(
                        "multi vector flat payload is ragged".to_string(),
                    ));
                }
                let count = multi.count();
                if count > max_vectors {
                    return Err(VectorSearchError::InvalidConfig(format!(
                        "multi vector count {count} exceeds configured limit {max_vectors}"
                    )));
                }
                if multi.dim != dim {
                    return Err(VectorSearchError::InvalidVectorDimension {
                        expected: dim,
                        actual: multi.dim,
                    });
                }
                validate_finite(&multi.flat)?;
            }
        }
        Ok(())
    }
}

fn validate_finite(values: &[f32]) -> Result<()> {
    for (i, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(VectorSearchError::NonFiniteElement(i));
        }
    }
    Ok(())
}

/// A point carries one vector per declared space. The default space is used
/// by single-space collections and by the dense storage/WAL paths; named
/// collections simply add more entries to the same map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorPoint {
    pub id: PointId,
    pub vectors: BTreeMap<String, VectorData>,
    pub payload: Option<Payload>,
}

impl VectorPoint {
    pub fn new(id: impl Into<PointId>, vector: Vec<f32>) -> Self {
        Self::with_data(id, VectorData::Dense(vector))
    }

    /// Build a point carrying the payload of one space.
    pub fn with_data(id: impl Into<PointId>, data: VectorData) -> Self {
        Self {
            id: id.into(),
            vectors: BTreeMap::from([(DEFAULT_VECTOR_NAME.to_string(), data)]),
            payload: None,
        }
    }

    /// Build a point carrying one named-space vector.
    pub fn with_named(id: impl Into<PointId>, space: impl Into<String>, data: VectorData) -> Self {
        Self {
            id: id.into(),
            vectors: BTreeMap::from([(space.into(), data)]),
            payload: None,
        }
    }

    /// Attach or replace one named space.
    pub fn with_named_vector(mut self, space: impl Into<String>, data: VectorData) -> Self {
        self.vectors.insert(space.into(), data);
        self
    }

    /// Replace the default space's dense vector, preserving other spaces.
    pub fn set_dense(&mut self, vector: Vec<f32>) {
        self.vectors
            .insert(DEFAULT_VECTOR_NAME.to_string(), VectorData::Dense(vector));
    }

    /// Sort every sparse space so merge-based scoring is always valid.
    pub fn normalize(&mut self) {
        for data in self.vectors.values_mut() {
            *data = data.clone().normalized();
        }
    }

    /// Default-space vector, empty when the point has no dense default space.
    pub fn vector(&self) -> &[f32] {
        match self.vectors.get(DEFAULT_VECTOR_NAME) {
            Some(VectorData::Dense(vector)) => vector,
            _ => &[],
        }
    }

    /// Default-space payload of any kind.
    pub fn data(&self) -> Option<&VectorData> {
        self.vectors.get(DEFAULT_VECTOR_NAME)
    }

    /// Payload of one named space.
    pub fn space(&self, name: &str) -> Option<&VectorData> {
        self.vectors.get(name)
    }

    /// Kind of the default space, or the first space for named-only points.
    pub fn point_kind(&self) -> Option<VectorKind> {
        self.vectors
            .get(DEFAULT_VECTOR_NAME)
            .or_else(|| self.vectors.values().next())
            .map(VectorData::kind)
    }

    pub fn with_payload(mut self, payload: Payload) -> Self {
        self.payload = Some(payload);
        self
    }

    pub fn with_payload_kv(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        let payload = self.payload.get_or_insert_with(HashMap::new);
        payload.insert(key.into(), value);
        self
    }

    pub fn dimension(&self) -> usize {
        match self.vectors.get(DEFAULT_VECTOR_NAME) {
            Some(VectorData::Dense(vector)) => vector.len(),
            Some(VectorData::Multi(multi)) => multi.dim,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorPoints {
    pub points: Vec<VectorPoint>,
}

impl VectorPoints {
    pub fn new(points: Vec<VectorPoint>) -> Self {
        Self { points }
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }
}

impl From<Vec<VectorPoint>> for VectorPoints {
    fn from(points: Vec<VectorPoint>) -> Self {
        Self::new(points)
    }
}
