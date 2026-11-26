use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::filter_cond::PointId;
use crate::types::config::VectorKind;

pub type Payload = std::collections::HashMap<String, serde_json::Value>;

/// Vector payload shape for non-dense collection kinds.
///
/// Mirrors `vector-contracts::types::VectorData` field for field: sparse
/// indices are strictly increasing and paired with finite weights of the
/// same length; multi holds a non-empty sequence of dense vectors that all
/// share the collection's single-vector dimension.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VectorData {
    /// A single dense embedding vector.
    Dense(Vec<f32>),
    /// Sparse vector: strictly increasing indices paired with finite weights.
    Sparse {
        indices: Vec<u32>,
        values: Vec<f32>,
    },
    /// Multiple dense vectors of identical dimension scored with MaxSim.
    Multi(Vec<Vec<f32>>),
}

impl VectorData {
    /// Kind described by this payload.
    pub fn kind(&self) -> VectorKind {
        match self {
            VectorData::Dense(_) => VectorKind::Dense,
            VectorData::Sparse { .. } => VectorKind::Sparse,
            VectorData::Multi(_) => VectorKind::Multi,
        }
    }

    /// Dense payload builder.
    pub fn dense(vector: Vec<f32>) -> Self {
        Self::Dense(vector)
    }

    /// Sparse payload builder.
    pub fn sparse(indices: Vec<u32>, values: Vec<f32>) -> Self {
        Self::Sparse { indices, values }
    }

    /// Multi-vector payload builder.
    pub fn multi(vectors: Vec<Vec<f32>>) -> Self {
        Self::Multi(vectors)
    }

    /// Validate the payload shape against the collection parameters.
    ///
    /// `dim` is the collection `vector_size` (dense dimension, sparse
    /// vocabulary bound or multi single-vector dimension); `max_vectors`
    /// bounds the per-point sub-vector count for Multi collections.
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
                if indices.is_empty() {
                    return Err(VectorSearchError::InvalidConfig(
                        "sparse vector must not be empty".to_string(),
                    ));
                }
                for pair in indices.windows(2) {
                    if pair[0] >= pair[1] {
                        return Err(VectorSearchError::InvalidConfig(
                            "sparse indices must be strictly increasing".to_string(),
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
                validate_finite(values)?;
            }
            VectorData::Multi(vectors) => {
                if vectors.is_empty() {
                    return Err(VectorSearchError::InvalidConfig(
                        "multi vector must contain at least one sub-vector".to_string(),
                    ));
                }
                if vectors.len() > max_vectors {
                    return Err(VectorSearchError::InvalidConfig(format!(
                        "multi vector count {} exceeds configured limit {max_vectors}",
                        vectors.len()
                    )));
                }
                for vector in vectors {
                    if vector.len() != dim {
                        return Err(VectorSearchError::InvalidVectorDimension {
                            expected: dim,
                            actual: vector.len(),
                        });
                    }
                    validate_finite(vector)?;
                }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorPoint {
    pub id: PointId,
    pub vector: Vec<f32>,
    /// Non-dense payload. Exactly one of `vector` and `data` must be set:
    /// `vector` for Dense collections, `data` for Sparse/Multi collections.
    #[serde(default)]
    pub data: Option<VectorData>,
    pub payload: Option<Payload>,
}

impl VectorPoint {
    pub fn new(id: impl Into<PointId>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
            data: None,
            payload: None,
        }
    }

    /// Build a point carrying a non-dense payload.
    pub fn with_data(id: impl Into<PointId>, data: VectorData) -> Self {
        Self {
            id: id.into(),
            vector: Vec::new(),
            data: Some(data),
            payload: None,
        }
    }

    /// The payload kind this point carries (`None` when both fields are empty).
    pub fn point_kind(&self) -> Option<VectorKind> {
        if !self.vector.is_empty() {
            Some(VectorKind::Dense)
        } else {
            self.data.as_ref().map(VectorData::kind)
        }
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
        self.vector.len()
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
