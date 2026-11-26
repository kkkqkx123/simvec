use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};

use super::CompressionRatio;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Quantization method. The `always_ram` flag on every variant is a
/// Qdrant-compat reserved bit: storage is always segmented mmap and the flag
/// has no effect on placement.
pub enum QuantizationType {
    Scalar {
        quantile: Option<f32>,
        always_ram: Option<bool>,
    },
    Product {
        compression: CompressionRatio,
        always_ram: Option<bool>,
    },
    Binary {
        always_ram: Option<bool>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub quant_type: QuantizationType,
}

impl QuantizationConfig {
    pub fn scalar(quantile: f32) -> Self {
        Self {
            quant_type: QuantizationType::Scalar {
                quantile: Some(quantile),
                always_ram: Some(true),
            },
        }
    }

    pub fn product(compression: CompressionRatio) -> Self {
        Self {
            quant_type: QuantizationType::Product {
                compression,
                always_ram: Some(true),
            },
        }
    }

    pub fn binary() -> Self {
        Self {
            quant_type: QuantizationType::Binary {
                always_ram: Some(true),
            },
        }
    }

    pub fn with_always_ram(mut self, always_ram: bool) -> Self {
        match &mut self.quant_type {
            QuantizationType::Scalar { always_ram: ar, .. }
            | QuantizationType::Product { always_ram: ar, .. }
            | QuantizationType::Binary { always_ram: ar } => *ar = Some(always_ram),
        }
        self
    }

    pub fn always_ram(&self) -> bool {
        match &self.quant_type {
            QuantizationType::Scalar { always_ram, .. }
            | QuantizationType::Product { always_ram, .. }
            | QuantizationType::Binary { always_ram } => always_ram.unwrap_or(true),
        }
    }

    pub fn quantile_or_default(&self) -> f32 {
        match &self.quant_type {
            QuantizationType::Scalar { quantile, .. } => quantile.unwrap_or(0.99),
            _ => 0.99,
        }
    }

    pub fn validate(&self, dim: usize) -> Result<()> {
        match &self.quant_type {
            QuantizationType::Scalar { quantile, .. } => {
                if let Some(q) = quantile {
                    if !q.is_finite() || *q <= 0.0 || *q > 1.0 {
                        return Err(VectorSearchError::InvalidConfig(format!(
                            "scalar quantization quantile {q} must be in (0,1]"
                        )));
                    }
                }
            }
            QuantizationType::Product { compression, .. } => {
                let m = compression.pq_m(dim);
                if m == 0 || !dim.is_multiple_of(m) {
                    return Err(VectorSearchError::InvalidConfig(format!(
                        "product quantization dim {dim} not divisible by M={m} for compression {compression:?}"
                    )));
                }
            }
            QuantizationType::Binary { .. } => {}
        }
        Ok(())
    }

    pub fn quant_bytes_per_vector(&self, dim: usize) -> usize {
        match &self.quant_type {
            QuantizationType::Scalar { .. } => dim,
            QuantizationType::Binary { .. } => dim.div_ceil(8),
            QuantizationType::Product { compression, .. } => compression.pq_m(dim),
        }
    }

    pub fn is_product(&self) -> bool {
        matches!(self.quant_type, QuantizationType::Product { .. })
    }
}
