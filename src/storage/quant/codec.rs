use crate::error::{Result, VectorSearchError};
use crate::types::{DistanceMetric, QuantizationType};

use super::QuantStore;

/// Preprocessed query for quantized scoring.
///
/// Built once per search via [`QuantStore::encode_query`], so the per-slot
/// hot path needs no locks and no per-slot metadata access. Mirrors qdrant's
/// `encode_query` model (see `ref/qdrant/lib/quantization/src/`).
pub enum QuantQuery<'a> {
    Scalar {
        /// Full-precision query; dequantized codes are compared against it.
        query: &'a [f32],
        min: f32,
        scale: f32,
        metric: DistanceMetric,
    },
    Binary {
        bits: Vec<u8>,
    },
    Product {
        /// Per-subspace query->centroid term, `lut[s * 256 + k]`:
        /// Euclid = squared L2 distance; Dot = negated dot product;
        /// L1 = sum of absolute differences; Cosine = dot scaled by `1/||q||`.
        lut: Vec<f32>,
        /// Cosine only: squared norm of each centroid, `norms[s * 256 + k]`.
        norms: Option<Vec<f32>>,
        metric: DistanceMetric,
    },
}

impl QuantQuery<'_> {
    fn metric(&self) -> DistanceMetric {
        match self {
            QuantQuery::Scalar { metric, .. } | QuantQuery::Product { metric, .. } => *metric,
            // Binary scores are Hamming-based regardless of the collection
            // metric (lower hamming = nearer); the negation happens in
            // `QuantStore::score_quantized`.
            QuantQuery::Binary { .. } => DistanceMetric::Euclid,
        }
    }
}

impl QuantStore {
    /// Encode a vector into its quantized byte representation using current meta.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let meta = self.meta.read();
        match &self.config.quant_type {
            QuantizationType::Scalar { .. } => {
                let min = meta.scalar_min.unwrap_or(0.0);
                let max = meta.scalar_max.unwrap_or(1.0);
                let scale = meta.scalar_scale.unwrap_or((max - min) / 255.0);
                let inv_scale = if scale.abs() < 1e-9 { 1.0 } else { 1.0 / scale };
                let mut out = Vec::with_capacity(self.dim);
                for &v in vector {
                    let clamped = v.clamp(min, max);
                    let q = ((clamped - min) * inv_scale).round().clamp(0.0, 255.0) as u8;
                    out.push(q);
                }
                Ok(out)
            }
            QuantizationType::Binary { .. } => {
                let mut out = vec![0u8; self.bytes_per_vector];
                for (i, &v) in vector.iter().enumerate() {
                    if v > 0.0 {
                        out[i / 8] |= 1 << (i % 8);
                    }
                }
                Ok(out)
            }
            QuantizationType::Product { compression, .. } => {
                if !meta.ready {
                    return Err(VectorSearchError::Internal(
                        "product quantization codebook not ready".to_string(),
                    ));
                }
                let codebook = meta.codebook.as_ref().ok_or_else(|| {
                    VectorSearchError::CorruptData("missing product codebook".to_string())
                })?;
                let m = meta.pq_m.unwrap_or(compression.pq_m(self.dim));
                let subdim = meta.pq_subdim.unwrap_or(self.dim / m.max(1));
                let mut out = Vec::with_capacity(m);
                for s in 0..m {
                    let start = s * subdim;
                    let end = start + subdim;
                    let sub = &vector[start..end];
                    // Find nearest centroid in codebook[s*256:(s+1)*256]
                    let base = s * 256 * subdim;
                    let mut best = 0u8;
                    let mut best_dist = f32::INFINITY;
                    for k in 0..256 {
                        let centroid = &codebook[base + k * subdim..base + (k + 1) * subdim];
                        let d = squared_l2(sub, centroid);
                        if d < best_dist {
                            best_dist = d;
                            best = k as u8;
                        }
                    }
                    out.push(best);
                }
                Ok(out)
            }
        }
    }

    /// Preprocess a query into a [`QuantQuery`] so the per-slot scoring loop
    /// is lock-free and, for PQ, a pure LUT lookup (asymmetric distance
    /// computation). Called once per search.
    pub fn encode_query<'a>(
        &self,
        query: &'a [f32],
        metric: DistanceMetric,
    ) -> Result<QuantQuery<'a>> {
        if query.len() != self.dim {
            return Err(VectorSearchError::InvalidVectorDimension {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let meta = self.meta.read();
        match &self.config.quant_type {
            QuantizationType::Scalar { .. } => {
                let min = meta.scalar_min.unwrap_or(0.0);
                let scale = meta.scalar_scale.unwrap_or(1.0 / 255.0);
                Ok(QuantQuery::Scalar {
                    query,
                    min,
                    scale,
                    metric,
                })
            }
            QuantizationType::Binary { .. } => Ok(QuantQuery::Binary {
                bits: encode_binary_for_query(query),
            }),
            QuantizationType::Product { compression, .. } => {
                let codebook = meta.codebook.as_ref().ok_or_else(|| {
                    VectorSearchError::CorruptData("missing product codebook".to_string())
                })?;
                if !meta.ready {
                    return Err(VectorSearchError::Internal(
                        "product quantization codebook not ready".to_string(),
                    ));
                }
                let m = meta.pq_m.unwrap_or(compression.pq_m(self.dim));
                let subdim = meta.pq_subdim.unwrap_or(self.dim / m.max(1));
                let mut lut = Vec::with_capacity(m * 256);
                let mut norms = if matches!(metric, DistanceMetric::Cosine) {
                    Some(Vec::with_capacity(m * 256))
                } else {
                    None
                };
                let inv_qlen = match metric {
                    DistanceMetric::Cosine => {
                        let n: f32 = query.iter().map(|v| v * v).sum();
                        if n <= f32::EPSILON {
                            0.0
                        } else {
                            1.0 / n.sqrt()
                        }
                    }
                    _ => 1.0,
                };
                for s in 0..m {
                    let start = s * subdim;
                    let qsub = &query[start..start + subdim];
                    let base = s * 256 * subdim;
                    for k in 0..256 {
                        let centroid = &codebook[base + k * subdim..base + (k + 1) * subdim];
                        lut.push(match metric {
                            DistanceMetric::Euclid => squared_l2(qsub, centroid),
                            DistanceMetric::Dot => -dot(qsub, centroid),
                            DistanceMetric::Manhattan => l1(qsub, centroid),
                            DistanceMetric::Cosine => dot(qsub, centroid) * inv_qlen,
                        });
                        if let Some(n) = norms.as_mut() {
                            n.push(centroid.iter().map(|v| v * v).sum::<f32>());
                        }
                    }
                }
                Ok(QuantQuery::Product { lut, norms, metric })
            }
        }
    }

    /// Search score (higher = nearer) for a preprocessed query and a stored
    /// code. Applies the collection's score convention, including the binary
    /// Hamming negation. Lock-free: reads only the query and code bytes.
    pub(crate) fn score_quantized_for_query(query: &QuantQuery, code: &[u8]) -> f32 {
        match query {
            QuantQuery::Binary { bits } => {
                let mut ham = 0u32;
                for (a, b) in bits.iter().zip(code.iter()) {
                    ham += (a ^ b).count_ones();
                }
                -(ham as f32)
            }
            q => crate::distance::to_score(q.metric(), Self::distance_quantized_for_query(q, code)),
        }
    }

    /// Quantized distance between a preprocessed query and a stored code.
    /// Returns internal distance (smaller = nearer) consistent with `distance`.
    /// Lock-free: reads only the [`QuantQuery`] and the code bytes — no meta
    /// lock, no per-slot metadata access.
    pub(crate) fn distance_quantized_for_query(query: &QuantQuery, code: &[u8]) -> f32 {
        match query {
            QuantQuery::Scalar {
                query: q,
                min,
                scale,
                metric,
            } => match metric {
                DistanceMetric::Euclid => {
                    let mut sum = 0.0f32;
                    for (i, &qv) in q.iter().enumerate() {
                        let approx = code[i] as f32 * *scale + *min;
                        let d = qv - approx;
                        sum += d * d;
                    }
                    sum
                }
                DistanceMetric::Dot => {
                    let mut acc = 0.0f32;
                    for (i, &qv) in q.iter().enumerate() {
                        let approx = code[i] as f32 * *scale + *min;
                        acc += qv * approx;
                    }
                    -acc
                }
                DistanceMetric::Manhattan => {
                    let mut sum = 0.0f32;
                    for (i, &qv) in q.iter().enumerate() {
                        let approx = code[i] as f32 * *scale + *min;
                        sum += (qv - approx).abs();
                    }
                    sum
                }
                DistanceMetric::Cosine => {
                    let mut acc = 0.0f32;
                    let mut norm_q = 0.0f32;
                    let mut norm_c = 0.0f32;
                    for (i, &qv) in q.iter().enumerate() {
                        let approx = code[i] as f32 * *scale + *min;
                        acc += qv * approx;
                        norm_q += qv * qv;
                        norm_c += approx * approx;
                    }
                    let denom = (norm_q * norm_c).sqrt();
                    if denom == 0.0 {
                        return 1.0;
                    }
                    1.0 - (acc / denom).clamp(-1.0, 1.0)
                }
            },
            QuantQuery::Binary { bits } => {
                let mut ham = 0u32;
                for (a, b) in bits.iter().zip(code.iter()) {
                    ham += (a ^ b).count_ones();
                }
                ham as f32
            }
            QuantQuery::Product { lut, norms, .. } => match norms {
                None => {
                    let mut total = 0.0f32;
                    for (s, &c) in code.iter().enumerate() {
                        total += lut[s * 256 + c as usize];
                    }
                    total
                }
                Some(norms) => {
                    // Cosine: dot sum (already scaled by 1/||q||) over
                    // sqrt of summed centroid squared norms.
                    let mut acc = 0.0f32;
                    let mut norm_c = 0.0f32;
                    for (s, &c) in code.iter().enumerate() {
                        acc += lut[s * 256 + c as usize];
                        norm_c += norms[s * 256 + c as usize];
                    }
                    let denom = norm_c.sqrt();
                    if denom <= f32::EPSILON || !acc.is_finite() {
                        return 1.0;
                    }
                    1.0 - (acc / denom).clamp(-1.0, 1.0)
                }
            },
        }
    }
}

pub(super) fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(super) fn l1(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

pub(crate) fn encode_binary_for_query(vector: &[f32]) -> Vec<u8> {
    let bytes = vector.len().div_ceil(8);
    let mut out = vec![0u8; bytes];
    for (i, &v) in vector.iter().enumerate() {
        if v > 0.0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}
