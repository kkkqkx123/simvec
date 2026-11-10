use std::sync::Arc;

use memmap2::Mmap;

use crate::error::{Result, VectorSearchError};
use crate::types::{DistanceMetric, QuantizationConfig, QuantizationType};

use super::{QuantMeta, QuantStore};

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

    /// Quantized distance between query f32 and stored code.
    /// Returns internal distance (smaller = nearer) consistent with `distance`.
    pub fn distance_quantized(&self, query: &[f32], code: &[u8], metric: DistanceMetric) -> f32 {
        let meta = self.meta.read();
        match &self.config.quant_type {
            QuantizationType::Scalar { .. } => {
                // Dequantize code to approx f32 and compute true distance.
                let min = meta.scalar_min.unwrap_or(0.0);
                let scale = meta.scalar_scale.unwrap_or(1.0 / 255.0);
                // Avoid per-slot allocation by computing on the fly in naive loop.
                // We'll materialize dequantized vector into stack small vec.
                // For performance we inline distance computation.
                match metric {
                    DistanceMetric::Euclid => {
                        let mut sum = 0.0f32;
                        for (i, &q) in query.iter().enumerate() {
                            let approx = code[i] as f32 * scale + min;
                            let d = q - approx;
                            sum += d * d;
                        }
                        sum
                    }
                    DistanceMetric::Dot => {
                        let mut dot = 0.0f32;
                        for (i, &q) in query.iter().enumerate() {
                            let approx = code[i] as f32 * scale + min;
                            dot += q * approx;
                        }
                        -dot
                    }
                    DistanceMetric::Cosine => {
                        let mut dot = 0.0f32;
                        let mut norm_q = 0.0f32;
                        let mut norm_c = 0.0f32;
                        for (i, &q) in query.iter().enumerate() {
                            let approx = code[i] as f32 * scale + min;
                            dot += q * approx;
                            norm_q += q * q;
                            norm_c += approx * approx;
                        }
                        let denom = (norm_q * norm_c).sqrt();
                        if denom == 0.0 {
                            return 1.0;
                        }
                        1.0 - (dot / denom).clamp(-1.0, 1.0)
                    }
                    DistanceMetric::Manhattan => {
                        let mut sum = 0.0f32;
                        for (i, &q) in query.iter().enumerate() {
                            let approx = code[i] as f32 * scale + min;
                            sum += (q - approx).abs();
                        }
                        sum
                    }
                }
            }
            QuantizationType::Binary { .. } => {
                // Hamming distance; for all metrics lower hamming = nearer.
                // Compute popcount of xor between query bits and stored bits.
                let qbits = encode_binary(query);
                let mut ham = 0u32;
                for (a, b) in qbits.iter().zip(code.iter()) {
                    ham += (a ^ b).count_ones();
                }
                ham as f32
            }
            QuantizationType::Product { .. } => {
                if let (Some(cb), Some(&m), Some(&subdim)) = (
                    meta.codebook.as_ref(),
                    meta.pq_m.as_ref(),
                    meta.pq_subdim.as_ref(),
                ) {
                    match metric {
                        DistanceMetric::Euclid => {
                            // ADC: precompute distance tables per subspace.
                            // For Euclid we use squared L2.
                            let mut total = 0.0f32;
                            for (s, &code_byte) in code.iter().enumerate().take(m) {
                                let start = s * subdim;
                                let qsub = &query[start..start + subdim];
                                let base = s * 256 * subdim;
                                let c = code_byte as usize;
                                let centroid = &cb[base + c * subdim..base + (c + 1) * subdim];
                                total += squared_l2(qsub, centroid);
                            }
                            total
                        }
                        DistanceMetric::Dot => {
                            // Approximate inner product via codebook.
                            let mut dot = 0.0f32;
                            for (s, &code_byte) in code.iter().enumerate().take(m) {
                                let start = s * subdim;
                                let qsub = &query[start..start + subdim];
                                let base = s * 256 * subdim;
                                let c = code_byte as usize;
                                let centroid = &cb[base + c * subdim..base + (c + 1) * subdim];
                                for (a, b) in qsub.iter().zip(centroid.iter()) {
                                    dot += a * b;
                                }
                            }
                            -dot
                        }
                        DistanceMetric::Cosine => {
                            // Norm on the fly for both query and reconstructed approx vector.
                            // Reconstruct approx vector.
                            let mut dot = 0.0f32;
                            let mut norm_q = 0.0f32;
                            let mut norm_c = 0.0f32;
                            for (s, &code_byte) in code.iter().enumerate().take(m) {
                                let start = s * subdim;
                                let qsub = &query[start..start + subdim];
                                let base = s * 256 * subdim;
                                let c = code_byte as usize;
                                let centroid = &cb[base + c * subdim..base + (c + 1) * subdim];
                                for (a, b) in qsub.iter().zip(centroid.iter()) {
                                    dot += a * b;
                                    norm_q += a * a;
                                    norm_c += b * b;
                                }
                            }
                            let denom = (norm_q * norm_c).sqrt();
                            if denom == 0.0 {
                                return 1.0;
                            }
                            1.0 - (dot / denom).clamp(-1.0, 1.0)
                        }
                        DistanceMetric::Manhattan => {
                            let mut sum = 0.0f32;
                            for (s, &code_byte) in code.iter().enumerate().take(m) {
                                let start = s * subdim;
                                let qsub = &query[start..start + subdim];
                                let base = s * 256 * subdim;
                                let c = code_byte as usize;
                                let centroid = &cb[base + c * subdim..base + (c + 1) * subdim];
                                for (a, b) in qsub.iter().zip(centroid.iter()) {
                                    sum += (a - b).abs();
                                }
                            }
                            sum
                        }
                    }
                } else {
                    f32::INFINITY
                }
            }
        }
    }
}

pub(super) fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

pub(super) fn encode_binary(vector: &[f32]) -> Vec<u8> {
    encode_binary_for_query(vector)
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
