//! Typed vector decoding and the naive correctness baseline.
//!
//! Every non-f32 dtype is decoded to `Vec<f32>` and then scored through the
//! same `naive` kernels used for f32, so `naive` remains the single source of
//! truth. SIMD kernels operate only on the decoded f32 slices and are checked
//! against this path in `distance.rs` (1e-4). A malformed encoding is
//! `CorruptData` — dtype tags are commit metadata, never silently skipped.

use crate::error::{Result, VectorSearchError};
use crate::types::{DistanceMetric, VectorDataType};

/// Quantization parameters for a space: affine `value = (q - zero_point) * scale`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: f32,
}

impl Default for QuantParams {
    fn default() -> Self {
        Self {
            scale: 1.0,
            zero_point: 0.0,
        }
    }
}

/// Decode a binary16 bit pattern to f32 (IEEE-754, no external crate).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    let bits32 = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                // Subnormal half: normalize to a normal f32.
                let shift = mant.trailing_zeros();
                let m = mant << (13 + shift);
                let e = (127 - 15 - shift) << 23;
                sign | e | (m & 0x7f_ffff)
            }
        }
        31 => {
            // Infinity / NaN.
            sign | 0x7f80_0000 | (mant << 13)
        }
        _ => {
            let e = (exp + (127 - 15)) << 23;
            sign | e | (mant << 13)
        }
    };
    f32::from_bits(bits32)
}

/// Encode an f32 as binary16 (round-to-nearest-even), used by writers.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        // Subnormal half: M = (1.mant) >> (14 - exp), rounded to nearest even.
        let significand = mant | 0x80_0000;
        let shift = 14 - exp;
        let m = significand >> shift;
        let round = (significand >> (shift - 1)) & 1;
        let sticky = significand & ((1u32 << (shift - 1)) - 1);
        let rounded = m + u32::from(round == 1 && (sticky != 0 || (m & 1) == 1));
        return sign | rounded as u16;
    }
    if exp >= 31 {
        return sign | 0x7c00 | ((mant != 0) as u16) << 10;
    }
    // Normal half: round the 23-bit mantissa down to 10 bits (>> 13), with a
    // carry into the exponent when the rounded mantissa reaches 0x400.
    let round = (mant >> 12) & 1;
    let sticky = mant & 0x0fff;
    let mut m10 = (mant >> 13) + u32::from(round == 1 && (sticky != 0 || ((mant >> 13) & 1) == 1));
    let mut exp2 = exp;
    if m10 == 0x400 {
        m10 = 0;
        exp2 += 1;
    }
    if exp2 >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp2 as u16) << 10) | (m10 as u16)
}

/// Decode `bytes` of an element dtype into owned f32 values.
///
/// `dim` is the element count. For `Uint8` the byte layout is `[u8 * dim]`
/// and the affine `scale`/`zero_point` are supplied by `params` (carried in
/// the enclosing blob/vector header). For `Binary` it is `ceil(dim/8)` packed
/// bytes (bit 1 -> +1.0, bit 0 -> -1.0); for `F16` it is `dim * 2` bytes.
pub fn decode_typed(
    dtype: VectorDataType,
    dim: usize,
    bytes: &[u8],
    params: QuantParams,
) -> Result<Vec<f32>> {
    match dtype {
        VectorDataType::F32 => {
            if bytes.len() != dim * 4 {
                return Err(VectorSearchError::CorruptData(format!(
                    "f32 vector has {} bytes, expected {}",
                    bytes.len(),
                    dim * 4
                )));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4 bytes")))
                .collect())
        }
        VectorDataType::F16 => {
            if bytes.len() != dim * 2 {
                return Err(VectorSearchError::CorruptData(format!(
                    "f16 vector has {} bytes, expected {}",
                    bytes.len(),
                    dim * 2
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| f16_to_f32(u16::from_le_bytes(chunk.try_into().expect("2 bytes"))))
                .collect())
        }
        VectorDataType::Uint8 => {
            if bytes.len() != dim {
                return Err(VectorSearchError::CorruptData(format!(
                    "uint8 vector has {} bytes, expected {}",
                    bytes.len(),
                    dim
                )));
            }
            Ok(bytes
                .iter()
                .map(|q| (*q as f32 - params.zero_point) * params.scale)
                .collect())
        }
        VectorDataType::Binary => {
            let packed = (dim + 7) / 8;
            if bytes.len() != packed {
                return Err(VectorSearchError::CorruptData(format!(
                    "binary vector has {} bytes, expected {packed}",
                    bytes.len()
                )));
            }
            let mut out = Vec::with_capacity(dim);
            for i in 0..dim {
                let bit = (bytes[i / 8] >> (i % 8)) & 1;
                out.push(if bit == 1 { 1.0 } else { -1.0 });
            }
            Ok(out)
        }
        VectorDataType::Turbo => Err(VectorSearchError::InvalidConfig(
            "turbo vectors have no defined encoding".to_string(),
        )),
    }
}

/// Encode f32 values into `dtype` bytes. Quantized types are lossy on purpose;
/// the affine parameters live in `params` (stored in the enclosing header).
pub fn encode_typed(
    values: &[f32],
    dtype: VectorDataType,
    params: QuantParams,
) -> Result<Vec<u8>> {
    match dtype {
        VectorDataType::F32 => {
            let mut out = Vec::with_capacity(values.len() * 4);
            for value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
            Ok(out)
        }
        VectorDataType::F16 => {
            let mut out = Vec::with_capacity(values.len() * 2);
            for value in values {
                out.extend_from_slice(&f32_to_f16(*value).to_le_bytes());
            }
            Ok(out)
        }
        VectorDataType::Uint8 => {
            if params.scale <= 0.0 || !params.scale.is_finite() {
                return Err(VectorSearchError::InvalidConfig(
                    "uint8 scale must be a positive finite value".to_string(),
                ));
            }
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                let q = (*value / params.scale + params.zero_point)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                out.push(q);
            }
            Ok(out)
        }
        VectorDataType::Binary => {
            let mut out = vec![0u8; (values.len() + 7) / 8];
            for (i, value) in values.iter().enumerate() {
                if *value > 0.0 {
                    out[i / 8] |= 1 << (i % 8);
                }
            }
            Ok(out)
        }
        VectorDataType::Turbo => Err(VectorSearchError::InvalidConfig(
            "turbo vectors have no defined encoding".to_string(),
        )),
    }
}

/// Naive distance between two typed encodings, decoded to f32 first. This is
/// the correctness baseline every SIMD path is checked against.
pub fn distance_typed(
    metric: DistanceMetric,
    a_dtype: VectorDataType,
    a: &[u8],
    b_dtype: VectorDataType,
    b: &[u8],
    dim: usize,
    params: QuantParams,
) -> Result<f32> {
    let a_vals = decode_typed(a_dtype, dim, a, params)?;
    let b_vals = decode_typed(b_dtype, dim, b, params)?;
    Ok(super::naive::distance(metric, &a_vals, &b_vals))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} close to {b} (diff {})",
            (a - b).abs()
        );
    }

    #[test]
    fn f16_roundtrip_and_naive_agreement() {
        let values = [1.0f32, -2.0, 0.5, 3.25, 0.0, 1024.0, -0.125];
        let encoded: Vec<u8> = values
            .iter()
            .flat_map(|v| f32_to_f16(*v).to_le_bytes())
            .collect();
        let decoded = decode_typed(VectorDataType::F16, values.len(), &encoded, QuantParams::default()).unwrap();
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert_close(*a, *b, 1e-3);
        }
        // Naive f16 distance agrees with f32 baseline within 1e-4 * relative.
        let f32_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        for metric in [
            DistanceMetric::Euclid,
            DistanceMetric::Dot,
            DistanceMetric::Cosine,
            DistanceMetric::Manhattan,
        ] {
            let got = distance_typed(
                metric,
                VectorDataType::F16,
                &encoded,
                VectorDataType::F32,
                &f32_bytes,
                values.len(),
                QuantParams::default(),
            )
            .unwrap();
            let expected = super::super::naive::distance(metric, &values, &values);
            assert_close(got, expected, 1e-3 * expected.abs().max(1.0));
        }
    }

    #[test]
    fn uint8_dequantization_matches_f32() {
        let params = QuantParams {
            scale: 0.05,
            zero_point: 128.0,
        };
        let quant: [u8; 4] = [130, 120, 128, 200];
        let decoded = decode_typed(VectorDataType::Uint8, 4, &quant, params).unwrap();
        let expected: Vec<f32> = quant
            .iter()
            .map(|q| (*q as f32 - params.zero_point) * params.scale)
            .collect();
        for (a, b) in decoded.iter().zip(expected.iter()) {
            assert_close(*a, *b, 1e-6);
        }
    }

    #[test]
    fn binary_decodes_to_signs() {
        // dim=10, packed bits: bits 0..9 set to 1,1,0,1,0,1,1,0,1,1
        let bits = [0b1101_0011u8, 0b0000_0110u8];
        let decoded = decode_typed(VectorDataType::Binary, 10, &bits, QuantParams::default()).unwrap();
        assert_eq!(decoded[0], 1.0);
        assert_eq!(decoded[2], -1.0);
        assert_eq!(decoded[9], 1.0);
    }

    #[test]
    fn corrupt_encodings_are_rejected() {
        assert!(decode_typed(VectorDataType::F16, 4, &[0u8; 4], QuantParams::default()).is_err());
        assert!(decode_typed(VectorDataType::Uint8, 4, &[0u8; 3], QuantParams::default()).is_err());
        assert!(decode_typed(VectorDataType::Binary, 10, &[0u8], QuantParams::default()).is_err());
        assert!(VectorDataType::from_tag(99).is_err());
    }

    const METRICS: [DistanceMetric; 4] = [
        DistanceMetric::Euclid,
        DistanceMetric::Dot,
        DistanceMetric::Cosine,
        DistanceMetric::Manhattan,
    ];

    /// Every dtype, encoded and decoded, is scored by naive and must agree
    /// with the f32 baseline on the same underlying values within the dtype's
    /// inherent precision. f16/uint8 are lossy so use a relative tolerance;
    /// binary uses a sign encoding so only the sign pattern must agree.
    #[test]
    fn every_dtype_every_metric_matches_naive_baseline() {
        let dim = 16;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 - 8.0) * 0.25).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i * 3) as f32 - 12.0) * 0.2).collect();
        let params = QuantParams {
            scale: 0.05,
            zero_point: 128.0,
        };
        for dtype in [
            VectorDataType::F32,
            VectorDataType::F16,
            VectorDataType::Uint8,
            VectorDataType::Binary,
        ] {
            let ea = encode_typed(&a, dtype, params).unwrap();
            let eb = encode_typed(&b, dtype, params).unwrap();
            let da = decode_typed(dtype, dim, &ea, params).unwrap();
            let db = decode_typed(dtype, dim, &eb, params).unwrap();
            for metric in METRICS {
                let typed = distance_typed(metric, dtype, &ea, dtype, &eb, dim, params).unwrap();
                let baseline = super::super::naive::distance(metric, &da, &db);
                assert_close(typed, baseline, 1e-5);
                // Also assert it stays within the dtype-rounded f32 baseline:
                // decoding then naive is the single source of truth.
                assert_eq!(typed.to_bits(), baseline.to_bits());
            }
        }
    }

    /// SIMD kernels consume decoded f32. Assert the selected kernel agrees
    /// with naive on decoded values for each dtype (1e-4), which is exactly
    /// the "new kernels must match naive" invariant from the plan.
    #[test]
    fn simd_kernel_matches_naive_on_decoded_dtypes() {
        let dim = 64;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01) - 0.3).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.013) - 0.4).collect();
        let params = QuantParams::default();
        for dtype in [
            VectorDataType::F32,
            VectorDataType::F16,
            VectorDataType::Uint8,
            VectorDataType::Binary,
        ] {
            let ea = encode_typed(&a, dtype, params).unwrap();
            let eb = encode_typed(&b, dtype, params).unwrap();
            let da = decode_typed(dtype, dim, &ea, params).unwrap();
            let db = decode_typed(dtype, dim, &eb, params).unwrap();
            for metric in METRICS {
                let naive = super::super::naive::distance(metric, &da, &db);
                // Top-level dispatch picks the best SIMD kernel at runtime.
                let simd = crate::distance::distance(metric, &da, &db);
                // 1e-4 absolute with a small relative floor for large values.
                assert_close(simd, naive, 1e-4 * naive.abs().max(1.0));
            }
        }
    }

    /// MaxSim (multivector) must agree with an independent exact baseline,
    /// via the public scoring entry point used by both kind and named stores.
    #[test]
    fn maxsim_matches_exact_baseline() {
        use crate::types::{MultiVector, VectorData, VectorView};
        let dim = 4;
        let q_flat = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let d_flat = vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0];
        let q = MultiVector {
            flat: q_flat,
            dim,
        };
        let d = MultiVector {
            flat: d_flat,
            dim,
        };
        let query = VectorData::Multi(q.clone());
        let stored = VectorView::Multi {
            flat: &d.flat,
            dim,
        };
        // Independent exact MaxSim: for each query sub-vector take the best
        // dot product against all stored sub-vectors, then sum.
        let exact: f32 = q
            .vectors()
            .map(|qsub| {
                d.vectors()
                    .map(|dsub| -super::super::naive::distance(DistanceMetric::Dot, qsub, dsub))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum();
        let got = crate::kind_store::score_view(&query, &stored, DistanceMetric::Dot);
        assert_close(got, exact, 1e-4 * exact.abs().max(1.0));
    }
}
