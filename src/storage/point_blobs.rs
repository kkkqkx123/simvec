//! Vector blob encoding primitives shared by the physical segment stores.
//!
//! This module no longer owns an append-only file container: point payloads
//! live in `SegmentedBlobs` (`spaces/<space>/segments/<start>/`). What remains
//! here is the on-disk encoding of one vector (`BlobRef` handles plus the
//! dtype-aware encode/decode functions) shared by kind/named stores, WAL
//! replay and snapshots.
//!
//! Layout of one vector: a 12-byte header keeps the arrays 4-byte aligned.
//! Byte `[1]` carries the element dtype tag (`0` = f32 keeps the historical
//! layout); bytes `[2..4)` are reserved.

use crate::error::Result;
use crate::types::{VectorData, VectorView};

/// Handle to one point's bytes inside [`PointBlobs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BlobRef {
    pub off: u64,
    pub len: u32,
}

// ---------------------------------------------------------------------------
// Blob encoding
//
// Layout of one vector: a 12-byte header keeps the arrays 4-byte aligned.
//
//   [0]   kind (1 = dense, 2 = sparse, 3 = multi)
//   [1..4) padding
//   [4..8) a  — dense: length; sparse: pair count; multi: sub-vector dimension
//   [8..12) b — dense/sparse: unused; multi: sub-vector count
//   [12..) arrays — dense: f32[a]; sparse: u32[a] then f32[a]; multi: f32[a*b]
const VECTOR_HEADER_BYTES: usize = 12;
const KIND_DENSE: u8 = 1;
const KIND_SPARSE: u8 = 2;
const KIND_MULTI: u8 = 3;

/// Length of the encoding of `data`, so callers can reserve exactly.
pub fn encoded_len(data: &VectorData) -> usize {
    match data {
        VectorData::Dense(values) => VECTOR_HEADER_BYTES + values.len() * 4,
        VectorData::Sparse { indices, .. } => VECTOR_HEADER_BYTES + indices.len() * 8,
        VectorData::Multi(multi) => VECTOR_HEADER_BYTES + multi.flat.len() * 4,
    }
}

/// Encode one vector payload.
pub fn encode_vector(data: &VectorData) -> Vec<u8> {
    let mut out = Vec::with_capacity(encoded_len(data));
    match data {
        VectorData::Dense(values) => {
            push_header(&mut out, KIND_DENSE, values.len() as u32, 0);
            push_f32s(&mut out, values);
        }
        VectorData::Sparse { indices, values } => {
            push_header(&mut out, KIND_SPARSE, indices.len() as u32, 0);
            for index in indices {
                out.extend_from_slice(&index.to_le_bytes());
            }
            push_f32s(&mut out, values);
        }
        VectorData::Multi(multi) => {
            push_header(&mut out, KIND_MULTI, multi.dim as u32, multi.count() as u32);
            push_f32s(&mut out, &multi.flat);
        }
    }
    out
}

/// Decode one vector payload, borrowing its arrays from `bytes`.
pub fn decode_vector(bytes: &[u8]) -> Option<VectorView<'_>> {
    if bytes.len() < VECTOR_HEADER_BYTES {
        return None;
    }
    let kind = bytes[0];
    // Zero-copy decoding only handles the historical f32 encoding. A typed
    // body must go through `decode_vector_typed` instead.
    if bytes[1] != crate::types::VectorDataType::F32.tag() {
        return None;
    }
    let a = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let b = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let body = &bytes[VECTOR_HEADER_BYTES..];
    match kind {
        KIND_DENSE => {
            if body.len() != a * 4 {
                return None;
            }
            Some(VectorView::Dense(as_f32s(body)?))
        }
        KIND_SPARSE => {
            if body.len() != a * 8 {
                return None;
            }
            let (index_bytes, value_bytes) = body.split_at(a * 4);
            Some(VectorView::Sparse {
                indices: as_u32s(index_bytes)?,
                values: as_f32s(value_bytes)?,
            })
        }
        KIND_MULTI => {
            if a == 0 || body.len() != a * b * 4 {
                return None;
            }
            Some(VectorView::Multi {
                flat: as_f32s(body)?,
                dim: a,
            })
        }
        _ => None,
    }
}

/// One named space inside a point blob: `[u16 index][u16 pad][u32 len]` then a
/// 4-aligned vector encoding.
const NAMED_ENTRY_HEADER_BYTES: usize = 8;

/// Pack already-encoded per-space vectors into one named point blob. Used by
/// stores that apply a per-space element dtype before packing.
pub fn pack_named(mut entries: Vec<(u16, Vec<u8>)>) -> Vec<u8> {
    entries.sort_unstable_by_key(|(index, _)| *index);
    let mut out = Vec::with_capacity(4 + entries.len() * NAMED_ENTRY_HEADER_BYTES);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (index, encoded) in entries {
        out.extend_from_slice(&index.to_le_bytes());
        out.extend_from_slice(&[0u8, 0u8]);
        out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        out.extend_from_slice(&encoded);
    }
    out
}

/// Re-encode a point blob with some spaces replaced, copying the untouched
/// entries byte for byte. `replacements` maps a space index to its new
/// encoding.
pub fn rewrite_named(
    old: &[u8],
    replacements: &std::collections::BTreeMap<u16, Vec<u8>>,
) -> Option<Vec<u8>> {
    let mut entries: std::collections::BTreeMap<u16, &[u8]> = std::collections::BTreeMap::new();
    for (index, entry) in named_entries(old)? {
        entries.insert(index, entry);
    }
    for (index, encoded) in replacements {
        entries.insert(*index, encoded.as_slice());
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (index, entry) in entries {
        out.extend_from_slice(&index.to_le_bytes());
        out.extend_from_slice(&[0u8, 0u8]);
        out.extend_from_slice(&(entry.len() as u32).to_le_bytes());
        out.extend_from_slice(entry);
    }
    Some(out)
}

/// The `(space index, encoded vector)` entries of a point blob. One entry per
/// declared space, at most a handful, so a `Vec` keeps the call sites simple.
pub fn named_entries(bytes: &[u8]) -> Option<Vec<(u16, &[u8])>> {
    if bytes.len() < 4 {
        return None;
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let mut rest = &bytes[4..];
    let mut raw: Vec<(u16, &[u8])> = Vec::with_capacity(count);
    for _ in 0..count {
        if rest.len() < NAMED_ENTRY_HEADER_BYTES {
            return None;
        }
        let index = u16::from_le_bytes(rest[0..2].try_into().ok()?);
        let len = u32::from_le_bytes(rest[4..8].try_into().ok()?) as usize;
        let start = NAMED_ENTRY_HEADER_BYTES;
        if rest.len() < start + len {
            return None;
        }
        raw.push((index, &rest[start..start + len]));
        rest = &rest[start + len..];
    }
    Some(raw)
}

fn push_header(out: &mut Vec<u8>, kind: u8, a: u32, b: u32) {
    push_header_dtype(out, kind, crate::types::VectorDataType::F32, a, b);
}

/// Header with an explicit element dtype tag in byte `[1]`. Bytes `[2..4]`
/// stay zero (reserved). `F32` (tag 0) keeps the historical layout byte-for-byte.
fn push_header_dtype(
    out: &mut Vec<u8>,
    kind: u8,
    dtype: crate::types::VectorDataType,
    a: u32,
    b: u32,
) {
    out.push(kind);
    out.push(dtype.tag());
    out.extend_from_slice(&[0u8, 0u8]);
    out.extend_from_slice(&a.to_le_bytes());
    out.extend_from_slice(&b.to_le_bytes());
}

/// Read the dtype tag of one encoded vector. An unknown tag is `None` so the
/// caller can surface `CorruptData` rather than silently decoding as f32.
#[cfg(test)]
pub fn vector_dtype(bytes: &[u8]) -> Option<crate::types::VectorDataType> {
    if bytes.len() < VECTOR_HEADER_BYTES {
        return None;
    }
    crate::types::VectorDataType::from_tag(bytes[1]).ok()
}

/// Encode a dense vector using an explicit element dtype. The body holds the
/// dtype's native bytes (f32 / f16 / uint8 quantized / packed binary) and the
/// affine parameters for uint8 live in the header's `b`/reserved words so a
/// decode needs no external config.
///
/// Returns the encoded bytes plus the `QuantParams` actually used, so callers
/// that care can persist them alongside the blob.
pub fn encode_vector_typed(
    data: &VectorData,
    dtype: crate::types::VectorDataType,
    params: crate::distance::dtype::QuantParams,
) -> Result<(Vec<u8>, crate::distance::dtype::QuantParams)> {
    use crate::distance::dtype::encode_typed;
    use crate::error::VectorSearchError;
    match data {
        VectorData::Dense(values) => {
            let body = encode_typed(values, dtype, params)?;
            let mut out = Vec::with_capacity(VECTOR_HEADER_BYTES + body.len());
            // `a` is the element count (dim), matching `decode_vector`'s dense
            // convention; the decoder derives byte length from dtype.
            push_header_dtype(&mut out, KIND_DENSE, dtype, values.len() as u32, 0);
            out.extend_from_slice(&body);
            Ok((out, params))
        }
        // Sparse and multi payloads carry no element dtype in this format;
        // fall back to the f32 encoding but still stamp the requested tag only
        // when it is F32 (otherwise reject: a typed non-dense body is invalid).
        VectorData::Sparse { .. } | VectorData::Multi(_) => {
            if dtype != crate::types::VectorDataType::F32 {
                return Err(VectorSearchError::InvalidConfig(
                    "element dtype is only supported for dense vectors".to_string(),
                ));
            }
            Ok((encode_vector(data), params))
        }
    }
}

/// Produce a borrowable view for scoring. For the default f32 dtype the slice
/// is borrowed straight out of `bytes` (zero copy). For typed encodings the
/// decoded f32 values are placed into `scratch` and the view borrows that.
/// `scratch` lives for one scoring iteration, so no per-candidate allocation
/// beyond a reused buffer.
pub fn view_for_score<'a>(
    bytes: &'a [u8],
    dtype: crate::types::VectorDataType,
    params: crate::distance::dtype::QuantParams,
    scratch: &'a mut Vec<f32>,
) -> Option<VectorView<'a>> {
    use crate::types::VectorDataType;
    if dtype == VectorDataType::F32 {
        return decode_vector(bytes);
    }
    let data = decode_vector_typed(bytes, params).ok()?;
    match data {
        VectorData::Dense(values) => {
            *scratch = values;
            Some(VectorView::Dense(scratch))
        }
        _ => None,
    }
}

/// Decode any supported element dtype into an owned `VectorData`. Unlike
/// `decode_vector` (zero-copy f32 only), this materializes non-f32 encodings.
/// A malformed or unknown dtype tag is `CorruptData`.
pub fn decode_vector_typed(
    bytes: &[u8],
    params: crate::distance::dtype::QuantParams,
) -> Result<VectorData> {
    use crate::distance::dtype::decode_typed;
    use crate::error::VectorSearchError;
    if bytes.len() < VECTOR_HEADER_BYTES {
        return Err(VectorSearchError::CorruptData(
            "vector blob shorter than header".to_string(),
        ));
    }
    let dtype = crate::types::VectorDataType::from_tag(bytes[1])?;
    let kind = bytes[0];
    let a = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes")) as usize;
    let b = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as usize;
    let body = &bytes[VECTOR_HEADER_BYTES..];
    match kind {
        KIND_DENSE => {
            let values = decode_typed(dtype, a, body, params)?;
            Ok(VectorData::Dense(values))
        }
        KIND_SPARSE => {
            if dtype != crate::types::VectorDataType::F32 {
                return Err(VectorSearchError::CorruptData(
                    "sparse vector has a non-f32 dtype tag".to_string(),
                ));
            }
            decode_vector(bytes)
                .map(|view| view.to_data())
                .ok_or_else(|| VectorSearchError::CorruptData("bad sparse vector".to_string()))
        }
        KIND_MULTI => {
            if dtype != crate::types::VectorDataType::F32 {
                return Err(VectorSearchError::CorruptData(
                    "multi vector has a non-f32 dtype tag".to_string(),
                ));
            }
            let _ = b;
            decode_vector(bytes)
                .map(|view| view.to_data())
                .ok_or_else(|| VectorSearchError::CorruptData("bad multi vector".to_string()))
        }
        _ => Err(VectorSearchError::CorruptData(format!(
            "unknown vector kind tag {kind}"
        ))),
    }
}

fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn as_f32s(bytes: &[u8]) -> Option<&[f32]> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let ptr = bytes.as_ptr();
    debug_assert_eq!(ptr as usize % 4, 0, "blob arrays must be 4-byte aligned");
    Some(unsafe { std::slice::from_raw_parts(ptr as *const f32, bytes.len() / 4) })
}

fn as_u32s(bytes: &[u8]) -> Option<&[u32]> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let ptr = bytes.as_ptr();
    debug_assert_eq!(ptr as usize % 4, 0, "blob arrays must be 4-byte aligned");
    Some(unsafe { std::slice::from_raw_parts(ptr as *const u32, bytes.len() / 4) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MultiVector;

    fn dense() -> VectorData {
        VectorData::Dense(vec![1.0, -2.0, 3.5])
    }

    fn sparse() -> VectorData {
        VectorData::Sparse {
            indices: vec![3, 7, 11],
            values: vec![0.5, 0.25, 0.125],
        }
    }

    fn multi() -> VectorData {
        VectorData::Multi(MultiVector::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]))
    }

    #[test]
    fn vectors_round_trip_through_the_blob_encoding() {
        for data in [dense(), sparse(), multi()] {
            let bytes = encode_vector(&data);
            assert_eq!(bytes.len(), encoded_len(&data));
            let view = decode_vector(&bytes).expect("decode");
            assert_eq!(view.to_data(), data);
            assert_eq!(view.kind(), data.kind());
        }
    }

    #[test]
    fn truncated_encodings_are_rejected() {
        let bytes = encode_vector(&sparse());
        assert!(decode_vector(&bytes[..4]).is_none());
        assert!(decode_vector(&bytes[..bytes.len() - 1]).is_none());
    }

    #[test]
    fn named_entries_round_trip_and_patch() {
        let mut order = std::collections::BTreeMap::new();
        order.insert("a".to_string(), 0u16);
        order.insert("b".to_string(), 1u16);
        let mut vectors = std::collections::BTreeMap::new();
        vectors.insert("a".to_string(), dense());
        vectors.insert("b".to_string(), sparse());
        let entries = vectors
            .iter()
            .map(|(name, data)| (order.get(name).copied().unwrap_or(u16::MAX), encode_vector(data)))
            .collect();
        let bytes = pack_named(entries);
        let view_at = |blob: &[u8], space: u16| {
            named_entries(blob)
                .and_then(|entries| entries.into_iter().find(|(idx, _)| *idx == space).map(|(_, b)| b))
                .and_then(decode_vector)
                .map(|v| v.to_data())
        };
        assert_eq!(view_at(&bytes, 0), Some(dense()));
        assert_eq!(view_at(&bytes, 1), Some(sparse()));
        assert!(view_at(&bytes, 2).is_none());
        assert_eq!(named_entries(&bytes).unwrap().len(), 2);

        // Replace space 1 only: space 0 must survive byte for byte.
        let mut replacements = std::collections::BTreeMap::new();
        replacements.insert(1u16, encode_vector(&multi()));
        let patched = rewrite_named(&bytes, &replacements).unwrap();
        assert_eq!(view_at(&patched, 0), Some(dense()));
        assert_eq!(view_at(&patched, 1), Some(multi()));
        assert_eq!(named_entries(&patched).unwrap().len(), 2);
    }

    #[test]
    fn dtype_tag_is_persisted_and_f32_layout_is_unchanged() {
        use crate::types::VectorDataType;
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let data = VectorData::Dense(values.clone());
        // Default f32 encoding is byte-for-byte the historical layout.
        let legacy = encode_vector(&data);
        let (typed_f32, _) = encode_vector_typed(
            &data,
            VectorDataType::F32,
            crate::distance::dtype::QuantParams::default(),
        )
        .unwrap();
        assert_eq!(legacy, typed_f32);
        assert_eq!(vector_dtype(&legacy), Some(VectorDataType::F32));
        assert_eq!(
            decode_vector_typed(&typed_f32, crate::distance::dtype::QuantParams::default()).unwrap(),
            data
        );
    }

    #[test]
    fn non_f32_dtype_roundtrips_through_blob() {
        use crate::distance::dtype::QuantParams;
        use crate::types::VectorDataType;
        let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let data = VectorData::Dense(values.clone());
        let params = QuantParams {
            scale: 0.05,
            zero_point: 128.0,
        };
        for dtype in [
            VectorDataType::F16,
            VectorDataType::Uint8,
            VectorDataType::Binary,
        ] {
            let (bytes, used_params) = encode_vector_typed(&data, dtype, params).unwrap();
            assert_eq!(vector_dtype(&bytes), Some(dtype));
            // Zero-copy decoder must refuse typed bodies.
            assert!(decode_vector(&bytes).is_none());
            let decoded =
                decode_vector_typed(&bytes, used_params).unwrap();
            let VectorData::Dense(decoded_values) = decoded else {
                panic!("expected dense");
            };
            assert_eq!(decoded_values.len(), values.len());
            // Quantized values stay within one quantization step of source.
            for (a, b) in values.iter().zip(decoded_values.iter()) {
                let tol = match dtype {
                    VectorDataType::F16 => 1e-3,
                    VectorDataType::Uint8 => params.scale,
                    VectorDataType::Binary => 1.1,
                    _ => 0.0,
                };
                assert!((a - b).abs() <= tol, "{dtype:?}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn corrupt_dtype_tag_is_rejected() {
        use crate::distance::dtype::QuantParams;
        let bytes = encode_vector(&VectorData::Dense(vec![1.0, 2.0, 3.0]));
        let mut bad = bytes.clone();
        bad[1] = 99; // unknown dtype tag
        assert_eq!(vector_dtype(&bad), None);
        assert!(decode_vector_typed(&bad, QuantParams::default()).is_err());
        assert!(decode_vector(&bad).is_none());
    }
}
