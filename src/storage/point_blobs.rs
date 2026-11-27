//! `points.bin` — append-only blob file for exact-scan point payloads.
//!
//! Dense collections address their vectors by slot into a fixed-stride
//! `vectors.bin`. Sparse and multi points are variable length, so they are
//! appended here instead and addressed by `(offset, len)` — the pair the
//! snapshot stores per slot. Nothing is overwritten in place: an updated point
//! gets a new blob and the old bytes become unreachable until compaction
//! rewrites the file.
//!
//! The file is mapped read-only once and remapped when it has to grow, in
//! [`GROW_CHUNK_BYTES`] steps so a burst of small upserts costs one `mmap`
//! per chunk rather than one per point. Readers hold the map read lock and get
//! `&[u8]`/`VectorView` borrows straight out of the mapping; writers hold the
//! write lock to append. Blob starts are 4-byte aligned so a decoded view can
//! borrow `&[f32]`/`&[u32]` without copying.
//!
//! Durability: appends are plain `pwrite`s, `sync()` is the checkpoint's
//! fsync. The file may therefore be longer than the snapshot that describes it
//! (crash between append and checkpoint), which [`PointBlobs::open`] resolves
//! by truncating back to the snapshot's length: every reachable blob is one
//! the snapshot pointed at, so the tail is by definition uncommitted.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use parking_lot::{RwLock, RwLockReadGuard};

use crate::error::{Result, VectorSearchError};
use crate::types::{VectorData, VectorView};

/// How much the file grows by when an append does not fit in the mapping.
const GROW_CHUNK_BYTES: u64 = 4 * 1024 * 1024;

/// Handle to one point's bytes inside [`PointBlobs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BlobRef {
    pub off: u64,
    pub len: u32,
}

impl BlobRef {
    pub fn end(&self) -> u64 {
        self.off + self.len as u64
    }
}

struct BlobState {
    file: File,
    map: Option<Arc<Mmap>>,
    /// Logical end of the appended data (the file may be longer: it is
    /// pre-allocated in whole chunks).
    len: u64,
}

/// Append-only, memory-mapped blob file.
pub struct PointBlobs {
    path: PathBuf,
    state: RwLock<BlobState>,
    dirty: AtomicBool,
}

impl PointBlobs {
    /// Create an empty blob file. Fails if it already exists.
    pub fn create(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        file.sync_all()?;
        open_dir(path)?.sync_all()?;
        Ok(Self {
            path: path.to_path_buf(),
            state: RwLock::new(BlobState {
                file,
                map: None,
                len: 0,
            }),
            dirty: AtomicBool::new(false),
        })
    }

    /// Open an existing file, reconciling its length with the snapshot.
    ///
    /// `durable_len` is the length the snapshot recorded. A longer file is a
    /// crash tail (appends that never reached a checkpoint) and is cut back; a
    /// shorter one means the snapshot points at bytes that are gone, which is
    /// real corruption.
    pub fn open(path: &Path, durable_len: Option<u64>) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let actual = file.metadata()?.len();
        if let Some(expected) = durable_len {
            if actual > expected {
                tracing::warn!(
                    actual,
                    expected,
                    "points.bin longer than the snapshot; dropping the uncommitted tail"
                );
                file.set_len(expected)?;
                file.sync_all()?;
            } else if actual < expected {
                return Err(VectorSearchError::CorruptData(format!(
                    "points.bin holds {actual} bytes but the snapshot references {expected}"
                )));
            }
        }
        let len = file.metadata()?.len();
        let map = map_all(&file, len)?;
        Ok(Self {
            path: path.to_path_buf(),
            state: RwLock::new(BlobState { file, map, len }),
            dirty: AtomicBool::new(false),
        })
    }

    /// Append `bytes` and return the handle addressing them. `bytes` must not
    /// be empty (a zero-length blob would be indistinguishable from a tombstone
    /// reference).
    pub fn append(&self, bytes: &[u8]) -> Result<BlobRef> {
        if bytes.is_empty() {
            return Err(VectorSearchError::Internal(
                "refusing to append an empty point blob".to_string(),
            ));
        }
        let mut state = self.state.write();
        // Keep every blob start 4-byte aligned so decoded f32/u32 slices need
        // no copy.
        let pad = (4 - state.len % 4) % 4;
        if pad > 0 {
            write_at(&state.file, &[0u8; 4][..pad as usize], state.len)?;
        }
        let off = state.len + pad;
        let end = off + bytes.len() as u64;
        let mapped = state.map.as_ref().map_or(0, |map| map.len() as u64);
        if end > mapped {
            let new_len = end.div_ceil(GROW_CHUNK_BYTES) * GROW_CHUNK_BYTES;
            state.file.set_len(new_len)?;
            state.map = map_all(&state.file, new_len)?;
        }
        write_at(&state.file, bytes, off)?;
        state.len = end;
        drop(state);
        self.dirty.store(true, Ordering::Relaxed);
        Ok(BlobRef {
            off,
            len: bytes.len() as u32,
        })
    }

    /// Borrow the mapping for a read pass. One reader per operation: holding
    /// it keeps the map (and therefore every returned slice) stable.
    pub fn reader(&self) -> PointBlobsReader<'_> {
        PointBlobsReader {
            guard: self.state.read(),
        }
    }

    /// Logical end of the appended data.
    pub fn len(&self) -> u64 {
        self.state.read().len
    }

    /// Flush appends since the last `sync`. Returns whether anything was
    /// flushed.
    pub fn sync(&self) -> bool {
        if !self.dirty.swap(false, Ordering::Relaxed) {
            return false;
        }
        let state = self.state.read();
        state.file.sync_all().ok();
        true
    }

    /// Atomically replace the backing file with `tmp_path` (compaction) and
    /// remap. `len` is the staged logical length: the staging file is
    /// pre-allocated in whole chunks, so the tail past `len` is bloat the
    /// replacement has to drop. Readers keep their old mapping until the swap.
    pub fn replace_from(&self, tmp_path: &Path, len: u64) -> Result<()> {
        std::fs::rename(tmp_path, &self.path)?;
        let file = OpenOptions::new().read(true).write(true).open(&self.path)?;
        if file.metadata()?.len() > len {
            file.set_len(len)?;
        }
        file.sync_all()?;
        open_dir(&self.path)?.sync_all()?;
        let map = map_all(&file, len)?;
        let mut state = self.state.write();
        state.file = file;
        state.map = map;
        state.len = len;
        self.dirty.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Bytes currently mapped. Used by the resident-memory estimate.
    pub fn mapped_bytes(&self) -> u64 {
        self.state.read().len
    }

}

impl std::fmt::Debug for PointBlobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointBlobs")
            .field("path", &self.path)
            .field("len", &self.len())
            .finish()
    }
}

/// Read handle over a mapped blob file.
pub struct PointBlobsReader<'a> {
    guard: RwLockReadGuard<'a, BlobState>,
}

impl PointBlobsReader<'_> {
    /// Raw bytes of one blob. `None` when the reference is out of the map,
    /// which the open-time reconciliation is supposed to make impossible.
    pub fn bytes(&self, blob: BlobRef) -> Option<&[u8]> {
        let map = self.guard.map.as_ref()?;
        let end = blob.end() as usize;
        if end > map.len() {
            return None;
        }
        Some(&map[blob.off as usize..end])
    }

    /// Decode the vector of one blob without copying.
    pub fn view(&self, blob: BlobRef) -> Option<VectorView<'_>> {
        decode_vector(self.bytes(blob)?)
    }

    /// Decode the vector of one blob into an owned payload (fetch paths).
    pub fn to_data(&self, blob: BlobRef) -> Option<VectorData> {
        self.view(blob).map(|view| view.to_data())
    }

    /// Decode one named space of a multi-space blob, without copying.
    pub fn space_view(&self, blob: BlobRef, space: u16) -> Option<VectorView<'_>> {
        named_view(self.bytes(blob)?, space)
    }
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
            push_header(
                &mut out,
                KIND_MULTI,
                multi.dim as u32,
                multi.count() as u32,
            );
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

/// Encode a whole named-space map. `order` maps a space name to its stable
/// index (the position in the collection's space map), so a decode can address
/// a space without storing its name per point.
pub fn encode_named(
    vectors: &std::collections::BTreeMap<String, VectorData>,
    order: &std::collections::BTreeMap<String, u16>,
) -> Vec<u8> {
    let mut entries: Vec<(u16, Vec<u8>)> = Vec::with_capacity(vectors.len());
    for (space, data) in vectors {
        let index = order.get(space).copied().unwrap_or(u16::MAX);
        entries.push((index, encode_vector(data)));
    }
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

/// Borrow one space's vector out of an encoded point blob.
pub fn named_view(bytes: &[u8], space: u16) -> Option<VectorView<'_>> {
    let (_, entry) = named_entries(bytes)?
        .into_iter()
        .find(|(index, _)| *index == space)?;
    decode_vector(entry)
}

fn push_header(out: &mut Vec<u8>, kind: u8, a: u32, b: u32) {
    out.push(kind);
    out.extend_from_slice(&[0u8, 0u8, 0u8]);
    out.extend_from_slice(&a.to_le_bytes());
    out.extend_from_slice(&b.to_le_bytes());
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

fn map_all(file: &File, len: u64) -> Result<Option<Arc<Mmap>>> {
    if len == 0 {
        return Ok(None);
    }
    let map = unsafe { MmapOptions::new().len(len as usize).map(file)? };
    Ok(Some(Arc::new(map)))
}

fn open_dir(path: &Path) -> Result<File> {
    let dir = path.parent().ok_or_else(|| {
        VectorSearchError::Internal(format!("no parent dir for {}", path.display()))
    })?;
    Ok(File::open(dir)?)
}

#[cfg(unix)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(not(unix))]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut handle = file;
    handle.seek(SeekFrom::Start(offset))?;
    handle.write_all(buf)
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
        VectorData::Multi(MultiVector::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]))
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
    fn appends_are_addressable_and_aligned() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("points.bin");
        let blobs = PointBlobs::create(&path).unwrap();
        let dense_ref = blobs.append(&encode_vector(&dense())).unwrap();
        // A 1-byte blob forces the next append to be realigned.
        let odd = blobs.append(&[0xAB]).unwrap();
        let sparse_ref = blobs.append(&encode_vector(&sparse())).unwrap();
        assert_eq!(odd.len, 1);
        assert_eq!(sparse_ref.off % 4, 0, "blob starts stay 4-byte aligned");

        let reader = blobs.reader();
        assert_eq!(reader.view(dense_ref).unwrap().to_data(), dense());
        assert_eq!(reader.view(sparse_ref).unwrap().to_data(), sparse());
        assert_eq!(reader.bytes(odd).unwrap(), &[0xAB]);
    }

    #[test]
    fn a_long_file_is_truncated_to_the_snapshot_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("points.bin");
        let blobs = PointBlobs::create(&path).unwrap();
        let kept = blobs.append(&encode_vector(&dense())).unwrap();
        let durable = blobs.len();
        let dropped = blobs.append(&encode_vector(&sparse())).unwrap();
        blobs.sync();
        drop(blobs);

        // Reopen as if the snapshot only covered the first append.
        let reopened = PointBlobs::open(&path, Some(durable)).unwrap();
        assert_eq!(reopened.len(), durable);
        let reader = reopened.reader();
        assert!(reader.view(kept).is_some());
        assert!(
            reader.bytes(dropped).is_none(),
            "the uncommitted tail must be unreachable"
        );
    }

    #[test]
    fn a_short_file_is_reported_as_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("points.bin");
        let blobs = PointBlobs::create(&path).unwrap();
        blobs.append(&encode_vector(&dense())).unwrap();
        blobs.sync();
        let len = blobs.len();
        drop(blobs);
        // Cut the file below what the snapshot references.
        std::fs::write(&path, vec![0u8; len as usize - 4]).unwrap();
        let error = PointBlobs::open(&path, Some(len)).unwrap_err();
        assert!(matches!(error, VectorSearchError::CorruptData(_)));
    }

    #[test]
    fn growth_is_transparent_across_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("points.bin");
        let blobs = PointBlobs::create(&path).unwrap();
        // Pushing past GROW_CHUNK_BYTES forces at least one remap; every
        // earlier reference must still resolve.
        let big = VectorData::Dense(vec![7.0; 2 * 1024 * 1024]);
        let first = blobs.append(&encode_vector(&dense())).unwrap();
        let large = blobs.append(&encode_vector(&big)).unwrap();
        assert!(large.end() > GROW_CHUNK_BYTES);
        let reader = blobs.reader();
        assert_eq!(reader.view(first).unwrap().to_data(), dense());
        assert_eq!(reader.view(large).unwrap().to_data(), big);
    }

    #[test]
    fn named_entries_round_trip_and_patch() {
        let mut order = std::collections::BTreeMap::new();
        order.insert("a".to_string(), 0u16);
        order.insert("b".to_string(), 1u16);
        let mut vectors = std::collections::BTreeMap::new();
        vectors.insert("a".to_string(), dense());
        vectors.insert("b".to_string(), sparse());
        let bytes = encode_named(&vectors, &order);
        assert_eq!(named_view(&bytes, 0).unwrap().to_data(), dense());
        assert_eq!(named_view(&bytes, 1).unwrap().to_data(), sparse());
        assert!(named_view(&bytes, 2).is_none());
        assert_eq!(named_entries(&bytes).unwrap().len(), 2);

        // Replace space 1 only: space 0 must survive byte for byte.
        let mut replacements = std::collections::BTreeMap::new();
        replacements.insert(1u16, encode_vector(&multi()));
        let patched = rewrite_named(&bytes, &replacements).unwrap();
        assert_eq!(named_view(&patched, 0).unwrap().to_data(), dense());
        assert_eq!(named_view(&patched, 1).unwrap().to_data(), multi());
        assert_eq!(named_entries(&patched).unwrap().len(), 2);
    }
}
