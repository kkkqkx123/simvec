//! `spaces/<space>/segments/<start>/` — segmented append-only blob files for
//! exact-scan (kind/named) spaces.
//!
//! The public handle stays a global [`BlobRef`] `{off, len}`, so snapshots,
//! WAL records and every read call site keep working unchanged. Internally the
//! logical byte stream is partitioned into physical segment files of at most
//! `segment_bytes` each. A blob never straddles two segments: when the active
//! segment cannot fit the next record, a new segment directory is opened and
//! the record starts at its local offset 0. A global `off` maps to
//! `(segment.start, off - segment.start)`.
//!
//! Layout:
//! ```text
//!   segments/<start:010>/<file_name>   one append-only file per segment
//! ```
//!
//! Durability mirrors `PointBlobs`: appends are positional `pwrite`s, `sync()`
//! fsyncs every dirty segment, and `open(durable_len)` truncates the committed
//! tail segment back to the snapshot's length; segment directories entirely
//! beyond that length are crash tails and are removed.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use parking_lot::{RwLock, RwLockReadGuard};

use crate::error::{Result, VectorSearchError};
use crate::types::{VectorData, VectorView};

use super::point_blobs::{decode_vector, BlobRef};

/// Default logical capacity of one segment (64 MiB), matching the WAL segment
/// size so a segment is a meaningful checkpoint/compaction unit.
pub(crate) const DEFAULT_SEGMENT_BYTES: u64 = 64 * 1024 * 1024;

/// How much a segment file grows by when an append does not fit the mapping.
const GROW_CHUNK_BYTES: u64 = 4 * 1024 * 1024;

struct Segment {
    /// Global byte offset at which this segment starts.
    start: u64,
    file: std::fs::File,
    map: Option<Arc<Mmap>>,
    /// Logical end of appended data inside this segment.
    len: u64,
}

struct SegState {
    segments: Vec<Segment>,
    /// Global logical length = last.start + last.len.
    len: u64,
}

/// Segmented append-only blob store over a space's segment directories.
pub struct SegmentedBlobs {
    dir: PathBuf,
    file_name: &'static str,
    segment_bytes: u64,
    state: RwLock<SegState>,
    dirty: AtomicBool,
}

impl SegmentedBlobs {
    /// Create an empty segmented store. The first segment (start 0) is created.
    pub(crate) fn create(dir: &Path, file_name: &'static str, segment_bytes: u64) -> Result<Self> {
        if segment_bytes == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "segment_bytes must be >= 1".to_string(),
            ));
        }
        std::fs::create_dir_all(dir)?;
        let seg_dir = dir.join(format!("{:010}", 0u64));
        std::fs::create_dir_all(&seg_dir)?;
        let file = open_blob_file(&seg_dir.join(file_name), true)?;
        file.sync_all()?;
        open_dir(&seg_dir.join(file_name))?.sync_all()?;
        Ok(Self {
            dir: dir.to_path_buf(),
            file_name,
            segment_bytes,
            state: RwLock::new(SegState {
                segments: vec![Segment {
                    start: 0,
                    file,
                    map: None,
                    len: 0,
                }],
                len: 0,
            }),
            dirty: AtomicBool::new(false),
        })
    }

    /// Open a segmented store, truncating the committed tail segment back to
    /// `durable_len`. Segments entirely beyond it are crash tails and removed.
    pub(crate) fn open(
        dir: &Path,
        file_name: &'static str,
        segment_bytes: u64,
        durable_len: u64,
    ) -> Result<Self> {
        if segment_bytes == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "segment_bytes must be >= 1".to_string(),
            ));
        }
        let mut starts = Vec::new();
        if dir.exists() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                if !entry.file_type()?.is_dir() {
                    continue;
                }
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.len() != 10 || !name.bytes().all(|b| b.is_ascii_digit()) {
                    continue;
                }
                let start: u64 = name.parse().map_err(|_| {
                    VectorSearchError::CorruptData(format!("invalid segment directory {name}"))
                })?;
                starts.push(start);
            }
        }
        starts.sort_unstable();
        if starts.is_empty() {
            starts.push(0);
        }
        if starts[0] != 0 {
            return Err(VectorSearchError::CorruptData(
                "segmented blobs are missing the start-0 segment".to_string(),
            ));
        }
        for (i, &start) in starts.iter().enumerate() {
            if start != i as u64 * segment_bytes {
                return Err(VectorSearchError::CorruptData(format!(
                    "segment start {start} breaks the contiguous grid (expected {})",
                    i as u64 * segment_bytes
                )));
            }
        }

        let mut segments = Vec::with_capacity(starts.len());
        for &start in &starts {
            // Segments entirely beyond the committed length are crash tails.
            // Truncate just this store's file (never remove the whole segment
            // directory — other blob stores share the same segment root) and
            // drop the segment from the mapping.
            if start >= durable_len {
                let path = dir.join(format!("{start:010}")).join(file_name);
                if path.exists() {
                    let f = open_blob_file(&path, false)?;
                    f.set_len(0)?;
                    f.sync_all()?;
                }
                continue;
            }
            let seg_dir = dir.join(format!("{start:010}"));
            let path = seg_dir.join(file_name);
            let is_tail = start + segment_bytes > durable_len;
            let expected = if is_tail {
                Some(durable_len - start)
            } else {
                None
            };
            let file = open_blob_file(&path, expected.is_none() || expected == Some(0))?;
            let actual = file.metadata()?.len();
            if let Some(want) = expected {
                if actual > want {
                    file.set_len(want)?;
                    file.sync_all()?;
                } else if actual < want {
                    return Err(VectorSearchError::CorruptData(format!(
                        "segment {start} holds {actual} bytes but the snapshot references {want}"
                    )));
                }
            }
            let len = expected.unwrap_or(actual);
            let map = map_all(&file, actual)?;
            segments.push(Segment {
                start,
                file,
                map,
                len,
            });
        }
        if segments.is_empty() {
            let seg_dir = dir.join("0000000000");
            std::fs::create_dir_all(&seg_dir)?;
            let file = open_blob_file(&seg_dir.join(file_name), true)?;
            file.set_len(0)?;
            segments.push(Segment {
                start: 0,
                file,
                map: None,
                len: 0,
            });
        }
        let len = segments
            .last()
            .map(|seg| seg.start + seg.len)
            .unwrap_or(0);
        Ok(Self {
            dir: dir.to_path_buf(),
            file_name,
            segment_bytes,
            state: RwLock::new(SegState { segments, len }),
            dirty: AtomicBool::new(false),
        })
    }

    /// Append `bytes`, returning a global `BlobRef`. The record is placed
    /// wholly inside one segment.
    pub(crate) fn append(&self, bytes: &[u8]) -> Result<BlobRef> {
        if bytes.is_empty() {
            return Err(VectorSearchError::Internal(
                "refusing to append an empty point blob".to_string(),
            ));
        }
        if bytes.len() as u64 > self.segment_bytes {
            return Err(VectorSearchError::InvalidConfig(format!(
                "blob of {} bytes exceeds segment_bytes {}",
                bytes.len(),
                self.segment_bytes
            )));
        }
        let mut state = self.state.write();
        // Ensure the active (last) segment has room; otherwise open a new one.
        let need_new = {
            let last = state.segments.last().expect("at least one segment");
            last.len + bytes.len() as u64 > self.segment_bytes
        };
        if need_new {
            let start = state.len.div_ceil(self.segment_bytes) * self.segment_bytes;
            let seg_dir = self.dir.join(format!("{start:010}"));
            std::fs::create_dir_all(&seg_dir)?;
            let file = open_blob_file(&seg_dir.join(self.file_name), true)?;
            state.segments.push(Segment {
                start,
                file,
                map: None,
                len: 0,
            });
        }
        let seg = state.segments.last_mut().expect("at least one segment");
        // 4-byte align every blob start so decoded f32/u32 slices need no copy.
        let pad = (4 - seg.len % 4) % 4;
        if pad > 0 {
            write_at(&seg.file, &[0u8; 4][..pad as usize], seg.len)?;
        }
        let local_off = seg.len + pad;
        let end = local_off + bytes.len() as u64;
        let mapped = seg.map.as_ref().map_or(0, |m| m.len() as u64);
        if end > mapped {
            let new_len = end.div_ceil(GROW_CHUNK_BYTES) * GROW_CHUNK_BYTES;
            seg.file.set_len(new_len)?;
            seg.map = map_all(&seg.file, new_len)?;
        }
        write_at(&seg.file, bytes, local_off)?;
        seg.len = end;
        let global_off = seg.start + local_off;
        state.len = seg.start + seg.len;
        drop(state);
        self.dirty.store(true, Ordering::Relaxed);
        Ok(BlobRef {
            off: global_off,
            len: bytes.len() as u32,
        })
    }

    /// Borrow a read view across all segments.
    pub(crate) fn reader(&self) -> SegmentedBlobsReader<'_> {
        SegmentedBlobsReader {
            guard: self.state.read(),
        }
    }

    pub(crate) fn len(&self) -> u64 {
        self.state.read().len
    }

    /// fsync every dirty segment. Failure keeps the store dirty for retry.
    pub(crate) fn sync(&self) -> Result<bool> {
        if !self.dirty.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let mut state = self.state.write();
        for seg in state.segments.iter_mut() {
            if seg.file.sync_all().is_err() {
                return Err(VectorSearchError::Internal(
                    "segmented blob fsync failed".to_string(),
                ));
            }
            // Trim any preallocated slack back to the logical length so a
            // snapshot's length matches the file exactly after checkpoint.
            seg.file.set_len(seg.len)?;
            if seg.len == 0 {
                seg.map = None;
            } else {
                seg.map = map_all(&seg.file, seg.len)?;
            }
        }
        drop(state);
        self.dirty.store(false, Ordering::Relaxed);
        Ok(true)
    }

    pub(crate) fn mapped_bytes(&self) -> u64 {
        self.state.read().len
    }
}

/// Read handle over all segments of a [`SegmentedBlobs`].
pub(crate) struct SegmentedBlobsReader<'a> {
    guard: RwLockReadGuard<'a, SegState>,
}

impl SegmentedBlobsReader<'_> {
    pub(crate) fn bytes(&self, blob: BlobRef) -> Option<&[u8]> {
        for seg in &self.guard.segments {
            if blob.off >= seg.start && blob.off + blob.len as u64 <= seg.start + seg.len {
                let local = blob.off - seg.start;
                let map = seg.map.as_ref()?;
                let end = local as usize + blob.len as usize;
                if end > map.len() {
                    return None;
                }
                return Some(&map[local as usize..end]);
            }
        }
        None
    }

    pub(crate) fn view(&self, blob: BlobRef) -> Option<VectorView<'_>> {
        decode_vector(self.bytes(blob)?)
    }

    pub(crate) fn to_data(&self, blob: BlobRef) -> Option<VectorData> {
        self.view(blob).map(|view| view.to_data())
    }

}

fn open_blob_file(path: &Path, create: bool) -> Result<std::fs::File> {
    use std::fs::OpenOptions;
    let mut opts = OpenOptions::new();
    opts.read(true).write(true);
    if create {
        opts.create(true).truncate(false);
    }
    Ok(opts.open(path)?)
}

fn open_dir(path: &Path) -> Result<std::fs::File> {
    let dir = path.parent().ok_or_else(|| {
        VectorSearchError::Internal(format!("no parent dir for {}", path.display()))
    })?;
    Ok(std::fs::File::open(dir)?)
}

fn map_all(file: &std::fs::File, len: u64) -> Result<Option<Arc<Mmap>>> {
    if len == 0 {
        return Ok(None);
    }
    let map = unsafe { MmapOptions::new().len(len as usize).map(file)? };
    Ok(Some(Arc::new(map)))
}

#[cfg(unix)]
fn write_at(file: &std::fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(not(unix))]
fn write_at(file: &std::fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut handle = file;
    handle.seek(SeekFrom::Start(offset))?;
    handle.write_all(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::point_blobs::encode_vector;

    fn seg_bytes() -> u64 {
        256
    }

    #[test]
    fn roundtrip_across_segments() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("blobs");
        let store = SegmentedBlobs::create(&root, "points.bin", seg_bytes()).unwrap();
        let mut refs = Vec::new();
        for i in 0..40u32 {
            let data = VectorData::Dense(vec![i as f32, 1.0, 2.0, 3.0]);
            let bytes = encode_vector(&data);
            refs.push(store.append(&bytes).unwrap());
        }
        store.sync().unwrap();
        let reader = store.reader();
        for (i, blob) in refs.iter().enumerate() {
            match reader.view(*blob).unwrap() {
                VectorView::Dense(values) => assert_eq!(values[0], i as f32),
                _ => panic!("expected dense view"),
            }
        }
        assert!(root.join("0000000000").join("points.bin").exists());
        assert!(root.join("0000000256").join("points.bin").exists());
    }

    #[test]
    fn reopen_preserves_all_blobs_and_truncates_tail() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("blobs");
        let durable;
        let refs: Vec<BlobRef>;
        {
            let store = SegmentedBlobs::create(&root, "points.bin", seg_bytes()).unwrap();
            refs = (0..20u32)
                .map(|i| {
                    let data = VectorData::Dense(vec![i as f32; 4]);
                    store.append(&encode_vector(&data)).unwrap()
                })
                .collect();
            store.sync().unwrap();
            durable = store.len();
        }
        let reopened = SegmentedBlobs::open(&root, "points.bin", seg_bytes(), durable).unwrap();
        assert_eq!(reopened.len(), durable);
        let reader = reopened.reader();
        for (i, blob) in refs.iter().enumerate() {
            match reader.view(*blob).unwrap() {
                VectorView::Dense(values) => assert_eq!(values[0], i as f32),
                _ => panic!("expected dense view"),
            }
        }
    }
}
