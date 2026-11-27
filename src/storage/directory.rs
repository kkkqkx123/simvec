//! Directory file: `keys.bin` (slot -> point id).
//!
//! One layout — fixed header, fixed-size record array, and an append-only blob
//! area:
//!
//! ```text
//! [0..4)   magic           "VKEY"
//! [4..8)   version         u32 = 2
//! [8..12)  rec_size        u32 (bytes per record)
//! [12..16) reserved        u32 = 0
//! [16..24) rec_capacity    u64
//! [24..32) blob_len        u64
//! [32..)   rec array       rec_size * rec_capacity
//! [32 + rec_capacity*rec_size ..)  blob area (length blob_len)
//! ```
//!
//! `rec_size` is a header field rather than a caller-supplied constant: the
//! record layout is what the file itself declares, so a reader never has to
//! agree with a compile-time constant, and the only constraint left is that a
//! record is large enough for the `{off: u32, len: u32}` prefix every record
//! starts with.
//!
//! Blob offsets in records are relative to the start of the blob area. Growing
//! the record array relocates the blob area, so it is implemented as an atomic
//! temp-file rebuild. Writers go through the file handle; readers use a
//! read-only mmap snapshot published via `ArcSwap`.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;
use memmap2::{Mmap, MmapOptions};

use crate::error::{Result, VectorSearchError};

const HEADER_LEN: usize = 32;
const FILE_VERSION: u32 = 2;

/// Byte offsets of the header fields.
const OFF_VERSION: u64 = 4;
const OFF_REC_SIZE: u64 = 8;
const OFF_REC_CAPACITY: u64 = 16;
const OFF_BLOB_LEN: u64 = 24;

/// Smallest record a directory file can declare: the `{off, len}` prefix.
const MIN_REC_SIZE: usize = 8;

/// Keys record: `{ off: u32, len: u32 }`.
pub(crate) const KEY_REC_SIZE: usize = 8;

/// A read-only snapshot of a directory file.
pub struct DirView {
    mmap: Arc<Mmap>,
    /// Record stride declared by the file header.
    rec_size: usize,
    rec_capacity: usize,
    blob_len: u64,
    /// Bytes currently mapped. The physical file may carry headroom beyond
    /// the logical end (`HEADER_LEN + rec_capacity * rec_size + blob_len`);
    /// `open` tolerates that slack so appends only remap on geometric growth
    /// steps instead of once per append.
    mapped_len: usize,
}

impl DirView {
    fn blob_offset(&self) -> usize {
        HEADER_LEN + self.rec_capacity * self.rec_size
    }

    /// Logical end of the file: header + record array + blob area.
    fn logical_len(&self) -> usize {
        self.blob_offset() + self.blob_len as usize
    }

    /// Bytes per record, as declared by the file. Test-only observation hook:
    /// production readers only ever go through [`DirView::blob`].
    #[cfg(test)]
    pub fn rec_size(&self) -> usize {
        self.rec_size
    }

    /// Parse `{ off: u32, len: u32 }` of a record.
    fn rec_off_len(&self, slot: usize) -> Option<(u32, u32)> {
        if slot >= self.rec_capacity {
            return None;
        }
        let start = HEADER_LEN + slot * self.rec_size;
        let bytes = self.mmap.get(start..start + 8)?;
        let off = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let len = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        Some((off, len))
    }

    /// Blob for a slot. Returns `None` for empty records.
    pub fn blob(&self, slot: usize) -> Option<&[u8]> {
        let (off, len) = self.rec_off_len(slot)?;
        if len == 0 {
            return None;
        }
        let start = self.blob_offset() + off as usize;
        self.mmap.get(start..start + len as usize)
    }
}

/// Append-only directory file with a fixed-size record array and blob area.
pub struct BlobDirectory {
    path: PathBuf,
    rec_size: usize,
    file: parking_lot::Mutex<File>,
    view: ArcSwap<DirView>,
    /// Set on any write; cleared and fsynced by `sync` at checkpoint time.
    dirty: std::sync::atomic::AtomicBool,
}

/// Minimum geometric headroom added when the mapping has to grow.
const MAPPED_HEADROOM_MIN: usize = 64 * 1024;

impl BlobDirectory {
    /// Create a new file with `initial_capacity` records.
    pub fn create(
        path: &Path,
        magic: [u8; 4],
        rec_size: usize,
        initial_capacity: u64,
    ) -> Result<Self> {
        if rec_size < MIN_REC_SIZE {
            return Err(VectorSearchError::InvalidConfig(format!(
                "directory record size {rec_size} < minimum {MIN_REC_SIZE}"
            )));
        }
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        let file_len = HEADER_LEN + initial_capacity as usize * rec_size;
        file.set_len(file_len as u64)?;
        write_header(&file, magic, rec_size, initial_capacity, 0)?;
        file.sync_all()?;

        let mmap = map_file(&file, file_len)?;
        let view = Arc::new(DirView {
            mmap: Arc::new(mmap),
            rec_size,
            rec_capacity: initial_capacity as usize,
            blob_len: 0,
            mapped_len: file_len,
        });
        Ok(Self {
            path: path.to_path_buf(),
            rec_size,
            file: parking_lot::Mutex::new(file),
            view: ArcSwap::from(view),
            dirty: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Open an existing file. The record stride comes from the header.
    pub fn open(path: &Path, magic: [u8; 4]) -> Result<Self> {
        let file = File::options().read(true).write(true).open(path)?;
        let file_len = file.metadata()?.len() as usize;
        if file_len < HEADER_LEN {
            return Err(VectorSearchError::CorruptData(format!(
                "{} too short: {} bytes",
                path.display(),
                file_len
            )));
        }
        let actual_magic = read_magic(&file)?;
        if actual_magic != magic {
            return Err(VectorSearchError::CorruptData(format!(
                "{} bad magic",
                path.display()
            )));
        }
        let version = read_version(&file)?;
        if version != FILE_VERSION {
            return Err(VectorSearchError::CorruptData(format!(
                "{} unsupported version {}",
                path.display(),
                version
            )));
        }
        let rec_size = read_rec_size(&file)?;
        let rec_capacity = read_rec_capacity(&file)? as usize;
        let blob_len = read_blob_len(&file)?;
        let view = Arc::new(DirView {
            mmap: Arc::new(map_file(&file, file_len)?),
            rec_size,
            rec_capacity,
            blob_len,
            mapped_len: file_len,
        });
        // The physical file may be longer than the logical end: appends map
        // with headroom and never truncate back, so extra bytes are unused
        // reserved space rather than corruption.
        let expected = view.logical_len();
        if file_len < expected {
            return Err(VectorSearchError::CorruptData(format!(
                "{} length {} < expected {expected}",
                path.display(),
                file_len
            )));
        }
        Ok(Self {
            path: path.to_path_buf(),
            rec_size,
            file: parking_lot::Mutex::new(file),
            view: ArcSwap::from(view),
            dirty: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Flush writes since the last `sync` to stable storage. Returns whether
    /// anything was flushed. Called at checkpoint time only; the WAL carries
    /// the same data for recovery, so per-append fsync is not needed.
    pub fn sync(&self) -> Result<bool> {
        if !self.dirty.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(false);
        }
        let file = self.file.lock();
        file.sync_all()?;
        self.dirty
            .store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(true)
    }

    pub fn snapshot(&self) -> arc_swap::Guard<Arc<DirView>> {
        self.view.load()
    }

    /// Append a blob and wire up the record for `slot`.
    ///
    /// `slot` must be strictly less than the current record capacity.
    ///
    /// Bytes are written through the file handle without fsync; the mapping is
    /// remapped only when the blob area outgrows the mapped range (geometric
    /// headroom), so an append is O(1) amortized. Durability comes from the
    /// WAL plus the batched `sync` at checkpoint time.
    pub fn append_blob(&self, slot: usize, blob: &[u8], flags: u8) -> Result<()> {
        let view = self.view.load();
        if slot >= view.rec_capacity {
            return Err(VectorSearchError::Internal(format!(
                "slot {slot} out of record capacity {}",
                view.rec_capacity
            )));
        }
        let blob_offset = view.blob_offset();
        let new_blob_len = view.blob_len + blob.len() as u64;
        let end = blob_offset + new_blob_len as usize;

        if end > view.mapped_len {
            // Grow the physical file with headroom and remap once; readers
            // keep their previous snapshot until this store lands.
            let headroom = (view.mapped_len / 4).max(MAPPED_HEADROOM_MIN);
            let new_mapped = end.saturating_add(headroom).max(end);
            {
                let file = self.file.lock();
                file.set_len(new_mapped as u64)?;
            }
            let mmap = map_file(&self.file.lock(), new_mapped)?;
            self.view.store(Arc::new(DirView {
                mmap: Arc::new(mmap),
                rec_size: view.rec_size,
                rec_capacity: view.rec_capacity,
                blob_len: view.blob_len,
                mapped_len: new_mapped,
            }));
        }

        let view = self.view.load();
        // Blob bytes first, then the record, then the header water mark.
        let file = self.file.lock();
        write_at(&file, blob, (blob_offset + view.blob_len as usize) as u64)?;
        let rec_bytes = encode_rec(
            view.blob_len as u32,
            blob.len() as u32,
            flags,
            self.rec_size,
        );
        let rec_start = HEADER_LEN + slot * self.rec_size;
        write_at(&file, &rec_bytes, rec_start as u64)?;
        write_at(&file, &new_blob_len.to_le_bytes(), OFF_BLOB_LEN)?;
        drop(file);

        // Publish the new water mark reusing the same mapping.
        let next = DirView {
            mmap: Arc::clone(&view.mmap),
            rec_size: view.rec_size,
            rec_capacity: view.rec_capacity,
            blob_len: new_blob_len,
            mapped_len: view.mapped_len,
        };
        self.view.store(Arc::new(next));
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

impl std::fmt::Debug for BlobDirectory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let view = self.view.load();
        f.debug_struct("BlobDirectory")
            .field("path", &self.path)
            .field("rec_size", &self.rec_size)
            .field("rec_capacity", &view.rec_capacity)
            .field("blob_len", &view.blob_len)
            .finish()
    }
}

fn encode_rec(off: u32, len: u32, flags: u8, rec_size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rec_size);
    out.extend_from_slice(&off.to_le_bytes());
    out.extend_from_slice(&len.to_le_bytes());
    if rec_size >= 12 {
        out.push(flags);
        out.extend_from_slice(&[0u8; 3]);
    }
    out.truncate(rec_size);
    out
}

fn write_header(
    file: &File,
    magic: [u8; 4],
    rec_size: usize,
    rec_capacity: u64,
    blob_len: u64,
) -> Result<()> {
    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(&magic);
    header.extend_from_slice(&FILE_VERSION.to_le_bytes());
    header.extend_from_slice(&(rec_size as u32).to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes());
    header.extend_from_slice(&rec_capacity.to_le_bytes());
    header.extend_from_slice(&blob_len.to_le_bytes());
    write_at(file, &header, 0)?;
    Ok(())
}

fn read_magic(file: &File) -> Result<[u8; 4]> {
    let mut buf = [0u8; 4];
    read_exact_at(file, &mut buf, 0)?;
    Ok(buf)
}

fn read_version(file: &File) -> Result<u32> {
    let mut buf = [0u8; 4];
    read_exact_at(file, &mut buf, OFF_VERSION)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_rec_size(file: &File) -> Result<usize> {
    let mut buf = [0u8; 4];
    read_exact_at(file, &mut buf, OFF_REC_SIZE)?;
    let rec_size = u32::from_le_bytes(buf) as usize;
    if rec_size < MIN_REC_SIZE {
        return Err(VectorSearchError::CorruptData(format!(
            "directory record size {rec_size} < minimum {MIN_REC_SIZE}"
        )));
    }
    Ok(rec_size)
}

fn read_rec_capacity(file: &File) -> Result<u64> {
    let mut buf = [0u8; 8];
    read_exact_at(file, &mut buf, OFF_REC_CAPACITY)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_blob_len(file: &File) -> Result<u64> {
    let mut buf = [0u8; 8];
    read_exact_at(file, &mut buf, OFF_BLOB_LEN)?;
    Ok(u64::from_le_bytes(buf))
}

fn map_file(file: &File, len: usize) -> Result<Mmap> {
    let mmap = unsafe { MmapOptions::new().len(len).map(file) }?;
    Ok(mmap)
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut guard = file;
        guard.seek(SeekFrom::Start(offset))?;
        guard.read_exact(buf)
    }
}

#[cfg(unix)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(not(unix))]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut guard = file;
    guard.seek(SeekFrom::Start(offset))?;
    guard.write_all(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rec_size_round_trips_through_the_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dir.bin");
        {
            let dir_file = BlobDirectory::create(&path, *b"VTST", 12, 4).unwrap();
            dir_file.append_blob(0, b"hello", 1).unwrap();
            assert!(dir_file.sync().unwrap());
        }

        let dir_file = BlobDirectory::open(&path, *b"VTST").unwrap();
        let view = dir_file.snapshot();
        assert_eq!(view.rec_size(), 12, "record stride comes from the file");
        assert_eq!(view.blob(0), Some(&b"hello"[..]));
        assert_eq!(view.blob(1), None);
    }

    #[test]
    fn unusable_record_size_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dir.bin");
        {
            let dir_file = BlobDirectory::create(&path, *b"VTST", 12, 4).unwrap();
            dir_file.append_blob(0, b"x", 0).unwrap();
        }
        // Patch the header field to a stride that cannot hold `{off, len}`.
        {
            let file = File::options().write(true).open(&path).unwrap();
            write_at(&file, &4u32.to_le_bytes(), OFF_REC_SIZE).unwrap();
        }
        let err = BlobDirectory::open(&path, *b"VTST").unwrap_err();
        assert!(matches!(err, VectorSearchError::CorruptData(_)), "{err}");
    }
}
