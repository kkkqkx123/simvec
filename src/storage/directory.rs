//! Directory files: `keys.bin` / `payloads.bin`.
//!
//! Both files share one layout — fixed header, fixed-size record array, and an
//! append-only blob area:
//!
//! ```text
//! [0..4)   magic           "VKEY" / "VPLD"
//! [4..8)   version         u32 = 1
//! [8..16)  rec_capacity    u64
//! [16..24) blob_len        u64
//! [24..)   rec array       rec_size * rec_capacity
//! [24 + rec_capacity*rec_size ..)  blob area (length blob_len)
//! ```
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

const HEADER_LEN: usize = 24;
const FILE_VERSION: u32 = 1;

/// Keys record: `{ off: u32, len: u32 }`.
pub(crate) const KEY_REC_SIZE: usize = 8;
/// Payloads record: `{ off: u32, len: u32, flags: u8, pad: [u8; 3] }`.
pub(crate) const SLOT_REC_SIZE: usize = 12;

/// Flag bit 0 marks a tombstoned slot.
pub(crate) const FLAG_TOMBSTONE: u8 = 1;

/// A read-only snapshot of a directory file.
pub struct DirView {
    mmap: Arc<Mmap>,
    rec_capacity: usize,
    blob_len: u64,
}

impl DirView {
    fn blob_offset(&self, rec_size: usize) -> usize {
        HEADER_LEN + self.rec_capacity * rec_size
    }

    /// Parse `{ off: u32, len: u32 }` of a record.
    fn rec_off_len(&self, rec_size: usize, slot: usize) -> Option<(u32, u32)> {
        if slot >= self.rec_capacity {
            return None;
        }
        let start = HEADER_LEN + slot * rec_size;
        let bytes = self.mmap.get(start..start + 8)?;
        let off = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let len = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        Some((off, len))
    }

    /// Blob for a slot. Returns `None` for empty records.
    pub fn blob(&self, rec_size: usize, slot: usize) -> Option<&[u8]> {
        let (off, len) = self.rec_off_len(rec_size, slot)?;
        if len == 0 {
            return None;
        }
        let start = self.blob_offset(rec_size) + off as usize;
        self.mmap.get(start..start + len as usize)
    }
}

/// Append-only directory file with a fixed-size record array and blob area.
pub struct BlobDirectory {
    path: PathBuf,
    magic: [u8; 4],
    rec_size: usize,
    file: parking_lot::Mutex<File>,
    view: ArcSwap<DirView>,
}

impl BlobDirectory {
    /// Create a new file with `initial_capacity` records.
    pub fn create(
        path: &Path,
        magic: [u8; 4],
        rec_size: usize,
        initial_capacity: u64,
    ) -> Result<Self> {
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        let file_len = HEADER_LEN + initial_capacity as usize * rec_size;
        file.set_len(file_len as u64)?;
        write_header(&file, magic, initial_capacity, 0)?;
        file.sync_all()?;

        let mmap = map_file(&file, file_len)?;
        let view = Arc::new(DirView {
            mmap: Arc::new(mmap),
            rec_capacity: initial_capacity as usize,
            blob_len: 0,
        });
        Ok(Self {
            path: path.to_path_buf(),
            magic,
            rec_size,
            file: parking_lot::Mutex::new(file),
            view: ArcSwap::from(view),
        })
    }

    /// Open an existing file.
    pub fn open(path: &Path, magic: [u8; 4], rec_size: usize) -> Result<Self> {
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
        let rec_capacity = read_rec_capacity(&file)? as usize;
        let blob_len = read_blob_len(&file)?;
        let expected = HEADER_LEN + rec_capacity * rec_size + blob_len as usize;
        if file_len != expected {
            return Err(VectorSearchError::CorruptData(format!(
                "{} length {} != expected {expected}",
                path.display(),
                file_len
            )));
        }
        let mmap = map_file(&file, file_len)?;
        let view = Arc::new(DirView {
            mmap: Arc::new(mmap),
            rec_capacity,
            blob_len,
        });
        Ok(Self {
            path: path.to_path_buf(),
            magic,
            rec_size,
            file: parking_lot::Mutex::new(file),
            view: ArcSwap::from(view),
        })
    }

    pub fn snapshot(&self) -> arc_swap::Guard<Arc<DirView>> {
        self.view.load()
    }

    /// Append a blob and wire up the record for `slot`.
    ///
    /// `slot` must be strictly less than the current record capacity.
    pub fn append_blob(&self, slot: usize, blob: &[u8], flags: u8) -> Result<()> {
        let view = self.view.load();
        if slot >= view.rec_capacity {
            return Err(VectorSearchError::Internal(format!(
                "slot {slot} out of record capacity {}",
                view.rec_capacity
            )));
        }
        let blob_offset = view.blob_offset(self.rec_size);
        let new_blob_len = view.blob_len + blob.len() as u64;

        // Blob bytes first, then the record, then the header water mark.
        write_at(
            &self.file.lock(),
            blob,
            (blob_offset + view.blob_len as usize) as u64,
        )?;
        let rec_bytes = encode_rec(
            view.blob_len as u32,
            blob.len() as u32,
            flags,
            self.rec_size,
        );
        let rec_start = HEADER_LEN + slot * self.rec_size;
        write_at(&self.file.lock(), &rec_bytes, rec_start as u64)?;
        write_at(&self.file.lock(), &new_blob_len.to_le_bytes(), 16)?;
        self.file.lock().sync_all()?;

        let file_len =
            (HEADER_LEN + view.rec_capacity * self.rec_size + new_blob_len as usize) as u64;
        let mmap = map_file(&self.file.lock(), file_len as usize)?;
        self.view.store(Arc::new(DirView {
            mmap: Arc::new(mmap),
            rec_capacity: view.rec_capacity,
            blob_len: new_blob_len,
        }));
        Ok(())
    }

    /// Update the flags byte of an existing record in place.
    pub fn set_flags(&self, slot: usize, flags: u8) -> Result<()> {
        let view = self.view.load();
        if slot >= view.rec_capacity {
            return Err(VectorSearchError::Internal(format!(
                "slot {slot} out of record capacity {}",
                view.rec_capacity
            )));
        }
        let flag_start = HEADER_LEN + slot * self.rec_size + 8;
        write_at(&self.file.lock(), &[flags], flag_start as u64)?;
        Ok(())
    }

    /// Atomically replace the backing file with `tmp_path` (which must be a
    /// complete, valid directory file) and refresh the in-memory view.
    ///
    /// Used by compaction, which builds a compacted file and renames it over
    /// the live one. Readers keep their old mmap snapshot until the swap.
    pub fn replace_from(&self, tmp_path: &Path) -> Result<()> {
        let dir = self.path.parent().ok_or_else(|| {
            VectorSearchError::Internal(format!("no parent dir for {}", self.path.display()))
        })?;
        std::fs::rename(tmp_path, &self.path)?;
        open_dir(dir)?.sync_all()?;

        let file = File::options().read(true).write(true).open(&self.path)?;
        let file_len = file.metadata()?.len() as usize;
        if file_len < HEADER_LEN {
            return Err(VectorSearchError::CorruptData(format!(
                "{} too short: {} bytes",
                self.path.display(),
                file_len
            )));
        }
        if read_magic(&file)? != self.magic {
            return Err(VectorSearchError::CorruptData(format!(
                "{} bad magic",
                self.path.display()
            )));
        }
        if read_version(&file)? != FILE_VERSION {
            return Err(VectorSearchError::CorruptData(format!(
                "{} unsupported version",
                self.path.display()
            )));
        }
        let rec_capacity = read_rec_capacity(&file)? as usize;
        let blob_len = read_blob_len(&file)?;
        let expected = HEADER_LEN + rec_capacity * self.rec_size + blob_len as usize;
        if file_len != expected {
            return Err(VectorSearchError::CorruptData(format!(
                "{} length {} != expected {expected}",
                self.path.display(),
                file_len
            )));
        }
        let mmap = map_file(&file, file_len)?;
        *self.file.lock() = file;
        self.view.store(Arc::new(DirView {
            mmap: Arc::new(mmap),
            rec_capacity,
            blob_len,
        }));
        Ok(())
    }

    /// Grow the record array to at least `target_capacity` records.
    ///
    /// Relocates the blob area, so the file is rebuilt atomically via a
    /// temp-file + rename.
    pub fn grow_to(&self, target_capacity: u64) -> Result<()> {
        let view = self.view.load();
        if target_capacity as usize <= view.rec_capacity {
            return Ok(());
        }

        let tmp_path = self.path.with_extension("bin.tmp");
        let mut tmp = File::create(&tmp_path)?;
        tmp.set_len((HEADER_LEN + target_capacity as usize * self.rec_size) as u64)?;
        write_header(&tmp, self.magic, target_capacity, view.blob_len)?;

        // Copy live records then the blob area.
        let old_capacity = view.rec_capacity;
        let old_rec_bytes = old_capacity * self.rec_size;
        write_region(
            &mut tmp,
            &self.file.lock(),
            HEADER_LEN,
            HEADER_LEN,
            old_rec_bytes,
        )?;
        let old_blob_start = HEADER_LEN + old_capacity * self.rec_size;
        let new_blob_start = HEADER_LEN + target_capacity as usize * self.rec_size;
        write_region(
            &mut tmp,
            &self.file.lock(),
            new_blob_start,
            old_blob_start,
            view.blob_len as usize,
        )?;
        tmp.sync_all()?;

        let dir = self.path.parent().ok_or_else(|| {
            VectorSearchError::Internal(format!("no parent dir for {}", self.path.display()))
        })?;
        std::fs::rename(&tmp_path, &self.path)?;
        open_dir(dir)?.sync_all()?;

        let file = File::options().read(true).write(true).open(&self.path)?;
        let file_len = file.metadata()?.len() as usize;
        let mmap = map_file(&file, file_len)?;
        *self.file.lock() = file;
        self.view.store(Arc::new(DirView {
            mmap: Arc::new(mmap),
            rec_capacity: target_capacity as usize,
            blob_len: view.blob_len,
        }));
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

fn write_header(file: &File, magic: [u8; 4], rec_capacity: u64, blob_len: u64) -> Result<()> {
    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(&magic);
    header.extend_from_slice(&FILE_VERSION.to_le_bytes());
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
    read_exact_at(file, &mut buf, 4)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_rec_capacity(file: &File) -> Result<u64> {
    let mut buf = [0u8; 8];
    read_exact_at(file, &mut buf, 8)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_blob_len(file: &File) -> Result<u64> {
    let mut buf = [0u8; 8];
    read_exact_at(file, &mut buf, 16)?;
    Ok(u64::from_le_bytes(buf))
}

fn map_file(file: &File, len: usize) -> Result<Mmap> {
    let mmap = unsafe { MmapOptions::new().len(len).map(file) }?;
    Ok(mmap)
}

fn open_dir(dir: &Path) -> Result<File> {
    Ok(File::open(dir)?)
}

/// Copy `len` bytes from `src` at `src_off` into `dst` at `dst_off`.
fn write_region(
    dst: &mut File,
    src: &File,
    dst_off: usize,
    src_off: usize,
    len: usize,
) -> Result<()> {
    const CHUNK: usize = 64 * 1024;
    let mut remaining = len;
    let mut pos = 0;
    while remaining > 0 {
        let n = remaining.min(CHUNK);
        let mut buf = vec![0u8; n];
        read_exact_at(src, &mut buf, (src_off + pos) as u64)?;
        write_at(dst, &buf, (dst_off + pos) as u64)?;
        pos += n;
        remaining -= n;
    }
    Ok(())
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
