//! `vectors.bin` — dense row-major f32 vectors stored in fixed-size segments.
//!
//! The file grows segment by segment; each segment is mapped once and the
//! mapping is published through an `ArcSwap`. Writers mutate slot rows through
//! the file handle (visible to readers through the shared page cache), readers
//! access the immutable mmaps lock-free.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;
use memmap2::{Mmap, MmapOptions};

use crate::error::{Result, VectorSearchError};

/// Dense vector storage with segment-granular mmap.
pub struct Vectors {
    path: PathBuf,
    file: parking_lot::Mutex<File>,
    dim: usize,
    segment_slots: u32,
    segments: ArcSwap<Vec<Arc<Mmap>>>,
}

impl Vectors {
    /// Create a new file with a single initial segment.
    pub fn create(path: &Path, dim: usize, segment_slots: u32) -> Result<Self> {
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        let segment_bytes = segment_bytes(dim, segment_slots);
        file.set_len(segment_bytes)?;
        file.sync_all()?;

        let segment = map_segment(&file, segment_bytes, 0)?;
        let segments = Arc::new(vec![Arc::new(segment)]);
        Ok(Self {
            path: path.to_path_buf(),
            file: parking_lot::Mutex::new(file),
            dim,
            segment_slots,
            segments: ArcSwap::from(segments),
        })
    }

    /// Open an existing file, validating that its length matches the declared
    /// slot capacity.
    pub fn open(path: &Path, dim: usize, segment_slots: u32, slot_capacity: u64) -> Result<Self> {
        let file = File::options().read(true).write(true).open(path)?;
        let meta = file.metadata()?;
        let expected = slot_capacity as usize * dim * 4;
        if meta.len() != expected as u64 {
            return Err(VectorSearchError::CorruptData(format!(
                "vectors.bin length {} does not match capacity {slot_capacity} * dim {dim} * 4",
                meta.len()
            )));
        }

        let segment_bytes = segment_bytes(dim, segment_slots);
        let mut segments = Vec::with_capacity(slot_capacity as usize + segment_slots as usize - 1);
        let mut offset = 0u64;
        while offset < meta.len() {
            let len = segment_bytes.min(meta.len() - offset);
            segments.push(Arc::new(map_segment(&file, len, offset)?));
            offset += len;
        }

        Ok(Self {
            path: path.to_path_buf(),
            file: parking_lot::Mutex::new(file),
            dim,
            segment_slots,
            segments: ArcSwap::from(Arc::new(segments)),
        })
    }

    /// Atomically replace the backing file with `tmp_path` and remap all
    /// segments. Used by compaction; readers keep their old segment snapshot
    /// until the swap.
    pub fn replace_from(&self, tmp_path: &Path) -> Result<()> {
        let dir = self.path.parent().ok_or_else(|| {
            VectorSearchError::Internal(format!("no parent dir for {}", self.path.display()))
        })?;
        std::fs::rename(tmp_path, &self.path)?;
        open_dir(dir)?.sync_all()?;

        let file = File::options().read(true).write(true).open(&self.path)?;
        let len = file.metadata()?.len();
        let segment_bytes = segment_bytes(self.dim, self.segment_slots);
        let mut segments = Vec::new();
        let mut offset = 0u64;
        while offset < len {
            let n = segment_bytes.min(len - offset);
            segments.push(Arc::new(map_segment(&file, n, offset)?));
            offset += n;
        }
        *self.file.lock() = file;
        self.segments.store(Arc::new(segments));
        Ok(())
    }

    /// Total number of slots currently mapped.
    pub fn slot_capacity(&self) -> u64 {
        self.segments.load().len() as u64 * self.segment_slots as u64
    }

    /// Take a snapshot of the mapped segments for lock-free reads.
    pub fn snapshot(&self) -> arc_swap::Guard<Arc<Vec<Arc<Mmap>>>> {
        self.segments.load()
    }

    /// Append one segment worth of slots.
    pub fn grow(&self) -> Result<()> {
        let segment_bytes = segment_bytes(self.dim, self.segment_slots);
        let file = self.file.lock();
        let old_len = file.metadata()?.len();
        let new_len = old_len + segment_bytes;
        file.set_len(new_len)?;
        file.sync_all()?;

        let segment = map_segment(&file, segment_bytes, old_len)?;
        let mut segments = (**self.segments.load()).clone();
        segments.push(Arc::new(segment));
        self.segments.store(Arc::new(segments));
        Ok(())
    }

    /// Grow to at least `target_slots` slots.
    pub fn grow_to(&self, target_slots: u64) -> Result<()> {
        while self.slot_capacity() < target_slots {
            self.grow()?;
        }
        Ok(())
    }

    /// Write the vector for a slot. The slot must be within current capacity.
    pub fn write_slot(&self, slot: u64, data: &[f32]) -> Result<()> {
        if data.len() != self.dim {
            return Err(VectorSearchError::InvalidVectorDimension {
                expected: self.dim,
                actual: data.len(),
            });
        }
        if slot >= self.slot_capacity() {
            return Err(VectorSearchError::Internal(format!(
                "slot {slot} out of capacity {}",
                self.slot_capacity()
            )));
        }
        let offset = slot as usize * self.dim * 4;
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        write_at(&self.file.lock(), bytes, offset as u64)?;
        Ok(())
    }

    /// Read the vector for a slot from a snapshot. Borrowed from the mmap.
    pub fn read_slot(
        snapshot: &[Arc<Mmap>],
        slot: u64,
        segment_slots: u32,
        dim: usize,
    ) -> Option<&[f32]> {
        let seg_idx = (slot / segment_slots as u64) as usize;
        let in_seg = slot % segment_slots as u64;
        let seg = snapshot.get(seg_idx)?;
        let offset = in_seg as usize * dim * 4;
        let end = offset + dim * 4;
        if end > seg.len() {
            return None;
        }
        let bytes = &seg[offset..end];
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, dim) })
    }
}

fn segment_bytes(dim: usize, segment_slots: u32) -> u64 {
    dim as u64 * segment_slots as u64 * 4
}

fn open_dir(dir: &Path) -> Result<File> {
    Ok(File::open(dir)?)
}

fn map_segment(file: &File, len: u64, offset: u64) -> Result<Mmap> {
    let mmap = unsafe {
        MmapOptions::new()
            .offset(offset)
            .len(len as usize)
            .map(file)
    }?;
    Ok(mmap)
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

impl std::fmt::Debug for Vectors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vectors")
            .field("path", &self.path)
            .field("dim", &self.dim)
            .field("segment_slots", &self.segment_slots)
            .field("slot_capacity", &self.slot_capacity())
            .finish()
    }
}
