//! Dense vectors stored as one mmap-backed file per physical segment.
//!
//! The generation directory owns `segments/<start>/vectors.bin`, where each
//! file covers exactly one segment's slot range. A global slot is resolved as
//! `segment.start + local_offset`, and a read guard is an ordered list of the
//! segment mappings so a search sees one consistent generation.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use parking_lot::RwLock;

use crate::error::{Result, VectorSearchError};

use super::segments::segment_dir;

/// One physical segment's vector file.
struct VectorSegment {
    path: PathBuf,
    file: parking_lot::Mutex<File>,
    dim: usize,
    slots: u32,
    mapping: Arc<Mmap>,
    dirty: std::sync::atomic::AtomicBool,
}

impl VectorSegment {
    fn create(path: &Path, dim: usize, slots: u32) -> Result<Self> {
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        let len = segment_bytes(dim, slots);
        file.set_len(len)?;
        file.sync_all()?;
        Ok(Self::from_file(file, path.to_path_buf(), dim, slots, len)?)
    }

    fn open(path: &Path, dim: usize, slots: u32) -> Result<Self> {
        let file = File::options().read(true).write(true).open(path)?;
        let expected = segment_bytes(dim, slots);
        let actual = file.metadata()?.len();
        if actual < expected {
            file.set_len(expected)?;
            file.sync_all()?;
        }
        Self::from_file(file, path.to_path_buf(), dim, slots, expected)
    }

    fn from_file(file: File, path: PathBuf, dim: usize, slots: u32, len: u64) -> Result<Self> {
        let mapping = Arc::new(unsafe { MmapOptions::new().len(len as usize).map(&file)? });
        Ok(Self {
            path,
            file: parking_lot::Mutex::new(file),
            dim,
            slots,
            mapping,
            dirty: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn write(&self, local: u32, data: &[f32]) -> Result<()> {
        if data.len() != self.dim || local as u64 >= self.slots as u64 {
            return Err(VectorSearchError::Internal(format!(
                "vector write outside segment {}: local {local}, dim {}",
                self.path.display(),
                self.dim
            )));
        }
        let offset = local as u64 * self.dim as u64 * 4;
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        write_at(&self.file.lock(), bytes, offset)?;
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn sync(&self) -> Result<bool> {
        if !self.dirty.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(false);
        }
        self.file.lock().sync_all()?;
        self.dirty
            .store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(true)
    }
}

/// All dense vector files of one generation.
pub struct Vectors {
    dir: PathBuf,
    dim: usize,
    segment_slots: u32,
    segments: RwLock<Vec<Arc<VectorSegment>>>,
}

impl Vectors {
    /// Create the first physical segment of a new generation.
    pub fn create(dir: &Path, dim: usize, segment_slots: u32) -> Result<Self> {
        let path = segment_dir(dir, 0).join("vectors.bin");
        let segment = VectorSegment::create(&path, dim, segment_slots)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            dim,
            segment_slots,
            segments: RwLock::new(vec![Arc::new(segment)]),
        })
    }

    /// Open every serving segment declared by the slot capacity.
    pub fn open(dir: &Path, dim: usize, segment_slots: u32, slot_capacity: u64) -> Result<Self> {
        if segment_slots == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "segment_slots must be positive".to_string(),
            ));
        }
        let count = slot_capacity.div_ceil(segment_slots as u64).max(1) as usize;
        let mut segments = Vec::with_capacity(count);
        for index in 0..count {
            let path = segment_dir(dir, index as u32 * segment_slots).join("vectors.bin");
            segments.push(Arc::new(VectorSegment::open(&path, dim, segment_slots)?));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            dim,
            segment_slots,
            segments: RwLock::new(segments),
        })
    }

    pub fn slot_capacity(&self) -> u64 {
        self.segments.read().len() as u64 * self.segment_slots as u64
    }

    /// One read guard covering every physical segment in slot order.
    pub fn snapshot(&self) -> Vec<Arc<Mmap>> {
        self.segments
            .read()
            .iter()
            .map(|segment| Arc::clone(&segment.mapping))
            .collect()
    }

    /// Append one physical segment.
    pub fn grow(&self) -> Result<()> {
        let index = self.segments.read().len() as u32;
        let path = segment_dir(&self.dir, index * self.segment_slots).join("vectors.bin");
        let segment = VectorSegment::create(&path, self.dim, self.segment_slots)?;
        self.segments.write().push(Arc::new(segment));
        Ok(())
    }

    pub fn grow_to(&self, target_slots: u64) -> Result<()> {
        while self.slot_capacity() < target_slots {
            self.grow()?;
        }
        Ok(())
    }

    pub fn write_slot(&self, slot: u64, data: &[f32]) -> Result<()> {
        let index = (slot / self.segment_slots as u64) as usize;
        let local = (slot % self.segment_slots as u64) as u32;
        let segments = self.segments.read();
        let Some(segment) = segments.get(index) else {
            return Err(VectorSearchError::Internal(format!(
                "slot {slot} outside {} segments",
                segments.len()
            )));
        };
        segment.write(local, data)
    }

    /// Flush one physical segment.
    pub fn sync_segment(&self, index: usize) -> Result<bool> {
        let segments = self.segments.read();
        let Some(segment) = segments.get(index) else {
            return Ok(false);
        };
        segment.sync()
    }

    /// Read a vector from a captured snapshot.
    pub fn read_slot(
        snapshot: &[Arc<Mmap>],
        slot: u64,
        segment_slots: u32,
        dim: usize,
    ) -> Option<&[f32]> {
        if segment_slots == 0 {
            return None;
        }
        let index = (slot / segment_slots as u64) as usize;
        let local = slot % segment_slots as u64;
        let segment = snapshot.get(index)?;
        let offset = local as usize * dim * 4;
        let end = offset + dim * 4;
        if end > segment.len() {
            return None;
        }
        let bytes = &segment[offset..end];
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, dim) })
    }
}

fn segment_bytes(dim: usize, slots: u32) -> u64 {
    dim as u64 * slots as u64 * 4
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
            .field("dir", &self.dir)
            .field("dim", &self.dim)
            .field("segment_slots", &self.segment_slots)
            .field("slot_capacity", &self.slot_capacity())
            .finish()
    }
}
