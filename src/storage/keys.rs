//! Slot-to-point-id directories, one file per physical segment.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::error::{Result, VectorSearchError};
use crate::storage::directory::{BlobDirectory, DirView, KEY_REC_SIZE};

use super::segments::segment_dir;

const MAGIC: [u8; 4] = *b"VKEY";

struct KeySegment {
    dir: BlobDirectory,
}

impl KeySegment {
    fn create(path: &Path, capacity: u64) -> Result<Self> {
        Ok(Self {
            dir: BlobDirectory::create(path, MAGIC, KEY_REC_SIZE, capacity)?,
        })
    }

    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            dir: BlobDirectory::open(path, MAGIC)?,
        })
    }
}

/// Immutable id directory of one generation, in slot order.
pub struct KeysView {
    segment_slots: u32,
    segments: Vec<Arc<DirView>>,
}

impl KeysView {
    fn blob(&self, slot: usize) -> Option<&[u8]> {
        if self.segment_slots == 0 {
            return None;
        }
        let index = slot / self.segment_slots as usize;
        let local = slot % self.segment_slots as usize;
        self.segments.get(index)?.blob(local)
    }
}

/// All per-segment key directories of one generation.
pub struct Keys {
    base: PathBuf,
    segment_slots: u32,
    segments: RwLock<Vec<KeySegment>>,
}

impl Keys {
    pub fn create(dir: &Path, initial_capacity: u64, segment_slots: u32) -> Result<Self> {
        let path = segment_dir(dir, 0).join("keys.bin");
        Ok(Self {
            base: dir.to_path_buf(),
            segment_slots,
            segments: RwLock::new(vec![KeySegment::create(&path, initial_capacity)?]),
        })
    }

    pub fn open(dir: &Path, segment_slots: u32, slot_capacity: u64) -> Result<Self> {
        if segment_slots == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "segment_slots must be positive".to_string(),
            ));
        }
        let count = slot_capacity.div_ceil(segment_slots as u64).max(1) as usize;
        let mut segments = Vec::with_capacity(count);
        for index in 0..count {
            let path = segment_dir(dir, index as u32 * segment_slots).join("keys.bin");
            segments.push(KeySegment::open(&path)?);
        }
        Ok(Self {
            base: dir.to_path_buf(),
            segment_slots,
            segments: RwLock::new(segments),
        })
    }

    pub fn snapshot(&self) -> KeysView {
        KeysView {
            segment_slots: self.segment_slots,
            segments: self
                .segments
                .read()
                .iter()
                .map(|segment| Arc::clone(&*segment.dir.snapshot()))
                .collect(),
        }
    }

    fn ensure_segments(&self, target_capacity: u64) -> Result<()> {
        let needed = target_capacity.div_ceil(self.segment_slots as u64).max(1) as usize;
        let mut segments = self.segments.write();
        while segments.len() < needed {
            let start = segments.len() as u32 * self.segment_slots;
            let path = segment_dir(self.dir_for(start), start).join("keys.bin");
            segments.push(KeySegment::create(&path, self.segment_slots as u64)?);
        }
        Ok(())
    }

    fn dir_for(&self, _start: u32) -> &Path {
        // `Keys` is created inside a generation directory; the first segment
        // records the base path so later segments can be created on growth.
        &self.base
    }

    pub fn grow_to(&self, target_capacity: u64) -> Result<()> {
        self.ensure_segments(target_capacity)
    }

    pub fn sync_segment(&self, index: usize) -> Result<bool> {
        let segments = self.segments.read();
        let Some(segment) = segments.get(index) else {
            return Ok(false);
        };
        segment.dir.sync()
    }

    pub fn append_key(&self, slot: usize, key: &str) -> Result<()> {
        let index = slot / self.segment_slots as usize;
        let local = slot % self.segment_slots as usize;
        let segments = self.segments.read();
        let Some(segment) = segments.get(index) else {
            return Err(VectorSearchError::Internal(format!(
                "key slot {slot} outside {} segments",
                segments.len()
            )));
        };
        segment.dir.append_blob(local, key.as_bytes(), 0)
    }

    pub fn read_key(view: &KeysView, slot: usize) -> Result<Option<String>> {
        match view.blob(slot) {
            Some(bytes) => {
                let key = std::str::from_utf8(bytes).map_err(|e| {
                    VectorSearchError::CorruptData(format!("invalid utf8 key at slot {slot}: {e}"))
                })?;
                Ok(Some(key.to_string()))
            }
            None => Ok(None),
        }
    }
}
