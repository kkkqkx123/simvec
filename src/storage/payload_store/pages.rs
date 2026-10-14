//! mmap-backed data pages for the payload store.
//!
//! Each page is a fixed-size file (default 32 MiB) holding payload data
//! across contiguous 128-byte blocks. Pages are memory-mapped for fast
//! reads and written through a `File` handle.
//!
//! Pages are named `page_NNNN.bin` (zero-padded to 4 digits).

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};
use parking_lot::{Mutex, RwLock};

use crate::error::{Result, VectorSearchError};

use super::{StoreConfig, ValuePointer};

/// Pages storage: manages a collection of mmap-backed data page files.
pub struct Pages {
    base_path: PathBuf,
    page_files: RwLock<HashMap<u32, PageEntry>>,
    /// Page ids with bytes written since the last `sync`. A checkpoint flushes
    /// exactly these pages, keeping the per-write cost O(1).
    dirty: Mutex<HashSet<u32>>,
}

struct PageEntry {
    file: File,
    mmap: RwLock<Option<Mmap>>,
    _len: u64,
}

impl Pages {
    /// Create a fresh pages directory (no pages yet).
    pub fn create(path: &Path, _config: &StoreConfig) -> Result<Self> {
        let pages_dir = path.join("pages");
        std::fs::create_dir_all(&pages_dir)?;
        Ok(Self {
            base_path: pages_dir,
            page_files: RwLock::new(HashMap::new()),
            dirty: Mutex::new(HashSet::new()),
        })
    }

    /// Open existing pages from disk.
    pub fn open(path: &Path) -> Result<Self> {
        let pages_dir = path.join("pages");
        if !pages_dir.exists() {
            std::fs::create_dir_all(&pages_dir)?;
        }

        let mut page_files = HashMap::new();
        for entry in std::fs::read_dir(&pages_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if let Some(page_id) = name_str
                .strip_prefix("page_")
                .and_then(|s| s.strip_suffix(".bin"))
                .and_then(|s| s.parse::<u32>().ok())
            {
                let page = Self::open_page(entry.path())?;
                page_files.insert(page_id, page);
            }
        }

        Ok(Self {
            base_path: pages_dir,
            page_files: RwLock::new(page_files),
            dirty: Mutex::new(HashSet::new()),
        })
    }

    /// Ensure a page exists (creating it if necessary).
    pub fn ensure_page(&self, page_id: usize, config: &StoreConfig) -> Result<()> {
        let page_id = page_id as u32;
        {
            let pages = self.page_files.read();
            if pages.contains_key(&page_id) {
                return Ok(());
            }
        }
        let path = self.page_path(page_id);
        let file = File::create(&path)?;
        file.set_len(config.page_size as u64)?;
        let entry = Self::open_page(path)?;
        self.page_files.write().insert(page_id, entry);
        Ok(())
    }

    /// Flush every page written since the last `sync` and return how many
    /// pages were dirty. Called at checkpoint time only; WAL replay covers
    /// page contents a crash may have lost before this point.
    pub fn sync(&self) -> usize {
        let dirty = std::mem::take(&mut *self.dirty.lock());
        let count = dirty.len();
        let pages = self.page_files.read();
        for page_id in &dirty {
            if let Some(entry) = pages.get(page_id) {
                let _ = entry.file.sync_all();
            }
        }
        count
    }

    /// Read a value from the pages, following the pointer.
    pub fn read_value(&self, pointer: &ValuePointer, config: &StoreConfig) -> Result<Vec<u8>> {
        if pointer.is_empty() {
            return Ok(Vec::new());
        }

        let pages = self.page_files.read();
        let entry = pages.get(&pointer.page_id).ok_or_else(|| {
            VectorSearchError::CorruptData(format!(
                "page {} not found for payload read",
                pointer.page_id
            ))
        })?;

        let mmap_guard = entry.mmap.read();
        let mmap = mmap_guard.as_ref().ok_or_else(|| {
            VectorSearchError::CorruptData(format!("page {} has no mmap", pointer.page_id))
        })?;

        let start = pointer.block_offset as usize * config.block_size;
        let end = start + pointer.length as usize;
        if end > mmap.len() {
            return Err(VectorSearchError::CorruptData(format!(
                "payload read out of bounds: page {} offset {} len {} (page size {})",
                pointer.page_id,
                start,
                pointer.length,
                mmap.len(),
            )));
        }

        Ok(mmap[start..end].to_vec())
    }

    /// Write a value to the pages at the given pointer location.
    ///
    /// The bytes land in the page cache through the file handle and are
    /// visible to readers through the page's mmap (both map the same cache
    /// pages). The page is recorded as dirty so the next checkpoint flushes
    /// it; the WAL already carries the payload for replay-based recovery.
    pub fn write_value(
        &self,
        pointer: &ValuePointer,
        data: &[u8],
        config: &StoreConfig,
    ) -> Result<()> {
        let pages = self.page_files.read();
        let entry = pages.get(&pointer.page_id).ok_or_else(|| {
            VectorSearchError::CorruptData(format!(
                "page {} not found for payload write",
                pointer.page_id
            ))
        })?;

        let start = pointer.block_offset as usize * config.block_size;
        let _end = start + data.len();

        write_at(&entry.file, start as u64, data)?;
        drop(pages);
        self.dirty.lock().insert(pointer.page_id);

        Ok(())
    }

    /// Get the file path for a page.
    fn page_path(&self, page_id: u32) -> PathBuf {
        self.base_path.join(format!("page_{:04}.bin", page_id))
    }

    /// Open a single page file and create its mmap.
    fn open_page(path: PathBuf) -> Result<PageEntry> {
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        let len = file.metadata()?.len();
        let mmap = if len > 0 {
            Some(unsafe { MmapOptions::new().map(&file)? })
        } else {
            None
        };
        Ok(PageEntry {
            file,
            mmap: RwLock::new(mmap),
            _len: len,
        })
    }
}

#[cfg(unix)]
fn write_at(file: &File, offset: u64, buf: &[u8]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(not(unix))]
fn write_at(file: &File, offset: u64, buf: &[u8]) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let mut file = file;
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_config() -> StoreConfig {
        StoreConfig {
            block_size: 128,
            page_size: 1024,
        }
    }

    #[test]
    fn test_create_and_write() {
        let dir = tempdir().unwrap();
        let config = test_config();
        let pages = Pages::create(dir.path(), &config).unwrap();
        pages.ensure_page(0, &config).unwrap();

        let data = b"hello world";
        let ptr = ValuePointer::new(0, 0, data.len() as u32);
        pages.write_value(&ptr, data, &config).unwrap();

        let got = pages.read_value(&ptr, &config).unwrap();
        assert_eq!(got, data);
    }

    #[test]
    fn test_multiple_pages() {
        let dir = tempdir().unwrap();
        let config = test_config();
        let pages = Pages::create(dir.path(), &config).unwrap();
        pages.ensure_page(0, &config).unwrap();
        pages.ensure_page(1, &config).unwrap();

        let data0 = b"page zero";
        let data1 = b"page one";
        let ptr0 = ValuePointer::new(0, 0, data0.len() as u32);
        let ptr1 = ValuePointer::new(1, 0, data1.len() as u32);

        pages.write_value(&ptr0, data0, &config).unwrap();
        pages.write_value(&ptr1, data1, &config).unwrap();

        assert_eq!(pages.read_value(&ptr0, &config).unwrap(), data0);
        assert_eq!(pages.read_value(&ptr1, &config).unwrap(), data1);
    }
}
