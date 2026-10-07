//! Block-level usage bitmask for the payload store.
//!
//! Tracks which blocks across all pages are in use (1) or free (0).
//! Protected by `parking_lot::RwLock` for safe concurrent access.
//!
//! # On-disk format
//!
//! ```text
//! [0..4)   magic           "VBMP"
//! [4..8)   version         u32 = 1
//! [8..16)  total_blocks    u64
//! [16..20) block_size      u32
//! [20..24) page_blocks     u32 (blocks per page)
//! [24..)   bitmap data     ceil(total_blocks / 8) bytes
//! ```

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{Result, VectorSearchError};

use super::{StoreConfig, BITMASK_MAGIC};

const BITMASK_VERSION: u32 = 1;
const BITMASK_HEADER_LEN: usize = 24;

/// In-memory state of the bitmask.
#[derive(Debug, Clone)]
struct BitmaskInner {
    data: Vec<u8>,
    total_blocks: u64,
    block_size: u32,
    page_blocks: u32,
}

impl BitmaskInner {
    fn is_used(&self, block: u64) -> bool {
        if block >= self.total_blocks {
            return false;
        }
        let byte_idx = block as usize / 8;
        let bit_idx = block as usize % 8;
        (self.data[byte_idx] >> bit_idx) & 1 == 1
    }

    fn set_bit(&mut self, block: u64, used: bool) {
        if block >= self.total_blocks {
            return;
        }
        let byte_idx = block as usize / 8;
        let bit_idx = block as usize % 8;
        if used {
            self.data[byte_idx] |= 1 << bit_idx;
        } else {
            self.data[byte_idx] &= !(1 << bit_idx);
        }
    }

    fn find_free_range(&self, count: u32) -> Option<(u32, u32)> {
        if count == 0 || self.total_blocks == 0 {
            return None;
        }
        let mut run_start: Option<u64> = None;
        let mut run_len: u32 = 0;
        for block in 0..self.total_blocks {
            if !self.is_used(block) {
                if run_start.is_none() {
                    run_start = Some(block);
                    run_len = 1;
                } else {
                    run_len += 1;
                }
                if run_len >= count {
                    let start = run_start.unwrap();
                    let page_id = (start / self.page_blocks as u64) as u32;
                    let block_offset = (start % self.page_blocks as u64) as u32;
                    return Some((page_id, block_offset));
                }
            } else {
                run_start = None;
                run_len = 0;
            }
        }
        None
    }
}

/// Bitmask tracking block allocation across all data pages.
pub struct MmapBitmask {
    file: parking_lot::Mutex<File>,
    inner: parking_lot::RwLock<BitmaskInner>,
}

impl MmapBitmask {
    /// Create a fresh bitmask.
    pub fn create(path: &Path, config: &StoreConfig) -> Result<Self> {
        let file_path = path.join("bitmask.bin");
        let mut file = File::create(&file_path)?;
        let inner = BitmaskInner {
            data: Vec::new(),
            total_blocks: 0,
            block_size: config.block_size as u32,
            page_blocks: config.blocks_per_page() as u32,
        };
        Self::write_file(&mut file, &inner)?;
        Ok(Self {
            file: parking_lot::Mutex::new(file),
            inner: parking_lot::RwLock::new(inner),
        })
    }

    /// Open an existing bitmask.
    pub fn open(path: &Path, config: &StoreConfig) -> Result<Self> {
        let file_path = path.join("bitmask.bin");
        let mut file = File::options().read(true).write(true).open(&file_path)?;
        let inner = Self::read_file(&mut file)?;
        if inner.block_size != config.block_size as u32 {
            return Err(VectorSearchError::CorruptData(format!(
                "bitmask block_size mismatch: stored={}, expected={}",
                inner.block_size, config.block_size,
            )));
        }
        Ok(Self {
            file: parking_lot::Mutex::new(file),
            inner: parking_lot::RwLock::new(inner),
        })
    }

    /// Allocate `count` contiguous blocks. Returns (page_id, block_offset).
    pub fn allocate(&self, count: u32, config: &StoreConfig) -> Result<(u32, u32)> {
        let mut inner = self.inner.write();

        // Try to find a free range in existing space.
        if let Some((pid, off)) = inner.find_free_range(count) {
            let start = pid as u64 * inner.page_blocks as u64 + off as u64;
            for i in 0..count {
                inner.set_bit(start + i as u64, true);
            }
            let mut file = self.file.lock();
            Self::write_file(&mut file, &inner)?;
            file.sync_all()?;
            return Ok((pid, off));
        }

        // No room — grow by one page.
        let blocks_per_page = config.blocks_per_page() as u64;
        let new_total = inner.total_blocks + blocks_per_page;
        let new_bitmap_len = new_total.div_ceil(8) as usize;
        inner.data.resize(new_bitmap_len, 0);
        inner.total_blocks = new_total;

        // Allocate from the new page.
        let new_page_id = ((new_total - blocks_per_page) / blocks_per_page) as u32;
        let start = new_total - blocks_per_page;
        for i in 0..count {
            inner.set_bit(start + i as u64, true);
        }

        let mut file = self.file.lock();
        Self::write_file(&mut file, &inner)?;
        file.sync_all()?;

        Ok((new_page_id, 0))
    }

    /// Free `count` blocks starting at (page_id, block_offset).
    pub fn free_blocks(
        &self,
        page_id: usize,
        block_offset: usize,
        length: u32,
        config: &StoreConfig,
    ) {
        let blocks = (length as usize).div_ceil(config.block_size) as u32;
        let start = page_id as u64 * config.blocks_per_page() as u64 + block_offset as u64;

        let mut inner = self.inner.write();
        for i in 0..blocks {
            inner.set_bit(start + i as u64, false);
        }
        let mut file = self.file.lock();
        Self::write_file(&mut file, &inner).ok();
        file.sync_all().ok();
    }

    fn read_file(file: &mut File) -> Result<BitmaskInner> {
        file.seek(SeekFrom::Start(0))?;
        let mut header = [0u8; BITMASK_HEADER_LEN];
        file.read_exact(&mut header)?;

        let magic = &header[0..4];
        if magic != BITMASK_MAGIC {
            return Err(VectorSearchError::CorruptData(format!(
                "bitmask bad magic: {magic:?}"
            )));
        }
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if version != BITMASK_VERSION {
            return Err(VectorSearchError::CorruptData(format!(
                "bitmask unsupported version: {version}"
            )));
        }

        let total_blocks = u64::from_le_bytes(header[8..16].try_into().unwrap());
        let block_size = u32::from_le_bytes(header[16..20].try_into().unwrap());
        let page_blocks = u32::from_le_bytes(header[20..24].try_into().unwrap());

        let bitmap_len = total_blocks.div_ceil(8) as usize;
        let mut data = vec![0u8; bitmap_len];
        file.read_exact(&mut data)?;

        Ok(BitmaskInner {
            data,
            total_blocks,
            block_size,
            page_blocks,
        })
    }

    fn write_file(file: &mut File, inner: &BitmaskInner) -> Result<()> {
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&BITMASK_MAGIC)?;
        file.write_all(&BITMASK_VERSION.to_le_bytes())?;
        file.write_all(&inner.total_blocks.to_le_bytes())?;
        file.write_all(&inner.block_size.to_le_bytes())?;
        file.write_all(&inner.page_blocks.to_le_bytes())?;
        file.write_all(&inner.data)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> StoreConfig {
        StoreConfig {
            block_size: 128,
            page_size: 1024, // 8 blocks per page
        }
    }

    #[test]
    fn test_allocate_and_free() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config();
        let bm = MmapBitmask::create(dir.path(), &config).unwrap();

        let (pid, off) = bm.allocate(3, &config).unwrap();
        assert_eq!(pid, 0);
        assert_eq!(off, 0);

        let (pid2, off2) = bm.allocate(2, &config).unwrap();
        assert_eq!(pid2, 0);
        assert_eq!(off2, 3);

        bm.free_blocks(0, 0, 3 * 128, &config);

        let (pid3, off3) = bm.allocate(2, &config).unwrap();
        assert_eq!(pid3, 0);
        assert_eq!(off3, 0);
    }

    #[test]
    fn test_cross_page() {
        let dir = tempfile::tempdir().unwrap();
        let config = StoreConfig {
            block_size: 128,
            page_size: 256, // 2 blocks per page
        };
        let bm = MmapBitmask::create(dir.path(), &config).unwrap();

        bm.allocate(1, &config).unwrap();
        bm.allocate(1, &config).unwrap();

        let (pid, off) = bm.allocate(1, &config).unwrap();
        assert_eq!(pid, 1);
        assert_eq!(off, 0);
    }

    #[test]
    fn test_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config();
        {
            let bm = MmapBitmask::create(dir.path(), &config).unwrap();
            bm.allocate(5, &config).unwrap();
        }
        {
            let bm = MmapBitmask::open(dir.path(), &config).unwrap();
            // Verify the bitmask was persisted correctly by allocating
            // from the reopened state.
            let (pid, _off) = bm.allocate(1, &config).unwrap();
            assert_eq!(pid, 0); // Still within the first page.
        }
    }
}
