//! Block-level usage bitmask for the payload store.
//!
//! Tracks which blocks across all pages are in use (1) or free (0).
//! Protected by `parking_lot::RwLock` for safe concurrent access.
//!
//! Allocation state lives in memory during normal operation and is persisted
//! only at a payload-store checkpoint. Checkpoints publish the bitmap by
//! writing `bitmask.bin.tmp`, fsyncing it and renaming it over the live file,
//! so a crash cannot leave a half-written allocator state that fails to open.
//! A stale but complete bitmap is still safe: WAL replay re-applies every
//! allocation and converges in-memory state, and stale free bits can only
//! leak a block, never alias two live payloads.
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

use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{Result, VectorSearchError};

use super::{StoreConfig, ValuePointer, BITMASK_MAGIC};

const BITMASK_VERSION: u32 = 1;
const BITMASK_HEADER_LEN: usize = 24;

/// In-memory state of the bitmask.
#[derive(Debug, Clone)]
struct BitmaskInner {
    data: Vec<u8>,
    total_blocks: u64,
    block_size: u32,
    page_blocks: u32,
    /// Next-fit cursor: the search for a free run starts here instead of at
    /// block 0, which turns repeated `put` calls from a full-table scan per
    /// allocation into an amortised forward pass.
    next_fit: u64,
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

    /// Find a free run of `count` blocks that lies **entirely inside one page**.
    ///
    /// A pointer is `(page_id, block_offset, len)`, and `read_value` rejects
    /// any read that would cross the page's mmap. Allowing a run to straddle a
    /// page boundary therefore corrupts the page file (the write goes past the
    /// fixed page size) and breaks the block→offset invariant for every later
    /// allocation. The search is per page, in two passes around the next-fit
    /// cursor, so a run can never straddle by construction.
    fn find_free_range(&mut self, count: u32) -> Option<(u32, u32)> {
        let page_blocks = self.page_blocks as u64;
        if count == 0 || self.total_blocks == 0 || count as u64 > page_blocks {
            return None;
        }
        let total_pages = self.total_blocks.div_ceil(page_blocks);
        let cursor_page = self.next_fit / page_blocks;
        for pass in 0..2u32 {
            let (begin, end) = if pass == 0 {
                (cursor_page, total_pages)
            } else {
                (0, cursor_page.min(total_pages))
            };
            for page in begin..end {
                let page_start = page * page_blocks;
                let page_end = (page_start + page_blocks).min(self.total_blocks);
                let mut run_start: Option<u64> = None;
                let mut run_len: u32 = 0;
                for block in page_start..page_end {
                    if self.is_used(block) {
                        run_start = None;
                        run_len = 0;
                        continue;
                    }
                    match run_start {
                        Some(_) => run_len += 1,
                        None => {
                            run_start = Some(block);
                            run_len = 1;
                        }
                    }
                    if run_len >= count {
                        let start = run_start.expect("run start recorded");
                        self.next_fit = start + count as u64;
                        return Some(((start / page_blocks) as u32, (start % page_blocks) as u32));
                    }
                }
            }
        }
        None
    }
}

/// Bitmask tracking block allocation across all data pages.
pub struct MmapBitmask {
    path: PathBuf,
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
            next_fit: 0,
        };
        Self::write_file(&mut file, &inner)?;
        file.sync_all()?;
        Ok(Self {
            path: file_path,
            inner: parking_lot::RwLock::new(inner),
        })
    }

    /// Open the bitmask, rebuilding it from live pointers when it is damaged
    /// or when a stale bit would alias a live payload.
    ///
    /// The bitmap is derived state: the tracker is the authority for which
    /// blocks a live pointer owns. Orphaned used blocks are harmless and are
    /// left alone, but a free bit on a live pointer must never survive.
    pub fn open_or_rebuild(
        path: &Path,
        config: &StoreConfig,
        pointers: impl Iterator<Item = ValuePointer>,
    ) -> Result<Self> {
        let pointers: Vec<ValuePointer> = pointers.collect();
        match Self::open(path, config) {
            Ok(bitmask) if !bitmask.needs_rebuild(&pointers, config) => Ok(bitmask),
            Ok(bitmask) => {
                bitmask.rebuild(&pointers, config)?;
                bitmask.persist()?;
                Ok(bitmask)
            }
            Err(_) => {
                let bitmask = Self::create(path, config)?;
                bitmask.rebuild(&pointers, config)?;
                bitmask.persist()?;
                Ok(bitmask)
            }
        }
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
        if inner.page_blocks != config.blocks_per_page() as u32 {
            return Err(VectorSearchError::CorruptData(format!(
                "bitmask page_blocks mismatch: stored={}, expected={}",
                inner.page_blocks,
                config.blocks_per_page(),
            )));
        }
        Ok(Self {
            path: file_path,
            inner: parking_lot::RwLock::new(inner),
        })
    }

    /// A stale bitmap is safe as long as every live pointer still owns its
    /// blocks. Only a free bit on a live block requires a rebuild.
    fn needs_rebuild(&self, pointers: &[ValuePointer], config: &StoreConfig) -> bool {
        let inner = self.inner.read();
        Self::live_blocks(pointers, config).any(|block| !inner.is_used(block))
    }

    /// Rebuild the allocation exactly from the live pointer table.
    fn rebuild(&self, pointers: &[ValuePointer], config: &StoreConfig) -> Result<()> {
        let blocks: Vec<u64> = Self::live_blocks(pointers, config).collect();
        let mut inner = self.inner.write();
        let needed = blocks.iter().copied().max().unwrap_or(0) + 1;
        if needed > inner.total_blocks {
            let page_blocks = config.blocks_per_page() as u64;
            inner.total_blocks = needed.div_ceil(page_blocks) * page_blocks;
            let bitmap_len = inner.total_blocks.div_ceil(8) as usize;
            inner.data.resize(bitmap_len, 0);
        }
        inner.data.fill(0);
        for block in blocks {
            inner.set_bit(block, true);
        }
        inner.next_fit = 0;
        Ok(())
    }

    /// Expand every live pointer into the global block indices it owns.
    fn live_blocks<'a>(
        pointers: &'a [ValuePointer],
        config: &'a StoreConfig,
    ) -> impl Iterator<Item = u64> + 'a {
        let blocks_per_page = config.blocks_per_page() as u64;
        let mut seen = HashSet::new();
        pointers
            .iter()
            .filter(|p| !p.is_empty())
            .flat_map(move |p| {
                let start = p.page_id as u64 * blocks_per_page + p.block_offset as u64;
                let count = (p.length as u64).div_ceil(config.block_size as u64);
                start..start + count
            })
            .filter(move |block| seen.insert(*block))
    }

    /// Allocate `count` contiguous blocks. Returns (page_id, block_offset).
    ///
    /// The run is always page-local (see `find_free_range`). When no page has
    /// room the bitmap grows one page at a time until it does, so a request
    /// that fits in a page can never be satisfied by an out-of-range offset.
    /// `count` larger than a page's block count cannot be represented at all:
    /// a payload spanning pages would need a cross-page pointer format, so it
    /// is rejected before anything is written.
    ///
    /// In-memory only; the updated bitmap is persisted by
    /// [`MmapBitmask::persist`] at checkpoint time.
    pub fn allocate(&self, count: u32, config: &StoreConfig) -> Result<(u32, u32)> {
        let blocks_per_page = config.blocks_per_page() as u64;
        if count == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "payload allocation of zero blocks".to_string(),
            ));
        }
        if count as u64 > blocks_per_page {
            return Err(VectorSearchError::InvalidConfig(format!(
                "payload needs {count} blocks of {} bytes, which exceeds one {}-byte page",
                config.block_size, config.page_size,
            )));
        }

        let mut inner = self.inner.write();
        if let Some((pid, off)) = inner.find_free_range(count) {
            let start = pid as u64 * blocks_per_page + off as u64;
            for i in 0..count {
                inner.set_bit(start + i as u64, true);
            }
            return Ok((pid, off));
        }
        // No page has room — add one page; a single-page request always fits
        // in a fresh page.
        let new_page_id = (inner.total_blocks / blocks_per_page) as u32;
        let new_total = inner.total_blocks + blocks_per_page;
        let new_bitmap_len = new_total.div_ceil(8) as usize;
        inner.data.resize(new_bitmap_len, 0);
        inner.total_blocks = new_total;
        let start = new_page_id as u64 * blocks_per_page;
        for i in 0..count {
            inner.set_bit(start + i as u64, true);
        }
        inner.next_fit = start + count as u64;
        Ok((new_page_id, 0))
    }

    /// Free `count` blocks starting at (page_id, block_offset). In-memory
    /// only; persisted by [`MmapBitmask::persist`] at checkpoint time.
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
        // Rewind the cursor so the freed run is reused before new pages are
        // added; this keeps long-running upsert loops page-local.
        inner.next_fit = inner.next_fit.min(start);
    }

    /// Persist the current allocation state to disk and fsync it. Called at
    /// payload-store checkpoint time, so the full bitmap is written at most
    /// once per checkpoint instead of once per payload write.
    pub fn persist(&self) -> Result<()> {
        let inner = self.inner.read();
        let tmp = self.path.with_extension("bin.tmp");
        let mut file = File::create(&tmp)?;
        Self::write_file(&mut file, &inner)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, &self.path)?;
        if let Some(parent) = self.path.parent() {
            File::open(parent)?.sync_all()?;
        }
        Ok(())
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
            next_fit: 0,
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
            bm.persist().unwrap();
        }
        {
            let bm = MmapBitmask::open(dir.path(), &config).unwrap();
            // Verify the bitmask was persisted correctly by allocating
            // from the reopened state.
            let (pid, _off) = bm.allocate(1, &config).unwrap();
            assert_eq!(pid, 0); // Still within the first page.
        }
    }

    #[test]
    fn allocate_never_straddles_a_page() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(); // 8 blocks per page
        let bm = MmapBitmask::create(dir.path(), &config).unwrap();

        assert_eq!(bm.allocate(6, &config).unwrap(), (0, 0));
        // Page 0 only has blocks 6..8 left: a 4-block request must move to the
        // next page instead of writing past the page end.
        assert_eq!(bm.allocate(4, &config).unwrap(), (1, 0));
        // Whatever is handed out next stays inside one page; next-fit may pick
        // either the skipped tail of page 0 or the tail of page 1.
        let (pid, off) = bm.allocate(2, &config).unwrap();
        assert!(
            off as usize + 2 <= config.blocks_per_page(),
            "run must stay inside page {pid} (offset {off})"
        );
    }

    #[test]
    fn rejects_request_larger_than_a_page() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config();
        let bm = MmapBitmask::create(dir.path(), &config).unwrap();
        let err = bm.allocate(9, &config).expect_err("must reject");
        assert!(err.to_string().contains("exceeds one"), "{err}");
    }

    #[test]
    fn stale_free_bit_is_rebuilt_from_the_live_pointer() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config();
        let pointer = ValuePointer::new(0, 0, 2 * config.block_size as u32);
        {
            let bm = MmapBitmask::create(dir.path(), &config).unwrap();
            bm.allocate(2, &config).unwrap();
            bm.persist().unwrap();
        }
        let path = dir.path().join("bitmask.bin");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[BITMASK_HEADER_LEN] &= !1;
        std::fs::write(&path, bytes).unwrap();

        let bm =
            MmapBitmask::open_or_rebuild(dir.path(), &config, std::iter::once(pointer)).unwrap();
        let (_page, off) = bm.allocate(1, &config).unwrap();
        assert!(off >= 2, "live blocks must not be handed out again");
    }

    #[test]
    fn next_fit_reuses_the_freed_run() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config();
        let bm = MmapBitmask::create(dir.path(), &config).unwrap();
        assert_eq!(bm.allocate(3, &config).unwrap(), (0, 0));
        assert_eq!(bm.allocate(3, &config).unwrap(), (0, 3));
        bm.free_blocks(0, 3, 3 * 128, &config);
        // The cursor rewound to the freed run instead of appending a page.
        assert_eq!(bm.allocate(2, &config).unwrap(), (0, 3));
    }
}
