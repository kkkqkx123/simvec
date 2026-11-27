//! Per-segment HNSW files for dense collections.
//!
//! Each completed segment owns one graph whose node ids are local to that
//! segment. A missing or corrupt file only removes that segment's ANN
//! acceleration; its points stay searchable through the exact-scan fallback.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

use crate::index::HnswIndex;

/// One published segment graph.
#[derive(Clone)]
pub(super) struct SegmentHnsw {
    pub start: u32,
    pub end: u32,
    pub index: Arc<HnswIndex>,
    /// Codes for this segment, indexed by segment-local slot. `None` means
    /// the graph is traversed with exact vectors.
    pub quant: Option<Arc<Vec<Arc<Mmap>>>>,
}

/// File holding the graph of the segment starting at `start`.
pub(crate) fn file_name(start: u32) -> String {
    format!("hnsw_seg_{start:010}.bin")
}

pub(crate) fn file_path(dir: &Path, start: u32) -> PathBuf {
    dir.join(file_name(start))
}

pub(super) fn is_segment_file(name: &str) -> bool {
    name.starts_with("hnsw_seg_") && name.ends_with(".bin")
}
