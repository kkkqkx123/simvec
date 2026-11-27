//! File naming and guard type for segment-local IVF indexes.
//!
//! Each complete dense segment owns one `ivf_seg_<start>.bin` file. Slot
//! assignments inside the file are segment-local; searches add the segment
//! start before reading vectors or returning candidates.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::index::IvfIndex;

/// One published IVF segment.
#[derive(Clone)]
pub(crate) struct SegmentIvf {
    pub start: u32,
    pub end: u32,
    pub index: Arc<IvfIndex>,
}

pub(super) fn file_name(start: u32) -> String {
    format!("ivf_seg_{start:010}.bin")
}

pub(super) fn file_path(dir: &Path, start: u32) -> PathBuf {
    super::segments::segment_dir(dir, start).join(file_name(start))
}

pub(super) fn is_segment_file(name: &str) -> bool {
    name.strip_prefix("ivf_seg_")
        .and_then(|rest| rest.strip_suffix(".bin"))
        .is_some_and(|digits| digits.len() == 10 && digits.bytes().all(|b| b.is_ascii_digit()))
}
