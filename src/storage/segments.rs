//! Logical segment view of dense serving files.
//!
//! `vectors.bin` is already mapped segment by segment. This module publishes
//! the boundary between completed segments and the one segment still being
//! appended to, so ANN builds and searches share one definition of what is
//! frozen. The manifest is derived from `meta.bin`; a stale or missing copy
//! is rewritten instead of being treated as authoritative.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::snapshot;
use crate::error::Result;

const SEGMENTS_FILE: &str = "segments.json";
const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    segment_slots: u32,
    next_slot: u64,
}

/// Frozen prefix and appendable tail of one published generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SegmentSet {
    pub immutable_end: u32,
    pub appendable_from: u32,
}

/// Slots below this boundary may be indexed; the rest are scanned exactly.
///
/// A collection that has not filled its first segment has no appendable tail:
/// forcing it to exact scan would make every small collection permanently
/// ineligible for ANN. Once at least one full segment exists, only the final
/// partial segment stays appendable.
pub(crate) fn immutable_end(next_slot: u64, segment_slots: u32) -> u32 {
    if segment_slots == 0 || next_slot == 0 {
        return 0;
    }
    let full = (next_slot / segment_slots as u64) * segment_slots as u64;
    if full == 0 {
        next_slot.min(u32::MAX as u64) as u32
    } else {
        full.min(u32::MAX as u64) as u32
    }
}

impl SegmentSet {
    pub(crate) fn new(next_slot: u64, segment_slots: u32) -> Self {
        let end = immutable_end(next_slot, segment_slots);
        Self {
            immutable_end: end,
            appendable_from: end,
        }
    }
}

/// Publish the derived segment manifest for a generation.
pub(crate) fn write(dir: &Path, segment_slots: u32, next_slot: u64) -> Result<()> {
    snapshot::write(
        &dir.join(SEGMENTS_FILE),
        &Manifest {
            version: FORMAT_VERSION,
            segment_slots,
            next_slot,
        },
    )
}

/// Load the segment manifest, refreshing it when it disagrees with `meta.bin`.
pub(crate) fn read_or_derive(dir: &Path, segment_slots: u32, next_slot: u64) -> Result<SegmentSet> {
    let valid = snapshot::read::<Manifest>(&dir.join(SEGMENTS_FILE))?
        .filter(|m| m.version == FORMAT_VERSION && m.segment_slots == segment_slots);
    if !valid.is_some_and(|m| m.next_slot == next_slot) {
        write(dir, segment_slots, next_slot)?;
    }
    Ok(SegmentSet::new(next_slot, segment_slots))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_keeps_only_the_partial_tail_appendable() {
        assert_eq!(immutable_end(3, 4), 3);
        assert_eq!(immutable_end(4, 4), 4);
        assert_eq!(immutable_end(5, 4), 4);
        assert_eq!(immutable_end(8, 4), 8);
    }

    #[test]
    fn manifest_reports_a_stale_slot_count() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), 4, 5).unwrap();
        let set = read_or_derive(dir.path(), 4, 8).unwrap();
        assert_eq!(set.immutable_end, 8);
        let stored = snapshot::read::<Manifest>(&dir.path().join(SEGMENTS_FILE))
            .unwrap()
            .unwrap();
        assert_eq!(stored.next_slot, 8);
    }
}
