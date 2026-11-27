//! Logical segment view of dense serving files.
//!
//! `vectors.bin` is already mapped segment by segment. This module publishes
//! the boundary between completed segments and the one segment still being
//! appended to, so ANN builds and searches share one definition of what is
//! frozen. The manifest is derived from `meta.bin`; a stale or missing copy
//! is rewritten instead of being treated as authoritative.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::snapshot;
use super::{DataFiles, IdMap, StoreInner};
use crate::error::{Result, VectorSearchError};
use crate::types::PointId;

const SEGMENTS_FILE: &str = "segments.json";
const FORMAT_VERSION: u32 = 1;

/// Per-segment persisted accounting. Derived from `meta.bin` plus the
/// tombstone table; the manifest is never authoritative over them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub(crate) struct SegmentEntry {
    /// First slot of the segment.
    pub start: u32,
    /// One past the last slot covered by this entry.
    pub end: u32,
    /// Live slots in `[start, end)`.
    pub live_count: u64,
    /// Tombstoned slots in `[start, end)`.
    pub tombstone_count: u64,
    /// Highest LSN known durable for this segment at publish time.
    pub persisted_lsn: u64,
}

impl SegmentEntry {
    /// Fraction of garbage in the segment, for compaction planning.
    pub(crate) fn garbage_ratio(&self) -> f64 {
        let total = self.live_count + self.tombstone_count;
        if total == 0 {
            return 0.0;
        }
        self.tombstone_count as f64 / total as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    segment_slots: u32,
    next_slot: u64,
    #[serde(default)]
    segments: Vec<SegmentEntry>,
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

/// Directory holding one physical segment. Creating it on demand lets a
/// newly appended segment own all of its files without a generation-wide
/// rename.
pub(crate) fn segment_dir(generation: &Path, start: u32) -> PathBuf {
    let dir = segment_path(generation, start);
    if let Err(err) = std::fs::create_dir_all(&dir) {
        tracing::warn!(error = %err, path = %dir.display(), "cannot create segment directory");
    }
    dir
}

/// Path of one physical segment without creating it. Read paths use this so
/// opening a store never mutates the directory it is inspecting.
pub(crate) fn segment_path(generation: &Path, start: u32) -> PathBuf {
    generation.join("segments").join(format!("{start:010}"))
}

/// Every published physical segment directory of a generation.
pub(crate) fn segment_dirs(generation: &Path) -> Vec<PathBuf> {
    let root = generation.join("segments");
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    dirs
}

/// Which segment a slot belongs to.
pub(crate) fn segment_index(slot: u64, segment_slots: u32) -> usize {
    if segment_slots == 0 {
        return 0;
    }
    (slot / segment_slots as u64) as usize
}

/// Slot range covered by one segment index, clipped to `next_slot`.
pub(crate) fn segment_range(index: usize, segment_slots: u32, next_slot: u64) -> (u32, u32) {
    if segment_slots == 0 {
        return (0, next_slot.min(u32::MAX as u64) as u32);
    }
    let start = index as u64 * segment_slots as u64;
    let end = ((index as u64 + 1) * segment_slots as u64).min(next_slot);
    (
        start.min(u32::MAX as u64) as u32,
        end.min(u32::MAX as u64) as u32,
    )
}

/// Derive per-segment live/tombstone counts by scanning the tombstone table.
///
/// The caller passes a slot predicate so this module stays decoupled from the
/// tombstone representation; production passes the live tombstone bits.
pub(crate) fn derive_entries(
    next_slot: u64,
    segment_slots: u32,
    mut is_tombstoned: impl FnMut(usize) -> bool,
    persisted_lsn: u64,
) -> Vec<SegmentEntry> {
    if segment_slots == 0 || next_slot == 0 {
        return Vec::new();
    }
    let count = next_slot.div_ceil(segment_slots as u64) as usize;
    let mut entries = Vec::with_capacity(count);
    for index in 0..count {
        let (start, end) = segment_range(index, segment_slots, next_slot);
        let mut live = 0u64;
        let mut tomb = 0u64;
        for slot in start..end {
            if is_tombstoned(slot as usize) {
                tomb += 1;
            } else {
                live += 1;
            }
        }
        entries.push(SegmentEntry {
            start,
            end,
            live_count: live,
            tombstone_count: tomb,
            persisted_lsn,
        });
    }
    entries
}

/// Segments whose garbage ratio meets `threshold`. The appendable tail is
/// never selected: it is still receiving writes, so compacting it would race
/// the write path without reclaiming stable garbage.
pub(crate) fn select_for_compaction(entries: &[SegmentEntry], threshold: f64) -> Vec<u32> {
    if entries.is_empty() {
        return Vec::new();
    }
    let appendable = entries.last().map(|e| e.start).unwrap_or(0);
    entries
        .iter()
        .filter(|e| e.start != appendable || entries.len() == 1)
        .filter(|e| e.garbage_ratio() >= threshold && e.tombstone_count > 0)
        .map(|e| e.start)
        .collect()
}

/// One external id's authoritative placement inside a segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct IdTrackerEntry {
    pub id: String,
    pub offset: u32,
    pub point_lsn: u64,
    pub delete_lsn: u64,
    pub free: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct IdTracker {
    generation: u64,
    entries: Vec<IdTrackerEntry>,
}

/// Persist one segment's id tracker after its data files are durable.
///
/// `generation` is the id of the generation the tracker belongs to; binding it
/// (rather than a slot water mark) lets open reject a tracker that was left
/// behind by an aborted staging attempt.
pub(super) fn write_id_tracker(
    data: &DataFiles,
    start: u32,
    end: u32,
    inner: &StoreInner,
    generation: u64,
) -> Result<()> {
    let keys = data.keys.snapshot();
    let tombstones = data.tombstones.load();
    let mut entries = Vec::new();
    for offset in 0..(end - start) {
        let slot = start + offset;
        let Ok(Some(id_text)) = super::keys::Keys::read_key(&keys, slot as usize) else {
            continue;
        };
        let id = PointId::from(id_text.as_str());
        let live = inner.reverse.get(&id) == Some(slot);
        let deleted = tombstones.bit(slot as usize);
        let version = inner.versions.get_value(&id).unwrap_or(0);
        entries.push(IdTrackerEntry {
            id: id_text,
            offset,
            point_lsn: if live { version } else { 0 },
            delete_lsn: if deleted { version } else { 0 },
            free: !live || deleted,
        });
    }
    snapshot::write(
        &segment_dir(data.dir(), start).join(ID_TRACKER_FILE),
        &IdTracker {
            generation,
            entries,
        },
    )
}

/// File name of a segment's authoritative id directory.
pub(crate) const ID_TRACKER_FILE: &str = "id_tracker.bin";

/// One point's directory entry handed to a tracker writer (id, global slot,
/// point lsn, delete lsn, free).
pub(crate) type TrackerPoint = (String, u32, u64, u64, bool);

/// Write id trackers for a freshly built generation from an explicit list of
/// points, grouped by their segment. Used by compaction, which knows the final
/// slot of every live point but has no live `StoreInner` to read from.
pub(crate) fn write_trackers_from_points(
    dir: &Path,
    generation: u64,
    segment_slots: u32,
    points: &[TrackerPoint],
) -> Result<()> {
    if segment_slots == 0 {
        return Ok(());
    }
    let mut grouped: BTreeMap<u32, Vec<IdTrackerEntry>> = BTreeMap::new();
    for (id, slot, point_lsn, delete_lsn, free) in points {
        let start = (*slot / segment_slots) * segment_slots;
        grouped.entry(start).or_default().push(IdTrackerEntry {
            id: id.clone(),
            offset: *slot - start,
            point_lsn: *point_lsn,
            delete_lsn: *delete_lsn,
            free: *free,
        });
    }
    for (start, entries) in grouped {
        snapshot::write(
            &segment_dir(dir, start).join(ID_TRACKER_FILE),
            &IdTracker {
                generation,
                entries,
            },
        )?;
    }
    Ok(())
}

/// Reconstruct the id->slot directory and per-id water marks from the
/// per-segment id trackers.
///
/// Trackers are the authoritative directory of a published generation, replacing
/// the old whole-generation snapshot. Every physical segment must carry one:
/// a missing or mismatched tracker is `CorruptData` because the id directory
/// cannot be safely reconstructed from slot files alone (tombstoned slots keep
/// their key, so a scan cannot tell which id currently owns a live slot).
/// Returns `None` only for a brand-new generation that has never checkpointed
/// (no segment tracker exists yet); the caller then rebuilds from WAL replay.
pub(crate) fn read_id_trackers(
    dir: &Path,
    generation: u64,
    segment_slots: u32,
    slot_capacity: u64,
) -> Result<Option<(IdMap, IdMap<u64>)>> {
    if segment_slots == 0 {
        return Ok(None);
    }
    let count = slot_capacity.div_ceil(segment_slots as u64) as usize;
    let mut reverse = IdMap::with_capacity(slot_capacity as usize);
    let mut versions = IdMap::<u64>::default();
    let mut found = false;
    for index in 0..count {
        let start = index as u32 * segment_slots;
        let path = segment_path(dir, start).join(ID_TRACKER_FILE);
        if !path.exists() {
            // A segment that never received a checkpoint has no tracker;
            // its slots are rebuilt from the WAL.
            continue;
        }
        // The file exists but failed the CRC/format envelope: a committed
        // id directory is not reconstructable from slot files, so this is
        // corruption, not a silent fallback to WAL replay.
        let tracker = snapshot::read::<IdTracker>(&path)?.ok_or_else(|| {
            VectorSearchError::CorruptData(format!(
                "id tracker in segment {start} failed its checksum"
            ))
        })?;
        if tracker.generation != generation {
            return Err(VectorSearchError::CorruptData(format!(
                "id tracker in segment {start} belongs to generation {}, expected {generation}",
                tracker.generation
            )));
        }
        found = true;
        for entry in tracker.entries {
            let id = PointId::from(entry.id.as_str());
            let version = entry.point_lsn.max(entry.delete_lsn);
            versions.insert_value(&id, version);
            if entry.free {
                if entry.delete_lsn > 0 {
                    versions.insert_value(&id, entry.delete_lsn);
                }
                continue;
            }
            reverse.insert(&id, start + entry.offset);
        }
    }
    Ok(if found {
        Some((reverse, versions))
    } else {
        None
    })
}

/// Publish the derived segment manifest for a generation.
pub(crate) fn write(dir: &Path, segment_slots: u32, next_slot: u64) -> Result<()> {
    snapshot::write(
        &dir.join(SEGMENTS_FILE),
        &Manifest {
            version: FORMAT_VERSION,
            segment_slots,
            next_slot,
            segments: Vec::new(),
        },
    )
}

/// Publish the manifest together with per-segment accounting.
pub(crate) fn write_with_entries(
    dir: &Path,
    segment_slots: u32,
    next_slot: u64,
    entries: &[SegmentEntry],
) -> Result<()> {
    snapshot::write(
        &dir.join(SEGMENTS_FILE),
        &Manifest {
            version: FORMAT_VERSION,
            segment_slots,
            next_slot,
            segments: entries.to_vec(),
        },
    )
}

/// Load persisted per-segment accounting, if the manifest carries current data.
pub(crate) fn read_entries(dir: &Path, segment_slots: u32, next_slot: u64) -> Vec<SegmentEntry> {
    let Ok(Some(manifest)) = snapshot::read::<Manifest>(&dir.join(SEGMENTS_FILE)) else {
        return Vec::new();
    };
    if manifest.version != FORMAT_VERSION
        || manifest.segment_slots != segment_slots
        || manifest.next_slot != next_slot
    {
        return Vec::new();
    }
    manifest.segments
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

    #[test]
    fn entries_split_slots_and_report_garbage() {
        let tomb = [false, true, false, false, true, true, false, false];
        let entries = derive_entries(8, 4, |s| tomb[s], 9);
        assert_eq!(entries.len(), 2);
        assert_eq!((entries[0].start, entries[0].end), (0, 4));
        assert_eq!(entries[0].live_count, 3);
        assert_eq!(entries[0].tombstone_count, 1);
        assert_eq!((entries[1].start, entries[1].end), (4, 8));
        assert_eq!(entries[1].live_count, 2);
        assert_eq!(entries[1].tombstone_count, 2);
        let picked = select_for_compaction(&entries, 0.20);
        assert_eq!(picked, vec![0]);
    }

    #[test]
    fn entries_round_trip_through_the_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let entries = derive_entries(4, 4, |_| false, 7);
        write_with_entries(dir.path(), 4, 4, &entries).unwrap();
        let loaded = read_entries(dir.path(), 4, 4);
        assert_eq!(loaded, entries);
        assert!(read_entries(dir.path(), 4, 8).is_empty());
    }
}
