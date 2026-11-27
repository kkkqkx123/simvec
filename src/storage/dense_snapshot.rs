//! Checkpointed directory state of a published dense generation.
//!
//! `meta.bin` describes the shape of the files but rebuilding the id
//! directory from them means scanning every slot, every payload pointer and
//! every tombstone. This snapshot carries the directory itself, the per-id
//! water marks and the slot geometry at the generation's lsn, so opening a
//! checkpointed collection only replays the tail of the log.
//!
//! The snapshot is written before the generation manifest advances and is
//! usable only when its generation id and lsn both match the manifest. A
//! crash between those two writes leaves a snapshot ahead of the manifest;
//! open ignores it, scans the files and replays from the manifest, whose WAL
//! prefix is still intact.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::snapshot;
use super::{Generation, IdMap, Meta, TombstoneBits};
use crate::error::Result;
use crate::types::PointId;

pub(crate) const FILE_NAME: &str = "dense_snapshot.bin";

/// One id's directory entry. Deleted ids keep their water mark so an old
/// upsert from a resurrected WAL segment cannot bring them back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SnapshotPoint {
    pub id: String,
    pub slot: u32,
    pub lsn: u64,
    pub deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DenseSnapshot {
    pub generation: u64,
    pub lsn: u64,
    pub next_slot: u64,
    pub slot_capacity: u64,
    pub live_count: u64,
    pub tombstone_count: u64,
    pub points: Vec<SnapshotPoint>,
    pub tombstones: Vec<u32>,
}

impl DenseSnapshot {
    /// Snapshot the id directory and its water marks.
    pub(crate) fn capture_points(reverse: &IdMap, versions: &IdMap<u64>) -> Vec<SnapshotPoint> {
        versions
            .iter()
            .map(|(id, version)| match reverse.get(&id) {
                Some(slot) => SnapshotPoint {
                    id: id.to_string(),
                    slot,
                    lsn: version,
                    deleted: false,
                },
                None => SnapshotPoint {
                    id: id.to_string(),
                    slot: 0,
                    lsn: version,
                    deleted: true,
                },
            })
            .collect()
    }

    /// Capture the directory from immutable store state.
    pub(crate) fn capture(
        generation: u64,
        lsn: u64,
        meta: &Meta,
        reverse: &IdMap,
        versions: &IdMap<u64>,
        tombstones: &TombstoneBits,
    ) -> Self {
        let points = Self::capture_points(reverse, versions);
        let tombstone_slots = (0..meta.next_slot as usize)
            .filter(|&slot| tombstones.bit(slot))
            .map(|slot| slot as u32)
            .collect();
        Self {
            generation,
            lsn,
            next_slot: meta.next_slot,
            slot_capacity: meta.slot_capacity,
            live_count: meta.live_count,
            tombstone_count: meta.tombstone_count,
            points,
            tombstones: tombstone_slots,
        }
    }

    /// Whether this snapshot describes exactly the published generation.
    ///
    /// A mismatch means the snapshot belongs to an unpublished checkpoint or
    /// a different generation. It is safe to ignore: the manifest's lsn still
    /// bounds WAL replay and the files can always be scanned instead.
    pub(crate) fn matches(
        &self,
        generation: &Generation,
        meta: &Meta,
        tombstone_count: u64,
    ) -> bool {
        if self.generation != generation.id()
            || self.lsn != generation.lsn()
            || self.next_slot != meta.next_slot
            || self.slot_capacity != meta.slot_capacity
            || self.live_count != meta.live_count
            || self.tombstone_count != tombstone_count
            || self.tombstones.len() as u64 != tombstone_count
        {
            return false;
        }
        self.points.iter().all(|point| {
            point.lsn <= self.lsn
                && if point.deleted {
                    true
                } else {
                    (point.slot as u64) < self.next_slot
                }
        })
    }

    pub(crate) fn write(dir: &Path, value: &Self) -> Result<()> {
        snapshot::write(&dir.join(FILE_NAME), value)
    }

    pub(crate) fn read(dir: &Path) -> Result<Option<Self>> {
        snapshot::read(&dir.join(FILE_NAME))
    }

    pub(crate) fn point_id(point: &SnapshotPoint) -> PointId {
        PointId::from(point.id.as_str())
    }
}
