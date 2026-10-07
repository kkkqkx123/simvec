//! `index.bin` — persisted IVFFlat state (centroids + slot membership).
//!
//! The IVF index is a *derived* structure: it can be rebuilt from
//! `vectors.bin` at any time, so persistence is a pure optimization for open
//! latency. Any validation failure on load discards the file and falls back to
//! exact scan rather than blocking startup.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::types::DistanceMetric;

pub(crate) const INDEX_MAGIC: [u8; 4] = *b"VIVF";
/// Development stage: version locked at 1. No backward compatibility is
/// maintained (see the `FORMAT_VERSION` note in `storage::meta`): older files
/// fail version / checksum validation and are discarded in favor of exact
/// scan, exactly like the pgvector "corrupt index file is deleted and search
/// falls back" contract. The format is CRC32-protected tag
/// `[magic 4][version u16 LE][crc32 u32 LE][postcard payload]`.
pub(crate) const INDEX_VERSION: u16 = 1;
const INDEX_TMP: &str = "index_tmp.bin";

const HNSW_FILE: &str = "hnsw.bin";
const HNSW_MAGIC: [u8; 4] = *b"VHSW";
/// Same dev-stage policy as `INDEX_VERSION`; version locked at 1.
const HNSW_VERSION: u16 = 1;
const HNSW_TMP: &str = "hnsw_tmp.bin";

/// The persisted subset of the IVF index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PersistedIvf {
    pub lists: u32,
    pub dim: usize,
    pub distance: DistanceMetric,
    pub built_at_live_count: u64,
    /// Mean training-sample distance to the nearest centroid; drift baseline.
    pub baseline_mean_dist: f32,
    /// `lists x dim` centroids in list order.
    pub centroids: Vec<Vec<f32>>,
    /// slot -> list assignment; `u32::MAX` = unassigned. May be shorter than
    /// the collection's slot high-water mark (slots appended after the build
    /// are adopted through the pending path on publish/replay).
    pub slot_list: Vec<u32>,
}

impl PersistedIvf {
    /// Structural self-check independent of collection state.
    pub(crate) fn structurally_valid(&self) -> bool {
        self.lists > 0
            && self.dim > 0
            && self.baseline_mean_dist.is_finite()
            && self.baseline_mean_dist >= 0.0
            && self.centroids.len() == self.lists as usize
            && self.centroids.iter().all(|c| c.len() == self.dim)
            && self
                .slot_list
                .iter()
                .all(|&l| l == u32::MAX || (l as usize) < self.centroids.len())
    }

    /// Check against the live collection metadata on open.
    pub(crate) fn valid_for(&self, dim: usize, distance: DistanceMetric, next_slot: u64) -> bool {
        self.structurally_valid()
            && self.dim == dim
            && self.distance == distance
            && self.slot_list.len() as u64 <= next_slot
    }
}

/// Write `data` to `<dir>/index.bin` atomically (temp file + fsync + rename).
pub(crate) fn save(dir: &Path, data: &PersistedIvf) -> Result<()> {
    write_tagged(&dir.join(INDEX_TMP), &INDEX_MAGIC, INDEX_VERSION, data)?;
    std::fs::rename(dir.join(INDEX_TMP), dir.join("index.bin"))?;
    Ok(())
}

/// Load `<dir>/index.bin`. Returns `Ok(None)` when the file is absent or
/// fails any structural check; a failing check also deletes the file so the
/// next save starts clean.
pub(crate) fn load(dir: &Path) -> Result<Option<PersistedIvf>> {
    let path = dir.join("index.bin");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };

    let parsed = read_tagged::<PersistedIvf>(&bytes, &INDEX_MAGIC, INDEX_VERSION);

    match parsed {
        // Any malformed content is treated as absent: the index is a derived
        // structure and must never block startup.
        Ok(data) if data.structurally_valid() => Ok(Some(data)),
        _ => {
            discard(&path);
            Ok(None)
        }
    }
}

pub(crate) fn discard(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// One persisted HNSW node: slot plus its per-layer adjacency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PersistedNodeRecord {
    pub slot: u32,
    pub level: u8,
    /// Version counter for consistency detection. Cycles 1–15 like pgvector's
    /// 4-bit version field. Readers can compare before/after snapshots to
    /// detect concurrent adjacency mutations during iterative scans.
    pub version: u8,
    /// Adjacency for layers `0..=level`; outer index = layer.
    pub neighbors: Vec<Vec<u32>>,
}

/// The persisted subset of the HNSW index (`hnsw.bin`).
///
/// Like the IVF state this is a *derived* structure: any validation failure
/// on load discards the file and falls back to exact scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PersistedHnsw {
    pub dim: usize,
    pub distance: DistanceMetric,
    pub m: usize,
    pub ef_construct: usize,
    pub ef_search: usize,
    /// `(slot, level)` of the graph entry point; `None` = empty graph.
    pub entry: Option<(u32, i32)>,
    pub built_at_live_count: u64,
    /// Dense from slot 0 (compaction renumbers live slots densely and the
    /// insert path appends in ascending order).
    pub nodes: Vec<PersistedNodeRecord>,
}

impl PersistedHnsw {
    /// Structural self-check independent of collection state.
    pub(crate) fn structurally_valid(&self) -> bool {
        if self.dim == 0 || self.m < 2 || self.ef_construct == 0 || self.ef_search == 0 {
            return false;
        }
        let slots: std::collections::HashSet<u32> = self.nodes.iter().map(|n| n.slot).collect();
        self.nodes.iter().all(|n| {
            n.neighbors.len() == n.level as usize + 1
                && n.neighbors.iter().enumerate().all(|(lc, list)| {
                    let cap = if lc == 0 { self.m * 2 } else { self.m };
                    list.len() <= cap && list.iter().all(|&s| slots.contains(&s))
                })
        }) && match self.entry {
            None => true,
            Some((slot, level)) => self
                .nodes
                .iter()
                .any(|n| n.slot == slot && n.level as i32 == level),
        }
    }

    /// Check against the live collection metadata on open. Nodes are dense
    /// from slot 0, so every recorded slot must sit below `next_slot`.
    pub(crate) fn valid_for(&self, dim: usize, distance: DistanceMetric, next_slot: u64) -> bool {
        self.structurally_valid()
            && self.dim == dim
            && self.distance == distance
            && self.nodes.iter().all(|n| (n.slot as u64) < next_slot)
    }
}

/// Write `data` to `<dir>/hnsw.bin` atomically (temp file + fsync + rename).
pub(crate) fn save_hnsw(dir: &Path, data: &PersistedHnsw) -> Result<()> {
    write_tagged(&dir.join(HNSW_TMP), &HNSW_MAGIC, HNSW_VERSION, data)?;
    std::fs::rename(dir.join(HNSW_TMP), dir.join(HNSW_FILE))?;
    Ok(())
}

/// Load `<dir>/hnsw.bin`. Same contract as [`load`]: absent or invalid means
/// `Ok(None)` with the file removed so the next save starts clean.
pub(crate) fn load_hnsw(dir: &Path) -> Result<Option<PersistedHnsw>> {
    let path = dir.join(HNSW_FILE);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };

    let parsed = read_tagged::<PersistedHnsw>(&bytes, &HNSW_MAGIC, HNSW_VERSION);
    match parsed {
        Ok(data) if data.structurally_valid() => Ok(Some(data)),
        _ => {
            discard(&path);
            Ok(None)
        }
    }
}

fn write_tagged<T: serde::Serialize>(
    path: &Path,
    magic: &[u8; 4],
    version: u16,
    data: &T,
) -> Result<()> {
    let bytes = postcard::to_stdvec(data)?;
    let crc = crc32fast::hash(&bytes);
    let mut file = File::create(path)?;
    file.write_all(magic)?;
    file.write_all(&version.to_le_bytes())?;
    file.write_all(&crc.to_le_bytes())?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn read_tagged<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    magic: &[u8; 4],
    version: u16,
) -> Result<T> {
    if bytes.len() < 10 || &bytes[..4] != magic {
        return Err(VectorSearchError::CorruptData("bad magic".to_string()));
    }
    let stored = u16::from_le_bytes([bytes[4], bytes[5]]);
    if stored != version {
        return Err(VectorSearchError::CorruptData(format!(
            "unsupported version {stored}"
        )));
    }
    let expected_crc = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
    let payload = &bytes[10..];
    if crc32fast::hash(payload) != expected_crc {
        return Err(VectorSearchError::CorruptData("crc32 mismatch".to_string()));
    }
    Ok(postcard::from_bytes(payload)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(lists: u32, dim: usize) -> PersistedIvf {
        let centroids = (0..lists).map(|i| vec![i as f32; dim]).collect();
        let slot_list = vec![0u32, 1, u32::MAX, lists - 1];
        PersistedIvf {
            lists,
            dim,
            distance: DistanceMetric::Cosine,
            built_at_live_count: 4,
            baseline_mean_dist: 0.5,
            centroids,
            slot_list,
        }
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let data = sample(3, 4);
        save(dir.path(), &data).unwrap();
        let loaded = load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.lists, 3);
        assert_eq!(loaded.dim, 4);
        assert_eq!(loaded.slot_list.len(), 4);
        assert_eq!(loaded.built_at_live_count, 4);
        assert_eq!(loaded.baseline_mean_dist, 0.5);
    }

    #[test]
    fn test_load_absent_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load(dir.path()).unwrap().is_none());
    }

    #[test]
    fn test_load_corrupt_discards_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("index.bin"), b"XXXXgarbage").unwrap();
        assert!(load(dir.path()).unwrap().is_none());
        assert!(!dir.path().join("index.bin").exists());
    }

    #[test]
    fn test_load_truncated_postcard_discards_file() {
        let dir = tempfile::tempdir().unwrap();
        save(dir.path(), &sample(2, 3)).unwrap();
        let mut bytes = std::fs::read(dir.path().join("index.bin")).unwrap();
        bytes.truncate(10);
        std::fs::write(dir.path().join("index.bin"), &bytes).unwrap();
        assert!(load(dir.path()).unwrap().is_none());
        assert!(!dir.path().join("index.bin").exists());
    }

    #[test]
    fn test_valid_for_checks() {
        let data = sample(2, 3);
        // slot_list covers slots 0..4; valid only if next_slot >= 4.
        assert!(data.valid_for(3, DistanceMetric::Cosine, 4));
        assert!(
            !data.valid_for(4, DistanceMetric::Cosine, 4),
            "dim mismatch"
        );
        assert!(
            !data.valid_for(3, DistanceMetric::Euclid, 4),
            "metric mismatch"
        );
        assert!(
            !data.valid_for(3, DistanceMetric::Cosine, 2),
            "slot_list longer than next_slot"
        );
    }

    #[test]
    fn test_structurally_invalid_rejected_on_save_load() {
        let mut data = sample(2, 3);
        data.centroids[0] = vec![1.0; 9];
        assert!(!data.structurally_valid());
    }

    #[test]
    fn test_payload_bitflip_detected_by_crc() {
        let dir = tempfile::tempdir().unwrap();
        save(dir.path(), &sample(2, 3)).unwrap();
        let mut bytes = std::fs::read(dir.path().join("index.bin")).unwrap();
        assert!(bytes.len() > 10, "payload must exist");
        // Flip an arbitrary payload byte (header is 10 bytes: magic 4 + version 2 + crc 4).
        bytes[10] ^= 0xFF;
        std::fs::write(dir.path().join("index.bin"), &bytes).unwrap();
        assert!(
            load(dir.path()).unwrap().is_none(),
            "CRC mismatch must be treated as corrupt"
        );
        assert!(
            !dir.path().join("index.bin").exists(),
            "corrupt payload file must be deleted"
        );
    }

    #[test]
    fn test_crc_mismatch_on_hnsw_payload() {
        let dir = tempfile::tempdir().unwrap();
        let data = PersistedHnsw {
            dim: 4,
            distance: DistanceMetric::Cosine,
            m: 4,
            ef_construct: 16,
            ef_search: 16,
            entry: None,
            built_at_live_count: 0,
            nodes: vec![],
        };
        save_hnsw(dir.path(), &data).unwrap();
        let mut bytes = std::fs::read(dir.path().join("hnsw.bin")).unwrap();
        assert!(bytes.len() > 10);
        bytes[10] ^= 0xA5;
        std::fs::write(dir.path().join("hnsw.bin"), &bytes).unwrap();
        assert!(load_hnsw(dir.path()).unwrap().is_none());
        assert!(!dir.path().join("hnsw.bin").exists());
    }

    #[test]
    fn test_old_version_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        save(dir.path(), &sample(2, 3)).unwrap();
        let mut bytes = std::fs::read(dir.path().join("index.bin")).unwrap();
        // Flip version from 1 to 2 (bytes 4..6 LE). Keep CRC/payload otherwise valid.
        bytes[4] = 2;
        bytes[5] = 0;
        std::fs::write(dir.path().join("index.bin"), &bytes).unwrap();
        assert!(
            load(dir.path()).unwrap().is_none(),
            "unsupported version must be treated as corrupt"
        );
        assert!(!dir.path().join("index.bin").exists());
    }
}
