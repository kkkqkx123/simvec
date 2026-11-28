//! Persisted ANN state (per-segment IVF centroids + slot membership).
//!
//! Indexes are *derived* structures: they can be rebuilt from `vectors.bin`
//! at any time, so persistence is a pure optimization for open latency. Any
//! validation failure on load discards the file and falls back to exact scan
//! rather than blocking startup. One file is written per completed dense
//! segment (`hnsw_seg_<start>.bin` / `ivf_seg_<start>.bin` layout); there is
//! no whole-collection index file.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::types::DistanceMetric;

pub(crate) const INDEX_MAGIC: [u8; 4] = *b"VICF";
/// Index file format version. The file is a CRC32-protected tag
/// `[magic 4][version u16 LE][crc32 u32 LE][postcard payload]`; a file whose
/// version or checksum does not match is discarded and search falls back to
/// exact scan.
pub(crate) const INDEX_VERSION: u16 = 1;
const HNSW_MAGIC: [u8; 4] = *b"VHSC";
/// HNSW graph file format version.
const HNSW_VERSION: u16 = 1;

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
}

/// Write a single segment's IVF state to an explicit path.
pub(crate) fn save_at(path: &Path, data: &PersistedIvf) -> Result<()> {
    let tmp = path.with_extension("tmp.bin");
    write_tagged(&tmp, &INDEX_MAGIC, INDEX_VERSION, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Load one segment's IVF state. Derived-file failures become `None` so the
/// affected segment falls back to exact scan without blocking open.
pub(crate) fn load_at(path: &Path) -> Result<Option<PersistedIvf>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    match read_tagged::<PersistedIvf>(&bytes, &INDEX_MAGIC, INDEX_VERSION) {
        Ok(data) if data.structurally_valid() => Ok(Some(data)),
        _ => {
            discard(path);
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

/// The persisted subset of one HNSW segment graph.
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
    /// Dense from slot 0 (segment rebuilds number live slots densely and the
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
                    list.len() <= cap && list.iter().all(|s| slots.contains(s))
                })
        }) && match self.entry {
            None => true,
            Some((slot, level)) => self
                .nodes
                .iter()
                .any(|n| n.slot == slot && n.level as i32 == level),
        }
    }
}

/// Write one segment graph to its own atomic file.
pub(crate) fn save_hnsw_at(path: &Path, data: &PersistedHnsw) -> Result<()> {
    let tmp = path.with_extension("tmp");
    write_tagged(&tmp, &HNSW_MAGIC, HNSW_VERSION, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Load one segment graph. Damage is reported as absent so that only the
/// affected segment falls back to exact scan.
pub(crate) fn load_hnsw_at(path: &Path) -> Result<Option<PersistedHnsw>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    match read_tagged::<PersistedHnsw>(&bytes, &HNSW_MAGIC, HNSW_VERSION) {
        Ok(data) if data.structurally_valid() => Ok(Some(data)),
        _ => {
            discard(path);
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
    let crc = crate::checksum::hash(&bytes);
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
    if crate::checksum::hash(payload) != expected_crc {
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

    fn segment_path(dir: &tempfile::TempDir, name: &str) -> std::path::PathBuf {
        dir.path().join(name)
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let data = sample(3, 4);
        let path = segment_path(&dir, "ivf_seg_0000000000.bin");
        save_at(&path, &data).unwrap();
        let loaded = load_at(&path).unwrap().unwrap();
        assert_eq!(loaded.lists, 3);
        assert_eq!(loaded.dim, 4);
        assert_eq!(loaded.slot_list.len(), 4);
        assert_eq!(loaded.built_at_live_count, 4);
        assert_eq!(loaded.baseline_mean_dist, 0.5);
    }

    #[test]
    fn test_load_absent_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir, "ivf_seg_0000000000.bin");
        assert!(load_at(&path).unwrap().is_none());
    }

    #[test]
    fn test_load_corrupt_discards_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir, "ivf_seg_0000000000.bin");
        std::fs::write(&path, b"XXXXgarbage").unwrap();
        assert!(load_at(&path).unwrap().is_none());
        assert!(!path.exists());
    }

    #[test]
    fn test_load_truncated_postcard_discards_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir, "ivf_seg_0000000000.bin");
        save_at(&path, &sample(2, 3)).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(10);
        std::fs::write(&path, &bytes).unwrap();
        assert!(load_at(&path).unwrap().is_none());
        assert!(!path.exists());
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
        let path = segment_path(&dir, "ivf_seg_0000000000.bin");
        save_at(&path, &sample(2, 3)).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 10, "payload must exist");
        // Flip an arbitrary payload byte (header is 10 bytes: magic 4 + version 2 + crc 4).
        bytes[10] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            load_at(&path).unwrap().is_none(),
            "CRC mismatch must be treated as corrupt"
        );
        assert!(!path.exists(), "corrupt payload file must be deleted");
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
        let path = segment_path(&dir, "hnsw_seg_0000000000.bin");
        save_hnsw_at(&path, &data).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 10);
        bytes[10] ^= 0xA5;
        std::fs::write(&path, &bytes).unwrap();
        assert!(load_hnsw_at(&path).unwrap().is_none());
        assert!(!path.exists());
    }

    #[test]
    fn test_old_version_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir, "ivf_seg_0000000000.bin");
        save_at(&path, &sample(2, 3)).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        // Flip version from 1 to 2 (bytes 4..6 LE). Keep CRC/payload otherwise valid.
        bytes[4] = 2;
        bytes[5] = 0;
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            load_at(&path).unwrap().is_none(),
            "unsupported version must be treated as corrupt"
        );
        assert!(!path.exists());
    }
}
