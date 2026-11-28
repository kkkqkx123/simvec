//! Versioned, checksummed snapshot envelope shared by exact-scan stores.
//!
//! The snapshot is a derived structure: the WAL remains the commit authority.
//! A corrupt or truncated snapshot is ignored on open and the WAL is replayed
//! from the beginning, so a failed checkpoint can never lose committed data.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::{Result, VectorSearchError};

const SNAPSHOT_MAGIC: [u8; 4] = *b"SVPC";
const SNAPSHOT_VERSION: u16 = 1;

/// Serialize `value` into a complete 10-byte-envelope payload.
fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let bytes = postcard::to_stdvec(value)?;
    let mut payload = Vec::with_capacity(10 + bytes.len());
    payload.extend_from_slice(&SNAPSHOT_MAGIC);
    payload.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
    payload.extend_from_slice(&crate::checksum::hash(&bytes).to_le_bytes());
    payload.extend_from_slice(&bytes);
    Ok(payload)
}

/// Write `value` to a temporary file, fsync it, then publish by rename.
pub fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let payload = encode(value)?;
    let temp = path.with_extension("tmp");
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&temp)?;
    file.write_all(&payload)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temp, path)?;
    if let Some(parent) = path.parent() {
        let dir = File::open(parent)?;
        dir.sync_all()?;
    }
    Ok(())
}

/// Write the same envelope to `path` **without** renaming it into place.
///
/// Used by records that a later step in the same checkpoint publishes with an
/// explicit rename: the file is durable but invisible to readers until then,
/// which is what lets a flush run outside the queue gate (see the per-segment
/// id tracker in `segments.rs`). The parent directory is not fsynced here —
/// the publish step owns that.
pub fn write_staged<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let payload = encode(value)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    file.write_all(&payload)?;
    file.sync_all()?;
    Ok(())
}

/// Read a published snapshot. Missing or corrupt files read as `None`.
pub fn read<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    let Ok(mut file) = File::open(path) else {
        return Ok(None);
    };
    let mut header = [0u8; 10];
    if file.read_exact(&mut header).is_err() {
        return Ok(None);
    }
    if header[0..4] != SNAPSHOT_MAGIC
        || u16::from_le_bytes(header[4..6].try_into().expect("version slice")) != SNAPSHOT_VERSION
    {
        return Ok(None);
    }
    let expected = u32::from_le_bytes(header[6..10].try_into().expect("crc slice"));
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() || crate::checksum::hash(&bytes) != expected {
        return Ok(None);
    }
    postcard::from_bytes(&bytes)
        .map(Some)
        .map_err(|e| VectorSearchError::CorruptData(e.to_string()))
}

/// Strict variant of [`read`] for commit metadata: a missing file yields
/// `Ok(None)`, but a present file with a bad magic/version/CRC is
/// `CorruptData` instead of being silently ignored. Used by dtype headers,
/// which are authoritative layout metadata rather than derived snapshots.
pub fn read_strict<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    if !path.exists() {
        return Ok(None);
    }
    let Some(value) = read(path)? else {
        return Err(VectorSearchError::CorruptData(format!(
            "corrupt metadata file {}",
            path.display()
        )));
    };
    Ok(Some(value))
}

/// Manifest file name of a copied collection snapshot.
pub const SNAPSHOT_MANIFEST_FILE: &str = "snapshot.json";

pub const SNAPSHOT_FORMAT_VERSION: u32 = 1;

/// One copied file, addressed relative to the snapshot root.
///
/// `version` is the generation that produced the file; WAL files keep the
/// generation they were bound to. Incremental snapshots reuse a base file
/// only when all four fields agree, so a same-length rewrite is never reused.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotFile {
    pub path: String,
    pub version: u64,
    pub len: u64,
    pub crc32: u32,
}

/// Durable description of a point-in-time collection copy.
///
/// `lsn` is the generation replay floor at copy time. The snapshot also
/// carries every surviving WAL segment, so a restore can replay anything the
/// copied generation does not contain without reaching into the live store.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotInfo {
    pub version: u32,
    pub generation: u64,
    pub lsn: u64,
    /// Files linked from a verified base snapshot instead of copied.
    pub reused_files: usize,
    pub files: Vec<SnapshotFile>,
}

/// Publish a snapshot manifest. The envelope gives it the same torn-write
/// protection as the generation manifest.
pub fn write_info(path: &Path, info: &SnapshotInfo) -> Result<()> {
    write(path, info)
}

/// Verify that every file named by a snapshot manifest is present, has the
/// recorded length and matches its CRC. Extra files are allowed so a caller
/// can attach its own metadata without invalidating the snapshot.
pub fn verify_snapshot(dir: &Path) -> Result<SnapshotInfo> {
    let path = dir.join(SNAPSHOT_MANIFEST_FILE);
    let Some(info) = read::<SnapshotInfo>(&path)? else {
        return Err(VectorSearchError::CorruptData(format!(
            "missing or corrupt snapshot manifest {}",
            path.display()
        )));
    };
    if info.version != SNAPSHOT_FORMAT_VERSION {
        return Err(VectorSearchError::CorruptData(format!(
            "unsupported snapshot manifest version {}",
            info.version
        )));
    }
    let generation = super::generation::Generation::open(dir)?;
    if generation.id() != info.generation || generation.lsn() != info.lsn {
        return Err(VectorSearchError::CorruptData(format!(
            "snapshot manifest describes generation {} at lsn {}, but generation.bin says {} at {}",
            info.generation,
            info.lsn,
            generation.id(),
            generation.lsn()
        )));
    }
    for file in &info.files {
        // WAL segments carry their first-lsn as the version, so they remain
        // reusable across generation advances. Every other file is bound to
        // the generation that published it.
        let wal_segment_lsn = wal_segment_version(&file.path);
        if let Some(lsn) = wal_segment_lsn {
            if file.version != lsn {
                return Err(VectorSearchError::CorruptData(format!(
                    "snapshot WAL segment {} is bound to lsn {}, expected {}",
                    file.path, file.version, lsn
                )));
            }
        } else if file.version != info.generation {
            return Err(VectorSearchError::CorruptData(format!(
                "snapshot file {} is bound to generation {}, expected {}",
                file.path, file.version, info.generation
            )));
        }
        let target = checked_path(dir, &file.path)?;
        let (len, crc32) = file_checksum(&target)?;
        if len != file.len || crc32 != file.crc32 {
            return Err(VectorSearchError::CorruptData(format!(
                "snapshot file {} failed verification",
                file.path
            )));
        }
    }
    Ok(info)
}

/// Resolve a manifest path without allowing it to leave `base`.
pub(crate) fn checked_path(base: &Path, rel: &str) -> Result<PathBuf> {
    let path = Path::new(rel);
    let safe = path
        .components()
        .all(|part| matches!(part, Component::Normal(_)));
    if !safe || path.as_os_str().is_empty() {
        return Err(VectorSearchError::CorruptData(format!(
            "snapshot path escapes the snapshot root: {rel}"
        )));
    }
    Ok(base.join(path))
}

/// If `rel` names a WAL segment directly under `wal/`, return its first-lsn
/// (the zero-padded numeric file name). Other paths return `None`.
fn wal_segment_version(rel: &str) -> Option<u64> {
    let name = rel.strip_prefix("wal/")?;
    if name.contains('/') {
        return None;
    }
    name.strip_suffix(".bin")?.parse::<u64>().ok()
}

/// Stream a file's length and CRC without holding the whole file in memory.
pub(crate) fn file_checksum(path: &Path) -> Result<(u64, u32)> {
    let mut file = File::open(path)?;
    let mut hasher = crate::checksum::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut len = 0u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        len += read as u64;
    }
    Ok((len, hasher.finalize()))
}
