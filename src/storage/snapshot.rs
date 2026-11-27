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

/// Write `value` to a temporary file, fsync it, then publish by rename.
pub fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = postcard::to_stdvec(value)?;
    let mut payload = Vec::with_capacity(10 + bytes.len());
    payload.extend_from_slice(&SNAPSHOT_MAGIC);
    payload.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
    payload.extend_from_slice(&crate::checksum::hash(&bytes).to_le_bytes());
    payload.extend_from_slice(&bytes);

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
        if file.version != info.generation {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_crc32_envelope_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.bin");
        let body = postcard::to_stdvec(&42u32).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"SVSP");
        bytes.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&crc32fast::hash(&body).to_le_bytes());
        bytes.extend_from_slice(&body);
        std::fs::write(&path, bytes).unwrap();
        assert!(read::<u32>(&path).unwrap().is_none());
    }
}
