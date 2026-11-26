//! Versioned, checksummed snapshot envelope shared by exact-scan stores.
//!
//! The snapshot is a derived structure: the WAL remains the commit authority.
//! A corrupt or truncated snapshot is ignored on open and the WAL is replayed
//! from the beginning, so a failed checkpoint can never lose committed data.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::{Result, VectorSearchError};

const SNAPSHOT_MAGIC: [u8; 4] = *b"SVSP";
const SNAPSHOT_VERSION: u16 = 1;

/// Write `value` to a temporary file, fsync it, then publish by rename.
pub fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = postcard::to_stdvec(value)?;
    let mut payload = Vec::with_capacity(10 + bytes.len());
    payload.extend_from_slice(&SNAPSHOT_MAGIC);
    payload.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
    payload.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
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
    if file.read_to_end(&mut bytes).is_err() || crc32fast::hash(&bytes) != expected {
        return Ok(None);
    }
    postcard::from_bytes(&bytes)
        .map(Some)
        .map_err(|e| VectorSearchError::CorruptData(e.to_string()))
}
