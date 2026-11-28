//! Append-only blob file backing the reference-form WAL.
//!
//! The WAL no longer inlines full vectors and JSON payloads. At append time
//! each data-bearing record is encoded into a blob here, fsynced, and the WAL
//! stores only an 8-byte-aligned reference `{offset, length, crc32c, version,
//! kind}`. On replay the blob is read back by offset, its CRC and length are
//! verified, and the original record is reconstructed. A missing or corrupt
//! blob is `CorruptData` — replay never silently falls back to an old value.
//!
//! The file lives beside the WAL segments (`wal/blobs.bin`) so it is copied by
//! the same snapshot window and shares the log's lifetime. It is append-only:
//! a WAL append that fails after a blob was written leaves an unreferenced
//! blob (reclaimable garbage), which is harmless. Full truncation (compaction)
//! resets it; partial truncation (checkpoint) leaves it intact because older
//! references are simply never replayed again.

use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::error::{Result, VectorSearchError};

/// Blob file name inside the WAL directory.
const BLOBS_FILE: &str = "blobs.bin";
/// Record alignment in bytes.
const ALIGN: u64 = 8;
/// Per-blob header: `[crc32c u32][length u32][version u64]`, then a `kind u8`.
const HEADER_LEN: usize = 4 + 4 + 8 + 1;

/// Kind of payload carried by a blob.
pub(crate) mod kind {
    pub const DENSE_UPSERT: u8 = 1;
    pub const VECTORS_UPSERT: u8 = 2;
    pub const NAMED_PATCH: u8 = 3;
    pub const PAYLOAD_SET: u8 = 4;
}

/// Reference to one blob, stored in the WAL record in place of inline data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct BlobRefSpec {
    /// Byte offset of the blob record in `blobs.bin`.
    pub offset: u64,
    /// Length of the encoded payload (excludes header and padding).
    pub length: u32,
    /// CRC32C over header + payload.
    pub crc32c: u32,
    /// Version stamped into the blob header (the assigning LSN).
    pub version: u64,
    /// Payload kind, one of the constants in [`kind`].
    pub kind: u8,
}

/// One blob to append: its kind, version and encoded payload.
pub(crate) struct BlobInput {
    pub kind: u8,
    pub version: u64,
    pub payload: Vec<u8>,
}

fn aligned(len: usize) -> usize {
    (len + ALIGN as usize - 1) / ALIGN as usize * ALIGN as usize
}

/// Append-only blob store. Writes are serialized behind a mutex; reads use
/// positional I/O so replay never disturbs the append cursor.
pub(crate) struct WalBlobs {
    file: File,
}

impl WalBlobs {
    /// Open (or create) the blob file next to the WAL segments.
    pub(crate) fn open_or_create(wal_dir: &Path) -> Result<Self> {
        let path = wal_dir.join(BLOBS_FILE);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        Ok(Self { file })
    }

    /// Append a batch of blobs with a single fsync, returning their refs.
    ///
    /// The caller must invoke this *before* the WAL record that references the
    /// blobs is made durable, so a referenced blob is always already durable.
    pub(crate) fn append_many(&self, inputs: &[BlobInput]) -> Result<Vec<BlobRefSpec>> {
        let mut end = self.file.metadata()?.len();
        let mut refs = Vec::with_capacity(inputs.len());
        let mut buf = Vec::new();
        for input in inputs {
            let payload = input.payload.as_slice();
            let mut framed = Vec::with_capacity(HEADER_LEN + aligned(payload.len()));
            // crc placeholder, patched after the body is known.
            framed.extend_from_slice(&[0u8; 4]);
            framed.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            framed.extend_from_slice(&input.version.to_le_bytes());
            framed.push(input.kind);
            framed.extend_from_slice(payload);
            while framed.len() % ALIGN as usize != 0 {
                framed.push(0);
            }
            let crc = crate::checksum::hash(&framed[4..]);
            framed[0..4].copy_from_slice(&crc.to_le_bytes());
            let offset = end;
            end += framed.len() as u64;
            buf.extend_from_slice(&framed);
            refs.push(BlobRefSpec {
                offset,
                length: payload.len() as u32,
                crc32c: crc,
                version: input.version,
                kind: input.kind,
            });
        }
        if !buf.is_empty() {
            use std::os::unix::fs::FileExt;
            let start = end - buf.len() as u64;
            self.file.write_all_at(&buf, start)?;
            self.file.sync_all()?;
        }
        Ok(refs)
    }

    /// Read and verify a blob, returning its decoded payload.
    pub(crate) fn read(&self, reference: &BlobRefSpec) -> Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
        let mut head = [0u8; HEADER_LEN];
        self.file
            .read_exact_at(&mut head, reference.offset)
            .map_err(|_| {
                VectorSearchError::CorruptData(format!(
                    "blob at offset {} is missing or truncated",
                    reference.offset
                ))
            })?;
        let stored_crc = u32::from_le_bytes(head[0..4].try_into().unwrap());
        let length = u32::from_le_bytes(head[4..8].try_into().unwrap());
        let version = u64::from_le_bytes(head[8..16].try_into().unwrap());
        let kind = head[16];
        if length != reference.length
            || version != reference.version
            || kind != reference.kind
        {
            return Err(VectorSearchError::CorruptData(format!(
                "blob at offset {} has a mismatched header",
                reference.offset
            )));
        }
        let total = aligned(HEADER_LEN + length as usize) as u64;
        if reference.offset + total > self.file.metadata()?.len() {
            return Err(VectorSearchError::CorruptData(format!(
                "blob at offset {} is truncated",
                reference.offset
            )));
        }
        let mut framed = vec![0u8; total as usize];
        self.file
            .read_exact_at(&mut framed, reference.offset)
            .map_err(|_| {
                VectorSearchError::CorruptData(format!(
                    "blob at offset {} is unreadable",
                    reference.offset
                ))
            })?;
        if crate::checksum::hash(&framed[4..]) != stored_crc || stored_crc != reference.crc32c {
            return Err(VectorSearchError::CorruptData(format!(
                "blob at offset {} failed its checksum",
                reference.offset
            )));
        }
        Ok(framed[HEADER_LEN..HEADER_LEN + length as usize].to_vec())
    }

    /// Reset the blob file. Used only by full WAL truncation (test teardown:
    /// a full wipe discards every outstanding reference at once).
    pub(crate) fn reset(&self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.sync_all()?;
        Ok(())
    }

    /// Current length of the blob file.
    ///
    /// Blobs are appended for the lifetime of a log and are only reclaimed by
    /// a full wipe, so this figure grows with every upsert: it is reported
    /// through [`crate::types::MaintenanceStatus`] so the growth is visible
    /// instead of silent.
    pub(crate) fn len(&self) -> u64 {
        self.file.metadata().map(|m| m.len()).unwrap_or(0)
    }
}
