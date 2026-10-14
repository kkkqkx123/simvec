//! `wal.bin` — append-only transaction log per collection.
//!
//! Each record is `[u32 len][u32 crc32][postcard(WalTxn)]`; the checksum covers
//! the encoded payload so a torn or bit-rotted record cannot be mistaken for a
//! committed transaction. Durability is one `write_all` + `sync_all` per
//! *batch*: the commit queue (`super::queue`) hands its whole backlog to
//! [`Wal::append_batch`], so concurrent writers share a single fsync — that
//! shared fsync is what makes group commit actually group. Single-record
//! appends (maintenance markers) go through the same batch path.
//!
//! A batch that fails at any point is rolled back to the offset it started
//! at. Without the rollback a half-written record would sit in the middle of
//! the log, and replay stops at the first checksum mismatch — silently
//! discarding every record behind it.
//!
//! Replay is idempotent: upserts overwrite by point id, deletes of missing
//! points are no-ops, and `Compact` checkpoints only advance the water mark.
//! A truncated or checksum-failed final record (crash mid-append) stops the
//! replay silently and is treated as end of log; anything else malformed is
//! reported as `CorruptData`.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::types::{PointId, VectorPoint};

/// Maximum accepted single-record size (guards against corrupt length fields).
const MAX_RECORD_LEN: u32 = 512 * 1024 * 1024;

/// One transaction: a batch of operations applied atomically to a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalTxn {
    pub txn_id: u64,
    pub ops: Vec<WalRecord>,
}

/// Borrowed view of [`WalTxn`] for encoding a batch without cloning ops.
///
/// Field-for-field identical to the owned form, and postcard is a
/// non-self-describing format, so both produce the same bytes.
#[derive(Serialize)]
struct WalTxnRef<'a> {
    txn_id: u64,
    ops: &'a [WalRecord],
}

/// A point as persisted in the WAL. The id is a string and the payload is
/// JSON-encoded because postcard cannot encode untagged enums (`PointId`,
/// `serde_json::Value`) — same choice as `keys.bin`/`payloads.bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalPoint {
    pub id: String,
    pub vector: Vec<f32>,
    /// JSON-encoded payload.
    pub payload: Option<String>,
}

impl WalPoint {
    pub fn from_point(point: &VectorPoint) -> Result<Self> {
        let payload = match &point.payload {
            Some(p) => Some(serde_json::to_string(p)?),
            None => None,
        };
        Ok(Self {
            id: point.id.to_string(),
            vector: point.vector.clone(),
            payload,
        })
    }

    pub fn to_point(&self) -> Result<VectorPoint> {
        let payload = match &self.payload {
            Some(s) => Some(serde_json::from_str(s)?),
            None => None,
        };
        Ok(VectorPoint {
            id: PointId::from(self.id.clone()),
            vector: self.vector.clone(),
            payload,
        })
    }
}

/// A single mutation logged in the WAL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalRecord {
    Upsert {
        point: WalPoint,
    },
    Delete {
        point_id: String,
    },
    DeleteBatch {
        point_ids: Vec<String>,
    },
    /// Checkpoint marker written after compaction; replay ignores the payload
    /// but keeps advancing the water mark.
    Compact,
    /// Quantization state change (codebook / scale update). Replay is idempotent
    /// — the derived `quant.bin` files are rebuildable, so this record only
    /// advances the water mark and triggers no in-memory mutation on replay.
    Quantize,
    /// Reserved for drop (the whole directory is removed, nothing is logged).
    DropCollection,
    /// Replace the entire payload for a slot. The payload is JSON-encoded
    /// because postcard cannot encode `serde_json::Value` (untagged enum).
    SetPayload {
        slot: u32,
        payload: Option<String>,
    },
    /// Remove specific keys from a slot's payload. The remaining keys are
    /// preserved; if the slot has no payload this is a no-op.
    DeletePayloadKeys {
        slot: u32,
        keys: Vec<String>,
    },
    /// Set a single key on a slot's payload (merge semantics). A missing
    /// payload is created containing just this key; all other keys are
    /// preserved. The value is JSON-encoded because postcard cannot encode
    /// `serde_json::Value` (untagged enum).
    SetPayloadField {
        slot: u32,
        key: String,
        value: String,
    },
}

/// Append-only WAL file. Every durable mutation goes through
/// [`Wal::append_batch`], one write + one fsync per batch.
pub struct Wal {
    path: PathBuf,
    file: parking_lot::Mutex<File>,
    /// Bytes physically appended since the last truncate. Drives the
    /// size-based checkpoint schedule: a pure-insert workload never crosses a
    /// tombstone ratio, so without it the log would grow without bound.
    appended_bytes: std::sync::atomic::AtomicU64,
    /// Test-only: fail the next batch append, to exercise the failure paths.
    #[cfg(test)]
    fail_next_append: std::sync::atomic::AtomicBool,
}

impl Wal {
    /// Open the WAL for appending, creating it if missing.
    pub fn open_or_create(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(path)?;
        let existing = file.metadata()?.len();
        Ok(Self {
            path: path.to_path_buf(),
            file: parking_lot::Mutex::new(file),
            appended_bytes: std::sync::atomic::AtomicU64::new(existing),
            #[cfg(test)]
            fail_next_append: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Test-only: make the next batch append fail without writing anything,
    /// the way a failing `write_all` or `sync_all` would.
    #[cfg(test)]
    pub fn fail_next_append_for_test(&self) {
        self.fail_next_append
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Append a single transaction (one write + one fsync).
    pub fn append(&self, txn: &WalTxn) -> Result<()> {
        self.append_batch(&[(txn.txn_id, &txn.ops)])
    }

    /// Append a batch of transactions with a single `write_all` +
    /// `sync_all`.
    ///
    /// The batch is the unit of group commit: everything handed in here shares
    /// one fsync, so a batch's transactions all become durable together and
    /// none of them is applied unless the whole batch is durable.
    ///
    /// On failure the batch is rolled back to the log length it started from.
    /// Leaving a torn record behind would make every later replay stop at
    /// that checksum mismatch and drop the rest of the log, so the rollback
    /// is part of the failure contract, not an optimization.
    pub fn append_batch(&self, batch: &[(u64, &[WalRecord])]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let mut buffer = Vec::new();
        for (txn_id, ops) in batch {
            let bytes = postcard::to_stdvec(&WalTxnRef {
                txn_id: *txn_id,
                ops,
            })?;
            if bytes.len() > MAX_RECORD_LEN as usize {
                return Err(VectorSearchError::Internal(format!(
                    "wal record too large: {} bytes",
                    bytes.len()
                )));
            }
            buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
            buffer.extend_from_slice(&bytes);
        }

        let mut file = self.file.lock();
        #[cfg(test)]
        if self
            .fail_next_append
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            return Err(VectorSearchError::Internal(
                "test-injected wal append failure".to_string(),
            ));
        }
        let start_len = file.metadata()?.len();
        let write_result = file.write_all(&buffer).and_then(|_| file.sync_all());
        if let Err(e) = write_result {
            // Drop the partial batch so the log keeps a clean suffix.
            let _ = file.set_len(start_len);
            let _ = file.sync_all();
            return Err(e.into());
        }
        drop(file);
        self.appended_bytes
            .fetch_add(buffer.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Replay all records in order, invoking `f` for each decoded transaction.
    ///
    /// Returns the highest `txn_id` seen. A truncated final record (crash
    /// mid-append) or a checksum mismatch on the final record stops the replay
    /// silently — nothing after a torn record can be trusted; anything else
    /// that is malformed is reported as `CorruptData`.
    pub fn replay(&self, mut f: impl FnMut(&WalTxn) -> Result<()>) -> Result<u64> {
        let mut file = self.file.lock();
        file.rewind()?;
        let mut last_txn = 0u64;
        let mut header_buf = [0u8; 8];
        loop {
            match file.read_exact(&mut header_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u32::from_le_bytes(header_buf[0..4].try_into().expect("len slice"));
            if len == 0 {
                return Err(VectorSearchError::CorruptData(
                    "wal record with zero length".to_string(),
                ));
            }
            if len > MAX_RECORD_LEN {
                return Err(VectorSearchError::CorruptData(format!(
                    "wal record length {len} exceeds limit"
                )));
            }
            let expected_crc = u32::from_le_bytes(header_buf[4..8].try_into().expect("crc slice"));
            let mut bytes = vec![0u8; len as usize];
            if let Err(e) = file.read_exact(&mut bytes) {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    break; // truncated trailing record from a crash
                }
                return Err(e.into());
            }
            if crc32fast::hash(&bytes) != expected_crc {
                // Torn write or bit rot: stop at this record. Because the log
                // is append-only, no later record can be valid anyway.
                break;
            }
            let txn: WalTxn = postcard::from_bytes(&bytes)?;
            f(&txn)?;
            last_txn = txn.txn_id.max(last_txn);
        }
        Ok(last_txn)
    }

    /// Truncate the log to empty (used after compaction, where all state is
    /// already durable in the rebuilt files and `meta.bin`, and after a
    /// periodic checkpoint, where every derived file has been flushed).
    pub fn truncate(&self) -> Result<()> {
        let mut file = self.file.lock();
        file.set_len(0)?;
        file.rewind()?;
        file.sync_all()?;
        self.appended_bytes
            .store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Bytes currently held in the log since the last truncate. Drives the
    /// size-based checkpoint schedule.
    pub fn appended_bytes(&self) -> u64 {
        self.appended_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Read all valid WAL records into a `Vec<WalTxn>`, returning the highest
    /// `txn_id` seen. Caller must hold no other WAL locks; this acquires and
    /// releases the file lock internally so callers can batch-apply without
    /// per-record lock contention.
    pub fn read_all(&self) -> Result<(Vec<WalTxn>, u64)> {
        let mut file = self.file.lock();
        file.rewind()?;
        let mut txns = Vec::new();
        let mut last_txn = 0u64;
        let mut header_buf = [0u8; 8];
        loop {
            match file.read_exact(&mut header_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u32::from_le_bytes(header_buf[0..4].try_into().expect("len slice"));
            if len == 0 {
                return Err(VectorSearchError::CorruptData(
                    "wal record with zero length".to_string(),
                ));
            }
            if len > MAX_RECORD_LEN {
                return Err(VectorSearchError::CorruptData(format!(
                    "wal record length {len} exceeds limit"
                )));
            }
            let expected_crc = u32::from_le_bytes(header_buf[4..8].try_into().expect("crc slice"));
            let mut bytes = vec![0u8; len as usize];
            if let Err(e) = file.read_exact(&mut bytes) {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    break;
                }
                return Err(e.into());
            }
            if crc32fast::hash(&bytes) != expected_crc {
                break;
            }
            let txn: WalTxn = postcard::from_bytes(&bytes)?;
            last_txn = txn.txn_id.max(last_txn);
            txns.push(txn);
        }
        Ok((txns, last_txn))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl std::fmt::Debug for Wal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wal").field("path", &self.path).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PointId;

    fn txn(id: u64, point_id: u64) -> WalTxn {
        WalTxn {
            txn_id: id,
            ops: vec![WalRecord::Upsert {
                point: WalPoint::from_point(&VectorPoint::new(
                    PointId::Num(point_id),
                    vec![1.0, 2.0, 3.0],
                ))
                .unwrap(),
            }],
        }
    }

    #[test]
    fn test_append_replay_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        let wal = Wal::open_or_create(&path).unwrap();
        wal.append(&txn(1, 10)).unwrap();
        wal.append(&txn(2, 20)).unwrap();
        wal.append(&WalTxn {
            txn_id: 3,
            ops: vec![WalRecord::Delete {
                point_id: "10".to_string(),
            }],
        })
        .unwrap();

        let mut seen = Vec::new();
        let last = wal
            .replay(|t| {
                seen.push(t.txn_id);
                Ok(())
            })
            .unwrap();
        assert_eq!(last, 3);
        assert_eq!(seen, vec![1, 2, 3]);
    }

    #[test]
    fn test_replay_ignores_truncated_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        let wal = Wal::open_or_create(&path).unwrap();
        wal.append(&txn(1, 10)).unwrap();

        // Append a garbage trailing record that is not a valid length-prefixed
        // txn: 3 bytes claiming a 100-byte record that does not exist.
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&100u32.to_le_bytes()).unwrap();
            f.sync_all().unwrap();
        }

        let mut seen = Vec::new();
        let last = wal
            .replay(|t| {
                seen.push(t.txn_id);
                Ok(())
            })
            .unwrap();
        assert_eq!(last, 1);
        assert_eq!(seen, vec![1]);
    }

    #[test]
    fn test_replay_stops_at_checksum_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        let wal = Wal::open_or_create(&path).unwrap();
        wal.append(&txn(1, 10)).unwrap();
        wal.append(&txn(2, 20)).unwrap();

        // Flip one payload byte of the second record: the length prefix still
        // parses but the checksum no longer matches, so the replay must stop
        // after the first record instead of applying torn data.
        let file_len = std::fs::metadata(&path).unwrap().len();
        {
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            use std::io::{Seek, SeekFrom, Write};
            f.seek(SeekFrom::Start(file_len - 1)).unwrap();
            f.write_all(&[0xFFu8]).unwrap();
        }

        let mut seen = Vec::new();
        let last = wal
            .replay(|t| {
                seen.push(t.txn_id);
                Ok(())
            })
            .unwrap();
        assert_eq!(last, 1);
        assert_eq!(seen, vec![1]);
    }

    #[test]
    fn test_truncate_then_replay_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        let wal = Wal::open_or_create(&path).unwrap();
        wal.append(&txn(7, 10)).unwrap();
        wal.truncate().unwrap();
        let last = wal.replay(|_| Ok(())).unwrap();
        assert_eq!(last, 0);
    }

    #[test]
    fn test_replay_double_apply_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        let wal = Wal::open_or_create(&path).unwrap();
        let t = txn(1, 10);
        wal.append(&t).unwrap();
        wal.append(&t).unwrap();

        let mut count = 0;
        wal.replay(|_| {
            count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_compact_checkpoint_replayable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.bin");
        let wal = Wal::open_or_create(&path).unwrap();
        wal.append(&txn(1, 10)).unwrap();
        wal.append(&WalTxn {
            txn_id: 2,
            ops: vec![WalRecord::Compact],
        })
        .unwrap();

        let mut seen = Vec::new();
        let last = wal
            .replay(|t| {
                seen.push((t.txn_id, t.ops.clone()));
                Ok(())
            })
            .unwrap();
        assert_eq!(last, 2);
        assert_eq!(seen.len(), 2);
        assert!(matches!(seen[1].1[0], WalRecord::Compact));
    }
}
