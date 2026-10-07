//! `wal.bin` — append-only transaction log per collection.
//!
//! Each record is `[u32 len][u32 crc32][postcard(WalTxn)]`; the checksum covers
//! the encoded payload so a torn or bit-rotted record cannot be mistaken for a
//! committed transaction. Appends are fsync'ed before the caller applies the
//! transaction to memory. Replay is idempotent: upserts overwrite by point id,
//! deletes of missing points are no-ops, and `Compact` checkpoints only advance
//! the water mark. A truncated or checksum-failed final record (crash
//! mid-append) stops the replay silently and is treated as end of log;
//! anything else malformed is reported as `CorruptData`.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::types::{PointId, VectorPoint};

/// Maximum accepted single-record size (guards against corrupt length fields).
const MAX_RECORD_LEN: u32 = 512 * 1024 * 1024;

/// State shared between concurrent appenders for group commit.
struct CommitBatch {
    /// Buffered record bytes waiting to be flushed.
    buffer: Vec<u8>,
    /// Number of appenders currently waiting in this batch.
    pending: usize,
    /// True if the leader is currently writing + fsyncing.
    flushing: bool,
    /// Set to Ok(()) after a successful flush, Err on failure.
    result: std::result::Result<(), std::io::Error>,
}

/// Group commit state: multiple appenders share a single write+fsync.
struct GroupCommit {
    state: Mutex<CommitBatch>,
    condvar: Condvar,
}

/// One transaction: a batch of operations applied atomically to a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalTxn {
    pub txn_id: u64,
    pub ops: Vec<WalRecord>,
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

/// Append-only WAL file with group commit.
pub struct Wal {
    path: PathBuf,
    file: parking_lot::Mutex<File>,
    group: GroupCommit,
}

impl Wal {
    /// Open the WAL for appending, creating it if missing.
    pub fn open_or_create(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            file: parking_lot::Mutex::new(file),
            group: GroupCommit {
                state: Mutex::new(CommitBatch {
                    buffer: Vec::new(),
                    pending: 0,
                    flushing: false,
                    result: Ok(()),
                }),
                condvar: Condvar::new(),
            },
        })
    }

    /// Append a transaction using group commit: multiple concurrent appends
    /// share a single write_all + fsync to reduce per-transaction disk latency.
    pub fn append(&self, txn: &WalTxn) -> Result<()> {
        let bytes = postcard::to_stdvec(txn)?;
        if bytes.len() > MAX_RECORD_LEN as usize {
            return Err(VectorSearchError::Internal(format!(
                "wal record too large: {} bytes",
                bytes.len()
            )));
        }
        let mut record = Vec::with_capacity(8 + bytes.len());
        record.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        record.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
        record.extend_from_slice(&bytes);

        let mut state = self.group.state.lock();
        state.buffer.extend_from_slice(&record);
        state.pending += 1;

        // If someone is already flushing, wait for it.
        while state.flushing {
            self.group.condvar.wait(&mut state);
        }

        // If we are the leader (first to arrive or previous flush finished),
        // perform the write + fsync for the entire batch.
        if state.pending > 0 && !state.flushing {
            state.flushing = true;
            let batch_len = state.buffer.len();
            let batch = std::mem::replace(&mut state.buffer, Vec::with_capacity(batch_len));
            drop(state);

            let write_result = {
                let mut file = self.file.lock();
                file.write_all(&batch).and_then(|_| file.sync_all())
            };

            let mut state = self.group.state.lock();
            state.flushing = false;
            state.pending = 0;
            state.result = write_result;
            self.group.condvar.notify_all();

            return match &state.result {
                Ok(()) => Ok(()),
                Err(e) => Err(std::io::Error::new(e.kind(), e.to_string()).into()),
            };
        }

        // We are a follower: wait for the leader to finish.
        loop {
            self.group.condvar.wait(&mut state);
            if !state.flushing {
                break;
            }
        }
        match &state.result {
            Ok(()) => Ok(()),
            Err(e) => Err(std::io::Error::new(e.kind(), e.to_string()).into()),
        }
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
    /// already durable in the rebuilt files and `meta.bin`).
    pub fn truncate(&self) -> Result<()> {
        let mut file = self.file.lock();
        file.set_len(0)?;
        file.rewind()?;
        file.sync_all()?;
        Ok(())
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
