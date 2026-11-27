//! `wal/` — segmented append-only transaction log per collection.
//!
//! The log is a directory of segment files named after the first `txn_id`
//! they contain (`00000000000000000001.bin`), so lexical order is lsn order
//! and the name doubles as the segment's lower bound. A segment is created
//! lazily on the first append that needs it, and rolled over once its size
//! reaches [`Wal::segment_bytes`]; a batch is never split across segments.
//!
//! Each record is `[u32 len][u32 crc32][postcard(WalTxn)]`; the checksum
//! covers the encoded payload so a torn or bit-rotted record cannot be
//! mistaken for a committed transaction. Durability is one `write_all` +
//! `sync_all` per *batch*: the commit queue (`super::queue`) hands its whole
//! backlog to [`Wal::append_batch`], so concurrent writers share a single
//! fsync — that shared fsync is what makes group commit actually group.
//! Single-record appends (maintenance markers) go through the same batch path.
//!
//! A batch that fails at any point is rolled back to the offset it started
//! at. Without the rollback a half-written record would sit in the middle of
//! the log, and replay stops at the first checksum mismatch — silently
//! discarding every record behind it.
//!
//! Replay is idempotent: upserts overwrite by point id, deletes of missing
//! points are no-ops, and `Compact` checkpoints only advance the water mark.
//! A truncated or checksum-failed final record (crash mid-append) stops the
//! replay silently and is treated as end of log — but only in the *last*
//! segment: in any earlier segment the log continues after it, so the same
//! damage is reported as `CorruptData` instead of silently dropping the tail.
//!
//! Segments exist so that discarding logged history is `unlink`, not
//! `set_len(0)` on a file that then keeps growing: [`Wal::truncate_through`]
//! removes every segment whose whole lsn range is at or below a durable water
//! mark (the checkpoint's `last_applied_txn`), and [`Wal::truncate`] wipes the
//! directory for the paths that rewrite every derived file anyway. Both paths
//! publish the water mark in `wal/acked-lsn` *before* unlinking, bound to the
//! generation whose files it describes. A store binds its generation at open
//! and after a compaction publish; an ack from a different generation is
//! ignored, because only that generation's manifest knows which operations
//! its files contain.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::types::VectorPoint;

/// Maximum accepted single-record size (guards against corrupt length fields).
const MAX_RECORD_LEN: u32 = 512 * 1024 * 1024;

/// Segment file extension. Names are zero-padded `txn_id`s so that a plain
/// lexical sort of the directory is lsn order.
const SEGMENT_EXT: &str = "bin";
const SEGMENT_NAME_DIGITS: usize = 20;

/// Durable watermark: every transaction at or below this lsn is already
/// reflected in the collection's data files. It is written before covered
/// segments are unlinked, so a segment that reappears after a crash cannot
/// replay stale operations over newer data.
const ACK_FILE: &str = "acked-lsn";
const ACK_MAGIC: [u8; 4] = *b"SVAK";
const ACK_VERSION: u32 = 1;
const ACK_HEADER_LEN: usize = 28;

/// Segments kept as one transaction log; the bytes after which a new segment
/// is started. Default mirrors `StorageConfig::wal_segment_bytes`.
pub const DEFAULT_SEGMENT_BYTES: u64 = 64 * 1024 * 1024;

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
/// `serde_json::Value`) — same choice as `keys.bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalPoint {
    pub id: String,
    pub vector: Vec<f32>,
    /// JSON-encoded payload.
    pub payload: Option<String>,
}

impl WalPoint {
    pub fn from_point(point: &VectorPoint) -> Result<Self> {
        if point.vectors.len() != 1 || !point.vectors.contains_key(crate::types::DEFAULT_VECTOR_NAME) {
            return Err(VectorSearchError::InvalidConfig(
                "dense WAL upserts must carry exactly the default vector space".to_string(),
            ));
        }
        let payload = match &point.payload {
            Some(p) => Some(serde_json::to_string(p)?),
            None => None,
        };
        Ok(Self {
            id: point.id.to_string(),
            vector: point.vector().to_vec(),
            payload,
        })
    }

    pub fn to_point(&self) -> Result<VectorPoint> {
        let payload = match &self.payload {
            Some(s) => Some(serde_json::from_str(s)?),
            None => None,
        };
        let mut point = VectorPoint::new(self.id.clone(), self.vector.clone());
        point.payload = payload;
        Ok(point)
    }
}

/// A single mutation logged in the WAL.
///
/// Delete variants keep point ids as strings on purpose: the binary codec
/// cannot encode the untagged point id enum, so the typed boundary ends at
/// the transaction entry and replay converts back exactly once.
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
    /// Upsert a point with one or more named vector spaces. Kind collections
    /// require exactly the default space; named collections accept the full
    /// declared map. The dense store rejects this variant.
    UpsertVectors {
        id: String,
        vectors: BTreeMap<String, crate::types::VectorData>,
        /// JSON-encoded payload.
        payload: Option<String>,
    },
    /// Replace selected named spaces of an existing point, preserving all
    /// other spaces and the point's slot.
    PatchNamedVectors {
        id: String,
        vectors: BTreeMap<String, crate::types::VectorData>,
    },
}

/// Mutable WAL state: the segment directory listing plus the open handle of
/// the segment currently being appended to.
#[derive(Default)]
struct WalState {
    /// Segment names (first `txn_id` of each) in ascending order.
    segments: Vec<u64>,
    /// Handle of the highest segment, opened for append. `None` after the
    /// segment was retired by a truncation, so the next append creates one.
    active: Option<File>,
    /// Highest `txn_id` known to live in the active segment. Used by
    /// [`Wal::truncate_through`] to decide whether that segment — the one
    /// whose upper bound no segment name encodes — is fully covered by the
    /// water mark. `u64::MAX` while unknown (a segment that was never fully
    /// read and never appended to), which makes the check conservative.
    active_last_lsn: u64,
}

/// Segmented append-only WAL directory. Every durable mutation goes through
/// [`Wal::append_batch`], one write + one fsync per batch.
pub struct Wal {
    dir: PathBuf,
    segment_bytes: u64,
    state: parking_lot::Mutex<WalState>,
    /// Bytes physically held by the log since the last full truncate. Drives
    /// the size-based checkpoint schedule: a pure-insert workload never
    /// crosses a tombstone ratio, so without it the log would grow without
    /// bound.
    appended_bytes: std::sync::atomic::AtomicU64,
    /// Generation the watermark belongs to. An ack written by another
    /// generation describes a different set of data files and is ignored.
    ack_generation: AtomicU64,
    /// Generation whose log this is. Zero until the owning store binds it.
    generation: AtomicU64,
    /// Highest lsn whose effects are durable in the data files. Only
    /// advanced by checkpoint/truncation paths, never by ordinary appends.
    acked_lsn: AtomicU64,
    /// Test-only: fail the next batch append, to exercise the failure paths.
    #[cfg(test)]
    fail_next_append: std::sync::atomic::AtomicBool,
}

impl Wal {
    /// Open (or create) the log directory for appending.
    ///
    /// `segment_bytes` is the size at which a new segment is started; it is
    /// not persisted, so changing it applies to the next rollover only.
    pub fn open_or_create(dir: &Path, segment_bytes: u64) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let segments = list_segments(dir)?;
        let mut total = 0u64;
        for name in &segments {
            total += std::fs::metadata(segment_path(dir, *name))?.len();
        }
        let active = match segments.last() {
            Some(name) => Some(OpenOptions::new().read(true).append(true).open(segment_path(dir, *name))?),
            None => None,
        };
        let (ack_generation, ack_lsn) = read_ack(dir);
        Ok(Self {
            dir: dir.to_path_buf(),
            segment_bytes: segment_bytes.max(1),
            ack_generation: AtomicU64::new(ack_generation),
            generation: AtomicU64::new(0),
            state: parking_lot::Mutex::new(WalState {
                // An existing tail segment may hold transactions this process
                // has not read; `replay`/`read_all` refines the bound.
                active_last_lsn: if segments.is_empty() { 0 } else { u64::MAX },
                segments,
                active,
            }),
            appended_bytes: std::sync::atomic::AtomicU64::new(total),
            acked_lsn: AtomicU64::new(ack_lsn),
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

    /// Segment size threshold (bytes) at which the log rolls over.
    pub fn segment_bytes(&self) -> u64 {
        self.segment_bytes
    }

    /// Bind this log to the generation whose files it is replayed into.
    /// An ack from a different generation is discarded: only the generation
    /// manifest can say which operations its files already contain.
    pub fn bind_generation(&self, generation: u64) {
        self.generation.store(generation, AtomicOrdering::Release);
    }

    /// Highest lsn already covered by the bound generation's data files.
    pub fn acked_lsn(&self) -> u64 {
        if self.ack_generation.load(AtomicOrdering::Acquire)
            != self.generation.load(AtomicOrdering::Acquire)
        {
            return 0;
        }
        self.acked_lsn.load(AtomicOrdering::Acquire)
    }

    /// Generation that published the current watermark.
    pub fn ack_generation(&self) -> u64 {
        self.ack_generation.load(AtomicOrdering::Acquire)
    }

    /// Read every transaction newer than the durable watermark.
    ///
    /// Callers must only use this when the data files really do contain every
    /// operation up to `acked_lsn`; dense collections satisfy that after a
    /// checkpoint. Exact-scan stores that keep their own snapshot watermark
    /// use [`Self::replay_after`] with that snapshot's lsn instead.
    pub fn read_all_acked(&self) -> Result<(Vec<WalTxn>, u64)> {
        self.read_all_from(self.acked_lsn())
    }

    /// Replay everything newer than an explicit floor.
    ///
    /// Exact-scan stores pass their published generation's lsn instead of the
    /// WAL watermark: only a generation can say which operations are already
    /// reflected in its snapshot.
    pub fn read_all_from(&self, floor: u64) -> Result<(Vec<WalTxn>, u64)> {
        let mut txns = Vec::new();
        // An lsn of zero means "nothing checkpointed": transaction zero is
        // valid for the exact-scan stores, so replay the whole log.
        let last_txn = self.replay_after((floor > 0).then_some(floor), |txn| {
            txns.push(txn.clone());
            Ok(())
        })?;
        Ok((txns, last_txn.max(floor)))
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
        let mut first_txn = u64::MAX;
        let mut last_txn = 0u64;
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
            first_txn = first_txn.min(*txn_id);
            last_txn = last_txn.max(*txn_id);
            buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
            buffer.extend_from_slice(&bytes);
        }

        let mut state = self.state.lock();
        #[cfg(test)]
        if self
            .fail_next_append
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            return Err(VectorSearchError::Internal(
                "test-injected wal append failure".to_string(),
            ));
        }
        // Roll over before writing, so a segment never exceeds the threshold
        // by more than the size of one batch. The new segment is named after
        // the first transaction it will hold.
        let needs_segment = match &state.active {
            Some(file) => file.metadata()?.len().max(0) >= self.segment_bytes,
            None => true,
        };
        if needs_segment {
            let name = next_segment_name(&state.segments, first_txn);
            let file = OpenOptions::new()
                .read(true)
                .append(true)
                .create_new(true)
                .open(segment_path(&self.dir, name))?;
            // The directory entry has to be durable before any record in it
            // can be considered committed.
            File::open(&self.dir)?.sync_all()?;
            state.segments.push(name);
            state.active = Some(file);
            state.active_last_lsn = 0;
        }

        let file = state
            .active
            .as_mut()
            .expect("active segment opened above");
        let start_len = file.metadata()?.len();
        let write_result = file.write_all(&buffer).and_then(|_| file.sync_all());
        if let Err(e) = write_result {
            // Drop the partial batch so the log keeps a clean suffix.
            let _ = file.set_len(start_len);
            let _ = file.sync_all();
            return Err(e.into());
        }
        state.active_last_lsn = state.active_last_lsn.max(last_txn);
        drop(state);
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
    pub fn replay(&self, f: impl FnMut(&WalTxn) -> Result<()>) -> Result<u64> {
        self.replay_after(None, f)
    }

    /// Replay transactions with `txn_id > after`. Used when a snapshot has
    /// already covered every earlier transaction; skipped records still
    /// advance the returned high-water mark.
    pub fn replay_after(
        &self,
        after: Option<u64>,
        mut f: impl FnMut(&WalTxn) -> Result<()>,
    ) -> Result<u64> {
        let segments = self.state.lock().segments.clone();
        let mut last_txn = 0u64;
        for (index, name) in segments.iter().enumerate() {
            let last_segment = index + 1 == segments.len();
            let mut file = File::open(segment_path(&self.dir, *name))?;
            last_txn = read_segment(
                &mut file,
                last_segment,
                after,
                &self.dir,
                *name,
                &mut f,
                last_txn,
            )?;
        }
        // The log's highest lsn is now known, which lets `truncate_through`
        // decide about the tail segment without re-reading it.
        self.state.lock().active_last_lsn = last_txn;
        Ok(last_txn)
    }

    /// Remove the whole log: every segment is unlinked and the next append
    /// starts a fresh one. Used by the paths that rewrite every derived file
    /// (compaction), where no logged transaction remains authoritative.
    pub fn truncate(&self) -> Result<()> {
        let mut state = self.state.lock();
        // Every segment is being dropped because the data files already
        // contain the whole log. Publish that fact before unlinking so a
        // crash cannot leave a reused segment replayable.
        let covered = if state.active_last_lsn == u64::MAX {
            state.segments.last().map_or(0, |name| name.saturating_sub(1))
        } else {
            state.active_last_lsn
        };
        let new_ack = self.acked_lsn().max(covered);
        let generation = self.generation.load(AtomicOrdering::Acquire);
        write_ack(&self.dir, generation, new_ack)?;
        self.ack_generation
            .store(generation, AtomicOrdering::Release);
        for name in state.segments.drain(..) {
            remove_segment(&self.dir, name)?;
        }
        sync_dir(&self.dir)?;
        state.active = None;
        state.active_last_lsn = 0;
        drop(state);
        self.acked_lsn
            .store(new_ack, AtomicOrdering::Release);
        self.appended_bytes.store(0, AtomicOrdering::Relaxed);
        Ok(())
    }

    /// Drop every segment that is entirely at or below `watermark`.
    ///
    /// A segment is covered when the next segment starts at or below
    /// `watermark + 1` — segment names are first-lsn bounds, so everything in
    /// the earlier segment is `< next name` — and the tail segment is covered
    /// when its highest appended lsn is at or below the water mark. Whole
    /// segments only: a segment straddling the water mark is kept as is,
    /// because prefix-truncating it would rewrite the file this scheme exists
    /// to avoid rewriting.
    ///
    /// This is the checkpoint's truncate: the caller has just made everything
    /// up to `watermark` durable in the data files, so the covered records
    /// exist only to be re-read.
    pub fn truncate_through(&self, watermark: u64) -> Result<()> {
        let mut state = self.state.lock();
        // The caller has just made everything up to `watermark` durable in
        // the data files. Publish that watermark before unlinking anything:
        // if a covered segment survives a crash, replay skips it instead of
        // overwriting newer state with old operations.
        let new_ack = self.acked_lsn().max(watermark);
        let generation = self.generation.load(AtomicOrdering::Acquire);
        write_ack(&self.dir, generation, new_ack)?;
        self.ack_generation
            .store(generation, AtomicOrdering::Release);
        self.acked_lsn.store(new_ack, AtomicOrdering::Release);
        let segments = std::mem::take(&mut state.segments);
        let mut kept: Vec<u64> = Vec::with_capacity(segments.len());
        let mut removed_bytes = 0u64;
        for (index, name) in segments.iter().enumerate() {
            let covered = match segments.get(index + 1) {
                Some(next) => next.saturating_sub(1) <= watermark,
                None => state.active.is_none() || state.active_last_lsn <= watermark,
            };
            if covered {
                removed_bytes += std::fs::metadata(segment_path(&self.dir, *name))
                    .map(|meta| meta.len())
                    .unwrap_or(0);
                remove_segment(&self.dir, *name)?;
                if index + 1 == segments.len() {
                    // The tail segment is gone: retire the handle so the next
                    // append creates a new one.
                    state.active = None;
                    state.active_last_lsn = 0;
                }
            } else {
                kept.push(*name);
            }
        }
        state.segments = kept;
        sync_dir(&self.dir)?;
        drop(state);
        if removed_bytes > 0 {
            self.appended_bytes.fetch_sub(
                removed_bytes.min(self.appended_bytes()),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        Ok(())
    }

    /// Bytes currently held in the log. Drives the size-based checkpoint
    /// schedule.
    pub fn appended_bytes(&self) -> u64 {
        self.appended_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Read all valid WAL records into a `Vec<WalTxn>`, returning the highest
    /// `txn_id` seen. Caller must hold no other WAL locks; this acquires and
    /// releases the state lock internally so callers can batch-apply without
    /// per-record lock contention.
    pub fn read_all(&self) -> Result<(Vec<WalTxn>, u64)> {
        let mut txns = Vec::new();
        let last_txn = self.replay(|txn| {
            txns.push(txn.clone());
            Ok(())
        })?;
        Ok((txns, last_txn))
    }

    /// Segment names currently present, ascending. Observability only.
    pub fn segment_names(&self) -> Vec<u64> {
        self.state.lock().segments.clone()
    }

    /// Path of the segment the log is currently appending to, if any. Used by
    /// crash-injection tests that append a torn record directly.
    pub fn active_segment_path(&self) -> Option<PathBuf> {
        self.state
            .lock()
            .segments
            .last()
            .map(|name| segment_path(&self.dir, *name))
    }

    /// Directory holding the segments.
    pub fn path(&self) -> &Path {
        &self.dir
    }
}

impl std::fmt::Debug for Wal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wal")
            .field("dir", &self.dir)
            .field("segment_bytes", &self.segment_bytes)
            .finish()
    }
}

/// Read one segment, feeding transactions to `f`. Returns the highest lsn
/// seen so far.
fn read_segment(
    file: &mut File,
    last_segment: bool,
    after: Option<u64>,
    dir: &Path,
    name: u64,
    f: &mut impl FnMut(&WalTxn) -> Result<()>,
    mut last_txn: u64,
) -> Result<u64> {
    file.rewind()?;
    let mut header_buf = [0u8; 8];
    loop {
        match file.read_exact(&mut header_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes(header_buf[0..4].try_into().expect("len slice"));
        if len == 0 {
            return Err(VectorSearchError::CorruptData(format!(
                "wal record with zero length in {}",
                dir.join(segment_file_name(name)).display()
            )));
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
                if last_segment {
                    break; // truncated trailing record from a crash
                }
                return Err(VectorSearchError::CorruptData(format!(
                    "wal segment {} ends mid-record before the log does",
                    segment_file_name(name)
                )));
            }
            return Err(e.into());
        }
        if crc32fast::hash(&bytes) != expected_crc {
            // Torn write or bit rot. In the tail segment nothing after it can
            // be trusted either, so the replay simply ends; in an earlier
            // segment the log continues, which makes it real corruption.
            if last_segment {
                break;
            }
            return Err(VectorSearchError::CorruptData(format!(
                "wal segment {} failed its checksum before the log ended",
                segment_file_name(name)
            )));
        }
        let txn: WalTxn = postcard::from_bytes(&bytes)?;
        if after.is_none_or(|lsn| txn.txn_id > lsn) {
            f(&txn)?;
        }
        last_txn = txn.txn_id.max(last_txn);
    }
    Ok(last_txn)
}

/// Persist the generation and replay watermark with a checksum and a
/// directory fsync. A torn or corrupt file reads as an unbound zero, which
/// is always safe: the generation manifest remains the replay authority.
fn write_ack(dir: &Path, generation: u64, lsn: u64) -> Result<()> {
    let target = dir.join(ACK_FILE);
    let tmp = dir.join("acked-lsn.tmp");
    let mut bytes = Vec::with_capacity(ACK_HEADER_LEN);
    bytes.extend_from_slice(&ACK_MAGIC);
    bytes.extend_from_slice(&ACK_VERSION.to_le_bytes());
    bytes.extend_from_slice(&generation.to_le_bytes());
    bytes.extend_from_slice(&lsn.to_le_bytes());
    let mut checked = Vec::with_capacity(16);
    checked.extend_from_slice(&generation.to_le_bytes());
    checked.extend_from_slice(&lsn.to_le_bytes());
    bytes.extend_from_slice(&crc32fast::hash(&checked).to_le_bytes());

    let mut file = File::create(&tmp)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp, &target)?;
    sync_dir(dir)
}

fn read_ack(dir: &Path) -> (u64, u64) {
    let Ok(bytes) = std::fs::read(dir.join(ACK_FILE)) else {
        return (0, 0);
    };
    if bytes.len() != ACK_HEADER_LEN || bytes[0..4] != ACK_MAGIC {
        return (0, 0);
    }
    if u32::from_le_bytes(bytes[4..8].try_into().expect("ack version slice")) != ACK_VERSION {
        return (0, 0);
    }
    let generation_bytes: [u8; 8] = bytes[8..16].try_into().expect("ack generation slice");
    let lsn_bytes: [u8; 8] = bytes[16..24].try_into().expect("ack lsn slice");
    let expected = u32::from_le_bytes(bytes[24..28].try_into().expect("ack crc slice"));
    let mut checked = Vec::with_capacity(16);
    checked.extend_from_slice(&generation_bytes);
    checked.extend_from_slice(&lsn_bytes);
    if crc32fast::hash(&checked) != expected {
        return (0, 0);
    }
    (
        u64::from_le_bytes(generation_bytes),
        u64::from_le_bytes(lsn_bytes),
    )
}

/// Make WAL segment removals and new segment names durable as a directory
/// change, not just as file writes.
fn sync_dir(dir: &Path) -> Result<()> {
    File::open(dir)?.sync_all()?;
    Ok(())
}

fn segment_file_name(name: u64) -> String {
    format!(
        "{:0width$}.{}",
        name,
        SEGMENT_EXT,
        width = SEGMENT_NAME_DIGITS
    )
}

fn segment_path(dir: &Path, name: u64) -> PathBuf {
    dir.join(segment_file_name(name))
}

fn list_segments(dir: &Path) -> Result<Vec<u64>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        let Some(stem) = file_name.strip_suffix(&format!(".{SEGMENT_EXT}")) else {
            continue;
        };
        match stem.parse::<u64>() {
            Ok(name) => names.push(name),
            Err(_) => tracing::warn!(
                file = file_name,
                "ignoring unrecognized file in the wal directory"
            ),
        }
    }
    names.sort_unstable();
    names.dedup();
    Ok(names)
}

/// Pick the name for a new segment: the first transaction it will hold, but
/// always strictly above the existing tail so names stay ascending.
fn next_segment_name(segments: &[u64], first_txn: u64) -> u64 {
    match segments.last() {
        Some(last) => first_txn.max(last.saturating_add(1)),
        None => first_txn,
    }
}

fn remove_segment(dir: &Path, name: u64) -> Result<()> {
    match std::fs::remove_file(segment_path(dir, name)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
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

    fn wal(dir: &Path) -> Wal {
        Wal::open_or_create(dir, DEFAULT_SEGMENT_BYTES).unwrap()
    }

    #[test]
    fn test_append_replay_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let wal = wal(&path);
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
        let path = dir.path().join("wal");
        let wal = wal(&path);
        wal.append(&txn(1, 10)).unwrap();

        // Append a garbage trailing record that is not a valid length-prefixed
        // txn: 4 bytes claiming a 100-byte record that does not exist.
        {
            let mut f = OpenOptions::new()
                .append(true)
                .open(segment_path(&path, 1))
                .unwrap();
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
        let path = dir.path().join("wal");
        let wal = wal(&path);
        wal.append(&txn(1, 10)).unwrap();
        wal.append(&txn(2, 20)).unwrap();

        // Flip one payload byte of the second record: the length prefix still
        // parses but the checksum no longer matches, so the replay must stop
        // after the first record instead of applying torn data.
        let segment = segment_path(&path, 1);
        let file_len = std::fs::metadata(&segment).unwrap().len();
        {
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&segment)
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
        let path = dir.path().join("wal");
        let wal = wal(&path);
        wal.append(&txn(7, 10)).unwrap();
        wal.truncate().unwrap();
        let last = wal.replay(|_| Ok(())).unwrap();
        assert_eq!(last, 0);
        assert!(wal.segment_names().is_empty());
        assert_eq!(wal.appended_bytes(), 0);
    }

    #[test]
    fn test_replay_double_apply_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let wal = wal(&path);
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
        let path = dir.path().join("wal");
        let wal = wal(&path);
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

    #[test]
    fn test_segments_roll_over_and_replay_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        // One record is roughly 40 bytes; 100 bytes forces a segment every
        // couple of transactions.
        let wal = Wal::open_or_create(&path, 100).unwrap();
        for id in 1..=5u64 {
            wal.append(&txn(id, id * 10)).unwrap();
        }
        let names = wal.segment_names();
        assert!(
            names.len() > 1,
            "the log must have rolled over, got {names:?}"
        );
        assert_eq!(names[0], 1, "first segment is named after its first txn");
        assert!(
            names.windows(2).all(|pair| pair[0] < pair[1]),
            "segment names must ascend: {names:?}"
        );

        let mut seen = Vec::new();
        let last = wal
            .replay(|t| {
                seen.push(t.txn_id);
                Ok(())
            })
            .unwrap();
        assert_eq!(last, 5);
        assert_eq!(seen, vec![1, 2, 3, 4, 5], "replay follows segment order");

        // A reopen must find the same log through the directory listing.
        drop(wal);
        let reopened = Wal::open_or_create(&path, 100).unwrap();
        assert_eq!(reopened.segment_names(), names);
        assert!(reopened.appended_bytes() > 0);
        let mut count = 0;
        assert_eq!(
            reopened
                .replay(|_| {
                    count += 1;
                    Ok(())
                })
                .unwrap(),
            5
        );
        assert_eq!(count, 5);
    }

    #[test]
    fn test_truncate_through_removes_only_covered_segments() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let wal = Wal::open_or_create(&path, 40).unwrap();
        for id in 1..=6u64 {
            wal.append(&txn(id, id)).unwrap();
        }
        let names = wal.segment_names();
        assert!(names.len() >= 3, "need several segments, got {names:?}");
        // Highest lsn the log holds is 6 (learned by replay).
        assert_eq!(wal.replay(|_| Ok(())).unwrap(), 6);
        let before = wal.appended_bytes();

        // A water mark inside the second segment keeps it and everything
        // after it, and drops only the first segment.
        let watermark = names[1];
        wal.truncate_through(watermark).unwrap();
        let kept = wal.segment_names();
        assert!(
            !kept.contains(&names[0]),
            "the fully covered head segment must be unlinked"
        );
        assert!(
            kept.contains(&names[1]),
            "a segment that straddles the water mark stays"
        );
        assert!(wal.appended_bytes() < before);

        // Every surviving record is still replayable and ordered.
        let mut seen = Vec::new();
        wal.replay(|t| {
            seen.push(t.txn_id);
            Ok(())
        })
        .unwrap();
        assert!(seen.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(*seen.last().unwrap(), 6);
    }

    #[test]
    fn test_acked_replay_skips_checkpointed_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let wal = Wal::open_or_create(&path, DEFAULT_SEGMENT_BYTES).unwrap();
        wal.append(&txn(1, 10)).unwrap();
        wal.append(&txn(2, 20)).unwrap();

        wal.truncate_through(2).unwrap();
        assert_eq!(wal.acked_lsn(), 2);
        wal.append(&txn(3, 30)).unwrap();

        let (txns, last) = wal.read_all_acked().unwrap();
        assert_eq!(last, 3);
        assert_eq!(txns.iter().map(|t| t.txn_id).collect::<Vec<_>>(), vec![3]);

        // The watermark survives reopen, so a resurrected old segment is
        // skipped on the next recovery too.
        drop(wal);
        let reopened = Wal::open_or_create(&path, DEFAULT_SEGMENT_BYTES).unwrap();
        assert_eq!(reopened.acked_lsn(), 2);
        assert_eq!(reopened.read_all_acked().unwrap().0.len(), 1);
    }

    #[test]
    fn test_truncate_through_drops_the_tail_when_fully_covered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let wal = wal(&path);
        wal.append(&txn(1, 1)).unwrap();
        wal.append(&txn(2, 2)).unwrap();
        assert_eq!(wal.segment_names().len(), 1);

        wal.truncate_through(2).unwrap();
        assert!(
            wal.segment_names().is_empty(),
            "a checkpoint covering the whole log empties it"
        );
        assert_eq!(wal.appended_bytes(), 0);
        // The retired segment is replaced by a fresh one on the next append.
        wal.append(&txn(3, 3)).unwrap();
        assert_eq!(wal.segment_names(), vec![3]);
        assert_eq!(wal.replay(|_| Ok(())).unwrap(), 3);
    }

    #[test]
    fn test_ack_belongs_to_one_generation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let first = wal(&path);
        first.bind_generation(7);
        first.append(&txn(1, 10)).unwrap();
        first.truncate_through(1).unwrap();
        assert_eq!(first.ack_generation(), 7);
        assert_eq!(first.acked_lsn(), 1);
        drop(first);

        let reopened = wal(&path);
        assert_eq!(reopened.ack_generation(), 7);
        assert_eq!(reopened.acked_lsn(), 0, "unbound store cannot use the ack");
        reopened.bind_generation(8);
        assert_eq!(
            reopened.acked_lsn(),
            0,
            "another generation's ack must be ignored"
        );
        reopened.bind_generation(7);
        assert_eq!(reopened.acked_lsn(), 1);
    }

    #[test]
    fn test_corrupt_middle_segment_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let wal = Wal::open_or_create(&path, 100).unwrap();
        for id in 1..=6u64 {
            wal.append(&txn(id, id)).unwrap();
        }
        let names = wal.segment_names();
        assert!(names.len() >= 2, "need several segments, got {names:?}");
        drop(wal);

        // Corrupt the first segment while later segments still follow it: the
        // replay cannot treat that as a torn tail, because the log evidently
        // continues behind it.
        let segment = segment_path(&path, names[0]);
        let mut bytes = std::fs::read(&segment).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&segment, &bytes).unwrap();

        let wal = Wal::open_or_create(&path, 100).unwrap();
        let error = wal.replay(|_| Ok(())).unwrap_err();
        assert!(
            matches!(error, VectorSearchError::CorruptData(_)),
            "mid-log damage must surface as CorruptData, got {error:?}"
        );
    }
}
