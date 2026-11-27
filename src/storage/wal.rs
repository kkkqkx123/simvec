//! `wal/` — segmented append-only transaction log per collection.
//!
//! The log is a directory of segment files named after the first `txn_id`
//! they contain (`00000000000000000001.bin`), so lexical order is lsn order
//! and the name doubles as the segment's lower bound. Segments are
//! preallocated to [`Wal::segment_bytes`] on creation and served through
//! memory maps: appends stream encoded records into the active mapping with
//! one `flush_range` per group-commit batch, and replay walks read-only
//! mappings sequentially instead of buffering decoded transactions.
//!
//! On-disk segment format (version 2):
//!
//! ```text
//! header (32 B): [magic 4 = "SVWL"][version u32 = 2][seed u32]
//!                [first_lsn u64][reserved u64][reserved u64]
//! record:       [u32 len][u32 crc][postcard bytes][0..7 zero padding]
//! ```
//!
//! Records are 8-byte aligned. The CRC of each record chains the previous
//! record's CRC (the segment seed for the first record) over the length
//! prefix, the payload and the padding, so reusing a file under a new name
//! fails verification instead of replaying stale entries. A zero length
//! prefix is never emitted by a writer; it marks preallocated slack (or a
//! torn tail) and ends the valid prefix. Only the last segment may end
//! mid-record.
//!
//! A batch is never split across segments. If the active segment cannot hold
//! the whole batch, the log rolls over to a fresh preallocated segment named
//! after the batch's first transaction (fresh LSNs are guaranteed by the
//! commit queue, so the name is always strictly above the previous tail).
//! A batch bigger than a fresh segment grows that segment before writing.
//!
//! Rollover reuses a prebuilt spare file (`.spare`, capacity-matched) when
//! one is available, so steady-state appends never create files on the
//! critical path; the spare is replenished after the rollover. If the spare
//! is missing or capacity-mismatched, the segment is created directly.
//!
//! Replay is idempotent: upserts overwrite by point id, deletes of missing
//! points are no-ops, and `Compact` checkpoints only advance the water mark.
//! A truncated final record (crash mid-append) or a zero prefix in the last
//! segment stops the replay silently and is treated as end of log — but only
//! in the *last* segment: in any earlier segment the log continues after it,
//! so the same damage is reported as `CorruptData` instead of silently
//! dropping the tail.
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
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use memmap2::{Mmap, MmapMut};
use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::types::VectorPoint;

/// Maximum accepted single-record size (guards against corrupt length fields).
const MAX_RECORD_LEN: u32 = 512 * 1024 * 1024;

/// Segment file extension. Names are zero-padded `txn_id`s so that a plain
/// lexical sort of the directory is lsn order.
const SEGMENT_EXT: &str = "bin";
const SEGMENT_NAME_DIGITS: usize = 20;

/// Prebuilt spare segment file, capacity-matched on use. A different
/// extension keeps it out of the segment listing.
const SPARE_NAME: &str = ".spare";

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

/// Segment format version written into every header.
const SEG_VERSION: u32 = 2;
/// Segment magic: `SVWL`.
const SEG_MAGIC: [u8; 4] = *b"SVWL";
/// Segment header length in bytes.
const SEG_HEADER_LEN: u64 = 32;
/// Record alignment in bytes.
const RECORD_ALIGN: u64 = 8;
/// Record framing: `[u32 len][u32 crc]` before the payload.
const RECORD_HEADER_LEN: u64 = 8;

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
        if point.vectors.len() != 1
            || !point
                .vectors
                .contains_key(crate::types::DEFAULT_VECTOR_NAME)
        {
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
///
/// Payload variants are keyed by external point id, never by slot: live apply
/// and replay resolve the id through the same id tracker, so the two paths
/// cannot disagree about which slot a payload belongs to.
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
    /// Checkpoint marker written after the data files are durable. Replay
    /// ignores the payload but keeps advancing the water mark. Markers always
    /// carry freshly allocated LSNs (never the checkpoint watermark), so they
    /// can never collide with an existing segment name.
    Compact,
    /// Quantization state change (codebook / scale update). Replay is idempotent
    /// — the derived `quant.bin` files are rebuildable, so this record only
    /// advances the water mark and triggers no in-memory mutation on replay.
    Quantize,
    /// Replace the entire payload for a point. The payload is JSON-encoded
    /// because postcard cannot encode `serde_json::Value` (untagged enum).
    SetPayload {
        point_id: String,
        payload: Option<String>,
    },
    /// Remove specific keys from a point's payload. The remaining keys are
    /// preserved; if the point has no payload this is a no-op.
    DeletePayloadKeys {
        point_id: String,
        keys: Vec<String>,
    },
    /// Set a single key on a point's payload (merge semantics). A missing
    /// payload is created containing just this key; all other keys are
    /// preserved. The value is JSON-encoded because postcard cannot encode
    /// `serde_json::Value` (untagged enum).
    SetPayloadField {
        point_id: String,
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

/// Open handle of the segment currently being appended to.
struct ActiveSegment {
    mmap: MmapMut,
    file: File,
}

/// Mutable WAL state: the segment directory listing plus the append handle.
#[derive(Default)]
struct WalState {
    /// Segment names (first `txn_id` of each) in ascending order.
    segments: Vec<u64>,
    /// Valid data bytes per sealed (closed) segment, in `segments` order
    /// excluding the active tail. Closed segments are immutable, so their
    /// valid length never changes after rollover.
    sealed_valid: Vec<u64>,
    /// Handle of the highest segment, mapped for append. `None` after the
    /// segment was retired by a truncation, so the next append creates one.
    active: Option<ActiveSegment>,
    /// Highest `txn_id` known to live in the active segment. Used by
    /// [`Wal::truncate_through`] to decide whether that segment — the one
    /// whose upper bound no segment name encodes — is fully covered by the
    /// water mark. `u64::MAX` while unknown (a segment that was never fully
    /// read and never appended to), which makes the check conservative.
    active_last_lsn: u64,
    /// Length of the valid prefix of the active segment, from its start.
    /// Everything at or beyond it is preallocated slack or a torn tail.
    /// `None` only before the first replay or append establishes it.
    active_valid_len: Option<u64>,
    /// CRC chaining value at the valid frontier: the segment seed for an
    /// empty segment, otherwise the last valid record's CRC. Lets appends
    /// chain without re-walking the segment. `None` until established by
    /// open, rollover or replay.
    active_crc: Option<u32>,
}

/// Segmented append-only WAL directory. Every durable mutation goes through
/// [`Wal::append_batch`], one visibility publish + one flush per batch.
pub struct Wal {
    dir: PathBuf,
    segment_bytes: u64,
    state: parking_lot::Mutex<WalState>,
    /// Valid data bytes held by the log. Drives the size-based checkpoint
    /// schedule: a pure-insert workload never crosses a tombstone ratio, so
    /// without it the log would grow without bound. Preallocated slack is
    /// not counted, only records.
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
    /// `segment_bytes` is the preallocation size at which a new segment is
    /// started; it is not persisted, so changing it applies to the next
    /// rollover only. Existing segments are length-walked (never verified)
    /// to recover their valid prefixes and the valid-byte total; any damage
    /// is reported by replay, not by open.
    pub fn open_or_create(dir: &Path, segment_bytes: u64) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let segments = list_segments(dir)?;
        let capacity = normalize_capacity(segment_bytes);
        let mut sealed_valid = Vec::with_capacity(segments.len());
        let mut total_valid = 0u64;
        let mut tail_valid = 0u64;
        let mut tail_crc = None;
        for (index, name) in segments.iter().enumerate() {
            if index + 1 == segments.len() {
                // The tail is walked leniently (torn tails tolerated) and
                // yields the chaining CRC, so the first append needs no scan.
                let (valid, crc) = walk_tail_prefix(dir, *name);
                tail_valid = valid;
                tail_crc = crc;
                total_valid += valid;
            } else {
                let valid = match walk_valid_prefix(dir, *name) {
                    Ok(valid) => valid,
                    // An unreadable segment file reads as empty here; replay
                    // reports the damage with its segment context.
                    Err(_) => 0,
                };
                total_valid += valid;
                sealed_valid.push(valid);
            }
        }
        let active = match segments.last() {
            Some(name) => {
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(segment_path(dir, *name))?;
                // The mapping must cover the whole file (including slack) so
                // appends never remap on the fast path.
                let mmap = unsafe { MmapMut::map_mut(&file)? };
                Some(ActiveSegment { mmap, file })
            }
            None => None,
        };
        let (ack_generation, ack_lsn) = read_ack(dir);
        Ok(Self {
            dir: dir.to_path_buf(),
            segment_bytes: capacity,
            ack_generation: AtomicU64::new(ack_generation),
            generation: AtomicU64::new(0),
            state: parking_lot::Mutex::new(WalState {
                // An existing tail segment may hold transactions this process
                // has not read; `replay`/`read_all` refines the bound.
                active_last_lsn: if segments.is_empty() { 0 } else { u64::MAX },
                active_valid_len: if segments.is_empty() {
                    None
                } else {
                    Some(tail_valid)
                },
                sealed_valid,
                segments,
                active,
                active_crc: tail_crc,
            }),
            appended_bytes: std::sync::atomic::AtomicU64::new(total_valid),
            acked_lsn: AtomicU64::new(ack_lsn),
            #[cfg(test)]
            fail_next_append: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Test-only: make the next batch append fail without writing anything,
    /// the way a failing flush would.
    #[cfg(test)]
    pub fn fail_next_append_for_test(&self) {
        self.fail_next_append
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Segment preallocation size (bytes). New segments start at this size.
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

    /// Append a single transaction (one visibility publish + one flush).
    pub fn append(&self, txn: &WalTxn) -> Result<()> {
        self.append_batch(&[(txn.txn_id, &txn.ops)])
    }

    /// Append a batch of transactions with a single visibility publish and a
    /// single flush.
    ///
    /// The batch is the unit of group commit: everything handed in here
    /// becomes durable together, and none of it is applied unless the whole
    /// batch is durable.
    ///
    /// Records are encoded per transaction and streamed into the active
    /// mapping in order — no whole-batch buffer is ever assembled — then one
    /// `flush_range` makes the batch visible. On failure nothing is published:
    /// `active_valid_len` still points at the pre-batch prefix, so the torn
    /// bytes are preallocated slack again rather than a corrupt log.
    pub fn append_batch(&self, batch: &[(u64, &[WalRecord])]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        // Encode per transaction (small buffers, no whole-batch Vec) and
        // measure the framed footprint up front: the batch is never split, so
        // the rollover decision needs the total before writing anything.
        let mut encoded: Vec<(u64, Vec<u8>)> = Vec::with_capacity(batch.len());
        let mut first_txn = u64::MAX;
        let mut last_txn = 0u64;
        let mut total = 0u64;
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
            total += record_total_len(bytes.len());
            encoded.push((*txn_id, bytes));
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
        let valid_len = state.active_valid_len.unwrap_or(0);
        // Roll over before writing, so a segment grows past the threshold
        // only by the size of one batch. The new segment is named after the
        // first transaction it will hold; LSNs are freshly allocated by the
        // commit queue, so the name is always strictly above the old tail.
        let needs_segment = match &state.active {
            Some(_) => valid_len + total > self.segment_bytes,
            None => true,
        };
        let mut rolled = false;
        if needs_segment {
            self.roll_over_locked(&mut state, first_txn, total)?;
            rolled = true;
        }
        let cached_crc = state.active_crc;
        let append_start_snapshot = state.active_valid_len.unwrap_or(0);
        let tail_name = *state.segments.last().expect("active segment is listed");
        // A tail that never got its header (crash between file creation and
        // header write) is reheaded before any record is framed. The header
        // write goes through a separate file handle; the mapping already
        // covers the file, so no remap is needed.
        if cached_crc.is_none() || append_start_snapshot < SEG_HEADER_LEN {
            let seed = random_seed();
            write_segment_header(&segment_path(&self.dir, tail_name), seed, tail_name)?;
            state.active_valid_len = Some(SEG_HEADER_LEN);
            state.active_crc = Some(seed);
        }
        let append_start = state.active_valid_len.unwrap_or(SEG_HEADER_LEN);
        let mut crc = state.active_crc.expect("crc established above");
        let active = state.active.as_mut().expect("active segment opened above");
        // A batch bigger than a fresh segment grows it before writing; the
        // growth extends preallocated slack, never the valid prefix.
        let end = append_start + total;
        if end > active.mmap.len() as u64 {
            active.file.set_len(end)?;
            active.mmap = unsafe { MmapMut::map_mut(&active.file)? };
        }
        let mut offset = append_start;
        for (_, bytes) in &encoded {
            let data_len = bytes.len() as u64;
            let framed = record_total_len(bytes.len());
            let mmap_len = active.mmap.len();
            let dst = &mut active.mmap[offset as usize..(offset + framed) as usize];
            debug_assert!(offset + framed <= mmap_len as u64);
            dst[0..4].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
            // CRC chains the previous record (or the seed): it covers the
            // length prefix, the payload and the zero padding.
            let pad = (framed - RECORD_HEADER_LEN - data_len) as usize;
            crc = chain_crc(crc, &(bytes.len() as u32).to_le_bytes(), bytes, pad);
            dst[4..8].copy_from_slice(&crc.to_le_bytes());
            dst[8..8 + bytes.len()].copy_from_slice(bytes);
            for b in &mut dst[8 + bytes.len()..] {
                *b = 0;
            }
            offset += framed;
        }
        // One visibility publish + one flush for the whole batch.
        active
            .mmap
            .flush_range(append_start as usize, total as usize)?;
        active.file.sync_all()?;
        state.active_last_lsn = state.active_last_lsn.max(last_txn);
        state.active_valid_len = Some(end);
        state.active_crc = Some(crc);
        drop(state);
        // Rebuild the spare outside the WAL state lock: preallocating and
        // fsyncing 64 MiB must not block readers inspecting segment names.
        if rolled {
            let _ = make_spare(&self.dir, self.segment_bytes);
        }
        self.appended_bytes
            .fetch_add(total, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Roll over to a fresh segment while holding the state lock. The closed
    /// segment keeps its exact valid prefix; the spare file is consumed when
    /// capacity matches, otherwise the segment is preallocated directly.
    fn roll_over_locked(
        &self,
        state: &mut WalState,
        first_txn: u64,
        batch_total: u64,
    ) -> Result<bool> {
        // Seal the closed segment: truncate it to its valid prefix so no
        // torn tail or preallocated slack survives inside a segment that
        // replay will later treat as a middle segment (where any damage is
        // corruption, never a tolerated tail). Batches were already flushed
        // on append; the truncate only drops slack, never valid records.
        // The handle stays in place until the replacement is fully prepared,
        // so a failure here cannot strand an untracked active segment.
        if !state.segments.is_empty() {
            let closed_valid = state.active_valid_len.unwrap_or(0);
            if let Some(active) = state.active.as_ref() {
                active.file.set_len(closed_valid)?;
                active.file.sync_all()?;
            }
            state.sealed_valid.push(closed_valid);
        }
        let name = next_segment_name(&state.segments, first_txn);
        let capacity = normalize_capacity(self.segment_bytes.max(SEG_HEADER_LEN + batch_total));
        let path = segment_path(&self.dir, name);
        let seed = random_seed();
        let header = (|| -> Result<()> {
            if !take_spare(&self.dir, &path, capacity)? {
                create_segment_file(&path, capacity)?;
            }
            // The directory entry has to be durable before any record in it
            // can be considered committed.
            File::open(&self.dir)?.sync_all()?;
            write_segment_header(&path, seed, name)?;
            File::open(&path)?.sync_all()?;
            Ok(())
        })();
        if let Err(error) = header {
            // A named but unheaded segment would block every later rollover
            // with the same name. Remove it; the previous active handle is
            // still registered and serving.
            remove_segment(&self.dir, name)?;
            return Err(error);
        }
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        state.segments.push(name);
        state.active = Some(ActiveSegment { mmap, file });
        state.active_last_lsn = 0;
        state.active_valid_len = Some(SEG_HEADER_LEN);
        state.active_crc = Some(seed);
        Ok(true)
    }

    /// Replay all records in order, invoking `f` for each decoded transaction.
    ///
    /// Returns the highest `txn_id` seen. A truncated final record (crash
    /// mid-append), a zero prefix (preallocated slack) or a checksum mismatch
    /// on the final record stops the replay silently — nothing after a torn
    /// record can be trusted; anything else that is malformed is reported as
    /// `CorruptData`.
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
        let mut valid_len = SEG_HEADER_LEN;
        let mut tail_crc = None;
        for (index, name) in segments.iter().enumerate() {
            let last_segment = index + 1 == segments.len();
            let path = segment_path(&self.dir, *name);
            let mapped = map_segment(&path)?;
            let bytes: &[u8] = mapped.as_deref().unwrap_or(&[]);
            let (segment_last_txn, segment_valid_len, segment_crc) =
                read_segment_bytes(bytes, last_segment, after, &path, *name, &mut f, last_txn)?;
            // Segment names are lower bounds on their first transaction. A
            // closed segment must end exactly before the next one starts:
            // a gap means a segment was lost, an overlap means a name was
            // reused. Either way replaying only the surviving segments would
            // silently drop or duplicate committed transactions. LSNs are
            // allocated densely by the commit queue, so holes cannot occur.
            if !last_segment {
                if let Some(next) = segments.get(index + 1) {
                    let expected = segment_last_txn.checked_add(1).unwrap_or(u64::MAX);
                    if *next != expected {
                        return Err(VectorSearchError::CorruptData(format!(
                            "wal segments are not contiguous: {} ends at lsn {}, next segment starts at {}",
                            segment_file_name(*name),
                            segment_last_txn,
                            segment_file_name(*next),
                        )));
                    }
                }
            }
            last_txn = segment_last_txn;
            if last_segment {
                valid_len = segment_valid_len;
                tail_crc = Some(segment_crc);
            }
        }
        // The log's highest lsn and valid tail are now known: the first
        // repairs the truncation decision, the second keeps future appends
        // behind every torn record we just skipped, and the third lets the
        // next append chain its CRC without re-walking the segment.
        let mut state = self.state.lock();
        state.active_last_lsn = last_txn;
        if !segments.is_empty() {
            state.active_valid_len = Some(valid_len);
            state.active_crc = tail_crc;
        }
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
            state
                .segments
                .last()
                .map_or(0, |name| name.saturating_sub(1))
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
        state.sealed_valid.clear();
        sync_dir(&self.dir)?;
        state.active = None;
        state.active_last_lsn = 0;
        state.active_valid_len = None;
        drop(state);
        self.acked_lsn.store(new_ack, AtomicOrdering::Release);
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
        let sealed = std::mem::take(&mut state.sealed_valid);
        let mut kept: Vec<u64> = Vec::with_capacity(segments.len());
        let mut kept_valid: Vec<u64> = Vec::with_capacity(sealed.len());
        let mut removed_bytes = 0u64;
        for (index, name) in segments.iter().enumerate() {
            // Valid bytes of segment `index`: sealed table for heads, the
            // live tail counter for the last segment.
            let valid = match sealed.get(index) {
                Some(valid) => *valid,
                None => state.active_valid_len.unwrap_or(0),
            };
            let covered = match segments.get(index + 1) {
                Some(next) => next.saturating_sub(1) <= watermark,
                None => state.active.is_none() || state.active_last_lsn <= watermark,
            };
            if covered {
                removed_bytes += valid;
                remove_segment(&self.dir, *name)?;
                if index + 1 == segments.len() {
                    // The tail segment is gone: retire the handle so the next
                    // append creates a new one.
                    state.active = None;
                    state.active_last_lsn = 0;
                    state.active_valid_len = None;
                }
            } else {
                kept.push(*name);
                if index + 1 < segments.len() {
                    kept_valid.push(valid);
                }
            }
        }
        state.segments = kept;
        state.sealed_valid = kept_valid;
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

    /// Valid data bytes currently held in the log. Drives the size-based
    /// checkpoint schedule.
    pub fn appended_bytes(&self) -> u64 {
        self.appended_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
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

    /// Valid prefix length of the active segment, if any. Test-only hook so
    /// crash-injection tests can place torn bytes exactly at the frontier.
    #[cfg(test)]
    pub fn active_valid_len_for_test(&self) -> Option<u64> {
        self.state.lock().active_valid_len
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

/// Framed length of a record holding `data_len` payload bytes, 8-aligned.
fn record_total_len(data_len: usize) -> u64 {
    ((RECORD_HEADER_LEN as usize + data_len + RECORD_ALIGN as usize - 1) / RECORD_ALIGN as usize
        * RECORD_ALIGN as usize) as u64
}

/// Normalize a segment capacity: at least header plus one minimal record,
/// 8-byte aligned.
fn normalize_capacity(segment_bytes: u64) -> u64 {
    let min = SEG_HEADER_LEN + RECORD_ALIGN;
    let capacity = segment_bytes.max(1).max(min);
    (capacity + RECORD_ALIGN - 1) / RECORD_ALIGN * RECORD_ALIGN
}

fn random_seed() -> u32 {
    // Cheap non-cryptographic seed: address entropy plus time. Uniqueness per
    // segment is what matters (a reused file fails CRC), not randomness.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    std::time::SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    (hasher.finish() & 0xffff_ffff) as u32
}

/// Chained CRC: `prev` is the previous record's CRC (or the segment seed),
/// covering the length prefix, the payload and the zero padding.
fn chain_crc(prev: u32, len_le: &[u8; 4], data: &[u8], pad: usize) -> u32 {
    let mut hasher = crc32fast::Hasher::new_with_initial(prev);
    hasher.update(len_le);
    hasher.update(data);
    if pad > 0 {
        static ZEROES: [u8; 7] = [0; 7];
        hasher.update(&ZEROES[..pad]);
    }
    hasher.finalize()
}

/// Read one mapped segment, feeding transactions to `f`. Returns the highest
/// lsn seen so far and the end of its valid prefix. Records are framed, so a
/// sequential walk over the mapping is enough to resume at any floor without
/// buffering decoded transactions.
fn read_segment_bytes(
    bytes: &[u8],
    last_segment: bool,
    after: Option<u64>,
    path: &Path,
    name: u64,
    f: &mut impl FnMut(&WalTxn) -> Result<()>,
    mut last_txn: u64,
) -> Result<(u64, u64, u32)> {
    if bytes.len() < SEG_HEADER_LEN as usize {
        if last_segment {
            return Ok((last_txn, 0, 0));
        }
        return Err(VectorSearchError::CorruptData(format!(
            "wal segment {} is shorter than its header",
            path.display()
        )));
    }
    if &bytes[0..4] != SEG_MAGIC {
        if last_segment && bytes.iter().all(|b| *b == 0) {
            // Fully preallocated but never headed: a crash between file
            // creation and header write. Treat as an empty tail; the next
            // append rewrites the header.
            return Ok((last_txn, 0, 0));
        }
        return Err(VectorSearchError::CorruptData(format!(
            "wal segment {} has a bad magic",
            path.display()
        )));
    }
    if u32::from_le_bytes(bytes[4..8].try_into().expect("version slice")) != SEG_VERSION {
        return Err(VectorSearchError::CorruptData(format!(
            "wal segment {} has an unsupported version",
            path.display()
        )));
    }
    let seed = u32::from_le_bytes(bytes[8..12].try_into().expect("seed slice"));
    let first_lsn = u64::from_le_bytes(bytes[12..20].try_into().expect("first-lsn slice"));
    if first_lsn != name {
        return Err(VectorSearchError::CorruptData(format!(
            "wal segment {} header first-lsn {first_lsn} disagrees with its name",
            path.display()
        )));
    }
    let mut crc = seed;
    let mut offset = SEG_HEADER_LEN;
    let len = bytes.len() as u64;
    while offset + RECORD_HEADER_LEN <= len {
        let record_len = u32::from_le_bytes(
            bytes[offset as usize..offset as usize + 4]
                .try_into()
                .expect("len slice"),
        );
        if record_len == 0 {
            // Preallocated slack: end of the valid prefix.
            break;
        }
        if record_len > MAX_RECORD_LEN {
            return Err(VectorSearchError::CorruptData(format!(
                "wal record length {record_len} exceeds limit"
            )));
        }
        let framed = record_total_len(record_len as usize);
        if offset + framed > len {
            if last_segment {
                break; // truncated trailing record from a crash
            }
            return Err(VectorSearchError::CorruptData(format!(
                "wal segment {} ends mid-record before the log does",
                segment_file_name(name)
            )));
        }
        let data_start = (offset + RECORD_HEADER_LEN) as usize;
        let data_end = data_start + record_len as usize;
        let pad = (framed - RECORD_HEADER_LEN - record_len as u64) as usize;
        // Padding is always written as zeros; non-zero padding is damage,
        // handled exactly like a checksum mismatch below.
        let pad_ok = bytes[data_end..data_end + pad].iter().all(|b| *b == 0);
        let mut len_le = [0u8; 4];
        len_le.copy_from_slice(&bytes[offset as usize..offset as usize + 4]);
        let expected = chain_crc(crc, &len_le, &bytes[data_start..data_end], pad);
        let stored = u32::from_le_bytes(
            bytes[offset as usize + 4..offset as usize + 8]
                .try_into()
                .expect("crc slice"),
        );
        if !pad_ok || expected != stored {
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
        crc = stored;
        let txn: WalTxn = postcard::from_bytes(&bytes[data_start..data_end])?;
        if after.is_none_or(|lsn| txn.txn_id > lsn) {
            f(&txn)?;
        }
        last_txn = txn.txn_id.max(last_txn);
        offset += framed;
    }
    Ok((last_txn, offset, crc))
}

/// Map one segment for reading. A zero-length file is an empty view rather
/// than a mapping error, which keeps a crash-created segment tolerable.
fn map_segment(path: &Path) -> Result<Option<Mmap>> {
    let file = File::open(path)?;
    if file.metadata()?.len() == 0 {
        return Ok(None);
    }
    Ok(Some(unsafe { Mmap::map(&file)? }))
}

/// Lenient tail walk for open: like replay of the last segment (zero prefix,
/// truncation and checksum mismatch all end the prefix silently), but without
/// decoding. Returns the valid prefix and the chaining CRC at it (`None` when
/// the file cannot be read at all).
fn walk_tail_prefix(dir: &Path, name: u64) -> (u64, Option<u32>) {
    let Ok(Some(bytes)) = map_segment(&segment_path(dir, name)) else {
        return (0, None);
    };
    if bytes.len() < SEG_HEADER_LEN as usize
        || &bytes[0..4] != SEG_MAGIC
        || u32::from_le_bytes(bytes[4..8].try_into().expect("version slice")) != SEG_VERSION
        || u64::from_le_bytes(bytes[12..20].try_into().expect("first-lsn slice")) != name
    {
        return (0, None);
    }
    let seed = u32::from_le_bytes(bytes[8..12].try_into().expect("seed slice"));
    let mut crc = seed;
    let mut offset = SEG_HEADER_LEN;
    let len = bytes.len() as u64;
    while offset + RECORD_HEADER_LEN <= len {
        let record_len = u32::from_le_bytes(
            bytes[offset as usize..offset as usize + 4]
                .try_into()
                .expect("len slice"),
        );
        if record_len == 0 || record_len > MAX_RECORD_LEN {
            break;
        }
        let framed = record_total_len(record_len as usize);
        if offset + framed > len {
            break;
        }
        let data_start = (offset + RECORD_HEADER_LEN) as usize;
        let data_end = data_start + record_len as usize;
        let pad = (framed - RECORD_HEADER_LEN - record_len as u64) as usize;
        let pad_ok = bytes[data_end..data_end + pad].iter().all(|b| *b == 0);
        let mut len_le = [0u8; 4];
        len_le.copy_from_slice(&bytes[offset as usize..offset as usize + 4]);
        let expected = chain_crc(crc, &len_le, &bytes[data_start..data_end], pad);
        let stored = u32::from_le_bytes(
            bytes[offset as usize + 4..offset as usize + 8]
                .try_into()
                .expect("crc slice"),
        );
        if !pad_ok || expected != stored {
            break;
        }
        crc = stored;
        offset += framed;
    }
    (offset, Some(crc))
}

/// Length-walk a segment file without verifying checksums: used at open to
/// recover valid prefixes and the valid-byte total. Never fails; any damage
/// is reported later by replay with its segment context.
fn walk_valid_prefix(dir: &Path, name: u64) -> Result<u64> {
    let Some(bytes) = map_segment(&segment_path(dir, name))? else {
        return Ok(0);
    };
    if bytes.len() < SEG_HEADER_LEN as usize
        || &bytes[0..4] != SEG_MAGIC
        || u32::from_le_bytes(bytes[4..8].try_into().expect("version slice")) != SEG_VERSION
    {
        return Ok(0);
    }
    if u64::from_le_bytes(bytes[12..20].try_into().expect("first-lsn slice")) != name {
        return Ok(0);
    }
    let mut offset = SEG_HEADER_LEN;
    let len = bytes.len() as u64;
    while offset + RECORD_HEADER_LEN <= len {
        let record_len = u32::from_le_bytes(
            bytes[offset as usize..offset as usize + 4]
                .try_into()
                .expect("len slice"),
        );
        if record_len == 0 || record_len > MAX_RECORD_LEN {
            break;
        }
        let framed = record_total_len(record_len as usize);
        if offset + framed > len {
            break;
        }
        offset += framed;
    }
    Ok(offset)
}

/// Create and preallocate a segment file with a valid (zeroed) body.
fn create_segment_file(path: &Path, capacity: u64) -> Result<()> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)?;
    file.set_len(capacity)?;
    file.sync_all()?;
    drop(file);
    Ok(())
}

/// Write the segment header: magic, version, seed and first lsn.
fn write_segment_header(path: &Path, seed: u32, first_lsn: u64) -> Result<()> {
    let mut header = [0u8; SEG_HEADER_LEN as usize];
    header[0..4].copy_from_slice(&SEG_MAGIC);
    header[4..8].copy_from_slice(&SEG_VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&seed.to_le_bytes());
    header[12..20].copy_from_slice(&first_lsn.to_le_bytes());
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.write_all(&header)?;
    file.sync_all()?;
    Ok(())
}

/// Consume the prebuilt spare file when its capacity matches.
fn take_spare(dir: &Path, target: &Path, capacity: u64) -> Result<bool> {
    let spare = dir.join(SPARE_NAME);
    let meta = match std::fs::metadata(&spare) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e.into()),
    };
    if meta.len() != capacity {
        return Ok(false);
    }
    match std::fs::rename(&spare, target) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Prebuild the spare segment file for the next rollover. Best-effort: a
/// missing spare only means the next rollover creates its file directly.
/// A stale spare from a previous capacity is replaced.
fn make_spare(dir: &Path, capacity: u64) -> Result<()> {
    let spare = dir.join(SPARE_NAME);
    if spare.exists() {
        if std::fs::metadata(&spare)
            .map(|meta| meta.len())
            .unwrap_or(0)
            == capacity
        {
            return Ok(());
        }
        let _ = std::fs::remove_file(&spare);
    }
    let tmp = dir.join(".spare.tmp");
    let _ = std::fs::remove_file(&tmp);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&tmp)?;
    file.set_len(capacity)?;
    file.sync_all()?;
    drop(file);
    match std::fs::rename(&tmp, &spare) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let _ = std::fs::remove_file(&tmp);
        }
        Err(e) => return Err(e.into()),
    }
    sync_dir(dir)
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

        // Corrupt the tail: flip a payload byte of the last record so its
        // checksum no longer matches. The replay must stop after the valid
        // prefix instead of applying torn data.
        let segment = segment_path(&path, 1);
        let valid = wal.active_valid_len_for_test().unwrap();
        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&segment)
                .unwrap();
            let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
            let last = (valid - 1) as usize;
            mmap[last] ^= 0xFF;
            mmap.flush().unwrap();
        }

        let mut seen = Vec::new();
        let last = wal
            .replay(|t| {
                seen.push(t.txn_id);
                Ok(())
            })
            .unwrap();
        assert_eq!(last, 0, "the torn tail contributes no transaction");
        assert!(seen.is_empty());
    }

    #[test]
    fn test_appends_after_a_torn_tail_are_replayed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let first = wal(&path);
        first.append(&txn(1, 10)).unwrap();
        let valid = first.active_valid_len_for_test().unwrap();
        {
            let segment = segment_path(&path, 1);
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&segment)
                .unwrap();
            let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
            mmap[valid as usize] = 0xFF;
            mmap.flush().unwrap();
        }
        drop(first);

        let recovered = wal(&path);
        assert_eq!(recovered.replay(|_| Ok(())).unwrap(), 1);
        recovered.append(&txn(2, 20)).unwrap();
        drop(recovered);

        let reopened = wal(&path);
        let (txns, last) = reopened.read_all_from(0).unwrap();
        assert_eq!(last, 2);
        assert_eq!(
            txns.iter().map(|t| t.txn_id).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn test_torn_tail_does_not_survive_a_rollover() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let first = wal(&path);
        first.append(&txn(1, 10)).unwrap();
        // Extend the file past the valid prefix and plant a torn record
        // there. A reopen learns the valid prefix; the next append must
        // build on that prefix *before* closing the segment, or the stale
        // bytes become a corrupt middle segment on the following reopen.
        let valid = first.active_valid_len_for_test().unwrap();
        {
            use std::io::{Seek, SeekFrom};
            let segment = segment_path(&path, 1);
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&segment)
                .unwrap();
            file.set_len(valid + 16).unwrap();
            file.seek(SeekFrom::Start(valid)).unwrap();
            file.write_all(&[0xFF]).unwrap();
            file.sync_all().unwrap();
        }
        drop(first);

        let recovered = Wal::open_or_create(&path, 40).unwrap();
        assert_eq!(recovered.replay(|_| Ok(())).unwrap(), 1);
        recovered.append(&txn(2, 20)).unwrap();
        drop(recovered);

        let reopened = Wal::open_or_create(&path, DEFAULT_SEGMENT_BYTES).unwrap();
        let (txns, last) = reopened.read_all_from(0).unwrap();
        assert_eq!(last, 2);
        assert_eq!(
            txns.iter().map(|t| t.txn_id).collect::<Vec<_>>(),
            vec![1, 2]
        );
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
        let file_len = wal.active_valid_len_for_test().unwrap();
        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&segment)
                .unwrap();
            let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
            let last = (file_len - 1) as usize;
            mmap[last] ^= 0xFF;
            mmap.flush().unwrap();
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
        // Small segments force a rollover every couple of transactions.
        let wal = Wal::open_or_create(&path, 128).unwrap();
        let ids = append_until_segments(&wal, 2);
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
        assert_eq!(last, *ids.last().unwrap());
        assert_eq!(seen, ids, "replay follows segment order");

        // A reopen must find the same log through the directory listing.
        drop(wal);
        let reopened = Wal::open_or_create(&path, 128).unwrap();
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
            ids.len() as u64
        );
        assert_eq!(count, ids.len());
    }

    #[test]
    fn test_missing_wal_segment_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let wal = Wal::open_or_create(&path, 128).unwrap();
        let ids = append_until_segments(&wal, 3);
        let names = wal.segment_names();
        assert!(names.len() >= 3, "need several segments, got {names:?}");
        // The untouched log must be valid; only the removed segment should
        // trigger the continuity check.
        assert_eq!(wal.replay(|_| Ok(())).unwrap(), *ids.last().unwrap());
        drop(wal);

        let missing = names[names.len() / 2];
        std::fs::remove_file(segment_path(&path, missing)).unwrap();
        let reopened = Wal::open_or_create(&path, 128).unwrap();
        let err = reopened.replay(|_| Ok(())).unwrap_err();
        assert!(
            err.to_string().contains("not contiguous"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_truncate_through_removes_only_covered_segments() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let wal = Wal::open_or_create(&path, 128).unwrap();
        let ids = append_until_segments(&wal, 3);
        let names = wal.segment_names();
        assert!(names.len() >= 3, "need several segments, got {names:?}");
        // Highest lsn the log holds is learned by replay.
        assert_eq!(wal.replay(|_| Ok(())).unwrap(), *ids.last().unwrap());
        let before = wal.appended_bytes();

        // A water mark at the second segment's first lsn keeps it and
        // everything after it, and drops only the first segment.
        let watermark = names[1];
        wal.truncate_through(watermark.saturating_sub(1)).unwrap();
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
        assert_eq!(*seen.last().unwrap(), *ids.last().unwrap());
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

        let (txns, last) = wal.read_all_from(wal.acked_lsn()).unwrap();
        assert_eq!(last, 3);
        assert_eq!(txns.iter().map(|t| t.txn_id).collect::<Vec<_>>(), vec![3]);

        // The watermark survives reopen, so a resurrected old segment is
        // skipped on the next recovery too.
        drop(wal);
        let reopened = Wal::open_or_create(&path, DEFAULT_SEGMENT_BYTES).unwrap();
        assert_eq!(reopened.acked_lsn(), 2);
        assert_eq!(reopened.read_all_from(reopened.acked_lsn()).unwrap().0.len(), 1);
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
        let wal = Wal::open_or_create(&path, 128).unwrap();
        append_until_segments(&wal, 2);
        let names = wal.segment_names();
        assert!(names.len() >= 2, "need several segments, got {names:?}");
        drop(wal);

        // Corrupt the first segment while later segments still follow it: the
        // replay cannot treat that as a torn tail, because the log evidently
        // continues behind it.
        let segment = segment_path(&path, names[0]);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&segment)
            .unwrap();
        let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        // Flip a byte inside the first record's payload: the length prefix
        // still parses but the chained CRC no longer matches.
        mmap[(SEG_HEADER_LEN + RECORD_HEADER_LEN) as usize] ^= 0xFF;
        mmap.flush().unwrap();
        drop(mmap);

        let wal = Wal::open_or_create(&path, 128).unwrap();
        let error = wal.replay(|_| Ok(())).unwrap_err();
        assert!(
            matches!(error, VectorSearchError::CorruptData(_)),
            "mid-log damage must surface as CorruptData, got {error:?}"
        );
    }

    /// Append until the log holds at least `wanted` segments; returns the
    /// committed lsn sequence. Encoding-size agnostic: rollover timing
    /// depends on the framed record size, so tests grow the log instead of
    /// assuming a fixed batch count per segment.
    fn append_until_segments(wal: &Wal, wanted: usize) -> Vec<u64> {
        let mut id = 0u64;
        while wal.segment_names().len() < wanted {
            id += 1;
            assert!(id < 100, "rollover never happened");
            wal.append(&txn(id, id * 10)).unwrap();
        }
        (1..=id).collect()
    }

    #[test]
    fn test_rollover_reuses_prebuilt_spare() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        // A spare is prebuilt after the first segment is created.
        let wal = Wal::open_or_create(&path, 128).unwrap();
        let ids = append_until_segments(&wal, 2);
        assert!(wal.segment_names().len() > 1);
        assert!(
            path.join(SPARE_NAME).exists(),
            "a spare segment must be prebuilt after a rollover"
        );
        let mut seen = Vec::new();
        let last = wal
            .replay(|t| {
                seen.push(t.txn_id);
                Ok(())
            })
            .unwrap();
        assert_eq!(last, *ids.last().unwrap());
        assert_eq!(seen, ids);
    }

    #[test]
    fn test_header_first_lsn_mismatch_is_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let wal = wal(&path);
        wal.append(&txn(5, 10)).unwrap();
        drop(wal);

        // Rewrite the header's first-lsn so it disagrees with the file name.
        let segment = segment_path(&path, 5);
        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&segment)
                .unwrap();
            let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
            mmap[12..20].copy_from_slice(&99u64.to_le_bytes());
            mmap.flush().unwrap();
        }

        let wal = Wal::open_or_create(&path, DEFAULT_SEGMENT_BYTES).unwrap();
        let error = wal.replay(|_| Ok(())).unwrap_err();
        assert!(
            matches!(error, VectorSearchError::CorruptData(_)),
            "header/name disagreement must be CorruptData, got {error:?}"
        );
    }

    #[test]
    fn test_mapped_records_match_replay_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal");
        let wal = Wal::open_or_create(&path, 128).unwrap();
        let ids = append_until_segments(&wal, 2);
        // Walk every segment mapping directly and check that framed records
        // decode to the next lsn in log order.
        let mut expected = 1u64;
        for name in wal.segment_names() {
            let mmap = map_segment(&segment_path(&path, name))
                .unwrap()
                .expect("segment has a header");
            let mut offset = SEG_HEADER_LEN as usize;
            while offset + RECORD_HEADER_LEN as usize <= mmap.len() {
                let record_len =
                    u32::from_le_bytes(mmap[offset..offset + 4].try_into().expect("len slice"));
                if record_len == 0 {
                    break;
                }
                let framed = record_total_len(record_len as usize) as usize;
                let txn: WalTxn = postcard::from_bytes(
                    &mmap[offset + RECORD_HEADER_LEN as usize
                        ..offset + RECORD_HEADER_LEN as usize + record_len as usize],
                )
                .unwrap();
                assert_eq!(txn.txn_id, expected, "record index order");
                expected += 1;
                offset += framed;
            }
        }
        assert_eq!(expected, ids.len() as u64 + 1);
    }
}
