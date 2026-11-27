//! `postings/` — persisted sparse postings for exact-scan collections.
//!
//! The sparse inverted index used to be a `HashMap<term, Vec<(slot, weight)>>`
//! rebuilt on every open by deserializing the whole snapshot and re-inserting
//! every posting. This module keeps the same query semantics but moves the
//! postings themselves into bucket files that are appended to and read back
//! through a `mmap`, so restart neither rebuilds nor re-reads them and a
//! checkpoint writes only what was appended since the previous one.
//!
//! Layout:
//!
//! ```text
//! postings/<bucket>.bin   records `[u32 term][u32 count][(u32 slot, f32 w) * count]`
//!                         appended in non-decreasing term order
//! postings/manifest.bin   `{lsn, bucket lengths, term -> chain list}` (snapshot envelope)
//! ```
//!
//! A bucket covers `1 << BUCKET_SHIFT` consecutive terms, so a term names its
//! bucket arithmetically and a bucket holds at most that many distinct terms —
//! which is what makes the per-bucket chain rewrite below bounded work.
//!
//! Two kinds of state, and the split is the point:
//!
//! - **pending**: postings applied since the last flush, in memory. Bounded by
//!   [`SOFT_FLUSH_BYTES`]; exceeding it flushes early, so a long write burst
//!   cannot grow the heap without bound between checkpoints.
//! - **chains**: for each term, the list of `(bucket, offset, count)` records
//!   written so far. Appending is O(1) per term; a term with more than
//!   [`MAX_CHAINS`] records gets its bucket rewritten into one record, which
//!   is amortized against the bytes appended to that bucket.
//!
//! Durability contract: the manifest is written **only by a checkpoint**, and
//! carries the checkpoint's lsn. On open, a manifest whose lsn does not match
//! the snapshot's is stale — the bucket files may hold postings for
//! transactions the snapshot does not cover, which replay would add a second
//! time — so the whole posting set is discarded and rebuilt from the snapshot
//! plus the WAL. That is why a soft flush needs no manifest: it appends bytes
//! and nothing else, and the next open truncates the buckets back to the
//! lengths the manifest recorded.
//!
//! Deleted points leave their postings behind. That is deliberate: removing a
//! single posting would rewrite the record that contains it. Readers filter at
//! the point where the slot's liveness is already known (the scan skips
//! tombstoned slots), and compaction — which renumbers slots anyway — rebuilds
//! the index from scratch.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};

/// Terms covered by one bucket file, as a power of two exponent.
const BUCKET_SHIFT: u32 = 10;

/// Pending postings size at which the write path flushes without waiting for
/// a checkpoint.
const SOFT_FLUSH_BYTES: usize = 8 * 1024 * 1024;

/// Records a single term may accumulate before its bucket is rewritten into
/// one record per term.
const MAX_CHAINS: usize = 8;

/// File name of the term dictionary.
const MANIFEST_FILE: &str = "manifest.bin";

/// One posting on disk. `#[repr(C)]` so the `mmap` slice can be reinterpreted
/// as `&[Posting]` without copying.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct Posting {
    slot: u32,
    weight: f32,
}

/// Bytes of one posting.
const POSTING_BYTES: usize = std::mem::size_of::<Posting>();

/// `[u32 term][u32 count]` in front of every record.
const POSTING_RECORD_HEADER: usize = 8;

/// Address of one record's posting array inside its bucket file. `off` points
/// at the first `Posting`, past the `[term][count]` header.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Chain {
    off: u32,
    count: u32,
}

/// An open bucket file. The mapping grows by remapping the file after an
/// append — one `mmap` per dirty bucket per flush, not per posting.
struct Bucket {
    file: File,
    map: Option<Arc<Mmap>>,
    len: u64,
}

impl Bucket {
    fn open(path: &Path, len: u64) -> Result<Self> {
        let file = if len == 0 {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)?
        } else {
            // A bucket the manifest describes must already exist: recreating it
            // empty would silently drop postings.
            OpenOptions::new().read(true).write(true).open(path)?
        };
        let actual = file.metadata()?.len();
        if actual < len {
            return Err(VectorSearchError::CorruptData(format!(
                "posting bucket {} holds {actual} bytes but the manifest records {len}",
                path.display()
            )));
        }
        if actual > len {
            file.set_len(len)?;
        }
        let map = map_file(&file, len)?;
        Ok(Self { file, map, len })
    }

    fn remap(&mut self) -> Result<()> {
        self.map = map_file(&self.file, self.len)?;
        Ok(())
    }
}

#[derive(Default)]
struct State {
    /// bucket -> term -> record list.
    dict: HashMap<u32, HashMap<u32, Vec<Chain>>>,
    /// Number of distinct slots known for each term. Used for sparse planner
    /// selectivity; it never decides which candidates are eligible.
    doc_freq: HashMap<u32, u32>,
    /// term -> postings applied since the last flush.
    pending: HashMap<u32, Vec<Posting>>,
    pending_bytes: usize,
    /// Open bucket files, plus their durable length (what the manifest
    /// records for truncation on open).
    buckets: HashMap<u32, Bucket>,
    lens: HashMap<u32, u64>,
}

/// Persisted dictionary: everything needed to answer a lookup without reading
/// the bucket files.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Manifest {
    lsn: u64,
    buckets: Vec<(u32, u64)>,
    /// bucket -> (term -> record list)
    terms: Vec<(u32, Vec<(u32, Vec<Chain>)>)>,
    /// term -> distinct document count at the checkpoint.
    doc_freq: Vec<(u32, u32)>,
}

/// Persisted sparse postings of one space.
pub struct Postings {
    dir: PathBuf,
    state: Mutex<State>,
    /// Whether `open` loaded a dictionary instead of starting empty. A caller
    /// that restored a snapshot must re-index every live point when this is
    /// false.
    restored: bool,
}

impl Postings {
    /// Open the postings of one space.
    ///
    /// `snapshot_lsn` is the lsn of the snapshot the caller restored. Only a
    /// manifest carrying exactly that lsn describes postings the caller did
    /// not already rebuild, so anything else is discarded.
    pub fn open(dir: &Path, snapshot_lsn: Option<u64>) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let manifest = crate::storage::snapshot::read::<Manifest>(&dir.join(MANIFEST_FILE))?;
        let loaded = match manifest {
            Some(manifest) if Some(manifest.lsn) == snapshot_lsn => {
                match load_state(dir, manifest) {
                    Ok(state) => Some(state),
                    Err(error) => {
                        tracing::warn!(
                            error = %error,
                            dir = %dir.display(),
                            "discarding an unreadable postings index; it will be rebuilt"
                        );
                        None
                    }
                }
            }
            _ => None,
        };
        let restored = loaded.is_some();
        let postings = Self {
            dir: dir.to_path_buf(),
            state: Mutex::new(loaded.unwrap_or_default()),
            restored,
        };
        postings.prune_unlisted()?;
        Ok(postings)
    }

    /// Whether the dictionary came from disk (rather than starting empty).
    pub fn was_restored(&self) -> bool {
        self.restored
    }

    /// Drop bucket files the loaded dictionary does not mention. They hold
    /// postings appended after the last checkpoint, which the snapshot does
    /// not cover.
    fn prune_unlisted(&self) -> Result<()> {
        let state = self.state.lock();
        let listed: Vec<u64> = state.lens.keys().map(|bucket| *bucket as u64).collect();
        drop(state);
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if name == MANIFEST_FILE {
                continue;
            }
            let Some(stem) = name.strip_suffix(".bin") else {
                continue;
            };
            let Ok(bucket) = stem.parse::<u64>() else {
                continue;
            };
            if !listed.contains(&bucket) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
        Ok(())
    }

    /// Stage one point's postings. Flushes early once the pending window
    /// reaches [`SOFT_FLUSH_BYTES`].
    pub fn index(&self, slot: u32, indices: &[u32], values: &[f32]) -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        let flush = {
            let mut state = self.state.lock();
            for (index, value) in indices.iter().zip(values.iter()) {
                state
                    .pending
                    .entry(*index)
                    .or_default()
                    .push(Posting {
                        slot,
                        weight: *value,
                    });
            }
            state.pending_bytes += indices.len() * POSTING_BYTES;
            state.pending_bytes >= SOFT_FLUSH_BYTES
        };
        if flush {
            self.flush(None)?;
        }
        Ok(())
    }

    /// Distinct documents known for `term`: checkpointed frequencies plus
    /// the pending window. Soft-flushed bytes are included only after the
    /// next checkpoint, so the estimate can be low but never excludes a
    /// candidate from the posting path.
    pub fn document_frequency(&self, term: u32) -> usize {
        let state = self.state.lock();
        let pending = state.pending.get(&term).map_or(0, Vec::len);
        state.doc_freq.get(&term).copied().unwrap_or(0) as usize + pending
    }

    /// Largest document frequency among the query terms.
    pub fn max_document_frequency(&self, terms: &[u32]) -> usize {
        terms
            .iter()
            .map(|term| self.document_frequency(*term))
            .max()
            .unwrap_or(0)
    }

    /// Visit every posting of `term`, both flushed and pending.
    pub fn visit(&self, term: u32, mut f: impl FnMut(u32, f32)) {
        let state = self.state.lock();
        if let Some(pending) = state.pending.get(&term) {
            for posting in pending {
                f(posting.slot, posting.weight);
            }
        }
        let Some(chains) = state
            .dict
            .get(&bucket_of(term))
            .and_then(|terms| terms.get(&term))
        else {
            return;
        };
        let Some(bucket) = state.buckets.get(&bucket_of(term)) else {
            return;
        };
        let Some(map) = bucket.map.as_ref() else {
            return;
        };
        for chain in chains {
            let start = chain.off as usize;
            let end = start + chain.count as usize * POSTING_BYTES;
            if end > map.len() {
                tracing::warn!(term, "posting record out of range; skipping");
                continue;
            }
            debug_assert_eq!(start % POSTING_BYTES, 0);
            let slice = unsafe {
                std::slice::from_raw_parts(map[start..].as_ptr() as *const Posting, chain.count as usize)
            };
            for posting in slice {
                f(posting.slot, posting.weight);
            }
        }
    }

    /// Append the pending window to the bucket files. `lsn` is `Some` only at
    /// a checkpoint: it is what makes the appended postings durable, by
    /// recording them in the manifest together with the snapshot they match.
    pub fn flush(&self, lsn: Option<u64>) -> Result<()> {
        let mut state = self.state.lock();
        if state.pending.is_empty() {
            if let Some(lsn) = lsn {
                refresh_doc_freq(&mut state)?;
                write_manifest(&self.dir, &state, lsn)?;
            }
            return Ok(());
        }

        // Group by bucket, terms ascending inside each bucket so records stay
        // in term order.
        let mut by_bucket: BTreeMap<u32, Vec<(u32, Vec<Posting>)>> = BTreeMap::new();
        for (term, postings) in std::mem::take(&mut state.pending) {
            by_bucket
                .entry(bucket_of(term))
                .or_default()
                .push((term, postings));
        }
        state.pending_bytes = 0;

        let mut rewritten: Vec<u32> = Vec::new();
        for (bucket_id, mut records) in by_bucket {
            records.sort_unstable_by_key(|(term, _)| *term);
            match append_bucket(&mut state, &self.dir, bucket_id, &records) {
                Ok(needs_rewrite) => {
                    if needs_rewrite {
                        rewritten.push(bucket_id);
                    }
                }
                Err(error) => {
                    requeue(&mut state, records);
                    return Err(error);
                }
            }
        }

        rewritten.sort_unstable();
        rewritten.dedup();
        for bucket_id in rewritten {
            state.rewrite_bucket(&self.dir, bucket_id)?;
        }

        if let Some(lsn) = lsn {
            refresh_doc_freq(&mut state)?;
            write_manifest(&self.dir, &state, lsn)?;
        }
        Ok(())
    }

    /// Bytes the postings occupy on disk plus the pending window.
    pub fn disk_bytes(&self) -> u64 {
        let state = self.state.lock();
        state.lens.values().sum::<u64>() + state.pending_bytes as u64
    }

    /// Resident bytes: the dictionary plus the pending window. The postings
    /// themselves live in page cache behind the mappings.
    pub fn memory_bytes(&self) -> u64 {
        self.disk_bytes()
    }

    /// Number of distinct terms the dictionary knows.
    #[cfg(test)]
    pub fn term_count(&self) -> usize {
        self.state
            .lock()
            .dict
            .values()
            .map(|terms| terms.len())
            .sum()
    }
}

impl State {
    fn open_bucket(&mut self, dir: &Path, bucket_id: u32) -> Result<&mut Bucket> {
        if !self.buckets.contains_key(&bucket_id) {
            let len = self.lens.get(&bucket_id).copied().unwrap_or(0);
            let bucket = Bucket::open(&bucket_path(dir, bucket_id), len)?;
            self.buckets.insert(bucket_id, bucket);
        }
        Ok(self.buckets.get_mut(&bucket_id).expect("bucket opened above"))
    }

    /// Collapse one bucket into a single record per term. Bounded by the
    /// bucket's own terms (at most `1 << BUCKET_SHIFT`).
    fn rewrite_bucket(&mut self, dir: &Path, bucket_id: u32) -> Result<()> {
        // Collect first, mutate afterwards: gathering borrows the dictionary
        // and the mapping, publishing replaces both.
        let merged: BTreeMap<u32, Vec<Posting>> = {
            let Some(terms) = self.dict.get(&bucket_id) else {
                return Ok(());
            };
            let Some(bucket) = self.buckets.get(&bucket_id) else {
                return Ok(());
            };
            let Some(map) = bucket.map.as_ref() else {
                return Ok(());
            };
            let mut merged: BTreeMap<u32, Vec<Posting>> = BTreeMap::new();
            for (term, chains) in terms {
                let mut postings = Vec::new();
                for chain in chains {
                    let start = chain.off as usize;
                    let end = start + chain.count as usize * POSTING_BYTES;
                    if end > map.len() {
                        return Err(VectorSearchError::CorruptData(format!(
                            "posting record for term {term} out of range"
                        )));
                    }
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            map[start..].as_ptr() as *const Posting,
                            chain.count as usize,
                        )
                    };
                    postings.extend_from_slice(slice);
                }
                merged.insert(*term, postings);
            }
            merged
        };
        if merged.is_empty() {
            return Ok(());
        }

        let mut buffer = Vec::new();
        let mut placed: Vec<(u32, Chain)> = Vec::with_capacity(merged.len());
        for (term, postings) in &merged {
            placed.push((
                *term,
                Chain {
                    off: (buffer.len() + POSTING_RECORD_HEADER) as u32,
                    count: postings.len() as u32,
                },
            ));
            buffer.extend_from_slice(&term.to_le_bytes());
            buffer.extend_from_slice(&(postings.len() as u32).to_le_bytes());
            for posting in postings {
                buffer.extend_from_slice(&posting.slot.to_le_bytes());
                buffer.extend_from_slice(&posting.weight.to_le_bytes());
            }
        }

        // Publish through a temporary file and a rename: a crash leaves either
        // the old bucket (still described by the previous manifest) or the new
        // one.
        let tmp = dir.join(format!("{bucket_id:010}.rewrite"));
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp)?;
            std::io::Write::write_all(&mut file, &buffer)?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp, bucket_path(dir, bucket_id))?;
        File::open(dir)?.sync_all()?;

        let bucket = self.buckets.get_mut(&bucket_id).expect("bucket present");
        bucket.file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(bucket_path(dir, bucket_id))?;
        bucket.len = buffer.len() as u64;
        bucket.remap()?;
        self.lens.insert(bucket_id, bucket.len);
        let terms = self.dict.get_mut(&bucket_id).expect("terms present");
        for (term, chain) in placed {
            terms.insert(term, vec![chain]);
        }
        Ok(())
    }
}

/// Append one bucket's records and register their chains. Returns whether the
/// bucket now needs a rewrite (some term accumulated more than
/// [`MAX_CHAINS`] records).
fn append_bucket(
    state: &mut State,
    dir: &Path,
    bucket_id: u32,
    records: &[(u32, Vec<Posting>)],
) -> Result<bool> {
    let mut buffer = Vec::new();
    let mut placed: Vec<(u32, Chain)> = Vec::with_capacity(records.len());
    for (term, postings) in records {
        placed.push((
            *term,
            Chain {
                off: (buffer.len() + POSTING_RECORD_HEADER) as u32,
                count: postings.len() as u32,
            },
        ));
        buffer.extend_from_slice(&term.to_le_bytes());
        buffer.extend_from_slice(&(postings.len() as u32).to_le_bytes());
        for posting in postings {
            buffer.extend_from_slice(&posting.slot.to_le_bytes());
            buffer.extend_from_slice(&posting.weight.to_le_bytes());
        }
    }

    let base = {
        let bucket = state.open_bucket(dir, bucket_id)?;
        let base = bucket.len;
        if base + buffer.len() as u64 > u32::MAX as u64 {
            return Err(VectorSearchError::Internal(format!(
                "posting bucket {bucket_id} would exceed 4 GiB"
            )));
        }
        write_all_at(&bucket.file, &buffer, base)?;
        bucket.len = base + buffer.len() as u64;
        bucket.file.sync_all()?;
        bucket.remap()?;
        base
    };
    state.lens.insert(bucket_id, base + buffer.len() as u64);

    let mut needs_rewrite = false;
    let terms = state.dict.entry(bucket_id).or_default();
    for (term, chain) in placed {
        let entry = terms.entry(term).or_default();
        entry.push(Chain {
            off: base as u32 + chain.off,
            count: chain.count,
        });
        if entry.len() > MAX_CHAINS {
            needs_rewrite = true;
        }
    }
    Ok(needs_rewrite)
}

/// Recompute per-term distinct document counts from flushed and pending
/// postings. Checkpoint-only: the scan is bounded by the number of terms and
/// their chain lengths, which is exactly what the manifest persists.
fn refresh_doc_freq(state: &mut State) -> Result<()> {
    let mut docs: HashMap<u32, HashSet<u32>> = HashMap::new();
    for (term, postings) in &state.pending {
        let set = docs.entry(*term).or_default();
        for posting in postings {
            set.insert(posting.slot);
        }
    }
    for (bucket_id, terms) in &state.dict {
        let Some(bucket) = state.buckets.get(bucket_id) else {
            continue;
        };
        let Some(map) = bucket.map.as_ref() else {
            continue;
        };
        for (term, chains) in terms {
            let set = docs.entry(*term).or_default();
            for chain in chains {
                let start = chain.off as usize;
                let end = start + chain.count as usize * POSTING_BYTES;
                if end > map.len() {
                    return Err(VectorSearchError::CorruptData(format!(
                        "posting record for term {term} lies beyond its bucket"
                    )));
                }
                let slice = unsafe {
                    std::slice::from_raw_parts(map[start..].as_ptr() as *const Posting, chain.count as usize)
                };
                for posting in slice {
                    set.insert(posting.slot);
                }
            }
        }
    }
    state.doc_freq = docs
        .into_iter()
        .map(|(term, slots)| (term, slots.len() as u32))
        .collect();
    Ok(())
}

fn write_manifest(dir: &Path, state: &State, lsn: u64) -> Result<()> {
    let manifest = Manifest {
        lsn,
        buckets: state.lens.iter().map(|(bucket, len)| (*bucket, *len)).collect(),
        terms: state
            .dict
            .iter()
            .map(|(bucket, terms)| {
                (
                    *bucket,
                    terms
                        .iter()
                        .map(|(term, chains)| (*term, chains.clone()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect(),
        doc_freq: state
            .doc_freq
            .iter()
            .map(|(term, count)| (*term, *count))
            .collect(),
    };
    crate::storage::snapshot::write(&dir.join(MANIFEST_FILE), &manifest)
}

fn load_state(dir: &Path, manifest: Manifest) -> Result<State> {
    let mut state = State::default();
    for (bucket_id, len) in &manifest.buckets {
        if *len == 0 {
            continue;
        }
        let bucket = Bucket::open(&bucket_path(dir, *bucket_id), *len)?;
        state.buckets.insert(*bucket_id, bucket);
        state.lens.insert(*bucket_id, *len);
    }
    for (bucket_id, terms) in manifest.terms {
        let len = state.lens.get(&bucket_id).copied().unwrap_or(0);
        let mut entries = HashMap::with_capacity(terms.len());
        for (term, chains) in terms {
            for chain in &chains {
                let end = chain.off as u64 + chain.count as u64 * POSTING_BYTES as u64;
                if end > len {
                    return Err(VectorSearchError::CorruptData(format!(
                        "posting record for term {term} lies beyond its bucket"
                    )));
                }
            }
            entries.insert(term, chains);
        }
        state.dict.insert(bucket_id, entries);
    }
    state.doc_freq = manifest.doc_freq.into_iter().collect();
    Ok(state)
}

/// Put unflushed records back after a failed append, so a transient IO error
/// cannot silently drop postings from the query path.
fn requeue(state: &mut State, records: Vec<(u32, Vec<Posting>)>) {
    for (term, postings) in records {
        state.pending_bytes += postings.len() * POSTING_BYTES;
        state.pending.entry(term).or_default().extend(postings);
    }
}

fn bucket_of(term: u32) -> u32 {
    term >> BUCKET_SHIFT
}

fn bucket_path(dir: &Path, bucket_id: u32) -> PathBuf {
    dir.join(format!("{bucket_id:010}.bin"))
}

fn map_file(file: &File, len: u64) -> Result<Option<Arc<Mmap>>> {
    if len == 0 {
        return Ok(None);
    }
    let map = unsafe { MmapOptions::new().len(len as usize).map(file)? };
    Ok(Some(Arc::new(map)))
}

#[cfg(unix)]
fn write_all_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(not(unix))]
fn write_all_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut handle = file;
    handle.seek(SeekFrom::Start(offset))?;
    handle.write_all(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(dir: &Path, lsn: Option<u64>) -> Postings {
        Postings::open(dir, lsn).unwrap()
    }

    fn collect(postings: &Postings, term: u32) -> Vec<(u32, f32)> {
        let mut out = Vec::new();
        postings.visit(term, |slot, weight| out.push((slot, weight)));
        out
    }

    #[test]
    fn pending_postings_are_visible_before_a_flush() {
        let dir = tempfile::tempdir().unwrap();
        let postings = open(dir.path(), None);
        postings.index(0, &[5, 9], &[0.5, 1.5]).unwrap();
        assert_eq!(postings.document_frequency(5), 1);
        assert_eq!(postings.max_document_frequency(&[5, 9, 11]), 1);
        assert_eq!(collect(&postings, 5), vec![(0, 0.5)]);
        assert_eq!(collect(&postings, 9), vec![(0, 1.5)]);
        assert!(collect(&postings, 11).is_empty());
    }

    #[test]
    fn a_checkpointed_index_reopens_without_rebuilding() {
        let dir = tempfile::tempdir().unwrap();
        {
            let postings = open(dir.path(), None);
            postings.index(3, &[1, 2], &[0.25, 0.75]).unwrap();
            postings.index(4, &[2], &[1.0]).unwrap();
            postings.flush(Some(7)).unwrap();
        }
        // Same lsn as the snapshot: the dictionary is loaded, not rebuilt.
        let reopened = open(dir.path(), Some(7));
        assert_eq!(reopened.term_count(), 2);
        assert_eq!(collect(&reopened, 2), vec![(3, 0.75), (4, 1.0)]);
        assert_eq!(reopened.document_frequency(1), 1);
        assert_eq!(reopened.document_frequency(2), 2);
    }

    #[test]
    fn a_stale_manifest_discards_the_postings() {
        let dir = tempfile::tempdir().unwrap();
        {
            let postings = open(dir.path(), None);
            postings.index(3, &[1], &[0.25]).unwrap();
            postings.flush(Some(7)).unwrap();
        }
        // The snapshot is older than the manifest: replay would re-add the
        // same (slot, term) pairs, so the index must not be trusted.
        let reopened = open(dir.path(), Some(6));
        assert_eq!(reopened.term_count(), 0);
        assert_eq!(reopened.document_frequency(1), 0);
        assert!(collect(&reopened, 1).is_empty());
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            1,
            "only the manifest may survive a discard"
        );
    }

    #[test]
    fn a_soft_flush_keeps_the_last_checkpoint_as_the_durable_point() {
        let dir = tempfile::tempdir().unwrap();
        {
            let postings = open(dir.path(), None);
            postings.index(1, &[1], &[1.0]).unwrap();
            postings.flush(Some(4)).unwrap();
            // Appended after the checkpoint, without a manifest update.
            postings.index(2, &[1], &[2.0]).unwrap();
            postings.flush(None).unwrap();
            assert_eq!(postings.document_frequency(1), 1);
        }
        let reopened = open(dir.path(), Some(4));
        assert_eq!(
            collect(&reopened, 1),
            vec![(1, 1.0)],
            "postings appended after the checkpoint are not durable"
        );
    }

    #[test]
    fn terms_are_spread_over_buckets() {
        let dir = tempfile::tempdir().unwrap();
        let postings = open(dir.path(), None);
        let terms = [0u32, 1, (1 << BUCKET_SHIFT) - 1, 1 << BUCKET_SHIFT, 1 << 30];
        for (i, term) in terms.iter().enumerate() {
            postings.index(i as u32, std::slice::from_ref(term), &[1.0]).unwrap();
        }
        postings.flush(Some(1)).unwrap();
        let files = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_str() != Some(MANIFEST_FILE))
            .count();
        assert_eq!(
            files, 3,
            "terms 0/1/1023 share a bucket, 1024 and 1<<30 get their own"
        );
        for (i, term) in terms.iter().enumerate() {
            assert_eq!(collect(&postings, *term), vec![(i as u32, 1.0)]);
        }
    }

    #[test]
    fn repeated_flushes_collapse_into_one_record_per_term() {
        let dir = tempfile::tempdir().unwrap();
        let postings = open(dir.path(), None);
        for round in 0..(MAX_CHAINS as u32 + 4) {
            postings.index(round, &[7], &[1.0]).unwrap();
            postings.flush(None).unwrap();
        }
        postings.flush(Some(1)).unwrap();
        // The bucket was rewritten, so the term holds a single chain that
        // still yields every posting once.
        let state = postings.state.lock();
        let chains = state.dict.get(&bucket_of(7)).unwrap().get(&7).unwrap();
        assert!(
            chains.len() <= MAX_CHAINS,
            "chains must be bounded by a rewrite: {chains:?}"
        );
        drop(state);
        let weights: Vec<u32> = collect(&postings, 7).into_iter().map(|(s, _)| s).collect();
        assert_eq!(weights.len(), MAX_CHAINS as usize + 4);
        assert_eq!(weights, (0..(MAX_CHAINS as u32 + 4)).collect::<Vec<_>>());
    }

    #[test]
    fn document_frequencies_survive_a_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        {
            let postings = open(dir.path(), None);
            postings.index(1, &[7], &[1.0]).unwrap();
            postings.index(2, &[7, 8], &[1.0, 1.0]).unwrap();
            postings.flush(Some(4)).unwrap();
        }
        let reopened = open(dir.path(), Some(4));
        assert!(reopened.was_restored());
        assert_eq!(reopened.document_frequency(7), 2);
        assert_eq!(reopened.document_frequency(8), 1);
        assert_eq!(reopened.max_document_frequency(&[7, 8]), 2);
    }

    #[test]
    fn a_corrupt_manifest_makes_open_start_empty() {
        let dir = tempfile::tempdir().unwrap();
        {
            let postings = open(dir.path(), None);
            postings.index(1, &[1], &[1.0]).unwrap();
            postings.flush(Some(2)).unwrap();
        }
        // Truncate a bucket so the dictionary cannot be trusted.
        let bucket = bucket_path(dir.path(), bucket_of(1));
        let len = std::fs::metadata(&bucket).unwrap().len();
        std::fs::write(&bucket, vec![0u8; len as usize - 4]).unwrap();

        let reopened = open(dir.path(), Some(2));
        assert_eq!(reopened.term_count(), 0, "a short bucket forces a rebuild");
        assert!(collect(&reopened, 1).is_empty());
    }

}
