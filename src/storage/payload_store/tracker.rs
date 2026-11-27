//! Slot-to-pointer mapping for the payload store.
//!
//! The tracker is a flat array mapping each slot to an optional
//! `ValuePointer` (page_id, block_offset, length). It is persisted to
//! `tracker.bin` and loaded into memory for O(1) lookups.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use crate::error::{Result, VectorSearchError};

use super::{StoreConfig, TRACKER_MAGIC};

/// File format version.
const TRACKER_VERSION: u32 = 1;

/// Header: magic(4) + version(4) + slot_count(4) + block_size(4) + page_size(4)
/// + change counter (8) = 28 bytes.
const TRACKER_HEADER_LEN: usize = 28;

/// Append-only changes since the last published `tracker.bin`.
const TRACKER_JOURNAL_FILE: &str = "tracker.journal.bin";

/// One record: version(8) + lsn(8) + slot(4) + pointer(12) + crc32(4).
const JOURNAL_RECORD_LEN: usize = 36;
const JOURNAL_CRC_LEN: usize = 4;

/// A pointer to a contiguous range of blocks within a data page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ValuePointer {
    /// Page containing the data.
    pub page_id: u32,
    /// Starting block offset within the page.
    pub block_offset: u32,
    /// Payload length in bytes.
    pub length: u32,
}

impl ValuePointer {
    /// A null/empty pointer.
    pub const EMPTY: Self = Self {
        page_id: u32::MAX,
        block_offset: 0,
        length: 0,
    };

    pub fn new(page_id: u32, block_offset: u32, length: u32) -> Self {
        Self {
            page_id,
            block_offset,
            length,
        }
    }

    /// Create an empty pointer.
    pub fn empty() -> Self {
        Self::EMPTY
    }

    /// Whether this pointer refers to any data.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// The in-memory tracker: a Vec mapping slot → ValuePointer.
///
/// Persisted to `tracker.bin` with a simple binary format for fast reload.
pub struct Tracker {
    pointers: Vec<ValuePointer>,
    pub(crate) version: u64,
}

impl Tracker {
    /// Create a fresh tracker for a new store.
    pub fn create(path: &Path, config: &StoreConfig) -> Result<Self> {
        let tracker = Self {
            pointers: Vec::new(),
            version: 0,
        };
        tracker.save(path, config)?;
        Ok(tracker)
    }

    /// Open the published pointer table and replay its append-only journal.
    ///
    /// A torn journal tail is truncated: the WAL still owns every change made
    /// since the last checkpoint, so losing a derived record can never lose
    /// committed data.
    pub fn open(path: &Path, config: &StoreConfig) -> Result<Self> {
        Self::open_inner(path, config, None)
    }

    /// Open with journal rollback for a generation whose replay floor is
    /// already known. Changes newer than the floor are uncheckpointed: the
    /// compact tracker may contain half of them, so those slots are reset and
    /// the WAL replay that follows this call restores them exactly.
    pub fn open_at_floor(path: &Path, config: &StoreConfig, floor: u64) -> Result<Self> {
        Self::open_inner(path, config, Some(floor))
    }

    fn open_inner(path: &Path, config: &StoreConfig, floor: Option<u64>) -> Result<Self> {
        let mut tracker = Self::open_base(path, config)?;
        let journal = path.join(TRACKER_JOURNAL_FILE);
        let Ok(data) = std::fs::read(&journal) else {
            return Ok(tracker);
        };
        let mut valid_len = 0usize;
        let mut rolled_back = Vec::new();
        while valid_len + JOURNAL_RECORD_LEN <= data.len() {
            let record = &data[valid_len..valid_len + JOURNAL_RECORD_LEN];
            let payload = &record[..JOURNAL_RECORD_LEN - JOURNAL_CRC_LEN];
            let expected = u32::from_le_bytes(
                record[JOURNAL_RECORD_LEN - JOURNAL_CRC_LEN..]
                    .try_into()
                    .expect("journal crc slice"),
            );
            if crate::checksum::hash(payload) != expected {
                break;
            }
            let version = u64::from_le_bytes(payload[0..8].try_into().expect("journal version"));
            let lsn = u64::from_le_bytes(payload[8..16].try_into().expect("journal lsn"));
            let slot = u32::from_le_bytes(payload[16..20].try_into().expect("journal slot"));
            let pointer = ValuePointer {
                page_id: u32::from_le_bytes(payload[20..24].try_into().expect("journal page")),
                block_offset: u32::from_le_bytes(
                    payload[24..28].try_into().expect("journal block"),
                ),
                length: u32::from_le_bytes(payload[28..32].try_into().expect("journal length")),
            };
            if version > tracker.version {
                tracker.set(slot, pointer);
            }
            if floor.is_some_and(|f| lsn > f) {
                rolled_back.push(slot);
            }
            valid_len += JOURNAL_RECORD_LEN;
        }
        for slot in rolled_back {
            tracker.set(slot, ValuePointer::empty());
        }
        if valid_len != data.len() {
            // Drop only the unusable tail before the next append; otherwise a
            // torn record would hide every later change forever.
            let file = OpenOptions::new().write(true).open(&journal)?;
            file.set_len(valid_len as u64)?;
            file.sync_all()?;
        }
        Ok(tracker)
    }

    /// Read only the compact `tracker.bin`, without the journal.
    pub(super) fn open_base(path: &Path, config: &StoreConfig) -> Result<Self> {
        let file_path = path.join("tracker.bin");
        let data = std::fs::read(&file_path)
            .map_err(|e| VectorSearchError::CorruptData(format!("cannot read tracker.bin: {e}")))?;
        Self::deserialize(&data, config)
    }

    /// Get the pointer for a slot (O(1)).
    pub fn get(&self, slot: u32) -> Option<ValuePointer> {
        self.pointers.get(slot as usize).copied()
    }

    /// Monotonic count of pointer changes, observed by the persistence tests.
    #[cfg(test)]
    fn version(&self) -> u64 {
        self.version
    }

    /// All persisted pointers, used by open-time consistency validation.
    pub(super) fn entries(&self) -> impl Iterator<Item = (u32, ValuePointer)> + '_ {
        self.pointers
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, pointer)| (slot as u32, pointer))
    }

    /// Update a slot and append the change to the journal before it becomes
    /// visible in memory.
    pub fn set_and_record(
        &mut self,
        path: &Path,
        slot: u32,
        pointer: ValuePointer,
        lsn: u64,
    ) -> Result<()> {
        let version = self.version.wrapping_add(1);
        let mut record = Vec::with_capacity(JOURNAL_RECORD_LEN);
        record.extend_from_slice(&version.to_le_bytes());
        record.extend_from_slice(&lsn.to_le_bytes());
        record.extend_from_slice(&slot.to_le_bytes());
        record.extend_from_slice(&pointer.page_id.to_le_bytes());
        record.extend_from_slice(&pointer.block_offset.to_le_bytes());
        record.extend_from_slice(&pointer.length.to_le_bytes());
        let crc = crate::checksum::hash(&record);
        record.extend_from_slice(&crc.to_le_bytes());
        debug_assert_eq!(record.len(), JOURNAL_RECORD_LEN);

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.join(TRACKER_JOURNAL_FILE))?;
        file.write_all(&record)?;
        self.set(slot, pointer);
        Ok(())
    }

    /// Set the pointer for a slot, growing the array if needed.
    pub fn set(&mut self, slot: u32, pointer: ValuePointer) {
        let idx = slot as usize;
        if idx >= self.pointers.len() {
            self.pointers.resize(idx + 1, ValuePointer::empty());
        }
        self.pointers[idx] = pointer;
        self.version = self.version.wrapping_add(1);
    }

    /// Persist the tracker to `tracker.bin`.
    pub fn save(&self, path: &Path, config: &StoreConfig) -> Result<()> {
        let data = self.serialize(config);
        let file_path = path.join("tracker.bin");
        let tmp_path = path.join("tracker.bin.tmp");

        let mut file = File::create(&tmp_path)?;
        file.write_all(&data)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp_path, &file_path)?;
        File::open(path)?.sync_all()?;
        // The compact table now covers every journal record. Clearing only
        // after the rename makes a crash between the two steps harmless: the
        // stale records carry older versions and are ignored on replay.
        let journal = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.join(TRACKER_JOURNAL_FILE))?;
        journal.sync_all()?;
        drop(journal);
        File::open(path)?.sync_all()?;
        Ok(())
    }

    /// Serialize the tracker to bytes.
    fn serialize(&self, config: &StoreConfig) -> Vec<u8> {
        let slot_count = self.pointers.len() as u32;
        let entry_size = std::mem::size_of::<ValuePointer>() as u32;
        let total_size = TRACKER_HEADER_LEN + slot_count as usize * entry_size as usize;
        let mut buf = Vec::with_capacity(total_size);

        buf.extend_from_slice(&TRACKER_MAGIC);
        buf.extend_from_slice(&TRACKER_VERSION.to_le_bytes());
        buf.extend_from_slice(&slot_count.to_le_bytes());
        buf.extend_from_slice(&(config.block_size as u32).to_le_bytes());
        buf.extend_from_slice(&(config.page_size as u32).to_le_bytes());
        buf.extend_from_slice(&self.version.to_le_bytes());

        for ptr in &self.pointers {
            buf.extend_from_slice(&ptr.page_id.to_le_bytes());
            buf.extend_from_slice(&ptr.block_offset.to_le_bytes());
            buf.extend_from_slice(&ptr.length.to_le_bytes());
        }

        debug_assert_eq!(buf.len(), total_size);
        buf
    }

    /// Deserialize the tracker from bytes.
    fn deserialize(data: &[u8], config: &StoreConfig) -> Result<Self> {
        if data.len() < TRACKER_HEADER_LEN {
            return Err(VectorSearchError::CorruptData(
                "tracker.bin too short".into(),
            ));
        }

        let magic = &data[0..4];
        if magic != TRACKER_MAGIC {
            return Err(VectorSearchError::CorruptData(format!(
                "tracker.bin bad magic: {:?}",
                magic
            )));
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != TRACKER_VERSION {
            return Err(VectorSearchError::CorruptData(format!(
                "tracker.bin unsupported version: {version}"
            )));
        }

        let slot_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let block_size = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        let page_size = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
        let version = u64::from_le_bytes(data[20..28].try_into().expect("tracker version slice"));

        if block_size != config.block_size || page_size != config.page_size {
            return Err(VectorSearchError::CorruptData(format!(
                "tracker.bin config mismatch: stored block_size={block_size} page_size={page_size}, expected block_size={} page_size={}",
                config.block_size, config.page_size,
            )));
        }

        let entry_size = std::mem::size_of::<ValuePointer>();
        let expected_data_len = TRACKER_HEADER_LEN + slot_count * entry_size;
        if data.len() < expected_data_len {
            return Err(VectorSearchError::CorruptData(format!(
                "tracker.bin truncated: expected {} bytes, got {}",
                expected_data_len,
                data.len(),
            )));
        }

        let mut pointers = Vec::with_capacity(slot_count);
        for i in 0..slot_count {
            let off = TRACKER_HEADER_LEN + i * entry_size;
            let page_id =
                u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            let block_offset =
                u32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
            let length =
                u32::from_le_bytes([data[off + 8], data[off + 9], data[off + 10], data[off + 11]]);
            pointers.push(ValuePointer {
                page_id,
                block_offset,
                length,
            });
        }

        Ok(Self { pointers, version })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> StoreConfig {
        StoreConfig {
            block_size: 128,
            page_size: 1024,
            compression: crate::types::PayloadCompression::None,
        }
    }

    #[test]
    fn test_create_and_get() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config();
        let mut tracker = Tracker::create(dir.path(), &config).unwrap();
        // Empty tracker: slot 0 has no entry.
        assert!(tracker.get(0).is_none());

        tracker.set(0, ValuePointer::new(0, 0, 100));
        tracker.set(5, ValuePointer::new(1, 3, 200));
        assert_eq!(tracker.get(0).unwrap(), ValuePointer::new(0, 0, 100));
        assert_eq!(tracker.get(5).unwrap(), ValuePointer::new(1, 3, 200));
        // Slot 3 is within the Vec range but was set to empty by resize.
        assert_eq!(tracker.get(3), Some(ValuePointer::empty()));
    }

    #[test]
    fn test_journal_replays_and_truncates_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config();
        {
            let mut tracker = Tracker::create(dir.path(), &config).unwrap();
            tracker
                .set_and_record(dir.path(), 0, ValuePointer::new(0, 1, 64), 10)
                .unwrap();
            tracker
                .set_and_record(dir.path(), 3, ValuePointer::new(1, 2, 96), 11)
                .unwrap();
            let mut journal = OpenOptions::new()
                .append(true)
                .open(dir.path().join(TRACKER_JOURNAL_FILE))
                .unwrap();
            journal.write_all(&[1, 2, 3, 4, 5]).unwrap();
        }
        let tracker = Tracker::open(dir.path(), &config).unwrap();
        assert_eq!(tracker.get(0), Some(ValuePointer::new(0, 1, 64)));
        assert_eq!(tracker.get(3), Some(ValuePointer::new(1, 2, 96)));
        let len = std::fs::metadata(dir.path().join(TRACKER_JOURNAL_FILE))
            .unwrap()
            .len();
        assert_eq!(len, (2 * JOURNAL_RECORD_LEN) as u64);
    }

    #[test]
    fn test_journal_rolls_back_slots_newer_than_the_replay_floor() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config();
        {
            let mut tracker = Tracker::create(dir.path(), &config).unwrap();
            tracker.set(0, ValuePointer::new(0, 0, 64));
            tracker.save(dir.path(), &config).unwrap();

            let record = |version: u64, lsn: u64, slot: u32, pointer: ValuePointer| {
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&version.to_le_bytes());
                bytes.extend_from_slice(&lsn.to_le_bytes());
                bytes.extend_from_slice(&slot.to_le_bytes());
                bytes.extend_from_slice(&pointer.page_id.to_le_bytes());
                bytes.extend_from_slice(&pointer.block_offset.to_le_bytes());
                bytes.extend_from_slice(&pointer.length.to_le_bytes());
                let crc = crate::checksum::hash(&bytes);
                bytes.extend_from_slice(&crc.to_le_bytes());
                bytes
            };
            let mut journal = OpenOptions::new()
                .append(true)
                .open(dir.path().join(TRACKER_JOURNAL_FILE))
                .unwrap();
            journal
                .write_all(&record(1, 5, 0, ValuePointer::new(0, 2, 64)))
                .unwrap();
            journal
                .write_all(&record(2, 7, 1, ValuePointer::new(1, 0, 96)))
                .unwrap();
        }
        let tracker = Tracker::open_at_floor(dir.path(), &config, 6).unwrap();
        assert_eq!(tracker.get(0), Some(ValuePointer::new(0, 0, 64)));
        assert_eq!(tracker.get(1), Some(ValuePointer::empty()));
    }

    #[test]
    fn test_persist_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config();
        {
            let mut tracker = Tracker::create(dir.path(), &config).unwrap();
            tracker.set(0, ValuePointer::new(0, 5, 128));
            tracker.set(10, ValuePointer::new(2, 0, 512));
            tracker.save(dir.path(), &config).unwrap();
        }
        {
            let tracker = Tracker::open(dir.path(), &config).unwrap();
            assert_eq!(tracker.get(0).unwrap(), ValuePointer::new(0, 5, 128));
            assert_eq!(tracker.get(10).unwrap(), ValuePointer::new(2, 0, 512));
            assert_eq!(tracker.version(), 2);
        }
    }
}
