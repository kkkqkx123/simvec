//! Lock-free tombstone table held as fixed-size immutable chunks.
//!
//! The store publishes the table through an `ArcSwap`; readers snapshot the
//! chunk list and scan without locks. Setting one bit clones a single
//! 8 KiB chunk instead of the whole bitmap, whose size grows with collection
//! capacity, so scan cost is unchanged and updates stay cheap.
//!
//! # Durability
//!
//! The table is the only record of a logical delete once the WAL has been
//! truncated, so it is persisted as `tombstones.bin` at every checkpoint and
//! read back on open. Losing it would resurrect deleted points, which is why
//! a corrupt file is a hard error rather than a silent fallback to "empty".
//!
//! ```text
//! [0..4)   magic          "VSTB"
//! [4..8)   version        u32 = 1
//! [8..12)  crc32          over the bit bytes
//! [12..20) slot_capacity  u64
//! [20..)   bit bytes      ceil(slot_capacity / 8) bytes, bit N = slot N
//! ```

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use bitvec::prelude::*;

use crate::error::{Result, VectorSearchError};

use super::segments::segment_dir;

/// File name of the persisted tombstone table.
pub(crate) const TOMBSTONE_FILE: &str = "tombstones.bin";

const TOMBSTONE_MAGIC: [u8; 4] = *b"VSTC";
const TOMBSTONE_VERSION: u32 = 1;
const TOMBSTONE_HEADER_LEN: usize = 20;

/// Bits per tombstone chunk (8 KiB of bitmap).
const TOMBSTONE_CHUNK_BITS: usize = 1 << 16;

/// Tombstone table: slot -> deleted flag.
#[derive(Debug, Default)]
pub(crate) struct TombstoneBits {
    chunks: Vec<Arc<BitVec>>,
}

impl TombstoneBits {
    pub(super) fn new(slot_capacity: usize) -> Self {
        let chunk_count = slot_capacity.div_ceil(TOMBSTONE_CHUNK_BITS).max(1);
        Self {
            chunks: (0..chunk_count)
                .map(|_| Arc::new(bitvec::bitvec![0; TOMBSTONE_CHUNK_BITS]))
                .collect(),
        }
    }

    pub(crate) fn from_bits(bits: BitVec) -> Self {
        let mut chunks = Vec::new();
        for chunk in bits.chunks(TOMBSTONE_CHUNK_BITS) {
            chunks.push(Arc::new(chunk.to_bitvec()));
        }
        if chunks.is_empty() {
            chunks.push(Arc::new(bitvec::bitvec![0; TOMBSTONE_CHUNK_BITS]));
        }
        Self { chunks }
    }

    pub(crate) fn bit(&self, slot: usize) -> bool {
        let (chunk, offset) = (slot / TOMBSTONE_CHUNK_BITS, slot % TOMBSTONE_CHUNK_BITS);
        match self.chunks.get(chunk) {
            Some(c) => c.as_bitslice()[offset],
            None => false,
        }
    }

    pub(crate) fn count_ones(&self) -> u64 {
        self.chunks
            .iter()
            .map(|c| c.as_bitslice().count_ones() as u64)
            .sum()
    }

    /// Copy-on-write single-bit update: only the affected chunk is cloned.
    pub(super) fn with_slot(&self, slot: usize, value: bool) -> Self {
        let (chunk, offset) = (slot / TOMBSTONE_CHUNK_BITS, slot % TOMBSTONE_CHUNK_BITS);
        let mut next = Self {
            chunks: Vec::with_capacity(self.chunks.len()),
        };
        for (index, existing) in self.chunks.iter().enumerate() {
            if index == chunk {
                let mut copy = (**existing).clone();
                if offset < copy.len() {
                    copy.set(offset, value);
                }
                next.chunks.push(Arc::new(copy));
            } else {
                next.chunks.push(Arc::clone(existing));
            }
        }
        next
    }

    /// Grow or shrink to `slot_capacity` slots, preserving existing bits.
    pub(super) fn resized(&self, slot_capacity: usize) -> Self {
        let mut bits = self.to_bits();
        bits.resize(slot_capacity, false);
        Self::from_bits(bits)
    }

    /// Flatten the chunked table into one bitmap.
    fn to_bits(&self) -> BitVec {
        let mut bits = BitVec::new();
        for chunk in &self.chunks {
            bits.extend_from_bitslice(chunk.as_bitslice());
        }
        bits
    }

    /// Persist the table to one `tombstones.bin` (tmp + fsync + rename).
    ///
    /// [`TombstoneBits::save_segment`] is the per-segment writer and reaches
    /// this function. The caller must have this durable before `meta.bin` is
    /// written and before the WAL is truncated: once the log is cleared this
    /// file is the only evidence of the deletes.
    pub(super) fn save(&self, dir: &Path, slot_capacity: u64) -> Result<()> {
        let bits = self.to_bits();
        let mut bytes = Vec::with_capacity(bits.len().div_ceil(8));
        for octet in bits.chunks(8) {
            let mut byte = 0u8;
            for (index, bit) in octet.iter().by_vals().enumerate() {
                if bit {
                    byte |= 1 << index;
                }
            }
            bytes.push(byte);
        }

        let mut payload = Vec::with_capacity(TOMBSTONE_HEADER_LEN + bytes.len());
        payload.extend_from_slice(&TOMBSTONE_MAGIC);
        payload.extend_from_slice(&TOMBSTONE_VERSION.to_le_bytes());
        payload.extend_from_slice(&crate::checksum::hash(&bytes).to_le_bytes());
        payload.extend_from_slice(&slot_capacity.to_le_bytes());
        payload.extend_from_slice(&bytes);

        let target = dir.join(TOMBSTONE_FILE);
        let temp = dir.join("tombstones.bin.tmp");
        let mut file = File::create(&temp)?;
        file.write_all(&payload)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temp, &target)?;
        let parent = dir.parent().unwrap_or(dir);
        File::open(parent)?.sync_all()?;
        Ok(())
    }

    /// Persist one tombstone bitmap per physical segment.
    pub(super) fn save_segments(
        &self,
        dir: &Path,
        segment_slots: u32,
        slot_capacity: u64,
    ) -> Result<()> {
        if segment_slots == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "segment_slots must be positive".to_string(),
            ));
        }
        let mut start = 0u64;
        while start < slot_capacity {
            self.save_segment(dir, start as u32, segment_slots, slot_capacity)?;
            start += segment_slots as u64;
        }
        Ok(())
    }

    /// Persist the bits of one physical segment.
    pub(super) fn save_segment(
        &self,
        dir: &Path,
        start: u32,
        segment_slots: u32,
        slot_capacity: u64,
    ) -> Result<()> {
        let bits = self.to_bits();
        let begin = start as usize;
        let end = (begin + segment_slots as usize).min(slot_capacity as usize);
        let mut segment = BitVec::new();
        for bit in bits[begin..end].iter().by_vals() {
            segment.push(bit);
        }
        TombstoneBits::from_bits(segment)
            .resized(end - begin)
            .save(&segment_dir(dir, start), (end - begin) as u64)?;
        Ok(())
    }

    /// Load and concatenate the per-segment tombstone bitmaps.
    pub(super) fn load_segments(
        dir: &Path,
        segment_slots: u32,
        slot_capacity: usize,
        expected_tombstones: u64,
    ) -> Result<Option<Self>> {
        if segment_slots == 0 {
            return Err(VectorSearchError::InvalidConfig(
                "segment_slots must be positive".to_string(),
            ));
        }
        let mut bits = BitVec::new();
        let mut found = false;
        let mut start = 0usize;
        while start < slot_capacity {
            let end = (start + segment_slots as usize).min(slot_capacity);
            if let Some(table) = Self::load(&segment_dir(dir, start as u32), end - start, 0)? {
                found = true;
                for index in 0..(end - start) {
                    bits.push(table.bit(index));
                }
            } else {
                bits.resize(bits.len() + (end - start), false);
            }
            start = end;
        }
        if !found && expected_tombstones == 0 {
            return Ok(None);
        }
        if bits.count_ones() as u64 != expected_tombstones {
            return Err(VectorSearchError::CorruptData(format!(
                "tombstone segment files record {} deletes, meta records {expected_tombstones}",
                bits.count_ones()
            )));
        }
        bits.resize(slot_capacity, false);
        Ok(Some(Self::from_bits(bits)))
    }

    /// Load a single `tombstones.bin` and resize it to the capacity declared by
    /// meta.
    ///
    /// [`TombstoneBits::load_segments`] is the per-segment reader and goes
    /// through this function. A missing file is only acceptable when the
    /// metadata claims no tombstones (fresh collection, or a collection that
    /// never deleted). Otherwise the file is the authority for deletes and its
    /// absence or a checksum mismatch is corruption: falling back to "nothing
    /// deleted" would silently resurrect points.
    pub(super) fn load(
        dir: &Path,
        slot_capacity: usize,
        expected_tombstones: u64,
    ) -> Result<Option<Self>> {
        let path = dir.join(TOMBSTONE_FILE);
        let Ok(mut file) = File::open(&path) else {
            if expected_tombstones > 0 {
                return Err(VectorSearchError::CorruptData(format!(
                    "{} missing but metadata records {expected_tombstones} tombstones",
                    TOMBSTONE_FILE
                )));
            }
            return Ok(None);
        };

        let mut header = [0u8; TOMBSTONE_HEADER_LEN];
        file.read_exact(&mut header).map_err(|e| {
            VectorSearchError::CorruptData(format!("{} header truncated: {e}", TOMBSTONE_FILE))
        })?;
        if header[0..4] != TOMBSTONE_MAGIC {
            return Err(VectorSearchError::CorruptData(format!(
                "{} bad magic: {:?}",
                TOMBSTONE_FILE,
                &header[0..4]
            )));
        }
        let version = u32::from_le_bytes(header[4..8].try_into().expect("version slice"));
        if version != TOMBSTONE_VERSION {
            return Err(VectorSearchError::CorruptData(format!(
                "{} unsupported version {version}",
                TOMBSTONE_FILE
            )));
        }
        let expected_crc = u32::from_le_bytes(header[8..12].try_into().expect("crc slice"));
        let stored_capacity =
            u64::from_le_bytes(header[12..20].try_into().expect("capacity slice"));

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|e| {
            VectorSearchError::CorruptData(format!("{} unreadable: {e}", TOMBSTONE_FILE))
        })?;
        let actual_crc = crate::checksum::hash(&bytes);
        if actual_crc != expected_crc {
            return Err(VectorSearchError::CorruptData(format!(
                "{} checksum mismatch: stored {expected_crc:#x}, computed {actual_crc:#x}",
                TOMBSTONE_FILE
            )));
        }
        if stored_capacity != slot_capacity as u64 {
            tracing::warn!(
                stored_capacity,
                slot_capacity,
                "tombstone file capacity differs from meta; resizing"
            );
        }

        let mut bits = BitVec::with_capacity(slot_capacity);
        for (index, byte) in bytes.iter().enumerate() {
            for shift in 0..8 {
                bits.push(byte & (1 << shift) != 0);
            }
            if (index + 1) * 8 >= slot_capacity {
                break;
            }
        }
        bits.resize(slot_capacity, false);
        Ok(Some(Self::from_bits(bits)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_preserves_bits() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut table = TombstoneBits::new(1000);
        for slot in [0usize, 7, 64, 999] {
            table = table.with_slot(slot, true);
        }
        table.save(dir.path(), 1000).expect("save");

        let loaded = TombstoneBits::load(dir.path(), 1000, 4)
            .expect("load")
            .expect("file present");
        assert_eq!(loaded.count_ones(), 4);
        for slot in 0..1000 {
            assert_eq!(table.bit(slot), loaded.bit(slot), "slot {slot}");
        }
    }

    #[test]
    fn resize_on_capacity_change() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut table = TombstoneBits::new(16);
        table = table.with_slot(3, true);
        table.save(dir.path(), 16).expect("save");

        let loaded = TombstoneBits::load(dir.path(), 64, 1)
            .expect("load")
            .expect("file present");
        assert!(loaded.bit(3));
        assert!(!loaded.bit(40));
        assert_eq!(loaded.count_ones(), 1);
    }

    #[test]
    fn missing_file_with_tombstones_is_corruption() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = TombstoneBits::load(dir.path(), 16, 2).expect_err("must reject");
        assert!(err.to_string().contains("missing"), "{err}");
    }

    #[test]
    fn checksum_rejects_bit_flip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let table = TombstoneBits::new(64);
        table.save(dir.path(), 64).expect("save");
        let path = dir.path().join(TOMBSTONE_FILE);
        let mut bytes = std::fs::read(&path).expect("read");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write");
        let err = TombstoneBits::load(dir.path(), 64, 0).expect_err("must reject");
        assert!(err.to_string().contains("checksum"), "{err}");
    }
}
