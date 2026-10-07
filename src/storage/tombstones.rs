//! Lock-free tombstone table held as fixed-size immutable chunks.
//!
//! The store publishes the table through an `ArcSwap`; readers snapshot the
//! chunk list and scan without locks. Setting one bit clones a single
//! 8 KiB chunk instead of the whole bitmap, whose size grows with collection
//! capacity, so scan cost is unchanged and updates stay cheap.

use std::sync::Arc;

use bitvec::prelude::*;

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

    pub(super) fn count_ones(&self) -> u64 {
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
        let mut bits = BitVec::with_capacity(slot_capacity);
        for chunk in &self.chunks {
            bits.extend_from_bitslice(chunk.as_bitslice());
        }
        bits.resize(slot_capacity, false);
        Self::from_bits(bits)
    }
}
