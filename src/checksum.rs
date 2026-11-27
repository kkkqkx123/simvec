//! CRC32C (Castagnoli) checksums for every persisted envelope.
//!
//! Every on-disk checksum in this crate uses the same polynomial, including
//! WAL chains whose seed is the previous record's final CRC.

/// Reflected Castagnoli polynomial.
const POLYNOMIAL: u32 = 0x82F6_3B78;

const fn table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0;
    while index < 256 {
        let mut value = index as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 != 0 {
                (value >> 1) ^ POLYNOMIAL
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[index] = value;
        index += 1;
    }
    table
}

static TABLE: [u32; 256] = table();

/// Fold `bytes` into a running CRC state.
pub fn update(crc: u32, bytes: &[u8]) -> u32 {
    let mut crc = crc;
    for byte in bytes {
        crc = TABLE[((crc ^ *byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc
}

/// One-shot CRC32C of `bytes`.
pub fn hash(bytes: &[u8]) -> u32 {
    !update(!0, bytes)
}

/// Incremental CRC32C, used by chained WAL records.
pub struct Hasher(u32);

impl Hasher {
    /// Standard initial state.
    pub fn new() -> Self {
        Self(!0)
    }

    /// Continue a chain from a previous finalized CRC. The finalizer
    /// complements the internal state, so a resumed chain starts from the
    /// complement of the previous digest.
    pub fn new_with_initial(initial: u32) -> Self {
        Self(!initial)
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.0 = update(self.0, bytes);
    }

    pub fn finalize(self) -> u32 {
        !self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vector() {
        assert_eq!(hash(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn incremental_matches_one_shot() {
        let mut hasher = Hasher::new();
        hasher.update(b"hello ");
        hasher.update(b"world");
        assert_eq!(hasher.finalize(), hash(b"hello world"));
    }
}
