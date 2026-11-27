//! WAL acknowledge pins: hold back log truncation for readers that still
//! need old segments.
//!
//! A pin is a live guard over a version. While any guard is alive, the
//! effective acknowledge water mark must not advance past the lowest pinned
//! version. Snapshots and remote copies register a pin before capturing files
//! and release it afterwards, so a concurrent checkpoint cannot unlink the
//! segments they are still reading.

use std::sync::{Arc, Weak};

use parking_lot::Mutex;

/// Registry of live WAL acknowledge pins for one collection.
#[derive(Debug, Default)]
pub struct WalAckPins {
    pins: Mutex<Vec<Weak<parking_lot::Mutex<u64>>>>,
}

impl WalAckPins {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            pins: Mutex::new(Vec::new()),
        }
    }

    /// Pin the acknowledge at `version`. Operations from `version` onwards are
    /// retained while the returned guard is alive.
    pub fn pin(&self, version: u64) -> WalAckPinGuard {
        let slot = Arc::new(parking_lot::Mutex::new(version));
        {
            let mut pins = self.pins.lock();
            pins.retain(|p| p.strong_count() > 0);
            pins.push(Arc::downgrade(&slot));
        }
        WalAckPinGuard { version: slot }
    }

    /// Lowest pinned version, or `None` when no pin is live.
    pub fn lowest(&self) -> Option<u64> {
        let mut pins = self.pins.lock();
        pins.retain(|p| p.strong_count() > 0);
        pins
            .iter()
            .filter_map(Weak::upgrade)
            .map(|v| *v.lock())
            .min()
    }

    /// Clamp a candidate acknowledge water mark by the live pins.
    pub fn clamp(&self, watermark: u64) -> u64 {
        match self.lowest() {
            Some(lowest) if lowest <= watermark => lowest.saturating_sub(1),
            _ => watermark,
        }
    }
}

/// Live pin on the WAL acknowledge. Dropping it releases the pin.
pub struct WalAckPinGuard {
    version: Arc<parking_lot::Mutex<u64>>,
}

impl WalAckPinGuard {
    /// Move the pin forward to `version`. Never moves backwards.
    pub fn set(&self, version: u64) {
        let mut guard = self.version.lock();
        debug_assert!(version >= *guard);
        if version > *guard {
            *guard = version;
        }
    }

    /// Pinned version.
    pub fn version(&self) -> u64 {
        *self.version.lock()
    }
}

impl std::fmt::Debug for WalAckPinGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WalAckPinGuard")
            .field("version", &self.version())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowest_tracks_live_guards_only() {
        let pins = WalAckPins::new();
        assert_eq!(pins.lowest(), None);
        let a = pins.pin(10);
        let b = pins.pin(7);
        assert_eq!(pins.lowest(), Some(7));
        drop(b);
        assert_eq!(pins.lowest(), Some(10));
        drop(a);
        assert_eq!(pins.lowest(), None);
    }

    #[test]
    fn clamp_holds_back_the_ack() {
        let pins = WalAckPins::new();
        assert_eq!(pins.clamp(9), 9);
        let _guard = pins.pin(8);
        assert_eq!(pins.clamp(9), 7);
        assert_eq!(pins.clamp(5), 5);
    }
}
