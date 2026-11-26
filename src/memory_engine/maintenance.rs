use crate::error::Result;
use crate::metrics::MetricsSnapshot;

use super::MemoryEngine;

impl MemoryEngine {
    /// Index build is a compatible no-op in memory form.
    pub fn build_index(&self, collection: &str) -> Result<bool> {
        self.with_collection(collection, |_| Ok(false))
    }

    /// Whether an index is published. Always false in memory form.
    pub fn has_index(&self, _collection: &str) -> bool {
        false
    }

    /// Drop the published index. No-op in memory form; still rejects unknown collections.
    pub fn drop_index(&self, collection: &str) -> Result<()> {
        self.with_collection(collection, |_| Ok(()))
    }

    /// Drop the published index alias with explicit naming.
    pub fn drop_published_index(&self, collection: &str) -> Result<()> {
        self.drop_index(collection)
    }

    /// Quantization build is a compatible no-op in memory form.
    pub fn build_quantization(&self, collection: &str) -> Result<bool> {
        self.with_collection(collection, |_| Ok(false))
    }

    /// Whether quantization is ready. Always false in memory form.
    pub fn has_quantization(&self, _collection: &str) -> bool {
        false
    }

    /// Drain pending slots. No-op in memory form.
    pub fn drain_pending(&self, _collection: &str) {}

    /// Global maintenance sweep. No-op in memory form.
    pub fn run_maintenance_sweep(&self) {}

    /// Reclaim space. Points are removed eagerly, so this returns the live count.
    pub fn compact(&self, collection: &str) -> Result<u64> {
        self.count(collection)
    }

    /// Metrics snapshot. Memory form reports an empty snapshot.
    pub fn collection_metrics(&self, collection: &str) -> Result<MetricsSnapshot> {
        self.with_collection(collection, |_| Ok(MetricsSnapshot::default()))
    }
}
