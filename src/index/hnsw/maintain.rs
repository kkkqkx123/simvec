use std::sync::atomic::Ordering;

use super::*;

impl HnswIndex {
    /// Whether the node table covers every slot below `slots` with no holes.
    ///
    /// A hole means a live point that never entered the graph (it is sitting
    /// in `pending`, or the insert failed). Remapping after compaction cannot
    /// invent its adjacency, and a dense node table is what makes the
    /// remapped graph reloadable, so the caller falls back to a rebuild when
    /// this returns false.
    pub(crate) fn covers_all(&self, slots: usize) -> bool {
        let nodes = self.nodes.read();
        nodes.len() >= slots && nodes[..slots].iter().all(|n| n.is_some())
    }

    /// Staleness ratio combining overwrite count and distance delta.
    /// Returns `max(count_ratio, delta_ratio)` where:
    /// - `count_ratio = overwrites / built_at_live_count`
    /// - `delta_ratio = distance_delta / built_at_live_count`
    pub(crate) fn stale_ratio(&self) -> f64 {
        let count = self.overwrites_since_build.load(Ordering::Relaxed);
        let delta = self.overwrites_distance_delta.load(Ordering::Relaxed) as f64 / 1e6;
        let base = self.built_at_live_count.load(Ordering::Relaxed).max(1) as f64;
        let count_ratio = count as f64 / base;
        let delta_ratio = delta / base;
        count_ratio.max(delta_ratio)
    }
}
