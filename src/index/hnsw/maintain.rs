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

    /// Staleness ratio: overwrite upserts relative to the live count at
    /// build time (`overwrites / built_at_live_count`).
    ///
    /// This is a count-based ratio only. A distance-delta leg existed while
    /// overwritten slots kept their graph position, but N1a routes live
    /// overwrites to freshly appended slots (`register_slot` pushes them to
    /// `pending`), so the already-present branch that fed the delta is no
    /// longer reached on the live path and the delta was removed.
    pub(crate) fn stale_ratio(&self) -> f64 {
        let count = self.overwrites_since_build.load(Ordering::Relaxed);
        let base = self.built_at_live_count.load(Ordering::Relaxed).max(1) as f64;
        count as f64 / base
    }
}
