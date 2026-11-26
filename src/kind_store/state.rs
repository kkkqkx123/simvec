use crate::types::VectorData;

use super::Inner;

impl Inner {
    pub(super) fn tombstone_slot(&mut self, slot: u32) {
        let Some(point) = self.points[slot as usize].take() else {
            return;
        };
        self.reverse.remove(point.id.to_string().as_str());
        if let VectorData::Sparse { indices, .. } = &point.data {
            for index in indices {
                if let Some(postings) = self.inverted.get_mut(index) {
                    postings.retain(|(s, _)| *s != slot);
                    if postings.is_empty() {
                        self.inverted.remove(index);
                    }
                }
            }
        }
        self.live_count -= 1;
    }

    /// Candidate slots holding at least one of the query terms.
    pub(super) fn sparse_candidates(&self, indices: &[u32]) -> Vec<u32> {
        let mut candidates = Vec::new();
        for index in indices {
            if let Some(postings) = self.inverted.get(index) {
                for (slot, _) in postings {
                    candidates.push(*slot);
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }
}
