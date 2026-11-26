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
            self.norms.remove(&slot);
        }
        self.live_count -= 1;
    }

    /// Register one sparse vector in the postings index and cache its norm.
    pub(super) fn index_sparse(&mut self, slot: u32, indices: &[u32], values: &[f32]) {
        for (index, value) in indices.iter().zip(values.iter()) {
            self.inverted.entry(*index).or_default().push((slot, *value));
        }
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        self.norms.insert(slot, norm);
    }
}
