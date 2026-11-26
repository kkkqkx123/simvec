use crate::types::VectorData;

use super::Inner;

impl Inner {
    pub(super) fn tombstone_slot(&mut self, slot: u32, sparse_spaces: &[&str]) {
        let Some(point) = self.points[slot as usize].take() else {
            return;
        };
        self.reverse.remove(point.id.to_string().as_str());
        for space in sparse_spaces {
            if let Some(VectorData::Sparse { indices, .. }) = point.vectors.get(*space) {
                if let Some(space_index) = self.inverted.get_mut(*space) {
                    for index in indices {
                        if let Some(postings) = space_index.get_mut(index) {
                            postings.retain(|(s, _)| *s != slot);
                            if postings.is_empty() {
                                space_index.remove(index);
                            }
                        }
                    }
                }
                if let Some(norms) = self.norms.get_mut(*space) {
                    norms.remove(&slot);
                }
            }
        }
        self.live_count -= 1;
    }

    /// Drop one named sparse vector's postings and cached norm without
    /// retiring the point's slot.
    pub(super) fn remove_sparse(&mut self, space: &str, slot: u32, data: &VectorData) {
        if let VectorData::Sparse { indices, .. } = data {
            if let Some(space_index) = self.inverted.get_mut(space) {
                for index in indices {
                    if let Some(postings) = space_index.get_mut(index) {
                        postings.retain(|(s, _)| *s != slot);
                        if postings.is_empty() {
                            space_index.remove(index);
                        }
                    }
                }
            }
            if let Some(norms) = self.norms.get_mut(space) {
                norms.remove(&slot);
            }
        }
    }

    /// Register one named sparse vector in the postings index and cache its norm.
    pub(super) fn index_sparse(
        &mut self,
        space: &str,
        slot: u32,
        indices: &[u32],
        values: &[f32],
    ) {
        let space_index = self.inverted.entry(space.to_string()).or_default();
        for (index, value) in indices.iter().zip(values.iter()) {
            space_index.entry(*index).or_default().push((slot, *value));
        }
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        self.norms
            .entry(space.to_string())
            .or_default()
            .insert(slot, norm);
    }
}
