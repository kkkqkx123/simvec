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
                let space_index = self.inverted.get_mut(*space);
                if let Some(space_index) = space_index {
                    for index in indices {
                        if let Some(postings) = space_index.get_mut(index) {
                            postings.retain(|(s, _)| *s != slot);
                            if postings.is_empty() {
                                space_index.remove(index);
                            }
                        }
                    }
                }
            }
        }
        self.live_count -= 1;
    }
}
