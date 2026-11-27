use crate::error::Result;
use crate::storage::postings::Postings;

use super::Inner;

impl Inner {
    /// Retire a slot. Blob bytes and postings stay where they are: a reader
    /// already skips tombstoned slots, and rewriting a posting record to drop
    /// one entry would cost the whole record. Compaction reclaims both.
    pub(super) fn tombstone_slot(&mut self, slot: u32) {
        let Some(point) = self.points[slot as usize].take() else {
            return;
        };
        self.reverse.remove(&point.id);
        for norms in self.norms.values_mut() {
            norms.remove(&slot);
        }
        self.live_count -= 1;
    }

    /// Register one named sparse vector in the postings index and cache its
    /// norm.
    pub(super) fn index_sparse(
        &mut self,
        postings: &Postings,
        space: &str,
        slot: u32,
        indices: &[u32],
        values: &[f32],
    ) -> Result<()> {
        postings.index(slot, indices, values)?;
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        self.norms
            .entry(space.to_string())
            .or_default()
            .insert(slot, norm);
        Ok(())
    }

    /// Stage the postings of an already-encoded vector; dense and multi
    /// vectors are not inverted.
    pub(super) fn index_view(
        &mut self,
        postings: &Postings,
        space: &str,
        slot: u32,
        data: crate::types::VectorView<'_>,
    ) -> Result<()> {
        match data {
            crate::types::VectorView::Sparse { indices, values } => {
                self.index_sparse(postings, space, slot, indices, values)
            }
            _ => Ok(()),
        }
    }
}
