use crate::error::Result;
use crate::storage::postings::Postings;

use super::Inner;

impl Inner {
    /// Retire a slot. The vector bytes and the postings stay where they are:
    /// rewriting a posting record to remove one entry would cost the whole
    /// record, and every reader already skips tombstoned slots. Compaction is
    /// what reclaims them.
    pub(super) fn tombstone_slot(&mut self, slot: u32) {
        let Some(point) = self.points[slot as usize].take() else {
            return;
        };
        self.reverse.remove(&point.id);
        self.norms.remove(&slot);
        self.live_count -= 1;
    }

    /// Stage one sparse vector in the postings index and cache its norm.
    pub(super) fn index_sparse(
        &mut self,
        postings: &Postings,
        slot: u32,
        indices: &[u32],
        values: &[f32],
    ) -> Result<()> {
        postings.index(slot, indices, values)?;
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        self.norms.insert(slot, norm);
        Ok(())
    }

    /// Stage the postings of an already-encoded payload. Dense and multi
    /// payloads are not inverted.
    pub(super) fn index_view(
        &mut self,
        postings: &Postings,
        slot: u32,
        data: crate::types::VectorView<'_>,
    ) -> Result<()> {
        match data {
            crate::types::VectorView::Sparse { indices, values } => {
                self.index_sparse(postings, slot, indices, values)
            }
            _ => Ok(()),
        }
    }
}
