//! Per-space ANN index for named dense spaces.
//!
//! Each dense space that declares an ANN tier owns an independent HNSW graph
//! under `spaces/<space>/ann/`. The graph is a *derived* structure over a
//! space-local `vectors.bin` (rows addressed by the collection's global slot):
//! it is rebuilt at checkpoint from the live points and persisted for fast
//! open. A missing, corrupt or untrained graph makes the space fall back to
//! exact scan; it never affects correctness or other spaces.
//!
//! The graph's node ids are the collection's global slots (built with
//! `slot_base = 0`), so a probe result is directly usable as a slot into the
//! owning `NamedStore` after a liveness/filter check.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::RwLock;

use crate::error::Result;
use crate::index::persist::{load_hnsw_at, save_hnsw_at};
use crate::index::hnsw::HnswIndex;
use crate::index::IndexBuildParams;
use crate::storage::vectors::Vectors;
use crate::types::{DistanceMetric, HnswConfig, IndexConfig};

use super::spaces_meta;

const ANN_DIR: &str = "ann";
const HNSW_FILE: &str = "hnsw.bin";
/// Segment width for the space-local vectors file. Independent of the blob
/// segment width; chosen to keep a single segment for typical collections.
const ANN_SEGMENT_SLOTS: u32 = 1 << 20;

/// One dense space's optional ANN index.
pub(crate) struct SpaceAnn {
    space: String,
    dim: usize,
    metric: DistanceMetric,
    config: HnswConfig,
    ann_dir: PathBuf,
    vectors: RwLock<Vectors>,
    index: ArcSwap<Option<HnswIndex>>,
}

impl SpaceAnn {
    /// Extract the HNSW config from an index selection. Returns `None` for
    /// Flat (no ANN) and for IVF (handled by a future tier).
    pub(crate) fn hnsw_config(index: &IndexConfig) -> Option<HnswConfig> {
        match index {
            IndexConfig::Hnsw(config) => Some(config.clone()),
            _ => None,
        }
    }

    /// Create an empty ANN index for a dense space.
    pub(crate) fn create(
        gen_dir: &Path,
        space: &str,
        dim: usize,
        metric: DistanceMetric,
        config: HnswConfig,
    ) -> Result<Self> {
        let ann_dir = spaces_meta::segment_dir(gen_dir, space, 0)
            .parent()
            .expect("segment dir has a parent")
            .parent()
            .expect("segments dir has a parent")
            .join(ANN_DIR);
        std::fs::create_dir_all(&ann_dir)?;
        let vectors = Vectors::create(&ann_dir, dim, ANN_SEGMENT_SLOTS)?;
        Ok(Self {
            space: space.to_string(),
            dim,
            metric,
            config,
            ann_dir,
            vectors: RwLock::new(vectors),
            index: ArcSwap::from_pointee(None),
        })
    }

    /// Open an ANN index, loading a persisted graph if present and valid.
    /// Any failure leaves the index untrained (exact-scan fallback).
    pub(crate) fn open(
        gen_dir: &Path,
        space: &str,
        dim: usize,
        metric: DistanceMetric,
        config: HnswConfig,
        slot_capacity: u64,
    ) -> Result<Self> {
        let ann_dir = spaces_meta::segment_dir(gen_dir, space, 0)
            .parent()
            .expect("segment dir has a parent")
            .parent()
            .expect("segments dir has a parent")
            .join(ANN_DIR);
        std::fs::create_dir_all(&ann_dir)?;
        let vectors = Vectors::open(&ann_dir, dim, ANN_SEGMENT_SLOTS, slot_capacity)?;
        let index = match load_hnsw_at(&ann_dir.join(HNSW_FILE)) {
            Ok(Some(data)) => HnswIndex::from_persisted(data, &config).ok(),
            _ => None,
        };
        Ok(Self {
            space: space.to_string(),
            dim,
            metric,
            config,
            ann_dir,
            vectors: RwLock::new(vectors),
            index: ArcSwap::from_pointee(index),
        })
    }

    /// Rebuild the graph from a set of `(global_slot, vector)` pairs and
    /// persist it. The vectors are written into the space-local vectors file
    /// first, so the graph's reads resolve during build.
    pub(crate) fn rebuild(&self, rows: &[(u32, &[f32])]) -> Result<()> {
        if rows.is_empty() {
            self.index.store(Arc::new(None));
            return Ok(());
        }
        let max_slot = rows.iter().map(|(slot, _)| *slot).max().unwrap_or(0) as u64;
        {
            let vectors = self.vectors.write();
            vectors.grow_to(max_slot + 1)?;
            for (slot, vector) in rows {
                vectors.write_slot(*slot as u64, vector)?;
            }
            // One fsync per segment touched. The graph is only published after
            // its vectors are durable.
            let segments = (max_slot / ANN_SEGMENT_SLOTS as u64) as usize + 1;
            for index in 0..segments {
                vectors.sync_segment(index)?;
            }
        }
        let slots: Vec<u32> = rows.iter().map(|(slot, _)| *slot).collect();
        let snapshot = self.vectors.read().snapshot();
        let built = HnswIndex::build(
            &IndexBuildParams::new(
                &self.space,
                self.dim,
                self.metric,
                &slots,
                &snapshot,
                ANN_SEGMENT_SLOTS,
            ),
            &self.config,
        )?;
        save_hnsw_at(&self.ann_dir.join(HNSW_FILE), &built.to_persisted())?;
        self.index.store(Arc::new(Some(built)));
        Ok(())
    }

    /// Approximate kNN probe. Returns `(output_score, global_slot)` pairs.
    /// An untrained index returns no candidates, which callers treat as a
    /// signal to fall back to exact scan.
    pub(crate) fn probe(&self, query: &[f32], ef: usize, k: usize) -> Result<Vec<(f32, u32)>> {
        let guard = self.index.load_full();
        let Some(index) = guard.as_ref() else {
            return Ok(Vec::new());
        };
        let snapshot = self.vectors.read().snapshot();
        let ctx = crate::index::hnsw::HnswSearchContext {
            tombstones: None,
            vectors: &snapshot,
            segment_slots: ANN_SEGMENT_SLOTS,
            slot_base: 0,
            filter_mask: None,
            quant: None,
        };
        index.probe_candidates(query, ef, k, &ctx)
    }

    /// Whether a trained graph is available.
    pub(crate) fn is_trained(&self) -> bool {
        self.index.load().as_ref().is_some()
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: u32) -> Vec<(u32, Vec<f32>)> {
        (0..n).map(|i| (i, vec![i as f32, 0.0, 1.0])).collect()
    }

    #[test]
    fn rebuild_probe_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let gen = dir.path().join("g");
        std::fs::create_dir_all(&gen).unwrap();
        let config = HnswConfig::default();
        let ann = SpaceAnn::create(&gen, "dense", 3, DistanceMetric::Euclid, config.clone()).unwrap();
        let owned = rows(30);
        let refs: Vec<(u32, &[f32])> = owned.iter().map(|(s, v)| (*s, v.as_slice())).collect();
        ann.rebuild(&refs).unwrap();
        assert!(ann.is_trained());
        let hits = ann.probe(&[0.0, 0.0, 1.0], 32, 5).unwrap();
        assert!(!hits.is_empty());

        let reopened = SpaceAnn::open(
            &gen, "dense", 3, DistanceMetric::Euclid, config, 30,
        ).unwrap();
        assert!(reopened.is_trained());
        let hits2 = reopened.probe(&[0.0, 0.0, 1.0], 32, 5).unwrap();
        assert!(!hits2.is_empty());
    }
}
