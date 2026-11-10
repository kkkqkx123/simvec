use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use crate::error::{Result, VectorSearchError};
use crate::index::persist::{PersistedHnsw, PersistedNodeRecord};
use crate::index::IndexBuildParams;
use crate::metrics::{timed_write_lock, Metrics};
use crate::storage::vectors::Vectors;
use crate::storage::TombstoneBits;
use crate::types::{DistanceMetric, HnswConfig};

use bitvec::prelude::*;

use super::*;

impl HnswIndex {
    /// Rehydrate from persisted state. Structural violations produce
    /// `CorruptData` so callers discard the file and fall back to exact scan.
    pub(crate) fn from_persisted(data: PersistedHnsw, config: &HnswConfig) -> Result<Self> {
        if data.dim == 0 || data.m < MIN_M {
            return Err(VectorSearchError::CorruptData(
                "hnsw.bin invalid header".to_string(),
            ));
        }
        let index = Self::new(
            data.dim,
            data.distance,
            &HnswConfig {
                m: data.m,
                ef_construct: data.ef_construct,
                ..config.clone()
            },
            0,
        );

        let mut nodes = Vec::new();
        for record in &data.nodes {
            if record.slot as usize != nodes.len() {
                return Err(VectorSearchError::CorruptData(
                    "hnsw.bin nodes not dense from slot 0".to_string(),
                ));
            }
            if record.neighbors.len() != record.level as usize + 1 {
                return Err(VectorSearchError::CorruptData(
                    "hnsw.bin node layer count mismatch".to_string(),
                ));
            }
            nodes.push(Some(Arc::new(Node::new(record.level))));
        }

        // Resolve adjacency after all nodes exist.
        for (i, record) in data.nodes.iter().enumerate() {
            let node = nodes[i].as_ref().expect("just pushed");
            // Restore version from persisted state (cycling 1–15).
            // Clamp to valid range for safety; version 0 is invalid in pgvector.
            let version = if record.version == 0 {
                1
            } else {
                record.version.min(15)
            };
            node.version
                .store(version, std::sync::atomic::Ordering::Release);
            for (lc, list) in record.neighbors.iter().enumerate() {
                let cap = index.layer_cap(lc as u8);
                if list.len() > cap || list.iter().any(|&s| s as usize >= nodes.len()) {
                    return Err(VectorSearchError::CorruptData(format!(
                        "hnsw.bin adjacency violation at slot {}",
                        record.slot
                    )));
                }
                *node.layer(lc as u8).write() = list.clone();
            }
        }

        if let Some((slot, level)) = data.entry {
            match nodes.get(slot as usize) {
                Some(Some(node)) if node.level as i32 == level => {}
                _ => {
                    return Err(VectorSearchError::CorruptData(
                        "hnsw.bin entry point invalid".to_string(),
                    ))
                }
            }
            *index.entry.write() = Some(EntryPoint {
                slot,
                level: level as u8,
            });
        }

        *index.nodes.write() = nodes;
        index
            .built_at_live_count
            .store(data.built_at_live_count, Ordering::Relaxed);
        Ok(index)
    }

    /// Snapshot of the persisted representation (for `hnsw.bin`).
    pub(crate) fn to_persisted(&self) -> PersistedHnsw {
        let nodes = self.nodes.read();
        let records = nodes
            .iter()
            .enumerate()
            .filter_map(|(slot, node)| {
                let node = node.as_ref()?;
                Some(PersistedNodeRecord {
                    slot: slot as u32,
                    level: node.level,
                    version: node.version(),
                    neighbors: node
                        .neighbors
                        .iter()
                        .map(|l| l.read().clone())
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        PersistedHnsw {
            dim: self.dim,
            distance: self.metric,
            m: self.m,
            ef_construct: self.ef_construct,
            ef_search: self.ef_search,
            entry: (*self.entry.read()).map(|e| (e.slot, e.level as i32)),
            built_at_live_count: self.built_at_live_count.load(Ordering::Relaxed),
            nodes: records,
        }
    }
}
