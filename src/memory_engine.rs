//! In-memory vector engine for the browser form.
//!
//! Session-local exact-scan engine with no filesystem, no background
//! threads, and no parallel runtime. It shares the crate-wide type,
//! distance-score, and filter semantics with the native engine so the
//! same JSON shapes work on both forms.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use crate::error::{Result, VectorSearchError};
use crate::types::{
    CollectionConfig, Payload, PayloadSchemaType, PointId, SearchQuery, VectorPoint,
};

mod lifecycle;
mod maintenance;
mod payload;
mod points;
mod snapshot;
mod txn;

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.contains('\0')
    {
        return Err(VectorSearchError::InvalidCollectionName(name.to_string()));
    }
    Ok(())
}

fn validate_point(config: &CollectionConfig, point: &VectorPoint) -> Result<()> {
    if point.vectors.len() != 1
        || !point
            .vectors
            .contains_key(crate::types::DEFAULT_VECTOR_NAME)
    {
        return Err(VectorSearchError::InvalidConfig(
            "memory engine points must carry a single dense vector".to_string(),
        ));
    }
    if point
        .data()
        .is_some_and(|data| data.kind() != crate::types::VectorKind::Dense)
    {
        return Err(VectorSearchError::InvalidConfig(
            "memory engine points must be dense".to_string(),
        ));
    }
    if point.vector().len() != config.vector_size {
        return Err(VectorSearchError::InvalidVectorDimension {
            expected: config.vector_size,
            actual: point.vector().len(),
        });
    }
    for (idx, v) in point.vector().iter().enumerate() {
        if !v.is_finite() {
            return Err(VectorSearchError::NonFiniteElement(idx));
        }
    }
    Ok(())
}

fn project_payload(payload: Option<Payload>, query: &SearchQuery) -> Option<Payload> {
    let mut payload = payload?;
    if let Some(sel) = &query.payload_selector {
        if let Some(include) = &sel.include {
            payload.retain(|k, _| include.contains(k));
        }
        if let Some(exclude) = &sel.exclude {
            for k in exclude {
                payload.remove(k);
            }
        }
    }
    Some(payload)
}

#[derive(Debug)]
struct MemoryCollection {
    config: CollectionConfig,
    points: HashMap<PointId, VectorPoint>,
    payload_indexes: Vec<(String, PayloadSchemaType)>,
}

/// One coordinated transaction operation for the memory form.
#[derive(Debug, Clone)]
pub enum MemoryTxnOp {
    Upsert {
        collection: String,
        point: VectorPoint,
    },
    Delete {
        collection: String,
        point_id: PointId,
    },
}

/// Versioned whole-engine snapshot for caller-managed persistence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemorySnapshotCollection {
    pub name: String,
    pub config: CollectionConfig,
    pub points: Vec<VectorPoint>,
    pub payload_indexes: Vec<(String, PayloadSchemaType)>,
}

/// Exported snapshot envelope. Version stays fixed for this form.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemorySnapshot {
    pub version: u32,
    pub collections: Vec<MemorySnapshotCollection>,
}

impl MemoryCollection {
    fn live_count(&self) -> u64 {
        self.points.len() as u64
    }
}

/// Session-local engine. All operations are synchronous and apply inline.
#[derive(Debug, Default)]
pub struct MemoryEngine {
    collections: RwLock<HashMap<String, MemoryCollection>>,
    revision: AtomicU64,
}

impl MemoryEngine {
    fn bump_revision(&self) {
        self.revision.fetch_add(1, Ordering::SeqCst);
    }

    /// Monotonic counter advanced by every state-changing mutation.
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::SeqCst)
    }

    /// Create an empty engine with no collections.
    pub fn new() -> Self {
        Self {
            collections: RwLock::new(HashMap::new()),
            revision: AtomicU64::new(0),
        }
    }

    fn with_collection<R>(
        &self,
        name: &str,
        f: impl FnOnce(&MemoryCollection) -> Result<R>,
    ) -> Result<R> {
        let guard = self
            .collections
            .read()
            .map_err(|_| VectorSearchError::Internal("memory engine lock poisoned".to_string()))?;
        let collection = guard
            .get(name)
            .ok_or_else(|| VectorSearchError::CollectionNotFound(name.to_string()))?;
        f(collection)
    }

    fn with_collection_mut<R>(
        &self,
        name: &str,
        f: impl FnOnce(&mut MemoryCollection) -> Result<R>,
    ) -> Result<R> {
        let mut guard = self
            .collections
            .write()
            .map_err(|_| VectorSearchError::Internal("memory engine lock poisoned".to_string()))?;
        let collection = guard
            .get_mut(name)
            .ok_or_else(|| VectorSearchError::CollectionNotFound(name.to_string()))?;
        f(collection)
    }
}

#[cfg(test)]
#[path = "memory_engine_tests.rs"]
mod tests;
