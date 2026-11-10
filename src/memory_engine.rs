//! In-memory vector engine for the browser form.
//!
//! Session-local exact-scan engine with no filesystem, no background
//! threads, and no parallel runtime. It shares the crate-wide type,
//! distance-score, and filter semantics with the native engine so the
//! same JSON shapes work on both forms.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use crate::distance::{distance, to_score};
use crate::error::{Result, VectorSearchError};
use crate::metrics::MetricsSnapshot;
use crate::types::{
    CollectionConfig, CollectionInfo, CollectionStatus, DistanceMetric, Payload, PayloadSchemaType,
    PointId, SearchQuery, SearchResult, VectorFilter, VectorPoint,
};

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
    if point.vector.len() != config.vector_size {
        return Err(VectorSearchError::InvalidVectorDimension {
            expected: config.vector_size,
            actual: point.vector.len(),
        });
    }
    for (idx, v) in point.vector.iter().enumerate() {
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

    /// Names of all collections, sorted.
    pub fn collection_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .collections
            .read()
            .map(|c| c.keys().cloned().collect())
            .unwrap_or_default();
        names.sort_unstable();
        names
    }

    /// Create a collection. Fails when it already exists.
    pub fn create_collection(&self, name: &str, config: &CollectionConfig) -> Result<()> {
        validate_name(name)?;
        if config.vector_size == 0 {
            return Err(VectorSearchError::InvalidVectorDimension {
                expected: 1,
                actual: 0,
            });
        }
        if !matches!(
            config.distance,
            DistanceMetric::Cosine
                | DistanceMetric::Euclid
                | DistanceMetric::Dot
                | DistanceMetric::Manhattan
        ) {
            return Err(VectorSearchError::UnsupportedMetric(config.distance));
        }
        let mut guard = self
            .collections
            .write()
            .map_err(|_| VectorSearchError::Internal("memory engine lock poisoned".to_string()))?;
        if guard.contains_key(name) {
            return Err(VectorSearchError::CollectionAlreadyExists(name.to_string()));
        }
        guard.insert(
            name.to_string(),
            MemoryCollection {
                config: config.clone(),
                points: HashMap::new(),
                payload_indexes: Vec::new(),
            },
        );
        self.bump_revision();
        Ok(())
    }

    /// Drop a collection and all its points.
    pub fn delete_collection(&self, name: &str) -> Result<()> {
        let mut guard = self
            .collections
            .write()
            .map_err(|_| VectorSearchError::Internal("memory engine lock poisoned".to_string()))?;
        if guard.remove(name).is_none() {
            return Err(VectorSearchError::CollectionNotFound(name.to_string()));
        }
        self.bump_revision();
        Ok(())
    }

    /// Whether a collection exists.
    pub fn collection_exists(&self, name: &str) -> bool {
        self.collections
            .read()
            .map(|c| c.contains_key(name))
            .unwrap_or(false)
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

    /// Stored configuration for a collection.
    pub fn collection_config(&self, name: &str) -> Result<Option<CollectionConfig>> {
        self.with_collection(name, |c| Ok(Some(c.config.clone())))
    }

    /// Detailed collection info with exact live counts.
    pub fn collection_info(&self, name: &str) -> Result<CollectionInfo> {
        self.with_collection(name, |c| {
            let live = c.live_count();
            Ok(CollectionInfo {
                name: name.to_string(),
                vector_count: live,
                indexed_vector_count: live,
                points_count: live,
                segments_count: 1,
                config: c.config.clone(),
                status: CollectionStatus::Green,
                index: None,
                memory_bytes: live * c.config.vector_size as u64 * 4,
            })
        })
    }

    /// Upsert a single point.
    pub fn upsert(&self, collection: &str, point: VectorPoint) -> Result<()> {
        self.with_collection_mut(collection, |c| {
            validate_point(&c.config, &point)?;
            if let Some(cap) = c.config.max_points {
                let grows = !c.points.contains_key(&point.id);
                if grows && c.points.len() as u64 >= cap {
                    return Err(VectorSearchError::CapacityExceeded {
                        collection: collection.to_string(),
                        max_points: cap,
                        requested: c.points.len() as u64 + 1,
                    });
                }
            }
            c.points.insert(point.id.clone(), point);
            self.bump_revision();
            Ok(())
        })
    }

    /// Upsert a batch of points.
    pub fn upsert_batch(&self, collection: &str, points: &[VectorPoint]) -> Result<()> {
        for point in points {
            self.upsert(collection, point.clone())?;
        }
        Ok(())
    }

    /// Delete a point by id. Deleting a missing id is a no-op.
    pub fn delete(&self, collection: &str, point_id: &PointId) -> Result<()> {
        self.with_collection_mut(collection, |c| {
            if c.points.remove(point_id).is_some() {
                self.bump_revision();
            }
            Ok(())
        })
    }

    /// Delete a batch of points.
    pub fn delete_batch(&self, collection: &str, point_ids: &[PointId]) -> Result<()> {
        self.with_collection_mut(collection, |c| {
            let mut removed = false;
            for id in point_ids {
                if c.points.remove(id).is_some() {
                    removed = true;
                }
            }
            if removed {
                self.bump_revision();
            }
            Ok(())
        })
    }

    /// Delete every point matching the filter. Returns the number deleted.
    pub fn delete_by_filter(&self, collection: &str, filter: &VectorFilter) -> Result<u64> {
        self.with_collection_mut(collection, |c| {
            let mut doomed = Vec::new();
            for (id, point) in &c.points {
                if crate::filter::matches(filter, id, point.payload.as_ref())? {
                    doomed.push(id.clone());
                }
            }
            let count = doomed.len() as u64;
            for id in doomed {
                c.points.remove(&id);
            }
            if count > 0 {
                self.bump_revision();
            }
            Ok(count)
        })
    }

    /// Exact-scan search. Scores follow the crate-wide higher-is-better contract.
    /// Kernel dispatch degrades to the scalar baseline on wasm32.
    pub fn search(&self, collection: &str, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        self.with_collection(collection, |c| {
            if query.vector.len() != c.config.vector_size {
                return Err(VectorSearchError::InvalidVectorDimension {
                    expected: c.config.vector_size,
                    actual: query.vector.len(),
                });
            }
            for (idx, v) in query.vector.iter().enumerate() {
                if !v.is_finite() {
                    return Err(VectorSearchError::NonFiniteElement(idx));
                }
            }
            let mut scored: Vec<SearchResult> = Vec::new();
            for point in c.points.values() {
                if let Some(filter) = &query.filter {
                    if !crate::filter::matches(filter, &point.id, point.payload.as_ref())? {
                        continue;
                    }
                }
                let dist = distance(c.config.distance, &query.vector, &point.vector);
                let score = to_score(c.config.distance, dist);
                if let Some(threshold) = query.score_threshold {
                    if score < threshold {
                        continue;
                    }
                }
                let mut result = SearchResult::new(point.id.clone(), score);
                if query.with_payload.unwrap_or(true) {
                    result.payload = project_payload(point.payload.clone(), query);
                }
                if query.with_vector.unwrap_or(false) {
                    result.vector = Some(point.vector.clone());
                }
                scored.push(result);
            }
            scored.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let offset = query.offset.unwrap_or(0);
            let limit = query.effective_limit();
            Ok(scored.into_iter().skip(offset).take(limit).collect())
        })
    }

    /// Fetch a point by id.
    pub fn get(&self, collection: &str, point_id: &PointId) -> Result<Option<VectorPoint>> {
        self.with_collection(collection, |c| Ok(c.points.get(point_id).cloned()))
    }

    /// Number of live points.
    pub fn count(&self, collection: &str) -> Result<u64> {
        self.with_collection(collection, |c| Ok(c.live_count()))
    }

    /// Number of live points matching `filter`. Pure read.
    pub fn count_by_filter(&self, collection: &str, filter: &VectorFilter) -> Result<u64> {
        self.with_collection(collection, |c| {
            let mut count = 0u64;
            for (id, point) in &c.points {
                if crate::filter::matches(filter, id, point.payload.as_ref())? {
                    count += 1;
                }
            }
            Ok(count)
        })
    }

    /// Paginated scan ordered by point id string.
    pub fn scroll(
        &self,
        collection: &str,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        self.with_collection(collection, |c| {
            let mut ids: Vec<&PointId> = c.points.keys().collect();
            ids.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
            let mut skipping = offset.is_some();
            let mut results = Vec::new();
            let mut last: Option<PointId> = None;
            for id in ids {
                if skipping {
                    if Some(id) == offset {
                        skipping = false;
                    }
                    continue;
                }
                if let Some(point) = c.points.get(id) {
                    last = Some(id.clone());
                    results.push(VectorPoint {
                        id: id.clone(),
                        vector: if with_vector.unwrap_or(false) {
                            point.vector.clone()
                        } else {
                            Vec::new()
                        },
                        payload: if with_payload.unwrap_or(true) {
                            point.payload.clone()
                        } else {
                            None
                        },
                    });
                    if results.len() >= limit {
                        break;
                    }
                }
            }
            Ok((results, last))
        })
    }

    /// Paginated scan over points matching `filter`, ordered by id string.
    /// `limit` counts only matching points; the cursor is the last match.
    /// A short page does not imply exhaustion: page until an empty page.
    pub fn scroll_by_filter(
        &self,
        collection: &str,
        filter: &VectorFilter,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        self.with_collection(collection, |c| {
            let mut ids: Vec<&PointId> = c.points.keys().collect();
            ids.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
            let mut skipping = offset.is_some();
            let mut results = Vec::new();
            let mut last: Option<PointId> = None;
            for id in ids {
                if skipping {
                    if Some(id) == offset {
                        skipping = false;
                    }
                    continue;
                }
                if let Some(point) = c.points.get(id) {
                    if !crate::filter::matches(filter, id, point.payload.as_ref())? {
                        continue;
                    }
                    last = Some(id.clone());
                    results.push(VectorPoint {
                        id: id.clone(),
                        vector: if with_vector.unwrap_or(false) {
                            point.vector.clone()
                        } else {
                            Vec::new()
                        },
                        payload: if with_payload.unwrap_or(true) {
                            point.payload.clone()
                        } else {
                            None
                        },
                    });
                    if results.len() >= limit {
                        break;
                    }
                }
            }
            Ok((results, last))
        })
    }

    /// Replace the payload for a single point.
    pub fn set_payload(
        &self,
        collection: &str,
        point_id: &PointId,
        payload: Payload,
    ) -> Result<()> {
        self.with_collection_mut(collection, |c| {
            let point = c
                .points
                .get_mut(point_id)
                .ok_or_else(|| VectorSearchError::InvalidPointId(point_id.to_string()))?;
            point.payload = Some(payload);
            self.bump_revision();
            Ok(())
        })
    }

    /// Set a single payload field, preserving all other keys.
    pub fn set_payload_field(
        &self,
        collection: &str,
        point_id: &PointId,
        key: String,
        value: serde_json::Value,
    ) -> Result<()> {
        self.set_payload_fields(collection, point_id, Payload::from([(key, value)]))
    }

    /// Merge fields into a point payload.
    pub fn set_payload_fields(
        &self,
        collection: &str,
        point_id: &PointId,
        fields: Payload,
    ) -> Result<()> {
        if fields.is_empty() {
            return Ok(());
        }
        self.with_collection_mut(collection, |c| {
            let point = c
                .points
                .get_mut(point_id)
                .ok_or_else(|| VectorSearchError::InvalidPointId(point_id.to_string()))?;
            let payload = point.payload.get_or_insert_with(HashMap::new);
            for (k, v) in fields {
                payload.insert(k, v);
            }
            self.bump_revision();
            Ok(())
        })
    }

    /// Remove keys from a point payload.
    pub fn delete_payload_keys(
        &self,
        collection: &str,
        point_id: &PointId,
        keys: Vec<String>,
    ) -> Result<()> {
        self.with_collection_mut(collection, |c| {
            let point = c
                .points
                .get_mut(point_id)
                .ok_or_else(|| VectorSearchError::InvalidPointId(point_id.to_string()))?;
            if let Some(payload) = point.payload.as_mut() {
                let mut removed = false;
                for k in keys {
                    if payload.remove(&k).is_some() {
                        removed = true;
                    }
                }
                if removed {
                    self.bump_revision();
                }
            }
            Ok(())
        })
    }

    /// Declare a payload field index. Memory form tracks the declaration
    /// only; searches stay exact scan.
    pub fn create_payload_index(
        &self,
        collection: &str,
        field: &str,
        schema: PayloadSchemaType,
    ) -> Result<()> {
        self.with_collection_mut(collection, |c| {
            if !c.payload_indexes.iter().any(|(f, _)| f == field) {
                c.payload_indexes.push((field.to_string(), schema));
                self.bump_revision();
            }
            Ok(())
        })
    }

    /// Drop a payload field index. Returns whether it existed.
    pub fn delete_payload_index(&self, collection: &str, field: &str) -> Result<bool> {
        self.with_collection_mut(collection, |c| {
            let before = c.payload_indexes.len();
            c.payload_indexes.retain(|(f, _)| f != field);
            let removed = c.payload_indexes.len() != before;
            if removed {
                self.bump_revision();
            }
            Ok(removed)
        })
    }

    /// All declared payload indexes.
    pub fn list_payload_indexes(
        &self,
        collection: &str,
    ) -> Result<Vec<(String, PayloadSchemaType)>> {
        self.with_collection(collection, |c| Ok(c.payload_indexes.clone()))
    }

    /// Apply a batch of cross-collection operations in sorted collection order.
    pub fn apply_txn(&self, ops: Vec<MemoryTxnOp>) -> Result<()> {
        let mut grouped: HashMap<String, Vec<MemoryTxnOp>> = HashMap::new();
        for op in ops {
            match &op {
                MemoryTxnOp::Upsert { collection, .. } | MemoryTxnOp::Delete { collection, .. } => {
                    grouped.entry(collection.clone()).or_default().push(op);
                }
            }
        }
        let mut names: Vec<String> = grouped.keys().cloned().collect();
        names.sort_unstable();
        for name in names {
            let group = grouped.remove(&name).expect("group present");
            for op in group {
                match op {
                    MemoryTxnOp::Upsert { point, .. } => {
                        self.upsert(&name, point)?;
                    }
                    MemoryTxnOp::Delete { point_id, .. } => {
                        self.delete(&name, &point_id)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Export the whole engine as a versioned snapshot value.
    pub fn export_snapshot(&self) -> Result<MemorySnapshot> {
        let guard = self
            .collections
            .read()
            .map_err(|_| VectorSearchError::Internal("memory engine lock poisoned".to_string()))?;
        let mut collections: Vec<MemorySnapshotCollection> = Vec::with_capacity(guard.len());
        for (name, collection) in guard.iter() {
            let mut points: Vec<VectorPoint> = collection.points.values().cloned().collect();
            points.sort_by(|a, b| a.id.to_string().cmp(&b.id.to_string()));
            collections.push(MemorySnapshotCollection {
                name: name.clone(),
                config: collection.config.clone(),
                points,
                payload_indexes: collection.payload_indexes.clone(),
            });
        }
        collections.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(MemorySnapshot {
            version: 1,
            collections,
        })
    }

    /// Export one collection as a snapshot entry.
    pub fn export_collection(&self, name: &str) -> Result<MemorySnapshotCollection> {
        self.with_collection(name, |c| {
            let mut points: Vec<VectorPoint> = c.points.values().cloned().collect();
            points.sort_by(|a, b| a.id.to_string().cmp(&b.id.to_string()));
            Ok(MemorySnapshotCollection {
                name: name.to_string(),
                config: c.config.clone(),
                points,
                payload_indexes: c.payload_indexes.clone(),
            })
        })
    }

    /// Insert or replace one collection from a snapshot entry.
    pub fn import_collection(&self, entry: MemorySnapshotCollection) -> Result<()> {
        validate_name(&entry.name)?;
        if entry.config.vector_size == 0 {
            return Err(VectorSearchError::InvalidVectorDimension {
                expected: 1,
                actual: 0,
            });
        }
        for point in &entry.points {
            validate_point(&entry.config, point)?;
        }
        if let Some(cap) = entry.config.max_points {
            if entry.points.len() as u64 > cap {
                return Err(VectorSearchError::CapacityExceeded {
                    collection: entry.name,
                    max_points: cap,
                    requested: entry.points.len() as u64,
                });
            }
        }
        let mut points = HashMap::with_capacity(entry.points.len());
        for point in entry.points {
            points.insert(point.id.clone(), point);
        }
        let mut guard = self
            .collections
            .write()
            .map_err(|_| VectorSearchError::Internal("memory engine lock poisoned".to_string()))?;
        guard.insert(
            entry.name,
            MemoryCollection {
                config: entry.config,
                points,
                payload_indexes: entry.payload_indexes,
            },
        );
        self.bump_revision();
        Ok(())
    }

    /// Replace the whole engine from a snapshot after validating every entry.
    pub fn import_snapshot(&self, snapshot: MemorySnapshot) -> Result<()> {
        if snapshot.version != 1 {
            return Err(VectorSearchError::InvalidConfig(format!(
                "unsupported memory snapshot version {}",
                snapshot.version
            )));
        }
        let mut staged: HashMap<String, MemoryCollection> = HashMap::new();
        for entry in snapshot.collections {
            validate_name(&entry.name)?;
            if entry.config.vector_size == 0 {
                return Err(VectorSearchError::InvalidVectorDimension {
                    expected: 1,
                    actual: 0,
                });
            }
            for point in &entry.points {
                validate_point(&entry.config, point)?;
            }
            if staged.contains_key(&entry.name) {
                return Err(VectorSearchError::CollectionAlreadyExists(entry.name));
            }
            let mut points = HashMap::with_capacity(entry.points.len());
            for point in entry.points {
                points.insert(point.id.clone(), point);
            }
            if let Some(cap) = entry.config.max_points {
                if points.len() as u64 > cap {
                    return Err(VectorSearchError::CapacityExceeded {
                        collection: entry.name,
                        max_points: cap,
                        requested: points.len() as u64,
                    });
                }
            }
            staged.insert(
                entry.name,
                MemoryCollection {
                    config: entry.config,
                    points,
                    payload_indexes: entry.payload_indexes,
                },
            );
        }
        let mut guard = self
            .collections
            .write()
            .map_err(|_| VectorSearchError::Internal("memory engine lock poisoned".to_string()))?;
        *guard = staged;
        self.bump_revision();
        Ok(())
    }

    /// Index build is a compatible no-op in memory form.
    pub fn build_index(&self, collection: &str) -> Result<bool> {
        self.with_collection(collection, |_| Ok(false))
    }

    /// Whether an index is published. Always false in memory form.
    pub fn has_index(&self, _collection: &str) -> bool {
        false
    }

    /// Drop the published index. No-op in memory form; still rejects unknown collections.
    pub fn drop_index(&self, collection: &str) -> Result<()> {
        self.with_collection(collection, |_| Ok(()))
    }

    /// Drop the published index alias with explicit naming.
    pub fn drop_published_index(&self, collection: &str) -> Result<()> {
        self.drop_index(collection)
    }

    /// Quantization build is a compatible no-op in memory form.
    pub fn build_quantization(&self, collection: &str) -> Result<bool> {
        self.with_collection(collection, |_| Ok(false))
    }

    /// Whether quantization is ready. Always false in memory form.
    pub fn has_quantization(&self, _collection: &str) -> bool {
        false
    }

    /// Drain pending slots. No-op in memory form.
    pub fn drain_pending(&self, _collection: &str) {}

    /// Global maintenance sweep. No-op in memory form.
    pub fn run_maintenance_sweep(&self) {}

    /// Reclaim space. Points are removed eagerly, so this returns the live count.
    pub fn compact(&self, collection: &str) -> Result<u64> {
        self.count(collection)
    }

    /// Metrics snapshot. Memory form reports an empty snapshot.
    pub fn collection_metrics(&self, collection: &str) -> Result<MetricsSnapshot> {
        self.with_collection(collection, |_| Ok(MetricsSnapshot::default()))
    }
}

#[cfg(test)]
#[path = "memory_engine_tests.rs"]
mod tests;
