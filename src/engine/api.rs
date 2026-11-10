use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use super::{LocalVectorEngine, MaintenanceJob, TxnOp};
use super::*;
use crate::error::{EngineResult, Result, VectorEngineError, VectorSearchError};
use crate::metrics::MetricsSnapshot;
use crate::storage::CollectionStore;
use crate::types::{CollectionConfig, CollectionInfo, HealthStatus, IndexMetadata, Payload, PayloadSchemaType, PointId, SearchQuery, SearchResult, VectorFilter, VectorPoint};

// ---------------------------------------------------------------------------
// Unified async VectorEngine trait
// ---------------------------------------------------------------------------
//
// Both the local (in-process) engine and the remote Qdrant client implement
// this trait. Callers in the sync layer talk exclusively through this
// interface via a trait object (`Arc<dyn VectorEngine>`), eliminating the
// previous enum-dispatch boilerplate.
//
// The local engine is synchronous; its implementation wraps each call in
// `tokio::task::spawn_blocking` so that blocking I/O / CPU work never
// starves the Tokio worker pool.

use std::pin::Pin;

use futures::Stream;

/// Asynchronous vector engine abstraction.
///
/// Implementors must be `Send + Sync + Debug`.  The local engine runs
/// synchronously through `block_in_place`; remote engines execute
/// directly against the network.
#[async_trait]
pub trait VectorEngine: Send + Sync + std::fmt::Debug {
    /// Human-readable engine name (used for logging and health checks).
    fn name(&self) -> &str;

    /// Engine version string.
    fn version(&self) -> &str;

    /// Whether this is the built-in local engine.
    fn is_local(&self) -> bool {
        false
    }

    /// Whether the engine is currently unavailable (e.g. disabled Qdrant).
    fn is_disabled(&self) -> bool {
        false
    }

    /// Downcast support: returns `self` as `&dyn Any`.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Liveness / readiness probe.
    async fn health_check(&self) -> EngineResult<HealthStatus>;

    // ---- collection management ----

    /// Create a collection.  Fails if it already exists.
    async fn create_collection(&self, name: &str, config: &CollectionConfig) -> EngineResult<()>;

    /// Drop a collection and its persisted data.
    async fn delete_collection(&self, name: &str) -> EngineResult<()>;

    /// Whether a collection with this name exists.
    fn collection_exists(&self, name: &str) -> bool;

    /// Collection metadata, or `None` when not found.
    fn get_index_metadata(&self, name: &str) -> Option<IndexMetadata>;

    /// Create a payload field index for filter acceleration.
    async fn create_payload_index(
        &self,
        collection: &str,
        field: &str,
        schema: PayloadSchemaType,
    ) -> EngineResult<()>;

    /// Remove a payload field index.
    async fn delete_payload_index(&self, collection: &str, field: &str) -> EngineResult<()>;

    /// All declared payload indexes as `(field, schema_type)` pairs.
    async fn list_payload_indexes(
        &self,
        collection: &str,
    ) -> EngineResult<Vec<(String, PayloadSchemaType)>>;

    // ---- mutations ----

    /// Upsert a single point.
    async fn upsert(&self, collection: &str, point: VectorPoint) -> EngineResult<()>;

    /// Upsert a batch of points in a single transaction.
    async fn upsert_batch(&self, collection: &str, points: Vec<VectorPoint>) -> EngineResult<()>;

    /// Delete a point by id.
    async fn delete(&self, collection: &str, point_id: &PointId) -> EngineResult<()>;

    /// Delete a batch of points by id.
    async fn delete_batch(&self, collection: &str, point_ids: &[PointId]) -> EngineResult<()>;

    /// Delete every point matching the given filter.
    async fn delete_by_filter(&self, collection: &str, filter: VectorFilter) -> EngineResult<()>;

    /// Atomically replace the full payload of each listed point.
    async fn set_payload(
        &self,
        collection: &str,
        point_ids: Vec<PointId>,
        payload: Payload,
    ) -> EngineResult<()>;

    /// Merge the given fields into each point's payload (other keys preserved).
    async fn set_payload_fields(
        &self,
        collection: &str,
        point_ids: Vec<PointId>,
        fields: Payload,
    ) -> EngineResult<()>;

    /// Remove the given keys from each point's payload.
    async fn delete_payload(
        &self,
        collection: &str,
        point_ids: Vec<PointId>,
        keys: Vec<String>,
    ) -> EngineResult<()>;

    /// Paginated scan over live points in slot order.
    async fn scroll(
        &self,
        collection: &str,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> EngineResult<(Vec<VectorPoint>, Option<PointId>)>;

    /// Paginated scan over live points matching the given filter, in slot order.
    /// Limit counts only matching points and the cursor is the last match.
    /// A short page does not imply exhaustion with a filter present, page until an empty page.
    async fn scroll_by_filter(
        &self,
        collection: &str,
        filter: VectorFilter,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> EngineResult<(Vec<VectorPoint>, Option<PointId>)>;

    // ---- reads ----

    /// Full ANN / exact-scan search.
    async fn search(&self, collection: &str, query: SearchQuery)
        -> EngineResult<Vec<SearchResult>>;

    /// Fetch a single point by id.
    async fn get(&self, collection: &str, point_id: &PointId) -> EngineResult<Option<VectorPoint>>;

    /// Number of live points in a collection.
    async fn count(&self, collection: &str) -> EngineResult<u64>;

    /// Number of live points matching the given filter. Pure read.
    async fn count_by_filter(
        &self,
        collection: &str,
        filter: VectorFilter,
    ) -> EngineResult<u64>;

    /// Streaming search: yields results one by one as a stream.
    ///
    /// Default implementation executes a full search and streams the
    /// buffered results. Remote engines may override with a true gRPC
    /// streaming implementation. The stream is boxed to keep the trait
    /// object-safe.
    async fn search_stream(
        &self,
        collection: &str,
        query: SearchQuery,
    ) -> EngineResult<Pin<Box<dyn Stream<Item = EngineResult<SearchResult>> + Send>>> {
        let results = self.search(collection, query).await?;
        let stream = futures::stream::iter(results.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }

    /// Streaming scroll over a collection.
    ///
    /// Default implementation pages through `scroll` synchronously and
    /// yields points one by one. Engines with native streaming may
    /// override for efficiency.
    async fn scroll_stream(
        &self,
        collection: &str,
        batch_size: usize,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> EngineResult<Pin<Box<dyn Stream<Item = EngineResult<VectorPoint>> + Send>>> {
        let mut offset: Option<PointId> = None;
        let collection = collection.to_string();
        let mut all_points: Vec<VectorPoint> = Vec::new();
        loop {
            let (points, next) = self
                .scroll(
                    &collection,
                    batch_size,
                    offset.as_ref(),
                    with_payload,
                    with_vector,
                )
                .await?;
            let is_last = next.is_none();
            all_points.extend(points);
            offset = next;
            if is_last || offset.is_none() {
                break;
            }
        }
        let stream = futures::stream::iter(all_points.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }

    /// Streaming filtered scroll over points matching the given filter.
    ///
    /// Default implementation pages through `scroll_by_filter` and yields
    /// points one by one. Paging continues until an empty page because a short
    /// page does not imply exhaustion with a filter present.
    async fn scroll_by_filter_stream(
        &self,
        collection: &str,
        filter: VectorFilter,
        batch_size: usize,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> EngineResult<Pin<Box<dyn Stream<Item = EngineResult<VectorPoint>> + Send>>> {
        let mut offset: Option<PointId> = None;
        let collection = collection.to_string();
        let mut all_points: Vec<VectorPoint> = Vec::new();
        loop {
            let (points, next) = self
                .scroll_by_filter(
                    &collection,
                    filter.clone(),
                    batch_size,
                    offset.as_ref(),
                    with_payload,
                    with_vector,
                )
                .await?;
            let empty = points.is_empty();
            all_points.extend(points);
            offset = next;
            if empty {
                break;
            }
            if offset.is_none() {
                break;
            }
        }
        let stream = futures::stream::iter(all_points.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }
}

// ---- LocalVectorEngine implementation ---------------------------------------

#[async_trait]
impl VectorEngine for LocalVectorEngine {
    fn name(&self) -> &str {
        "simvec"
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn is_local(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_check(&self) -> EngineResult<HealthStatus> {
        Ok(HealthStatus::healthy(self.name(), self.version()))
    }

    async fn create_collection(&self, name: &str, config: &CollectionConfig) -> EngineResult<()> {
        let name = name.to_string();
        let config = config.clone();
        tokio::task::block_in_place(|| self.create_collection(&name, &config))
            .map_err(VectorEngineError::from)
    }

    async fn delete_collection(&self, name: &str) -> EngineResult<()> {
        let name = name.to_string();
        tokio::task::block_in_place(|| self.delete_collection(&name))
            .map_err(VectorEngineError::from)
    }

    fn collection_exists(&self, name: &str) -> bool {
        self.collections.read().contains_key(name)
    }

    fn get_index_metadata(&self, name: &str) -> Option<IndexMetadata> {
        let collections = self.collections.read();
        let store = collections.get(name)?;
        let meta = store.meta();
        let mut config = CollectionConfig::new(meta.vector_size, meta.distance);
        config.index = meta.index.clone();
        if meta.quantization_config.is_some() {
            config.quantization_config = meta.quantization_config.clone();
        }
        Some(IndexMetadata {
            name: name.to_string(),
            config,
            created_at: chrono::Utc::now(),
            vector_count: store.count(),
            index_name: None,
        })
    }

    async fn create_payload_index(
        &self,
        collection: &str,
        field: &str,
        schema: PayloadSchemaType,
    ) -> EngineResult<()> {
        let collection = collection.to_string();
        let field = field.to_string();
        tokio::task::block_in_place(|| {
            self.store(&collection)
                .and_then(|s| s.create_payload_index(&field, schema))
        })
        .map_err(VectorEngineError::from)
    }

    async fn delete_payload_index(&self, collection: &str, field: &str) -> EngineResult<()> {
        let collection = collection.to_string();
        let field = field.to_string();
        tokio::task::block_in_place(|| {
            self.store(&collection)
                .and_then(|s| s.delete_payload_index(&field))
                .map(|_| ())
        })
        .map_err(VectorEngineError::from)
    }

    async fn list_payload_indexes(
        &self,
        collection: &str,
    ) -> EngineResult<Vec<(String, PayloadSchemaType)>> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.store(&collection).map(|s| s.list_payload_indexes()))
            .map_err(VectorEngineError::from)
    }

    async fn upsert(&self, collection: &str, point: VectorPoint) -> EngineResult<()> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.upsert(&collection, point))
            .map_err(VectorEngineError::from)
    }

    async fn upsert_batch(&self, collection: &str, points: Vec<VectorPoint>) -> EngineResult<()> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.upsert_batch(&collection, &points))
            .map_err(VectorEngineError::from)
    }

    async fn delete(&self, collection: &str, point_id: &PointId) -> EngineResult<()> {
        let collection = collection.to_string();
        let point_id = point_id.clone();
        tokio::task::block_in_place(|| self.delete(&collection, &point_id))
            .map_err(VectorEngineError::from)
    }

    async fn delete_batch(&self, collection: &str, point_ids: &[PointId]) -> EngineResult<()> {
        let collection = collection.to_string();
        let ids: Vec<PointId> = point_ids.to_vec();
        tokio::task::block_in_place(|| self.delete_batch(&collection, &ids))
            .map_err(VectorEngineError::from)
    }

    async fn delete_by_filter(&self, collection: &str, filter: VectorFilter) -> EngineResult<()> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.delete_by_filter(&collection, &filter).map(|_| ()))
            .map_err(VectorEngineError::from)
    }

    async fn set_payload(
        &self,
        collection: &str,
        point_ids: Vec<PointId>,
        payload: Payload,
    ) -> EngineResult<()> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| {
            for id in &point_ids {
                self.set_payload(&collection, id, payload.clone())?;
            }
            Ok::<(), VectorSearchError>(())
        })
        .map_err(VectorEngineError::from)
    }

    async fn set_payload_fields(
        &self,
        collection: &str,
        point_ids: Vec<PointId>,
        fields: Payload,
    ) -> EngineResult<()> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| {
            for id in &point_ids {
                self.set_payload_fields(&collection, id, fields.clone())?;
            }
            Ok::<(), VectorSearchError>(())
        })
        .map_err(VectorEngineError::from)
    }

    async fn delete_payload(
        &self,
        collection: &str,
        point_ids: Vec<PointId>,
        keys: Vec<String>,
    ) -> EngineResult<()> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| {
            for id in &point_ids {
                self.delete_payload(&collection, id, keys.clone())?;
            }
            Ok::<(), VectorSearchError>(())
        })
        .map_err(VectorEngineError::from)
    }

    async fn scroll(
        &self,
        collection: &str,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> EngineResult<(Vec<VectorPoint>, Option<PointId>)> {
        let collection = collection.to_string();
        let offset = offset.cloned();
        tokio::task::block_in_place(|| {
            self.scroll(
                &collection,
                limit,
                offset.as_ref(),
                with_payload,
                with_vector,
            )
        })
        .map_err(VectorEngineError::from)
    }

    async fn scroll_by_filter(
        &self,
        collection: &str,
        filter: VectorFilter,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> EngineResult<(Vec<VectorPoint>, Option<PointId>)> {
        let collection = collection.to_string();
        let offset = offset.cloned();
        tokio::task::block_in_place(|| {
            self.scroll_by_filter(
                &collection,
                &filter,
                limit,
                offset.as_ref(),
                with_payload,
                with_vector,
            )
        })
        .map_err(VectorEngineError::from)
    }

    async fn search(
        &self,
        collection: &str,
        query: SearchQuery,
    ) -> EngineResult<Vec<SearchResult>> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.search(&collection, &query))
            .map_err(VectorEngineError::from)
    }

    async fn get(&self, collection: &str, point_id: &PointId) -> EngineResult<Option<VectorPoint>> {
        let collection = collection.to_string();
        let point_id = point_id.clone();
        tokio::task::block_in_place(|| self.get(&collection, &point_id))
            .map_err(VectorEngineError::from)
    }

    async fn count(&self, collection: &str) -> EngineResult<u64> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.count(&collection)).map_err(VectorEngineError::from)
    }

    async fn count_by_filter(
        &self,
        collection: &str,
        filter: VectorFilter,
    ) -> EngineResult<u64> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.count_by_filter(&collection, &filter))
            .map_err(VectorEngineError::from)
    }
}
