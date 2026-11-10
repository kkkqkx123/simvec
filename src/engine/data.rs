use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::{Builder as ThreadBuilder, JoinHandle};
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};

use super::*;
use super::{LocalVectorEngine, MaintenanceJob, TxnOp};
use crate::error::{Result, VectorSearchError};
use crate::metrics::MetricsSnapshot;
use crate::storage::{CollectionStore, WalPoint, WalRecord};
use crate::types::{
    CollectionConfig, CollectionInfo, CollectionStatus, HnswConfig, IndexConfig, IvfConfig,
    Payload, PayloadSchemaType, PointId, SearchQuery, SearchResult, VectorFilter, VectorPoint,
};

impl LocalVectorEngine {
    /// Upsert a point (WAL-backed).
    pub fn upsert(&self, collection: &str, point: VectorPoint) -> Result<()> {
        let store = self.store(collection)?;
        let result = store.apply_ops(&[WalRecord::Upsert {
            point: WalPoint::from_point(&point)?,
        }]);
        if result.is_err() {
            store.metrics().record_upsert_error();
        }
        result?;
        self.after_mutation(collection, &store);
        Ok(())
    }

    /// Upsert a batch of points (WAL-backed, single transaction).
    pub fn upsert_batch(&self, collection: &str, points: &[VectorPoint]) -> Result<()> {
        let store = self.store(collection)?;
        let ops: Result<Vec<_>> = points
            .iter()
            .map(|p| {
                Ok(WalRecord::Upsert {
                    point: WalPoint::from_point(p)?,
                })
            })
            .collect();
        let ops = match ops {
            Ok(ops) => ops,
            Err(e) => {
                store.metrics().record_upsert_error();
                return Err(e);
            }
        };
        let result = store.apply_ops(&ops);
        if result.is_err() {
            store.metrics().record_upsert_error();
        }
        result?;
        self.after_mutation(collection, &store);
        Ok(())
    }

    /// Delete a point by id (WAL-backed). Deleting a missing id is a no-op.
    pub fn delete(&self, collection: &str, point_id: &PointId) -> Result<()> {
        let store = self.store(collection)?;
        let result = store.apply_ops(&[WalRecord::Delete {
            point_id: point_id.to_string(),
        }]);
        if result.is_err() {
            store.metrics().record_delete_error();
        }
        result?;
        self.after_mutation(collection, &store);
        Ok(())
    }

    /// Delete a batch of points (WAL-backed, single transaction).
    pub fn delete_batch(&self, collection: &str, point_ids: &[PointId]) -> Result<()> {
        let store = self.store(collection)?;
        let result = store.apply_ops(&[WalRecord::DeleteBatch {
            point_ids: point_ids.iter().map(|id| id.to_string()).collect(),
        }]);
        if result.is_err() {
            store.metrics().record_delete_error();
        }
        result?;
        self.after_mutation(collection, &store);
        Ok(())
    }

    /// Delete every point matching `filter`. Returns the number deleted.
    pub fn delete_by_filter(&self, collection: &str, filter: &VectorFilter) -> Result<u64> {
        let store = self.store(collection)?;
        let deleted = match store.delete_by_filter(filter) {
            Ok(deleted) => deleted,
            Err(e) => {
                store.metrics().record_delete_error();
                return Err(e);
            }
        };
        self.after_mutation(collection, &store);
        Ok(deleted)
    }

    /// Exact full-scan search.
    ///
    /// Output scores follow the crate-wide contract: higher is better
    /// (cosine similarity, inner product, and `1/(1+distance)` for euclid).
    /// Remote Qdrant engines normalize their raw Euclid distances to the
    /// same contract at the client boundary.
    pub fn search(&self, collection: &str, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let store = self.store(collection)?;
        let results = store.search(query);
        if results.is_err() {
            store.metrics().record_search_error();
        }
        results
    }

    /// Fetch a point by id.
    pub fn get(&self, collection: &str, point_id: &PointId) -> Result<Option<VectorPoint>> {
        let store = self.store(collection)?;
        store.get(point_id)
    }

    /// Number of live points in a collection.
    pub fn count(&self, collection: &str) -> Result<u64> {
        Ok(self.store(collection)?.count())
    }

    /// Number of live points matching `filter`. Pure read.
    pub fn count_by_filter(&self, collection: &str, filter: &VectorFilter) -> Result<u64> {
        Ok(self.store(collection)?.count_by_filter(filter)?)
    }

    /// Replace the payload for a single point. The point must exist and not
    /// be tombstoned. The entire payload map is replaced atomically.
    pub fn set_payload(
        &self,
        collection: &str,
        point_id: &PointId,
        payload: Payload,
    ) -> Result<()> {
        let store = self.store(collection)?;
        store.set_payload(point_id, payload)
    }

    /// Remove specific keys from a point's payload. The remaining keys are
    /// preserved. If the point has no payload this is a no-op.
    pub fn delete_payload(
        &self,
        collection: &str,
        point_id: &PointId,
        keys: Vec<String>,
    ) -> Result<()> {
        let store = self.store(collection)?;
        store.delete_payload_keys(point_id, keys)
    }

    /// Set a single field on a point's payload (merge semantics). A missing
    /// payload is created containing just this field; all other keys are
    /// preserved. Applied atomically via a WAL-backed transaction.
    pub fn set_payload_field(
        &self,
        collection: &str,
        point_id: &PointId,
        key: String,
        value: serde_json::Value,
    ) -> Result<()> {
        self.set_payload_fields(collection, point_id, Payload::from([(key, value)]))
    }

    /// Merge the given fields into a point's payload within one WAL-backed
    /// transaction: keys in `fields` overwrite their previous values while
    /// all other keys are preserved. A missing payload is created.
    pub fn set_payload_fields(
        &self,
        collection: &str,
        point_id: &PointId,
        fields: Payload,
    ) -> Result<()> {
        let store = self.store(collection)?;
        store.set_payload_fields(point_id, fields)
    }

    /// Create a payload field index on a collection. The index is populated
    /// synchronously and its definition persisted with the collection.
    pub fn create_payload_index(
        &self,
        collection: &str,
        field: &str,
        schema: PayloadSchemaType,
    ) -> Result<()> {
        self.store(collection)?.create_payload_index(field, schema)
    }

    /// Drop the payload field index on `field`. Returns whether it existed.
    pub fn delete_payload_index(&self, collection: &str, field: &str) -> Result<bool> {
        self.store(collection)?.delete_payload_index(field)
    }

    /// All declared payload indexes as `(field, schema_type)` pairs.
    pub fn list_payload_indexes(
        &self,
        collection: &str,
    ) -> Result<Vec<(String, PayloadSchemaType)>> {
        Ok(self.store(collection)?.list_payload_indexes())
    }

    /// Paginated scan over live points in slot order.
    ///
    /// Returns up to `limit` points starting after `offset` (the last
    /// point_id from the previous page).
    pub fn scroll(
        &self,
        collection: &str,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        let store = self.store(collection)?;
        store.scroll(limit, offset, with_payload, with_vector)
    }

    /// Paginated scan over live points matching `filter`, in slot order.
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
        let store = self.store(collection)?;
        store.scroll_by_filter(filter, limit, offset, with_payload, with_vector)
    }

    /// Commit protocol entry for coordinated transactions.
    ///
    /// Ops are grouped by collection and each collection receives one WAL
    /// transaction. Collections are processed in lexicographic order so a
    /// crash always advances the per-collection water marks in a
    /// deterministic order. Replay is idempotent, so graph WAL replay may
    /// re-apply the same ops without double-applying data.
    ///
    /// The caller's transaction id is intentionally not carried into the
    /// per-collection logs: each collection assigns its own log sequence
    /// numbers, which is what keeps the log order, the apply order and the
    /// slot numbering of every collection consistent on its own. No atomicity
    /// is promised across collections — a crash can leave one collection
    /// committed and another not, and replay converges each one.
    pub fn apply_txn(&self, ops: Vec<TxnOp>) -> Result<()> {
        let mut by_collection: HashMap<String, Vec<WalRecord>> = HashMap::new();
        for op in ops {
            match op {
                TxnOp::Upsert { collection, point } => by_collection
                    .entry(collection)
                    .or_default()
                    .push(WalRecord::Upsert {
                        point: WalPoint::from_point(&point)?,
                    }),
                TxnOp::Delete {
                    collection,
                    point_id,
                } => by_collection
                    .entry(collection)
                    .or_default()
                    .push(WalRecord::Delete {
                        point_id: point_id.to_string(),
                    }),
            }
        }

        let mut names: Vec<String> = by_collection.keys().cloned().collect();
        names.sort_unstable();
        for name in names {
            let ops = by_collection.remove(&name).expect("key present");
            let store = self.store(&name)?;
            store.apply_txn(&ops)?;
            self.maybe_schedule_compaction(&name, &store);
        }
        Ok(())
    }
}
