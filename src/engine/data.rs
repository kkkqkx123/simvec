use std::collections::HashMap;

use super::{LocalVectorEngine, TxnOp};
use crate::error::{Result, VectorSearchError};
use crate::storage::{WalPoint, WalRecord};
use crate::types::{
    Payload, PayloadSchemaType, PointId, SearchQuery, SearchResult, VectorFilter, VectorPoint,
};

impl LocalVectorEngine {
    /// Upsert a point (WAL-backed).
    pub fn upsert(&self, collection: &str, point: VectorPoint) -> Result<()> {
        if let Some(named) = self.named_store(collection)? {
            return named
                .upsert_batch(std::slice::from_ref(&point))
                .inspect_err(|_| named.metrics().record_upsert_error());
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind
                .upsert_batch(std::slice::from_ref(&point))
                .inspect_err(|_| kind.metrics().record_upsert_error());
        }
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
        if let Some(named) = self.named_store(collection)? {
            return named
                .upsert_batch(points)
                .inspect_err(|_| named.metrics().record_upsert_error());
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind
                .upsert_batch(points)
                .inspect_err(|_| kind.metrics().record_upsert_error());
        }
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
        if let Some(named) = self.named_store(collection)? {
            return named
                .delete(point_id)
                .inspect_err(|_| named.metrics().record_delete_error());
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind
                .delete(point_id)
                .inspect_err(|_| kind.metrics().record_delete_error());
        }
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
        if let Some(named) = self.named_store(collection)? {
            return named
                .delete_batch(point_ids)
                .inspect_err(|_| named.metrics().record_delete_error());
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind
                .delete_batch(point_ids)
                .inspect_err(|_| kind.metrics().record_delete_error());
        }
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
        if let Some(named) = self.named_store(collection)? {
            return named
                .delete_by_filter(filter)
                .inspect_err(|_| named.metrics().record_delete_error());
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind
                .delete_by_filter(filter)
                .inspect_err(|_| kind.metrics().record_delete_error());
        }
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
        if let Some(named) = self.named_store(collection)? {
            return named
                .search(query)
                .inspect_err(|_| named.metrics().record_search_error());
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind
                .search(query)
                .inspect_err(|_| kind.metrics().record_search_error());
        }
        let store = self.store(collection)?;
        let results = store.search(query);
        if results.is_err() {
            store.metrics().record_search_error();
        }
        results
    }

    /// Fetch a point by id.
    pub fn get(&self, collection: &str, point_id: &PointId) -> Result<Option<VectorPoint>> {
        if let Some(named) = self.named_store(collection)? {
            return named.get(point_id);
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind.get(point_id);
        }
        let store = self.store(collection)?;
        store.get(point_id)
    }

    /// Number of live points in a collection.
    pub fn count(&self, collection: &str) -> Result<u64> {
        if let Some(named) = self.named_store(collection)? {
            return Ok(named.count());
        }
        if let Some(kind) = self.kind_store(collection)? {
            return Ok(kind.count());
        }
        Ok(self.store(collection)?.count())
    }

    /// Number of live points matching `filter`. Pure read.
    pub fn count_by_filter(&self, collection: &str, filter: &VectorFilter) -> Result<u64> {
        if let Some(named) = self.named_store(collection)? {
            return named.count_by_filter(filter);
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind.count_by_filter(filter);
        }
        self.store(collection)?.count_by_filter(filter)
    }

    /// Replace the payload for a single point. The point must exist and not
    /// be tombstoned. The entire payload map is replaced atomically.
    pub fn set_payload(
        &self,
        collection: &str,
        point_id: &PointId,
        payload: Payload,
    ) -> Result<()> {
        if let Some(named) = self.named_store(collection)? {
            return named.set_payload(point_id, payload);
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind.set_payload(point_id, payload);
        }
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
        if let Some(named) = self.named_store(collection)? {
            return named.delete_payload_keys(point_id, &keys);
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind.delete_payload_keys(point_id, &keys);
        }
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
        if let Some(named) = self.named_store(collection)? {
            return named.set_payload_fields(point_id, fields);
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind.set_payload_fields(point_id, fields);
        }
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
        if self.kind_store(collection)?.is_some() || self.named_store(collection)?.is_some() {
            return Err(VectorSearchError::InvalidConfig(
                "exact-scan collections do not support payload field indexes".to_string(),
            ));
        }
        self.store(collection)?.create_payload_index(field, schema)
    }

    /// Drop the payload field index on `field`. Returns whether it existed.
    pub fn delete_payload_index(&self, collection: &str, field: &str) -> Result<bool> {
        if self.kind_store(collection)?.is_some() || self.named_store(collection)?.is_some() {
            return Err(VectorSearchError::InvalidConfig(
                "exact-scan collections do not support payload field indexes".to_string(),
            ));
        }
        self.store(collection)?.delete_payload_index(field)
    }

    /// All declared payload indexes as `(field, schema_type)` pairs.
    pub fn list_payload_indexes(
        &self,
        collection: &str,
    ) -> Result<Vec<(String, PayloadSchemaType)>> {
        if self.kind_store(collection)?.is_some() || self.named_store(collection)?.is_some() {
            return Ok(Vec::new());
        }
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
        if let Some(named) = self.named_store(collection)? {
            return named.scroll(limit, offset, with_payload, with_vector);
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind.scroll(limit, offset, with_payload, with_vector);
        }
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
        if let Some(named) = self.named_store(collection)? {
            return named.scroll_by_filter(filter, limit, offset, with_payload, with_vector);
        }
        if let Some(kind) = self.kind_store(collection)? {
            return kind.scroll_by_filter(filter, limit, offset, with_payload, with_vector);
        }
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
                TxnOp::Upsert { collection, point } => {
                    if self.named_store(&collection)?.is_some() {
                        by_collection
                            .entry(collection)
                            .or_default()
                            .push(crate::named_store::named_upsert_record(&point)?);
                    } else if self.kind_store(&collection)?.is_some() {
                        by_collection
                            .entry(collection)
                            .or_default()
                            .push(kind_upsert_record(&point)?);
                    } else {
                        by_collection
                            .entry(collection)
                            .or_default()
                            .push(WalRecord::Upsert {
                                point: WalPoint::from_point(&point)?,
                            });
                    }
                }
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
            if let Some(named) = self.named_store(&name)? {
                named.apply_records(&ops)?;
                continue;
            }
            if let Some(kind) = self.kind_store(&name)? {
                kind.apply_records(&ops)?;
                continue;
            }
            let store = self.store(&name)?;
            store.apply_txn(&ops)?;
            self.maybe_schedule_compaction(&name, &store);
        }
        Ok(())
    }
}

/// WAL record for a kind-collection upsert inside a coordinated transaction.
fn kind_upsert_record(point: &VectorPoint) -> Result<WalRecord> {
    crate::kind_store::kind_upsert_record(point)
}
