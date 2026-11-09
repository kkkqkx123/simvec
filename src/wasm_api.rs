//! WASM boundary over the in-memory engine.
//!
//! Thin JSON boundary mirroring the C API shapes: complex values cross
//! as JSON text reusing the existing serde forms, vectors cross as
//! single-precision arrays. No retrieval logic lives here.

use wasm_bindgen::prelude::*;

use crate::error::VectorSearchError;
use crate::memory_engine::{MemoryEngine, MemorySnapshotCollection, MemoryTxnOp};
use crate::types::{CollectionConfig, Payload, PayloadSchemaType, PointId, SearchQuery, VectorPoint};

const CODE_INTERNAL: i32 = 2;
const CODE_NOMEM: i32 = 7;
const CODE_IOERR: i32 = 10;
const CODE_CORRUPT: i32 = 11;
const CODE_NOTFOUND: i32 = 12;
const CODE_ALREADY_EXISTS: i32 = 13;
const CODE_INVALID_CONFIG: i32 = 16;
const CODE_INVALID_DIM: i32 = 19;
const CODE_MISUSE: i32 = 20;
const CODE_NON_FINITE: i32 = 21;
const CODE_CAPACITY_EXCEEDED: i32 = 22;
const CODE_FILTER: i32 = 23;

fn code_for_error(err: &VectorSearchError) -> i32 {
    match err {
        VectorSearchError::CollectionNotFound(_) => CODE_NOTFOUND,
        VectorSearchError::CollectionAlreadyExists(_) => CODE_ALREADY_EXISTS,
        VectorSearchError::CollectionIncomplete { .. }
        | VectorSearchError::CorruptData(_)
        | VectorSearchError::Serialization(_) => CODE_CORRUPT,
        VectorSearchError::InvalidCollectionName(_) | VectorSearchError::InvalidPointId(_) => {
            CODE_MISUSE
        }
        VectorSearchError::InvalidConfig(_) | VectorSearchError::UnsupportedMetric(_) => {
            CODE_INVALID_CONFIG
        }
        VectorSearchError::InsufficientSpace { .. } => CODE_NOMEM,
        VectorSearchError::CapacityExceeded { .. } => CODE_CAPACITY_EXCEEDED,
        VectorSearchError::InvalidVectorDimension { .. } => CODE_INVALID_DIM,
        VectorSearchError::NonFiniteElement(_) => CODE_NON_FINITE,
        VectorSearchError::Filter(_) => CODE_FILTER,
        VectorSearchError::Io(_) => CODE_IOERR,
        VectorSearchError::Json(_) => CODE_MISUSE,
        VectorSearchError::Internal(_) => CODE_INTERNAL,
    }
}

fn err_to_js(err: impl ToString) -> JsValue {
    JsValue::from_str(&err.to_string())
}

fn engine_err_to_js(err: VectorSearchError) -> JsValue {
    JsValue::from_str(&format!("[simvec:{}] {err}", code_for_error(&err)))
}

fn input_err_to_js(err: impl ToString) -> JsValue {
    JsValue::from_str(&format!("[simvec:{CODE_MISUSE}] {err}", err = err.to_string()))
}

fn parse_input_json<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, JsValue> {
    serde_json::from_str(text).map_err(|err| {
        JsValue::from_str(&format!("[simvec:{CODE_MISUSE}] invalid JSON input: {err}"))
    })
}

fn parse_schema_type(raw: &str) -> Option<PayloadSchemaType> {
    match raw.trim().to_lowercase().as_str() {
        "keyword" => Some(PayloadSchemaType::Keyword),
        "integer" => Some(PayloadSchemaType::Integer),
        "float" => Some(PayloadSchemaType::Float),
        "text" => Some(PayloadSchemaType::Text),
        "bool" => Some(PayloadSchemaType::Bool),
        "geo" => Some(PayloadSchemaType::Geo),
        "datetime" => Some(PayloadSchemaType::Datetime),
        _ => None,
    }
}

#[derive(serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum TxnOpDto {
    Upsert { collection: String, point: VectorPoint },
    Delete { collection: String, id: String },
}

/// Browser engine handle. One instance owns all its collections.
#[wasm_bindgen]
pub struct WasmEngine {
    inner: MemoryEngine,
}

#[wasm_bindgen]
impl WasmEngine {
    /// Create an empty engine with no collections.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmEngine {
        WasmEngine {
            inner: MemoryEngine::new(),
        }
    }

    /// Create a collection from a JSON-encoded configuration.
    pub fn create_collection(&self, name: &str, config_json: Option<String>) -> Result<(), JsValue> {
        let config: CollectionConfig = match config_json {
            Some(raw) => parse_input_json(&raw)?,
            None => CollectionConfig::default(),
        };
        self.inner
            .create_collection(name, &config)
            .map_err(engine_err_to_js)
    }

    /// Drop a collection and all its points.
    pub fn delete_collection(&self, name: &str) -> Result<(), JsValue> {
        self.inner.delete_collection(name).map_err(engine_err_to_js)
    }

    /// Whether a collection exists.
    pub fn collection_exists(&self, name: &str) -> bool {
        self.inner.collection_exists(name)
    }

    /// List collection names as a JSON array string.
    pub fn list_collections(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner.collection_names()).map_err(err_to_js)
    }

    /// Collection info as a JSON string.
    pub fn collection_info(&self, name: &str) -> Result<String, JsValue> {
        let info = self.inner.collection_info(name).map_err(engine_err_to_js)?;
        serde_json::to_string(&info).map_err(err_to_js)
    }

    /// Stored collection configuration as a JSON string.
    pub fn collection_config(&self, name: &str) -> Result<String, JsValue> {
        let config = self
            .inner
            .collection_config(name)
            .map_err(engine_err_to_js)?;
        serde_json::to_string(&config).map_err(err_to_js)
    }

    /// Upsert one point. Payload crosses as an optional JSON object string.
    pub fn upsert(
        &self,
        collection: &str,
        id: &str,
        vector: Vec<f32>,
        payload_json: Option<String>,
    ) -> Result<(), JsValue> {
        let payload: Option<Payload> = match payload_json {
            Some(raw) => Some(parse_input_json(&raw)?),
            None => None,
        };
        let mut point = VectorPoint::new(PointId::from(id), vector);
        if let Some(payload) = payload {
            point = point.with_payload(payload);
        }
        self.inner.upsert(collection, point).map_err(engine_err_to_js)
    }

    /// Upsert a batch from a JSON array of points.
    pub fn upsert_batch(&self, collection: &str, points_json: &str) -> Result<(), JsValue> {
        let points: Vec<VectorPoint> = parse_input_json(points_json)?;
        self.inner
            .upsert_batch(collection, &points)
            .map_err(engine_err_to_js)
    }

    /// Delete one point by id. Missing ids are a no-op.
    pub fn delete(&self, collection: &str, id: &str) -> Result<(), JsValue> {
        self.inner
            .delete(collection, &PointId::from(id))
            .map_err(engine_err_to_js)
    }

    /// Delete a batch of ids from a JSON string array.
    pub fn delete_batch(&self, collection: &str, ids_json: &str) -> Result<(), JsValue> {
        let ids: Vec<String> = parse_input_json(ids_json)?;
        let point_ids: Vec<PointId> = ids.iter().map(|id| PointId::from(id.as_str())).collect();
        self.inner
            .delete_batch(collection, &point_ids)
            .map_err(engine_err_to_js)
    }

    /// Delete every point matching a JSON-encoded filter. Returns deleted count.
    pub fn delete_by_filter(&self, collection: &str, filter_json: &str) -> Result<f64, JsValue> {
        let filter = parse_input_json(filter_json)?;
        let deleted = self
            .inner
            .delete_by_filter(collection, &filter)
            .map_err(engine_err_to_js)?;
        Ok(deleted as f64)
    }

    /// Commit a JSON array of upsert/delete operations in collection order.
    pub fn apply_txn(&self, ops_json: &str) -> Result<(), JsValue> {
        let ops: Vec<TxnOpDto> = parse_input_json(ops_json)?;
        let ops: Vec<MemoryTxnOp> = ops
            .into_iter()
            .map(|item| match item {
                TxnOpDto::Upsert { collection, point } => MemoryTxnOp::Upsert { collection, point },
                TxnOpDto::Delete { collection, id } => MemoryTxnOp::Delete {
                    collection,
                    point_id: PointId::from(id.as_str()),
                },
            })
            .collect();
        self.inner.apply_txn(ops).map_err(engine_err_to_js)
    }

    /// Search with a JSON-encoded query. Returns results as a JSON string.
    pub fn search(&self, collection: &str, query_json: &str) -> Result<String, JsValue> {
        let query: SearchQuery = parse_input_json(query_json)?;
        let results = self.inner.search(collection, &query).map_err(engine_err_to_js)?;
        serde_json::to_string(&results).map_err(err_to_js)
    }

    /// Fetch one point by id. Missing points report not-found.
    pub fn get(&self, collection: &str, id: &str) -> Result<String, JsValue> {
        let point = self
            .inner
            .get(collection, &PointId::from(id))
            .map_err(engine_err_to_js)?;
        match point {
            Some(point) => serde_json::to_string(&point).map_err(err_to_js),
            None => Err(JsValue::from_str(&format!(
                "[simvec:{CODE_NOTFOUND}] point not found: {id}"
            ))),
        }
    }

    /// Live point count as a host number.
    pub fn count(&self, collection: &str) -> Result<f64, JsValue> {
        let count = self.inner.count(collection).map_err(engine_err_to_js)?;
        Ok(count as f64)
    }

    /// Paginated scan as a JSON object with points and the next offset.
    pub fn scroll(
        &self,
        collection: &str,
        limit: u32,
        offset: Option<String>,
        with_payload: bool,
        with_vector: bool,
    ) -> Result<String, JsValue> {
        let offset_id = offset.as_deref().map(PointId::from);
        let (points, next) = self
            .inner
            .scroll(
                collection,
                limit as usize,
                offset_id.as_ref(),
                Some(with_payload),
                Some(with_vector),
            )
            .map_err(engine_err_to_js)?;
        let body = serde_json::json!({
            "points": points,
            "next_offset": next.map(|id| id.to_string()),
        });
        serde_json::to_string(&body).map_err(err_to_js)
    }

    /// Replace the payload for one point from a JSON object string.
    pub fn set_payload(
        &self,
        collection: &str,
        id: &str,
        payload_json: &str,
    ) -> Result<(), JsValue> {
        let payload: Payload = parse_input_json(payload_json)?;
        self.inner
            .set_payload(collection, &PointId::from(id), payload)
            .map_err(engine_err_to_js)
    }

    /// Merge fields into one point payload from a JSON object string.
    pub fn merge_payload(
        &self,
        collection: &str,
        id: &str,
        fields_json: &str,
    ) -> Result<(), JsValue> {
        let fields: Payload = parse_input_json(fields_json)?;
        self.inner
            .set_payload_fields(collection, &PointId::from(id), fields)
            .map_err(engine_err_to_js)
    }

    /// Set one payload field from a JSON value string.
    pub fn set_payload_field(
        &self,
        collection: &str,
        id: &str,
        key: &str,
        value_json: &str,
    ) -> Result<(), JsValue> {
        let value: serde_json::Value = parse_input_json(value_json)?;
        self.inner
            .set_payload_field(collection, &PointId::from(id), key.to_string(), value)
            .map_err(engine_err_to_js)
    }

    /// Remove payload keys given as a JSON array string.
    pub fn delete_payload_keys(
        &self,
        collection: &str,
        id: &str,
        keys_json: &str,
    ) -> Result<(), JsValue> {
        let keys: Vec<String> = parse_input_json(keys_json)?;
        self.inner
            .delete_payload_keys(collection, &PointId::from(id), keys)
            .map_err(engine_err_to_js)
    }

    /// Declare a payload field index. Memory form tracks the declaration only.
    pub fn create_payload_index(
        &self,
        collection: &str,
        field: &str,
        schema: &str,
    ) -> Result<(), JsValue> {
        let schema_type = parse_schema_type(schema).ok_or_else(|| {
            input_err_to_js(format!("unknown payload schema: {schema}"))
        })?;
        self.inner
            .create_payload_index(collection, field, schema_type)
            .map_err(engine_err_to_js)
    }

    /// Drop a payload field index. Returns whether it existed.
    pub fn delete_payload_index(&self, collection: &str, field: &str) -> Result<bool, JsValue> {
        self.inner
            .delete_payload_index(collection, field)
            .map_err(engine_err_to_js)
    }

    /// List payload indexes as a JSON string.
    pub fn list_payload_indexes(&self, collection: &str) -> Result<String, JsValue> {
        let indexes = self
            .inner
            .list_payload_indexes(collection)
            .map_err(engine_err_to_js)?;
        serde_json::to_string(&indexes).map_err(err_to_js)
    }

    /// Index build is a compatible no-op reporting not-effective.
    pub fn build_index(&self, collection: &str) -> Result<bool, JsValue> {
        self.inner.build_index(collection).map_err(engine_err_to_js)
    }

    /// Whether an index is published. Always false in memory form.
    pub fn has_index(&self, collection: &str) -> bool {
        self.inner.has_index(collection)
    }

    /// Drop the published index. No-op in memory form.
    pub fn drop_index(&self, collection: &str) -> Result<(), JsValue> {
        self.inner.drop_index(collection).map_err(engine_err_to_js)
    }

    /// Drop the published index alias. No-op in memory form.
    pub fn drop_published_index(&self, collection: &str) -> Result<(), JsValue> {
        self.inner
            .drop_published_index(collection)
            .map_err(engine_err_to_js)
    }

    /// Quantization build is a compatible no-op reporting not-effective.
    pub fn build_quantization(&self, collection: &str) -> Result<bool, JsValue> {
        self.inner
            .build_quantization(collection)
            .map_err(engine_err_to_js)
    }

    /// Whether quantization is ready. Always false in memory form.
    pub fn has_quantization(&self, collection: &str) -> bool {
        self.inner.has_quantization(collection)
    }

    /// Reclaim space eagerly. Returns the live count as a host number.
    pub fn compact(&self, collection: &str) -> Result<f64, JsValue> {
        let live = self.inner.compact(collection).map_err(engine_err_to_js)?;
        Ok(live as f64)
    }

    /// Drain pending slots. No-op in memory form.
    pub fn drain_pending(&self, collection: &str) -> Result<(), JsValue> {
        self.inner.drain_pending(collection);
        Ok(())
    }

    /// Maintenance sweep. No-op in memory form.
    pub fn run_maintenance_sweep(&self) -> Result<(), JsValue> {
        self.inner.run_maintenance_sweep();
        Ok(())
    }

    /// Metrics snapshot as a JSON string.
    pub fn collection_metrics(&self, collection: &str) -> Result<String, JsValue> {
        let snapshot = self
            .inner
            .collection_metrics(collection)
            .map_err(engine_err_to_js)?;
        serde_json::to_string(&snapshot).map_err(err_to_js)
    }

    /// Monotonic revision advanced by every state-changing mutation.
    pub fn revision(&self) -> f64 {
        self.inner.revision() as f64
    }

    /// Export the whole engine as a versioned JSON snapshot.
    pub fn export_snapshot(&self) -> Result<String, JsValue> {
        let snapshot = self.inner.export_snapshot().map_err(engine_err_to_js)?;
        serde_json::to_string(&snapshot).map_err(err_to_js)
    }

    /// Replace the whole engine from a snapshot after validating every entry.
    pub fn import_snapshot(&self, snapshot_json: &str) -> Result<(), JsValue> {
        let snapshot = parse_input_json(snapshot_json)?;
        self.inner.import_snapshot(snapshot).map_err(engine_err_to_js)
    }

    /// Export one collection as a JSON snapshot entry.
    pub fn export_collection(&self, name: &str) -> Result<String, JsValue> {
        let entry = self
            .inner
            .export_collection(name)
            .map_err(engine_err_to_js)?;
        serde_json::to_string(&entry).map_err(err_to_js)
    }

    /// Insert or replace one collection from a snapshot entry.
    pub fn import_collection(&self, collection_json: &str) -> Result<(), JsValue> {
        let entry: MemorySnapshotCollection = parse_input_json(collection_json)?;
        self.inner
            .import_collection(entry)
            .map_err(engine_err_to_js)
    }

}
