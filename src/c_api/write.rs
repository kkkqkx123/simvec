//! Point writes, payload edits and payload indexes over FFI.

use std::ffi::{c_char, c_int};

use crate::types::PointId;
use crate::{Payload, PayloadSchemaType, TxnOp, VectorFilter, VectorPoint};

use super::error::{
    emit_json, guard, input_floats, input_str, opt_input_str, parse_input_json,
    simvec_error_code_t,
};
use super::types::simvec_engine_t;

/// One batch point as carried by `simvec_upsert_batch` JSON.
#[derive(serde::Deserialize)]
struct BatchPoint {
    id: String,
    vector: Vec<f32>,
    payload: Option<Payload>,
}

/// Translate a schema name into its declared payload type.
fn parse_schema_type(raw: &str) -> Option<PayloadSchemaType> {
    match raw.trim().to_ascii_lowercase().as_str() {
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

/// Upsert one point; `payload_json` may be null for a bare vector.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` and `id` must be valid null-terminated UTF-8 strings.
/// - `vector` must point to `vector_len` readable floats.
/// - `payload_json` must be null or a valid JSON object string.
#[no_mangle]
pub unsafe extern "C" fn simvec_upsert(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    id: *const c_char,
    vector: *const f32,
    vector_len: usize,
    payload_json: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let id = match input_str(id) {
            Ok(id) => id,
            Err(code) => return handle.fail_code(code, "invalid point id".to_string()),
        };
        let vector = match input_floats(vector, vector_len) {
            Ok(vector) => vector,
            Err(code) => return handle.fail_code(code, "invalid vector pointer".to_string()),
        };
        let raw_payload = match opt_input_str(payload_json) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid payload string".to_string()),
        };
        let payload: Option<Payload> = match raw_payload {
            None => None,
            Some(text) => match parse_input_json(text) {
                Ok(payload) => Some(payload),
                Err(code) => return handle.fail_code(code, "invalid payload JSON".to_string()),
            },
        };
        let mut point = VectorPoint::new(id.to_string(), vector);
        if let Some(payload) = payload {
            point = point.with_payload(payload);
        }
        match handle.engine.upsert(collection, point) {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Upsert a JSON array of `{id, vector, payload?}` points in one transaction.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` and `points_json` must be valid null-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn simvec_upsert_batch(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    points_json: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let raw = match input_str(points_json) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid points string".to_string()),
        };
        let batch: Vec<BatchPoint> = match parse_input_json(raw) {
            Ok(batch) => batch,
            Err(code) => return handle.fail_code(code, "invalid points JSON".to_string()),
        };
        let points: Vec<VectorPoint> = batch
            .into_iter()
            .map(|item| {
                let mut point = VectorPoint::new(item.id, item.vector);
                if let Some(payload) = item.payload {
                    point = point.with_payload(payload);
                }
                point
            })
            .collect();
        match handle.engine.upsert_batch(collection, &points) {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Delete one point by id; deleting a missing id is a no-op.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` and `id` must be valid null-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn simvec_delete(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    id: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let id = match input_str(id) {
            Ok(id) => id,
            Err(code) => return handle.fail_code(code, "invalid point id".to_string()),
        };
        match handle.engine.delete(collection, &PointId::from(id)) {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Delete a JSON array of point id strings in one transaction.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` and `ids_json` must be valid null-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn simvec_delete_batch(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    ids_json: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let raw = match input_str(ids_json) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid ids string".to_string()),
        };
        let ids: Vec<String> = match parse_input_json(raw) {
            Ok(ids) => ids,
            Err(code) => return handle.fail_code(code, "invalid ids JSON".to_string()),
        };
        let ids: Vec<PointId> = ids.iter().map(|id| PointId::from(id.as_str())).collect();
        match handle.engine.delete_batch(collection, &ids) {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Delete every point matching a JSON `VectorFilter`; `*out` takes the count.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` and `filter_json` must be valid null-terminated UTF-8 strings.
/// - `out` must be a valid out pointer.
#[no_mangle]
pub unsafe extern "C" fn simvec_delete_by_filter(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    filter_json: *const c_char,
    out: *mut u64,
) -> c_int {
    guard(engine, |handle| {
        if out.is_null() {
            return handle.fail_code(
                simvec_error_code_t::SIMVEC_MISUSE,
                "null out pointer".to_string(),
            );
        }
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let raw = match input_str(filter_json) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid filter string".to_string()),
        };
        let filter: VectorFilter = match parse_input_json(raw) {
            Ok(filter) => filter,
            Err(code) => return handle.fail_code(code, "invalid filter JSON".to_string()),
        };
        match handle.engine.delete_by_filter(collection, &filter) {
            Ok(deleted) => {
                *out = deleted;
                handle.ok()
            }
            Err(err) => handle.fail(&err),
        }
    })
}

/// Replace the whole payload of one point with a JSON object.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection`, `id` and `payload_json` must be valid null-terminated
///   UTF-8 strings; the payload must be a JSON object.
#[no_mangle]
pub unsafe extern "C" fn simvec_set_payload(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    id: *const c_char,
    payload_json: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let id = match input_str(id) {
            Ok(id) => id,
            Err(code) => return handle.fail_code(code, "invalid point id".to_string()),
        };
        let raw = match input_str(payload_json) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid payload string".to_string()),
        };
        let payload: Payload = match parse_input_json(raw) {
            Ok(payload) => payload,
            Err(code) => return handle.fail_code(code, "invalid payload JSON".to_string()),
        };
        match handle
            .engine
            .set_payload(collection, &PointId::from(id), payload)
        {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Merge a JSON object into one point's payload, preserving other keys.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection`, `id` and `fields_json` must be valid null-terminated
///   UTF-8 strings; the fields must be a JSON object.
#[no_mangle]
pub unsafe extern "C" fn simvec_merge_payload(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    id: *const c_char,
    fields_json: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let id = match input_str(id) {
            Ok(id) => id,
            Err(code) => return handle.fail_code(code, "invalid point id".to_string()),
        };
        let raw = match input_str(fields_json) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid fields string".to_string()),
        };
        let fields: Payload = match parse_input_json(raw) {
            Ok(fields) => fields,
            Err(code) => return handle.fail_code(code, "invalid fields JSON".to_string()),
        };
        match handle
            .engine
            .set_payload_fields(collection, &PointId::from(id), fields)
        {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Remove keys listed in a JSON string array from one point's payload.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection`, `id` and `keys_json` must be valid null-terminated UTF-8
///   strings; the keys must be a JSON string array.
#[no_mangle]
pub unsafe extern "C" fn simvec_delete_payload_keys(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    id: *const c_char,
    keys_json: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let id = match input_str(id) {
            Ok(id) => id,
            Err(code) => return handle.fail_code(code, "invalid point id".to_string()),
        };
        let raw = match input_str(keys_json) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid keys string".to_string()),
        };
        let keys: Vec<String> = match parse_input_json(raw) {
            Ok(keys) => keys,
            Err(code) => return handle.fail_code(code, "invalid keys JSON".to_string()),
        };
        match handle
            .engine
            .delete_payload(collection, &PointId::from(id), keys)
        {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Create a payload field index; `schema` names keyword/integer/float/text/
/// bool/geo/datetime case-insensitively.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection`, `field` and `schema` must be valid null-terminated UTF-8
///   strings.
#[no_mangle]
pub unsafe extern "C" fn simvec_create_payload_index(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    field: *const c_char,
    schema: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let field = match input_str(field) {
            Ok(field) => field,
            Err(code) => return handle.fail_code(code, "invalid field name".to_string()),
        };
        let schema = match input_str(schema) {
            Ok(schema) => schema,
            Err(code) => return handle.fail_code(code, "invalid schema name".to_string()),
        };
        let Some(schema_type) = parse_schema_type(schema) else {
            return handle.fail_code(
                simvec_error_code_t::SIMVEC_MISUSE,
                format!("unknown payload schema: {schema}"),
            );
        };
        match handle
            .engine
            .create_payload_index(collection, field, schema_type)
        {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Drop a payload field index; `*out` receives 1 when one existed.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` and `field` must be valid null-terminated UTF-8 strings.
/// - `out` must be a valid out pointer.
#[no_mangle]
pub unsafe extern "C" fn simvec_delete_payload_index(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    field: *const c_char,
    out: *mut c_int,
) -> c_int {
    guard(engine, |handle| {
        if out.is_null() {
            return handle.fail_code(
                simvec_error_code_t::SIMVEC_MISUSE,
                "null out pointer".to_string(),
            );
        }
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let field = match input_str(field) {
            Ok(field) => field,
            Err(code) => return handle.fail_code(code, "invalid field name".to_string()),
        };
        match handle.engine.delete_payload_index(collection, field) {
            Ok(existed) => {
                *out = i32::from(existed);
                handle.ok()
            }
            Err(err) => handle.fail(&err),
        }
    })
}

/// Set one payload field on a point, preserving all other keys.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection`, `id` and `key` must be valid null-terminated UTF-8 strings.
/// - `value_json` must be a valid null-terminated UTF-8 string holding any JSON value.
#[no_mangle]
pub unsafe extern "C" fn simvec_set_payload_field(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    id: *const c_char,
    key: *const c_char,
    value_json: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let id = match input_str(id) {
            Ok(id) => id,
            Err(code) => return handle.fail_code(code, "invalid point id".to_string()),
        };
        let key = match input_str(key) {
            Ok(key) => key,
            Err(code) => return handle.fail_code(code, "invalid field name".to_string()),
        };
        let raw = match input_str(value_json) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid value string".to_string()),
        };
        let value: serde_json::Value = match parse_input_json(raw) {
            Ok(value) => value,
            Err(code) => return handle.fail_code(code, "invalid value JSON".to_string()),
        };
        match handle.engine.set_payload_field(
            collection,
            &PointId::from(id),
            key.to_string(),
            value,
        ) {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// One transaction operation as carried by `simvec_apply_txn` JSON.
#[derive(serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")] 
enum TxnOpDto {
    Upsert { collection: String, point: VectorPoint },
    Delete { collection: String, id: String },
}

/// Commit a JSON array of upsert/delete operations grouped per collection.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `ops_json` must be a valid null-terminated UTF-8 string holding a JSON
///   array of `{op: upsert|delete, ...}` objects.
#[no_mangle]
pub unsafe extern "C" fn simvec_apply_txn(
    engine: *mut simvec_engine_t,
    ops_json: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let raw = match input_str(ops_json) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid ops string".to_string()),
        };
        let ops: Vec<TxnOpDto> = match parse_input_json(raw) {
            Ok(ops) => ops,
            Err(code) => return handle.fail_code(code, "invalid ops JSON".to_string()),
        };
        let ops: Vec<TxnOp> = ops
            .into_iter()
            .map(|item| match item {
                TxnOpDto::Upsert { collection, point } => TxnOp::Upsert { collection, point },
                TxnOpDto::Delete { collection, id } => TxnOp::Delete {
                    collection,
                    point_id: PointId::from(id.as_str()),
                },
            })
            .collect();
        match handle.engine.apply_txn(ops) {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// List declared payload indexes as a JSON array of field/schema pairs.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer freed with `simvec_free_string`.
#[no_mangle]
pub unsafe extern "C" fn simvec_list_payload_indexes(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        match handle.engine.list_payload_indexes(collection) {
            Ok(indexes) => {
                let items: Vec<serde_json::Value> = indexes
                    .into_iter()
                    .map(|(field, schema)| {
                        serde_json::json!({"field": field, "schema": schema.as_str()})
                    })
                    .collect();
                match serde_json::to_string(&items) {
                    Ok(body) => {
                        let emit = emit_json(out, &body);
                        if emit == simvec_error_code_t::SIMVEC_OK as c_int {
                            handle.ok()
                        } else {
                            handle.fail_code(
                                simvec_error_code_t::SIMVEC_ERROR,
                                "failed to hand out payload indexes".to_string(),
                            )
                        }
                    }
                    Err(err) => handle.fail_code(
                        simvec_error_code_t::SIMVEC_INTERNAL,
                        format!("failed to serialize payload indexes: {err}"),
                    ),
                }
            }
            Err(err) => handle.fail(&err),
        }
    })
}
