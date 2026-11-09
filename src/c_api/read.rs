//! Reads and similarity search over FFI.

use std::ffi::{c_char, c_int};

use crate::types::PointId;
use crate::{SearchQuery, VectorFilter};

use super::error::{
    emit_json, guard, input_floats, input_str, opt_input_str, parse_input_json, simvec_error_code_t,
};
use super::types::simvec_engine_t;

/// Run a JSON `SearchQuery`; results come back as a JSON `SearchResult` array.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` and `query_json` must be valid null-terminated UTF-8 strings.
/// - `out` must be a valid out pointer freed with `simvec_free_string`.
#[no_mangle]
pub unsafe extern "C" fn simvec_search(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    query_json: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let raw = match input_str(query_json) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid query string".to_string()),
        };
        let query: SearchQuery = match parse_input_json(raw) {
            Ok(query) => query,
            Err(code) => return handle.fail_code(code, "invalid query JSON".to_string()),
        };
        match handle.engine.search(collection, &query) {
            Ok(results) => match serde_json::to_string(&results) {
                Ok(body) => {
                    let emit = emit_json(out, &body);
                    if emit == simvec_error_code_t::SIMVEC_OK as c_int {
                        handle.ok()
                    } else {
                        handle.fail_code(
                            simvec_error_code_t::SIMVEC_ERROR,
                            "failed to hand out search results".to_string(),
                        )
                    }
                }
                Err(err) => handle.fail_code(
                    simvec_error_code_t::SIMVEC_INTERNAL,
                    format!("failed to serialize search results: {err}"),
                ),
            },
            Err(err) => handle.fail(&err),
        }
    })
}

/// Convenience search without building query JSON.
///
/// A null `filter_json` disables filtering; a negative `score_threshold`
/// disables the similarity cutoff. Non-zero `with_payload`/`with_vector`
/// echo the stored payload/vector back in each hit.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` must be a valid null-terminated UTF-8 string.
/// - `vector` must point to `vector_len` readable floats.
/// - `filter_json` must be null or a valid JSON `VectorFilter` string.
/// - `out` must be a valid out pointer freed with `simvec_free_string`.
#[no_mangle]
pub unsafe extern "C" fn simvec_search_vector(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    vector: *const f32,
    vector_len: usize,
    limit: u64,
    filter_json: *const c_char,
    score_threshold: f32,
    with_payload: c_int,
    with_vector: c_int,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let vector = match input_floats(vector, vector_len) {
            Ok(vector) => vector,
            Err(code) => return handle.fail_code(code, "invalid vector pointer".to_string()),
        };
        let raw_filter = match opt_input_str(filter_json) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid filter string".to_string()),
        };
        let filter: Option<VectorFilter> = match raw_filter {
            None => None,
            Some(text) => match parse_input_json(text) {
                Ok(filter) => Some(filter),
                Err(code) => return handle.fail_code(code, "invalid filter JSON".to_string()),
            },
        };
        let mut query = SearchQuery::new(vector, limit as usize);
        query.filter = filter;
        if score_threshold >= 0.0 {
            query.score_threshold = Some(score_threshold);
        }
        query.with_payload = Some(with_payload != 0);
        query.with_vector = Some(with_vector != 0);
        match handle.engine.search(collection, &query) {
            Ok(results) => match serde_json::to_string(&results) {
                Ok(body) => {
                    let emit = emit_json(out, &body);
                    if emit == simvec_error_code_t::SIMVEC_OK as c_int {
                        handle.ok()
                    } else {
                        handle.fail_code(
                            simvec_error_code_t::SIMVEC_ERROR,
                            "failed to hand out search results".to_string(),
                        )
                    }
                }
                Err(err) => handle.fail_code(
                    simvec_error_code_t::SIMVEC_INTERNAL,
                    format!("failed to serialize search results: {err}"),
                ),
            },
            Err(err) => handle.fail(&err),
        }
    })
}

/// Fetch one point as JSON; a missing id reports not-found.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` and `id` must be valid null-terminated UTF-8 strings.
/// - `out` must be a valid out pointer freed with `simvec_free_string`.
#[no_mangle]
pub unsafe extern "C" fn simvec_get(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    id: *const c_char,
    out: *mut *mut c_char,
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
        match handle.engine.get(collection, &PointId::from(id)) {
            Ok(Some(point)) => match serde_json::to_string(&point) {
                Ok(body) => {
                    let emit = emit_json(out, &body);
                    if emit == simvec_error_code_t::SIMVEC_OK as c_int {
                        handle.ok()
                    } else {
                        handle.fail_code(
                            simvec_error_code_t::SIMVEC_ERROR,
                            "failed to hand out point".to_string(),
                        )
                    }
                }
                Err(err) => handle.fail_code(
                    simvec_error_code_t::SIMVEC_INTERNAL,
                    format!("failed to serialize point: {err}"),
                ),
            },
            Ok(None) => handle.fail_code(
                simvec_error_code_t::SIMVEC_NOTFOUND,
                format!("point not found: {id}"),
            ),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Count live points; `*out` takes the count.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer.
#[no_mangle]
pub unsafe extern "C" fn simvec_count(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
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
        match handle.engine.count(collection) {
            Ok(count) => {
                *out = count;
                handle.ok()
            }
            Err(err) => handle.fail(&err),
        }
    })
}

/// Paginated scan as a JSON object with points and the next offset.
///
/// A null `offset` starts from the beginning. `next_offset` echoes the last
/// returned point id for use as the next page's cursor; a short page
/// (fewer points than requested) signals exhaustion.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` must be a valid null-terminated UTF-8 string.
/// - `offset` must be null or a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer freed with `simvec_free_string`.
#[no_mangle]
pub unsafe extern "C" fn simvec_scroll(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    limit: u64,
    offset: *const c_char,
    with_payload: c_int,
    with_vector: c_int,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let raw_offset = match opt_input_str(offset) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid offset string".to_string()),
        };
        let offset = raw_offset.map(PointId::from);
        match handle.engine.scroll(
            collection,
            limit as usize,
            offset.as_ref(),
            Some(with_payload != 0),
            Some(with_vector != 0),
        ) {
            Ok((points, next)) => {
                let body = serde_json::json!({
                    "points": points,
                    "next_offset": next.map(|id| id.to_string()),
                });
                match serde_json::to_string(&body) {
                    Ok(body) => {
                        let emit = emit_json(out, &body);
                        if emit == simvec_error_code_t::SIMVEC_OK as c_int {
                            handle.ok()
                        } else {
                            handle.fail_code(
                                simvec_error_code_t::SIMVEC_ERROR,
                                "failed to hand out scroll page".to_string(),
                            )
                        }
                    }
                    Err(err) => handle.fail_code(
                        simvec_error_code_t::SIMVEC_INTERNAL,
                        format!("failed to serialize scroll page: {err}"),
                    ),
                }
            }
            Err(err) => handle.fail(&err),
        }
    })
}
