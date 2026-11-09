//! Index maintenance and metrics over FFI.

use std::ffi::{c_char, c_int};

use super::error::{emit_json, guard, input_str, simvec_error_code_t};
use super::types::simvec_engine_t;

/// Build and publish the index; `*out` receives 1 when one is published.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer.
#[no_mangle]
pub unsafe extern "C" fn simvec_build_index(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
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
        match handle.engine.build_index(collection) {
            Ok(published) => {
                *out = i32::from(published);
                handle.ok()
            }
            Err(err) => handle.fail(&err),
        }
    })
}

/// Build quantized storage; `*out` receives 1 when quantization is ready.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer.
#[no_mangle]
pub unsafe extern "C" fn simvec_build_quantization(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
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
        match handle.engine.build_quantization(collection) {
            Ok(ready) => {
                *out = i32::from(ready);
                handle.ok()
            }
            Err(err) => handle.fail(&err),
        }
    })
}

/// Whether an index is published; `*out` receives 1 or 0.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer.
#[no_mangle]
pub unsafe extern "C" fn simvec_has_index(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
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
        *out = i32::from(handle.engine.has_index(collection));
        handle.ok()
    })
}

/// Whether quantization is ready; `*out` receives 1 or 0.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer.
#[no_mangle]
pub unsafe extern "C" fn simvec_has_quantization(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
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
        *out = i32::from(handle.engine.has_quantization(collection));
        handle.ok()
    })
}

/// Drop the published index structures; the data stays searchable by scan.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn simvec_drop_index(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        match handle.engine.drop_index(collection) {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Reclaim tombstoned slots; `*out` takes the live point count.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer.
#[no_mangle]
pub unsafe extern "C" fn simvec_compact(
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
        match handle.engine.compact_collection(collection) {
            Ok(live) => {
                *out = live;
                handle.ok()
            }
            Err(err) => handle.fail(&err),
        }
    })
}

/// Drain pending HNSW slots into the published graph immediately.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn simvec_drain_pending(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        match handle.engine.drain_pending(collection) {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Operational metrics snapshot as JSON.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `collection` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer freed with `simvec_free_string`.
#[no_mangle]
pub unsafe extern "C" fn simvec_collection_metrics(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(collection) => collection,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        match handle.engine.collection_metrics(collection) {
            Ok(snapshot) => match serde_json::to_string(&snapshot) {
                Ok(body) => {
                    let emit = emit_json(out, &body);
                    if emit == simvec_error_code_t::SIMVEC_OK as c_int {
                        handle.ok()
                    } else {
                        handle.fail_code(
                            simvec_error_code_t::SIMVEC_ERROR,
                            "failed to hand out metrics".to_string(),
                        )
                    }
                }
                Err(err) => handle.fail_code(
                    simvec_error_code_t::SIMVEC_INTERNAL,
                    format!("failed to serialize metrics: {err}"),
                ),
            },
            Err(err) => handle.fail(&err),
        }
    })
}

/// Run one maintenance sweep immediately.
///
/// # Safety
/// - `engine` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn simvec_run_maintenance_sweep(engine: *mut simvec_engine_t) -> c_int {
    guard(engine, |handle| {
        handle.engine.run_maintenance_sweep();
        handle.ok()
    })
}
