//! Point-in-time collection snapshots over FFI.
//!
//! Snapshot metadata crosses the boundary as JSON and is released with
//! `simvec_free_string`. Restore creates an independent collection directory;
//! callers open it with `simvec_open` afterwards.

use std::ffi::{c_char, c_int};
use std::path::Path;

use super::error::{emit_json, guard, input_str, simvec_error_code_t};
use super::types::simvec_engine_t;
use crate::storage::{restore_snapshot, verify_snapshot};

fn emit<T: serde::Serialize>(out: *mut *mut c_char, value: &T) -> c_int {
    match serde_json::to_string(value) {
        Ok(text) => unsafe { emit_json(out, &text) },
        Err(_) => simvec_error_code_t::SIMVEC_ERROR as c_int,
    }
}

/// Copy a collection's serving generation and WAL into `dest`.
#[no_mangle]
pub unsafe extern "C" fn simvec_snapshot_collection(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    dest: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(value) => value,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let dest = match input_str(dest) {
            Ok(value) => value,
            Err(code) => return handle.fail_code(code, "invalid snapshot path".to_string()),
        };
        match handle.engine.snapshot_collection(collection, dest) {
            Ok(info) => {
                let code = emit(out, &info);
                if code == simvec_error_code_t::SIMVEC_OK as c_int {
                    handle.ok()
                } else {
                    handle.fail_code(simvec_error_code_t::SIMVEC_ERROR, "serialize snapshot".into())
                }
            }
            Err(err) => handle.fail(&err),
        }
    })
}

/// Take a snapshot that hard-links unchanged files from a verified base.
#[no_mangle]
pub unsafe extern "C" fn simvec_snapshot_collection_incremental(
    engine: *mut simvec_engine_t,
    collection: *const c_char,
    base: *const c_char,
    dest: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let collection = match input_str(collection) {
            Ok(value) => value,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let base = match input_str(base) {
            Ok(value) => value,
            Err(code) => return handle.fail_code(code, "invalid base snapshot path".to_string()),
        };
        let dest = match input_str(dest) {
            Ok(value) => value,
            Err(code) => return handle.fail_code(code, "invalid snapshot path".to_string()),
        };
        match handle
            .engine
            .snapshot_collection_incremental(collection, base, dest)
        {
            Ok(info) => {
                let code = emit(out, &info);
                if code == simvec_error_code_t::SIMVEC_OK as c_int {
                    handle.ok()
                } else {
                    handle.fail_code(simvec_error_code_t::SIMVEC_ERROR, "serialize snapshot".into())
                }
            }
            Err(err) => handle.fail(&err),
        }
    })
}

/// Validate a snapshot manifest and every file it names.
#[no_mangle]
pub unsafe extern "C" fn simvec_verify_snapshot(
    engine: *mut simvec_engine_t,
    src: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let src = match input_str(src) {
            Ok(value) => value,
            Err(code) => return handle.fail_code(code, "invalid snapshot path".to_string()),
        };
        match verify_snapshot(Path::new(src)) {
            Ok(info) => {
                let code = emit(out, &info);
                if code == simvec_error_code_t::SIMVEC_OK as c_int {
                    handle.ok()
                } else {
                    handle.fail_code(simvec_error_code_t::SIMVEC_ERROR, "serialize snapshot".into())
                }
            }
            Err(err) => handle.fail(&err),
        }
    })
}

/// Copy a verified snapshot into a new directory and validate it by opening.
#[no_mangle]
pub unsafe extern "C" fn simvec_restore_snapshot(
    engine: *mut simvec_engine_t,
    src: *const c_char,
    dest: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let src = match input_str(src) {
            Ok(value) => value,
            Err(code) => return handle.fail_code(code, "invalid snapshot path".to_string()),
        };
        let dest = match input_str(dest) {
            Ok(value) => value,
            Err(code) => return handle.fail_code(code, "invalid restore path".to_string()),
        };
        match restore_snapshot(Path::new(src), Path::new(dest)) {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}
