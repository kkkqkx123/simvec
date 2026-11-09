//! Engine lifecycle: open, close, memory release and error readout.

use std::ffi::{c_char, c_int};

use crate::LocalVectorEngine;

use super::error::{
    emit_json, error_code_from_error, guard, input_str, set_thread_error, simvec_error_code_t,
    EngineHandle,
};
use super::types::simvec_engine_t;

/// Open (or create) the engine rooted at `path`.
///
/// # Safety
/// - `path` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer; on success it receives the engine
///   handle, on failure it is set to null. Open-time diagnostics are kept in
///   thread-local storage readable with `simvec_last_error_text`.
#[no_mangle]
pub unsafe extern "C" fn simvec_open(
    path: *const c_char,
    out: *mut *mut simvec_engine_t,
) -> c_int {
    if out.is_null() {
        return simvec_error_code_t::SIMVEC_MISUSE as c_int;
    }
    *out = std::ptr::null_mut();
    let path = match input_str(path) {
        Ok(path) => path,
        Err(code) => {
            set_thread_error("engine path must be a non-null UTF-8 string".to_string());
            return code as c_int;
        }
    };
    match LocalVectorEngine::open(path) {
        Ok(engine) => {
            let handle = Box::new(EngineHandle::new(engine));
            *out = Box::into_raw(handle) as *mut simvec_engine_t;
            simvec_error_code_t::SIMVEC_OK as c_int
        }
        Err(err) => {
            let code = error_code_from_error(&err);
            set_thread_error(err.to_string());
            code as c_int
        }
    }
}

/// Close an engine and join its maintenance worker.
///
/// # Safety
/// - `engine` must be a handle from `simvec_open` not previously closed.
#[no_mangle]
pub unsafe extern "C" fn simvec_close(engine: *mut simvec_engine_t) -> c_int {
    if engine.is_null() {
        return simvec_error_code_t::SIMVEC_MISUSE as c_int;
    }
    drop(Box::from_raw(engine as *mut EngineHandle));
    simvec_error_code_t::SIMVEC_OK as c_int
}

/// Release a string allocated by the C API.
///
/// # Safety
/// - `s` must be null or a pointer previously returned by this API.
#[no_mangle]
pub unsafe extern "C" fn simvec_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(std::ffi::CString::from_raw(s));
    }
}

/// Most recent status code recorded on the handle.
///
/// Returns misuse for a null handle without touching any error state.
#[no_mangle]
pub unsafe extern "C" fn simvec_errcode(engine: *mut simvec_engine_t) -> c_int {
    if engine.is_null() {
        return simvec_error_code_t::SIMVEC_MISUSE as c_int;
    }
    (&*(engine as *mut EngineHandle)).last_code()
}

/// Most recent error message as a newly allocated string.
///
/// The caller releases the result with `simvec_free_string`. An empty
/// string means the last call succeeded.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `out` must be a valid out pointer.
#[no_mangle]
pub unsafe extern "C" fn simvec_last_error(
    engine: *mut simvec_engine_t,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let body = handle
            .last_message()
            .map(|owned| owned.into_string().unwrap_or_default())
            .unwrap_or_default();
        let code = handle.last_code();
        let emit = emit_json(out, &body);
        if emit != simvec_error_code_t::SIMVEC_OK as c_int {
            return emit;
        }
        let _ = code;
        simvec_error_code_t::SIMVEC_OK as c_int
    })
}
