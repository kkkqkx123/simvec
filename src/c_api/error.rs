//! C API error codes, engine handle state and input helpers.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use parking_lot::Mutex;

use crate::{LocalVectorEngine, VectorSearchError};

use super::types::simvec_engine_t;

/// Stable C status codes for every C API entry point.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum simvec_error_code_t {
    SIMVEC_OK = 0,
    SIMVEC_ERROR = 1,
    SIMVEC_INTERNAL = 2,
    SIMVEC_NOMEM = 7,
    SIMVEC_IOERR = 10,
    SIMVEC_CORRUPT = 11,
    SIMVEC_NOTFOUND = 12,
    SIMVEC_ALREADY_EXISTS = 13,
    SIMVEC_INVALID_CONFIG = 16,
    SIMVEC_INVALID_DIM = 19,
    SIMVEC_MISUSE = 20,
    SIMVEC_NON_FINITE = 21,
    SIMVEC_CAPACITY_EXCEEDED = 22,
    SIMVEC_FILTER = 23,
}

/// Map an engine error onto its C status code.
pub(crate) fn error_code_from_error(err: &VectorSearchError) -> simvec_error_code_t {
    use simvec_error_code_t as Code;
    match err {
        VectorSearchError::CollectionNotFound(_) => Code::SIMVEC_NOTFOUND,
        VectorSearchError::CollectionAlreadyExists(_) => Code::SIMVEC_ALREADY_EXISTS,
        VectorSearchError::CollectionIncomplete { .. }
        | VectorSearchError::CorruptData(_)
        | VectorSearchError::Serialization(_) => Code::SIMVEC_CORRUPT,
        VectorSearchError::InvalidCollectionName(_) | VectorSearchError::InvalidPointId(_) => {
            Code::SIMVEC_MISUSE
        }
        VectorSearchError::InvalidConfig(_) | VectorSearchError::UnsupportedMetric(_) => {
            Code::SIMVEC_INVALID_CONFIG
        }
        VectorSearchError::InsufficientSpace { .. } => Code::SIMVEC_NOMEM,
        VectorSearchError::CapacityExceeded { .. } => Code::SIMVEC_CAPACITY_EXCEEDED,
        VectorSearchError::InvalidVectorDimension { .. } => Code::SIMVEC_INVALID_DIM,
        VectorSearchError::NonFiniteElement(_) => Code::SIMVEC_NON_FINITE,
        VectorSearchError::Filter(_) => Code::SIMVEC_FILTER,
        VectorSearchError::Io(_) => Code::SIMVEC_IOERR,
        VectorSearchError::Json(_) => Code::SIMVEC_MISUSE,
        VectorSearchError::Internal(_) => Code::SIMVEC_INTERNAL,
    }
}

/// Heap handle behind the opaque C engine pointer.
pub(crate) struct EngineHandle {
    pub(crate) engine: LocalVectorEngine,
    code: Mutex<c_int>,
    message: Mutex<Option<CString>>,
}

impl EngineHandle {
    pub(crate) fn new(engine: LocalVectorEngine) -> Self {
        Self {
            engine,
            code: Mutex::new(simvec_error_code_t::SIMVEC_OK as c_int),
            message: Mutex::new(None),
        }
    }

    pub(crate) fn ok(&self) -> c_int {
        *self.code.lock() = simvec_error_code_t::SIMVEC_OK as c_int;
        *self.message.lock() = None;
        simvec_error_code_t::SIMVEC_OK as c_int
    }

    pub(crate) fn fail(&self, err: &VectorSearchError) -> c_int {
        self.fail_code(error_code_from_error(err), err.to_string())
    }

    pub(crate) fn fail_code(&self, code: simvec_error_code_t, msg: String) -> c_int {
        let raw = code as c_int;
        *self.code.lock() = raw;
        *self.message.lock() = CString::new(msg).ok();
        raw
    }

    pub(crate) fn last_code(&self) -> c_int {
        *self.code.lock()
    }

    pub(crate) fn last_message(&self) -> Option<CString> {
        self.message.lock().clone()
    }
}

thread_local! {
    static THREAD_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Remember an error that has no engine handle yet, such as open failures.
pub(crate) fn set_thread_error(msg: String) {
    THREAD_ERROR.with(|slot| *slot.borrow_mut() = CString::new(msg).ok());
}

/// Borrow a required null-terminated UTF-8 input string.
pub(crate) unsafe fn input_str<'a>(ptr: *const c_char) -> Result<&'a str, simvec_error_code_t> {
    if ptr.is_null() {
        return Err(simvec_error_code_t::SIMVEC_MISUSE);
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|_| simvec_error_code_t::SIMVEC_MISUSE)
}

/// Borrow an optional null-terminated UTF-8 input string.
pub(crate) unsafe fn opt_input_str<'a>(
    ptr: *const c_char,
) -> Result<Option<&'a str>, simvec_error_code_t> {
    if ptr.is_null() {
        return Ok(None);
    }
    input_str(ptr).map(Some)
}

/// Copy a float slice provided by the caller into an owned vector.
pub(crate) unsafe fn input_floats(
    ptr: *const f32,
    len: usize,
) -> Result<Vec<f32>, simvec_error_code_t> {
    if ptr.is_null() {
        return Err(simvec_error_code_t::SIMVEC_MISUSE);
    }
    Ok(std::slice::from_raw_parts(ptr, len).to_vec())
}

/// Parse caller-supplied JSON input; malformed input is caller misuse.
pub(crate) fn parse_input_json<T: serde::de::DeserializeOwned>(
    text: &str,
) -> Result<T, simvec_error_code_t> {
    serde_json::from_str(text).map_err(|_| simvec_error_code_t::SIMVEC_MISUSE)
}

/// Hand a JSON document to the caller through an out pointer.
pub(crate) unsafe fn emit_json(out: *mut *mut c_char, body: &str) -> c_int {
    if out.is_null() {
        return simvec_error_code_t::SIMVEC_MISUSE as c_int;
    }
    match CString::new(body) {
        Ok(owned) => {
            *out = owned.into_raw();
            simvec_error_code_t::SIMVEC_OK as c_int
        }
        Err(_) => {
            *out = std::ptr::null_mut();
            simvec_error_code_t::SIMVEC_ERROR as c_int
        }
    }
}

/// Null-check the handle, run the body and convert panics into errors.
pub(crate) unsafe fn guard(
    engine: *mut simvec_engine_t,
    run: impl FnOnce(&EngineHandle) -> c_int,
) -> c_int {
    if engine.is_null() {
        return simvec_error_code_t::SIMVEC_MISUSE as c_int;
    }
    let handle = &*(engine as *mut EngineHandle);
    match catch_unwind(AssertUnwindSafe(|| run(handle))) {
        Ok(code) => code,
        Err(_) => handle.fail_code(
            simvec_error_code_t::SIMVEC_INTERNAL,
            "panic inside simvec".to_string(),
        ),
    }
}

/// Most recent pre-handle error as a newly allocated string.
///
/// The caller releases the result with `simvec_free_string`. An empty
/// string means no recorded error.
#[no_mangle]
pub unsafe extern "C" fn simvec_last_error_text() -> *mut c_char {
    let owned: CString = THREAD_ERROR.with(|slot| slot.borrow().clone().unwrap_or_default());
    owned.into_raw()
}
