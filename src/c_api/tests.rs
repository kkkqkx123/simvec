//! C API round-trip coverage: misuse guards and a full lifecycle.

use std::ffi::{c_char, c_int, CStr, CString};
use std::ptr;

use super::error::simvec_error_code_t;
use super::types::simvec_engine_t;
use super::{collections, engine, read, write};

const OK: c_int = simvec_error_code_t::SIMVEC_OK as c_int;
const MISUSE: c_int = simvec_error_code_t::SIMVEC_MISUSE as c_int;
const NOTFOUND: c_int = simvec_error_code_t::SIMVEC_NOTFOUND as c_int;

fn _assert_send_sync() {
    fn bound<T: Send + Sync>() {}
    bound::<crate::LocalVectorEngine>();
}

fn cstr(text: &str) -> CString {
    CString::new(text).expect("test string")
}

unsafe fn take_string(out: *mut c_char) -> String {
    assert!(!out.is_null());
    let owned = CStr::from_ptr(out).to_string_lossy().into_owned();
    engine::simvec_free_string(out);
    owned
}

unsafe fn open_engine() -> *mut simvec_engine_t {
    let dir = tempfile::tempdir().expect("tempdir");
    // The engine takes over the directory contents, so detach the tempdir
    // from cleanup for the life of the test process.
    let root = dir.path().join("engine");
    std::mem::forget(dir);
    let path = cstr(&root.to_string_lossy());
    let mut handle: *mut simvec_engine_t = ptr::null_mut();
    assert_eq!(engine::simvec_open(path.as_ptr(), &mut handle), OK);
    assert!(!handle.is_null());
    handle
}

unsafe fn close_engine(handle: *mut simvec_engine_t) {
    assert_eq!(engine::simvec_close(handle), OK);
}

#[test]
fn null_arguments_report_misuse() {
    unsafe {
        assert_eq!(
            engine::simvec_open(ptr::null(), ptr::null_mut()),
            MISUSE
        );
        assert_eq!(engine::simvec_close(ptr::null_mut()), MISUSE);
        engine::simvec_free_string(ptr::null_mut());
        assert_eq!(engine::simvec_errcode(ptr::null_mut()), MISUSE);
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            read::simvec_search(ptr::null_mut(), ptr::null(), ptr::null(), &mut out),
            MISUSE
        );
        assert!(out.is_null());
        assert_eq!(
            collections::simvec_list_collections(ptr::null_mut(), ptr::null_mut()),
            MISUSE
        );
        assert_eq!(
            write::simvec_upsert(
                ptr::null_mut(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                ptr::null()
            ),
            MISUSE
        );
    }
}

#[test]
fn invalid_json_reports_misuse() {
    unsafe {
        let handle = open_engine();
        let name = cstr("docs");
        let bad = cstr("{not json");
        assert_eq!(
            collections::simvec_create_collection(handle, name.as_ptr(), bad.as_ptr()),
            MISUSE
        );
        let bad_query = cstr("[1,2");
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            read::simvec_search(handle, name.as_ptr(), bad_query.as_ptr(), &mut out),
            MISUSE
        );
        assert!(out.is_null());
        close_engine(handle);
    }
}

#[test]
fn missing_objects_report_not_found() {
    unsafe {
        let handle = open_engine();
        let name = cstr("ghost");
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            collections::simvec_collection_info(handle, name.as_ptr(), &mut out),
            NOTFOUND
        );
        assert!(out.is_null());
        let id = cstr("1");
        assert_eq!(
            read::simvec_get(handle, name.as_ptr(), id.as_ptr(), &mut out),
            NOTFOUND
        );
        assert!(out.is_null());
        let mut count: u64 = 0;
        assert_eq!(
            read::simvec_count(handle, name.as_ptr(), &mut count),
            NOTFOUND
        );
        close_engine(handle);
    }
}

#[test]
fn lifecycle_round_trip() {
    unsafe {
        let handle = open_engine();
        let name = cstr("docs");
        let config = cstr(r#"{"vector_size":4,"distance":"Cosine","index":"Flat"}"#);
        assert_eq!(
            collections::simvec_create_collection(handle, name.as_ptr(), config.as_ptr()),
            OK
        );

        let id = cstr("1");
        let vector = [1.0f32, 0.0, 0.0, 0.0];
        let payload = cstr(r#"{"title":"hello"}"#);
        assert_eq!(
            write::simvec_upsert(
                handle,
                name.as_ptr(),
                id.as_ptr(),
                vector.as_ptr(),
                vector.len(),
                payload.as_ptr()
            ),
            OK
        );

        let mut count: u64 = 0;
        assert_eq!(read::simvec_count(handle, name.as_ptr(), &mut count), OK);
        assert_eq!(count, 1);

        let query = cstr(r#"{"vector":[1.0,0.0,0.0,0.0],"limit":5}"#);
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            read::simvec_search(handle, name.as_ptr(), query.as_ptr(), &mut out),
            OK
        );
        let hits: serde_json::Value =
            serde_json::from_str(&take_string(out)).expect("search results");
        assert_eq!(hits.as_array().map(Vec::len), Some(1));

        let mut point_out: *mut c_char = ptr::null_mut();
        assert_eq!(
            read::simvec_get(handle, name.as_ptr(), id.as_ptr(), &mut point_out),
            OK
        );
        let point: serde_json::Value =
            serde_json::from_str(&take_string(point_out)).expect("point");
        assert_eq!(
            point.get("id").map(|id| id.to_string()).as_deref(),
            Some("1")
        );

        assert_eq!(
            write::simvec_delete(handle, name.as_ptr(), id.as_ptr()),
            OK
        );
        assert_eq!(read::simvec_count(handle, name.as_ptr(), &mut count), OK);
        assert_eq!(count, 0);

        assert_eq!(
            collections::simvec_delete_collection(handle, name.as_ptr()),
            OK
        );
        close_engine(handle);
    }
}
