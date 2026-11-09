//! C API round-trip coverage: misuse guards and a full lifecycle.

use std::ffi::{c_char, c_int, CStr, CString};
use std::ptr;

use super::error::simvec_error_code_t;
use super::types::simvec_engine_t;
use super::{collections, engine, maintenance, read, write};

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
        assert_eq!(engine::simvec_open(ptr::null(), ptr::null_mut()), MISUSE);
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

        assert_eq!(write::simvec_delete(handle, name.as_ptr(), id.as_ptr()), OK);
        assert_eq!(read::simvec_count(handle, name.as_ptr(), &mut count), OK);
        assert_eq!(count, 0);

        assert_eq!(
            collections::simvec_delete_collection(handle, name.as_ptr()),
            OK
        );
        close_engine(handle);
    }
}

#[test]
fn advanced_paths_round_trip() {
    unsafe {
        let handle = open_engine();
        let name = cstr("docs");
        let config = cstr(r#"{"vector_size":4,"distance":"Cosine","index":"Flat"}"#);
        assert_eq!(
            collections::simvec_create_collection(handle, name.as_ptr(), config.as_ptr()),
            OK
        );

        let ops = cstr(
            r#"[{"op":"upsert","collection":"docs","point":{"id":"1","vector":[1.0,0.0,0.0,0.0]}},{"op":"delete","collection":"docs","id":"1"}]"#,
        );
        assert_eq!(write::simvec_apply_txn(handle, ops.as_ptr()), OK);
        let id = cstr("1");
        let vector = [1.0f32, 0.0, 0.0, 0.0];
        assert_eq!(
            write::simvec_upsert(
                handle,
                name.as_ptr(),
                id.as_ptr(),
                vector.as_ptr(),
                vector.len(),
                ptr::null()
            ),
            OK
        );

        let key = cstr("lang");
        let value = cstr(r#""en""#);
        assert_eq!(
            write::simvec_set_payload_field(
                handle,
                name.as_ptr(),
                id.as_ptr(),
                key.as_ptr(),
                value.as_ptr()
            ),
            OK
        );

        let mut flag: std::ffi::c_int = 0;
        assert_eq!(
            maintenance::simvec_has_index(handle, name.as_ptr(), &mut flag),
            OK
        );
        assert_eq!(
            maintenance::simvec_build_index(handle, name.as_ptr(), &mut flag),
            OK
        );
        assert_eq!(
            maintenance::simvec_build_quantization(handle, name.as_ptr(), &mut flag),
            OK
        );
        assert_eq!(maintenance::simvec_drain_pending(handle, name.as_ptr()), OK);
        assert_eq!(
            maintenance::simvec_drop_published_index(handle, name.as_ptr()),
            OK
        );

        let hnsw = cstr(r#"{"m":8,"ef_construct":32}"#);
        assert_eq!(
            engine::simvec_set_default_hnsw_config(handle, hnsw.as_ptr()),
            OK
        );
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(engine::simvec_get_default_hnsw_config(handle, &mut out), OK);
        let body: serde_json::Value =
            serde_json::from_str(&take_string(out)).expect("default hnsw");
        assert_eq!(body.get("m").and_then(|m| m.as_u64()), Some(8));

        let temp = cstr("docs.rebuild-1");
        assert_eq!(
            collections::simvec_create_collection(handle, temp.as_ptr(), config.as_ptr()),
            OK
        );
        let mut list_out: *mut c_char = ptr::null_mut();
        assert_eq!(
            collections::simvec_list_temp_collections(handle, &mut list_out),
            OK
        );
        let temps: serde_json::Value =
            serde_json::from_str(&take_string(list_out)).expect("temp list");
        assert!(temps.as_array().map(|items| items.len()) == Some(1));
        let live = name;
        let suffix = cstr(".old-1");
        assert_eq!(
            collections::simvec_promote_temp_collection(
                handle,
                live.as_ptr(),
                temp.as_ptr(),
                suffix.as_ptr()
            ),
            OK
        );
        let mut backups_out: *mut c_char = ptr::null_mut();
        assert_eq!(
            collections::simvec_list_old_backups(handle, live.as_ptr(), &mut backups_out),
            OK
        );
        let backups: serde_json::Value =
            serde_json::from_str(&take_string(backups_out)).expect("backup list");
        assert_eq!(backups.as_array().map(Vec::len), Some(1));
        let mut restored_out: *mut c_char = ptr::null_mut();
        assert_eq!(
            collections::simvec_restore_promote_backup(handle, live.as_ptr(), &mut restored_out),
            OK
        );
        let restored: String =
            serde_json::from_str(&take_string(restored_out)).expect("backup name");
        assert_eq!(restored, "docs.old-1");

        assert_eq!(
            collections::simvec_delete_collection(handle, live.as_ptr()),
            OK
        );
        close_engine(handle);
    }
}
