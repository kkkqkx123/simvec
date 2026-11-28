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
fn filtered_count_scroll_round_trip() {
    unsafe {
        let handle = open_engine();
        let name = cstr("docs");
        let config = cstr(r#"{"vector_size":4,"distance":"Cosine","index":"Flat"}"#);
        assert_eq!(
            collections::simvec_create_collection(handle, name.as_ptr(), config.as_ptr()),
            OK
        );
        for (id, version) in [("1", "v1"), ("2", "v1"), ("3", "v2")] {
            let id_c = cstr(id);
            let vector = [1.0f32, 0.0, 0.0, 0.0];
            let payload = cstr(&format!(r#"{{"version":"{version}"}}"#));
            assert_eq!(
                write::simvec_upsert(
                    handle,
                    name.as_ptr(),
                    id_c.as_ptr(),
                    vector.as_ptr(),
                    vector.len(),
                    payload.as_ptr()
                ),
                OK
            );
        }

        let filter = cstr(r#"{"must":[{"field":"version","condition":{"Match":{"value":"v1"}}}]}"#);
        let mut count: u64 = 0;
        assert_eq!(
            read::simvec_count_by_filter(handle, name.as_ptr(), filter.as_ptr(), &mut count),
            OK
        );
        assert_eq!(count, 2);

        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            read::simvec_scroll_by_filter(
                handle,
                name.as_ptr(),
                1,
                ptr::null(),
                filter.as_ptr(),
                1,
                0,
                &mut out
            ),
            OK
        );
        let first: serde_json::Value = serde_json::from_str(&take_string(out)).expect("first page");
        assert_eq!(
            first.get("points").map(|p| p.as_array().map(Vec::len)),
            Some(Some(1))
        );
        let cursor = first
            .get("next_offset")
            .and_then(|v| v.as_str())
            .expect("cursor");
        let cursor_c = cstr(cursor);
        let mut tail_out: *mut c_char = ptr::null_mut();
        assert_eq!(
            read::simvec_scroll_by_filter(
                handle,
                name.as_ptr(),
                10,
                cursor_c.as_ptr(),
                filter.as_ptr(),
                1,
                0,
                &mut tail_out
            ),
            OK
        );
        let tail: serde_json::Value =
            serde_json::from_str(&take_string(tail_out)).expect("tail page");
        assert_eq!(
            tail.get("points").map(|p| p.as_array().map(Vec::len)),
            Some(Some(1))
        );

        // Misuse: null out, null filter, malformed JSON.
        let mut bad_count: u64 = 0;
        assert_eq!(
            read::simvec_count_by_filter(handle, name.as_ptr(), ptr::null(), &mut bad_count),
            MISUSE
        );
        assert_eq!(
            read::simvec_count_by_filter(handle, name.as_ptr(), filter.as_ptr(), ptr::null_mut()),
            MISUSE
        );
        let bad = cstr("{not json");
        assert_eq!(
            read::simvec_count_by_filter(handle, name.as_ptr(), bad.as_ptr(), &mut bad_count),
            MISUSE
        );
        let mut bad_out: *mut c_char = ptr::null_mut();
        assert_eq!(
            read::simvec_scroll_by_filter(
                handle,
                name.as_ptr(),
                10,
                ptr::null(),
                bad.as_ptr(),
                1,
                0,
                &mut bad_out
            ),
            MISUSE
        );
        assert!(bad_out.is_null());

        // Not-found: unknown collection reports NOTFOUND for both entries.
        let ghost = cstr("ghost");
        assert_eq!(
            read::simvec_count_by_filter(handle, ghost.as_ptr(), filter.as_ptr(), &mut bad_count),
            NOTFOUND
        );
        let mut ghost_out: *mut c_char = ptr::null_mut();
        assert_eq!(
            read::simvec_scroll_by_filter(
                handle,
                ghost.as_ptr(),
                10,
                ptr::null(),
                filter.as_ptr(),
                1,
                0,
                &mut ghost_out
            ),
            NOTFOUND
        );
        assert!(ghost_out.is_null());

        assert_eq!(
            collections::simvec_delete_collection(handle, name.as_ptr()),
            OK
        );
        close_engine(handle);
    }
}

/// `simvec_maintenance_status` reports the per-segment durability array, not
/// just the health boolean: a binding that only forwards JSON would otherwise
/// silently drop the segment water marks.
#[test]
fn maintenance_status_reports_segment_shape() {
    unsafe {
        let handle = open_engine();
        let name = cstr("segments");
        let config = cstr(r#"{"vector_size":4,"distance":"Cosine","index":"Flat"}"#);
        assert_eq!(
            collections::simvec_create_collection(handle, name.as_ptr(), config.as_ptr()),
            OK
        );
        // Eight points at the default segment width fill a single partial
        // segment. The assertion below checks the array's shape (numeric
        // fields, increasing starts) rather than its length, so a future
        // width change that splits it does not false-fail here.
        let ops: String = (1..=8u64)
            .map(|id| {
                format!(
                    r#"{{"op":"upsert","collection":"segments","point":{{"id":"{id}","vector":[1.0,0.0,0.0,0.0]}}}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(
            write::simvec_apply_txn(handle, cstr(&format!("[{ops}]")).as_ptr()),
            OK
        );

        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            maintenance::simvec_maintenance_status(handle, name.as_ptr(), &mut out),
            OK
        );
        let status: serde_json::Value =
            serde_json::from_str(&take_string(out)).expect("maintenance status json");

        assert_eq!(status.get("healthy").and_then(|h| h.as_bool()), Some(true));
        assert!(
            status.get("persisted_lsn").and_then(|v| v.as_u64()).is_some(),
            "persisted_lsn must be a number: {status}"
        );
        assert!(
            status
                .get("wal_blob_bytes")
                .and_then(|v| v.as_u64())
                .is_some(),
            "wal_blob_bytes must be a number: {status}"
        );
        let segments = status
            .get("segments")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("segments must be an array: {status}"));
        assert_eq!(segments.len(), 1, "one partial segment is reported");
        let mut last_start: Option<u64> = None;
        for segment in segments {
            for field in ["start", "applied_lsn", "persisted_lsn"] {
                assert!(
                    segment.get(field).and_then(|v| v.as_u64()).is_some(),
                    "segment entry needs a numeric {field}: {segment}"
                );
            }
            let start = segment["start"].as_u64().unwrap();
            if let Some(previous) = last_start {
                assert!(start > previous, "segment starts must increase");
            }
            last_start = Some(start);
        }
        assert!(
            status.get("error").map(|v| v.is_null()).unwrap_or(true),
            "a healthy collection carries no error: {status}"
        );
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

#[test]
fn named_patch_round_trip() {
    unsafe {
        let handle = open_engine();
        let name = cstr("named");
        let config = cstr(
            r#"{"vector_size":2,"distance":"Cosine","kind":"Dense","vectors":{"text":{"kind":"Dense","vector_size":2,"distance":"Cosine"},"tags":{"kind":"Sparse","vector_size":4294967295,"distance":"Dot"}},"index":"Flat"}"#,
        );
        assert_eq!(
            collections::simvec_create_collection(handle, name.as_ptr(), config.as_ptr()),
            OK
        );
        let points = cstr(
            r#"[{"id":"p1","vectors":{"text":{"Dense":[1.0,0.0]},"tags":{"Sparse":{"indices":[1],"values":[1.0]}}}}]"#,
        );
        assert_eq!(
            write::simvec_upsert_batch(handle, name.as_ptr(), points.as_ptr()),
            OK
        );
        let id = cstr("p1");
        let patch = cstr(r#"{"tags":{"Sparse":{"indices":[2],"values":[2.0]}}}"#);
        assert_eq!(
            write::simvec_update_named_vectors(handle, name.as_ptr(), id.as_ptr(), patch.as_ptr()),
            OK
        );
        let query = cstr(
            r#"{"vectors":[{"space":"tags","data":{"Sparse":{"indices":[2],"values":[1.0]}}}],"limit":5,"with_vector":true}"#,
        );
        let mut out: *mut c_char = std::ptr::null_mut();
        assert_eq!(
            read::simvec_search(handle, name.as_ptr(), query.as_ptr(), &mut out),
            OK
        );
        let hits: serde_json::Value = serde_json::from_str(&take_string(out)).expect("named hits");
        let first = hits
            .as_array()
            .and_then(|items| items.first())
            .expect("one hit");
        let spaces = first
            .get("vectors")
            .and_then(|v| v.as_object())
            .expect("vector map");
        assert!(spaces.contains_key("text"));
        assert!(spaces.contains_key("tags"));
        let bad = cstr(r#"{"ghost":{"Dense":[1.0,0.0]}}"#);
        assert_ne!(
            write::simvec_update_named_vectors(handle, name.as_ptr(), id.as_ptr(), bad.as_ptr()),
            OK
        );
        assert_eq!(
            collections::simvec_delete_collection(handle, name.as_ptr()),
            OK
        );
        close_engine(handle);
    }
}

#[test]
fn snapshot_c_abi_roundtrip() {
    unsafe {
        let handle = open_engine();
        let name = cstr("snap_col");
        let config = cstr(r#"{"vector_size":2,"distance":"Cosine","index":"Flat"}"#);
        assert_eq!(
            collections::simvec_create_collection(handle, name.as_ptr(), config.as_ptr()),
            OK
        );
        let vector = [1.0f32, 0.0];
        assert_eq!(
            write::simvec_upsert(
                handle,
                name.as_ptr(),
                cstr("1").as_ptr(),
                vector.as_ptr(),
                vector.len(),
                ptr::null(),
            ),
            OK
        );
        let root = std::env::temp_dir().join(format!(
            "simvec-c-snapshot-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let snapshot = cstr(&root.join("snapshot").to_string_lossy());
        let restored_root = root.join("restored");
        let restored = cstr(&restored_root.join("snap_col").to_string_lossy());
        let mut out: *mut c_char = ptr::null_mut();
        assert_eq!(
            super::snapshots::simvec_snapshot_collection(
                handle,
                name.as_ptr(),
                snapshot.as_ptr(),
                &mut out,
            ),
            OK
        );
        let info: serde_json::Value =
            serde_json::from_str(&take_string(out)).expect("snapshot info JSON");
        assert!(info.get("generation").is_some());
        assert_eq!(
            super::snapshots::simvec_verify_snapshot(handle, snapshot.as_ptr(), &mut out),
            OK
        );
        engine::simvec_free_string(out);
        assert_eq!(
            super::snapshots::simvec_restore_snapshot(handle, snapshot.as_ptr(), restored.as_ptr()),
            OK
        );
        let mut restored_engine: *mut simvec_engine_t = ptr::null_mut();
        assert_eq!(
            engine::simvec_open(
                cstr(&restored_root.to_string_lossy()).as_ptr(),
                &mut restored_engine
            ),
            OK
        );
        let mut count = 0u64;
        assert_eq!(
            read::simvec_count(restored_engine, name.as_ptr(), &mut count),
            OK
        );
        assert_eq!(count, 1);
        close_engine(restored_engine);
        close_engine(handle);
        let _ = std::fs::remove_dir_all(&root);
    }
}
