//! Collection management over FFI.

use std::ffi::{c_char, c_int};

use crate::CollectionConfig;

use super::error::{
    emit_json, guard, input_str, opt_input_str, parse_input_json, simvec_error_code_t,
};
use super::types::simvec_engine_t;

/// Create a collection from a JSON `CollectionConfig`.
///
/// A null `config_json` selects the library default configuration.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `name` must be a valid null-terminated UTF-8 string.
/// - `config_json` must be null or a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn simvec_create_collection(
    engine: *mut simvec_engine_t,
    name: *const c_char,
    config_json: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let name = match input_str(name) {
            Ok(name) => name,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        let raw = match opt_input_str(config_json) {
            Ok(raw) => raw,
            Err(code) => return handle.fail_code(code, "invalid config string".to_string()),
        };
        let config: CollectionConfig = match raw {
            None => CollectionConfig::default(),
            Some(text) => match parse_input_json(text) {
                Ok(config) => config,
                Err(code) => return handle.fail_code(code, "invalid config JSON".to_string()),
            },
        };
        match handle.engine.create_collection(name, &config) {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Delete a collection and its persisted directory.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `name` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn simvec_delete_collection(
    engine: *mut simvec_engine_t,
    name: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let name = match input_str(name) {
            Ok(name) => name,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        match handle.engine.delete_collection(name) {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Whether a collection exists: `*out` receives 1 or 0.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `name` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer.
#[no_mangle]
pub unsafe extern "C" fn simvec_collection_exists(
    engine: *mut simvec_engine_t,
    name: *const c_char,
    out: *mut c_int,
) -> c_int {
    guard(engine, |handle| {
        if out.is_null() {
            return handle.fail_code(
                simvec_error_code_t::SIMVEC_MISUSE,
                "null out pointer".to_string(),
            );
        }
        let name = match input_str(name) {
            Ok(name) => name,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        *out = i32::from(handle.engine.collection_exists(name));
        handle.ok()
    })
}

/// List loaded collection names as a JSON array of strings.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `out` must be a valid out pointer freed with `simvec_free_string`.
#[no_mangle]
pub unsafe extern "C" fn simvec_list_collections(
    engine: *mut simvec_engine_t,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let names = handle.engine.collection_names();
        match serde_json::to_string(&names) {
            Ok(body) => {
                let emit = emit_json(out, &body);
                if emit == simvec_error_code_t::SIMVEC_OK as c_int {
                    handle.ok()
                } else {
                    handle.fail_code(
                        simvec_error_code_t::SIMVEC_ERROR,
                        "failed to hand out collection list".to_string(),
                    )
                }
            }
            Err(err) => handle.fail_code(
                simvec_error_code_t::SIMVEC_INTERNAL,
                format!("failed to serialize collection list: {err}"),
            ),
        }
    })
}

/// Describe a collection as a JSON `CollectionInfo`.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `name` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer freed with `simvec_free_string`.
#[no_mangle]
pub unsafe extern "C" fn simvec_collection_info(
    engine: *mut simvec_engine_t,
    name: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let name = match input_str(name) {
            Ok(name) => name,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        match handle.engine.collection_info(name) {
            Ok(info) => match serde_json::to_string(&info) {
                Ok(body) => {
                    let emit = emit_json(out, &body);
                    if emit == simvec_error_code_t::SIMVEC_OK as c_int {
                        handle.ok()
                    } else {
                        handle.fail_code(
                            simvec_error_code_t::SIMVEC_ERROR,
                            "failed to hand out collection info".to_string(),
                        )
                    }
                }
                Err(err) => handle.fail_code(
                    simvec_error_code_t::SIMVEC_INTERNAL,
                    format!("failed to serialize collection info: {err}"),
                ),
            },
            Err(err) => handle.fail(&err),
        }
    })
}

/// Read back a collection configuration as JSON.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `name` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer freed with `simvec_free_string`.
#[no_mangle]
pub unsafe extern "C" fn simvec_collection_config(
    engine: *mut simvec_engine_t,
    name: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let name = match input_str(name) {
            Ok(name) => name,
            Err(code) => return handle.fail_code(code, "invalid collection name".to_string()),
        };
        match handle.engine.collection_config(name) {
            Ok(Some(config)) => match serde_json::to_string(&config) {
                Ok(body) => {
                    let emit = emit_json(out, &body);
                    if emit == simvec_error_code_t::SIMVEC_OK as c_int {
                        handle.ok()
                    } else {
                        handle.fail_code(
                            simvec_error_code_t::SIMVEC_ERROR,
                            "failed to hand out collection config".to_string(),
                        )
                    }
                }
                Err(err) => handle.fail_code(
                    simvec_error_code_t::SIMVEC_INTERNAL,
                    format!("failed to serialize collection config: {err}"),
                ),
            },
            Ok(None) => handle.fail_code(
                simvec_error_code_t::SIMVEC_NOTFOUND,
                format!("collection not found: {name}"),
            ),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Promote a temp collection over a live name, keeping a backup suffix copy.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `live`, `temp` and `backup_suffix` must be valid null-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn simvec_promote_temp_collection(
    engine: *mut simvec_engine_t,
    live: *const c_char,
    temp: *const c_char,
    backup_suffix: *const c_char,
) -> c_int {
    guard(engine, |handle| {
        let live = match input_str(live) {
            Ok(live) => live,
            Err(code) => return handle.fail_code(code, "invalid live name".to_string()),
        };
        let temp = match input_str(temp) {
            Ok(temp) => temp,
            Err(code) => return handle.fail_code(code, "invalid temp name".to_string()),
        };
        let backup_suffix = match input_str(backup_suffix) {
            Ok(suffix) => suffix,
            Err(code) => return handle.fail_code(code, "invalid backup suffix".to_string()),
        };
        match handle
            .engine
            .promote_temp_collection(live, temp, backup_suffix)
        {
            Ok(()) => handle.ok(),
            Err(err) => handle.fail(&err),
        }
    })
}

/// Restore the newest promote backup over a live name; `*out` takes the backup name.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `live` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer freed with `simvec_free_string`.
#[no_mangle]
pub unsafe extern "C" fn simvec_restore_promote_backup(
    engine: *mut simvec_engine_t,
    live: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let live = match input_str(live) {
            Ok(live) => live,
            Err(code) => return handle.fail_code(code, "invalid live name".to_string()),
        };
        match handle.engine.restore_promote_backup(live) {
            Ok(backup) => match serde_json::to_string(&backup) {
                Ok(body) => {
                    let emit = emit_json(out, &body);
                    if emit == simvec_error_code_t::SIMVEC_OK as c_int {
                        handle.ok()
                    } else {
                        handle.fail_code(
                            simvec_error_code_t::SIMVEC_ERROR,
                            "failed to hand out backup name".to_string(),
                        )
                    }
                }
                Err(err) => handle.fail_code(
                    simvec_error_code_t::SIMVEC_INTERNAL,
                    format!("failed to serialize backup name: {err}"),
                ),
            },
            Err(err) => handle.fail(&err),
        }
    })
}

/// List temp rebuild collections as a JSON array of strings.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `out` must be a valid out pointer freed with `simvec_free_string`.
#[no_mangle]
pub unsafe extern "C" fn simvec_list_temp_collections(
    engine: *mut simvec_engine_t,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let names = handle.engine.list_temp_collections();
        match serde_json::to_string(&names) {
            Ok(body) => {
                let emit = emit_json(out, &body);
                if emit == simvec_error_code_t::SIMVEC_OK as c_int {
                    handle.ok()
                } else {
                    handle.fail_code(
                        simvec_error_code_t::SIMVEC_ERROR,
                        "failed to hand out temp list".to_string(),
                    )
                }
            }
            Err(err) => handle.fail_code(
                simvec_error_code_t::SIMVEC_INTERNAL,
                format!("failed to serialize temp list: {err}"),
            ),
        }
    })
}

/// List promote backups of one live collection as a JSON array of strings.
///
/// # Safety
/// - `engine` must be a valid handle.
/// - `live` must be a valid null-terminated UTF-8 string.
/// - `out` must be a valid out pointer freed with `simvec_free_string`.
#[no_mangle]
pub unsafe extern "C" fn simvec_list_old_backups(
    engine: *mut simvec_engine_t,
    live: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard(engine, |handle| {
        let live = match input_str(live) {
            Ok(live) => live,
            Err(code) => return handle.fail_code(code, "invalid live name".to_string()),
        };
        let names = handle.engine.list_old_backups_for(live);
        match serde_json::to_string(&names) {
            Ok(body) => {
                let emit = emit_json(out, &body);
                if emit == simvec_error_code_t::SIMVEC_OK as c_int {
                    handle.ok()
                } else {
                    handle.fail_code(
                        simvec_error_code_t::SIMVEC_ERROR,
                        "failed to hand out backup list".to_string(),
                    )
                }
            }
            Err(err) => handle.fail_code(
                simvec_error_code_t::SIMVEC_INTERNAL,
                format!("failed to serialize backup list: {err}"),
            ),
        }
    })
}
