/* Uniform .Call bridge over the simvec C API.
 *
 * Each entry returns a 3-list: code (integer status), payload (result value
 * or NULL), errmsg (message string or NULL). Error mapping stays in R: the
 * bridge never signals simvec failures itself, it only reports them. Only
 * programming errors in the SEXP shapes use Rf_error.
 *
 * Text arrives as UTF-8 strings and JSON outputs leave marked CE_UTF8.
 * Vectors arrive as numeric and are copied to contiguous float32 arrays
 * held only for the call. Outputs allocated by the library are converted
 * to R values and released with simvec_free_string before returning.
 */
#include <R.h>
#include <Rinternals.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "simvec.h"

typedef struct simvec_engine_t engine_t;

static engine_t *engine_ptr(SEXP ext) {
    if (TYPEOF(ext) != EXTPTRSXP) {
        return NULL;
    }
    return (engine_t *)R_ExternalPtrAddr(ext);
}

static SEXP make_result(int code, SEXP payload, const char *msg) {
    SEXP res = PROTECT(allocVector(VECSXP, 3));
    SEXP names = PROTECT(allocVector(STRSXP, 3));
    SET_STRING_ELT(names, 0, mkChar("code"));
    SET_STRING_ELT(names, 1, mkChar("payload"));
    SET_STRING_ELT(names, 2, mkChar("errmsg"));
    setAttrib(res, R_NamesSymbol, names);
    SET_VECTOR_ELT(res, 0, ScalarInteger(code));
    SET_VECTOR_ELT(res, 1, payload ? payload : R_NilValue);
    if (msg) {
        SET_VECTOR_ELT(res, 2, ScalarString(mkCharCE(msg, CE_UTF8)));
    } else {
        SET_VECTOR_ELT(res, 2, R_NilValue);
    }
    UNPROTECT(2);
    return res;
}

#define RETURN3(code, payload, msg)                 \
    do {                                            \
        SEXP keep_ = (payload);                     \
        PROTECT(keep_);                             \
        SEXP out_ = make_result((code), keep_, (msg)); \
        UNPROTECT(1);                               \
        return out_;                                \
    } while (0)

static const char *borrow_str(SEXP value, int *bad) {
    *bad = 0;
    if (isNull(value)) {
        return NULL;
    }
    if (TYPEOF(value) != STRSXP || XLENGTH(value) != 1) {
        *bad = 1;
        return NULL;
    }
    SEXP elt = STRING_ELT(value, 0);
    if (elt == NA_STRING) {
        *bad = 1;
        return NULL;
    }
    return translateCharUTF8(elt);
}

#define NEED_STR(name, sexp)                                                \
    int name##_bad = 0;                                                     \
    const char *name = borrow_str((sexp), &name##_bad);                      \
    if (name##_bad) {                                                       \
        RETURN3(20, R_NilValue, "string argument must be a non-missing single string"); \
    }

static char *fetch_message(engine_t *handle) {
    char *text = NULL;
    if (simvec_last_error(handle, &text) != 0) {
        if (text) {
            simvec_free_string(text);
        }
        return NULL;
    }
    if (!text) {
        return NULL;
    }
    size_t len = strlen(text) + 1;
    char *copy = (char *)malloc(len);
    if (copy) {
        memcpy(copy, text, len);
    }
    simvec_free_string(text);
    return copy;
}

#define FAIL_WITH_MESSAGE(handle, code)             \
    do {                                            \
        char *msg_ = fetch_message(handle);         \
        SEXP res_ = make_result((code), R_NilValue, msg_); \
        free(msg_);                                 \
        return res_;                                \
    } while (0)

static SEXP json_payload(char *text) {
    if (!text) {
        return R_NilValue;
    }
    SEXP out = ScalarString(mkCharCE(text, CE_UTF8));
    simvec_free_string(text);
    return out;
}

static float *copy_vector(SEXP value, size_t *len_out, int *oom) {
    *len_out = 0;
    *oom = 0;
    if (isNull(value)) {
        return NULL;
    }
    SEXP coerced = PROTECT(coerceVector(value, REALSXP));
    R_xlen_t n = XLENGTH(coerced);
    float *buf = NULL;
    if (n > 0) {
        buf = (float *)malloc((size_t)n * sizeof(float));
        if (!buf) {
            *oom = 1;
        } else {
            for (R_xlen_t i = 0; i < n; i++) {
                buf[i] = (float)REAL(coerced)[i];
            }
        }
    }
    *len_out = (size_t)n;
    UNPROTECT(1);
    return buf;
}

#define NEED_VECTOR(name, sexp)                                              \
    size_t name##_len = 0;                                                  \
    int name##_oom = 0;                                                     \
    float *name = copy_vector((sexp), &name##_len, &name##_oom);             \
    if (name##_oom) {                                                       \
        RETURN3(7, R_NilValue, "out of memory");                             \
    }

static void finalize_engine(SEXP ptr) {
    engine_t *handle = engine_ptr(ptr);
    if (handle) {
        simvec_close(handle);
        R_ClearExternalPtr(ptr);
    }
}

SEXP C_simvec_open(SEXP r_path) {
    NEED_STR(path, r_path);
    if (!path) {
        RETURN3(20, R_NilValue, "path must be a non-missing single string");
    }
    engine_t *handle = NULL;
    int rc = simvec_open(path, &handle);
    if (rc != 0) {
        char *msg = simvec_last_error_text();
        SEXP res = make_result(rc, R_NilValue, msg);
        if (msg) {
            simvec_free_string(msg);
        }
        return res;
    }
    SEXP ptr = R_MakeExternalPtr((void *)handle, R_NilValue, R_NilValue);
    R_RegisterCFinalizerEx(ptr, finalize_engine, TRUE);
    RETURN3(0, ptr, NULL);
}

SEXP C_simvec_close(SEXP r_handle) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    int rc = simvec_close(handle);
    if (rc != 0) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    R_ClearExternalPtr(r_handle);
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_closed(SEXP r_handle) {
    RETURN3(0, ScalarLogical(engine_ptr(r_handle) ? FALSE : TRUE), NULL);
}

SEXP C_simvec_last_code(SEXP r_handle) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(0, ScalarInteger(20), NULL);
    }
    RETURN3(0, ScalarInteger(simvec_errcode(handle)), NULL);
}

SEXP C_simvec_last_message(SEXP r_handle) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(0, ScalarString(mkCharCE("engine is closed", CE_UTF8)), NULL);
    }
    char *msg = fetch_message(handle);
    SEXP payload = msg ? ScalarString(mkCharCE(msg, CE_UTF8))
                       : ScalarString(mkCharCE("", CE_UTF8));
    free(msg);
    RETURN3(0, payload, NULL);
}

SEXP C_simvec_create_collection(SEXP r_handle, SEXP r_name, SEXP r_config) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(name, r_name);
    NEED_STR(config, r_config);
    int rc = simvec_create_collection(handle, name, config);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_delete_collection(SEXP r_handle, SEXP r_name) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(name, r_name);
    int rc = simvec_delete_collection(handle, name);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_collection_exists(SEXP r_handle, SEXP r_name) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(name, r_name);
    int flag = 0;
    int rc = simvec_collection_exists(handle, name, &flag);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, ScalarLogical(flag ? TRUE : FALSE), NULL);
}

SEXP C_simvec_list_collections(SEXP r_handle) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    char *text = NULL;
    int rc = simvec_list_collections(handle, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_collection_info(SEXP r_handle, SEXP r_name) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(name, r_name);
    char *text = NULL;
    int rc = simvec_collection_info(handle, name, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_collection_config(SEXP r_handle, SEXP r_name) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(name, r_name);
    char *text = NULL;
    int rc = simvec_collection_config(handle, name, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_upsert(SEXP r_handle, SEXP r_collection, SEXP r_id, SEXP r_vector,
                     SEXP r_payload) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(id, r_id);
    NEED_STR(payload, r_payload);
    NEED_VECTOR(vector, r_vector);
    int rc = simvec_upsert(handle, collection, id, vector, vector_len, payload);
    free(vector);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_upsert_batch(SEXP r_handle, SEXP r_collection, SEXP r_points) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(points, r_points);
    int rc = simvec_upsert_batch(handle, collection, points);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_delete(SEXP r_handle, SEXP r_collection, SEXP r_id) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(id, r_id);
    int rc = simvec_delete(handle, collection, id);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_delete_batch(SEXP r_handle, SEXP r_collection, SEXP r_ids) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(ids, r_ids);
    int rc = simvec_delete_batch(handle, collection, ids);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_delete_by_filter(SEXP r_handle, SEXP r_collection, SEXP r_filter) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(filter, r_filter);
    uint64_t removed = 0;
    int rc = simvec_delete_by_filter(handle, collection, filter, &removed);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, ScalarReal((double)removed), NULL);
}

SEXP C_simvec_set_payload(SEXP r_handle, SEXP r_collection, SEXP r_id, SEXP r_payload) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(id, r_id);
    NEED_STR(payload, r_payload);
    int rc = simvec_set_payload(handle, collection, id, payload);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_merge_payload(SEXP r_handle, SEXP r_collection, SEXP r_id,
                            SEXP r_fields) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(id, r_id);
    NEED_STR(fields, r_fields);
    int rc = simvec_merge_payload(handle, collection, id, fields);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_delete_payload_keys(SEXP r_handle, SEXP r_collection, SEXP r_id,
                                  SEXP r_keys) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(id, r_id);
    NEED_STR(keys, r_keys);
    int rc = simvec_delete_payload_keys(handle, collection, id, keys);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_create_payload_index(SEXP r_handle, SEXP r_collection, SEXP r_field,
                                   SEXP r_schema) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(field, r_field);
    NEED_STR(schema, r_schema);
    int rc = simvec_create_payload_index(handle, collection, field, schema);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_delete_payload_index(SEXP r_handle, SEXP r_collection, SEXP r_field) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(field, r_field);
    int hit = 0;
    int rc = simvec_delete_payload_index(handle, collection, field, &hit);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, ScalarLogical(hit ? TRUE : FALSE), NULL);
}

SEXP C_simvec_list_payload_indexes(SEXP r_handle, SEXP r_collection) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    char *text = NULL;
    int rc = simvec_list_payload_indexes(handle, collection, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_search(SEXP r_handle, SEXP r_collection, SEXP r_query) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(query, r_query);
    char *text = NULL;
    int rc = simvec_search(handle, collection, query, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_search_vector(SEXP r_handle, SEXP r_collection, SEXP r_vector,
                            SEXP r_limit, SEXP r_filter, SEXP r_threshold,
                            SEXP r_with_payload, SEXP r_with_vector) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(filter, r_filter);
    NEED_VECTOR(vector, r_vector);
    uint64_t limit = (uint64_t)asReal(r_limit);
    float threshold = (float)asReal(r_threshold);
    int rc;
    char *text = NULL;
    rc = simvec_search_vector(handle, collection, vector, vector_len, limit, filter,
                              threshold, asLogical(r_with_payload) ? 1 : 0,
                              asLogical(r_with_vector) ? 1 : 0, &text);
    free(vector);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_get(SEXP r_handle, SEXP r_collection, SEXP r_id) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(id, r_id);
    char *text = NULL;
    int rc = simvec_get(handle, collection, id, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_count(SEXP r_handle, SEXP r_collection) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    uint64_t count = 0;
    int rc = simvec_count(handle, collection, &count);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, ScalarReal((double)count), NULL);
}

SEXP C_simvec_scroll(SEXP r_handle, SEXP r_collection, SEXP r_limit, SEXP r_offset,
                     SEXP r_with_payload, SEXP r_with_vector) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(offset, r_offset);
    uint64_t limit = (uint64_t)asReal(r_limit);
    char *text = NULL;
    int rc = simvec_scroll(handle, collection, limit, offset,
                           asLogical(r_with_payload) ? 1 : 0,
                           asLogical(r_with_vector) ? 1 : 0, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_build_index(SEXP r_handle, SEXP r_collection) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    int done = 0;
    int rc = simvec_build_index(handle, collection, &done);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, ScalarLogical(done ? TRUE : FALSE), NULL);
}

SEXP C_simvec_build_quantization(SEXP r_handle, SEXP r_collection) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    int done = 0;
    int rc = simvec_build_quantization(handle, collection, &done);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, ScalarLogical(done ? TRUE : FALSE), NULL);
}

SEXP C_simvec_has_index(SEXP r_handle, SEXP r_collection) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    int present = 0;
    int rc = simvec_has_index(handle, collection, &present);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, ScalarLogical(present ? TRUE : FALSE), NULL);
}

SEXP C_simvec_has_quantization(SEXP r_handle, SEXP r_collection) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    int present = 0;
    int rc = simvec_has_quantization(handle, collection, &present);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, ScalarLogical(present ? TRUE : FALSE), NULL);
}

SEXP C_simvec_drop_index(SEXP r_handle, SEXP r_collection) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    int rc = simvec_drop_index(handle, collection);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_drop_published_index(SEXP r_handle, SEXP r_collection) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    int rc = simvec_drop_published_index(handle, collection);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_compact(SEXP r_handle, SEXP r_collection) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    uint64_t live = 0;
    int rc = simvec_compact(handle, collection, &live);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, ScalarReal((double)live), NULL);
}

SEXP C_simvec_drain_pending(SEXP r_handle, SEXP r_collection) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    int rc = simvec_drain_pending(handle, collection);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_collection_metrics(SEXP r_handle, SEXP r_collection) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    char *text = NULL;
    int rc = simvec_collection_metrics(handle, collection, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_run_maintenance_sweep(SEXP r_handle) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    int rc = simvec_run_maintenance_sweep(handle);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_set_payload_field(SEXP r_handle, SEXP r_collection, SEXP r_id,
                                SEXP r_key, SEXP r_value) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(collection, r_collection);
    NEED_STR(id, r_id);
    NEED_STR(key, r_key);
    NEED_STR(value, r_value);
    int rc = simvec_set_payload_field(handle, collection, id, key, value);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_apply_txn(SEXP r_handle, SEXP r_ops) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(ops, r_ops);
    int rc = simvec_apply_txn(handle, ops);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_promote_temp_collection(SEXP r_handle, SEXP r_live, SEXP r_temp,
                                      SEXP r_suffix) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(live, r_live);
    NEED_STR(temp, r_temp);
    NEED_STR(suffix, r_suffix);
    int rc = simvec_promote_temp_collection(handle, live, temp, suffix);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_restore_promote_backup(SEXP r_handle, SEXP r_live) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(live, r_live);
    char *text = NULL;
    int rc = simvec_restore_promote_backup(handle, live, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_list_temp_collections(SEXP r_handle) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    char *text = NULL;
    int rc = simvec_list_temp_collections(handle, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_list_old_backups(SEXP r_handle, SEXP r_live) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(live, r_live);
    char *text = NULL;
    int rc = simvec_list_old_backups(handle, live, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_set_default_hnsw_config(SEXP r_handle, SEXP r_config) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(config, r_config);
    int rc = simvec_set_default_hnsw_config(handle, config);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_get_default_hnsw_config(SEXP r_handle) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    char *text = NULL;
    int rc = simvec_get_default_hnsw_config(handle, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_set_default_ivf_config(SEXP r_handle, SEXP r_config) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(config, r_config);
    int rc = simvec_set_default_ivf_config(handle, config);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_get_default_ivf_config(SEXP r_handle) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    char *text = NULL;
    int rc = simvec_get_default_ivf_config(handle, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}

SEXP C_simvec_set_default_quantization_config(SEXP r_handle, SEXP r_config) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    NEED_STR(config, r_config);
    int rc = simvec_set_default_quantization_config(handle, config);
    if (rc) {
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, R_NilValue, NULL);
}

SEXP C_simvec_get_default_quantization_config(SEXP r_handle) {
    engine_t *handle = engine_ptr(r_handle);
    if (!handle) {
        RETURN3(20, R_NilValue, "engine is closed");
    }
    char *text = NULL;
    int rc = simvec_get_default_quantization_config(handle, &text);
    if (rc) {
        if (text) {
            simvec_free_string(text);
        }
        FAIL_WITH_MESSAGE(handle, rc);
    }
    RETURN3(0, json_payload(text), NULL);
}
