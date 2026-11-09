/* Thin JNI bridge over the simvec C API.
 *
 * Every text parameter arrives as a UTF-8 byte array and every JSON output
 * leaves as a UTF-8 byte array, so standard UTF-8 is preserved end to end.
 * Null byte arrays map to NULL pointers. Outputs allocated by the library
 * are copied to Java and released with simvec_free_string before returning.
 * Error mapping stays in Java: each entry returns the raw status code.
 */
#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "simvec.h"

#define SIMVEC_JNI_NOMEM 7

typedef struct simvec_engine_t engine_t;

static const char *acquire_utf8(JNIEnv *env, jbyteArray arr, char **owned) {
    *owned = NULL;
    if (arr == NULL) {
        return NULL;
    }
    jsize len = (*env)->GetArrayLength(env, arr);
    char *buf = (char *)malloc((size_t)len + 1);
    if (buf == NULL) {
        return NULL;
    }
    (*env)->GetByteArrayRegion(env, arr, 0, len, (jbyte *)buf);
    buf[len] = '\0';
    *owned = buf;
    return buf;
}

#define ACQUIRE(name, arr)                          \
    char *name##_owned = NULL;                      \
    const char *name = acquire_utf8(env, arr, &name##_owned); \
    if ((arr) != NULL && (name) == NULL) return SIMVEC_JNI_NOMEM

#define RELEASE(name) free(name##_owned)

static void emit_bytes(JNIEnv *env, jobjectArray out, char *s) {
    if (out == NULL) {
        if (s != NULL) {
            simvec_free_string(s);
        }
        return;
    }
    if (s == NULL) {
        (*env)->SetObjectArrayElement(env, out, 0, NULL);
        return;
    }
    size_t len = strlen(s);
    jbyteArray bytes = (*env)->NewByteArray(env, (jsize)len);
    if (bytes != NULL) {
        (*env)->SetByteArrayRegion(env, bytes, 0, (jsize)len, (const jbyte *)s);
        (*env)->SetObjectArrayElement(env, out, 0, bytes);
    }
    simvec_free_string(s);
}

static void emit_int(JNIEnv *env, jintArray out, int value) {
    if (out == NULL) {
        return;
    }
    jint v = (jint)value;
    (*env)->SetIntArrayRegion(env, out, 0, 1, &v);
}

static void emit_long(JNIEnv *env, jlongArray out, uint64_t value) {
    if (out == NULL) {
        return;
    }
    jlong v = (jlong)value;
    (*env)->SetLongArrayRegion(env, out, 0, 1, &v);
}

static const float *acquire_floats(JNIEnv *env, jfloatArray arr, jfloat **owned,
                                   jsize *len_out) {
    *owned = NULL;
    *len_out = 0;
    if (arr == NULL) {
        return NULL;
    }
    jsize len = (*env)->GetArrayLength(env, arr);
    *len_out = len;
    if (len == 0) {
        return NULL;
    }
    jfloat *buf = (jfloat *)malloc((size_t)len * sizeof(jfloat));
    if (buf == NULL) {
        return NULL;
    }
    (*env)->GetFloatArrayRegion(env, arr, 0, len, buf);
    *owned = buf;
    return buf;
}

#define ACQUIRE_FLOATS(name, arr)                        \
    jfloat *name##_owned = NULL;                         \
    jsize name##_len = 0;                                \
    const float *name = acquire_floats(env, arr, &name##_owned, &name##_len); \
    if ((arr) != NULL && name##_len > 0 && (name) == NULL) return SIMVEC_JNI_NOMEM

#define RELEASE_FLOATS(name) free(name##_owned)

static engine_t *to_engine(jlong handle) {
    return (engine_t *)(uintptr_t)handle;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nOpen(JNIEnv *env, jclass cls,
                                                      jbyteArray path, jlongArray handle_out) {
    (void)cls;
    ACQUIRE(dir, path);
    engine_t *engine = NULL;
    int rc = simvec_open(dir, &engine);
    RELEASE(dir);
    if (rc == 0 && handle_out != NULL) {
        jlong h = (jlong)(uintptr_t)engine;
        (*env)->SetLongArrayRegion(env, handle_out, 0, 1, &h);
    }
    return rc;
}

JNIEXPORT jbyteArray JNICALL Java_io_simvec_NativeLib_nLastErrorText(JNIEnv *env,
                                                                     jclass cls) {
    (void)cls;
    char *s = simvec_last_error_text();
    if (s == NULL) {
        return NULL;
    }
    size_t len = strlen(s);
    jbyteArray out = (*env)->NewByteArray(env, (jsize)len);
    if (out != NULL) {
        (*env)->SetByteArrayRegion(env, out, 0, (jsize)len, (const jbyte *)s);
    }
    simvec_free_string(s);
    return out;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nClose(JNIEnv *env, jclass cls,
                                                       jlong handle) {
    (void)env;
    (void)cls;
    if (handle == 0) {
        return 0;
    }
    return simvec_close(to_engine(handle));
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nErrcode(JNIEnv *env, jclass cls,
                                                         jlong handle) {
    (void)env;
    (void)cls;
    return simvec_errcode(to_engine(handle));
}

JNIEXPORT jbyteArray JNICALL Java_io_simvec_NativeLib_nLastError(JNIEnv *env, jclass cls,
                                                                 jlong handle) {
    (void)cls;
    char *s = NULL;
    if (simvec_last_error(to_engine(handle), &s) != 0 || s == NULL) {
        if (s != NULL) {
            simvec_free_string(s);
        }
        return NULL;
    }
    size_t len = strlen(s);
    jbyteArray out = (*env)->NewByteArray(env, (jsize)len);
    if (out != NULL) {
        (*env)->SetByteArrayRegion(env, out, 0, (jsize)len, (const jbyte *)s);
    }
    simvec_free_string(s);
    return out;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nCreateCollection(JNIEnv *env, jclass cls,
                                                                  jlong handle, jbyteArray name,
                                                                  jbyteArray config) {
    (void)cls;
    ACQUIRE(c_name, name);
    ACQUIRE(c_config, config);
    int rc = simvec_create_collection(to_engine(handle), c_name, c_config);
    RELEASE(c_name);
    RELEASE(c_config);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nDeleteCollection(JNIEnv *env, jclass cls,
                                                                  jlong handle, jbyteArray name) {
    (void)cls;
    ACQUIRE(c_name, name);
    int rc = simvec_delete_collection(to_engine(handle), c_name);
    RELEASE(c_name);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nCollectionExists(JNIEnv *env, jclass cls,
                                                                  jlong handle, jbyteArray name,
                                                                  jintArray out) {
    (void)cls;
    ACQUIRE(c_name, name);
    int flag = 0;
    int rc = simvec_collection_exists(to_engine(handle), c_name, &flag);
    RELEASE(c_name);
    emit_int(env, out, flag);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nListCollections(JNIEnv *env, jclass cls,
                                                                  jlong handle, jobjectArray out) {
    (void)cls;
    char *s = NULL;
    int rc = simvec_list_collections(to_engine(handle), &s);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nCollectionInfo(JNIEnv *env, jclass cls,
                                                                 jlong handle, jbyteArray name,
                                                                 jobjectArray out) {
    (void)cls;
    ACQUIRE(c_name, name);
    char *s = NULL;
    int rc = simvec_collection_info(to_engine(handle), c_name, &s);
    RELEASE(c_name);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nCollectionConfig(JNIEnv *env, jclass cls,
                                                                   jlong handle, jbyteArray name,
                                                                   jobjectArray out) {
    (void)cls;
    ACQUIRE(c_name, name);
    char *s = NULL;
    int rc = simvec_collection_config(to_engine(handle), c_name, &s);
    RELEASE(c_name);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nUpsert(JNIEnv *env, jclass cls,
                                                        jlong handle, jbyteArray collection,
                                                        jbyteArray id, jfloatArray vector,
                                                        jbyteArray payload) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_id, id);
    ACQUIRE(c_payload, payload);
    ACQUIRE_FLOATS(c_vector, vector);
    int rc = simvec_upsert(to_engine(handle), c_collection, c_id, c_vector,
                           (size_t)c_vector_len, c_payload);
    RELEASE(c_collection);
    RELEASE(c_id);
    RELEASE(c_payload);
    RELEASE_FLOATS(c_vector);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nUpsertBatch(JNIEnv *env, jclass cls,
                                                             jlong handle, jbyteArray collection,
                                                             jbyteArray points) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_points, points);
    int rc = simvec_upsert_batch(to_engine(handle), c_collection, c_points);
    RELEASE(c_collection);
    RELEASE(c_points);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nDelete(JNIEnv *env, jclass cls,
                                                        jlong handle, jbyteArray collection,
                                                        jbyteArray id) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_id, id);
    int rc = simvec_delete(to_engine(handle), c_collection, c_id);
    RELEASE(c_collection);
    RELEASE(c_id);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nDeleteBatch(JNIEnv *env, jclass cls,
                                                             jlong handle, jbyteArray collection,
                                                             jbyteArray ids) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_ids, ids);
    int rc = simvec_delete_batch(to_engine(handle), c_collection, c_ids);
    RELEASE(c_collection);
    RELEASE(c_ids);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nDeleteByFilter(JNIEnv *env, jclass cls,
                                                                 jlong handle,
                                                                 jbyteArray collection,
                                                                 jbyteArray filter,
                                                                 jlongArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_filter, filter);
    uint64_t removed = 0;
    int rc = simvec_delete_by_filter(to_engine(handle), c_collection, c_filter, &removed);
    RELEASE(c_collection);
    RELEASE(c_filter);
    emit_long(env, out, removed);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nSetPayload(JNIEnv *env, jclass cls,
                                                            jlong handle, jbyteArray collection,
                                                            jbyteArray id, jbyteArray payload) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_id, id);
    ACQUIRE(c_payload, payload);
    int rc = simvec_set_payload(to_engine(handle), c_collection, c_id, c_payload);
    RELEASE(c_collection);
    RELEASE(c_id);
    RELEASE(c_payload);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nMergePayload(JNIEnv *env, jclass cls,
                                                              jlong handle, jbyteArray collection,
                                                              jbyteArray id, jbyteArray fields) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_id, id);
    ACQUIRE(c_fields, fields);
    int rc = simvec_merge_payload(to_engine(handle), c_collection, c_id, c_fields);
    RELEASE(c_collection);
    RELEASE(c_id);
    RELEASE(c_fields);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nDeletePayloadKeys(JNIEnv *env, jclass cls,
                                                                    jlong handle,
                                                                    jbyteArray collection,
                                                                    jbyteArray id,
                                                                    jbyteArray keys) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_id, id);
    ACQUIRE(c_keys, keys);
    int rc = simvec_delete_payload_keys(to_engine(handle), c_collection, c_id, c_keys);
    RELEASE(c_collection);
    RELEASE(c_id);
    RELEASE(c_keys);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nCreatePayloadIndex(JNIEnv *env, jclass cls,
                                                                     jlong handle,
                                                                     jbyteArray collection,
                                                                     jbyteArray field,
                                                                     jbyteArray schema) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_field, field);
    ACQUIRE(c_schema, schema);
    int rc = simvec_create_payload_index(to_engine(handle), c_collection, c_field, c_schema);
    RELEASE(c_collection);
    RELEASE(c_field);
    RELEASE(c_schema);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nDeletePayloadIndex(JNIEnv *env, jclass cls,
                                                                     jlong handle,
                                                                     jbyteArray collection,
                                                                     jbyteArray field,
                                                                     jintArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_field, field);
    int hit = 0;
    int rc = simvec_delete_payload_index(to_engine(handle), c_collection, c_field, &hit);
    RELEASE(c_collection);
    RELEASE(c_field);
    emit_int(env, out, hit);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nListPayloadIndexes(JNIEnv *env, jclass cls,
                                                                     jlong handle,
                                                                     jbyteArray collection,
                                                                     jobjectArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    char *s = NULL;
    int rc = simvec_list_payload_indexes(to_engine(handle), c_collection, &s);
    RELEASE(c_collection);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nSearch(JNIEnv *env, jclass cls,
                                                        jlong handle, jbyteArray collection,
                                                        jbyteArray query, jobjectArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_query, query);
    char *s = NULL;
    int rc = simvec_search(to_engine(handle), c_collection, c_query, &s);
    RELEASE(c_collection);
    RELEASE(c_query);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nSearchVector(JNIEnv *env, jclass cls,
                                                              jlong handle, jbyteArray collection,
                                                              jfloatArray vector, jlong limit,
                                                              jbyteArray filter,
                                                              jfloat score_threshold,
                                                              jboolean with_payload,
                                                              jboolean with_vector,
                                                              jobjectArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_filter, filter);
    ACQUIRE_FLOATS(c_vector, vector);
    char *s = NULL;
    int rc = simvec_search_vector(to_engine(handle), c_collection, c_vector,
                                  (size_t)c_vector_len, (uint64_t)limit, c_filter,
                                  (float)score_threshold, with_payload ? 1 : 0,
                                  with_vector ? 1 : 0, &s);
    RELEASE(c_collection);
    RELEASE(c_filter);
    RELEASE_FLOATS(c_vector);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nGet(JNIEnv *env, jclass cls,
                                                     jlong handle, jbyteArray collection,
                                                     jbyteArray id, jobjectArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_id, id);
    char *s = NULL;
    int rc = simvec_get(to_engine(handle), c_collection, c_id, &s);
    RELEASE(c_collection);
    RELEASE(c_id);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nCount(JNIEnv *env, jclass cls,
                                                       jlong handle, jbyteArray collection,
                                                       jlongArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    uint64_t count = 0;
    int rc = simvec_count(to_engine(handle), c_collection, &count);
    RELEASE(c_collection);
    emit_long(env, out, count);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nScroll(JNIEnv *env, jclass cls,
                                                        jlong handle, jbyteArray collection,
                                                        jlong limit, jbyteArray offset,
                                                        jboolean with_payload,
                                                        jboolean with_vector, jobjectArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_offset, offset);
    char *s = NULL;
    int rc = simvec_scroll(to_engine(handle), c_collection, (uint64_t)limit, c_offset,
                           with_payload ? 1 : 0, with_vector ? 1 : 0, &s);
    RELEASE(c_collection);
    RELEASE(c_offset);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nBuildIndex(JNIEnv *env, jclass cls,
                                                             jlong handle, jbyteArray collection,
                                                             jintArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    int done = 0;
    int rc = simvec_build_index(to_engine(handle), c_collection, &done);
    RELEASE(c_collection);
    emit_int(env, out, done);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nBuildQuantization(JNIEnv *env, jclass cls,
                                                                    jlong handle,
                                                                    jbyteArray collection,
                                                                    jintArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    int done = 0;
    int rc = simvec_build_quantization(to_engine(handle), c_collection, &done);
    RELEASE(c_collection);
    emit_int(env, out, done);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nHasIndex(JNIEnv *env, jclass cls,
                                                          jlong handle, jbyteArray collection,
                                                          jintArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    int present = 0;
    int rc = simvec_has_index(to_engine(handle), c_collection, &present);
    RELEASE(c_collection);
    emit_int(env, out, present);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nHasQuantization(JNIEnv *env, jclass cls,
                                                                 jlong handle,
                                                                 jbyteArray collection,
                                                                 jintArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    int present = 0;
    int rc = simvec_has_quantization(to_engine(handle), c_collection, &present);
    RELEASE(c_collection);
    emit_int(env, out, present);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nDropIndex(JNIEnv *env, jclass cls,
                                                           jlong handle, jbyteArray collection) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    int rc = simvec_drop_index(to_engine(handle), c_collection);
    RELEASE(c_collection);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nDropPublishedIndex(JNIEnv *env, jclass cls,
                                                                     jlong handle,
                                                                     jbyteArray collection) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    int rc = simvec_drop_published_index(to_engine(handle), c_collection);
    RELEASE(c_collection);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nCompact(JNIEnv *env, jclass cls,
                                                         jlong handle, jbyteArray collection,
                                                         jlongArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    uint64_t live = 0;
    int rc = simvec_compact(to_engine(handle), c_collection, &live);
    RELEASE(c_collection);
    emit_long(env, out, live);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nDrainPending(JNIEnv *env, jclass cls,
                                                               jlong handle,
                                                               jbyteArray collection) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    int rc = simvec_drain_pending(to_engine(handle), c_collection);
    RELEASE(c_collection);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nCollectionMetrics(JNIEnv *env, jclass cls,
                                                                    jlong handle,
                                                                    jbyteArray collection,
                                                                    jobjectArray out) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    char *s = NULL;
    int rc = simvec_collection_metrics(to_engine(handle), c_collection, &s);
    RELEASE(c_collection);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nRunMaintenanceSweep(JNIEnv *env, jclass cls,
                                                                      jlong handle) {
    (void)env;
    (void)cls;
    return simvec_run_maintenance_sweep(to_engine(handle));
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nSetPayloadField(JNIEnv *env, jclass cls,
                                                                  jlong handle,
                                                                  jbyteArray collection,
                                                                  jbyteArray id, jbyteArray key,
                                                                  jbyteArray value) {
    (void)cls;
    ACQUIRE(c_collection, collection);
    ACQUIRE(c_id, id);
    ACQUIRE(c_key, key);
    ACQUIRE(c_value, value);
    int rc = simvec_set_payload_field(to_engine(handle), c_collection, c_id, c_key, c_value);
    RELEASE(c_collection);
    RELEASE(c_id);
    RELEASE(c_key);
    RELEASE(c_value);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nApplyTxn(JNIEnv *env, jclass cls,
                                                          jlong handle, jbyteArray ops) {
    (void)cls;
    ACQUIRE(c_ops, ops);
    int rc = simvec_apply_txn(to_engine(handle), c_ops);
    RELEASE(c_ops);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nPromoteTempCollection(JNIEnv *env, jclass cls,
                                                                       jlong handle,
                                                                       jbyteArray live,
                                                                       jbyteArray temp,
                                                                       jbyteArray suffix) {
    (void)cls;
    ACQUIRE(c_live, live);
    ACQUIRE(c_temp, temp);
    ACQUIRE(c_suffix, suffix);
    int rc = simvec_promote_temp_collection(to_engine(handle), c_live, c_temp, c_suffix);
    RELEASE(c_live);
    RELEASE(c_temp);
    RELEASE(c_suffix);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nRestorePromoteBackup(JNIEnv *env, jclass cls,
                                                                       jlong handle,
                                                                       jbyteArray live,
                                                                       jobjectArray out) {
    (void)cls;
    ACQUIRE(c_live, live);
    char *s = NULL;
    int rc = simvec_restore_promote_backup(to_engine(handle), c_live, &s);
    RELEASE(c_live);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nListTempCollections(JNIEnv *env, jclass cls,
                                                                      jlong handle,
                                                                      jobjectArray out) {
    (void)cls;
    char *s = NULL;
    int rc = simvec_list_temp_collections(to_engine(handle), &s);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nListOldBackups(JNIEnv *env, jclass cls,
                                                                 jlong handle, jbyteArray live,
                                                                 jobjectArray out) {
    (void)cls;
    ACQUIRE(c_live, live);
    char *s = NULL;
    int rc = simvec_list_old_backups(to_engine(handle), c_live, &s);
    RELEASE(c_live);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nSetDefaultHnswConfig(JNIEnv *env, jclass cls,
                                                                       jlong handle,
                                                                       jbyteArray config) {
    (void)cls;
    ACQUIRE(c_config, config);
    int rc = simvec_set_default_hnsw_config(to_engine(handle), c_config);
    RELEASE(c_config);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nGetDefaultHnswConfig(JNIEnv *env, jclass cls,
                                                                       jlong handle,
                                                                       jobjectArray out) {
    (void)cls;
    char *s = NULL;
    int rc = simvec_get_default_hnsw_config(to_engine(handle), &s);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nSetDefaultIvfConfig(JNIEnv *env, jclass cls,
                                                                       jlong handle,
                                                                       jbyteArray config) {
    (void)cls;
    ACQUIRE(c_config, config);
    int rc = simvec_set_default_ivf_config(to_engine(handle), c_config);
    RELEASE(c_config);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nGetDefaultIvfConfig(JNIEnv *env, jclass cls,
                                                                      jlong handle,
                                                                      jobjectArray out) {
    (void)cls;
    char *s = NULL;
    int rc = simvec_get_default_ivf_config(to_engine(handle), &s);
    emit_bytes(env, out, s);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nSetDefaultQuantizationConfig(JNIEnv *env,
                                                                               jclass cls,
                                                                               jlong handle,
                                                                               jbyteArray config) {
    (void)cls;
    ACQUIRE(c_config, config);
    int rc = simvec_set_default_quantization_config(to_engine(handle), c_config);
    RELEASE(c_config);
    return rc;
}

JNIEXPORT jint JNICALL Java_io_simvec_NativeLib_nGetDefaultQuantizationConfig(JNIEnv *env,
                                                                               jclass cls,
                                                                               jlong handle,
                                                                               jobjectArray out) {
    (void)cls;
    char *s = NULL;
    int rc = simvec_get_default_quantization_config(to_engine(handle), &s);
    emit_bytes(env, out, s);
    return rc;
}
