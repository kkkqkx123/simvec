#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

SEXP C_simvec_open(SEXP r_path);
SEXP C_simvec_close(SEXP r_handle);
SEXP C_simvec_closed(SEXP r_handle);
SEXP C_simvec_last_code(SEXP r_handle);
SEXP C_simvec_last_message(SEXP r_handle);
SEXP C_simvec_create_collection(SEXP r_handle, SEXP r_name, SEXP r_config);
SEXP C_simvec_delete_collection(SEXP r_handle, SEXP r_name);
SEXP C_simvec_collection_exists(SEXP r_handle, SEXP r_name);
SEXP C_simvec_list_collections(SEXP r_handle);
SEXP C_simvec_collection_info(SEXP r_handle, SEXP r_name);
SEXP C_simvec_collection_config(SEXP r_handle, SEXP r_name);
SEXP C_simvec_upsert(SEXP r_handle, SEXP r_collection, SEXP r_id, SEXP r_vector,
                     SEXP r_payload);
SEXP C_simvec_upsert_batch(SEXP r_handle, SEXP r_collection, SEXP r_points);
SEXP C_simvec_delete(SEXP r_handle, SEXP r_collection, SEXP r_id);
SEXP C_simvec_delete_batch(SEXP r_handle, SEXP r_collection, SEXP r_ids);
SEXP C_simvec_delete_by_filter(SEXP r_handle, SEXP r_collection, SEXP r_filter);
SEXP C_simvec_set_payload(SEXP r_handle, SEXP r_collection, SEXP r_id, SEXP r_payload);
SEXP C_simvec_merge_payload(SEXP r_handle, SEXP r_collection, SEXP r_id,
                            SEXP r_fields);
SEXP C_simvec_delete_payload_keys(SEXP r_handle, SEXP r_collection, SEXP r_id,
                                  SEXP r_keys);
SEXP C_simvec_create_payload_index(SEXP r_handle, SEXP r_collection, SEXP r_field,
                                   SEXP r_schema);
SEXP C_simvec_delete_payload_index(SEXP r_handle, SEXP r_collection, SEXP r_field);
SEXP C_simvec_list_payload_indexes(SEXP r_handle, SEXP r_collection);
SEXP C_simvec_search(SEXP r_handle, SEXP r_collection, SEXP r_query);
SEXP C_simvec_search_vector(SEXP r_handle, SEXP r_collection, SEXP r_vector,
                            SEXP r_limit, SEXP r_filter, SEXP r_threshold,
                            SEXP r_with_payload, SEXP r_with_vector);
SEXP C_simvec_get(SEXP r_handle, SEXP r_collection, SEXP r_id);
SEXP C_simvec_count(SEXP r_handle, SEXP r_collection);
SEXP C_simvec_count_by_filter(SEXP r_handle, SEXP r_collection, SEXP r_filter);
SEXP C_simvec_scroll(SEXP r_handle, SEXP r_collection, SEXP r_limit, SEXP r_offset,
                     SEXP r_with_payload, SEXP r_with_vector);
SEXP C_simvec_scroll_by_filter(SEXP r_handle, SEXP r_collection, SEXP r_limit, SEXP r_offset,
                               SEXP r_filter, SEXP r_with_payload, SEXP r_with_vector);
SEXP C_simvec_build_index(SEXP r_handle, SEXP r_collection);
SEXP C_simvec_build_quantization(SEXP r_handle, SEXP r_collection);
SEXP C_simvec_has_index(SEXP r_handle, SEXP r_collection);
SEXP C_simvec_has_quantization(SEXP r_handle, SEXP r_collection);
SEXP C_simvec_drop_index(SEXP r_handle, SEXP r_collection);
SEXP C_simvec_drop_published_index(SEXP r_handle, SEXP r_collection);
SEXP C_simvec_compact(SEXP r_handle, SEXP r_collection);
SEXP C_simvec_drain_pending(SEXP r_handle, SEXP r_collection);
SEXP C_simvec_collection_metrics(SEXP r_handle, SEXP r_collection);
SEXP C_simvec_run_maintenance_sweep(SEXP r_handle);
SEXP C_simvec_set_payload_field(SEXP r_handle, SEXP r_collection, SEXP r_id,
                                SEXP r_key, SEXP r_value);
SEXP C_simvec_apply_txn(SEXP r_handle, SEXP r_ops);
SEXP C_simvec_promote_temp_collection(SEXP r_handle, SEXP r_live, SEXP r_temp,
                                      SEXP r_suffix);
SEXP C_simvec_restore_promote_backup(SEXP r_handle, SEXP r_live);
SEXP C_simvec_list_temp_collections(SEXP r_handle);
SEXP C_simvec_list_old_backups(SEXP r_handle, SEXP r_live);
SEXP C_simvec_set_default_hnsw_config(SEXP r_handle, SEXP r_config);
SEXP C_simvec_get_default_hnsw_config(SEXP r_handle);
SEXP C_simvec_set_default_ivf_config(SEXP r_handle, SEXP r_config);
SEXP C_simvec_get_default_ivf_config(SEXP r_handle);
SEXP C_simvec_set_default_quantization_config(SEXP r_handle, SEXP r_config);
SEXP C_simvec_get_default_quantization_config(SEXP r_handle);

static const R_CallMethodDef call_defs[] = {
    {"C_simvec_open", (DL_FUNC)&C_simvec_open, 1},
    {"C_simvec_close", (DL_FUNC)&C_simvec_close, 1},
    {"C_simvec_closed", (DL_FUNC)&C_simvec_closed, 1},
    {"C_simvec_last_code", (DL_FUNC)&C_simvec_last_code, 1},
    {"C_simvec_last_message", (DL_FUNC)&C_simvec_last_message, 1},
    {"C_simvec_create_collection", (DL_FUNC)&C_simvec_create_collection, 3},
    {"C_simvec_delete_collection", (DL_FUNC)&C_simvec_delete_collection, 2},
    {"C_simvec_collection_exists", (DL_FUNC)&C_simvec_collection_exists, 2},
    {"C_simvec_list_collections", (DL_FUNC)&C_simvec_list_collections, 1},
    {"C_simvec_collection_info", (DL_FUNC)&C_simvec_collection_info, 2},
    {"C_simvec_collection_config", (DL_FUNC)&C_simvec_collection_config, 2},
    {"C_simvec_upsert", (DL_FUNC)&C_simvec_upsert, 5},
    {"C_simvec_upsert_batch", (DL_FUNC)&C_simvec_upsert_batch, 3},
    {"C_simvec_delete", (DL_FUNC)&C_simvec_delete, 3},
    {"C_simvec_delete_batch", (DL_FUNC)&C_simvec_delete_batch, 3},
    {"C_simvec_delete_by_filter", (DL_FUNC)&C_simvec_delete_by_filter, 3},
    {"C_simvec_set_payload", (DL_FUNC)&C_simvec_set_payload, 4},
    {"C_simvec_merge_payload", (DL_FUNC)&C_simvec_merge_payload, 4},
    {"C_simvec_delete_payload_keys", (DL_FUNC)&C_simvec_delete_payload_keys, 4},
    {"C_simvec_create_payload_index", (DL_FUNC)&C_simvec_create_payload_index, 4},
    {"C_simvec_delete_payload_index", (DL_FUNC)&C_simvec_delete_payload_index, 3},
    {"C_simvec_list_payload_indexes", (DL_FUNC)&C_simvec_list_payload_indexes, 2},
    {"C_simvec_search", (DL_FUNC)&C_simvec_search, 3},
    {"C_simvec_search_vector", (DL_FUNC)&C_simvec_search_vector, 8},
    {"C_simvec_get", (DL_FUNC)&C_simvec_get, 3},
    {"C_simvec_count", (DL_FUNC)&C_simvec_count, 2},
    {"C_simvec_count_by_filter", (DL_FUNC)&C_simvec_count_by_filter, 3},
    {"C_simvec_scroll", (DL_FUNC)&C_simvec_scroll, 6},
    {"C_simvec_scroll_by_filter", (DL_FUNC)&C_simvec_scroll_by_filter, 7},
    {"C_simvec_build_index", (DL_FUNC)&C_simvec_build_index, 2},
    {"C_simvec_build_quantization", (DL_FUNC)&C_simvec_build_quantization, 2},
    {"C_simvec_has_index", (DL_FUNC)&C_simvec_has_index, 2},
    {"C_simvec_has_quantization", (DL_FUNC)&C_simvec_has_quantization, 2},
    {"C_simvec_drop_index", (DL_FUNC)&C_simvec_drop_index, 2},
    {"C_simvec_drop_published_index", (DL_FUNC)&C_simvec_drop_published_index, 2},
    {"C_simvec_compact", (DL_FUNC)&C_simvec_compact, 2},
    {"C_simvec_drain_pending", (DL_FUNC)&C_simvec_drain_pending, 2},
    {"C_simvec_collection_metrics", (DL_FUNC)&C_simvec_collection_metrics, 2},
    {"C_simvec_run_maintenance_sweep", (DL_FUNC)&C_simvec_run_maintenance_sweep, 1},
    {"C_simvec_set_payload_field", (DL_FUNC)&C_simvec_set_payload_field, 5},
    {"C_simvec_apply_txn", (DL_FUNC)&C_simvec_apply_txn, 2},
    {"C_simvec_promote_temp_collection", (DL_FUNC)&C_simvec_promote_temp_collection, 4},
    {"C_simvec_restore_promote_backup", (DL_FUNC)&C_simvec_restore_promote_backup, 2},
    {"C_simvec_list_temp_collections", (DL_FUNC)&C_simvec_list_temp_collections, 1},
    {"C_simvec_list_old_backups", (DL_FUNC)&C_simvec_list_old_backups, 2},
    {"C_simvec_set_default_hnsw_config", (DL_FUNC)&C_simvec_set_default_hnsw_config, 2},
    {"C_simvec_get_default_hnsw_config", (DL_FUNC)&C_simvec_get_default_hnsw_config, 1},
    {"C_simvec_set_default_ivf_config", (DL_FUNC)&C_simvec_set_default_ivf_config, 2},
    {"C_simvec_get_default_ivf_config", (DL_FUNC)&C_simvec_get_default_ivf_config, 1},
    {"C_simvec_set_default_quantization_config",
     (DL_FUNC)&C_simvec_set_default_quantization_config, 2},
    {"C_simvec_get_default_quantization_config",
     (DL_FUNC)&C_simvec_get_default_quantization_config, 1},
    {NULL, NULL, 0},
};

void R_init_simvec(DllInfo *dll) {
    R_registerRoutines(dll, NULL, call_defs, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
