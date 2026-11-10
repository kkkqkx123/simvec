#ifndef SIMVEC_H
#define SIMVEC_H

/* Hand-maintained mirror of the cbindgen output from src/c_api.
 * Regenerate with `cargo build --features c_api` once cbindgen is available
 * and diff the result against this file. */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Stable C status codes for every simvec entry point.
 */
typedef enum simvec_error_code_t {
  /**
   * Success.
   */
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
  /**
   * Null pointer, invalid UTF-8 or malformed caller-supplied JSON.
   */
  SIMVEC_MISUSE = 20,
  SIMVEC_NON_FINITE = 21,
  SIMVEC_CAPACITY_EXCEEDED = 22,
  SIMVEC_FILTER = 23,
} simvec_error_code_t;

/**
 * Opaque engine handle owned by `simvec_open` / `simvec_close`.
 */
typedef struct simvec_engine_t {
  uint8_t _dummy;
} simvec_engine_t;

#ifdef __cplusplus
extern "C" {
#endif

int simvec_open(const char *path, struct simvec_engine_t **out);

int simvec_close(struct simvec_engine_t *engine);

void simvec_free_string(char *s);

int simvec_errcode(struct simvec_engine_t *engine);

int simvec_last_error(struct simvec_engine_t *engine, char **out);

char *simvec_last_error_text(void);

int simvec_create_collection(struct simvec_engine_t *engine,
                             const char *name,
                             const char *config_json);

int simvec_delete_collection(struct simvec_engine_t *engine, const char *name);

int simvec_collection_exists(struct simvec_engine_t *engine, const char *name, int *out);

int simvec_list_collections(struct simvec_engine_t *engine, char **out);

int simvec_collection_info(struct simvec_engine_t *engine, const char *name, char **out);

int simvec_collection_config(struct simvec_engine_t *engine, const char *name, char **out);

int simvec_upsert(struct simvec_engine_t *engine,
                  const char *collection,
                  const char *id,
                  const float *vector,
                  size_t vector_len,
                  const char *payload_json);

int simvec_upsert_batch(struct simvec_engine_t *engine,
                        const char *collection,
                        const char *points_json);

int simvec_delete(struct simvec_engine_t *engine, const char *collection, const char *id);

int simvec_delete_batch(struct simvec_engine_t *engine,
                        const char *collection,
                        const char *ids_json);

int simvec_delete_by_filter(struct simvec_engine_t *engine,
                            const char *collection,
                            const char *filter_json,
                            uint64_t *out);

int simvec_set_payload(struct simvec_engine_t *engine,
                       const char *collection,
                       const char *id,
                       const char *payload_json);

int simvec_merge_payload(struct simvec_engine_t *engine,
                         const char *collection,
                         const char *id,
                         const char *fields_json);

int simvec_delete_payload_keys(struct simvec_engine_t *engine,
                               const char *collection,
                               const char *id,
                               const char *keys_json);

int simvec_create_payload_index(struct simvec_engine_t *engine,
                                const char *collection,
                                const char *field,
                                const char *schema);

int simvec_delete_payload_index(struct simvec_engine_t *engine,
                                const char *collection,
                                const char *field,
                                int *out);

int simvec_list_payload_indexes(struct simvec_engine_t *engine,
                                const char *collection,
                                char **out);

int simvec_search(struct simvec_engine_t *engine,
                  const char *collection,
                  const char *query_json,
                  char **out);

int simvec_search_vector(struct simvec_engine_t *engine,
                         const char *collection,
                         const float *vector,
                         size_t vector_len,
                         uint64_t limit,
                         const char *filter_json,
                         float score_threshold,
                         int with_payload,
                         int with_vector,
                         char **out);

int simvec_get(struct simvec_engine_t *engine,
               const char *collection,
               const char *id,
               char **out);

int simvec_count(struct simvec_engine_t *engine, const char *collection, uint64_t *out);

int simvec_count_by_filter(struct simvec_engine_t *engine,
                           const char *collection,
                           const char *filter_json,
                           uint64_t *out);

int simvec_scroll(struct simvec_engine_t *engine,
                  const char *collection,
                  uint64_t limit,
                  const char *offset,
                  int with_payload,
                  int with_vector,
                  char **out);

int simvec_scroll_by_filter(struct simvec_engine_t *engine,
                            const char *collection,
                            uint64_t limit,
                            const char *offset,
                            const char *filter_json,
                            int with_payload,
                            int with_vector,
                            char **out);

int simvec_build_index(struct simvec_engine_t *engine, const char *collection, int *out);

int simvec_build_quantization(struct simvec_engine_t *engine, const char *collection, int *out);

int simvec_has_index(struct simvec_engine_t *engine, const char *collection, int *out);

int simvec_has_quantization(struct simvec_engine_t *engine, const char *collection, int *out);

int simvec_drop_index(struct simvec_engine_t *engine, const char *collection);

int simvec_compact(struct simvec_engine_t *engine, const char *collection, uint64_t *out);

int simvec_drain_pending(struct simvec_engine_t *engine, const char *collection);

int simvec_collection_metrics(struct simvec_engine_t *engine, const char *collection, char **out);

int simvec_run_maintenance_sweep(struct simvec_engine_t *engine);

int simvec_set_payload_field(struct simvec_engine_t *engine,
                              const char *collection,
                              const char *id,
                              const char *key,
                              const char *value_json);

int simvec_apply_txn(struct simvec_engine_t *engine, const char *ops_json);

int simvec_drop_published_index(struct simvec_engine_t *engine, const char *collection);

int simvec_promote_temp_collection(struct simvec_engine_t *engine,
                                     const char *live,
                                     const char *temp,
                                     const char *backup_suffix);

int simvec_restore_promote_backup(struct simvec_engine_t *engine, const char *live, char **out);

int simvec_list_temp_collections(struct simvec_engine_t *engine, char **out);

int simvec_list_old_backups(struct simvec_engine_t *engine, const char *live, char **out);

int simvec_set_default_hnsw_config(struct simvec_engine_t *engine, const char *config_json);

int simvec_get_default_hnsw_config(struct simvec_engine_t *engine, char **out);

int simvec_set_default_ivf_config(struct simvec_engine_t *engine, const char *config_json);

int simvec_get_default_ivf_config(struct simvec_engine_t *engine, char **out);

int simvec_set_default_quantization_config(struct simvec_engine_t *engine, const char *config_json);

int simvec_get_default_quantization_config(struct simvec_engine_t *engine, char **out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SIMVEC_H */
