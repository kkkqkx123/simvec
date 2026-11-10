tostring <- function(x, what) {
  if (is.null(x)) {
    return(NULL)
  }
  if (length(x) != 1L || is.na(x)) {
    stop_simvec(20L, sprintf("%s must be a non-missing single string", what))
  }
  as.character(x)
}

to_json_or_null <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null", digits = NA))
}

from_json <- function(text) {
  jsonlite::fromJSON(text, simplifyVector = FALSE, simplifyDataFrame = FALSE,
    simplifyMatrix = FALSE)
}

check_engine <- function(engine) {
  if (!inherits(engine, "simvec_engine") || is.null(engine$ptr)) {
    stop_simvec(20L, "engine is closed")
  }
  engine
}

#' Open the engine rooted at a directory.
#'
#' The caller keeps the binding and libsimvec from the same checkout.
#' @export
simvec_open <- function(path) {
  path <- tostring(path, "path")
  res <- .Call(C_simvec_open, path)
  if (as.integer(res$code) != 0L) {
    stop_simvec(res$code, res$errmsg)
  }
  engine <- new.env(parent = emptyenv())
  engine$ptr <- res$payload
  class(engine) <- "simvec_engine"
  reg.finalizer(engine, function(e) {
    tryCatch(simvec_close(e), error = function(unused) NULL)
  }, onexit = TRUE)
  engine
}

#' Close the engine; safe to call more than once.
#' @export
simvec_close <- function(engine) {
  if (!inherits(engine, "simvec_engine") || is.null(engine$ptr)) {
    return(invisible(FALSE))
  }
  res <- .Call(C_simvec_close, engine$ptr)
  engine$ptr <- NULL
  if (as.integer(res$code) != 0L) {
    stop_simvec(res$code, res$errmsg)
  }
  invisible(TRUE)
}

#' Whether the engine has been closed.
#' @export
simvec_closed <- function(engine) {
  if (!inherits(engine, "simvec_engine") || is.null(engine$ptr)) {
    return(TRUE)
  }
  isTRUE(.Call(C_simvec_closed, engine$ptr)$payload)
}

#' Most recent status code recorded on the native handle.
#' @export
simvec_last_code <- function(engine) {
  if (!inherits(engine, "simvec_engine") || is.null(engine$ptr)) {
    return(20L)
  }
  as.integer(.Call(C_simvec_last_code, engine$ptr)$payload)
}

#' Most recent error text recorded on the native handle.
#' @export
simvec_last_message <- function(engine) {
  if (!inherits(engine, "simvec_engine") || is.null(engine$ptr)) {
    return("engine is closed")
  }
  .Call(C_simvec_last_message, engine$ptr)$payload
}

#' @export
simvec_create_collection <- function(engine, name, config = NULL) {
  check_engine(engine)
  res <- .Call(C_simvec_create_collection, engine$ptr,
    tostring(name, "name"), to_json_or_null(config))
  raise_if_failed(res)
}

#' @export
simvec_delete_collection <- function(engine, name) {
  check_engine(engine)
  res <- .Call(C_simvec_delete_collection, engine$ptr, tostring(name, "name"))
  raise_if_failed(res)
}

#' @export
simvec_collection_exists <- function(engine, name) {
  check_engine(engine)
  res <- .Call(C_simvec_collection_exists, engine$ptr, tostring(name, "name"))
  isTRUE(payload_or_raise(res))
}

#' @export
simvec_list_collections <- function(engine) {
  check_engine(engine)
  res <- .Call(C_simvec_list_collections, engine$ptr)
  out <- from_json(payload_or_raise(res))
  if (is.null(out)) {
    return(character())
  }
  as.character(out)
}

#' @export
simvec_collection_info <- function(engine, name) {
  check_engine(engine)
  res <- .Call(C_simvec_collection_info, engine$ptr, tostring(name, "name"))
  from_json(payload_or_raise(res))
}

#' @export
simvec_collection_config <- function(engine, name) {
  check_engine(engine)
  res <- .Call(C_simvec_collection_config, engine$ptr, tostring(name, "name"))
  from_json(payload_or_raise(res))
}

#' @export
simvec_upsert <- function(engine, collection, id, vector, payload = NULL) {
  check_engine(engine)
  res <- .Call(C_simvec_upsert, engine$ptr,
    tostring(collection, "collection"), tostring(id, "id"),
    as.numeric(vector), to_json_or_null(payload))
  raise_if_failed(res)
}

#' @export
simvec_upsert_batch <- function(engine, collection, points) {
  check_engine(engine)
  res <- .Call(C_simvec_upsert_batch, engine$ptr,
    tostring(collection, "collection"), to_json_or_null(points))
  raise_if_failed(res)
}

#' @export
simvec_delete <- function(engine, collection, id) {
  check_engine(engine)
  res <- .Call(C_simvec_delete, engine$ptr,
    tostring(collection, "collection"), tostring(id, "id"))
  raise_if_failed(res)
}

#' @export
simvec_delete_batch <- function(engine, collection, ids) {
  check_engine(engine)
  res <- .Call(C_simvec_delete_batch, engine$ptr,
    tostring(collection, "collection"), to_json_or_null(as.list(as.character(ids))))
  raise_if_failed(res)
}

#' @export
simvec_delete_by_filter <- function(engine, collection, filter) {
  check_engine(engine)
  res <- .Call(C_simvec_delete_by_filter, engine$ptr,
    tostring(collection, "collection"), to_json_or_null(filter))
  as.numeric(payload_or_raise(res))
}

#' @export
simvec_set_payload <- function(engine, collection, id, payload) {
  check_engine(engine)
  res <- .Call(C_simvec_set_payload, engine$ptr,
    tostring(collection, "collection"), tostring(id, "id"),
    to_json_or_null(payload))
  raise_if_failed(res)
}

#' @export
simvec_merge_payload <- function(engine, collection, id, fields) {
  check_engine(engine)
  res <- .Call(C_simvec_merge_payload, engine$ptr,
    tostring(collection, "collection"), tostring(id, "id"),
    to_json_or_null(fields))
  raise_if_failed(res)
}

#' @export
simvec_delete_payload_keys <- function(engine, collection, id, keys) {
  check_engine(engine)
  res <- .Call(C_simvec_delete_payload_keys, engine$ptr,
    tostring(collection, "collection"), tostring(id, "id"),
    to_json_or_null(as.list(as.character(keys))))
  raise_if_failed(res)
}

#' @export
simvec_set_payload_field <- function(engine, collection, id, key, value) {
  check_engine(engine)
  res <- .Call(C_simvec_set_payload_field, engine$ptr,
    tostring(collection, "collection"), tostring(id, "id"),
    tostring(key, "key"), to_json_or_null(value))
  raise_if_failed(res)
}

#' @export
simvec_apply_txn <- function(engine, ops) {
  check_engine(engine)
  res <- .Call(C_simvec_apply_txn, engine$ptr, to_json_or_null(ops))
  raise_if_failed(res)
}

#' @export
simvec_create_payload_index <- function(engine, collection, field, schema) {
  check_engine(engine)
  res <- .Call(C_simvec_create_payload_index, engine$ptr,
    tostring(collection, "collection"), tostring(field, "field"),
    tostring(schema, "schema"))
  raise_if_failed(res)
}

#' @export
simvec_delete_payload_index <- function(engine, collection, field) {
  check_engine(engine)
  res <- .Call(C_simvec_delete_payload_index, engine$ptr,
    tostring(collection, "collection"), tostring(field, "field"))
  isTRUE(payload_or_raise(res))
}

#' @export
simvec_list_payload_indexes <- function(engine, collection) {
  check_engine(engine)
  res <- .Call(C_simvec_list_payload_indexes, engine$ptr,
    tostring(collection, "collection"))
  out <- from_json(payload_or_raise(res))
  if (is.null(out)) {
    return(list())
  }
  out
}

#' @export
simvec_search <- function(engine, collection, query) {
  check_engine(engine)
  res <- .Call(C_simvec_search, engine$ptr,
    tostring(collection, "collection"), to_json_or_null(query))
  out <- from_json(payload_or_raise(res))
  if (is.null(out)) {
    return(list())
  }
  out
}

#' @export
simvec_search_vector <- function(engine, collection, vector, limit, filter = NULL,
                                 score_threshold = -1, with_payload = TRUE,
                                 with_vector = FALSE) {
  check_engine(engine)
  res <- .Call(C_simvec_search_vector, engine$ptr,
    tostring(collection, "collection"), as.numeric(vector), as.numeric(limit),
    to_json_or_null(filter), as.numeric(score_threshold),
    isTRUE(with_payload), isTRUE(with_vector))
  out <- from_json(payload_or_raise(res))
  if (is.null(out)) {
    return(list())
  }
  out
}

#' @export
simvec_get <- function(engine, collection, id) {
  check_engine(engine)
  res <- .Call(C_simvec_get, engine$ptr,
    tostring(collection, "collection"), tostring(id, "id"))
  from_json(payload_or_raise(res))
}

#' @export
simvec_count <- function(engine, collection) {
  check_engine(engine)
  res <- .Call(C_simvec_count, engine$ptr, tostring(collection, "collection"))
  as.numeric(payload_or_raise(res))
}

#' @export
simvec_count_by_filter <- function(engine, collection, filter) {
  check_engine(engine)
  res <- .Call(C_simvec_count_by_filter, engine$ptr,
    tostring(collection, "collection"), to_json_or_null(filter))
  as.numeric(payload_or_raise(res))
}

#' @export
simvec_scroll <- function(engine, collection, limit, offset = NULL,
                          with_payload = TRUE, with_vector = FALSE) {
  check_engine(engine)
  res <- .Call(C_simvec_scroll, engine$ptr,
    tostring(collection, "collection"), as.numeric(limit),
    tostring(offset, "offset"), isTRUE(with_payload), isTRUE(with_vector))
  from_json(payload_or_raise(res))
}

#' @export
simvec_scroll_by_filter <- function(engine, collection, filter, limit, offset = NULL,
                                    with_payload = TRUE, with_vector = FALSE) {
  check_engine(engine)
  res <- .Call(C_simvec_scroll_by_filter, engine$ptr,
    tostring(collection, "collection"), as.numeric(limit),
    tostring(offset, "offset"), to_json_or_null(filter),
    isTRUE(with_payload), isTRUE(with_vector))
  from_json(payload_or_raise(res))
}

#' @export
simvec_build_index <- function(engine, collection) {
  check_engine(engine)
  res <- .Call(C_simvec_build_index, engine$ptr,
    tostring(collection, "collection"))
  isTRUE(payload_or_raise(res))
}

#' @export
simvec_build_quantization <- function(engine, collection) {
  check_engine(engine)
  res <- .Call(C_simvec_build_quantization, engine$ptr,
    tostring(collection, "collection"))
  isTRUE(payload_or_raise(res))
}

#' @export
simvec_has_index <- function(engine, collection) {
  check_engine(engine)
  res <- .Call(C_simvec_has_index, engine$ptr,
    tostring(collection, "collection"))
  isTRUE(payload_or_raise(res))
}

#' @export
simvec_has_quantization <- function(engine, collection) {
  check_engine(engine)
  res <- .Call(C_simvec_has_quantization, engine$ptr,
    tostring(collection, "collection"))
  isTRUE(payload_or_raise(res))
}

#' @export
simvec_drop_index <- function(engine, collection) {
  check_engine(engine)
  res <- .Call(C_simvec_drop_index, engine$ptr,
    tostring(collection, "collection"))
  raise_if_failed(res)
}

#' @export
simvec_drop_published_index <- function(engine, collection) {
  check_engine(engine)
  res <- .Call(C_simvec_drop_published_index, engine$ptr,
    tostring(collection, "collection"))
  raise_if_failed(res)
}

#' @export
simvec_compact <- function(engine, collection) {
  check_engine(engine)
  res <- .Call(C_simvec_compact, engine$ptr,
    tostring(collection, "collection"))
  as.numeric(payload_or_raise(res))
}

#' @export
simvec_drain_pending <- function(engine, collection) {
  check_engine(engine)
  res <- .Call(C_simvec_drain_pending, engine$ptr,
    tostring(collection, "collection"))
  raise_if_failed(res)
}

#' @export
simvec_collection_metrics <- function(engine, collection) {
  check_engine(engine)
  res <- .Call(C_simvec_collection_metrics, engine$ptr,
    tostring(collection, "collection"))
  from_json(payload_or_raise(res))
}

#' @export
simvec_run_maintenance_sweep <- function(engine) {
  check_engine(engine)
  res <- .Call(C_simvec_run_maintenance_sweep, engine$ptr)
  raise_if_failed(res)
}

#' @export
simvec_promote_temp_collection <- function(engine, live, temp, backup_suffix) {
  check_engine(engine)
  res <- .Call(C_simvec_promote_temp_collection, engine$ptr,
    tostring(live, "live"), tostring(temp, "temp"),
    tostring(backup_suffix, "backup_suffix"))
  raise_if_failed(res)
}

#' @export
simvec_restore_promote_backup <- function(engine, live) {
  check_engine(engine)
  res <- .Call(C_simvec_restore_promote_backup, engine$ptr,
    tostring(live, "live"))
  from_json(payload_or_raise(res))
}

#' @export
simvec_list_temp_collections <- function(engine) {
  check_engine(engine)
  res <- .Call(C_simvec_list_temp_collections, engine$ptr)
  out <- from_json(payload_or_raise(res))
  if (is.null(out)) {
    return(character())
  }
  as.character(out)
}

#' @export
simvec_list_old_backups <- function(engine, live) {
  check_engine(engine)
  res <- .Call(C_simvec_list_old_backups, engine$ptr, tostring(live, "live"))
  out <- from_json(payload_or_raise(res))
  if (is.null(out)) {
    return(character())
  }
  as.character(out)
}

#' @export
simvec_set_default_hnsw_config <- function(engine, config) {
  check_engine(engine)
  res <- .Call(C_simvec_set_default_hnsw_config, engine$ptr,
    to_json_or_null(config))
  raise_if_failed(res)
}

#' @export
simvec_get_default_hnsw_config <- function(engine) {
  check_engine(engine)
  res <- .Call(C_simvec_get_default_hnsw_config, engine$ptr)
  from_json(payload_or_raise(res))
}

#' @export
simvec_set_default_ivf_config <- function(engine, config) {
  check_engine(engine)
  res <- .Call(C_simvec_set_default_ivf_config, engine$ptr,
    to_json_or_null(config))
  raise_if_failed(res)
}

#' @export
simvec_get_default_ivf_config <- function(engine) {
  check_engine(engine)
  res <- .Call(C_simvec_get_default_ivf_config, engine$ptr)
  from_json(payload_or_raise(res))
}

#' @export
simvec_set_default_quantization_config <- function(engine, config) {
  check_engine(engine)
  res <- .Call(C_simvec_set_default_quantization_config, engine$ptr,
    to_json_or_null(config))
  raise_if_failed(res)
}

#' @export
simvec_get_default_quantization_config <- function(engine) {
  check_engine(engine)
  res <- .Call(C_simvec_get_default_quantization_config, engine$ptr)
  from_json(payload_or_raise(res))
}
