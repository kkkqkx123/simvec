library(simvec)

fail <- function(msg) {
  stop(sprintf("smoke failure: %s", msg))
}

expect_equal <- function(actual, expected, what) {
  if (!isTRUE(all.equal(actual, expected, check.attributes = FALSE))) {
    fail(sprintf("%s: got %s, want %s",
      what, paste(deparse(actual), collapse = " "),
      paste(deparse(expected), collapse = " ")))
  }
}

expect_true <- function(value, what) {
  if (!isTRUE(value)) {
    fail(what)
  }
}

expect_false <- function(value, what) {
  if (!isFALSE(value)) {
    fail(what)
  }
}

flat_config <- function(dims) {
  list(vector_size = dims, distance = "Cosine", index = "Flat")
}

# -- lifecycle --

root <- tempfile("simvec-r-")
engine <- simvec_open(root)
expect_true(simvec_collection_exists(engine, "docs") == FALSE, "fresh engine has no docs")

simvec_create_collection(engine, "docs", flat_config(4))
expect_true(simvec_collection_exists(engine, "docs"), "collection exists after creation")
expect_equal(simvec_list_collections(engine), "docs", "collection list")

simvec_upsert(engine, "docs", "1", c(1, 0, 0, 0), list(title = "hello"))
simvec_upsert_batch(engine, "docs", list(
  list(id = "2", vector = c(0, 1, 0, 0), payload = list(title = "world"))))
expect_equal(simvec_count(engine, "docs"), 2, "count after writes")
expect_equal(simvec_last_code(engine), 0, "last code is zero on success")

hits <- simvec_search_vector(engine, "docs", c(1, 0, 0, 0), 5)
expect_equal(length(hits), 2, "search vector hit count")
expect_equal(as.character(hits[[1]]$id), "1", "top hit id")
expect_equal(hits[[1]]$payload$title, "hello", "top hit payload")

query_hits <- simvec_search(engine, "docs", list(vector = c(1, 0, 0, 0), limit = 5))
expect_equal(length(query_hits), 2, "query hit count")

point <- simvec_get(engine, "docs", "1")
expect_equal(length(point$vector), 4, "point vector length")

simvec_merge_payload(engine, "docs", "1", list(lang = "en"))
expect_equal(simvec_get(engine, "docs", "1")$payload$lang, "en", "merged payload field")
simvec_set_payload_field(engine, "docs", "1", "topic", "intro")
expect_equal(simvec_get(engine, "docs", "1")$payload$topic, "intro", "payload field write")
simvec_delete_payload_keys(engine, "docs", "1", "topic")
expect_true(is.null(simvec_get(engine, "docs", "1")$payload$topic), "payload field delete")

simvec_create_payload_index(engine, "docs", "title", "keyword")
indexes <- simvec_list_payload_indexes(engine, "docs")
expect_equal(length(indexes), 1, "payload index count")
expect_equal(indexes[[1]]$field, "title", "payload index field")
expect_true(simvec_delete_payload_index(engine, "docs", "title"), "delete index hits")
expect_false(simvec_delete_payload_index(engine, "docs", "title"), "repeated delete misses")

page <- simvec_scroll(engine, "docs", 10)
expect_equal(length(page$points), 2, "scroll page size")
expect_equal(page$next_offset, "2", "scroll cursor")
tail <- simvec_scroll(engine, "docs", 10, offset = page$next_offset)
expect_equal(length(tail$points), 0, "scroll tail is empty")

expect_equal(simvec_collection_info(engine, "docs")$vector_count, 2, "collection info count")
expect_equal(simvec_collection_config(engine, "docs")$vector_size, 4, "collection config size")
expect_true(!is.null(simvec_collection_metrics(engine, "docs")), "metrics document")

expect_false(simvec_has_index(engine, "docs"), "no index before build")
expect_false(simvec_has_quantization(engine, "docs"), "no quantization before build")
invisible(simvec_build_index(engine, "docs"))
invisible(simvec_build_quantization(engine, "docs"))
simvec_drain_pending(engine, "docs")
simvec_drop_published_index(engine, "docs")
simvec_drop_index(engine, "docs")
simvec_run_maintenance_sweep(engine)

simvec_delete(engine, "docs", "2")
expect_equal(simvec_count(engine, "docs"), 1, "count after delete")
not_found <- tryCatch({ simvec_get(engine, "docs", "2"); NULL }, error = function(e) e)
expect_true(is_not_found(not_found), "missing point raises not-found")
expect_equal(simvec_code(not_found), 12, "not-found code")
expect_true(simvec_last_code(engine) != 0, "last code is nonzero after failure")
expect_true(nzchar(simvec_last_message(engine)), "last message is nonempty after failure")

simvec_delete_collection(engine, "docs")
expect_false(simvec_collection_exists(engine, "docs"), "collection gone after deletion")

# -- batch and filtered deletes --

simvec_create_collection(engine, "batch", flat_config(3))
simvec_upsert_batch(engine, "batch", list(
  list(id = "a", vector = c(1, 0, 0), payload = list(color = "red")),
  list(id = "b", vector = c(0, 1, 0), payload = list(color = "blue")),
  list(id = "c", vector = c(0, 0, 1), payload = list(color = "red"))))
expect_equal(simvec_count(engine, "batch"), 3, "batch count")

simvec_apply_txn(engine, list(
  list(op = "upsert", collection = "batch",
       point = list(id = "d", vector = c(1, 1, 0))),
  list(op = "delete", collection = "batch", id = "d")))
expect_equal(simvec_count(engine, "batch"), 3, "count after txn")

deleted <- simvec_delete_by_filter(engine, "batch",
  list(must = list(list(field = "color",
    condition = list(Match = list(value = "red"))))))
expect_equal(deleted, 2, "filtered delete count")
expect_equal(simvec_count(engine, "batch"), 1, "count after filtered delete")

simvec_delete_batch(engine, "batch", "b")
expect_equal(simvec_count(engine, "batch"), 0, "count after batch delete")
invisible(simvec_compact(engine, "batch"))

simvec_delete_collection(engine, "batch")
gone <- tryCatch({ simvec_get(engine, "batch", "b"); NULL }, error = function(e) e)
expect_true(is_not_found(gone), "deleted collection reads as not-found")

# -- promote and defaults --

simvec_create_collection(engine, "live", flat_config(2))
simvec_upsert(engine, "live", "1", c(1, 0))
simvec_create_collection(engine, "live.rebuild-7", flat_config(2))
simvec_upsert(engine, "live.rebuild-7", "2", c(0, 1))
expect_equal(simvec_list_temp_collections(engine), "live.rebuild-7", "temp list")

simvec_promote_temp_collection(engine, "live", "live.rebuild-7", ".old-7")
expect_equal(simvec_get(engine, "live", "2")$vector, list(0, 1), "promoted point")
expect_equal(simvec_list_old_backups(engine, "live"), "live.old-7", "backup list")
expect_equal(simvec_restore_promote_backup(engine, "live"), "live.old-7", "restore backup")
expect_equal(simvec_get(engine, "live", "1")$vector, list(1, 0), "restored point")
simvec_delete_collection(engine, "live")

expect_true(is.null(simvec_get_default_hnsw_config(engine)), "hnsw default unset")
expect_true(is.null(simvec_get_default_ivf_config(engine)), "ivf default unset")
expect_true(is.null(simvec_get_default_quantization_config(engine)), "quant default unset")
simvec_set_default_hnsw_config(engine, list(m = 8, ef_construct = 32))
simvec_set_default_ivf_config(engine, list(
  min_build_points = 1000, sample_limit = 1024, kmeans_max_iter = 10,
  drift_threshold = 0.1, drift_check_interval = 1000, default_nprobe = 4,
  auto_promotion = FALSE))
simvec_set_default_quantization_config(engine,
  list(quant_type = list(Binary = list(always_ram = TRUE))))
expect_equal(simvec_get_default_hnsw_config(engine)$m, 8, "hnsw default round trip")
expect_true(!is.null(simvec_get_default_ivf_config(engine)), "ivf default round trip")
expect_true(!is.null(simvec_get_default_quantization_config(engine)),
  "quant default round trip")

simvec_close(engine)
simvec_close(engine)
expect_true(simvec_closed(engine), "engine reports closed")
closed_err <- tryCatch({ simvec_count(engine, "docs"); NULL }, error = function(e) e)
expect_true(is_misuse(closed_err), "use after close is misuse")

cat("R smoke: OK\n")
