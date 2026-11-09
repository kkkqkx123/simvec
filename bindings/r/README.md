# simvec R binding (C over libsimvec)

Thin R binding around the simvec C API. The package ships no native
code: build the Rust dynamic library first, then install the R package
against it. A C toolchain is required.

```sh
cargo build --features c_api
R CMD INSTALL bindings/r
Rscript bindings/r/tests/smoke.R
```

`configure` resolves the library by checking the `SIMVEC_LIB` environment
variable first (naming the library file explicitly), then the
`target/debug` and `target/release` outputs under the workspace root. The
resolved directory is baked into the bridge link with a run path, so the
smoke script needs no extra environment. Against a custom location later,
override with `SIMVEC_LIB` at install time or `LD_LIBRARY_PATH` at run
time. A missing library fails `configure` with the tried locations.

Configs, filters, payloads and queries are plain lists serialized with
jsonlite to the JSON shapes the Rust side already understands. Vectors
are numeric vectors crossing the boundary as contiguous float32 arrays
with narrowing conversion; keep ids as strings and convert integer ids
with `as.character` before calling.

Function names are snake case and map one-to-one onto the C API:

| R | C API |
|---|---|
| `simvec_create_collection` / `simvec_delete_collection` / `simvec_collection_exists` / `simvec_list_collections` / `simvec_collection_info` / `simvec_collection_config` | `simvec_create_collection` et al |
| `simvec_upsert` / `simvec_upsert_batch` / `simvec_delete` / `simvec_delete_batch` / `simvec_delete_by_filter` | `simvec_upsert` et al |
| `simvec_set_payload` / `simvec_merge_payload` / `simvec_set_payload_field` / `simvec_apply_txn` / `simvec_delete_payload_keys` | `simvec_set_payload` et al |
| `simvec_create_payload_index` / `simvec_delete_payload_index` / `simvec_list_payload_indexes` | `simvec_create_payload_index` et al |
| `simvec_search` / `simvec_search_vector` / `simvec_get` / `simvec_count` / `simvec_scroll` | `simvec_search` et al |
| `simvec_build_index` / `simvec_build_quantization` / `simvec_has_index` / `simvec_has_quantization` / `simvec_drop_index` / `simvec_drop_published_index` / `simvec_compact` / `simvec_drain_pending` / `simvec_collection_metrics` / `simvec_run_maintenance_sweep` | `simvec_build_*` et al |
| `simvec_promote_temp_collection` / `simvec_restore_promote_backup` / `simvec_list_temp_collections` / `simvec_list_old_backups` | `simvec_promote_temp_collection` et al |
| `simvec_set_default_hnsw_config` / `simvec_get_default_hnsw_config` / `simvec_set_default_ivf_config` / `simvec_get_default_ivf_config` / `simvec_set_default_quantization_config` / `simvec_get_default_quantization_config` | `simvec_set_default_*` et al |

Error mapping: seven status codes signal typed conditions carrying the
`simvec_error` base class (`simvec_not_found`, `simvec_already_exists`,
`simvec_invalid_config`, `simvec_invalid_dimension`, `simvec_misuse`,
`simvec_capacity_exceeded`, `simvec_filter`). All other non-zero codes
(generic error, internal error, out of memory, IO error, corrupt data,
non-finite element) signal the `simvec_error` base with the numeric code
kept in the message and in the `code` field. Use `simvec_code()` or the
`is_not_found()` family to branch without parsing message text. The
`simvec_last_code()` and `simvec_last_message()` accessors expose the
most recent native status without signalling, useful for logging around
handled failures.

Engines are environments holding one external pointer. `simvec_close()`
is idempotent, and a finalizer closes engines the caller forgot to close,
but explicit closing is still expected:

```r
engine <- simvec_open("./data")
on.exit(simvec_close(engine))
simvec_create_collection(engine, "docs", list(vector_size = 4, distance = "Cosine"))
simvec_upsert(engine, "docs", "1", c(1, 0, 0, 0), list(title = "hello"))
print(simvec_search_vector(engine, "docs", c(1, 0, 0, 0), 5))
```

Notes: numeric vectors narrow from double to float32; counts cross as
doubles and stay exact below 2^53. Scalar string arguments must not be
`NA`; nested `NA` inside payloads serializes to JSON null. One-element
vectors inside batch JSON unbox to scalars under `auto_unbox`, so wrap
length-1 vectors in `I()` for single-dimension collections. The same
engine object must not be shared across parallel workers; use one engine
per process.

Bundling the shared library into a distributable package is follow-up
distribution work and intentionally out of scope for this binding
revision. Until then the recommended pairing is the `cargo build
--features c_api` output above against the same checkout, keeping
binding and library from the same source.
