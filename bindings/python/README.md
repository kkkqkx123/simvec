# simvec Python binding (ctypes over libsimvec)

Pure-Python wrapper around the simvec C API. The package ships no native
code: build the Rust dynamic library first and point the wrapper at it.

```sh
cargo build --features c_api
SIMVEC_LIB=$PWD/target/debug/libsimvec.so python -m pytest bindings/python/tests -q
```

The loader resolves the library by checking the `SIMVEC_LIB` environment
variable first, then the `target/debug` and `target/release` outputs under
the workspace root. A load failure raises with the list of tried paths.

Configs, filters, payloads and queries are plain dicts serialized to the
JSON shapes the Rust side already understands. Vectors accept any
sequence of floats and cross the boundary as contiguous float32 arrays;
numpy arrays are used in place when available, otherwise the sequence is
copied into a temporary float32 buffer held for the duration of the call.

Method names map one-to-one onto the C API:

| Python | C API |
|---|---|
| `create_collection` / `delete_collection` / `collection_exists` / `list_collections` / `collection_info` / `collection_config` | `simvec_create_collection` et al |
| `upsert` / `upsert_batch` / `delete` / `delete_batch` / `delete_by_filter` | `simvec_upsert` et al |
| `set_payload` / `merge_payload` / `set_payload_field` / `apply_txn` / `delete_payload_keys` | `simvec_set_payload` et al |
| `create_payload_index` / `delete_payload_index` / `list_payload_indexes` | `simvec_create_payload_index` et al |
| `search` / `search_vector` / `get` / `count` / `scroll` | `simvec_search` et al |
| `build_index` / `build_quantization` / `has_index` / `has_quantization` / `drop_index` / `drop_published_index` / `compact` / `drain_pending` / `collection_metrics` / `run_maintenance_sweep` | `simvec_build_index` et al |
| `promote_temp_collection` / `restore_promote_backup` / `list_temp_collections` / `list_old_backups` | `simvec_promote_temp_collection` et al |
| `set_default_hnsw_config` / `get_default_hnsw_config` / `set_default_ivf_config` / `get_default_ivf_config` / `set_default_quantization_config` / `get_default_quantization_config` | `simvec_set_default_*` et al |

Error mapping: seven status codes raise typed subclasses of `SimvecError`
(`NotFoundError`, `AlreadyExistsError`, `InvalidConfigError`,
`InvalidDimensionError`, `MisuseError`, `CapacityExceededError`,
`FilterError`). All other non-zero codes (generic error, internal error,
out of memory, IO error, corrupt data, non-finite element) raise the
`SimvecError` base with the numeric code kept in the message.
The `last_code` and `last_message` properties expose the most recent
native status without raising, useful for logging around caught errors.

`Engine` supports context management and repeatable close:

```python
from simvec import Engine

with Engine("./data") as engine:
    engine.create_collection("docs", {"vector_size": 4, "distance": "Cosine"})
    engine.upsert("docs", "1", [1, 0, 0, 0], {"title": "hello"})
    print(engine.search_vector("docs", [1, 0, 0, 0], 5))
```

Type hints ship with the package (`py.typed`, `engine.pyi`).

Bundling the shared library into wheels is follow-up distribution work
and intentionally out of scope for this binding revision. Until then the
recommended pairing is the `SIMVEC_LIB` override above against a local
`cargo build --features c_api` output, keeping binding and library from
the same checkout.
