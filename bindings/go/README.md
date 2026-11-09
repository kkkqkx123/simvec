# simvec Go binding (cgo over libsimvec)

Thin Go wrapper around the simvec C API. The package ships no native
code: build the Rust dynamic library first, then build or test the Go
module against it. A C toolchain with cgo enabled is required.

```sh
cargo build --features c_api
cd bindings/go && go test ./...
```

The cgo directives point at `../../include/simvec.h` for declarations and
at the `target/debug` and `target/release` outputs under the workspace root
for linking, with matching run paths embedded. Against a custom library
location, override with `CGO_CFLAGS`, `CGO_LDFLAGS`, or `LD_LIBRARY_PATH`
instead of relying on the default search paths. A link failure names the
missing library rather than a list of tried paths, which is the one place
where this binding cannot mirror the dynamic loaders of the other two
bindings.

Configs, filters, payloads and queries are plain maps or serializable
values encoded to the JSON shapes the Rust side already understands.
Vectors are `[]float32` slices crossing the boundary as contiguous arrays;
use the `Float32s` helper to convert `[]float64` callers, noting the
narrowing conversion. Point ids are strings; convert integer ids with
`strconv.Itoa` before calling.

Method names follow Go exported camel case and map one-to-one onto the C API:

| Go | C API |
|---|---|
| `CreateCollection` / `DeleteCollection` / `CollectionExists` / `ListCollections` / `CollectionInfo` / `CollectionConfig` | `simvec_create_collection` et al |
| `Upsert` / `UpsertBatch` / `Delete` / `DeleteBatch` / `DeleteByFilter` | `simvec_upsert` et al |
| `SetPayload` / `MergePayload` / `SetPayloadField` / `ApplyTxn` / `DeletePayloadKeys` | `simvec_set_payload` et al |
| `CreatePayloadIndex` / `DeletePayloadIndex` / `ListPayloadIndexes` | `simvec_create_payload_index` et al |
| `Search` / `SearchVector` / `Get` / `Count` / `Scroll` | `simvec_search` et al |
| `BuildIndex` / `BuildQuantization` / `HasIndex` / `HasQuantization` / `DropIndex` / `DropPublishedIndex` / `Compact` / `DrainPending` / `CollectionMetrics` / `RunMaintenanceSweep` | `simvec_build_*` et al |
| `PromoteTempCollection` / `RestorePromoteBackup` / `ListTempCollections` / `ListOldBackups` | `simvec_promote_temp_collection` et al |
| `SetDefaultHnswConfig` / `GetDefaultHnswConfig` / `SetDefaultIvfConfig` / `GetDefaultIvfConfig` / `SetDefaultQuantizationConfig` / `GetDefaultQuantizationConfig` | `simvec_set_default_*` et al |

Error mapping: seven status codes return typed errors wrapping
`SimvecError` (`NotFoundError`, `AlreadyExistsError`,
`InvalidConfigError`, `InvalidDimensionError`, `MisuseError`,
`CapacityExceededError`, `FilterError`). All other non-zero codes
(generic error, internal error, out of memory, IO error, corrupt data,
non-finite element) return the `SimvecError` base with the numeric code
kept in the message. Use the `IsNotFound` family or `CodeOf` to branch
without parsing message text. The `LastCode` and `LastMessage` accessors
expose the most recent native status without returning an error, useful
for logging around handled failures.

`Engine` instances are safe for concurrent use except that `Close` must
not run concurrently with in-flight calls. `Close` is idempotent, and a
finalizer closes engines the caller forgot to close, but explicit closing
is still expected:

```go
engine, err := simvec.NewEngine("./data")
if err != nil {
    log.Fatal(err)
}
defer engine.Close()

err = engine.CreateCollection("docs", map[string]any{"vector_size": 4, "distance": "Cosine"})
```

Bundling the shared library into a distributable module is follow-up
distribution work and intentionally out of scope for this binding revision.
Until then the recommended pairing is the `cargo build --features c_api`
output above against the same checkout, keeping binding and library from
the same source.
