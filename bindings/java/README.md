# simvec Java binding (JNI over libsimvec)

Thin Java wrapper around the simvec C API. The package ships no native
code: build the Rust dynamic library and the JNI bridge first, then build
or test the Java package against them. A C toolchain with the JDK headers
is required for the bridge.

```sh
cargo build --features c_api
sh bindings/java/scripts/build-native.sh
cd bindings/java && mvn test
```

The loader resolves the bridge by checking the `simvec.jni.lib` system
property first, then the `SIMVEC_JNI_LIB` environment variable, then the
`target/debug` and `target/release` outputs under the workspace root, and
finally `java.library.path`. When `SIMVEC_LIB` names an explicit Rust
library path, that file is preloaded so the bridge resolves against it.
A load failure names the missing library with the list of tried paths.

Configs, filters, payloads and queries are plain maps or lists encoded to
the JSON shapes the Rust side already understands. Vectors are `float[]`
slices crossing the boundary as contiguous arrays; `double[]` overloads
copy with narrowing conversion. Point ids are strings; convert integer
ids with `String.valueOf` before calling.

Method names follow Java camel case and map one-to-one onto the C API:

| Java | C API |
|---|---|
| `createCollection` / `deleteCollection` / `collectionExists` / `listCollections` / `collectionInfo` / `collectionConfig` | `simvec_create_collection` et al |
| `upsert` / `upsertBatch` / `delete` / `deleteBatch` / `deleteByFilter` | `simvec_upsert` et al |
| `setPayload` / `mergePayload` / `setPayloadField` / `applyTxn` / `deletePayloadKeys` | `simvec_set_payload` et al |
| `createPayloadIndex` / `deletePayloadIndex` / `listPayloadIndexes` | `simvec_create_payload_index` et al |
| `search` / `searchVector` / `get` / `count` / `scroll` | `simvec_search` et al |
| `buildIndex` / `buildQuantization` / `hasIndex` / `hasQuantization` / `dropIndex` / `dropPublishedIndex` / `compact` / `drainPending` / `collectionMetrics` / `runMaintenanceSweep` | `simvec_build_*` et al |
| `promoteTempCollection` / `restorePromoteBackup` / `listTempCollections` / `listOldBackups` | `simvec_promote_temp_collection` et al |
| `setDefaultHnswConfig` / `getDefaultHnswConfig` / `setDefaultIvfConfig` / `getDefaultIvfConfig` / `setDefaultQuantizationConfig` / `getDefaultQuantizationConfig` | `simvec_set_default_*` et al |

Error mapping: seven status codes throw typed subclasses of
`SimvecError` (`NotFoundError`, `AlreadyExistsError`,
`InvalidConfigError`, `InvalidDimensionError`, `MisuseError`,
`CapacityExceededError`, `FilterError`). All other non-zero codes
(generic error, internal error, out of memory, IO error, corrupt data,
non-finite element) throw the `SimvecError` base with the numeric code
kept in the message and in `getCode()`. Use the `isNotFound` family or
`codeOf` to branch without parsing message text. The `lastCode` and
`lastMessage` accessors expose the most recent native status without
throwing, useful for logging around handled failures.

`Engine` instances are safe for concurrent use; every method is
synchronized, so concurrent searches serialize. `close()` is idempotent,
and a cleaner closes engines the caller forgot to close, but explicit
closing (preferably try-with-resources) is still expected:

```java
try (Engine engine = new Engine("./data")) {
    engine.createCollection("docs", Map.of("vector_size", 4, "distance", "Cosine"));
    engine.upsert("docs", "1", new float[]{1, 0, 0, 0}, Map.of("title", "hello"));
    System.out.println(engine.searchVector("docs", new float[]{1, 0, 0, 0}, 5));
}
```

Bundling the native libraries into a distributable artifact is follow-up
distribution work and intentionally out of scope for this binding revision.
Until then the recommended pairing is the `cargo build --features c_api`
plus `scripts/build-native.sh` outputs above against the same checkout,
keeping binding, bridge, and library from the same source.
