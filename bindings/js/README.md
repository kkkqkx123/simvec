# simvec JavaScript binding (node:ffi over libsimvec)

Zero-dependency wrapper around the simvec C API using the Node.js built-in
`node:ffi` module. The package ships no native code: build the Rust dynamic
library first and point the wrapper at it.

```sh
cargo build --features c_api
SIMVEC_LIB=$PWD/target/debug/libsimvec.so node --experimental-ffi --test bindings/js/tests/
```

The loader resolves the library by checking the `SIMVEC_LIB` environment
variable first, then the `target/debug` and `target/release` outputs under
the workspace root. A load failure throws with the list of tried paths.

`node:ffi` is experimental and requires the `--experimental-ffi` flag
(Node 22+). If permission flags are in use, also pass `--allow-ffi`.

Configs, filters, payloads and queries are plain objects serialized to the
JSON shapes the Rust side already understands. Vectors accept any float
sequence and cross the boundary as contiguous float32 arrays (`Float32Array`
is used as-is, other sequences are copied).

Method names follow JavaScript camelCase and map one-to-one onto the C API:

| JavaScript | C API |
|---|---|
| `createCollection` / `deleteCollection` / `collectionExists` / `listCollections` / `collectionInfo` / `collectionConfig` | `simvec_create_collection` et al |
| `upsert` / `upsertBatch` / `delete` / `deleteBatch` / `deleteByFilter` | `simvec_upsert` et al |
| `setPayload` / `mergePayload` / `setPayloadField` / `applyTxn` / `deletePayloadKeys` | `simvec_set_payload` et al |
| `createPayloadIndex` / `deletePayloadIndex` / `listPayloadIndexes` | `simvec_create_payload_index` et al |
| `search` / `searchVector` / `get` / `count` / `scroll` | `simvec_search` et al |
| `buildIndex` / `buildQuantization` / `hasIndex` / `hasQuantization` / `dropIndex` / `dropPublishedIndex` / `compact` / `drainPending` / `collectionMetrics` / `runMaintenanceSweep` | `simvec_build_index` et al |
| `promoteTempCollection` / `restorePromoteBackup` / `listTempCollections` / `listOldBackups` | `simvec_promote_temp_collection` et al |
| `setDefaultHnswConfig` / `getDefaultHnswConfig` / `setDefaultIvfConfig` / `getDefaultIvfConfig` / `setDefaultQuantizationConfig` / `getDefaultQuantizationConfig` | `simvec_set_default_*` et al |

Error mapping: seven status codes throw typed subclasses of `SimvecError`
(`NotFoundError`, `AlreadyExistsError`, `InvalidConfigError`,
`InvalidDimensionError`, `MisuseError`, `CapacityExceededError`,
`FilterError`). All other non-zero codes (generic error, internal error,
out of memory, IO error, corrupt data, non-finite element) throw the
`SimvecError` base with the numeric code kept in the message.
The `lastCode` and `lastMessage` getters expose the most recent native
status without throwing, useful for logging around caught errors.

Type declarations ship with the package (`src/engine.d.ts`,
`src/errors.d.ts`).

`Engine` supports explicit resource management:

```js
import { Engine } from "simvec";

{
  using engine = new Engine("./data");
  engine.createCollection("docs", { vector_size: 4, distance: "Cosine" });
  engine.upsert("docs", "1", [1, 0, 0, 0], { title: "hello" });
  console.log(engine.searchVector("docs", [1, 0, 0, 0], 5));
}
```

Or close manually with `engine.close()`, which is safe to call more than once.

Bundling the shared library into an npm package is follow-up distribution
work and intentionally out of scope for this binding revision. Until then the
recommended pairing is the `SIMVEC_LIB` override above against a local
`cargo build --features c_api` output, keeping binding and library from
the same checkout.
