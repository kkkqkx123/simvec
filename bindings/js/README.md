# simvec JavaScript binding (node:ffi over libsimvec)

Zero-dependency wrapper around the simvec C API using the Node.js built-in
`node:ffi` module. The package ships no native code: build the Rust dynamic
library first and point the wrapper at it.

```sh
cargo build --features c_api
SIMVEC_LIB=/home/kkkqkx/code/simvec/target/debug/libsimvec.so node --experimental-ffi --test bindings/js/tests/
```

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
| `setPayload` / `mergePayload` / `deletePayloadKeys` | `simvec_set_payload` et al |
| `createPayloadIndex` / `deletePayloadIndex` / `listPayloadIndexes` | `simvec_create_payload_index` et al |
| `search` / `searchVector` / `get` / `count` / `scroll` | `simvec_search` et al |
| `buildIndex` / `buildQuantization` / `hasIndex` / `hasQuantization` / `dropIndex` / `compact` / `drainPending` / `collectionMetrics` / `runMaintenanceSweep` | `simvec_build_index` et al |

Non-zero status codes throw typed errors from `src/errors.js` mirroring the
Python binding: `NotFoundError`, `AlreadyExistsError`, `InvalidConfigError`,
`InvalidDimensionError`, `MisuseError`, `CapacityExceededError`,
`FilterError`, all extending `SimvecError`.

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
work and intentionally out of scope for this binding revision.
