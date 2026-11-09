# simvec-wasm

Browser binding for the simvec in-memory engine. Pure WebAssembly with no
native dynamic library and no filesystem: collections live for the page
session and disappear on reload unless the caller persists a snapshot.

Built from the same crate with the `wasm` feature. Complex values cross
as JSON text reusing the Rust serde shapes, vectors cross as
single-precision arrays. Scores follow the crate-wide higher-is-better
contract on every metric. Errors carry the same numeric codes as the
native bindings with matching typed classes.

## Build

Run `npm run build` inside this directory. It calls `wasm-pack` against
the workspace root with the `wasm` feature and emits `pkg/`. The Rust
side needs the `wasm32-unknown-unknown` target installed.

## Use

Construct one engine per page with the async factory, create a
collection, upsert points, and search. Payloads, filters, and queries are
plain objects. Collection listing, info, config, query results, and index
lists return parsed structures. Counts return host numbers. Scroll returns
a page object with the point array and the next offset cursor.

```js
const engine = await Engine.create();
engine.createCollection("docs", { vector_size: 2, distance: "Dot" });
engine.upsert("docs", "a", [1, 0], { color: "red" });
const hits = engine.searchVector("docs", [1, 0], 10);
engine.close();
```

Full chain coverage mirrors the native bindings: collection management,
writes, deletes, filter deletes, transactions, payload merge, indexes,
search, scroll, maintenance no-ops, metrics, and snapshot roundtrip.
See `tests/smoke.mjs` for the reference flow.

Index building, quantization, compaction, and maintenance are compatible
no-ops in this form and always report not-effective. Publish, backup,
and engine-default configuration groups are not exposed here.

## Persistence

The engine holds no file handle and runs no background sync. Callers
persist explicitly with `exportSnapshot` and `importSnapshot`. The
optional `persist.js` helper writes the snapshot to the browser
persistent directory and reads it back. Missing platform interfaces fall
back to an explicit error with the snapshot kept in memory only.
Snapshot version is fixed at one with no history compatibility.
