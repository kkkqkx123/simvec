# simvec-wasm

Browser binding for the simvec in-memory engine. Pure WebAssembly with no
native dynamic library and no filesystem: collections live for the page
session and disappear on reload unless the caller persists a snapshot.

Search is single-threaded linear exact scan for small and medium
collections; large collections stay server-side. There are no background
threads, no memory-mapped files, and no parallel runtime in this form.

Built from the same crate with the `wasm` feature. Complex values cross
as JSON text reusing the Rust serde shapes, vectors cross as
single-precision arrays. Scores follow the crate-wide higher-is-better
contract on every metric. Errors carry the same numeric codes as the
native bindings with matching typed classes.

## Build

Run `npm run build` inside this directory. It builds the workspace-root
crate (`../..`) with the `wasm` feature and emits `bindings/wasm/pkg/`
(`src/index.js` loads `../pkg/simvec.js`). The Rust side needs the
`wasm32-unknown-unknown` target installed. The npm package ships pure
JS only; `pkg/` is built by the caller and is not embedded in the
published package.

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
persistent directory and reads it back. Writes go through a temp file
with read-back verification plus move when the host supports it,
falling back to verified direct overwrite. The previous good snapshot
is retained as a `.bak` file. Missing platform interfaces fall
back to an explicit error with the snapshot kept in memory only.
Snapshot version is fixed at one with no history compatibility.

`revision` advances on every state-changing mutation so callers can
skip unchanged saves. `exportCollection` and `importCollection` persist
one collection without touching the others. `saveCollectionsToOpfs`
and `loadCollectionsFromOpfs` store one file per collection plus a
manifest under a directory. `snapshotExists`, `removeSnapshot`,
`estimateOpfsQuota`, `requestOpfsPersistence`, and `createOpfsAutoSave`
cover existence checks, cleanup, quota, and debounced saves with page
hide flushing.
