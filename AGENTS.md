# AGENTS.md — simvec

Local vector search engine in Rust: HNSW-first ANN with exact-scan fallback, WAL durability, mmap storage, plus a C ABI (`--features c_api`) that all language bindings wrap. `crate-type = ["cdylib", "staticlib", "rlib"]`, rust-version 1.97, edition 2021.

## Commands

- `cargo build` — plain rlib (no C ABI). `cargo build --features c_api` — C ABI / dynamic library (required before any binding test).
- `cargo test -p simvec` — unit + integration tests.
- `cargo bench -p simvec` — criterion benches. `QUICK=1 cargo bench` — seconds-long smoke gear; QUICK vs full numbers (and numbers across machines) are never comparable.
- No rustfmt.toml / clippy.toml — defaults apply.
- `scripts/publish.sh [--dry-run]` — crates.io publish.

## Layout

`src/engine` (LocalVectorEngine, maintenance worker) · `src/storage` (CollectionStore: mmap vectors, segmented WAL, payloads, tombstones, search; plus `point_blobs` / `postings` used by the multi-space stores) · `src/index` (HNSW, IVF/kmeans, persist) · `src/distance` (AVX2/AVX-512/NEON/naive kernels) · `src/kind_store`, `src/named_store` (multi-space) · `src/c_api` + `include/simvec.h` · `src/memory_engine` + `src/wasm_api` (wasm feature) · `bindings/{python,js,go,java,r,wasm}` · `docs/binding-matrix.md` (capability matrix per binding, Chinese).

## Non-obvious rules

- **Test files are not auto-discovered.** Unit tests hang off modules via `#[cfg(test)] #[path = "..."] mod tests;`; top-level files are `src/<module>_tests.rs`. A new `*_tests.rs` does nothing until wired into its parent module with `#[path]`.
- **Score contract:** public scores are higher-is-better on every metric (conversion table in README); internal distances are lower-is-better. Cosine normalizes on the fly — `vectors.bin` must keep the original bytes.
- **`naive` is the correctness baseline.** New distance kernels must match naive within 1e-4. `simd_portable` currently delegates to naive (portable_simd is unstable).
- **`include/simvec.h` is committed.** `build.rs` regenerates it only when the `c_api` feature is on AND cbindgen is installed; otherwise the committed header is used silently. After changing `src/c_api`, rebuild with cbindgen present and commit the regenerated header, and update `docs/binding-matrix.md` plus the affected bindings.
- **Bindings:** every non-wasm binding is FFI over the built cdylib — `cargo build --features c_api` first, then point `SIMVEC_LIB` (or fallback paths under `target/debug|release`) at it. JS needs Node 22+ `--experimental-ffi`; Java needs `sh bindings/java/scripts/build-native.sh` then `mvn test`; R needs `R CMD INSTALL`; wasm needs the `wasm32-unknown-unknown` target and `npm run build` in `bindings/wasm`.
- **`target_arch = "wasm32"` compiles out** `engine`, `storage`, `index`, `kind_store`, `named_store`, `c_api` (see `src/lib.rs` cfgs) — native-only code must stay behind those gates.
- **On-disk layout (per collection).** Dense: `meta.bin`, `keys.bin`, `vectors.bin`. `kind` / `named` additionally use `points.bin` (append-only blob file: 4-byte-aligned records, 12-byte header, grown in 4 MiB chunks, truncated to the snapshot's `blobs_len` at open), `payloads.bin` (same blob format for JSON payloads, truncated to the snapshot's `payloads_len` at open or to zero when no snapshot is published), and `postings/` (bucket files keyed by `term >> 10`, records `[term u32][count u32][(slot u32, weight f32) * count]`, plus `manifest.bin`). Their snapshots keep only the `{slot, id, blob, payload_blob, norm}` directory; vectors and JSON payload bytes live in the blob files, vectors read through the zero-copy `VectorView<'a>`. A point's `payload_blob` is `None` exactly when its in-memory payload changed since the last checkpoint — the checkpoint appends new JSON only for those points.
- **The WAL is a directory.** `wal/<20-digit first-lsn>.bin`; a txn never spans segments; cap is `StorageConfig::wal_segment_bytes` (default 64 MiB). Checkpoints call `truncate_through(lsn)` (unlink fully covered segments); `truncate()` means "delete every segment" and is only for dropping a collection. A torn tail record is tolerated in the **last** segment only — damage anywhere else is `CorruptData`, never silent truncation.
- **Checkpoint order for `kind` / `named` is load-bearing:** `blobs.sync() → postings.flush(Some(lsn)) → checkpoint_payloads (append + sync) → snapshot::write → wal.truncate_through(lsn)`. `postings/manifest.bin` must carry the same `lsn` as the snapshot; when they differ the dictionary is discarded and rebuilt from the snapshot, never patched incrementally. `points.bin` and `payloads.bin` are therefore **derived data** — `kind`/`named` WAL records still inline the full vector and JSON payload, so replay re-appends both kinds of blobs (WAL vector externalization is designed but not done; see `docs/plan/14-payload-externalization-and-wal-vectors.md`).
- **Scratch dirs** `*.rebuild-*`, `.old-*`, `.suspect-*` exist only for crash recovery and are never loaded into the serving registry — don't scan them in tests.
- HNSW promotion is automatic past `full_scan_threshold` (default 10 000); a background maintenance worker runs every 30 s. ANN and quantization apply only to the single default dense space — named/sparse spaces use exact scan; unsupported combinations are rejected at config time.

## Maintenance

When project structure, build/test commands, architecture boundaries, development conventions, or other facts documented here change, update this file in the same change. Replace stale statements with current facts instead of appending a history of each change. Keep current rules, commands, boundaries, and non-obvious gotchas here; put detailed incident reports, benchmark tables, experimental transcripts, and completed migration stories in linked docs (e.g. `docs/plan/`, `docs/binding-matrix.md`) and read those only when relevant.
