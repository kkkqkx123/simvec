# simvec

[中文文档](README_zh.md)

`simvec` is a local vector search engine library for Rust: exact-scan storage
and search with optional ANN indexes, WAL-backed durability, and mmap-based
vector storage — all in a single process, with no external services.

It is a leaf crate: it does not depend on any graphdb crate and does not pull
in the Qdrant networking stack. The type surface (`VectorPoint`,
`SearchQuery`, `VectorFilter`, …) is shared with the Qdrant client used
elsewhere in the workspace, so a collection created locally can be served by
either backend without changing query code.

## Features

- **Exact scan by default, ANN when it pays off.** Collections start as
  brute-force scans and are automatically promoted to the published index
  once the live-point count crosses `full_scan_threshold` (default
  10 000, matching Qdrant's default).
- **HNSW and IVF-Flat indexes.** HNSW supports concurrent build
  (`max_indexing_threads`), iterative filtered-scan expansion, staleness
  rebuilds, and pending-slot draining. IVF supports automatic `lists`
  selection, `nprobe` per query, drift-triggered rebuilds, and promotion from
  exact scan.
- **WAL-backed durability.** Every mutation is appended and fsynced before it
  is applied to memory; replay is idempotent, so crash recovery can re-apply
  transactions without double-applying data. Coordinated transactions land in
  `LocalVectorEngine::apply_txn`.
- **Simulated two-phase publish.** Index builds write to `*.rebuild-*`
  scratch directories, publish atomically, and keep `.old-*` backups for
  reverse recovery (`restore_promote_backup`).
- **Payload storage with filtering.** Full JSON payloads, optional payload
  field indexes, and a rich filter language: match, range, values-count,
  geo (radius / bounding box), nested conditions, and `must` / `should` /
  `must_not` combinators.
- **SIMD distance kernels.** AVX2+FMA and AVX-512 on x86-64, NEON on
  aarch64, with a naive fallback that always serves as the correctness
  baseline. Metrics: Cosine, Euclid, Dot, Manhattan.
- **Quantization.** Scalar and product quantization with k-means codebook
  training, used as a search accelerator (exact scores are always recomputed).
- **Tombstones and compaction.** Deletes are logical; `compact_collection`
  physically reclaims slots. A background maintenance worker runs builds,
  compaction, drift sweeps, and HNSW promotion every 30 s.
- **Metrics.** Per-collection `MetricsSnapshot` with search path counters
  (scan / IVF / HNSW, retries, fallbacks). Optional `lock-metrics` feature
  instruments lock wait times for contention investigations.

## Usage

```rust
use simvec::{
    CollectionConfig, DistanceMetric, LocalVectorEngine, SearchQuery, VectorPoint,
};

// Open (or create) an engine rooted at a directory. Every collection is a
// subdirectory; existing collections are loaded on open.
let engine = LocalVectorEngine::open("./data")?;

// Create a collection: 384-dim cosine, default HNSW tier.
let config = CollectionConfig::new(384, DistanceMetric::Cosine);
engine.create_collection("docs", &config)?;

// Upsert points (WAL-backed; fsync before applying to memory).
let point = VectorPoint::new("doc-1", vec![0.1; 384])
    .with_payload_kv("title", "hello".into());
engine.upsert("docs", point)?;

// Search. Scores follow the crate-wide contract: higher is better.
let query = SearchQuery::new(vec![0.1; 384], 10);
let results = engine.search("docs", &query)?;
for r in &results {
    println!("{} -> {}", r.id, r.score);
}
```

Filtering, kNN mode, range queries, pagination, and payload projection are
builders on `SearchQuery`:

```rust
use simvec::{FilterCondition, SearchQuery, VectorFilter};

let filter = VectorFilter::new().must(FilterCondition::match_value("color", "red"));
let query = SearchQuery::new(vec![0.1; 384], 10)
    .with_knn(10, Some(64))                       // kNN with explicit ef_search
    .with_filter(filter)
    .with_score_threshold(0.5)
    .with_payload_include(vec!["title".into()]);
```

## Architecture

| Module | Responsibility |
|--------|----------------|
| `engine` | `LocalVectorEngine`: collection registry, WAL-backed mutations, background maintenance worker, publish / promote / recovery |
| `storage` | `CollectionStore`: one collection directory — vectors (mmap), payloads, WAL, tombstones, metadata, payload indexes, quantization, search |
| `index` | HNSW graph, IVF (k-means), and index persistence |
| `distance` | Distance kernels (AVX2 / AVX-512 / NEON / naive) and the score contract |
| `filter` / `filter_cond` | Filter evaluation and condition types |
| `metrics` | `MetricsSnapshot` and search-path instrumentation |
| `types` | Shared type surface (also used by the remote Qdrant backend) |

### Storage layout

Each collection is a directory under the engine root containing
`vectors.bin` (mmap'd raw f32 rows), the WAL segment files, payload store,
metadata, and index artifacts. Deletes are tombstones; compaction rewrites
the directory. Rebuild scratch (`*.rebuild-*`), publish backups (`*.old-*`),
and quarantined (`.suspect-*`) directories are never loaded into the serving
registry — they exist only for crash recovery and inspection.

### Score contract

The engine ranks by an internal distance (smaller = nearer) and converts it
to a similarity score on output. The score **is** the public contract:
higher is better on every metric and backend, and `score_threshold` is a
lower bound.

| metric | internal distance | output score |
|--------|-------------------|--------------|
| Euclid | `Σ(a-b)²` (squared, no sqrt) | `1/(1+sqrt(d²))` |
| Dot | `-Σ(a·b)` | `Σ(a·b)` |
| Cosine | `1 - similarity` clamped to `[-1, 1]` | `similarity` |
| Manhattan | `Σ\|a-b\|` | `1/(1+sqrt(d))` |

Cosine computes norms on the fly rather than normalizing at insert time:
`vectors.bin` is the single copy of the data and `get()` / `with_vector`
must return the original bytes.

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `lock-metrics` | off | Compile in adjacency/list write-lock wait instrumentation; metrics flow through `MetricsSnapshot` |
| `simd_portable` | off | Portable `std::simd` kernel (unstable on stable Rust); currently delegates to `naive` as a placeholder until `std::simd` stabilizes |

## Testing and benchmarks

```sh
cargo test -p simvec          # unit + integration tests
cargo bench -p simvec         # criterion benchmarks
```

Benchmarks cover vector scan, IVF, HNSW build and ingest, concurrent search,
allocation statistics, and persistence CRC overhead.

## License

Apache-2.0
