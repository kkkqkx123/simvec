//! Concurrency tests for simvec: metrics wiring, concurrent
//! HNSW/IVF search and build operations.

use std::sync::Arc;
use std::thread;

use simvec::{
    CollectionConfig, DistanceMetric, HnswConfig, IvfConfig, LocalVectorEngine, SearchQuery,
    VectorPoint,
};

const DIM: usize = 16;

fn hnsw_config() -> HnswConfig {
    HnswConfig {
        m: 8,
        ef_construct: 16,
        ef_search: 16,
        ..HnswConfig::default()
    }
}

fn ivf_config() -> IvfConfig {
    IvfConfig {
        lists: Some(4),
        min_build_points: 10,
        sample_limit: 1000,
        kmeans_max_iter: 10,
        drift_threshold: 0.5,
        drift_check_interval: 1000,
        default_nprobe: 2,
        auto_promotion: false,
        max_probes: None,
    }
}

fn point(id: u64, dim: usize) -> VectorPoint {
    VectorPoint::new(
        id,
        (0..dim)
            .map(|i| ((id as usize * 31 + i) % 100) as f32 / 100.0)
            .collect(),
    )
}

// ── Metrics Wiring Tests ────────────────────────────────────────────

#[test]
fn metrics_record_mutations_and_searches() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path()).unwrap();
    engine
        .create_collection("col", &CollectionConfig::new(DIM, DistanceMetric::Cosine))
        .unwrap();

    let before = engine.collection_metrics("col").unwrap();
    assert_eq!(before.txns_applied, 0);
    assert_eq!(before.search_total, 0);

    // One batch = one WAL transaction.
    let points: Vec<VectorPoint> = (0..50).map(|i| point(i as u64, DIM)).collect();
    engine.upsert_batch("col", &points).unwrap();
    engine.delete("col", "0").unwrap();

    let after_writes = engine.collection_metrics("col").unwrap();
    assert_eq!(after_writes.txns_applied, 2);
    assert_eq!(after_writes.points_upserted, 50);
    assert_eq!(after_writes.points_deleted, 1);
    assert_eq!(
        after_writes.apply_txn.count, 2,
        "apply latency recorded per transaction"
    );

    // No published index on a tiny collection: exact scan path.
    engine
        .search("col", &SearchQuery::new(vec![0.5; DIM], 5))
        .unwrap();
    let after_search = engine.collection_metrics("col").unwrap();
    assert_eq!(after_search.search_total, 1);
    assert_eq!(after_search.search_exact, 1);
}

#[test]
fn metrics_record_ann_path_and_build() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path()).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Cosine)
                .with_index_type(simvec::IndexType::HNSW)
                .with_hnsw(hnsw_config()),
        )
        .unwrap();

    let points: Vec<VectorPoint> = (0..50).map(|i| point(i as u64, DIM)).collect();
    engine.upsert_batch("col", &points).unwrap();
    engine.build_index("col").unwrap();

    let m = engine.collection_metrics("col").unwrap();
    assert_eq!(m.hnsw_builds, 1);
    assert_eq!(m.hnsw_build.count, 1);

    engine
        .search(
            "col",
            &SearchQuery::new(vec![0.5; DIM], 5).with_knn(5, Some(16)),
        )
        .unwrap();
    let m = engine.collection_metrics("col").unwrap();
    assert_eq!(m.search_total, 1);
    assert_eq!(m.search_hnsw, 1);
    assert_eq!(m.search_exact, 0);
}

// ── Concurrent HNSW Search Tests ────────────────────────────────────

#[test]
fn concurrent_hnsw_search() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path()).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Cosine)
                .with_index_type(simvec::IndexType::HNSW)
                .with_hnsw(hnsw_config()),
        )
        .unwrap();

    // Insert points.
    let points: Vec<VectorPoint> = (0..50).map(|i| point(i as u64, DIM)).collect();
    engine.upsert_batch("col", &points).unwrap();
    engine.build_index("col").unwrap();

    // Spawn concurrent searches.
    let engine = Arc::new(engine);
    let mut handles = Vec::new();
    for t in 0..4 {
        let engine = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            for i in 0..20 {
                let q: Vec<f32> = (0..DIM)
                    .map(|j| ((t * 100 + i + j) % 100) as f32 / 100.0)
                    .collect();
                let results = engine
                    .search("col", &SearchQuery::new(q, 10).with_knn(10, Some(16)))
                    .unwrap();
                assert!(
                    !results.is_empty(),
                    "query {i} on thread {t} returned no results"
                );
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn concurrent_hnsw_insert_and_search() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path()).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid)
                .with_index_type(simvec::IndexType::HNSW)
                .with_hnsw(hnsw_config()),
        )
        .unwrap();

    // Build initial index.
    let points: Vec<VectorPoint> = (0..30).map(|i| point(i as u64, DIM)).collect();
    engine.upsert_batch("col", &points).unwrap();
    engine.build_index("col").unwrap();

    // Concurrent writers + readers.
    let engine = Arc::new(engine);
    let mut handles = Vec::new();

    // Writers.
    for t in 0..2 {
        let engine = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            for i in 0..10 {
                let id = 1000 + t * 100 + i;
                engine.upsert("col", point(id as u64, DIM)).unwrap();
            }
        }));
    }

    // Readers.
    for t in 0..2 {
        let engine = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            for i in 0..10 {
                let q: Vec<f32> = (0..DIM)
                    .map(|j| ((t * 100 + i + j) % 100) as f32 / 100.0)
                    .collect();
                let _ = engine.search("col", &SearchQuery::new(q, 5).with_knn(5, Some(16)));
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

/// Cross write/read pressure test: writers keep upserting fresh slots (each
/// maintenance-style drain links them into shared hub neighborhoods) while
/// readers probe concurrently. The version double-read protocol must detect
/// at least one concurrent adjacency mutation (`search_version_reloads`),
/// proving the torn-read recovery path is genuinely exercised, not just
/// compiled in. Search results stay valid throughout.
#[test]
fn concurrent_write_read_covers_version_double_read() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path()).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid)
                .with_index_type(simvec::IndexType::HNSW)
                .with_hnsw(hnsw_config()),
        )
        .unwrap();

    // Small dense base graph: every later insert reverse-links into the same
    // few hubs, maximizing adjacency mutation under the concurrent probes.
    let base: Vec<VectorPoint> = (0..40)
        .map(|i| VectorPoint::new(i, vec![50.0; DIM]))
        .collect();
    engine.upsert_batch("col", &base).unwrap();
    engine.build_index("col").unwrap();
    assert!(engine.has_index("col"));

    const WRITER_BATCHES: usize = 30;
    const BATCH: usize = 20;

    let engine = Arc::new(engine);

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut handles = Vec::new();

    // Writer: batches of fresh clustered slots, drained into the graph so
    // mutations actually land while readers are mid-probe.
    {
        let engine = Arc::clone(&engine);
        let next_id = std::sync::atomic::AtomicU64::new(1000);
        let next_id = Arc::new(next_id);
        handles.push(thread::spawn(move || {
            for _ in 0..WRITER_BATCHES {
                let ids: Vec<u64> = (0..BATCH)
                    .map(|_| next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
                    .collect();
                let points: Vec<VectorPoint> = ids
                    .into_iter()
                    .map(|id| {
                        let jitter = (id % 7) as f32 * 0.1;
                        let mut v = vec![50.0f32; DIM];
                        v[(id % DIM as u64) as usize] += jitter;
                        VectorPoint::new(id, v)
                    })
                    .collect();
                engine.upsert_batch("col", &points).unwrap();
                // Push pending slots into the graph immediately instead of
                // waiting for the maintenance tick.
                engine.drain_pending("col").unwrap();
            }
        }));
    }

    // Readers: hammer the graph path until the writer finishes.
    for t in 0..2 {
        let engine = Arc::clone(&engine);
        let stop = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            let mut rounds = 0u32;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let q = vec![50.0; DIM];
                let results = engine
                    .search("col", &SearchQuery::new(q, 5).with_knn(5, Some(8)))
                    .unwrap();
                assert!(!results.is_empty(), "reader {t} got no results");
                rounds += 1;
                assert!(rounds < 1_000_000, "reader {t} spun too long");
            }
        }));
    }

    // Join the writer (releasing the readers even on failure), then reap.
    let writer = handles.swap_remove(0);
    let joined = writer.join();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }
    joined.unwrap();

    // Every written point must be incorporated into the graph.
    assert_eq!(
        engine.count("col").unwrap(),
        40 + (WRITER_BATCHES * BATCH) as u64
    );
    engine.drain_pending("col").unwrap();

    let m = engine.collection_metrics("col").unwrap();
    assert!(
        m.search_version_reloads > 0,
        "write/read contention never hit the version double-read reload path"
    );

    // Post-storm sanity: results remain sane after all the mutation traffic.
    let results = engine
        .search(
            "col",
            &SearchQuery::new(vec![50.0; DIM], 5).with_knn(5, Some(16)),
        )
        .unwrap();
    assert_eq!(results.len(), 5);
}

/// The in-flight build progress surfaced through `collection_info` must be
/// consistent before and after a build completes.
#[test]
fn collection_info_reports_build_progress() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path()).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Cosine)
                .with_index_type(simvec::IndexType::HNSW)
                .with_hnsw(hnsw_config()),
        )
        .unwrap();
    let points: Vec<VectorPoint> = (0..60).map(|i| point(i as u64, DIM)).collect();
    engine.upsert_batch("col", &points).unwrap();

    let info = engine.collection_info("col").unwrap();
    let index = info.index.expect("ANN tier configured");
    assert!(!index.building);
    assert_eq!(index.build_points_total, 0);

    engine.build_index("col").unwrap();
    let info = engine.collection_info("col").unwrap();
    let index = info.index.unwrap();
    assert!(!index.building, "build finished");
    assert_eq!(index.build_inserted, index.build_points_total);
    assert_eq!(index.build_points_total, 60);
}

// ── Concurrent IVF Search Tests ─────────────────────────────────────

#[test]
fn concurrent_ivf_search() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path()).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Cosine)
                .with_index_type(simvec::IndexType::IVF)
                .with_ivf(ivf_config()),
        )
        .unwrap();

    let points: Vec<VectorPoint> = (0..50).map(|i| point(i as u64, DIM)).collect();
    engine.upsert_batch("col", &points).unwrap();
    engine.build_index("col").unwrap();

    let engine = Arc::new(engine);
    let mut handles = Vec::new();
    for t in 0..4 {
        let engine = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            for i in 0..20 {
                let q: Vec<f32> = (0..DIM)
                    .map(|j| ((t * 100 + i + j) % 100) as f32 / 100.0)
                    .collect();
                let results = engine
                    .search("col", &SearchQuery::new(q, 10).with_nprobe(4))
                    .unwrap();
                assert!(!results.is_empty());
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

// ── HNSW Concurrent Build Test ──────────────────────────────────────

#[test]
fn hnsw_concurrent_build() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path()).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Cosine)
                .with_index_type(simvec::IndexType::HNSW)
                .with_hnsw(HnswConfig {
                    m: 8,
                    ef_construct: 16,
                    ef_search: 16,
                    max_indexing_threads: Some(4),
                    ..HnswConfig::default()
                }),
        )
        .unwrap();

    let points: Vec<VectorPoint> = (0..50).map(|i| point(i as u64, DIM)).collect();
    engine.upsert_batch("col", &points).unwrap();

    // Build with multiple threads.
    engine.build_index("col").unwrap();

    // Verify search works after concurrent build.
    let results = engine
        .search(
            "col",
            &SearchQuery::new(vec![0.5; DIM], 5).with_knn(5, Some(16)),
        )
        .unwrap();
    assert_eq!(results.len(), 5);
}
