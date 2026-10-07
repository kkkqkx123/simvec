//! Concurrent search scaling baseline.
//!
//! Compares single-threaded HNSW search throughput against an 8-worker
//! `par_iter` fan-out over the same published graph, giving the speedup
//! reference for lock-contention work (the search path takes only short
//! adjacency read locks plus atomic entry loads, so near-linear scaling is
//! the expectation until contention says otherwise).
//!
//! Run: `cargo bench -p simvec -- concurrent_search`

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use sim_vec::{
    CollectionConfig, DistanceMetric, IndexType, LocalVectorEngine, SearchQuery, VectorPoint,
};

const DIM: usize = 128;
const POINTS: usize = 20_000;
const QUERIES: usize = 256;
const SEED: usize = 42;

fn engine_with_index(path: &std::path::Path) -> LocalVectorEngine {
    let mut rng = StdRng::seed_from_u64(SEED as u64);
    let engine = LocalVectorEngine::open(path).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid).with_index_type(IndexType::HNSW),
        )
        .unwrap();
    let points: Vec<VectorPoint> = (0..POINTS)
        .map(|i| {
            VectorPoint::new(
                i as u64,
                (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            )
        })
        .collect();
    // Batched ingest; the promotion threshold keeps a small collection on
    // exact scan, so build explicitly.
    engine.upsert_batch("col", &points).unwrap();
    assert!(engine.build_index("col").unwrap());
    engine
}

fn queries() -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64((SEED + 1) as u64);
    (0..QUERIES)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// Run the query set once; returns the number of results for black-boxing.
fn run_query_set(engine: &LocalVectorEngine, queries: &[Vec<f32>]) -> usize {
    let mut hits = 0usize;
    for q in queries {
        let results = engine
            .search(
                "col",
                &SearchQuery::new(q.clone(), 10).with_knn(10, Some(40)),
            )
            .unwrap();
        hits += results.len();
    }
    hits
}

fn bench_concurrent_search(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(engine_with_index(&dir.path().join("vec")));
    let queries = Arc::new(queries());

    let mut group = c.benchmark_group("concurrent_search");
    group.throughput(Throughput::Elements(QUERIES as u64));

    group.bench_function(BenchmarkId::new("threads", 1), |b| {
        b.iter(|| {
            black_box(run_query_set(&engine, &queries));
        })
    });

    for threads in [4usize, 8] {
        group.bench_function(BenchmarkId::new("threads", threads), |b| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            b.iter(|| {
                let total: usize = pool.install(|| {
                    queries
                        .par_iter()
                        .map(|q| {
                            engine
                                .search(
                                    "col",
                                    &SearchQuery::new(q.clone(), 10).with_knn(10, Some(40)),
                                )
                                .unwrap()
                                .len()
                        })
                        .sum()
                });
                black_box(total);
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_concurrent_search);
criterion_main!(benches);
