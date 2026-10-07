//! HNSW build-time baseline.
//!
//! Records graph construction duration at two corpus sizes across the three
//! `max_indexing_threads` build shapes:
//! - `sequential`: unset, global rayon pool (deterministic);
//! - `single_pool`: dedicated one-thread pool (deterministic);
//! - `workers_4`: four concurrent builders over disjoint slot subsets.
//!
//! Deterministic-shape topology equality is asserted by the unit tests;
//! the multi-worker shape is gated on recall invariants instead.
//!
//! Run: `cargo bench -p simvec -- hnsw_build`

use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sim_vec::{CollectionConfig, DistanceMetric, HnswConfig, IndexType, LocalVectorEngine};

const DIM: usize = 128;
const SEED: u64 = 0xC0FFEE;

fn random_points(count: usize) -> Vec<sim_vec::VectorPoint> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..count)
        .map(|i| {
            sim_vec::VectorPoint::new(
                i as u64,
                (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            )
        })
        .collect()
}

fn build_once(threads: Option<usize>, n: usize) -> std::time::Duration {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    let hnsw = HnswConfig {
        max_indexing_threads: threads,
        ..HnswConfig::default()
    };
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid)
                .with_index_type(IndexType::HNSW)
                .with_hnsw(hnsw),
        )
        .unwrap();
    engine.upsert_batch("col", &random_points(n)).unwrap();

    let start = Instant::now();
    assert!(engine.build_index("col").unwrap());
    black_box(&engine);
    start.elapsed()
}

fn bench_hnsw_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_build");
    for &n in &[2_000usize, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(BenchmarkId::new("sequential", n), |b| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    total += build_once(None, n);
                }
                total
            })
        });
        group.bench_function(BenchmarkId::new("single_pool", n), |b| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    total += build_once(Some(1), n);
                }
                total
            })
        });
        group.bench_function(BenchmarkId::new("workers_4", n), |b| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    total += build_once(Some(4), n);
                }
                total
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hnsw_build);
criterion_main!(benches);
