//! HNSW ingest throughput: write-path decoupling baseline.
//!
//! Compares batched upsert throughput into (a) an exact-scan collection and
//! (b) a collection with a published HNSW index. The published-index path
//! routes fresh slots into the pending queue (O(1) per upsert under the
//! store lock) instead of running the incremental graph insert inline.
//!
//! Run: `cargo bench -p simvec -- hnsw_ingest`

use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sim_vec::{CollectionConfig, DistanceMetric, IndexType, LocalVectorEngine, VectorPoint};

const DIM: usize = 128;
const BATCH: usize = 256;
const SEED: u64 = 0xF00D;

struct PointSource {
    rng: StdRng,
    next_id: u64,
}

impl PointSource {
    fn new() -> Self {
        Self {
            rng: StdRng::seed_from_u64(SEED),
            next_id: 0,
        }
    }

    fn batch(&mut self) -> Vec<VectorPoint> {
        (0..BATCH)
            .map(|_| {
                let p = VectorPoint::new(
                    self.next_id,
                    (0..DIM).map(|_| self.rng.gen_range(-1.0..1.0)).collect(),
                );
                self.next_id += 1;
                p
            })
            .collect()
    }
}

fn ingest(engine: &LocalVectorEngine, iters: u64) -> std::time::Duration {
    let mut src = PointSource::new();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..iters {
        let batch = src.batch();
        let start = Instant::now();
        engine.upsert_batch("col", black_box(&batch)).unwrap();
        total += start.elapsed();
    }
    total
}

fn bench_hnsw_ingest(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_ingest");
    group.throughput(Throughput::Elements(BATCH as u64));

    // Exact scan: no ANN tier, pure storage write path.
    let flat_dir = tempfile::tempdir().unwrap();
    let flat = LocalVectorEngine::open(flat_dir.path().join("vec")).unwrap();
    flat.create_collection(
        "col",
        &CollectionConfig::new(DIM, DistanceMetric::Euclid).with_index_type(IndexType::FLAT),
    )
    .unwrap();
    group.bench_function(BenchmarkId::new("exact_scan", BATCH), |b| {
        b.iter_custom(|iters| ingest(&flat, iters))
    });

    // Published HNSW: writes land in pending; the maintenance worker drains.
    let hnsw_dir = tempfile::tempdir().unwrap();
    let hnsw_engine = LocalVectorEngine::open(hnsw_dir.path().join("vec")).unwrap();
    hnsw_engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid).with_index_type(IndexType::HNSW),
        )
        .unwrap();
    // Seed enough points to publish a graph up front.
    let mut seed_src = PointSource::new();
    hnsw_engine.upsert_batch("col", &seed_src.batch()).unwrap();
    assert!(hnsw_engine.build_index("col").unwrap());
    assert!(hnsw_engine.has_index("col"));

    group.bench_function(BenchmarkId::new("published_hnsw", BATCH), |b| {
        b.iter_custom(|iters| ingest(&hnsw_engine, iters))
    });

    group.finish();
}

criterion_group!(benches, bench_hnsw_ingest);
criterion_main!(benches);
