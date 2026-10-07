//! Exact scan vs IVF index (IVFFlat) benchmark.
//!
//! Measures build time, query latency vs nprobe, recall@10 against the
//! exact-scan ground truth, and upsert overhead with/without a published
//! index. The promotion rule for flipping `IvfConfig::auto_promotion` on by
//! default lives at the bottom of this file.
//!
//! Run: `cargo bench -p simvec -- ivf`

use std::sync::Arc;
use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sim_vec::{
    CollectionConfig, DistanceMetric, IvfConfig, LocalVectorEngine, SearchQuery, VectorPoint,
};

const DIM: usize = 128;
const SEED: u64 = 0xBEEF;

fn unit_vectors(count: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..count)
        .map(|_| {
            let mut v: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect()
}

fn ivf_config(lists: u32) -> IvfConfig {
    IvfConfig {
        lists: Some(lists),
        min_build_points: 1,
        sample_limit: 65_536,
        kmeans_max_iter: 10,
        drift_threshold: 0.10,
        drift_check_interval: u64::MAX,
        default_nprobe: 8,
        auto_promotion: false,
        max_probes: None,
    }
}

fn engine_with(path: &std::path::Path, n: usize, lists: u32) -> LocalVectorEngine {
    let engine = LocalVectorEngine::open(path).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Cosine).with_ivf(ivf_config(lists)),
        )
        .unwrap();
    let vectors = unit_vectors(n);
    let points: Vec<VectorPoint> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, v)| VectorPoint::new(format!("p{i}"), v))
        .collect();
    engine.upsert_batch("col", &points).unwrap();
    engine
}

/// Ground truth via exact scan; returns point-id strings in rank order.
fn ground_truth(engine: &LocalVectorEngine, query: &[f32], k: usize) -> Vec<String> {
    engine
        .search("col", &SearchQuery::new(query.to_vec(), k))
        .unwrap()
        .into_iter()
        .map(|r| r.id.to_string())
        .collect()
}

fn build_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("ivf_build_time");
    const N: usize = 100_000;
    group.bench_function(BenchmarkId::new("build", N), |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let dir = tempfile::tempdir().unwrap();
                let engine = engine_with(dir.path(), N, 512);
                let start = Instant::now();
                engine.build_index("col").unwrap();
                total += start.elapsed();
            }
            total
        })
    });
    group.finish();
}

fn latency_and_recall(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let n = 100_000;
    let engine = engine_with(dir.path(), n, 256);

    // Ground truth must come from the exact-scan path, so compute it before
    // the index is published; afterwards searches route through IVF.
    let queries = unit_vectors(20);
    let truth: Vec<Vec<String>> = queries
        .iter()
        .map(|q| ground_truth(&engine, q, 10))
        .collect();
    engine.build_index("col").unwrap();

    let mut group = c.benchmark_group("ivf_latency_vs_nprobe");
    let mut report =
        String::from("\nnprobe | latency_mean | recall@10\n------- | ------------ | ---------\n");
    for nprobe in [1usize, 4, 16, 64, 256] {
        // Recall measurement (outside criterion timing).
        let mut hits = 0usize;
        for (q, expected) in queries.iter().zip(truth.iter()) {
            let got: Vec<String> = engine
                .search("col", &SearchQuery::new(q.clone(), 10).with_nprobe(nprobe))
                .unwrap()
                .into_iter()
                .map(|r| r.id.to_string())
                .collect();
            hits += got.iter().filter(|id| expected.contains(id)).count();
        }
        let recall = hits as f64 / (queries.len() * 10) as f64;

        let label = format!("{nprobe}");
        group.bench_function(BenchmarkId::new("100k", &label), |b| {
            b.iter(|| {
                let r = engine
                    .search(
                        "col",
                        &SearchQuery::new(queries[0].clone(), 10).with_nprobe(nprobe),
                    )
                    .unwrap();
                black_box(r.len())
            })
        });
        report.push_str(&format!("{nprobe:>6} |              | {recall:.4}\n"));
    }
    group.finish();
    println!("{}", report);
}

fn exact_vs_index_crossover(_c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let n = 100_000;
    let engine = engine_with(dir.path(), n, 256);
    let query = unit_vectors(1).pop().unwrap();

    let mut samples_exact = Vec::new();
    for _ in 0..50 {
        let s = Instant::now();
        black_box(
            engine
                .search("col", &SearchQuery::new(query.clone(), 10))
                .unwrap(),
        );
        samples_exact.push(s.elapsed());
    }

    engine.build_index("col").unwrap();
    let mut samples_ivf = Vec::new();
    for _ in 0..50 {
        let s = Instant::now();
        black_box(
            engine
                .search("col", &SearchQuery::new(query.clone(), 10).with_nprobe(8))
                .unwrap(),
        );
        samples_ivf.push(s.elapsed());
    }

    let mean = |v: &[std::time::Duration]| v.iter().sum::<std::time::Duration>() / v.len() as u32;
    println!(
        "\nexact_vs_index_crossover @ {n} points, dim {DIM}: exact={:?} ivf(nprobe=8)={:?}",
        mean(&samples_exact),
        mean(&samples_ivf)
    );
}

fn ivf_upsert_overhead(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(engine_with(dir.path(), 50_000, 256));

    let data = unit_vectors(100);
    let mut txn_id = 1u64;
    let mut group = c.benchmark_group("ivf_upsert_overhead");

    group.bench_function("without_index", |b| {
        b.iter(|| {
            txn_id += 1;
            let ops: Vec<VectorPoint> = data
                .iter()
                .enumerate()
                .map(|(i, v)| VectorPoint::new(format!("u{txn_id}_{i}"), v.clone()))
                .collect();
            engine.upsert_batch("col", &ops).unwrap();
        })
    });

    engine.build_index("col").unwrap();
    group.bench_function("with_index", |b| {
        b.iter(|| {
            txn_id += 1;
            let ops: Vec<VectorPoint> = data
                .iter()
                .enumerate()
                .map(|(i, v)| VectorPoint::new(format!("u{txn_id}_{i}"), v.clone()))
                .collect();
            engine.upsert_batch("col", &ops).unwrap();
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    build_time,
    latency_and_recall,
    exact_vs_index_crossover,
    ivf_upsert_overhead
);
criterion_main!(benches);

// Promotion rule:
// Flip `IvfConfig::default()`'s `auto_promotion` to true iff, at 1M points:
//   - IVF (default_nprobe) latency < exact scan latency / 3, and
//   - recall@10 >= 0.98 in `ivf_latency_vs_nprobe`.
