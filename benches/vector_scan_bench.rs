//! Benchmark baseline.
//!
//! Establishes the exact-scan baseline before any optimization work:
//! scan latency/throughput vs. dataset size, SIMD vs naive ratio, filter
//! selectivity impact, and WAL-backed upsert throughput.
//!
//! Run: `cargo bench -p simvec`

use std::collections::HashMap;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sim_vec::distance::kernel::{selected, Kernel};
use sim_vec::{
    CollectionConfig, DistanceMetric, FilterCondition, LocalVectorEngine, SearchQuery, TxnOp,
    VectorFilter, VectorPoint,
};

const DIM: usize = 128;
const SEED: u64 = 0xC0FFEE;

/// Random unit vectors (dim 128) with a fixed seed.
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

fn unit_vector(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// Exact scan loop: distance to every candidate, keep the nearest.
fn scan_best(kernel: Kernel, metric: DistanceMetric, query: &[f32], data: &[Vec<f32>]) -> f32 {
    let mut best = f32::INFINITY;
    for v in data {
        let d = kernel.distance(metric, query, v);
        if d < best {
            best = d;
        }
    }
    best
}

fn scan_latency(c: &mut Criterion) {
    let query = unit_vector(SEED);
    let data = unit_vectors(10_000);
    let kernel = selected();
    let mut group = c.benchmark_group("scan_latency");
    group.throughput(Throughput::Elements(10_000));
    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::Euclid,
        DistanceMetric::Dot,
    ] {
        group.bench_with_input(
            BenchmarkId::new("10k", format!("{metric:?}")),
            &(),
            |b, _| b.iter(|| black_box(scan_best(kernel, metric, &query, &data))),
        );
    }
    group.finish();

    let data = unit_vectors(100_000);
    let mut group = c.benchmark_group("scan_latency");
    group.throughput(Throughput::Elements(100_000));
    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::Euclid,
        DistanceMetric::Dot,
    ] {
        group.bench_with_input(
            BenchmarkId::new("100k", format!("{metric:?}")),
            &(),
            |b, _| b.iter(|| black_box(scan_best(kernel, metric, &query, &data))),
        );
    }
    group.finish();

    let data = unit_vectors(1_000_000);
    let mut group = c.benchmark_group("scan_latency");
    group.throughput(Throughput::Elements(1_000_000));
    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::Euclid,
        DistanceMetric::Dot,
    ] {
        group.bench_with_input(
            BenchmarkId::new("1M", format!("{metric:?}")),
            &(),
            |b, _| b.iter(|| black_box(scan_best(kernel, metric, &query, &data))),
        );
    }
    group.finish();
}

/// SIMD vs naive ratio on the same input (multi-kernel).
///
/// All kernels must agree within 1e-4 (tested in unit tests) and the bench
/// here establishes the latency win (>15% for AVX-512 to be promoted).
fn simd_vs_naive(c: &mut Criterion) {
    // Build the list of available kernels for this host.
    let mut kernels = vec![Kernel::Naive];
    #[cfg(target_arch = "x86_64")]
    {
        if Kernel::Avx2.is_available() {
            kernels.push(Kernel::Avx2);
        }
        if Kernel::Avx512.is_available() {
            kernels.push(Kernel::Avx512);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if Kernel::Neon.is_available() {
            kernels.push(Kernel::Neon);
        }
    }
    #[cfg(feature = "simd_portable")]
    {
        if Kernel::Portable.is_available() {
            kernels.push(Kernel::Portable);
        }
    }
    if kernels.len() <= 1 {
        eprintln!("no SIMD kernels available; skipping simd_vs_naive");
        return;
    }
    let query = unit_vector(SEED);
    let data = unit_vectors(100_000);
    let mut group = c.benchmark_group("simd_vs_naive");
    for kernel in kernels {
        group.bench_with_input(
            BenchmarkId::new("100k_cosine", kernel.to_string()),
            &(),
            |b, _| b.iter(|| black_box(scan_best(kernel, DistanceMetric::Cosine, &query, &data))),
        );
    }
    group.finish();
}

/// Dim-scaling bench for the promotion gate (dim=384/768/1536).
///
/// AVX-512 is only promoted to the default `best_available` ranking when it
/// sustains >15% win across these dims without p99 regression.
fn simd_dim_scaling(c: &mut Criterion) {
    // Collect available kernels again for dim scaling.
    let mut kernels = vec![Kernel::Naive];
    #[cfg(target_arch = "x86_64")]
    {
        if Kernel::Avx2.is_available() {
            kernels.push(Kernel::Avx2);
        }
        if Kernel::Avx512.is_available() {
            kernels.push(Kernel::Avx512);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if Kernel::Neon.is_available() {
            kernels.push(Kernel::Neon);
        }
    }
    #[cfg(feature = "simd_portable")]
    {
        if Kernel::Portable.is_available() {
            kernels.push(Kernel::Portable);
        }
    }
    if kernels.len() <= 1 {
        return;
    }
    for dim in [384usize, 768, 1536] {
        // Generate dim-specific data (still unit vectors for cosine).
        let mut rng = StdRng::seed_from_u64(SEED + dim as u64);
        let data: Vec<Vec<f32>> = (0..10_000)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm.max(1e-6);
                }
                v
            })
            .collect();
        let mut q: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut q {
            *x /= norm.max(1e-6);
        }
        let mut group = c.benchmark_group(format!("simd_dim_{dim}"));
        group.throughput(Throughput::Elements(10_000));
        for kernel in &kernels {
            for metric in [DistanceMetric::Cosine, DistanceMetric::Dot] {
                group.bench_with_input(
                    BenchmarkId::new(format!("{metric:?}"), kernel.to_string()),
                    &(),
                    |b, _| b.iter(|| black_box(scan_best(*kernel, metric, &q, &data))),
                );
            }
        }
        group.finish();
    }
}

/// End-to-end engine search latency vs filter selectivity.
fn filter_selectivity(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(LocalVectorEngine::open(dir.path()).unwrap());
    let collection = "sel";
    engine
        .create_collection(
            collection,
            &CollectionConfig {
                vector_size: DIM,
                distance: DistanceMetric::Cosine,
                ..Default::default()
            },
        )
        .unwrap();

    // 100k points, each tagged with a `bucket` string in [0, 100).
    let count = 100_000;
    let mut rng = StdRng::seed_from_u64(SEED + 1);
    let points: Vec<VectorPoint> = (0..count)
        .map(|i| {
            let mut v: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            let payload = HashMap::from([("bucket".to_string(), (i % 100).to_string().into())]);
            VectorPoint::new(format!("p{i}"), v).with_payload(payload)
        })
        .collect();
    engine
        .apply_txn(
            1,
            points
                .iter()
                .map(|p| TxnOp::Upsert {
                    collection: collection.to_string(),
                    point: p.clone(),
                })
                .collect(),
        )
        .unwrap();

    let query = unit_vector(SEED + 2);
    let mut group = c.benchmark_group("filter_selectivity");
    for (name, buckets) in [
        ("100%", None),
        ("50%", Some((0..50).collect::<Vec<i64>>())),
        ("10%", Some((0..10).collect::<Vec<i64>>())),
        ("1%", Some(vec![7])),
    ] {
        let filter = buckets.map(|bs| {
            VectorFilter::new().must(FilterCondition::match_any(
                "bucket",
                bs.into_iter()
                    .map(|b| serde_json::json!(b.to_string()))
                    .collect(),
            ))
        });
        let search_query = match &filter {
            Some(f) => SearchQuery::new(query.clone(), 10).with_filter(f.clone()),
            None => SearchQuery::new(query.clone(), 10),
        };
        group.bench_with_input(BenchmarkId::new("100k", name), &(), |b, _| {
            b.iter(|| {
                let r = engine.search(collection, &search_query).unwrap();
                black_box(r.len())
            })
        });
    }
    group.finish();
}

/// WAL-backed upsert throughput (single vs batch).
fn upsert_wal(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(LocalVectorEngine::open(dir.path()).unwrap());
    let collection = "up";
    engine
        .create_collection(
            collection,
            &CollectionConfig {
                vector_size: DIM,
                distance: DistanceMetric::Cosine,
                ..Default::default()
            },
        )
        .unwrap();

    let data = unit_vectors(1_000);
    let mut group = c.benchmark_group("upsert_wal");

    // Single point per txn.
    let mut rng = StdRng::seed_from_u64(SEED + 3);
    let single: Vec<(u64, VectorPoint)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64 + 1, VectorPoint::new(format!("s{i}"), v.clone())))
        .collect();
    group.bench_function("single", |b| {
        b.iter(|| {
            let (txn_id, point) = &single[rng.gen_range(0..single.len())];
            engine
                .apply_txn(
                    *txn_id,
                    vec![TxnOp::Upsert {
                        collection: collection.to_string(),
                        point: point.clone(),
                    }],
                )
                .unwrap();
            black_box(())
        })
    });

    // Batch of 100 points per txn (unique ids per batch).
    let mut batch_txn = 0u64;
    group.bench_function("batch_100", |b| {
        b.iter(|| {
            batch_txn += 1;
            let ops: Vec<TxnOp> = data
                .iter()
                .enumerate()
                .map(|(i, v)| TxnOp::Upsert {
                    collection: collection.to_string(),
                    point: VectorPoint::new(format!("b{batch_txn}_{i}"), v.clone()),
                })
                .collect();
            engine.apply_txn(batch_txn, ops).unwrap();
            black_box(())
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    scan_latency,
    simd_vs_naive,
    simd_dim_scaling,
    filter_selectivity,
    upsert_wal
);
criterion_main!(benches);
