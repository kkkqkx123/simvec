//! Distance-kernel baseline.
//!
//! Measures the shipped distance kernel (`distance::kernel::selected`) across
//! metrics and dimensions. This replaces the former root `vector_distance`
//! stub, which hand-rolled an L1 sum unrelated to any shipped kernel.
//!
//! Run: `cargo bench -p simvec --bench distance_bench`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use simvec::distance::kernel::selected;
use simvec::DistanceMetric;

fn pair(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.73).cos()).collect();
    (a, b)
}

fn bench_selected_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_kernel");
    for dim in [128usize, 256, 512] {
        let (a, b) = pair(dim);
        group.throughput(Throughput::Elements(dim as u64));
        for metric in [
            DistanceMetric::Euclid,
            DistanceMetric::Dot,
            DistanceMetric::Manhattan,
            DistanceMetric::Cosine,
        ] {
            let kernel = selected();
            group.bench_function(BenchmarkId::new(format!("{metric:?}"), dim), |bencher| {
                bencher.iter(|| black_box(kernel.distance(metric, &a, &b)));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_selected_kernel);
criterion_main!(benches);
