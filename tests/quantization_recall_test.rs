//! Quantized search recall guard tests: the quantized path (scan and HNSW
//! traversal) must recover a high fraction of the exact top-k, and the
//! `quantization.ignore` / `exact` controls must fall back to the exact path.

use simvec::storage::CollectionStore;
use simvec::types::{
    CollectionConfig, CompressionRatio, DistanceMetric, QuantizationConfig, QuantizationSearchParams,
    SearchQuery,
};

/// Clustered dataset: `clusters` blobs around unit-axis centers shifted to be
/// zero-mean, so binary sign bits vary (all-non-negative data makes binary
/// codes degenerate). Deterministic.
fn sample(i: u64, dim: usize) -> Vec<f32> {
    let mut v = vec![-0.5f32; dim];
    let center = (i / 50) % (dim as u64);
    v[center as usize] = 1.0;
    for (j, x) in v.iter_mut().enumerate() {
        let noise = ((i as usize * 31 + j * 17) % 23) as f32 * 0.02 - 0.22;
        *x += noise;
    }
    v
}

fn seed(store: &CollectionStore, n: u64, dim: usize) {
    let ops: Vec<simvec::storage::WalRecord> = (0..n)
        .map(|i| simvec::storage::WalRecord::Upsert {
            point: simvec::storage::WalPoint::from_point(&simvec::types::VectorPoint::new(
                i,
                sample(i, dim),
            ))
            .unwrap(),
        })
        .collect();
    store.apply_ops(&ops).unwrap();
}

fn exact_topk(store: &CollectionStore, query: &[f32], k: usize) -> Vec<String> {
    store
        .search(
            &SearchQuery::new(query.to_vec(), k)
                .with_quantization_params(QuantizationSearchParams {
                    ignore: true,
                    ..Default::default()
                }),
        )
        .unwrap()
        .into_iter()
        .map(|r| r.id.to_string())
        .collect()
}

fn recall(got: &[String], truth: &[String]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let hit = got.iter().filter(|g| truth.contains(g)).count();
    hit as f64 / truth.len() as f64
}

fn recall_guard_case(metric: DistanceMetric, qc: QuantizationConfig) {
    // Binary 1-bit codes are coarse; the binary case passes an explicit
    // larger oversampling to clear the guard (the default 4x yields ~0.7).
    let oversampling = matches!(qc.quant_type, simvec::types::QuantizationType::Binary { .. })
        .then_some(8.0f64);
    let dim = 16;
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &CollectionConfig::new(dim, metric).with_quantization(qc),
    )
    .unwrap();
    seed(&store, 400, dim);
    store.build_quantization().unwrap();
    assert!(store.has_quantization(), "quantization must be ready");

    let query = sample(7, dim);
    let truth = exact_topk(&store, &query, 10);

    // Default quantized path (rescore on to keep scores exact).
    let got = store
        .search(
            &SearchQuery::new(query.clone(), 10)
                .with_quantization_params(QuantizationSearchParams {
                    rescore: Some(true),
                    oversampling,
                    ..Default::default()
                }),
        )
        .unwrap();
    let got_ids: Vec<String> = got.iter().map(|r| r.id.to_string()).collect();
    let r = recall(&got_ids, &truth);
    assert!(
        r >= 0.8,
        "{metric:?} quantized recall@10 {r} below guard 0.8 (got {got_ids:?}, truth {truth:?})"
    );

    // ignore=true must match the exact path exactly.
    let ignored = store
        .search(
            &SearchQuery::new(query.clone(), 10).with_quantization_params(QuantizationSearchParams {
                ignore: true,
                ..Default::default()
            }),
        )
        .unwrap();
    let ignored_ids: Vec<String> = ignored.iter().map(|r| r.id.to_string()).collect();
    assert_eq!(ignored_ids, truth, "ignore=true must use the exact path");

    // exact=true must match the exact path exactly.
    let exact = store
        .search(&SearchQuery::new(query.clone(), 10).with_exact(true))
        .unwrap();
    let exact_ids: Vec<String> = exact.iter().map(|r| r.id.to_string()).collect();
    assert_eq!(exact_ids, truth, "exact=true must use the exact path");
}

#[test]
fn recall_guard_scalar_cosine() {
    recall_guard_case(DistanceMetric::Cosine, QuantizationConfig::scalar(0.99));
}

#[test]
fn recall_guard_binary_cosine() {
    recall_guard_case(DistanceMetric::Cosine, QuantizationConfig::binary());
}

#[test]
fn recall_guard_product_euclid() {
    recall_guard_case(
        DistanceMetric::Euclid,
        QuantizationConfig::product(CompressionRatio::X8),
    );
}
