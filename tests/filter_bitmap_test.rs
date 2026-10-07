//! Pre-filter bitmap integration tests and streaming top-K regression.

use std::collections::HashSet;

use sim_vec::storage::CollectionStore;
use sim_vec::types::{
    CollectionConfig, DistanceMetric, FilterCondition, HnswConfig, PointId, SearchQuery,
    VectorFilter, VectorPoint,
};

const DIM: usize = 8;

fn hnsw_config() -> HnswConfig {
    HnswConfig {
        m: 8,
        ef_construct: 16,
        ef_search: 16,
        ..HnswConfig::default()
    }
}

fn config(metric: DistanceMetric, dim: usize) -> CollectionConfig {
    CollectionConfig::new(dim, metric)
}

/// Deterministic vector with unique perturbation so every id has a distinct direction.
fn unit(i: u64, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[(i as usize) % dim] = 1.0;
    for (j, x) in v.iter_mut().enumerate() {
        if j != (i as usize) % dim {
            *x = ((i as usize * 7 + j * 13) % 100) as f32 * 0.001;
        }
    }
    v
}

fn blob_point(id: u64, dim_values: &[f32], tag: &str) -> VectorPoint {
    VectorPoint::new(id, dim_values.to_vec()).with_payload_kv("tag", serde_json::json!(tag))
}

/// Two well-separated blobs tagged by payload `tag` = "special"/"common".
fn clustered_points(n_special: usize, n_common: usize) -> Vec<VectorPoint> {
    let mut out = Vec::new();
    let mut id = 0u64;
    for (center, tag, n) in [
        ([0.0f32; DIM], "special", n_special),
        ([50.0; DIM], "common", n_common),
    ] {
        for i in 0..n {
            let mut v = center;
            v[i % DIM] += (i % 7) as f32 * 0.1 + i as f32 * 0.003;
            out.push(blob_point(id, &v, tag));
            id += 1;
        }
    }
    out
}

fn publish_hnsw(store: &CollectionStore) {
    store.build_index().expect("HNSW build must succeed");
}

/// Batch upsert through the WAL-backed apply path: one WAL append and one
/// meta.bin save per batch instead of per-point saves.
fn upsert_batch(store: &CollectionStore, points: &[VectorPoint]) {
    let ops: Vec<sim_vec::storage::WalRecord> = points
        .iter()
        .map(|p| sim_vec::storage::WalRecord::Upsert {
            point: sim_vec::storage::WalPoint::from_point(p).unwrap(),
        })
        .collect();
    store.apply_ops(&ops).unwrap();
}

/// Oracle: sort ids by descending score to `query_vector` among `ids` and
/// return them in rank order.
fn oracle(query_vector: &[f32], ids: &[u64]) -> Vec<u64> {
    let mut scored: Vec<(f32, u64)> = ids
        .iter()
        .copied()
        .map(|id| {
            let dist =
                sim_vec::distance::naive::distance_cosine(query_vector, &unit(id, DIM));
            (
                sim_vec::distance::to_score(DistanceMetric::Cosine, dist),
                id,
            )
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, id)| id).collect()
}

#[test]
fn test_prefilter_bitmap_high_selectivity_recall() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, DIM)
            .with_index_type(sim_vec::types::IndexType::HNSW)
            .with_hnsw(hnsw_config()),
    )
    .unwrap();

    let points = clustered_points(10, 40);
    upsert_batch(&store, &points);
    publish_hnsw(&store);

    let query_vec = [0.0f32; DIM];
    let filter = VectorFilter::new().must(FilterCondition::match_value("tag", "special"));

    let indexed = store
        .search(
            &SearchQuery::new(query_vec.to_vec(), 10)
                .with_filter(filter.clone())
                .with_knn(10, Some(16)),
        )
        .unwrap();

    // Drop index and re-search with exact scan + same filter: must agree on
    // returned ids (top-10 among all special points by true cosine score).
    store.drop_index().unwrap();
    let exact = store
        .search(&SearchQuery::new(query_vec.to_vec(), 10).with_filter(filter.clone()))
        .unwrap();

    let indexed_ids: HashSet<u64> = indexed
        .iter()
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    let exact_ids: HashSet<u64> = exact
        .iter()
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    assert_eq!(
        indexed_ids, exact_ids,
        "HNSW pre-filter must match exact-scan oracle on tag filter"
    );

    // Re-open: bitmap must be rebuilt from the payload file.
    drop(store);
    let reopened = CollectionStore::open(dir.path().join("col")).unwrap();
    let reopened = reopened
        .search(&SearchQuery::new(query_vec.to_vec(), 10).with_filter(filter.clone()))
        .unwrap();
    let reopened_ids: HashSet<u64> = reopened
        .iter()
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    assert_eq!(
        indexed_ids, reopened_ids,
        "reopened collection must produce the same filtered results"
    );
}

#[test]
fn test_filter_bitmap_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    {
        let store = CollectionStore::create(
            &store_dir,
            "col",
            &config(DistanceMetric::Cosine, DIM)
                .with_index_type(sim_vec::types::IndexType::FLAT),
        )
        .unwrap();
        for i in 0..10u64 {
            let tag = if i < 3 { "red" } else { "blue" };
            store.upsert(&blob_point(i, &unit(i, DIM), tag)).unwrap();
        }
    }

    let store = CollectionStore::open(&store_dir).unwrap();
    let query = unit(0, DIM);
    let filter = VectorFilter::new().must(FilterCondition::match_value("tag", "red"));
    let results = store
        .search(&SearchQuery::new(query.clone(), 5).with_filter(filter))
        .unwrap();
    let got: Vec<u64> = results
        .iter()
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    let expected: HashSet<u64> = [0, 1, 2].into_iter().collect();
    let got_set: HashSet<u64> = got.iter().copied().collect();
    assert_eq!(
        got_set, expected,
        "reopen must expose bitmap-filtered results"
    );
}

#[test]
fn test_filter_bitmap_after_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    {
        let store = CollectionStore::create(
            &store_dir,
            "col",
            &config(DistanceMetric::Cosine, DIM)
                .with_index_type(sim_vec::types::IndexType::FLAT),
        )
        .unwrap();
        for i in 0..20u64 {
            let tag = if i < 5 { "red" } else { "blue" };
            store.upsert(&blob_point(i, &unit(i, DIM), tag)).unwrap();
        }
        for i in 0..5u64 {
            store.delete(&PointId::Num(i)).unwrap();
        }
        store.compact().unwrap();
    }

    let store = CollectionStore::open(&store_dir).unwrap();
    let query = unit(5, DIM);
    let filter = VectorFilter::new().must(FilterCondition::match_value("tag", "blue"));
    let results = store
        .search(&SearchQuery::new(query.clone(), 10).with_filter(filter))
        .unwrap();
    let got: HashSet<u64> = results
        .iter()
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    let expected_ids: Vec<u64> = (5..20).collect();
    let expected: HashSet<u64> = oracle(&query, &expected_ids).into_iter().take(10).collect();
    assert_eq!(got, expected, "compacted bitmap must reflect removed slots");
}

#[test]
fn test_overwrite_changes_bitmap_membership() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, DIM).with_index_type(sim_vec::types::IndexType::FLAT),
    )
    .unwrap();
    store.upsert(&blob_point(0, &unit(0, DIM), "red")).unwrap();

    let filter_red = VectorFilter::new().must(FilterCondition::match_value("tag", "red"));
    let filter_blue = VectorFilter::new().must(FilterCondition::match_value("tag", "blue"));

    let r = store
        .search(&SearchQuery::new(unit(0, DIM), 5).with_filter(filter_red.clone()))
        .unwrap();
    assert_eq!(r.len(), 1, "red filter must find the point");

    store.upsert(&blob_point(0, &unit(0, DIM), "blue")).unwrap();
    let r = store
        .search(&SearchQuery::new(unit(0, DIM), 5).with_filter(filter_red))
        .unwrap();
    assert!(r.is_empty(), "overwrite must remove point from red bitmap");

    let r = store
        .search(&SearchQuery::new(unit(0, DIM), 5).with_filter(filter_blue))
        .unwrap();
    assert_eq!(r.len(), 1, "blue filter must find the updated point");
}

#[test]
fn test_filter_bitmap_with_non_indexed_conditions() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, DIM).with_index_type(sim_vec::types::IndexType::FLAT),
    )
    .unwrap();
    for i in 0..10u64 {
        let price = (i + 1) as f64;
        store
            .upsert(
                &VectorPoint::new(i, unit(i, DIM))
                    .with_payload_kv("price", serde_json::json!(price)),
            )
            .unwrap();
    }

    let filter = VectorFilter::new().must(FilterCondition::range(
        "price",
        sim_vec::types::RangeCondition::new().gte(5.0),
    ));
    let results = store
        .search(&SearchQuery::new(unit(0, DIM), 10).with_filter(filter))
        .unwrap();
    let got: HashSet<u64> = results
        .iter()
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    let price_ids: Vec<u64> = (0..10).filter(|i| *i >= 4).collect();
    let expected: HashSet<u64> = oracle(&unit(0, DIM), &price_ids)
        .into_iter()
        .take(10)
        .collect();
    assert_eq!(got, expected, "range filter must fall back to post-filter");
}

#[test]
fn test_streaming_topk_matches_collect_all() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, DIM).with_index_type(sim_vec::types::IndexType::FLAT),
    )
    .unwrap();
    let points: Vec<VectorPoint> = (0..50u64)
        .map(|i| VectorPoint::new(i, unit(i, DIM)))
        .collect();
    upsert_batch(&store, &points);

    let ids: Vec<u64> = (0..50).collect();
    for limit in [3usize, 10, 20] {
        for offset in [0usize, 5, 40] {
            let query = unit(42, DIM);
            let indexed = store
                .search(&SearchQuery::new(query.clone(), limit).with_offset(offset))
                .unwrap();
            let expected = oracle(&query, &ids);
            let expected_slice: HashSet<u64> =
                expected.into_iter().skip(offset).take(limit).collect();
            let got: HashSet<u64> = indexed
                .iter()
                .map(|r| r.id.to_string().parse().unwrap())
                .collect();
            assert_eq!(
                got, expected_slice,
                "streaming top-K must match oracle (limit={limit}, offset={offset})"
            );
        }
    }
}

#[test]
fn test_streaming_topk_with_filter() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, DIM).with_index_type(sim_vec::types::IndexType::FLAT),
    )
    .unwrap();
    let points: Vec<VectorPoint> = (0..50u64)
        .map(|i| {
            let tag = if i % 10 == 0 { "rare" } else { "common" };
            VectorPoint::new(i, unit(i, DIM)).with_payload_kv("tag", serde_json::json!(tag))
        })
        .collect();
    upsert_batch(&store, &points);

    let query = unit(42, DIM);
    let filter = VectorFilter::new().must(FilterCondition::match_value("tag", "rare"));
    let indexed = store
        .search(&SearchQuery::new(query.clone(), 5).with_filter(filter))
        .unwrap();

    let special_ids: Vec<u64> = (0..50).filter(|i| i % 10 == 0).collect();
    let expected_ids = oracle(&query, &special_ids)
        .into_iter()
        .take(5)
        .collect::<HashSet<_>>();
    let got_ids: HashSet<u64> = indexed
        .iter()
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    assert_eq!(
        got_ids, expected_ids,
        "streaming top-K with filter must match oracle"
    );
}

#[test]
fn test_prefilter_bitmap_falls_back_for_complex_filter() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, DIM).with_index_type(sim_vec::types::IndexType::FLAT),
    )
    .unwrap();
    for i in 0..10u64 {
        store
            .upsert(
                &VectorPoint::new(i, unit(i, DIM)).with_payload_kv("tag", serde_json::json!("red")),
            )
            .unwrap();
    }

    let filter = VectorFilter::new()
        .must(FilterCondition::match_value("tag", "red"))
        .must_not(FilterCondition::match_value("tag", "red"));
    let results = store
        .search(&SearchQuery::new(unit(0, DIM), 10).with_filter(filter))
        .unwrap();
    assert!(
        results.is_empty(),
        "must+must_not on same field must return nothing"
    );
}
