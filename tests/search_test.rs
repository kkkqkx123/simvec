//! Exact scan search integration tests: distance, filtering, score threshold,
//! pagination, and payload/vector trimming.

use simvec::distance::naive;
use simvec::storage::CollectionStore;
use simvec::types::{
    CollectionConfig, ConditionType, DistanceMetric, FilterCondition, PointId, SearchQuery,
    VectorFilter, VectorPoint,
};

fn config(metric: DistanceMetric, dim: usize) -> CollectionConfig {
    CollectionConfig::new(dim, metric)
}

/// A deterministic unit-axis vector with a unique small perturbation, so every
/// id has a distinct direction (no ties, no collinearity).
fn unit(i: u64, dim: usize) -> VectorPoint {
    let mut v = vec![0.0f32; dim];
    v[(i as usize) % dim] = 1.0;
    for (j, x) in v.iter_mut().enumerate() {
        if j != (i as usize) % dim {
            *x = ((i as usize * 7 + j * 13) % 100) as f32 * 0.001;
        }
    }
    VectorPoint::new(i, v)
}

fn color_point(i: u64, dim: usize, color: &str) -> VectorPoint {
    unit(i, dim).with_payload_kv("color", serde_json::json!(color))
}

fn seed(store: &CollectionStore, n: u64, dim: usize, with_color: bool) {
    // One WAL-backed batch: a single WAL append + meta save for the seed set.
    let points: Vec<VectorPoint> = (0..n)
        .map(|i| {
            if with_color {
                color_point(i, dim, if i % 2 == 0 { "red" } else { "blue" })
            } else {
                unit(i, dim)
            }
        })
        .collect();
    let ops: Vec<simvec::storage::WalRecord> = points
        .iter()
        .map(|p| simvec::storage::WalRecord::Upsert {
            point: simvec::storage::WalPoint::from_point(p).unwrap(),
        })
        .collect();
    store.apply_ops(&ops).unwrap();
}

/// Expected cosine ordering for ids in `points`, excluding `exclude`, computed
/// with the naive kernel (tie-break by id ascending).
fn expected_order(query: &[f32], points: &[u64], exclude: &[u64]) -> Vec<u64> {
    let mut v: Vec<(f32, u64)> = points
        .iter()
        .copied()
        .filter(|i| !exclude.contains(i))
        .map(|i| {
            let p = unit(i, query.len());
            let sim = 1.0 - naive::distance_cosine(query, &p.vector);
            (sim, i)
        })
        .collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    v.into_iter().map(|(_, i)| i).collect()
}

#[test]
fn test_topk_cosine() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, 8),
    )
    .unwrap();
    seed(&store, 10, 8, false);

    let query = unit(3, 8).vector;
    let results = store
        .search(&SearchQuery::new(query.clone(), 3).with_vector(true))
        .unwrap();
    assert_eq!(results.len(), 3);
    let expected = expected_order(&query, &(0..10).collect::<Vec<_>>(), &[]);
    let got: Vec<u64> = results
        .iter()
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    assert_eq!(got, expected[..3], "top-3 must match naive ordering");
    assert_eq!(got[0], 3, "exact match is the top hit");
    assert!((results[0].score - 1.0).abs() < 1e-4);
    for pair in results.windows(2) {
        assert!(pair[0].score >= pair[1].score, "scores must descend");
    }
    // with_vector=true returns vectors.
    assert_eq!(results[0].vector, Some(unit(3, 8).vector));
}

#[test]
fn test_topk_l2_and_dot() {
    let dir = tempfile::tempdir().unwrap();

    let store = CollectionStore::create(
        dir.path().join("col_l2"),
        "col_l2",
        &config(DistanceMetric::Euclid, 4),
    )
    .unwrap();
    seed(&store, 10, 4, false);
    let query = unit(7, 4).vector;
    let results = store.search(&SearchQuery::new(query.clone(), 2)).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id.to_string(), "7");
    assert!((results[0].score - 1.0).abs() < 1e-4);

    let store = CollectionStore::create(
        dir.path().join("col_dot"),
        "col_dot",
        &config(DistanceMetric::Dot, 4),
    )
    .unwrap();
    seed(&store, 10, 4, false);
    let query = unit(5, 4).vector;
    let results = store.search(&SearchQuery::new(query.clone(), 2)).unwrap();
    assert_eq!(results[0].id.to_string(), "5");
    // Dot score of the exact match is |v|^2 (the vector is not unit-length).
    let expected = naive::inner_product(&query, &query);
    assert!((results[0].score - expected).abs() < 1e-4);
    assert!(results[0].score > results[1].score);
}

#[test]
fn test_manhattan_topk_and_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col_man"),
        "col_man",
        &config(DistanceMetric::Manhattan, 8),
    )
    .unwrap();
    seed(&store, 10, 8, false);

    // Query is an exact copy of id 4; Manhattan distance to itself is 0 -> score 1.0.
    let query = unit(4, 8).vector;
    let results = store.search(&SearchQuery::new(query.clone(), 3)).unwrap();
    assert_eq!(results.len(), 3, "manhattan top-3");
    assert_eq!(results[0].id.to_string(), "4");
    assert!(
        (results[0].score - 1.0).abs() < 1e-4,
        "exact match score must be 1.0"
    );

    // Oracle: brute-force Manhattan distance via naive kernel, score = 1/(1+sqrt(l1)).
    let oracle: Vec<(f32, u64)> = (0..10)
        .map(|i| {
            let p = unit(i, 8);
            let l1 = naive::distance(DistanceMetric::Manhattan, &query, &p.vector);
            let score = 1.0 / (1.0 + l1.sqrt());
            (score, i)
        })
        .collect();
    let mut oracle_sorted = oracle.clone();
    oracle_sorted.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let expected_ids: Vec<String> = oracle_sorted
        .iter()
        .take(3)
        .map(|(_, id)| id.to_string())
        .collect();
    let got_ids: Vec<String> = results.iter().map(|r| r.id.to_string()).collect();
    assert_eq!(
        got_ids, expected_ids,
        "manhattan ranking must match naive oracle"
    );

    // Scores must be 1/(1+sqrt(l1)) and descending.
    for r in &results {
        let pid: u64 = r.id.to_string().parse().unwrap();
        let l1 = naive::distance(DistanceMetric::Manhattan, &query, &unit(pid, 8).vector);
        let expected_score = 1.0 / (1.0 + l1.sqrt());
        assert!(
            (r.score - expected_score).abs() < 1e-4,
            "manhattan score id={pid} got {} expected {}",
            r.score,
            expected_score
        );
    }
    for pair in results.windows(2) {
        assert!(pair[0].score >= pair[1].score, "scores must descend");
    }

    // Threshold filtering between best and second best keeps exactly the top hit.
    let all = store.search(&SearchQuery::new(query.clone(), 10)).unwrap();
    assert!(all.len() >= 2);
    let threshold = (all[0].score + all[1].score) / 2.0;
    let filtered = store
        .search(&SearchQuery::new(query, 10).with_score_threshold(threshold))
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].id.to_string(), "4");
}

#[test]
fn test_score_threshold_filters_below() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, 8),
    )
    .unwrap();
    seed(&store, 10, 8, false);

    // Threshold between the best and second-best score keeps exactly the top
    // hit (the exact match).
    let query = unit(1, 8).vector;
    let all = store.search(&SearchQuery::new(query.clone(), 10)).unwrap();
    assert_eq!(all[0].id.to_string(), "1");
    let threshold = (all[0].score + all[1].score) / 2.0;
    let results = store
        .search(&SearchQuery::new(query, 10).with_score_threshold(threshold))
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id.to_string(), "1");
}

#[test]
fn test_offset_limit_pagination() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, 8),
    )
    .unwrap();
    seed(&store, 10, 8, false);

    let query = unit(0, 8).vector;
    let all = store.search(&SearchQuery::new(query.clone(), 10)).unwrap();
    assert_eq!(all.len(), 10);

    let page1 = store.search(&SearchQuery::new(query.clone(), 4)).unwrap();
    let page2 = store
        .search(&SearchQuery::new(query.clone(), 4).with_offset(4))
        .unwrap();
    let page3 = store
        .search(&SearchQuery::new(query, 4).with_offset(8))
        .unwrap();
    assert_eq!(page1.len(), 4);
    assert_eq!(page2.len(), 4);
    assert_eq!(page3.len(), 2);
    let got: Vec<u64> = page1
        .iter()
        .chain(page2.iter())
        .chain(page3.iter())
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    let expected: Vec<u64> = all
        .iter()
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    assert_eq!(got, expected, "pages concatenate to the full ranking");
}

#[test]
fn test_filter_post_filtering() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, 8),
    )
    .unwrap();
    seed(&store, 10, 8, true);

    // Query is a blue point; the red filter must exclude it and the top red
    // hit must match the oracle restricted to red points.
    let query = unit(3, 8).vector;
    let reds: Vec<u64> = (0..10).filter(|i| i % 2 == 0).collect();
    let expected = expected_order(&query, &reds, &[]);
    let filter = VectorFilter::new().must(FilterCondition::match_value("color", "red"));
    let results = store
        .search(&SearchQuery::new(query, 10).with_filter(filter))
        .unwrap();
    let got: Vec<u64> = results
        .iter()
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    assert_eq!(got, expected, "filtered ranking must match the oracle");
    assert!(!got.contains(&3));
    // Default with_payload=true attaches payloads.
    assert!(results[0].payload.is_some());
}

#[test]
fn test_with_payload_false_skips_payload() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, 4),
    )
    .unwrap();
    seed(&store, 5, 4, true);

    let query = unit(1, 4).vector;
    let results = store
        .search(&SearchQuery::new(query, 5).with_payload(false))
        .unwrap();
    assert!(results.iter().all(|r| r.payload.is_none()));

    let results = store
        .search(&SearchQuery::new(unit(1, 4).vector, 5))
        .unwrap();
    assert!(results.iter().all(|r| r.payload.is_some()));
}

#[test]
fn test_deleted_points_are_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, 8),
    )
    .unwrap();
    seed(&store, 10, 8, false);
    store.delete(&PointId::Num(3)).unwrap();
    store.delete(&PointId::Num(7)).unwrap();

    let query = unit(3, 8).vector;
    let results = store.search(&SearchQuery::new(query.clone(), 10)).unwrap();
    let got: Vec<u64> = results
        .iter()
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    let all: Vec<u64> = (0..10).collect();
    let expected = expected_order(&query, &all, &[3, 7]);
    assert_eq!(got, expected, "tombstoned points must be excluded");
    assert!(!got.contains(&3) && !got.contains(&7));
}

#[test]
fn test_search_after_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, 8),
    )
    .unwrap();
    seed(&store, 20, 8, false);
    for i in 0..5u64 {
        store.delete(&PointId::Num(i)).unwrap();
    }
    // Tombstoned slots stay invisible immediately; physical reclamation is
    // explicit here (the engine schedules it in the background).
    assert_eq!(store.meta().tombstone_count, 5);
    store.compact().unwrap();
    assert_eq!(store.meta().tombstone_count, 0);

    let query = unit(6, 8).vector;
    let results = store.search(&SearchQuery::new(query.clone(), 10)).unwrap();
    let got: Vec<u64> = results
        .iter()
        .map(|r| r.id.to_string().parse().unwrap())
        .collect();
    let all: Vec<u64> = (0..20).collect();
    let expected = expected_order(&query, &all, &[0, 1, 2, 3, 4]);
    assert_eq!(got, expected[..10]);
    assert_eq!(got[0], 6);
}

#[test]
fn test_search_rejects_wrong_dimension() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, 8),
    )
    .unwrap();
    let err = store
        .search(&SearchQuery::new(vec![1.0, 2.0], 5))
        .unwrap_err();
    assert!(matches!(
        err,
        simvec::VectorSearchError::InvalidVectorDimension {
            expected: 8,
            actual: 2
        }
    ));
}

#[test]
fn test_nested_filter_and_geo() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(
        dir.path().join("col"),
        "col",
        &config(DistanceMetric::Cosine, 4),
    )
    .unwrap();
    let mut p = unit(1, 4);
    p.payload = Some(
        serde_json::json!({
            "addresses": [{"city": "paris"}, {"city": "lyon"}],
            "location": {"lat": 48.8566, "lon": 2.3522},
        })
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect(),
    );
    store.upsert(&p).unwrap();
    let mut p2 = unit(2, 4);
    p2.payload = Some(
        serde_json::json!({
            "addresses": [{"city": "berlin"}],
            "location": {"lat": 52.5200, "lon": 13.4050},
        })
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect(),
    );
    store.upsert(&p2).unwrap();

    let nested = VectorFilter::new().must(FilterCondition::new(
        "addresses",
        ConditionType::Nested {
            filter: Box::new(
                VectorFilter::new().must(FilterCondition::match_value("city", "lyon")),
            ),
        },
    ));
    let query = unit(1, 4).vector;
    let results = store
        .search(&SearchQuery::new(query.clone(), 10).with_filter(nested))
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id.to_string(), "1");

    let geo = VectorFilter::new().must(FilterCondition::new(
        "location",
        ConditionType::GeoRadius(simvec::types::GeoRadius::new(
            simvec::types::GeoPoint::new(48.8566, 2.3522),
            1000.0,
        )),
    ));
    let results = store
        .search(&SearchQuery::new(query, 10).with_filter(geo))
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id.to_string(), "1");
}
