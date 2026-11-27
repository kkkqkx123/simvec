//! Kind store tests: kind validation, recall ordering and restart durability.

use std::sync::Arc;

use super::*;
use crate::error::VectorSearchError;
use crate::types::{
    CollectionConfig, DistanceMetric, IndexConfig, MultiConfig, PointId, SearchQuery, VectorData,
    VectorKind, VectorPoint,
};
use crate::{LocalVectorEngine, VectorEngine};

fn sparse_config() -> CollectionConfig {
    CollectionConfig::with_sparse(DistanceMetric::Dot)
}

fn multi_config() -> CollectionConfig {
    CollectionConfig::with_multi(
        2,
        DistanceMetric::Dot,
        MultiConfig::new(4),
    )
}

fn sparse_point(id: &str, pairs: &[(u32, f32)]) -> VectorPoint {
    let (indices, values): (Vec<u32>, Vec<f32>) = pairs.iter().copied().unzip();
    VectorPoint::with_data(id, VectorData::sparse(indices, values))
}

fn multi_point(id: &str, vectors: Vec<Vec<f32>>) -> VectorPoint {
    VectorPoint::with_data(id, VectorData::multi(vectors))
}

#[test]
fn kind_configs_validate() {
    let mut config = sparse_config();
    config.distance = DistanceMetric::Euclid;
    assert!(config.validate_kind().is_err());

    assert!(CollectionConfig::with_sparse(DistanceMetric::Cosine)
        .validate_kind()
        .is_ok());

    let mut config = multi_config();
    config.index = IndexConfig::Hnsw(crate::types::HnswConfig::default());
    assert!(config.validate_kind().is_err());

    assert!(sparse_config().validate_kind().is_ok());
    assert!(multi_config().validate_kind().is_ok());
}

#[test]
fn dense_config_rejects_kind_params() {
    let mut config = CollectionConfig::new(3, DistanceMetric::Cosine);
    config.multi_config = Some(MultiConfig::new(2));
    assert!(config.validate_kind().is_err());
}

#[test]
fn sparse_weights_allow_negative() {
    // Negative weights are valid: without posting pruning they cannot hide
    // an otherwise-matching candidate.
    let data = VectorData::sparse(vec![1, 2], vec![1.0, -0.5]);
    assert!(data.validate(100, 1).is_ok());
}

#[test]
fn sparse_empty_and_zero_weight_queries_return_no_results() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(tmp.path()).unwrap();
    engine.create_collection("kb", &sparse_config()).unwrap();
    engine
        .upsert_batch(
            "kb",
            &[sparse_point("p1", &[(1, 1.0), (2, 2.0)])],
        )
        .unwrap();

    let empty = SearchQuery::with_data(
        VectorData::sparse(vec![], vec![]),
        10,
    );
    assert!(engine.search("kb", &empty).unwrap().is_empty());

    let zero = SearchQuery::with_data(
        VectorData::sparse(vec![1, 2], vec![0.0, 0.0]),
        10,
    );
    assert!(engine.search("kb", &zero).unwrap().is_empty());
}

#[test]
fn sparse_recall_orders_by_weighted_dot() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(tmp.path()).unwrap();
    engine.create_collection("kb", &sparse_config()).unwrap();
    engine
        .upsert_batch(
            "kb",
            &[
                sparse_point("weak", &[(1, 0.5), (2, 0.5)]),
                sparse_point("strong", &[(1, 2.0), (3, 1.0)]),
            ],
        )
        .unwrap();
    let query = SearchQuery::with_data(VectorData::sparse(vec![1, 3], vec![1.0, 1.0]), 2);
    let results = engine.search("kb", &query).unwrap();
    // strong: 1*2 + 1*1 = 3, weak: 1*0.5 = 0.5
    assert_eq!(results[0].id, PointId::from("strong"));
    assert_eq!(results[1].id, PointId::from("weak"));
    assert!((results[0].score - 3.0).abs() < 1e-5);
    assert!((results[1].score - 0.5).abs() < 1e-5);
}

#[test]
fn multi_recall_scores_with_max_sim() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(tmp.path()).unwrap();
    engine.create_collection("kb", &multi_config()).unwrap();
    engine
        .upsert_batch(
            "kb",
            &[
                multi_point("a", vec![vec![1.0, 0.0], vec![0.0, 0.5]]),
                multi_point("b", vec![vec![0.0, 1.0], vec![0.0, 1.0]]),
            ],
        )
        .unwrap();
    let query = SearchQuery::with_data(VectorData::multi(vec![vec![1.0, 0.0]]), 2);
    let results = engine.search("kb", &query).unwrap();
    // a: max(1.0, 0.0) = 1.0, b: max(0.0, 0.0) = 0.0
    assert_eq!(results[0].id, PointId::from("a"));
    assert!((results[0].score - 1.0).abs() < 1e-5);
    assert!(results[1].score.abs() < 1e-5);
}

#[test]
fn kind_payload_must_match_collection_kind() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(tmp.path()).unwrap();
    engine.create_collection("kb", &sparse_config()).unwrap();

    let err = engine
        .upsert("kb", VectorPoint::new("dense-in-sparse", vec![1.0, 2.0]))
        .unwrap_err();
    assert!(matches!(
        err,
        VectorSearchError::InvalidConfig(_) | VectorSearchError::Internal(_)
    ));

    let query = SearchQuery::new(vec![1.0, 2.0], 2);
    assert!(engine.search("kb", &query).is_err());
}

#[test]
fn kind_collections_recover_from_wal() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = LocalVectorEngine::open(tmp.path()).unwrap();
        engine.create_collection("kb", &sparse_config()).unwrap();
        engine
            .upsert_batch(
                "kb",
                &[
                    sparse_point("a", &[(1, 1.0)]),
                    sparse_point("b", &[(2, 2.0)]),
                ],
            )
            .unwrap();
        engine.delete("kb", &PointId::from("a")).unwrap();
        engine
            .set_payload_field("kb", &PointId::from("b"), "lang".into(), "zh".into())
            .unwrap();
    }

    let engine = LocalVectorEngine::open(tmp.path()).unwrap();
    assert!(engine.collection_exists("kb"));
    assert_eq!(engine.count("kb").unwrap(), 1);
    assert!(engine.get("kb", &PointId::from("a")).unwrap().is_none());

    let point = engine.get("kb", &PointId::from("b")).unwrap().unwrap();
    assert!(point.data().is_some());

    let query = SearchQuery::with_data(VectorData::sparse(vec![2], vec![1.0]), 10);
    let results = engine.search("kb", &query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, PointId::from("b"));
    assert_eq!(
        results[0].payload.as_ref().and_then(|p| p.get("lang")),
        Some(&serde_json::json!("zh"))
    );
}

#[test]
fn kind_delete_by_filter_and_scroll() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(tmp.path()).unwrap();
    engine.create_collection("kb", &sparse_config()).unwrap();
    engine
        .upsert_batch(
            "kb",
            &[
                sparse_point("a", &[(1, 1.0)]),
                sparse_point("b", &[(2, 2.0)]),
            ],
        )
        .unwrap();
    engine
        .set_payload_field("kb", &PointId::from("a"), "lang".into(), "en".into())
        .unwrap();

    let filter = crate::types::VectorFilter::new()
        .must(crate::types::FilterCondition::match_value("lang", "en"));
    let (page, _) = engine
        .scroll_by_filter("kb", &filter, 10, None, Some(true), Some(true))
        .unwrap();
    assert_eq!(page.len(), 1);
    assert_eq!(page[0].id, PointId::from("a"));

    let deleted = engine
        .delete_by_filter("kb", &crate::types::VectorFilter::default())
        .unwrap();
    assert_eq!(deleted, 2);
    assert_eq!(engine.count("kb").unwrap(), 0);
}

#[test]
fn local_engine_claims_all_kinds() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(tmp.path()).unwrap();
    let engine_trait: Arc<dyn VectorEngine> = Arc::new(engine);
    let kinds = engine_trait.supported_kinds();
    assert!(kinds.contains(&VectorKind::Dense));
    assert!(kinds.contains(&VectorKind::Sparse));
    assert!(kinds.contains(&VectorKind::Multi));
}

#[test]
fn coordinated_upsert_is_durable() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = LocalVectorEngine::open(tmp.path()).unwrap();
        engine
            .create_collection("kb", &sparse_config())
            .expect("create");
        engine
            .apply_txn(vec![crate::TxnOp::Upsert {
                collection: "kb".to_string(),
                point: sparse_point("p1", &[(1, 1.0), (3, 2.0)]),
            }])
            .expect("apply txn");
    }
    let engine = LocalVectorEngine::open(tmp.path()).unwrap();
    assert_eq!(engine.count("kb").unwrap(), 1);
    let hits = engine
        .search("kb", &SearchQuery::with_data(VectorData::sparse(vec![1, 3], vec![1.0, 1.0]), 5))
        .unwrap();
    assert_eq!(hits.len(), 1);
}

#[test]
fn sparse_unsorted_input_is_normalized() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(tmp.path()).unwrap();
    engine
        .create_collection("kb", &sparse_config())
        .expect("create");
    engine
        .upsert(
            "kb",
            VectorPoint::with_data("p1", VectorData::sparse(vec![3, 1, 2], vec![3.0, 1.0, 2.0])),
        )
        .expect("unsorted sparse upsert");
    let hits = engine
        .search("kb", &SearchQuery::with_data(VectorData::sparse(vec![2], vec![1.0]), 5))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].score, 2.0);
}

#[test]
fn exact_store_compaction_reclaims_slots_and_cap_is_enforced() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = sparse_config();
    config.max_points = Some(2);
    let engine = LocalVectorEngine::open(tmp.path()).unwrap();
    engine.create_collection("kb", &config).expect("create");
    engine.upsert("kb", sparse_point("p1", &[(1, 1.0)])).unwrap();
    engine.upsert("kb", sparse_point("p2", &[(2, 1.0)])).unwrap();
    assert!(engine.upsert("kb", sparse_point("p3", &[(3, 1.0)])).is_err());
    engine.delete("kb", &PointId::from("p1")).unwrap();
    assert!(engine.compact_collection("kb").unwrap() >= 1);
    assert_eq!(engine.count("kb").unwrap(), 1);
    drop(engine);
    let reopened = LocalVectorEngine::open(tmp.path()).unwrap();
    assert_eq!(reopened.count("kb").unwrap(), 1);
}

#[test]
fn multi_vector_try_new_rejects_ragged_and_zero_dimension() {
    assert!(VectorData::try_multi(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).is_ok());
    assert!(VectorData::try_multi(vec![vec![1.0, 0.0], vec![0.0]]).is_err());
    assert!(VectorData::try_multi(vec![vec![]]).is_err());
    assert!(VectorData::try_multi(vec![]).is_err());

    // A deserialized ragged payload must be rejected by validation too.
    let ragged = VectorData::Multi(crate::types::MultiVector {
        flat: vec![1.0, 0.0, 1.0],
        dim: 2,
    });
    assert!(ragged.validate(2, 4).is_err());
}

#[test]
fn filtered_sparse_search_matches_post_filter_and_cost_paths_agree() {
    use crate::types::{FilterCondition, PayloadSelector, VectorFilter};

    let tmp = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(tmp.path()).unwrap();
    engine.create_collection("kb", &sparse_config()).unwrap();
    let mut points = Vec::new();
    for i in 0..40u32 {
        let lang = if i % 2 == 0 { "en" } else { "fr" };
        points.push(
            sparse_point(&format!("p{i}",), &[(i % 5, 1.0 + (i % 3) as f32)])
                .with_payload_kv("lang", serde_json::json!(lang)),
        );
    }
    engine.upsert_batch("kb", &points).unwrap();

    let query_vec = VectorData::sparse(vec![0, 1, 2, 3, 4], vec![1.0; 5]);
    let filter = VectorFilter::new().must(FilterCondition::match_value("lang", "en"));

    // Filtered path (lazy accept over posting hits).
    let filtered = engine
        .search("kb", &SearchQuery::with_data(query_vec.clone(), 100).with_filter(filter.clone()))
        .unwrap();
    // Post-filter path: search everything, then keep matching payloads.
    let unfiltered = engine
        .search(
            "kb",
            &SearchQuery::with_data(query_vec.clone(), 100)
                .with_payload_selector(PayloadSelector::default()),
        )
        .unwrap();
    let mut expected: Vec<_> = unfiltered
        .into_iter()
        .filter(|r| {
            r.payload
                .as_ref()
                .and_then(|p| p.get("lang"))
                .and_then(|v| v.as_str())
                == Some("en")
        })
        .collect();
    // Engine tie-break is ascending insertion slot, which here follows the
    // numeric suffix of the id (p0, p1, ... inserted in order).
    let slot_order = |r: &crate::types::SearchResult| -> u64 {
        r.id
            .to_string()
            .trim_start_matches('p')
            .parse()
            .unwrap_or(u64::MAX)
    };
    expected.sort_by(|a, b| b.score.total_cmp(&a.score).then(slot_order(a).cmp(&slot_order(b))));

    let got_ids: Vec<_> = filtered.iter().map(|r| r.id.to_string()).collect();
    let expected_ids: Vec<_> = expected.iter().map(|r| r.id.to_string()).collect();
    assert_eq!(got_ids, expected_ids);
    assert!(filtered.iter().all(|r| r.score != 0.0));
}

#[test]
fn invalid_upsert_never_reaches_wal_and_reopen_succeeds() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = LocalVectorEngine::open(tmp.path()).unwrap();
        engine.create_collection("kb", &sparse_config()).expect("create");
        // A dense payload on a sparse collection is rejected before append.
        let bad = VectorPoint::with_data("bad", VectorData::Dense(vec![1.0, 2.0]));
        assert!(engine.upsert("kb", bad).is_err());
        assert_eq!(engine.count("kb").unwrap(), 0);
    }
    // Replay of a WAL with no committed bad record opens cleanly.
    let reopened = LocalVectorEngine::open(tmp.path()).unwrap();
    assert_eq!(reopened.count("kb").unwrap(), 0);
}

// Storage layout -----------------------------------------------------------

fn sparse_wide(id: &str, terms: u32, weight: f32) -> VectorPoint {
    let pairs: Vec<(u32, f32)> = (0..terms).map(|term| (term, weight)).collect();
    sparse_point(id, &pairs)
}

#[test]
fn kind_restart_loads_the_posting_dictionary() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("kb");
    let config = sparse_config();
    let storage = crate::types::StorageConfig::default();
    let (blob_bytes, posting_bytes);
    {
        let store = KindStore::create(&dir, "kb", &config, &storage).unwrap();
        let points: Vec<VectorPoint> = (0..40u32)
            .map(|i| sparse_wide(&format!("p{i}"), 64, 1.0))
            .collect();
        store.upsert_batch(&points).unwrap();
        store.checkpoint().unwrap();
        (blob_bytes, posting_bytes) = store.storage_bytes();
        assert!(blob_bytes > 0, "vectors must land in points.bin");
        assert!(posting_bytes > 0, "postings must land in postings/");
        // The snapshot is a directory: it references blobs instead of copying
        // the 200-byte vectors they hold.
        let snapshot = std::fs::metadata(dir.join("kind_snapshot.bin"))
            .unwrap()
            .len();
        assert!(
            snapshot * 4 < blob_bytes,
            "snapshot ({snapshot} B) must stay far below the vector bytes ({blob_bytes} B)"
        );
    }

    let reopened = KindStore::open(&dir, &storage).unwrap();
    assert_eq!(reopened.count(), 40);
    assert!(
        reopened.postings_restored(),
        "a checkpointed collection must load its dictionary, not re-invert every point"
    );
    assert_eq!(reopened.storage_bytes(), (blob_bytes, posting_bytes));
    let query = SearchQuery::with_data(VectorData::sparse(vec![3, 7], vec![1.0, 1.0]), 2);
    assert_eq!(reopened.search(&query).unwrap().len(), 2);
}

#[test]
fn kind_stale_posting_manifest_is_rebuilt() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("kb");
    let config = sparse_config();
    let storage = crate::types::StorageConfig::default();
    {
        let store = KindStore::create(&dir, "kb", &config, &storage).unwrap();
        let points: Vec<VectorPoint> = (0..10u32)
            .map(|i| sparse_wide(&format!("p{i}"), 8, 1.0))
            .collect();
        store.upsert_batch(&points).unwrap();
        store.checkpoint().unwrap();
    }
    // Simulate the crash window: the manifest is gone, so it can no longer
    // vouch for the buckets.
    std::fs::remove_file(dir.join("postings").join("manifest.bin")).unwrap();

    let reopened = KindStore::open(&dir, &storage).unwrap();
    assert!(!reopened.postings_restored());
    assert_eq!(reopened.count(), 10);
    // Rebuilt from the blobs, so queries still find every point.
    let query = SearchQuery::with_data(VectorData::sparse(vec![2], vec![1.0]), 10);
    assert_eq!(reopened.search(&query).unwrap().len(), 10);
}

#[test]
fn kind_deleted_points_leave_no_phantom_hits() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("kb");
    let config = sparse_config();
    let storage = crate::types::StorageConfig::default();
    let query = SearchQuery::with_data(VectorData::sparse(vec![1], vec![1.0]), 10);
    {
        let store = KindStore::create(&dir, "kb", &config, &storage).unwrap();
        let points: Vec<VectorPoint> = (0..3u32)
            .map(|i| sparse_point(&format!("p{i}"), &[(1, 1.0 + i as f32)]))
            .collect();
        store.upsert_batch(&points).unwrap();
        store.checkpoint().unwrap();
        store.delete(&PointId::from("p1")).unwrap();
        // The tombstoned slot's postings are still on disk; the scan must skip
        // them.
        assert_eq!(store.search(&query).unwrap().len(), 2);
    }
    let reopened = KindStore::open(&dir, &storage).unwrap();
    assert_eq!(reopened.count(), 2);
    let hits = reopened.search(&query).unwrap();
    assert_eq!(hits.len(), 2, "stale postings must not resurrect deleted points");
    assert!(hits.iter().all(|hit| hit.id != PointId::from("p1")));
}

#[test]
fn kind_compaction_rewrites_blobs_and_postings() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("kb");
    let config = sparse_config();
    let storage = crate::types::StorageConfig::default();
    let query = SearchQuery::with_data(VectorData::sparse(vec![1], vec![1.0]), 10);
    let store = KindStore::create(&dir, "kb", &config, &storage).unwrap();
    let points: Vec<VectorPoint> = (0..20u32)
        .map(|i| sparse_wide(&format!("p{i}"), 16, 1.0))
        .collect();
    store.upsert_batch(&points).unwrap();
    store.checkpoint().unwrap();
    let before = store.storage_bytes();
    for i in 0..10u32 {
        store.delete(&PointId::from(format!("p{i}"))).unwrap();
    }
    let reclaimed = store.compact().unwrap();
    assert_eq!(reclaimed, 10);
    let after = store.storage_bytes();
    assert!(
        after.0 <= before.0 / 2,
        "compaction must drop the bytes of deleted points: {before:?} -> {after:?}"
    );
    assert_eq!(store.search(&query).unwrap().len(), 10);

    let reopened = KindStore::open(&dir, &storage).unwrap();
    assert_eq!(reopened.count(), 10);
    assert!(reopened.postings_restored());
    assert_eq!(reopened.storage_bytes(), after);
    assert_eq!(reopened.search(&query).unwrap().len(), 10);
}

#[test]
fn kind_payload_edits_survive_a_restart_without_touching_vectors() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("kb");
    let config = sparse_config();
    let storage = crate::types::StorageConfig::default();
    {
        let store = KindStore::create(&dir, "kb", &config, &storage).unwrap();
        store
            .upsert_batch(&[sparse_wide("p1", 4, 1.0)])
            .unwrap();
        store.set_payload_fields(
            &PointId::from("p1"),
            [("tag".to_string(), serde_json::json!("a"))]
                .into_iter()
                .collect(),
        )
        .unwrap();
        store.checkpoint().unwrap();
    }
    let reopened = KindStore::open(&dir, &storage).unwrap();
    let point = reopened.get(&PointId::from("p1")).unwrap().unwrap();
    assert_eq!(
        point.payload.as_ref().unwrap()["tag"],
        serde_json::json!("a")
    );
    assert_eq!(point.vectors.len(), 1);
}
