//! Tests for `NamedStore` and the engine's named-collection dispatch.

use std::collections::BTreeMap;

use super::NamedStore;
use crate::engine::LocalVectorEngine;
use crate::types::{
    CollectionConfig, DistanceMetric, PointId, SearchQuery, VectorData, VectorPoint,
    VectorSpaceConfig,
};

fn data_dir(root: &std::path::Path) -> std::path::PathBuf {
    crate::storage::Generation::open(root).unwrap().path()
}

fn spaces() -> BTreeMap<String, VectorSpaceConfig> {
    BTreeMap::from([
        (
            "text".to_string(),
            VectorSpaceConfig::dense(4, DistanceMetric::Cosine),
        ),
        (
            "keywords".to_string(),
            VectorSpaceConfig::sparse(DistanceMetric::Dot),
        ),
    ])
}

fn named_config() -> CollectionConfig {
    CollectionConfig::with_named(spaces())
}

fn point(id: &str, text: Vec<f32>, pairs: Vec<(u32, f32)>) -> VectorPoint {
    let (indices, values): (Vec<u32>, Vec<f32>) = pairs.into_iter().unzip();
    VectorPoint::with_named(id, "text", VectorData::dense(text))
        .with_named_vector("keywords", VectorData::sparse(indices, values))
}

fn named_query(space: &str, data: VectorData) -> SearchQuery {
    SearchQuery::with_named(space, data, 10)
}

// Engine-level dispatch -------------------------------------------------

#[test]
fn named_collection_roundtrip_search_and_replay() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LocalVectorEngine::open(dir.path()).expect("open");
    engine
        .create_collection("named", &named_config())
        .expect("create");
    engine
        .upsert(
            "named",
            point("p1", vec![1.0, 0.0, 0.0, 0.0], vec![(1, 1.0), (2, 2.0)]),
        )
        .expect("upsert");
    engine
        .upsert(
            "named",
            point("p2", vec![0.0, 1.0, 0.0, 0.0], vec![(2, 1.0)]),
        )
        .expect("upsert");

    // Dense space: cosine similarity against the text vectors.
    let results = engine
        .search(
            "named",
            &named_query("text", VectorData::dense(vec![1.0, 0.0, 0.0, 0.0])),
        )
        .expect("search");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, PointId::from("p1"));

    // Sparse space: weighted dot product over shared terms.
    let results = engine
        .search(
            "named",
            &named_query("keywords", VectorData::sparse(vec![1, 2], vec![1.0, 1.0])),
        )
        .expect("search");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, PointId::from("p1"));

    // Whole-point replacement removes p1's dense vector.
    engine
        .upsert(
            "named",
            point("p1", vec![0.0, 1.0, 0.0, 0.0], vec![(3, 1.0)]),
        )
        .expect("upsert");
    let results = engine
        .search(
            "named",
            &named_query("text", VectorData::dense(vec![1.0, 0.0, 0.0, 0.0]))
                .with_score_threshold(0.5),
        )
        .expect("search");
    assert!(results.iter().all(|r| r.id != PointId::from("p1")));

    // Persistence: reopen the engine and verify the replayed state.
    drop(engine);
    let engine = LocalVectorEngine::open(dir.path()).expect("reopen");
    assert_eq!(engine.count("named").expect("count"), 2);
    let results = engine
        .search(
            "named",
            &named_query("keywords", VectorData::sparse(vec![1], vec![1.0])),
        )
        .expect("search");
    assert!(results.is_empty());
}

#[test]
fn named_collection_rejects_shape_mismatches() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LocalVectorEngine::open(dir.path()).expect("open");
    engine
        .create_collection("named", &named_config())
        .expect("create");

    // Dense upsert into a named collection.
    assert!(engine
        .upsert("named", VectorPoint::new("p1", vec![1.0, 0.0, 0.0, 0.0]))
        .is_err());
    // Missing space on the point.
    let partial =
        VectorPoint::with_named("p1", "text", VectorData::dense(vec![1.0, 0.0, 0.0, 0.0]));
    assert!(engine.upsert("named", partial).is_err());
    // Dense query without a target.
    let query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0], 10);
    assert!(engine.search("named", &query).is_err());
    // Unknown target space.
    assert!(engine
        .search(
            "named",
            &named_query("nope", VectorData::dense(vec![0.0; 4]))
        )
        .is_err());
    // Dim mismatch inside the space.
    assert!(engine
        .search(
            "named",
            &named_query("text", VectorData::dense(vec![0.0; 3]))
        )
        .is_err());

    // Named payload routed to a single-space collection.
    engine
        .create_collection("dense", &CollectionConfig::new(4, DistanceMetric::Cosine))
        .expect("create");
    assert!(engine
        .upsert(
            "dense",
            point("p1", vec![1.0, 0.0, 0.0, 0.0], vec![(1, 1.0)])
        )
        .is_err());
}

#[test]
fn named_collection_get_delete_payload_and_scroll() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LocalVectorEngine::open(dir.path()).expect("open");
    engine
        .create_collection("named", &named_config())
        .expect("create");
    engine
        .upsert(
            "named",
            point("p1", vec![1.0, 0.0, 0.0, 0.0], vec![(1, 1.0)])
                .with_payload_kv("color", serde_json::json!("red")),
        )
        .expect("upsert");

    let fetched = engine
        .get("named", &PointId::from("p1"))
        .expect("get")
        .expect("present");
    assert!(fetched.vectors.contains_key("text"));
    assert!(fetched.vectors.contains_key("keywords"));

    engine
        .delete("named", &PointId::from("p1"))
        .expect("delete");
    assert_eq!(engine.count("named").expect("count"), 0);

    engine
        .upsert(
            "named",
            point("p2", vec![0.0, 1.0, 0.0, 0.0], vec![(2, 1.0)]),
        )
        .expect("upsert");
    let (page, cursor) = engine
        .scroll("named", 10, None, Some(true), Some(true))
        .expect("scroll");
    assert_eq!(page.len(), 1);
    assert!(cursor.is_some());
    assert!(page[0].vectors.contains_key("keywords"));
}

#[test]
fn named_collection_persists_across_reopen_with_deletes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LocalVectorEngine::open(dir.path()).expect("open");
    engine
        .create_collection("named", &named_config())
        .expect("create");
    for i in 0..5 {
        let text = vec![i as f32, 0.0, 0.0, 0.0];
        engine
            .upsert("named", point(&format!("p{i}"), text, vec![(i, 1.0)]))
            .expect("upsert");
    }
    engine
        .delete_batch("named", &[PointId::from("p0"), PointId::from("p4")])
        .expect("delete batch");
    drop(engine);

    let engine = LocalVectorEngine::open(dir.path()).expect("reopen");
    assert_eq!(engine.count("named").expect("count"), 3);
    let fetched = engine
        .get("named", &PointId::from("p3"))
        .expect("get")
        .expect("present");
    assert_eq!(fetched.vectors.len(), 2);
}

// Store-level validation ------------------------------------------------

#[test]
fn named_store_upsert_record_requires_named_payload() {
    let point = VectorPoint::new("p1", vec![1.0]);
    assert!(super::named_upsert_record(&point).is_err());
}

#[test]
fn named_store_full_replacement_drops_postings() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store =
        NamedStore::create(dir.path(), "c", &spaces(), None, &Default::default()).expect("create");
    store
        .upsert_batch(&[point("p1", vec![1.0, 0.0, 0.0, 0.0], vec![(5, 2.0)])])
        .expect("upsert");
    store
        .upsert_batch(&[point("p1", vec![0.0, 1.0, 0.0, 0.0], vec![(6, 1.0)])])
        .expect("upsert");
    let results = store
        .search(&named_query(
            "keywords",
            VectorData::sparse(vec![5], vec![1.0]),
        ))
        .expect("search");
    assert!(results.is_empty());
    let results = store
        .search(&named_query(
            "keywords",
            VectorData::sparse(vec![6], vec![1.0]),
        ))
        .expect("search");
    assert_eq!(results.len(), 1);
}

#[test]
fn named_collection_create_validates_layout_before_opening_storage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LocalVectorEngine::open(dir.path()).expect("open");

    let empty = CollectionConfig::with_named(BTreeMap::new());
    assert!(engine.create_collection("empty", &empty).is_err());

    let mut multi = named_config();
    multi.vectors.insert(
        "multi".to_string(),
        crate::types::VectorSpaceConfig::multi(
            4,
            DistanceMetric::Cosine,
            crate::types::MultiConfig::new(2),
        ),
    );
    assert!(engine.create_collection("multi", &multi).is_ok());

    let mut ann = named_config();
    ann.index = crate::types::IndexConfig::Hnsw(Default::default());
    assert!(engine.create_collection("ann", &ann).is_err());

    let mut quantized = named_config();
    quantized.quantization_config = Some(crate::types::QuantizationConfig::binary());
    assert!(engine.create_collection("quantized", &quantized).is_err());
}

#[test]
fn named_collection_rejects_dense_recall_modes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LocalVectorEngine::open(dir.path()).expect("open");
    engine
        .create_collection("named", &named_config())
        .expect("create");
    let query = named_query("text", VectorData::dense(vec![1.0, 0.0, 0.0, 0.0]))
        .with_search_mode(crate::types::SearchMode::TopK(5));
    assert!(engine.search("named", &query).is_err());
}

#[test]
fn named_coordinated_upsert_is_durable() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = LocalVectorEngine::open(dir.path()).unwrap();
        engine.create_collection("named", &named_config()).unwrap();
        engine
            .apply_txn(vec![crate::TxnOp::Upsert {
                collection: "named".to_string(),
                point: point("p1", vec![1.0, 0.0, 0.0, 0.0], vec![(1, 1.0)]),
            }])
            .expect("apply txn");
    }
    let engine = LocalVectorEngine::open(dir.path()).unwrap();
    assert_eq!(engine.count("named").unwrap(), 1);
    assert!(engine.get("named", &PointId::from("p1")).unwrap().is_some());
}

#[test]
fn named_patch_preserves_other_spaces() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path()).unwrap();
    engine.create_collection("named", &named_config()).unwrap();
    engine.upsert(
        "named",
        point("p1", vec![1.0, 0.0, 0.0, 0.0], vec![(1, 1.0)]),
    ).unwrap();
    engine
        .update_named_vectors(
            "named",
            &PointId::from("p1"),
            BTreeMap::from([(
                "keywords".to_string(),
                VectorData::sparse(vec![2], vec![1.0]),
            )]),
        )
        .expect("patch");
    let point = engine.get("named", &PointId::from("p1")).unwrap().unwrap();
    assert_eq!(point.vectors.len(), 2);
    assert!(point.space("text").is_some());
    let hits = engine
        .search(
            "named",
            &named_query("keywords", VectorData::sparse(vec![1], vec![1.0])),
        )
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn named_multi_space_and_multi_target_fusion() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path()).unwrap();
    let mut config = named_config();
    config.vectors.insert(
        "late".to_string(),
        VectorSpaceConfig::multi(2, DistanceMetric::Dot, crate::types::MultiConfig::new(2)),
    );
    engine.create_collection("named", &config).unwrap();
    let point = VectorPoint::with_named("p1", "text", VectorData::dense(vec![1.0, 0.0, 0.0, 0.0]))
        .with_named_vector("keywords", VectorData::sparse(vec![1], vec![1.0]))
        .with_named_vector("late", VectorData::multi(vec![vec![1.0, 0.0], vec![0.0, 1.0]]));
    engine.upsert("named", point).unwrap();

    let hits = engine
        .search(
            "named",
            &named_query("late", VectorData::multi(vec![vec![1.0, 0.0]])),
        )
        .unwrap();
    assert_eq!(hits.len(), 1);

    let fused = engine
        .search(
            "named",
            &named_query("text", VectorData::dense(vec![1.0, 0.0, 0.0, 0.0]))
                .with_named_vector("keywords", VectorData::sparse(vec![1], vec![1.0]))
                .with_vector(true),
        )
        .unwrap();
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].vectors.len(), 3);
}

#[test]
fn fusion_configs_are_configurable_and_effective() {
    use crate::types::{FusionConfig, SearchResult};

    let a = || vec![
        SearchResult::new("p1", 1.0).with_named_vector("s1", VectorData::dense(vec![1.0])),
        SearchResult::new("p2", 0.5).with_named_vector("s1", VectorData::dense(vec![0.0])),
    ];
    let b = || vec![
        SearchResult::new("p2", 1.0).with_named_vector("s2", VectorData::dense(vec![1.0])),
        SearchResult::new("p1", 0.5).with_named_vector("s2", VectorData::dense(vec![0.0])),
    ];

    // Equal weights: both appear once at the top; union carries both spaces.
    let even = crate::types::query::fuse_rankings(
        vec![a(), b()],
        &FusionConfig::Rrf { k: None, weights: vec![] },
        0,
        10,
    );
    assert_eq!(even.len(), 2);
    assert!(even.iter().all(|r| r.vectors.len() == 2));

    // Weighting a disjoint second list heavily promotes its sole result.
    let only_p2 = || vec![
        SearchResult::new("p2", 1.0).with_named_vector("s2", VectorData::dense(vec![1.0])),
    ];
    let weighted = crate::types::query::fuse_rankings(
        vec![a(), only_p2()],
        &FusionConfig::Rrf { k: None, weights: vec![1.0, 100.0] },
        0,
        10,
    );
    assert_eq!(weighted[0].id, PointId::from("p2"));

    // DBSF is a usable alternative fusion that still returns both points.
    let dbsf = crate::types::query::fuse_rankings(
        vec![a(), b()],
        &FusionConfig::Dbsf { weights: vec![] },
        0,
        10,
    );
    assert_eq!(dbsf.len(), 2);
}

#[test]
fn with_vector_spaces_filters_echoed_spaces() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path()).unwrap();
    engine.create_collection("named", &named_config()).unwrap();
    engine
        .upsert(
            "named",
            point("p1", vec![1.0, 0.0, 0.0, 0.0], vec![(1, 1.0)]),
        )
        .unwrap();
    let query = named_query("text", VectorData::dense(vec![1.0, 0.0, 0.0, 0.0]))
        .with_vector(true)
        .with_vector_spaces(vec!["keywords".to_string()]);
    let hits = engine.search("named", &query).unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].vectors.contains_key("keywords"));
    assert!(!hits[0].vectors.contains_key("text"));
}

#[test]
fn invalid_patch_never_reaches_wal_and_reopen_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = LocalVectorEngine::open(dir.path()).unwrap();
        engine.create_collection("named", &named_config()).unwrap();
        engine
            .upsert(
                "named",
                point("p1", vec![1.0, 0.0, 0.0, 0.0], vec![(1, 1.0)]),
            )
            .unwrap();
        // Patching an unknown space is rejected before the WAL append.
        let bad = BTreeMap::from([(
            "nope".to_string(),
            VectorData::dense(vec![1.0, 0.0, 0.0, 0.0]),
        )]);
        assert!(engine
            .update_named_vectors("named", &PointId::from("p1"), bad)
            .is_err());
    }
    let reopened = LocalVectorEngine::open(dir.path()).unwrap();
    assert_eq!(reopened.count("named").unwrap(), 1);
}

// Storage layout -----------------------------------------------------------

#[test]
fn named_restart_loads_the_posting_dictionaries() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("named");
    let storage = crate::types::StorageConfig::default();
    let (blob_bytes, posting_bytes);
    {
        let store = NamedStore::create(&dir, "named", &spaces(), None, &storage).unwrap();
        let points: Vec<VectorPoint> = (0..20u32)
            .map(|i| point(&format!("p{i}"), vec![1.0, 0.0, 0.0, 0.0], (0..32).map(|t| (t, 1.0)).collect()))
            .collect();
        store.upsert_batch(&points).unwrap();
        store.checkpoint().unwrap();
        (blob_bytes, posting_bytes) = store.storage_bytes();
        assert!(blob_bytes > 0);
        assert!(posting_bytes > 0, "sparse spaces must persist their postings");
        let snapshot = std::fs::metadata(data_dir(&dir).join("named_snapshot.bin"))
            .unwrap()
            .len();
        assert!(
            snapshot * 2 < blob_bytes,
            "snapshot ({snapshot} B) must stay below the vector bytes ({blob_bytes} B)"
        );
    }

    let reopened = NamedStore::open(&dir, &storage).unwrap();
    assert!(reopened.postings_restored());
    assert_eq!(reopened.count(), 20);
    assert_eq!(reopened.storage_bytes(), (blob_bytes, posting_bytes));
    // Every point carries term 5, so the page is full (the query asks for 10).
    let query = named_query("keywords", VectorData::sparse(vec![5], vec![1.0]));
    assert_eq!(reopened.search(&query).unwrap().len(), 10);
    let text = named_query("text", VectorData::dense(vec![1.0, 0.0, 0.0, 0.0]));
    assert_eq!(reopened.search(&text).unwrap().len(), 10);
}

#[test]
fn named_patch_reallocates_the_point_and_its_postings() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("named");
    let storage = crate::types::StorageConfig::default();
    let store = NamedStore::create(&dir, "named", &spaces(), None, &storage).unwrap();
    store
        .upsert_batch(&[point("p1", vec![1.0, 0.0, 0.0, 0.0], vec![(1, 1.0), (2, 1.0)])])
        .unwrap();
    store.checkpoint().unwrap();
    store
        .patch_named(
            &PointId::from("p1"),
            BTreeMap::from([(
                "keywords".to_string(),
                VectorData::sparse(vec![3], vec![2.0]),
            )]),
        )
        .unwrap();

    // The replaced space's old terms must be gone; a stale posting on a live
    // slot would surface as a phantom hit.
    assert!(store
        .search(&named_query("keywords", VectorData::sparse(vec![1], vec![1.0])))
        .unwrap()
        .is_empty());
    assert_eq!(
        store
            .search(&named_query("keywords", VectorData::sparse(vec![3], vec![1.0])))
            .unwrap()
            .len(),
        1
    );
    // The untouched dense space survives the rewrite.
    let point = store.get(&PointId::from("p1")).unwrap().unwrap();
    assert_eq!(point.vectors.len(), 2);

    let reopened = NamedStore::open(&dir, &storage).unwrap();
    assert_eq!(reopened.count(), 1);
    assert!(reopened
        .search(&named_query("keywords", VectorData::sparse(vec![1], vec![1.0])))
        .unwrap()
        .is_empty());
    assert_eq!(
        reopened
            .search(&named_query("keywords", VectorData::sparse(vec![3], vec![1.0])))
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn named_compaction_reclaims_blob_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("named");
    let storage = crate::types::StorageConfig::default();
    let store = NamedStore::create(&dir, "named", &spaces(), None, &storage).unwrap();
    let points: Vec<VectorPoint> = (0..10u32)
        .map(|i| point(&format!("p{i}"), vec![1.0, 0.0, 0.0, 0.0], vec![(1, 1.0)]))
        .collect();
    store.upsert_batch(&points).unwrap();
    store.checkpoint().unwrap();
    let before = store.storage_bytes().0;
    for i in 0..5u32 {
        store.delete(&PointId::from(format!("p{i}"))).unwrap();
    }
    assert_eq!(store.compact().unwrap(), 5);
    assert!(store.storage_bytes().0 <= before / 2);

    let reopened = NamedStore::open(&dir, &storage).unwrap();
    assert_eq!(reopened.count(), 5);
    assert!(reopened.postings_restored());
    assert_eq!(
        reopened
            .search(&named_query("keywords", VectorData::sparse(vec![1], vec![1.0])))
            .unwrap()
            .len(),
        5
    );
}

#[test]
fn named_payloads_are_blobbed_outside_the_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("named");
    let storage = crate::types::StorageConfig::default();
    let marker = "named-payload-outside-snapshot";
    let big = marker.repeat(48);
    let id = PointId::from("p1");
    {
        let store = NamedStore::create(&dir, "named", &spaces(), None, &storage).unwrap();
        let point = point("p1", vec![1.0, 0.0, 0.0, 0.0], vec![(1, 1.0)]).with_payload(
            [("note".to_string(), serde_json::json!(big))]
                .into_iter()
                .collect(),
        );
        store.upsert_batch(&[point]).unwrap();
        store.checkpoint().unwrap();

        assert!(store.payload_bytes() >= (marker.len() * 48) as u64);
        let snapshot = std::fs::read(data_dir(&dir).join("named_snapshot.bin")).unwrap();
        assert!(
            !snapshot
                .windows(marker.len())
                .any(|w| w == marker.as_bytes()),
            "checkpointed payload bytes must not be inlined in the snapshot"
        );
        let bytes = store.payload_bytes();
        store.checkpoint().unwrap();
        assert_eq!(store.payload_bytes(), bytes);
    }
    let reopened = NamedStore::open(&dir, &storage).unwrap();
    let point = reopened.get(&id).unwrap().unwrap();
    assert_eq!(
        point.payload.as_ref().unwrap()["note"],
        serde_json::json!(big)
    );
    assert_eq!(point.vectors.len(), 2);
}
