//! Tests for `NamedStore` and the engine's named-collection dispatch.

use std::collections::BTreeMap;

use super::NamedStore;
use crate::engine::LocalVectorEngine;
use crate::types::{
    CollectionConfig, DistanceMetric, PointId, SearchQuery, VectorData, VectorPoint,
    VectorSpaceConfig, VectorsConfig,
};

fn spaces() -> BTreeMap<String, VectorSpaceConfig> {
    BTreeMap::from([
        (
            "text".to_string(),
            VectorSpaceConfig {
                kind: crate::types::VectorKind::Dense,
                vector_size: 4,
                distance: DistanceMetric::Cosine,
            },
        ),
        (
            "keywords".to_string(),
            VectorSpaceConfig {
                kind: crate::types::VectorKind::Sparse,
                vector_size: 16,
                distance: DistanceMetric::Dot,
            },
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
    let mut query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0], 10);
    query.target = Some("text".to_string());
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
    assert!(fetched.named.contains_key("text"));
    assert!(fetched.named.contains_key("keywords"));

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
    assert!(page[0].named.contains_key("keywords"));
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
    assert_eq!(fetched.named.len(), 2);
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
        NamedStore::create(dir.path(), "c", &spaces(), &Default::default()).expect("create");
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
    let _ = VectorsConfig::Single;
}

#[test]
fn named_collection_create_validates_layout_before_opening_storage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LocalVectorEngine::open(dir.path()).expect("open");

    let empty = CollectionConfig::with_named(BTreeMap::new());
    assert!(engine.create_collection("empty", &empty).is_err());

    let mut multi = named_config();
    if let crate::types::VectorsConfig::Named(spaces) = &mut multi.vectors {
        spaces.insert(
            "multi".to_string(),
            crate::types::VectorSpaceConfig {
                kind: crate::types::VectorKind::Multi,
                vector_size: 4,
                distance: DistanceMetric::Cosine,
            },
        );
    }
    assert!(engine.create_collection("multi", &multi).is_err());

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
