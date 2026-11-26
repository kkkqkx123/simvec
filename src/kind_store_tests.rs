//! Kind store tests: kind validation, recall ordering and restart durability.

use std::sync::Arc;

use super::*;
use crate::error::VectorSearchError;
use crate::types::{
    CollectionConfig, DistanceMetric, IndexConfig, MultiConfig, PointId, SearchQuery, SparseConfig,
    VectorKind, VectorPoint,
};
use crate::{LocalVectorEngine, VectorEngine};

fn sparse_config() -> CollectionConfig {
    CollectionConfig::with_sparse(100, DistanceMetric::Dot, SparseConfig { vocab_size: 100 })
}

fn multi_config() -> CollectionConfig {
    CollectionConfig::with_multi(
        2,
        DistanceMetric::Dot,
        MultiConfig {
            max_vectors_per_point: 4,
        },
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
    config.sparse_config = None;
    assert!(config.validate_kind().is_err());

    let mut config = multi_config();
    config.index = IndexConfig::Hnsw(crate::types::HnswConfig::default());
    assert!(config.validate_kind().is_err());

    assert!(sparse_config().validate_kind().is_ok());
    assert!(multi_config().validate_kind().is_ok());
}

#[test]
fn dense_config_rejects_kind_params() {
    let mut config = CollectionConfig::new(3, DistanceMetric::Cosine);
    config.sparse_config = Some(SparseConfig { vocab_size: 10 });
    assert!(config.validate_kind().is_err());
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
    assert!(point.data.is_some());

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
