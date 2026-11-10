use super::*;
    use crate::types::{CollectionConfig, DistanceMetric, SearchQuery, VectorPoint};

    fn engine_with_points() -> MemoryEngine {
        let engine = MemoryEngine::new();
        let config = CollectionConfig::new(2, DistanceMetric::Dot);
        engine.create_collection("docs", &config).unwrap();
        engine
            .upsert("docs", VectorPoint::new("a", vec![1.0, 0.0]))
            .unwrap();
        engine
            .upsert("docs", VectorPoint::new("b", vec![0.0, 1.0]))
            .unwrap();
        engine
    }

    #[test]
    fn memory_search_scores_higher_is_better() {
        let engine = engine_with_points();
        let results = engine
            .search("docs", &SearchQuery::new(vec![1.0, 0.0], 10))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id.to_string(), "a");
    }

    #[test]
    fn memory_rejects_dimension_mismatch() {
        let engine = engine_with_points();
        let err = engine
            .upsert("docs", VectorPoint::new("c", vec![1.0]))
            .unwrap_err();
        assert!(matches!(
            err,
            VectorSearchError::InvalidVectorDimension { .. }
        ));
    }

    #[test]
    fn memory_apply_txn_groups_by_collection() {
        let engine = MemoryEngine::new();
        let config = CollectionConfig::new(2, DistanceMetric::Dot);
        engine.create_collection("a", &config).unwrap();
        engine.create_collection("b", &config).unwrap();
        engine
            .apply_txn(vec![
                MemoryTxnOp::Upsert {
                    collection: "b".to_string(),
                    point: VectorPoint::new("p1", vec![1.0, 0.0]),
                },
                MemoryTxnOp::Upsert {
                    collection: "a".to_string(),
                    point: VectorPoint::new("p2", vec![0.0, 1.0]),
                },
            ])
            .unwrap();
        assert_eq!(engine.count("a").unwrap(), 1);
        assert_eq!(engine.count("b").unwrap(), 1);
    }

    #[test]
    fn memory_snapshot_roundtrip() {
        let engine = engine_with_points();
        let snapshot = engine.export_snapshot().unwrap();
        let restored = MemoryEngine::new();
        restored.import_snapshot(snapshot).unwrap();
        assert_eq!(restored.count("docs").unwrap(), 2);
    }

    #[test]
    fn memory_snapshot_rejects_bad_version() {
        let engine = MemoryEngine::new();
        let err = engine
            .import_snapshot(MemorySnapshot {
                version: 999,
                collections: Vec::new(),
            })
            .unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidConfig(_)));
    }

    #[test]
    fn memory_revision_advances_on_change_only() {
        let engine = MemoryEngine::new();
        assert_eq!(engine.revision(), 0);
        let config = CollectionConfig::new(2, DistanceMetric::Dot);
        engine.create_collection("docs", &config).unwrap();
        let after_create = engine.revision();
        assert!(after_create > 0);
        engine
            .search("docs", &SearchQuery::new(vec![1.0, 0.0], 10))
            .unwrap();
        assert_eq!(engine.revision(), after_create);
        engine.delete("docs", &PointId::from("missing")).unwrap();
        assert_eq!(engine.revision(), after_create);
        engine
            .upsert("docs", VectorPoint::new("a", vec![1.0, 0.0]))
            .unwrap();
        assert!(engine.revision() > after_create);
    }

    #[test]
    fn memory_collection_snapshot_roundtrip() {
        let engine = engine_with_points();
        let entry = engine.export_collection("docs").unwrap();
        assert_eq!(entry.name, "docs");
        assert_eq!(entry.points.len(), 2);
        let restored = MemoryEngine::new();
        restored.import_collection(entry).unwrap();
        assert_eq!(restored.count("docs").unwrap(), 2);
        assert!(restored.export_snapshot().unwrap().version == 1);
    }

    #[test]
    fn memory_export_orders_points_by_id() {
        let engine = MemoryEngine::new();
        let config = CollectionConfig::new(2, DistanceMetric::Dot);
        engine.create_collection("docs", &config).unwrap();
        engine
            .upsert("docs", VectorPoint::new("b", vec![0.0, 1.0]))
            .unwrap();
        engine
            .upsert("docs", VectorPoint::new("a", vec![1.0, 0.0]))
            .unwrap();
        let snapshot = engine.export_snapshot().unwrap();
        let ids: Vec<String> = snapshot.collections[0]
            .points
            .iter()
            .map(|p| p.id.to_string())
            .collect();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
    }
