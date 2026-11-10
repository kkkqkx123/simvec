use super::*;
    use crate::types::{DistanceMetric, FilterCondition, Payload, VectorFilter};

    pub(super) fn config(dim: usize) -> CollectionConfig {
        CollectionConfig::new(dim, DistanceMetric::Cosine)
    }

    pub(super) fn point(id: u64, dim: usize) -> VectorPoint {
        VectorPoint::new(
            id,
            (0..dim)
                .map(|i| ((id as usize * 31 + i) % 100) as f32 / 100.0)
                .collect(),
        )
    }

    pub(super) fn point_with_color(id: u64, dim: usize, color: &str) -> VectorPoint {
        let mut payload: Payload = HashMap::new();
        payload.insert("color".to_string(), serde_json::json!(color));
        VectorPoint::new(id, (0..dim).map(|_| 0.5).collect()).with_payload(payload)
    }

    pub(super) fn engine() -> LocalVectorEngine {
        let dir = tempfile::tempdir().unwrap();
        LocalVectorEngine::open(dir.path().join("vec")).unwrap()
    }

    #[test]
    pub(super) fn test_deletes_schedule_background_compaction() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        let engine = LocalVectorEngine::open(&root).unwrap();
        engine.create_collection("col", &config(4)).unwrap();
        for i in 0..10u64 {
            engine.upsert("col", point(i, 4)).unwrap();
        }
        // 3/10 live-slot ratio crosses the 20% threshold: the mutation path
        // must enqueue a background compaction without blocking.
        engine.delete("col", &PointId::from("0")).unwrap();
        engine.delete("col", &PointId::from("1")).unwrap();
        engine.delete("col", &PointId::from("2")).unwrap();

        let store = engine.store("col").unwrap();
        // Deletes are visible the moment they return, so the caller never
        // waits for the compaction they trigger. Whether the threshold still
        // reads as "needs compaction" here is deliberately not asserted: the
        // maintenance worker races this point and may already have drained
        // it. The threshold itself is covered on the store path, which no
        // background worker touches.
        assert_eq!(store.count(), 7);
        for id in 0..3u64 {
            assert!(engine.get("col", &PointId::from(id)).unwrap().is_none());
        }

        // The maintenance worker drains the queue promptly; poll briefly so
        // the test does not depend on exact thread scheduling.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while store.meta().tombstone_count > 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(store.meta().tombstone_count, 0, "background compaction ran");
        assert_eq!(store.count(), 7);
    }

    #[test]
    pub(super) fn test_create_open_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        {
            let engine = LocalVectorEngine::open(&root).unwrap();
            engine.create_collection("col_a", &config(4)).unwrap();
            engine.upsert("col_a", point(1, 4)).unwrap();
            engine.upsert("col_a", point(2, 4)).unwrap();
            assert_eq!(engine.count("col_a").unwrap(), 2);
        }
        let reopened = LocalVectorEngine::open(&root).unwrap();
        assert!(reopened.collection_exists("col_a"));
        assert_eq!(reopened.count("col_a").unwrap(), 2);
        let got = reopened.get("col_a", &PointId::from("1")).unwrap().unwrap();
        assert_eq!(got.vector, point(1, 4).vector);
        assert!(reopened
            .get("col_a", &PointId::from("99"))
            .unwrap()
            .is_none());
    }

    #[test]
    pub(super) fn test_create_duplicate_fails() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        let err = engine.create_collection("col", &config(4)).unwrap_err();
        assert!(matches!(err, VectorSearchError::CollectionAlreadyExists(_)));
    }

    #[test]
    pub(super) fn test_invalid_collection_name() {
        let engine = engine();
        let err = engine.create_collection("a/b", &config(4)).unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidCollectionName(_)));
        let err = engine.create_collection("", &config(4)).unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidCollectionName(_)));
    }

    #[test]
    pub(super) fn test_delete_collection() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        {
            let engine = LocalVectorEngine::open(&root).unwrap();
            engine.create_collection("col", &config(4)).unwrap();
            engine.upsert("col", point(1, 4)).unwrap();
            engine.delete_collection("col").unwrap();
            assert!(!engine.collection_exists("col"));
        }
        let reopened = LocalVectorEngine::open(&root).unwrap();
        assert!(!reopened.collection_exists("col"));
    }

    #[test]
    pub(super) fn test_delete_collection_missing() {
        let engine = engine();
        let err = engine.delete_collection("nope").unwrap_err();
        assert!(matches!(err, VectorSearchError::CollectionNotFound(_)));
    }

    #[test]
    pub(super) fn test_promote_temp_collection_swaps_live_and_keeps_backup() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        let engine = LocalVectorEngine::open(&root).unwrap();
        engine.create_collection("live", &config(4)).unwrap();
        engine.upsert("live", point(1, 4)).unwrap();
        engine
            .create_collection("live.rebuild-2", &config(4))
            .unwrap();
        engine.upsert("live.rebuild-2", point(2, 4)).unwrap();

        engine
            .promote_temp_collection("live", "live.rebuild-2", ".old-2")
            .unwrap();

        assert_eq!(engine.count("live").unwrap(), 1);
        let got = engine.get("live", &PointId::from("2")).unwrap().unwrap();
        assert_eq!(got.vector, point(2, 4).vector);
        assert!(engine.get("live", &PointId::from("1")).unwrap().is_none());
        assert_eq!(engine.count("live.old-2").unwrap(), 1);
        // Scratch and backups serve no reads through discovery names.
        assert!(!engine
            .collection_names()
            .contains(&"live.old-2".to_string()));
        assert!(engine
            .list_collections()
            .contains(&"live.old-2".to_string()));
        assert!(!engine
            .list_temp_collections()
            .contains(&"live.rebuild-2".to_string()));

        // Second publish prunes the older backup, keeping exactly one.
        engine
            .create_collection("live.rebuild-3", &config(4))
            .unwrap();
        engine.upsert("live.rebuild-3", point(3, 4)).unwrap();
        engine
            .promote_temp_collection("live", "live.rebuild-3", ".old-3")
            .unwrap();
        assert_eq!(engine.count("live").unwrap(), 1);
        assert!(engine
            .list_collections()
            .contains(&"live.old-3".to_string()));
        assert!(!engine
            .list_collections()
            .contains(&"live.old-2".to_string()));

        // Restarted engines never adopt scratch or backups as serving data.
        drop(engine);
        let reopened = LocalVectorEngine::open(&root).unwrap();
        assert_eq!(reopened.count("live").unwrap(), 1);
        assert!(!reopened
            .collection_names()
            .contains(&"live.old-3".to_string()));
        assert!(reopened
            .list_collections()
            .contains(&"live.old-3".to_string()));
    }

    #[test]
    pub(super) fn test_promote_temp_collection_missing_temp_fails_without_touching_live() {
        let engine = engine();
        engine.create_collection("live", &config(4)).unwrap();
        engine.upsert("live", point(1, 4)).unwrap();
        let err = engine
            .promote_temp_collection("live", "live.rebuild-9", ".old-9")
            .unwrap_err();
        assert!(matches!(err, VectorSearchError::CollectionNotFound(_)));
        assert_eq!(engine.count("live").unwrap(), 1);
    }

    #[test]
    pub(super) fn test_restore_promote_backup_reverses_bad_publish() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        let engine = LocalVectorEngine::open(&root).unwrap();
        engine.create_collection("live", &config(4)).unwrap();
        engine.upsert("live", point(1, 4)).unwrap();
        engine
            .create_collection("live.rebuild-2", &config(4))
            .unwrap();
        engine.upsert("live.rebuild-2", point(2, 4)).unwrap();
        engine
            .promote_temp_collection("live", "live.rebuild-2", ".old-2")
            .unwrap();

        let restored = engine.restore_promote_backup("live").unwrap();
        assert_eq!(restored, "live.old-2");
        assert_eq!(engine.count("live").unwrap(), 1);
        let got = engine.get("live", &PointId::from("1")).unwrap().unwrap();
        assert_eq!(got.vector, point(1, 4).vector);

        let err = engine.restore_promote_backup("live").unwrap_err();
        assert!(matches!(err, VectorSearchError::CollectionNotFound(_)));
    }

    #[test]
    pub(super) fn test_open_skips_scratch_and_backup_directories() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        {
            let engine = LocalVectorEngine::open(&root).unwrap();
            engine.create_collection("live", &config(4)).unwrap();
            engine.upsert("live", point(1, 4)).unwrap();
            engine
                .create_collection("live.rebuild-2", &config(4))
                .unwrap();
            engine
                .promote_temp_collection("live", "live.rebuild-2", ".old-2")
                .unwrap();
        }
        let reopened = LocalVectorEngine::open(&root).unwrap();
        assert!(reopened.collection_exists("live"));
        assert!(!reopened
            .collection_names()
            .contains(&"live.old-2".to_string()));
        // Scratch residue is deletable without being opened first.
        reopened.delete_collection("live.old-2").unwrap();
        assert!(!reopened
            .list_collections()
            .contains(&"live.old-2".to_string()));
    }

    #[test]
    pub(super) fn test_collection_config_and_info() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        let cfg = engine.collection_config("col").unwrap().unwrap();
        assert_eq!(cfg.vector_size, 4);
        assert_eq!(cfg.distance, DistanceMetric::Cosine);
        assert!(engine.collection_config("nope").unwrap().is_none());

        engine.upsert("col", point(1, 4)).unwrap();
        let info = engine.collection_info("col").unwrap();
        assert_eq!(info.name, "col");
        assert_eq!(info.points_count, 1);
        assert_eq!(info.vector_count, 1);
        assert_eq!(info.segments_count, 1);
        assert_eq!(info.status, CollectionStatus::Green);
    }

    #[test]
    pub(super) fn test_search_returns_topk() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        engine.upsert("col", point(1, 4)).unwrap();
        engine.upsert("col", point(2, 4)).unwrap();
        engine.upsert("col", point(3, 4)).unwrap();

        let results = engine
            .search("col", &SearchQuery::new(vec![0.0; 4], 2).with_payload(true))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            results[0].score >= results[1].score,
            "results sorted descending: {:?}",
            results
        );
    }

    #[test]
    pub(super) fn test_search_dimension_mismatch() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        engine.upsert("col", point(1, 4)).unwrap();
        let err = engine
            .search("col", &SearchQuery::new(vec![1.0, 2.0], 1))
            .unwrap_err();
        assert!(matches!(
            err,
            VectorSearchError::InvalidVectorDimension {
                expected: 4,
                actual: 2
            }
        ));
    }

    #[test]
    pub(super) fn test_delete_by_id() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        engine.upsert("col", point(1, 4)).unwrap();
        engine.upsert("col", point(2, 4)).unwrap();
        engine.delete("col", &PointId::from("1")).unwrap();
        assert_eq!(engine.count("col").unwrap(), 1);
        assert!(engine.get("col", &PointId::from("1")).unwrap().is_none());
        engine.delete("col", &PointId::from("99")).unwrap();
        assert_eq!(engine.count("col").unwrap(), 1);
    }

    #[test]
    pub(super) fn test_delete_by_filter() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        engine.upsert("col", point_with_color(1, 4, "red")).unwrap();
        engine
            .upsert("col", point_with_color(2, 4, "blue"))
            .unwrap();
        engine.upsert("col", point_with_color(3, 4, "red")).unwrap();
        assert_eq!(engine.count("col").unwrap(), 3);

        let filter = VectorFilter::new().must(FilterCondition::match_value("color", "red"));
        let deleted = engine.delete_by_filter("col", &filter).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(engine.count("col").unwrap(), 1);
        assert!(engine.get("col", &PointId::from("2")).unwrap().is_some());
        assert!(engine.get("col", &PointId::from("1")).unwrap().is_none());
        assert!(engine.get("col", &PointId::from("3")).unwrap().is_none());
    }

    #[test]
    pub(super) fn test_delete_by_filter_no_match() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        engine.upsert("col", point(1, 4)).unwrap();
        let filter = VectorFilter::new().must(FilterCondition::match_value("color", "red"));
        assert_eq!(engine.delete_by_filter("col", &filter).unwrap(), 0);
        assert_eq!(engine.count("col").unwrap(), 1);
    }

    #[test]
    pub(super) fn test_apply_txn_multi_collection() {
        let engine = engine();
        engine.create_collection("col_b", &config(4)).unwrap();
        engine.create_collection("col_a", &config(4)).unwrap();

        let ops = vec![
            TxnOp::Upsert {
                collection: "col_b".to_string(),
                point: point(1, 4),
            },
            TxnOp::Upsert {
                collection: "col_a".to_string(),
                point: point(2, 4),
            },
            TxnOp::Delete {
                collection: "col_a".to_string(),
                point_id: PointId::from("42"),
            },
        ];
        engine.apply_txn(ops).unwrap();

        assert_eq!(engine.count("col_a").unwrap(), 1);
        assert_eq!(engine.count("col_b").unwrap(), 1);
        assert!(engine.get("col_b", &PointId::from("1")).unwrap().is_some());

        // Replaying the same txn id is idempotent.
        engine
            .apply_txn(vec![TxnOp::Upsert {
                collection: "col_a".to_string(),
                point: point(2, 4),
            }])
            .unwrap();
        assert_eq!(engine.count("col_a").unwrap(), 1);
    }

    #[test]
    pub(super) fn test_apply_txn_unknown_collection() {
        let engine = engine();
        let err = engine
            .apply_txn(vec![TxnOp::Upsert {
                collection: "nope".to_string(),
                point: point(1, 4),
            }])
            .unwrap_err();
        assert!(matches!(err, VectorSearchError::CollectionNotFound(_)));
    }

    #[test]
    pub(super) fn test_apply_txn_validates_dimension() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        let err = engine
            .apply_txn(vec![TxnOp::Upsert {
                collection: "col".to_string(),
                point: VectorPoint::new(1u64, vec![1.0, 2.0]),
            }])
            .unwrap_err();
        assert!(matches!(
            err,
            VectorSearchError::InvalidVectorDimension {
                expected: 4,
                actual: 2
            }
        ));
        assert_eq!(engine.count("col").unwrap(), 0);
    }

    #[test]
    pub(super) fn test_wal_recovery_after_mutation() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        {
            let engine = LocalVectorEngine::open(&root).unwrap();
            engine.create_collection("col", &config(4)).unwrap();
            engine
                .apply_txn(vec![TxnOp::Upsert {
                    collection: "col".to_string(),
                    point: point(1, 4),
                }])
                .unwrap();
            engine.upsert("col", point(2, 4)).unwrap();
            engine.delete("col", &PointId::from("2")).unwrap();
        }
        let reopened = LocalVectorEngine::open(&root).unwrap();
        assert_eq!(reopened.count("col").unwrap(), 1);
        assert!(reopened.get("col", &PointId::from("1")).unwrap().is_some());
        assert!(reopened.get("col", &PointId::from("2")).unwrap().is_none());
    }
