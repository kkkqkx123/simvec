use std::collections::HashMap;

use crate::error::{Result, VectorSearchError};

use super::*;

    #[test]
    fn test_point_id_from_u64() {
        let id: PointId = 42u64.into();
        assert_eq!(id, PointId::Num(42));
    }

    #[test]
    fn test_point_id_from_string_numeric() {
        let id = PointId::from("123");
        assert_eq!(id, PointId::Num(123));
    }

    #[test]
    fn test_point_id_from_string_uuid() {
        let id = PointId::from("uuid-abc");
        assert_eq!(id, PointId::Uuid("uuid-abc".into()));
    }

    #[test]
    fn test_point_id_from_str() {
        let id = PointId::from("456");
        assert_eq!(id, PointId::Num(456));
    }

    #[test]
    fn test_point_id_from_str_uuid() {
        let id = PointId::from("550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(
            id,
            PointId::Uuid("550e8400-e29b-41d4-a716-446655440000".into())
        );
    }

    #[test]
    fn test_point_id_display_num() {
        let id = PointId::Num(42);
        assert_eq!(format!("{}", id), "42");
    }

    #[test]
    fn test_point_id_display_uuid() {
        let id = PointId::Uuid("abc".into());
        assert_eq!(format!("{}", id), "abc");
    }

    #[test]
    fn test_point_id_serialize_deserialize() {
        let id = PointId::Num(42);
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: PointId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn test_point_id_serialize_deserialize_uuid() {
        let id = PointId::Uuid("test-uuid".into());
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: PointId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn test_hnsw_config_default() {
        let cfg = HnswConfig::default();
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.ef_construct, 100);
        assert!(cfg.full_scan_threshold.is_none());
    }

    #[test]
    fn test_hnsw_config_new() {
        let cfg = HnswConfig::new(32, 200);
        assert_eq!(cfg.m, 32);
        assert_eq!(cfg.ef_construct, 200);
    }

    #[test]
    fn test_hnsw_config_builder() {
        let cfg = HnswConfig::new(16, 100)
            .with_full_scan_threshold(10000)
            .with_max_indexing_threads(4)
            .with_on_disk(true)
            .with_payload_m(8);
        assert_eq!(cfg.full_scan_threshold, Some(10000));
        assert_eq!(cfg.max_indexing_threads, Some(4));
        assert_eq!(cfg.on_disk, Some(true));
        assert_eq!(cfg.payload_m, Some(8));
    }

    #[test]
    fn test_hnsw_config_validate() {
        assert!(HnswConfig::default().validate().is_ok());
        // Boundary: ef_construct == max(2m, 4) is accepted.
        assert!(HnswConfig::new(8, 16).validate().is_ok());
        assert!(HnswConfig::new(2, 4).validate().is_ok());
        assert!(HnswConfig::new(1, 4).validate().is_ok());

        let err = HnswConfig::new(16, 20).validate().unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidConfig(_)));
        assert!(HnswConfig::new(2, 3).validate().is_err());
    }

    #[test]
    fn test_quantization_config_scalar() {
        let cfg = QuantizationConfig::scalar(0.99);
        match cfg.quant_type {
            QuantizationType::Scalar { .. } => {}
            _ => panic!("expected Scalar"),
        }
    }

    #[test]
    fn test_quantization_config_product() {
        let cfg = QuantizationConfig::product(CompressionRatio::X8);
        match cfg.quant_type {
            QuantizationType::Product { compression, .. } => {
                assert!(matches!(compression, CompressionRatio::X8));
            }
            _ => panic!("expected Product"),
        }
    }

    #[test]
    fn test_quantization_config_binary() {
        let cfg = QuantizationConfig::binary();
        match cfg.quant_type {
            QuantizationType::Binary { .. } => {}
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn test_quantization_config_with_always_ram() {
        let cfg = QuantizationConfig::scalar(0.5).with_always_ram(false);
        match cfg.quant_type {
            QuantizationType::Scalar { always_ram, .. } => {
                assert_eq!(always_ram, Some(false));
            }
            _ => panic!("expected Scalar"),
        }
    }

    #[test]
    fn test_collection_config_new() {
        let cfg = CollectionConfig::new(768, DistanceMetric::Cosine);
        assert_eq!(cfg.vector_size, 768);
        assert_eq!(cfg.distance, DistanceMetric::Cosine);
    }

    #[test]
    fn test_collection_config_with_index_type() {
        let cfg =
            CollectionConfig::new(384, DistanceMetric::Euclid).with_index_type(IndexType::FLAT);
        assert_eq!(cfg.index, IndexConfig::Flat);
        assert_eq!(cfg.index_type(), IndexType::FLAT);
    }

    #[test]
    fn test_collection_config_with_hnsw() {
        let hnsw = HnswConfig::new(32, 200);
        let cfg = CollectionConfig::new(128, DistanceMetric::Dot).with_hnsw(hnsw.clone());
        assert_eq!(cfg.index, IndexConfig::Hnsw(hnsw));
        assert_eq!(cfg.index_type(), IndexType::HNSW);
    }

    #[test]
    fn test_collection_config_tier_switch_resets() {
        let cfg = CollectionConfig::new(128, DistanceMetric::Cosine)
            .with_hnsw(HnswConfig::new(32, 200))
            .with_index_type(IndexType::FLAT);
        assert_eq!(cfg.index, IndexConfig::Flat);
        let cfg = cfg.with_index_type(IndexType::IVF);
        assert_eq!(cfg.index, IndexConfig::Ivf(None));
        let ivf = IvfConfig::default();
        let cfg = cfg.with_ivf(ivf.clone());
        assert_eq!(cfg.index, IndexConfig::Ivf(Some(ivf)));
        let cfg = cfg.with_index_type(IndexType::HNSW);
        assert_eq!(cfg.index, IndexConfig::Hnsw(HnswConfig::default()));
    }

    #[test]
    fn test_collection_config_serde_roundtrip() {
        for cfg in [
            CollectionConfig::new(64, DistanceMetric::Cosine),
            CollectionConfig::new(64, DistanceMetric::Cosine).with_index_type(IndexType::FLAT),
            CollectionConfig::new(64, DistanceMetric::Cosine).with_index_type(IndexType::IVF),
            CollectionConfig::new(64, DistanceMetric::Cosine).with_ivf(IvfConfig::default()),
        ] {
            let json = serde_json::to_string(&cfg).expect("serialize");
            let back: CollectionConfig = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(cfg, back);
        }
    }

    #[test]
    fn test_collection_config_with_quantization() {
        let q = QuantizationConfig::scalar(0.99);
        let cfg = CollectionConfig::new(1536, DistanceMetric::Cosine).with_quantization(q);
        assert!(cfg.quantization_config.is_some());
    }

    #[test]
    fn test_collection_config_with_shard_number() {
        let cfg = CollectionConfig::new(768, DistanceMetric::Cosine).with_shard_number(2);
        assert_eq!(cfg.shard_number, Some(2));
    }

    #[test]
    fn test_collection_config_with_on_disk_payload() {
        let cfg = CollectionConfig::new(768, DistanceMetric::Cosine).with_on_disk_payload(true);
        assert_eq!(cfg.on_disk_payload, Some(true));
    }

    #[test]
    fn test_collection_config_default() {
        let cfg = CollectionConfig::default();
        assert_eq!(cfg.vector_size, 1536);
        assert_eq!(cfg.distance, DistanceMetric::Cosine);
    }

    #[test]
    fn test_payload_schema_type_as_str() {
        assert_eq!(PayloadSchemaType::Keyword.as_str(), "keyword");
        assert_eq!(PayloadSchemaType::Integer.as_str(), "integer");
        assert_eq!(PayloadSchemaType::Float.as_str(), "float");
        assert_eq!(PayloadSchemaType::Text.as_str(), "text");
        assert_eq!(PayloadSchemaType::Bool.as_str(), "bool");
        assert_eq!(PayloadSchemaType::Geo.as_str(), "geo");
        assert_eq!(PayloadSchemaType::Datetime.as_str(), "datetime");
    }

    #[test]
    fn test_health_status_healthy() {
        let h = HealthStatus::healthy("test-engine", "1.0");
        assert!(h.is_healthy);
        assert_eq!(h.engine_name, "test-engine");
        assert_eq!(h.engine_version, "1.0");
        assert!(h.message.is_none());
    }

    #[test]
    fn test_health_status_unhealthy() {
        let h = HealthStatus::unhealthy("test-engine", "1.0", "not ready");
        assert!(!h.is_healthy);
        assert_eq!(h.message, Some("not ready".to_string()));
    }

    #[test]
    fn test_index_type_default() {
        assert_eq!(IndexType::default(), IndexType::HNSW);
    }

    #[test]
    fn test_distance_metric_default() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
    }

    #[test]
    fn test_collection_status_debug() {
        assert_eq!(format!("{:?}", CollectionStatus::Green), "Green");
        assert_eq!(format!("{:?}", CollectionStatus::Yellow), "Yellow");
        assert_eq!(format!("{:?}", CollectionStatus::Red), "Red");
        assert_eq!(format!("{:?}", CollectionStatus::Grey), "Grey");
    }

    #[test]
    fn test_geo_point_new() {
        let p = GeoPoint::new(1.0, 2.0);
        assert!((p.lat - 1.0).abs() < f64::EPSILON);
        assert!((p.lon - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_geo_radius_new() {
        let center = GeoPoint::new(0.0, 0.0);
        let r = GeoRadius::new(center, 100.0);
        assert!((r.radius - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_geo_bounding_box_new() {
        let tl = GeoPoint::new(1.0, 2.0);
        let br = GeoPoint::new(3.0, 4.0);
        let bbox = GeoBoundingBox::new(tl, br);
        assert!((bbox.top_left.lat - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_values_count_condition_default() {
        let v = ValuesCountCondition::default();
        assert!(v.gt.is_none());
        assert!(v.gte.is_none());
        assert!(v.lt.is_none());
        assert!(v.lte.is_none());
    }

    #[test]
    fn test_values_count_condition_builder() {
        let v = ValuesCountCondition::new().gt(5).lt(10);
        assert_eq!(v.gt, Some(5));
        assert_eq!(v.lt, Some(10));
        assert!(v.gte.is_none());
        assert!(v.lte.is_none());
    }

    #[test]
    fn test_range_condition_default() {
        let r = RangeCondition::default();
        assert!(r.gt.is_none());
        assert!(r.gte.is_none());
        assert!(r.lt.is_none());
        assert!(r.lte.is_none());
    }

    #[test]
    fn test_range_condition_builder() {
        let r = RangeCondition::new().gte(1.5).lt(10.0);
        assert_eq!(r.gte, Some(1.5));
        assert_eq!(r.lt, Some(10.0));
        assert!(r.gt.is_none());
        assert!(r.lte.is_none());
    }

    #[test]
    fn test_vector_filter_default() {
        let f = VectorFilter::default();
        assert!(f.must.is_none());
        assert!(f.must_not.is_none());
        assert!(f.should.is_none());
        assert!(f.min_should.is_none());
    }

    #[test]
    fn test_vector_filter_must() {
        let f = VectorFilter::new()
            .must(FilterCondition::match_value("color", "red"))
            .must(FilterCondition::match_value("size", "large"));
        assert_eq!(f.must.as_ref().map(|v| v.len()), Some(2));
    }

    #[test]
    fn test_vector_filter_must_not() {
        let f = VectorFilter::new().must_not(FilterCondition::is_null("deleted"));
        assert_eq!(f.must_not.as_ref().map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_vector_filter_should() {
        let f = VectorFilter::new()
            .should(FilterCondition::match_value("tag", "a"))
            .should(FilterCondition::match_value("tag", "b"));
        assert_eq!(f.should.as_ref().map(|v| v.len()), Some(2));
    }

    #[test]
    fn test_filter_condition_match_value() {
        let c = FilterCondition::match_value("color", "blue");
        assert_eq!(c.field, "color");
        match c.condition {
            ConditionType::Match { value } => assert_eq!(value, "blue"),
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn test_filter_condition_match_any() {
        let values = vec![serde_json::json!("a"), serde_json::json!("b")];
        let c = FilterCondition::match_any("tags", values);
        assert_eq!(c.field, "tags");
    }

    #[test]
    fn test_filter_condition_range() {
        let range = RangeCondition::new().gt(10.0);
        let c = FilterCondition::range("price", range);
        assert_eq!(c.field, "price");
    }

    #[test]
    fn test_filter_condition_is_empty() {
        let c = FilterCondition::is_empty("description");
        assert_eq!(c.field, "description");
        assert!(matches!(c.condition, ConditionType::IsEmpty));
    }

    #[test]
    fn test_filter_condition_is_null() {
        let c = FilterCondition::is_null("deleted_at");
        assert!(matches!(c.condition, ConditionType::IsNull));
    }

    #[test]
    fn test_filter_condition_has_id() {
        let ids = vec![PointId::from("1"), PointId::from("2")];
        let c = FilterCondition::has_id(ids);
        assert_eq!(c.field, "_id");
    }

    #[test]
    fn test_filter_condition_geo_radius() {
        let center = GeoPoint::new(1.0, 2.0);
        let radius = GeoRadius::new(center, 500.0);
        let c = FilterCondition::geo_radius("location", radius);
        assert_eq!(c.field, "location");
    }

    #[test]
    fn test_filter_condition_geo_bounding_box() {
        let tl = GeoPoint::new(1.0, 2.0);
        let br = GeoPoint::new(3.0, 4.0);
        let bbox = GeoBoundingBox::new(tl, br);
        let c = FilterCondition::geo_bounding_box("location", bbox);
        assert!(matches!(c.condition, ConditionType::GeoBoundingBox(_)));
    }

    #[test]
    fn test_filter_condition_values_count() {
        let count = ValuesCountCondition::new().gt(2);
        let c = FilterCondition::values_count("tags", count);
        assert!(matches!(c.condition, ConditionType::ValuesCount(_)));
    }

    #[test]
    fn test_filter_condition_contains() {
        let c = FilterCondition::contains("title", "rust");
        assert_eq!(c.field, "title");
        match c.condition {
            ConditionType::Contains { value } => assert_eq!(value, "rust"),
            _ => panic!("expected Contains"),
        }
    }

    #[test]
    fn test_payload_selector_include() {
        let sel = PayloadSelector::include(vec!["a".into(), "b".into()]);
        assert_eq!(sel.include.as_ref().map(|v| v.len()), Some(2));
        assert!(sel.exclude.is_none());
    }

    #[test]
    fn test_payload_selector_exclude() {
        let sel = PayloadSelector::exclude(vec!["c".into()]);
        assert_eq!(sel.exclude.as_ref().map(|v| v.len()), Some(1));
        assert!(sel.include.is_none());
    }

    #[test]
    fn test_payload_selector_all() {
        let sel = PayloadSelector::all();
        assert!(sel.include.is_none());
        assert!(sel.exclude.is_none());
    }

    #[test]
    fn test_min_should_condition() {
        let condition = FilterCondition::match_value("field", "val");
        let ms = MinShouldCondition {
            conditions: vec![condition],
            min_count: 1,
        };
        assert_eq!(ms.min_count, 1);
        assert_eq!(ms.conditions.len(), 1);
    }

    #[test]
    fn test_vector_point_new() {
        let p = VectorPoint::new(42u64, vec![1.0, 2.0, 3.0]);
        assert_eq!(p.id, PointId::Num(42));
        assert_eq!(p.vector, vec![1.0, 2.0, 3.0]);
        assert!(p.payload.is_none());
    }

    #[test]
    fn test_vector_point_with_payload() {
        let mut payload = HashMap::new();
        payload.insert("key".into(), serde_json::json!("val"));
        let p = VectorPoint::new("1", vec![1.0]).with_payload(payload);
        assert!(p.payload.is_some());
    }

    #[test]
    fn test_vector_point_with_payload_kv() {
        let p = VectorPoint::new(1u64, vec![1.0])
            .with_payload_kv("color", serde_json::json!("red"))
            .with_payload_kv("size", serde_json::json!(42));
        let payload = p.payload.expect("payload expected");
        assert_eq!(payload.get("color").and_then(|v| v.as_str()), Some("red"));
        assert_eq!(payload.get("size").and_then(|v| v.as_i64()), Some(42));
    }

    #[test]
    fn test_vector_point_dimension() {
        let p = VectorPoint::new(1u64, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(p.dimension(), 4);
    }

    #[test]
    fn test_vector_point_dimension_empty() {
        let p = VectorPoint::new(1u64, vec![]);
        assert_eq!(p.dimension(), 0);
    }

    #[test]
    fn test_vector_points_new() {
        let points = vec![VectorPoint::new(1u64, vec![1.0])];
        let vp = VectorPoints::new(points);
        assert_eq!(vp.len(), 1);
        assert!(!vp.is_empty());
    }

    #[test]
    fn test_vector_points_empty() {
        let vp = VectorPoints::new(vec![]);
        assert!(vp.is_empty());
        assert_eq!(vp.len(), 0);
    }

    #[test]
    fn test_vector_points_from() {
        let points = vec![VectorPoint::new(1u64, vec![1.0])];
        let vp: VectorPoints = points.into();
        assert_eq!(vp.len(), 1);
    }

    #[test]
    fn test_upsert_result() {
        let r = UpsertResult {
            operation_id: Some(123),
            status: UpsertStatus::Completed,
        };
        assert_eq!(r.operation_id, Some(123));
        assert_eq!(r.status, UpsertStatus::Completed);
    }

    #[test]
    fn test_delete_result() {
        let r = DeleteResult {
            operation_id: None,
            deleted_count: 5,
        };
        assert!(r.operation_id.is_none());
        assert_eq!(r.deleted_count, 5);
    }

    #[test]
    fn test_upsert_status_debug() {
        assert_eq!(format!("{:?}", UpsertStatus::Completed), "Completed");
        assert_eq!(format!("{:?}", UpsertStatus::Acknowledged), "Acknowledged");
    }

    #[test]
    fn test_search_query_new() {
        let q = SearchQuery::new(vec![1.0, 2.0], 10);
        assert_eq!(q.vector, vec![1.0, 2.0]);
        assert_eq!(q.limit, 10);
        assert_eq!(q.with_payload, Some(true));
        assert!(q.with_vector.is_none());
    }

    #[test]
    fn test_search_query_with_offset() {
        let q = SearchQuery::new(vec![1.0], 10).with_offset(5);
        assert_eq!(q.offset, Some(5));
    }

    #[test]
    fn test_search_query_with_score_threshold() {
        let q = SearchQuery::new(vec![1.0], 10).with_score_threshold(0.5);
        assert_eq!(q.score_threshold, Some(0.5));
    }

    #[test]
    fn test_search_query_with_payload() {
        let q = SearchQuery::new(vec![1.0], 10).with_payload(false);
        assert_eq!(q.with_payload, Some(false));
    }

    #[test]
    fn test_search_query_with_vector() {
        let q = SearchQuery::new(vec![1.0], 10).with_vector(true);
        assert_eq!(q.with_vector, Some(true));
    }

    #[test]
    fn test_search_query_with_nprobe() {
        let q = SearchQuery::new(vec![1.0], 10).with_nprobe(64);
        assert_eq!(q.nprobe, Some(64));
    }

    #[test]
    fn test_search_query_effective_limit_default() {
        let q = SearchQuery::new(vec![1.0], 10);
        assert_eq!(q.effective_limit(), 10);
    }

    #[test]
    fn test_search_query_effective_limit_topk() {
        let q = SearchQuery::new(vec![1.0], 10).with_search_mode(SearchMode::TopK(5));
        assert_eq!(q.effective_limit(), 5);
    }

    #[test]
    fn test_search_query_effective_limit_knn() {
        let q = SearchQuery::new(vec![1.0], 10).with_search_mode(SearchMode::KNN {
            k: 20,
            ef_search: Some(100),
        });
        assert_eq!(q.effective_limit(), 20);
    }

    #[test]
    fn test_search_query_effective_limit_range() {
        let q = SearchQuery::new(vec![1.0], 10).with_search_mode(SearchMode::Range {
            radius: 0.5,
            max_results: Some(30),
        });
        assert_eq!(q.effective_limit(), 30);
    }

    #[test]
    fn test_search_query_hnsw_ef_default() {
        let q = SearchQuery::new(vec![1.0], 10);
        assert!(q.hnsw_ef().is_none());
    }

    #[test]
    fn test_search_query_hnsw_ef_knn() {
        let q = SearchQuery::new(vec![1.0], 10).with_knn(5, Some(128));
        assert_eq!(q.hnsw_ef(), Some(128));
    }

    #[test]
    fn test_search_query_hnsw_ef_topk() {
        let q = SearchQuery::new(vec![1.0], 10).with_search_mode(SearchMode::TopK(5));
        assert!(q.hnsw_ef().is_none());
    }

    #[test]
    fn test_search_query_with_knn_sets_limit() {
        let q = SearchQuery::new(vec![1.0], 10).with_knn(42, None);
        assert_eq!(q.limit, 42);
        assert!(matches!(q.search_mode, Some(SearchMode::KNN { k: 42, .. })));
    }

    #[test]
    fn test_search_query_with_range_sets_limit() {
        let q = SearchQuery::new(vec![1.0], 10).with_range(0.3, Some(25));
        assert_eq!(q.limit, 25);
        assert_eq!(q.score_threshold, Some(0.3));
    }

    #[test]
    fn test_search_query_with_range_no_max_keeps_limit() {
        let q = SearchQuery::new(vec![1.0], 10).with_range(0.3, None);
        assert_eq!(q.limit, 10);
    }

    #[test]
    fn test_search_result_new() {
        let r = SearchResult::new(42u64, 0.95);
        assert_eq!(r.id, PointId::Num(42));
        assert!((r.score - 0.95).abs() < f32::EPSILON);
        assert!(r.payload.is_none());
        assert!(r.vector.is_none());
    }

    #[test]
    fn test_search_result_with_payload() {
        let mut payload = HashMap::new();
        payload.insert("key".into(), serde_json::json!("val"));
        let r = SearchResult::new("1", 0.5).with_payload(payload);
        assert!(r.payload.is_some());
    }

    #[test]
    fn test_search_result_with_vector() {
        let r = SearchResult::new("id", 0.1).with_vector(vec![1.0, 2.0]);
        assert_eq!(r.vector, Some(vec![1.0, 2.0]));
    }

    #[test]
    fn test_search_results_empty() {
        let r = SearchResults::new(vec![]);
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn test_search_results_non_empty() {
        let results = vec![SearchResult::new(1u64, 0.9), SearchResult::new(2u64, 0.8)];
        let r = SearchResults::new(results);
        assert!(!r.is_empty());
        assert_eq!(r.len(), 2);
        assert_eq!(r.total, Some(2));
    }

    #[test]
    fn test_search_results_from_vec() {
        let results = vec![SearchResult::new(1u64, 0.9)];
        let r: SearchResults = results.into();
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_batch_search_query_new() {
        let q = SearchQuery::new(vec![1.0], 5);
        let batch = BatchSearchQuery::new(vec![q]);
        assert_eq!(batch.queries.len(), 1);
    }

    #[test]
    fn test_batch_search_query_from() {
        let q = SearchQuery::new(vec![1.0], 5);
        let batch: BatchSearchQuery = vec![q].into();
        assert_eq!(batch.queries.len(), 1);
    }

    #[test]
    fn test_search_mode_topk() {
        let mode = SearchMode::TopK(10);
        match mode {
            SearchMode::TopK(k) => assert_eq!(k, 10),
            _ => panic!("expected TopK"),
        }
    }

    #[test]
    fn test_search_mode_knn() {
        let mode = SearchMode::KNN {
            k: 20,
            ef_search: Some(200),
        };
        match mode {
            SearchMode::KNN { k, ef_search } => {
                assert_eq!(k, 20);
                assert_eq!(ef_search, Some(200));
            }
            _ => panic!("expected KNN"),
        }
    }

    #[test]
    fn test_search_mode_range() {
        let mode = SearchMode::Range {
            radius: 0.7,
            max_results: Some(50),
        };
        match mode {
            SearchMode::Range {
                radius,
                max_results,
            } => {
                assert!((radius - 0.7).abs() < f32::EPSILON);
                assert_eq!(max_results, Some(50));
            }
            _ => panic!("expected Range"),
        }
    }

    #[test]
    fn test_search_query_score_threshold_range_mode() {
        let q = SearchQuery::new(vec![1.0], 10).with_range(0.5, None);
        assert_eq!(q.score_threshold(), Some(0.5));
    }

    #[test]
    fn test_search_query_score_threshold_default() {
        let q = SearchQuery::new(vec![1.0], 10);
        assert!(q.score_threshold().is_none());
    }

    #[test]
    fn test_scan_limit_config_defaults() {
        let hnsw = HnswConfig::default();
        assert_eq!(hnsw.iterative_max_rounds, None);
        assert_eq!(hnsw.max_scan_tuples, None);
        let ivf = IvfConfig::default();
        assert_eq!(ivf.max_probes, None);
        // Missing fields fall back to engine defaults on deserialization.
        let hnsw: HnswConfig = serde_json::from_str(r#"{"m": 16, "ef_construct": 100}"#).unwrap();
        assert_eq!(hnsw.iterative_max_rounds, None);
        assert_eq!(hnsw.max_scan_tuples, None);
        assert_eq!(hnsw.ef_search, 40);
        let ivf: IvfConfig = serde_json::from_str(
            r#"{
                "min_build_points": 1,
                "sample_limit": 10,
                "kmeans_max_iter": 1,
                "drift_threshold": 0.1,
                "drift_check_interval": 1,
                "default_nprobe": 4,
                "auto_promotion": false
            }"#,
        )
        .unwrap();
        assert_eq!(ivf.max_probes, None);
        assert_eq!(ivf.default_nprobe, 4);
    }

    #[test]
    fn test_hnsw_config_validate_scan_limits() {
        let cfg = HnswConfig::new(16, 100)
            .with_iterative_max_rounds(1)
            .with_max_scan_tuples(1);
        assert!(cfg.validate().is_ok());

        let err = HnswConfig::new(16, 100)
            .with_iterative_max_rounds(0)
            .validate()
            .unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidConfig(_)));

        let err = HnswConfig::new(16, 100)
            .with_max_scan_tuples(0)
            .validate()
            .unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidConfig(_)));
    }

    #[test]
    fn test_ivf_config_validate_and_effective_max_probes() {
        assert!(IvfConfig::default().validate().is_ok());
        assert!(IvfConfig::default().with_max_probes(1).validate().is_ok());
        let err = IvfConfig::default()
            .with_max_probes(0)
            .validate()
            .unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidConfig(_)));

        // The cap never exceeds the list count and never drops below one.
        let with_cap = |max_probes| IvfConfig {
            max_probes,
            ..IvfConfig::default()
        };
        assert_eq!(with_cap(Some(2)).effective_max_probes(8), 2);
        assert_eq!(with_cap(Some(16)).effective_max_probes(8), 8);
        assert_eq!(with_cap(None).effective_max_probes(8), 8);
        assert_eq!(with_cap(None).effective_max_probes(0), 1);
    }
