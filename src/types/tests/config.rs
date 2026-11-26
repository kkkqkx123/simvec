use crate::error::VectorSearchError;
use crate::types::*;

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
    let cfg = CollectionConfig::new(384, DistanceMetric::Euclid).with_index_type(IndexType::FLAT);
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
fn test_index_type_default() {
    assert_eq!(IndexType::default(), IndexType::HNSW);
}

#[test]
fn test_distance_metric_default() {
    assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
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
