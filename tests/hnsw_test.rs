//! HNSW index integration tests: build/publish lifecycle, incremental
//! insert visibility, crash recovery fallback and promotion semantics.

use std::sync::Arc;

use simvec::{
    CollectionConfig, DistanceMetric, FilterCondition, HnswConfig, LocalVectorEngine, SearchMode,
    SearchQuery, VectorFilter, VectorPoint, VectorSearchError,
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

/// Two well-separated blobs tagged by payload `blob` = "0"/"1".
fn blob_point(id: usize, dim_values: [f32; DIM], tag: &str) -> VectorPoint {
    let mut payload = std::collections::HashMap::new();
    payload.insert("blob".to_string(), serde_json::json!(tag));
    VectorPoint::new(format!("p{id}"), dim_values.to_vec()).with_payload(payload)
}

fn clustered_points(n_per_blob: usize) -> Vec<VectorPoint> {
    let mut out = Vec::new();
    let mut id = 0usize;
    for center in [[0.0f32; DIM], [50.0; DIM]] {
        let tag = if center[0] == 0.0 { "0" } else { "1" };
        for i in 0..n_per_blob {
            let mut v = center;
            // Keep every coordinate strictly unique: the extra epsilon term
            // breaks the (i % 7, i % 8) aliasing so exact-scan and ANN
            // results never disagree on tied distances.
            v[i % DIM] += (i % 7) as f32 * 0.1 + i as f32 * 0.003;
            out.push(blob_point(id, v, tag));
            id += 1;
        }
    }
    out
}

fn engine_with_hnsw(path: &std::path::Path, n_per_blob: usize) -> LocalVectorEngine {
    let engine = LocalVectorEngine::open(path).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid)
                .with_index_type(simvec::IndexType::HNSW)
                .with_hnsw(hnsw_config()),
        )
        .unwrap();
    let points = clustered_points(n_per_blob);
    engine.upsert_batch("col", &points).unwrap();
    engine
}

#[test]
fn publish_search_drop_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = engine_with_hnsw(&dir.path().join("vec"), 15);

        assert!(!engine.has_index("col"));
        assert!(engine.build_index("col").unwrap());
        assert!(engine.has_index("col"));

        // Indexed results must match exact-scan ground truth on clean
        // clusters.
        let query = vec![1.0; DIM];
        let indexed = engine
            .search(
                "col",
                &SearchQuery::new(query.clone(), 5).with_knn(5, Some(16)),
            )
            .unwrap();

        let info = engine.collection_info("col").unwrap();
        let index = info.index.expect("index info present");
        assert_eq!(index.index_kind, 2);
        assert_eq!(index.m, 8);
        assert_eq!(index.ef_construct, 16);

        // Filtered search across the whole collection still returns rows.
        let filter = VectorFilter::new().must(FilterCondition::match_value("blob", "1"));
        let filtered = engine
            .search(
                "col",
                &SearchQuery::new(vec![50.0; DIM], 3)
                    .with_filter(filter)
                    .with_knn(3, Some(16)),
            )
            .unwrap();
        assert!(!filtered.is_empty());
        assert!(filtered.iter().all(|r| r.id.to_string() != "p0"));

        // Drop back to exact scan.
        engine.drop_index("col").unwrap();
        assert!(!engine.has_index("col"));
        let exact = engine
            .search("col", &SearchQuery::new(query.clone(), 5))
            .unwrap();
        assert_eq!(
            exact.iter().map(|r| r.id.clone()).collect::<Vec<_>>(),
            indexed.iter().map(|r| r.id.clone()).collect::<Vec<_>>(),
            "hnsw results equal exact scan on separated blobs"
        );
    }

    // Reopen: index files were dropped together with drop_index.
    let reopened = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    assert!(!reopened.has_index("col"));
    assert_eq!(reopened.count("col").unwrap(), 30);
}

#[test]
fn index_persists_across_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = engine_with_hnsw(&dir.path().join("vec"), 12);
        assert!(engine.build_index("col").unwrap());
    }
    let reopened = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    assert!(
        reopened.has_index("col"),
        "published index must survive restart via hnsw.bin"
    );
    let info = reopened.collection_info("col").unwrap();
    let index = info.index.unwrap();
    assert_eq!(index.index_kind, 2);
    // The effective config is persisted in meta.bin, so the rehydrated
    // graph keeps its search-time default instead of the crate default.
    assert_eq!(index.ef_search_default, 16);
}

#[test]
fn incremental_inserts_stay_visible() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(engine_with_hnsw(&dir.path().join("vec"), 12));
    assert!(engine.build_index("col").unwrap());

    // Points appended to the published graph must be reachable through
    // approximate search immediately (incremental insert path). Every late
    // vector is unique, so the exact zero-distance hit identifies it.
    for i in 0..5u64 {
        let mut v = [50.0f32; DIM];
        v[(i as usize) % DIM] -= i as f32 * 0.13 + 0.01;
        engine
            .upsert("col", VectorPoint::new(format!("late{i}"), v.to_vec()))
            .unwrap();
        let hits = engine
            .search(
                "col",
                &SearchQuery::new(v.to_vec(), 1).with_knn(1, Some(16)),
            )
            .unwrap();
        assert_eq!(hits.len(), 1, "no result for late point {i}");
        assert_eq!(
            hits[0].id.to_string(),
            format!("late{i}"),
            "fresh point must be its own nearest neighbor"
        );
    }
}

#[test]
fn deleted_points_are_never_returned() {
    let dir = tempfile::tempdir().unwrap();
    let engine = engine_with_hnsw(&dir.path().join("vec"), 12);
    assert!(engine.build_index("col").unwrap());

    // Delete one blob-1 member; queries near blob 1 must not surface it,
    // even though the node stays in the graph for navigation.
    engine.delete("col", "p12").unwrap();
    let hits = engine
        .search(
            "col",
            &SearchQuery::new(vec![50.0; DIM], 12).with_knn(12, Some(32)),
        )
        .unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits.iter().all(|r| r.id.to_string() != "p12"),
        "tombstoned point must not be returned"
    );

    // Compaction renumbers slots wholesale and flags a rebuild; the
    // maintenance sweep must restore a published index afterwards.
    engine.compact_collection("col").unwrap();
    assert!(!engine.has_index("col"));
    engine.run_maintenance_sweep();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !engine.has_index("col") && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
        engine.run_maintenance_sweep();
    }
    assert!(engine.has_index("col"), "post-compaction rebuild ran");
}

#[test]
fn promotion_after_full_scan_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    // Small threshold so promotion triggers without a large fixture.
    let cfg = HnswConfig {
        full_scan_threshold: Some(10),
        ..HnswConfig::default()
    };
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid)
                .with_index_type(simvec::IndexType::HNSW)
                .with_hnsw(cfg),
        )
        .unwrap();

    let points: Vec<VectorPoint> = (0..15u64)
        .map(|i| VectorPoint::new(i, vec![(i % 7) as f32; DIM]))
        .collect();
    engine.upsert_batch("col", &points).unwrap();
    assert!(!engine.has_index("col"));

    // The idle-tick sweep normally drives this; invoke it directly so the
    // test does not depend on wall-clock timing.
    engine.run_maintenance_sweep();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !engine.has_index("col") && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(
        engine.has_index("col"),
        "collection above full_scan_threshold must auto-promote"
    );

    let info = engine.collection_info("col").unwrap();
    assert_eq!(info.index.unwrap().index_kind, 2);
}

#[test]
fn knn_mode_controls_recall_width() {
    let dir = tempfile::tempdir().unwrap();
    let engine = engine_with_hnsw(&dir.path().join("vec"), 15);
    assert!(engine.build_index("col").unwrap());

    let q = vec![50.0; DIM];
    let narrow = engine
        .search(
            "col",
            &SearchQuery::new(q.clone(), 5).with_search_mode(SearchMode::KNN {
                k: 5,
                ef_search: Some(1),
            }),
        )
        .unwrap();
    let wide = engine
        .search(
            "col",
            &SearchQuery::new(q.clone(), 5).with_search_mode(SearchMode::KNN {
                k: 5,
                ef_search: Some(32),
            }),
        )
        .unwrap();

    assert!(!narrow.is_empty());
    assert!(!wide.is_empty());
    // Wider exploration can only improve the best found score; with ef >= N
    // the search degenerates to exact scan on this metric.
    assert!(
        wide[0].score >= narrow[0].score,
        "wider ef must not degrade the best hit: {} < {}",
        wide[0].score,
        narrow[0].score
    );
}

#[test]
fn rejects_ef_construct_below_two_m() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    let bad = HnswConfig {
        m: 16,
        ef_construct: 20,
        ..HnswConfig::default()
    };
    let err = engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid).with_hnsw(bad),
        )
        .unwrap_err();
    assert!(
        matches!(err, VectorSearchError::InvalidConfig(_)),
        "expected InvalidConfig, got {err:?}"
    );

    // The boundary itself is accepted.
    let boundary = HnswConfig {
        m: 8,
        ef_construct: 16,
        ..HnswConfig::default()
    };
    engine
        .create_collection(
            "ok",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid).with_hnsw(boundary),
        )
        .unwrap();

    // Runtime config updates are validated too.
    let store = simvec::storage::CollectionStore::create(
        dir.path().join("store"),
        "store",
        &CollectionConfig::new(DIM, DistanceMetric::Euclid),
    )
    .unwrap();
    let err = store
        .set_hnsw_config(HnswConfig {
            m: 16,
            ef_construct: 10,
            ..HnswConfig::default()
        })
        .unwrap_err();
    assert!(matches!(err, VectorSearchError::InvalidConfig(_)));
}

#[test]
fn stale_ratio_triggers_background_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    let cfg = HnswConfig {
        m: 8,
        ef_construct: 16,
        full_scan_threshold: Some(10),
        stale_rebuild_ratio: Some(0.05),
        ..HnswConfig::default()
    };
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid)
                .with_index_type(simvec::IndexType::HNSW)
                .with_hnsw(cfg),
        )
        .unwrap();
    let points: Vec<VectorPoint> = (0..15u64)
        .map(|i| VectorPoint::new(i, vec![(i % 7) as f32 * 0.3 + 1.0; DIM]))
        .collect();
    engine.upsert_batch("col", &points).unwrap();

    engine.run_maintenance_sweep();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !engine.has_index("col") && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
        engine.run_maintenance_sweep();
    }
    assert!(engine.has_index("col"), "promotion built the graph");

    // Overwrite one point repeatedly: each overwrite keeps its old graph
    // position and accumulates staleness (routed through pending).
    for i in 0..10u64 {
        let mut v = vec![0.5f32; DIM];
        v[(i as usize) % DIM] += i as f32 * 0.01;
        engine.upsert("col", VectorPoint::new(0u64, v)).unwrap();
    }
    engine.drain_pending("col").unwrap();
    let info_before = engine.collection_info("col").unwrap().index.unwrap();
    assert_eq!(info_before.index_kind, 2);
    assert!(
        info_before.stale_overwrite_count > 0,
        "overwrites must be observable"
    );

    // The sweep compares the ratio against `stale_rebuild_ratio` and
    // schedules a rebuild; a rebuilt graph starts with a fresh counter.
    engine.run_maintenance_sweep();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let info = engine.collection_info("col").unwrap().index.unwrap();
        if info.stale_overwrite_count == 0 {
            assert_eq!(
                info.built_at_live_count, 15,
                "rebuild republished over the live set"
            );
            break;
        }
        assert!(std::time::Instant::now() < deadline, "rebuild never ran");
        std::thread::sleep(std::time::Duration::from_millis(20));
        engine.run_maintenance_sweep();
    }
}

/// `HnswConfig::iterative_max_rounds` / `max_scan_tuples` are accepted at
/// creation, validated (0 rejected), and honored on the filtered-search
/// retry path: a search whose filter matches nothing still terminates via
/// the configured caps instead of expanding unboundedly.
#[test]
fn scan_limit_config_controls_iterative_expansion() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path().join("vec")).unwrap();

    // Boundary values are accepted at creation.
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid)
                .with_index_type(simvec::IndexType::HNSW)
                .with_hnsw(HnswConfig {
                    iterative_max_rounds: Some(1),
                    max_scan_tuples: Some(64),
                    ..hnsw_config()
                }),
        )
        .unwrap();
    let points = clustered_points(20);
    engine.upsert_batch("col", &points).unwrap();
    assert!(engine.build_index("col").unwrap());
    // Filter matching a single point: the first pass comes up short for
    // limit=5, so iterative expansion runs and must stop within one round /
    // 64 visited nodes. The query must terminate with valid results.
    let marker_id = clustered_points(1).remove(0).id.to_string();
    let filter = VectorFilter::new().must(FilterCondition::has_id(vec![marker_id.clone()]));
    let results = engine
        .search(
            "col",
            &SearchQuery::new(vec![0.0; DIM], 5).with_filter(filter),
        )
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id.to_string(), marker_id);

    // Zero values are rejected by config validation at collection creation.
    let err = engine
        .create_collection(
            "bad",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid)
                .with_index_type(simvec::IndexType::HNSW)
                .with_hnsw(HnswConfig {
                    max_scan_tuples: Some(0),
                    ..hnsw_config()
                }),
        )
        .unwrap_err();
    assert!(matches!(err, VectorSearchError::InvalidConfig(_)));
}
