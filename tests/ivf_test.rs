//! IVF index integration tests: build/probe lifecycle, compaction
//! invalidation, crash recovery fallback and concurrent-search safety.

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

use sim_vec::{
    CollectionConfig, DistanceMetric, FilterCondition, IvfConfig, LocalVectorEngine, SearchQuery,
    VectorFilter, VectorPoint,
};

const DIM: usize = 8;

fn ivf_config() -> IvfConfig {
    IvfConfig {
        lists: Some(4),
        min_build_points: 1,
        sample_limit: 256,
        kmeans_max_iter: 5,
        drift_threshold: 0.10,
        drift_check_interval: u64::MAX, // no automatic checks in most tests
        default_nprobe: 2,
        auto_promotion: false,
        max_probes: None,
    }
}

/// Two well-separated blobs tagged by payload `blob` = "0"/"1".
fn blob_point(id: usize, dim_values: [f32; DIM], tag: &str) -> VectorPoint {
    let mut payload: HashMap<String, serde_json::Value> = HashMap::new();
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
            v[i % DIM] += (i % 7) as f32 * 0.1;
            out.push(blob_point(id, v, tag));
            id += 1;
        }
    }
    out
}

fn engine_with_blobs(path: &std::path::Path, n_per_blob: usize) -> LocalVectorEngine {
    let engine = LocalVectorEngine::open(path).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid).with_ivf(ivf_config()),
        )
        .unwrap();
    let points = clustered_points(n_per_blob);
    engine.upsert_batch("col", &points).unwrap();
    engine
}

#[test]
fn promote_search_rebuild_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = engine_with_blobs(&dir.path().join("vec"), 20);

        assert!(!engine.has_index("col"));
        let published = engine.build_index("col").unwrap();
        assert!(published);
        assert!(engine.has_index("col"));

        // Indexed results must match exact-scan ground truth on clean clusters.
        let query = vec![1.0; DIM];
        let indexed = engine
            .search(
                "col",
                &SearchQuery::new(query.clone(), 5).with_nprobe(4), // all lists = exact
            )
            .unwrap();

        let info = engine.collection_info("col").unwrap();
        let index = info.index.expect("index info present");
        assert_eq!(index.index_kind, 1);
        assert_eq!(index.lists, 4);

        // Filtered search across the whole collection still returns rows.
        let filter = VectorFilter::new().must(FilterCondition::match_value("blob", "1"));
        let filtered = engine
            .search(
                "col",
                &SearchQuery::new(vec![50.0; DIM], 3)
                    .with_filter(filter)
                    .with_nprobe(4),
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
            "nprobe=all probe results equal exact scan"
        );
    }

    // Reopen: index.bin was dropped together with drop_index -> exact scan.
    let reopened = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    assert!(!reopened.has_index("col"));
    assert_eq!(reopened.count("col").unwrap(), 40);
}

#[test]
fn index_persists_across_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = engine_with_blobs(&dir.path().join("vec"), 30);
        assert!(engine.build_index("col").unwrap());
    }
    let reopened = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    assert!(
        reopened.has_index("col"),
        "published index must survive restart via index.bin"
    );
    let info = reopened.collection_info("col").unwrap();
    assert_eq!(info.index.unwrap().index_kind, 1);
}

#[test]
fn restart_applies_runtime_config_to_published_index() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = engine_with_blobs(&dir.path().join("vec"), 30);
        assert!(engine.build_index("col").unwrap());
    }

    let reopened = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    assert!(reopened.has_index("col"));
    // The effective creation-time config is persisted in meta.bin, so the
    // rehydrated index restores `ivf_config()`
    // (`default_nprobe = 2`) instead of falling back to `IvfConfig::default()`.
    assert_eq!(
        reopened
            .collection_info("col")
            .unwrap()
            .index
            .unwrap()
            .nprobe_default,
        2,
        "published index must restore the persisted IVF config after restart"
    );

    // Injecting runtime settings must reach the already-published index,
    // not only future builds.
    let custom = IvfConfig {
        default_nprobe: 3,
        ..IvfConfig::default()
    };
    reopened.set_default_ivf_config(custom);
    let info = reopened.collection_info("col").unwrap();
    assert_eq!(
        info.index.unwrap().nprobe_default,
        3,
        "published index must adopt engine-supplied settings after restart"
    );

    // Unprobed queries honor the new default nprobe.
    let results = reopened
        .search("col", &SearchQuery::new(vec![50.0; DIM], 5))
        .unwrap();
    assert!(!results.is_empty());
}

#[test]
fn compact_invalidates_index_and_rebuild_restores() {
    let dir = tempfile::tempdir().unwrap();
    let engine = engine_with_blobs(&dir.path().join("vec"), 20);
    assert!(engine.build_index("col").unwrap());

    // Delete one point per blob: below the 20% tombstone threshold, so the
    // delete does not auto-compact.
    engine.delete("col", "p0").unwrap();
    engine.delete("col", "p20").unwrap();
    assert!(engine.has_index("col"));

    engine.compact_collection("col").unwrap();
    assert!(
        !engine.has_index("col"),
        "compaction renumbers slots wholesale; the index must be invalidated"
    );

    // Rebuild restores the index over the compacted slot space.
    assert!(engine.build_index("col").unwrap());
    let query = vec![50.0; DIM];
    let results = engine.search("col", &SearchQuery::new(query, 5)).unwrap();
    assert!(!results.is_empty());
    assert!(
        results.iter().all(|r| r.id.to_string() != "p20"),
        "deleted points stay deleted after compact + rebuild"
    );
}

#[test]
fn corrupt_index_bin_falls_back_to_exact_scan() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = engine_with_blobs(&dir.path().join("vec"), 20);
        assert!(engine.build_index("col").unwrap());
    }
    let col_dir = dir.path().join("vec").join("col");
    std::fs::write(col_dir.join("index.bin"), b"VIVFgarbage-not-parseable").unwrap();

    let reopened = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    assert!(
        !reopened.has_index("col"),
        "corrupt index.bin must fall back to exact scan"
    );
    assert_eq!(reopened.count("col").unwrap(), 40);
    let results = reopened
        .search("col", &SearchQuery::new(vec![1.0; DIM], 3))
        .unwrap();
    assert_eq!(results.len(), 3);
    // The corrupt file is cleaned up so a later save starts fresh.
    assert!(!col_dir.join("index.bin").exists());
}

#[test]
fn wal_replayed_points_are_searchable_after_promote() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
        engine
            .create_collection(
                "col",
                &CollectionConfig::new(DIM, DistanceMetric::Euclid).with_ivf(ivf_config()),
            )
            .unwrap();
        let points = clustered_points(20);
        engine.upsert_batch("col", &points).unwrap();
        assert!(engine.build_index("col").unwrap());

        // Insert AFTER publication: goes straight into a list.
        let late = blob_point(9999, [49.0; DIM], "1");
        engine.upsert("col", late).unwrap();
        drop(engine);
    }

    // Simulate crash recovery where WAL replay happens before the persisted
    // index loads: the replayed point must still be searchable through the
    // pending path.
    let reopened = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    let results = reopened
        .search("col", &SearchQuery::new(vec![49.0; DIM], 3))
        .unwrap();
    assert!(
        results.iter().any(|r| r.id.to_string() == "p9999"),
        "point written after publication must be found via its list"
    );
}

#[test]
fn concurrent_search_during_build_and_drop() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(engine_with_blobs(&dir.path().join("vec"), 20));

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut readers = Vec::new();
    for _ in 0..4 {
        let engine = Arc::clone(&engine);
        let stop = Arc::clone(&stop);
        readers.push(thread::spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let results = engine
                    .search("col", &SearchQuery::new(vec![1.0; DIM], 5))
                    .unwrap();
                assert_eq!(results.len(), 5);
            }
        }));
    }

    for _ in 0..3 {
        engine.build_index("col").unwrap();
        engine.drop_index("col").unwrap();
    }

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for r in readers {
        r.join().expect("reader thread must not panic");
    }
}

#[test]
fn filtered_probe_semantics_and_retry() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid).with_ivf(IvfConfig {
                lists: Some(8),
                min_build_points: 1,
                default_nprobe: 1,
                ..ivf_config()
            }),
        )
        .unwrap();

    // Eight tight blobs far apart; only blob 7 carries the marker payload.
    let mut points = Vec::new();
    for blob in 0..8i32 {
        for i in 0..10 {
            let mut v = [0.0f32; DIM];
            v[0] = blob as f32 * 100.0 + i as f32 * 0.01;
            let tag = if blob == 7 { "marker" } else { "other" };
            points.push(blob_point(points.len(), v, tag));
        }
    }
    engine.upsert_batch("col", &points).unwrap();
    assert!(engine.build_index("col").unwrap());

    let filter = VectorFilter::new().must(FilterCondition::match_value("blob", "marker"));

    // Query inside blob 7: the closest list holds the markers, so even the
    // default nprobe=1 finds them.
    let near = engine
        .search(
            "col",
            &SearchQuery::new(vec![706.0; DIM], 10).with_filter(filter.clone()),
        )
        .unwrap();
    assert_eq!(near.len(), 10);

    // Multi-round probe widening keeps doubling the probe width until every
    // list has been probed, so a filtered query reaches full recall even
    // when the matching points live far from the query.
    let far = engine
        .search(
            "col",
            &SearchQuery::new(vec![0.0; DIM], 10).with_filter(filter.clone()),
        )
        .unwrap();
    assert_eq!(
        far.len(),
        10,
        "unbounded probe widening must recover all markers"
    );

    // nprobe=all degenerates to exact: every marker is found from anywhere.
    let exact = engine
        .search(
            "col",
            &SearchQuery::new(vec![0.0; DIM], 10)
                .with_filter(filter)
                .with_nprobe(8),
        )
        .unwrap();
    assert_eq!(exact.len(), 10);
}

/// Eight tight blobs far apart; only blob 7 carries the marker payload.
fn engine_with_marker_blobs(
    path: &std::path::Path,
    max_probes: Option<usize>,
) -> LocalVectorEngine {
    let engine = LocalVectorEngine::open(path).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid).with_ivf(IvfConfig {
                lists: Some(8),
                min_build_points: 1,
                default_nprobe: 1,
                max_probes,
                ..ivf_config()
            }),
        )
        .unwrap();

    let mut points = Vec::new();
    for blob in 0..8i32 {
        for i in 0..10 {
            let mut v = [0.0f32; DIM];
            v[0] = blob as f32 * 100.0 + i as f32 * 0.01;
            let tag = if blob == 7 { "marker" } else { "other" };
            points.push(blob_point(points.len(), v, tag));
        }
    }
    engine.upsert_batch("col", &points).unwrap();
    assert!(engine.build_index("col").unwrap());
    engine
}

/// `IvfConfig::max_probes` must truncate the multi-round widening loop:
/// each configured ceiling bounds how many doublings a short filtered
/// search may attempt (observable through the retry counter).
#[test]
fn max_probes_caps_probe_widening() {
    let filter = || VectorFilter::new().must(FilterCondition::match_value("blob", "marker"));

    // cap == initial nprobe (1): widening disabled entirely, zero retries.
    let dir = tempfile::tempdir().unwrap();
    let engine = engine_with_marker_blobs(&dir.path().join("vec"), Some(1));
    let before = engine
        .collection_metrics("col")
        .unwrap()
        .search_nprobe_retries;
    let results = engine
        .search(
            "col",
            &SearchQuery::new(vec![0.0; DIM], 10).with_filter(filter()),
        )
        .unwrap();
    assert!(
        results.len() < 10,
        "cap=1 must stop before reaching the marker lists"
    );
    assert_eq!(
        engine
            .collection_metrics("col")
            .unwrap()
            .search_nprobe_retries,
        before,
        "no widening attempt may be recorded once the cap is already reached"
    );

    // cap == 2 from nprobe 1: exactly one widening is allowed.
    let dir = tempfile::tempdir().unwrap();
    let engine = engine_with_marker_blobs(&dir.path().join("vec"), Some(2));
    let before = engine
        .collection_metrics("col")
        .unwrap()
        .search_nprobe_retries;
    let results = engine
        .search(
            "col",
            &SearchQuery::new(vec![0.0; DIM], 10).with_filter(filter()),
        )
        .unwrap();
    assert!(results.len() < 10, "two lists cannot cover eight blobs");
    assert_eq!(
        engine
            .collection_metrics("col")
            .unwrap()
            .search_nprobe_retries
            - before,
        1,
        "exactly one doubling between nprobe 1 and the cap of 2"
    );

    // No cap: widening proceeds until all lists are probed (full recall),
    // with one retry per doubling (1 -> 2 -> 4 -> 8).
    let dir = tempfile::tempdir().unwrap();
    let engine = engine_with_marker_blobs(&dir.path().join("vec"), None);
    let before = engine
        .collection_metrics("col")
        .unwrap()
        .search_nprobe_retries;
    let results = engine
        .search(
            "col",
            &SearchQuery::new(vec![0.0; DIM], 10).with_filter(filter()),
        )
        .unwrap();
    assert_eq!(results.len(), 10);
    assert_eq!(
        engine
            .collection_metrics("col")
            .unwrap()
            .search_nprobe_retries
            - before,
        3,
        "one retry per doubling until the list count is reached"
    );
}

#[test]
fn build_window_no_missing_results() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(engine_with_blobs(&dir.path().join("vec"), 50));

    // Keep inserting while builds run so slots land in every routing state:
    // published-list assignment, pending set, and plain exact-scan windows.
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let inserted: Arc<std::sync::Mutex<Vec<Vec<f32>>>> = Arc::default();
    let mut writers = Vec::new();
    for w in 0..2usize {
        let engine = Arc::clone(&engine);
        let stop = Arc::clone(&stop);
        let inserted = Arc::clone(&inserted);
        writers.push(thread::spawn(move || {
            let mut i = 0usize;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let mut v = [50.0f32; DIM];
                v[(i + w) % DIM] += (i % 9) as f32 * 0.01;
                if engine
                    .upsert("col", VectorPoint::new(format!("w{w}_{i}"), v.to_vec()))
                    .is_err()
                {
                    break;
                }
                inserted.lock().unwrap().push(v.to_vec());
                i += 1;
            }
        }));
    }

    for _ in 0..2 {
        engine.build_index("col").unwrap();
        engine.drop_index("col").unwrap();
    }
    // End with a published index so verification exercises probe searches.
    engine.build_index("col").unwrap();

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for w in writers {
        w.join().expect("writer thread must not panic");
    }

    let all = inserted.lock().unwrap().clone();
    assert!(!all.is_empty(), "writers must have inserted during builds");
    // A zero-distance query pins the point to its own nearest list, so even
    // the default nprobe must surface it if it was routed correctly.
    for v in &all {
        let hits = engine
            .search("col", &SearchQuery::new(v.clone(), 1).with_vector(true))
            .unwrap();
        assert_eq!(hits.len(), 1, "no result for query {v:?}");
        assert_eq!(
            hits[0].vector.as_deref(),
            Some(v.as_slice()),
            "point written during a build window went missing"
        );
    }
}

#[test]
fn drift_triggers_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    engine
        .create_collection(
            "col",
            &CollectionConfig::new(DIM, DistanceMetric::Euclid).with_ivf(IvfConfig {
                lists: Some(2),
                min_build_points: 1,
                drift_check_interval: 1,
                // Drift maintenance must fire even without auto promotion.
                auto_promotion: false,
                ..ivf_config()
            }),
        )
        .unwrap();

    let mut points = Vec::new();
    for i in 0..40usize {
        let center = if i % 2 == 0 { [0.0; DIM] } else { [50.0; DIM] };
        points.push(blob_point(i, center, "0"));
    }
    engine.upsert_batch("col", &points).unwrap();
    assert!(engine.build_index("col").unwrap());
    let before = engine
        .collection_info("col")
        .unwrap()
        .index
        .expect("index info present")
        .built_at_live_count;
    assert_eq!(before, 40);

    // Pull the distribution toward a brand-new region: the stale centroids
    // sit far from much of the data, so the measured drift ratio explodes.
    let shifted: Vec<VectorPoint> = (0..30usize)
        .map(|i| blob_point(1000 + i, [200.0; DIM], "0"))
        .collect();
    engine.upsert_batch("col", &shifted).unwrap();

    engine.run_maintenance_sweep();

    // The sweep enqueued a rebuild; the live maintenance worker executes it.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut after = before;
    while after == before && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
        after = engine
            .collection_info("col")
            .unwrap()
            .index
            .expect("index info present")
            .built_at_live_count;
    }
    assert_eq!(
        after, 70,
        "drift above threshold must trigger an automatic rebuild"
    );
}
