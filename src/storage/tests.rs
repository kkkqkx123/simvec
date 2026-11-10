use super::*;
use crate::types::{CollectionConfig, DistanceMetric, VectorPoint};

fn config(dim: usize) -> CollectionConfig {
    CollectionConfig::new(dim, DistanceMetric::Cosine)
}

fn point(id: u64, dim: usize) -> VectorPoint {
    VectorPoint::new(
        id,
        (0..dim)
            .map(|i| ((id as usize * 31 + i) % 100) as f32 / 100.0)
            .collect(),
    )
}

fn point_with_payload(id: u64, dim: usize, payload: crate::types::Payload) -> VectorPoint {
    VectorPoint::new(id, (0..dim).map(|_| 0.5).collect()).with_payload(payload)
}

/// Upsert through the WAL-backed path, which is the only write path the
/// engine uses: the log carries the payload so derived files can stay
/// unsynced until a checkpoint.
fn upsert_logged(store: &CollectionStore, point: &VectorPoint) {
    store
        .apply_ops(&[WalRecord::Upsert {
            point: WalPoint::from_point(point).unwrap(),
        }])
        .unwrap();
}

/// Delete through the WAL-backed path. Returns whether the id was live.
fn delete_logged(store: &CollectionStore, id: &PointId) -> bool {
    let existed = store.get(id).unwrap().is_some();
    store
        .apply_ops(&[WalRecord::Delete {
            point_id: id.to_string(),
        }])
        .unwrap();
    existed
}

/// Result-returning upsert for the rejection tests.
fn upsert_logged_result(store: &CollectionStore, point: &VectorPoint) -> Result<()> {
    store.apply_ops(&[WalRecord::Upsert {
        point: WalPoint::from_point(point).unwrap(),
    }])
}

#[test]
fn test_create_open_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_a");
    let store = CollectionStore::create(&store_dir, "col_a", &config(4)).unwrap();
    upsert_logged(&store, &point(1, 4));
    upsert_logged(&store, &point(2, 4));
    assert_eq!(store.count(), 2);
    drop(store);

    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), 2);
    let p1 = reopened.get(&PointId::Num(1)).unwrap().unwrap();
    assert_eq!(p1.vector.len(), 4);
    assert_eq!(p1.vector, point(1, 4).vector);
    let p2 = reopened.get(&PointId::Num(2)).unwrap().unwrap();
    assert_eq!(p2.vector, point(2, 4).vector);
    assert!(reopened.get(&PointId::Num(99)).unwrap().is_none());
}

#[test]
fn test_payload_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_p");
    let store = CollectionStore::create(&store_dir, "col_p", &config(2)).unwrap();
    let mut payload = HashMap::new();
    payload.insert("color".to_string(), serde_json::json!("red"));
    payload.insert("size".to_string(), serde_json::json!(42));
    upsert_logged(&store, &point_with_payload(1, 2, payload.clone()));

    let got = store.get(&PointId::Num(1)).unwrap().unwrap();
    assert_eq!(got.payload, Some(payload.clone()));

    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    let got = reopened.get(&PointId::Num(1)).unwrap().unwrap();
    assert_eq!(got.payload, Some(payload));
}

#[test]
fn test_set_payload_fields_merges_and_reindexes_bitmap() {
    use crate::filter;
    use crate::types::{ConditionType, FilterCondition, VectorFilter};

    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_pf");
    let store = CollectionStore::create(&store_dir, "col_pf", &config(2)).unwrap();
    let mut payload = HashMap::new();
    payload.insert("color".to_string(), serde_json::json!("red"));
    upsert_logged(&store, &point_with_payload(1, 2, payload));

    // Partial merge overwrites the given keys and preserves the rest.
    let fields: Payload = [
        ("color", serde_json::json!("blue")),
        ("size", serde_json::json!(7)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    store.set_payload_fields(&PointId::Num(1), fields).unwrap();
    let got = store.get(&PointId::Num(1)).unwrap().unwrap();
    let got = got.payload.expect("payload present");
    assert_eq!(got.get("color"), Some(&serde_json::json!("blue")));
    assert_eq!(got.get("size"), Some(&serde_json::json!(7)));

    // The pre-filter bitmap must reflect the merged values so an
    // equality filter on the new value finds the slot and on the old
    // value no longer claims it.
    let make_filter = |value: &str| {
        VectorFilter::new().must(FilterCondition {
            field: "color".to_string(),
            condition: ConditionType::Match {
                value: value.to_string(),
            },
        })
    };
    let inner = store.inner.read();
    let blue = inner
        .filter_bitmap
        .build_mask(&make_filter("blue"))
        .expect("merged value is indexed");
    assert!(!blue.not_any());
    match inner.filter_bitmap.build_mask(&make_filter("red")) {
        None => {}
        Some(m) => assert!(!m.any()),
    }
    // Full filter evaluation agrees with the stored payload.
    let id = PointId::Num(1);
    assert!(filter::matches(&make_filter("blue"), &id, Some(&got)).unwrap());
    assert!(!filter::matches(&make_filter("red"), &id, Some(&got)).unwrap());
}

#[test]
fn test_set_payload_field_wal_replay_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_r");
    let store = CollectionStore::create(&store_dir, "col_r", &config(2)).unwrap();
    upsert_logged(&store, &point(1, 2));
    store
        .set_payload_field(
            &PointId::Num(1),
            "env".to_string(),
            serde_json::json!("prod"),
        )
        .unwrap();
    drop(store);

    let reopened = CollectionStore::open(&store_dir).unwrap();
    let payload = reopened
        .get(&PointId::Num(1))
        .unwrap()
        .unwrap()
        .payload
        .expect("payload recreated by partial update");
    assert_eq!(payload.get("env"), Some(&serde_json::json!("prod")));
}

#[test]
fn test_numeric_index_range_accelerated_search() {
    use crate::types::{
        ConditionType, FilterCondition, PayloadSchemaType, RangeCondition, SearchQuery,
        VectorFilter,
    };

    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_num");
    let store = CollectionStore::create(&store_dir, "col_num", &config(2)).unwrap();
    for (id, price) in [(1u64, 10.0f64), (2, 50.0), (3, 120.0)] {
        let mut p = HashMap::new();
        p.insert("price".to_string(), serde_json::json!(price));
        upsert_logged(&store, &point_with_payload(id, 2, p));
    }
    assert!(store.list_payload_indexes().is_empty());
    store
        .create_payload_index("price", PayloadSchemaType::Float)
        .unwrap();
    assert_eq!(
        store.list_payload_indexes(),
        vec![("price".to_string(), PayloadSchemaType::Float)]
    );

    // Range filter via the numeric index mask; identical results must
    // hold with and without the index because the post-filter still
    // re-evaluates the full filter. All vectors score identically, so
    // ties break by slot ascending.
    let make_query = || {
        SearchQuery::new(vec![0.5, 0.5], 10).with_filter(VectorFilter::new().must(
            FilterCondition {
                field: "price".to_string(),
                condition: ConditionType::Range(RangeCondition::new().gt(20.0).lte(130.0)),
            },
        ))
    };
    let hits = store
        .search(&make_query())
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect::<Vec<_>>();
    assert_eq!(hits, vec![PointId::Num(2), PointId::Num(3)]);

    // Definitions survive reopen and contents rebuild from payloads.
    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    let hits = reopened
        .search(&make_query())
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect::<Vec<_>>();
    assert_eq!(hits, vec![PointId::Num(2), PointId::Num(3)]);
    assert!(reopened.delete_payload_index("price").unwrap());
    assert!(!reopened.delete_payload_index("price").unwrap());
    assert!(reopened.list_payload_indexes().is_empty());
}

#[test]
fn test_search_payload_selector_projection() {
    use crate::types::{PayloadSelector, SearchQuery};

    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_sel");
    let store = CollectionStore::create(&store_dir, "col_sel", &config(2)).unwrap();
    let mut p = HashMap::new();
    p.insert("keep".to_string(), serde_json::json!("yes"));
    p.insert("drop".to_string(), serde_json::json!("gone"));
    upsert_logged(&store, &point_with_payload(1, 2, p.clone()));

    let query = SearchQuery::new(vec![0.5, 0.5], 5)
        .with_payload_selector(PayloadSelector::include(vec!["keep".to_string()]));
    let results = store.search(&query).unwrap();
    let payload = results[0].payload.as_ref().expect("payload present");
    assert_eq!(payload.len(), 1);
    assert_eq!(payload.get("keep"), Some(&serde_json::json!("yes")));

    // Selector building helpers merge into one selector.
    let mut sel = PayloadSelector::all();
    sel.include = Some(vec!["a".to_string()]);
    assert_eq!(sel.apply(&p).len(), 0, "missing field yields empty map");
}

#[test]
fn test_upsert_overwrite_reuses_slot() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_o");
    let store = CollectionStore::create(&store_dir, "col_o", &config(4)).unwrap();
    upsert_logged(&store, &point(1, 4));
    upsert_logged(&store, &point(2, 4));

    let overwritten = VectorPoint::new(1u64, vec![9.0; 4]);
    upsert_logged(&store, &overwritten);

    assert_eq!(store.count(), 2, "overwrite must not grow live count");
    let got = store.get(&PointId::Num(1)).unwrap().unwrap();
    assert_eq!(got.vector, vec![9.0; 4]);
    let got2 = store.get(&PointId::Num(2)).unwrap().unwrap();
    assert_eq!(got2.vector, point(2, 4).vector);
}

#[test]
fn test_grow_across_segments() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_g");
    let store =
        CollectionStore::create_with_segment_slots(&store_dir, "col_g", &config(4), 16).unwrap();

    let total = 100u64;
    for i in 0..total {
        upsert_logged(&store, &point(i, 4));
    }
    assert_eq!(store.count(), total);
    let meta = store.meta();
    assert!(
        meta.slot_capacity >= total,
        "capacity grown to {}",
        meta.slot_capacity
    );
    assert_eq!(meta.slot_capacity % meta.segment_slots as u64, 0);

    for i in (0..total).step_by(7) {
        let got = store.get(&PointId::Num(i)).unwrap().unwrap();
        assert_eq!(got.vector, point(i, 4).vector);
    }

    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), total);
    for i in (0..total).step_by(11) {
        let got = reopened.get(&PointId::Num(i)).unwrap().unwrap();
        assert_eq!(got.vector, point(i, 4).vector);
    }
}

#[test]
fn test_delete_tombstone_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_d");
    let store = CollectionStore::create(&store_dir, "col_d", &config(4)).unwrap();
    upsert_logged(&store, &point(1, 4));
    upsert_logged(&store, &point(2, 4));
    upsert_logged(&store, &point(3, 4));

    assert!(delete_logged(&store, &PointId::Num(2)));
    assert!(
        !delete_logged(&store, &PointId::Num(2)),
        "double delete is a no-op"
    );
    assert!(store.get(&PointId::Num(2)).unwrap().is_none());
    assert_eq!(store.count(), 2);
    // The tombstone stays pending until compaction runs (the engine
    // schedules it in the background); visibility never depends on it.
    let meta = store.meta();
    assert_eq!(meta.tombstone_count, 1);
    store.compact().unwrap();
    let meta = store.meta();
    assert_eq!(meta.tombstone_count, 0);
    assert_eq!(meta.next_slot, 2);

    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), 2);
    assert!(reopened.get(&PointId::Num(2)).unwrap().is_none());
    assert_eq!(
        reopened.get(&PointId::Num(1)).unwrap().unwrap().vector,
        point(1, 4).vector
    );
    assert_eq!(
        reopened.get(&PointId::Num(3)).unwrap().unwrap().vector,
        point(3, 4).vector
    );
}

#[test]
fn test_upsert_then_delete_then_reinsert() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_r");
    let store = CollectionStore::create(&store_dir, "col_r", &config(4)).unwrap();
    upsert_logged(&store, &point(1, 4));
    assert!(delete_logged(&store, &PointId::Num(1)));

    // Without an intervening compaction the re-insert takes a fresh slot:
    // the stale tombstone stays pending and flags the collection for
    // background compaction, while visibility is already correct.
    upsert_logged(&store, &point(1, 4));
    assert_eq!(store.count(), 1);
    let meta = store.meta();
    assert_eq!(meta.next_slot, 2);
    assert_eq!(meta.tombstone_count, 1);
    assert!(store.needs_compaction());
    store.compact().unwrap();
    assert!(!store.needs_compaction());
    assert!(store.get(&PointId::Num(1)).unwrap().is_some());
}

#[test]
fn test_rejects_invalid_dimension() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(dir.path().join("col_v"), "col_v", &config(4)).unwrap();
    let bad = VectorPoint::new(1u64, vec![1.0, 2.0]);
    let err = upsert_logged_result(&store, &bad).unwrap_err();
    assert!(matches!(
        err,
        VectorSearchError::InvalidVectorDimension {
            expected: 4,
            actual: 2
        }
    ));
}

#[test]
fn test_rejects_non_finite_elements() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(dir.path().join("col_n"), "col_n", &config(2)).unwrap();
    let bad = VectorPoint::new(1u64, vec![f32::NAN, 1.0]);
    let err = upsert_logged_result(&store, &bad).unwrap_err();
    assert!(matches!(err, VectorSearchError::NonFiniteElement(0)));

    let bad_inf = VectorPoint::new(1u64, vec![1.0, f32::INFINITY]);
    let err = upsert_logged_result(&store, &bad_inf).unwrap_err();
    assert!(matches!(err, VectorSearchError::NonFiniteElement(1)));
}

#[test]
fn test_create_validates_inputs() {
    let dir = tempfile::tempdir().unwrap();

    let err = CollectionStore::create(dir.path().join("bad"), "", &config(4)).unwrap_err();
    assert!(matches!(err, VectorSearchError::InvalidCollectionName(_)));

    let err = CollectionStore::create(dir.path().join("bad"), "a/b", &config(4)).unwrap_err();
    assert!(matches!(err, VectorSearchError::InvalidCollectionName(_)));

    let mut cfg = config(0);
    cfg.vector_size = 0;
    let err = CollectionStore::create(dir.path().join("bad"), "col", &cfg).unwrap_err();
    assert!(matches!(
        err,
        VectorSearchError::InvalidVectorDimension {
            expected: 1,
            actual: 0
        }
    ));

    let mut cfg = config(4);
    cfg.distance = DistanceMetric::Manhattan;
    let manhattan_dir = dir.path().join("manhattan_ok");
    let store = CollectionStore::create(&manhattan_dir, "col", &cfg).unwrap();
    assert_eq!(store.meta().distance, DistanceMetric::Manhattan);
}

#[test]
fn test_create_existing_dir_conflicts() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_x");
    let _store = CollectionStore::create(&store_dir, "col_x", &config(4)).unwrap();
    let err = CollectionStore::create(&store_dir, "col_x", &config(4)).unwrap_err();
    assert!(matches!(err, VectorSearchError::CollectionAlreadyExists(_)));
}

#[test]
fn test_corrupt_vectors_length_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_c");
    let store = CollectionStore::create(&store_dir, "col_c", &config(4)).unwrap();
    upsert_logged(&store, &point(1, 4));
    drop(store);

    // Corrupt vectors.bin by truncating it.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(store_dir.join("vectors.bin"))
        .unwrap();
    file.set_len(16).unwrap();
    drop(file);

    let err = CollectionStore::open(&store_dir).unwrap_err();
    assert!(matches!(err, VectorSearchError::CorruptData(_)));
}

/// Pending slots accumulate while an HNSW graph is published, then the
/// drain moves them into the graph itself: the queue empties and the
/// points stay reachable with only the graph answering.
#[test]
fn test_drain_pending_empty_backlog_and_stay_visible() {
    use crate::types::{IndexType, SearchQuery};

    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_d");
    let cfg = config(2).with_index_type(IndexType::HNSW);
    let store = CollectionStore::create(&store_dir, "col_d", &cfg).unwrap();

    // Build the graph over the first block of points.
    for id in 0..40u64 {
        upsert_logged(&store, &VectorPoint::new(id, vec![id as f32 * 0.01, 1.0]));
    }
    assert!(store.build_index().unwrap(), "HNSW graph must publish");
    assert_eq!(store.pending_len(), 0, "build adopts its own slots");

    // Later writes are queued instead of being inserted inline.
    for id in 40..60u64 {
        upsert_logged(&store, &VectorPoint::new(id, vec![id as f32 * 0.01, 1.0]));
    }
    assert_eq!(store.pending_len(), 20, "post-publish writes are pending");

    store.drain_pending_hnsw();
    assert_eq!(store.pending_len(), 0, "drain must empty the queue");

    // With an empty queue the graph is the only index that can answer, so
    // a hit proves the drained points landed in it.
    let hits = store.search(&SearchQuery::new(vec![0.55, 1.0], 3)).unwrap();
    assert!(!hits.is_empty(), "drained points must stay reachable");
    for hit in &hits {
        assert!(
            store.get(&hit.id).unwrap().is_some(),
            "hit {} must resolve to a live point",
            hit.id
        );
    }
}

/// A graph with no tombstone must not be walked: the repair reports no work
/// and leaves the published structure untouched.
#[test]
fn test_repair_skips_graph_without_tombstones() {
    use crate::types::IndexType;

    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_rp");
    let cfg = config(2).with_index_type(IndexType::HNSW);
    let store = CollectionStore::create(&store_dir, "col_rp", &cfg).unwrap();
    for id in 0..12u64 {
        upsert_logged(&store, &VectorPoint::new(id, vec![id as f32, 1.0]));
    }
    assert!(store.build_index().unwrap(), "HNSW graph must publish");

    assert_eq!(
        store.repair_hnsw().unwrap(),
        0,
        "zero tombstones must early-exit"
    );

    // Tombstone one slot: now the repair does real work.
    store
        .apply_ops(&[WalRecord::Delete {
            point_id: "3".to_string(),
        }])
        .unwrap();
    assert!(
        store.repair_hnsw().unwrap() > 0,
        "a tombstoned neighbour must be repaired"
    );
}

/// meta.bin is force-saved only when capacity grows, so a store that stops
/// with a stale meta.bin must still reopen with every logged point: the
/// slot water marks and counts are replayed from the WAL.
#[test]
fn test_stale_meta_bin_recovers_from_wal() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_m");
    {
        let store = CollectionStore::create(&store_dir, "col_m", &config(2)).unwrap();
        for id in 0..5u64 {
            upsert_logged(&store, &VectorPoint::new(id, vec![id as f32, 0.5]));
        }
        // Fewer batches than META_SAVE_INTERVAL and no capacity growth, so
        // meta.bin still carries the pre-write water mark.
        let on_disk = Meta::load(&store_dir).unwrap();
        assert_eq!(on_disk.next_slot, 0, "meta.bin must lag the live set");
    }
    {
        let reopened = CollectionStore::open(&store_dir).unwrap();
        assert_eq!(reopened.count(), 5);
        for id in 0..5u64 {
            assert!(
                reopened.get(&PointId::Num(id)).unwrap().is_some(),
                "point {id} must come back from the log"
            );
        }
    }
}

/// `max_points` is an admission contract: a write past the cap fails
/// before it is logged, leaves the live set untouched, and survives
/// reopen as the same cap on the same data.
#[test]
fn test_max_points_cap_rejects_write_and_keeps_data() {
    use crate::types::SearchQuery;

    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_cap");
    let cfg = config(2).with_max_points(4);
    let store = CollectionStore::create(&store_dir, "col_cap", &cfg).unwrap();
    for id in 0..4u64 {
        upsert_logged(&store, &VectorPoint::new(id, vec![id as f32, 0.5]));
    }
    assert_eq!(store.count(), 4);

    let err = store
        .apply_ops(&[WalRecord::Upsert {
            point: WalPoint::from_point(&VectorPoint::new(4, vec![4.0, 0.5])).unwrap(),
        }])
        .unwrap_err();
    assert!(
        matches!(err, VectorSearchError::CapacityExceeded { .. }),
        "unexpected error: {err}"
    );
    assert_eq!(store.count(), 4, "rejected write must not allocate a slot");
    assert!(store.get(&PointId::Num(4)).unwrap().is_none());
    store.checkpoint().unwrap();

    // The rejected record must never have entered the log, so the cap
    // cannot poison a later open.
    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), 4);
    assert_eq!(reopened.meta().max_points, Some(4));
    assert!(reopened.get(&PointId::Num(4)).unwrap().is_none());
    let err = reopened
        .apply_ops(&[WalRecord::Upsert {
            point: WalPoint::from_point(&VectorPoint::new(5, vec![5.0, 0.5])).unwrap(),
        }])
        .unwrap_err();
    assert!(matches!(err, VectorSearchError::CapacityExceeded { .. }));
    let hits = reopened
        .search(&SearchQuery::new(vec![3.0, 0.5], 4))
        .unwrap();
    assert_eq!(hits.len(), 4, "existing points must stay searchable");
}
