use super::dense_snapshot::DenseSnapshot;
use super::*;
use crate::error::{Result, VectorSearchError};
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
    assert_eq!(p1.vector().len(), 4);
    assert_eq!(p1.vector(), point(1, 4).vector());
    let p2 = reopened.get(&PointId::Num(2)).unwrap().unwrap();
    assert_eq!(p2.vector(), point(2, 4).vector());
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
    assert_eq!(got.vector(), vec![9.0; 4]);
    let got2 = store.get(&PointId::Num(2)).unwrap().unwrap();
    assert_eq!(got2.vector(), point(2, 4).vector());
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
        assert_eq!(got.vector(), point(i, 4).vector());
    }

    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), total);
    for i in (0..total).step_by(11) {
        let got = reopened.get(&PointId::Num(i)).unwrap().unwrap();
        assert_eq!(got.vector(), point(i, 4).vector());
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
        reopened.get(&PointId::Num(1)).unwrap().unwrap().vector(),
        point(1, 4).vector()
    );
    assert_eq!(
        reopened.get(&PointId::Num(3)).unwrap().unwrap().vector(),
        point(3, 4).vector()
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

/// `meta.bin` is the authority for capacity: a short `vectors.bin` is a crash
/// artifact of the grow sequence, not a reason to make the collection
/// permanently unopenable. The file is zero-extended and the WAL restores the
/// slots it still covers.
#[test]
fn test_short_vectors_file_is_repaired_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_c");
    let store = CollectionStore::create(&store_dir, "col_c", &config(4)).unwrap();
    upsert_logged(&store, &point(1, 4));
    let capacity = store.meta().slot_capacity;
    let data_dir = store.data_dir();
    drop(store);

    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(data_dir.join("vectors.bin"))
        .unwrap();
    file.set_len(16).unwrap();
    drop(file);

    let store = CollectionStore::open(&store_dir).expect("open must repair the length");
    assert_eq!(store.meta().slot_capacity, capacity);
    let len = std::fs::metadata(data_dir.join("vectors.bin"))
        .unwrap()
        .len();
    assert_eq!(len, capacity * 4 * 4);
    // The vector itself comes back from the WAL, which was still intact.
    assert!(store.get(&PointId::from(1u64)).unwrap().is_some());
}

/// A `vectors.bin` longer than the declared capacity is the other half of the
/// same crash window: the capacity from meta wins and the extra tail is
/// ignored instead of blocking startup.
#[test]
fn test_oversized_vectors_file_is_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_c2");
    let store = CollectionStore::create(&store_dir, "col_c2", &config(4)).unwrap();
    upsert_logged(&store, &point(1, 4));
    let capacity = store.meta().slot_capacity;
    let data_dir = store.data_dir();
    drop(store);

    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(data_dir.join("vectors.bin"))
        .unwrap();
    file.set_len(capacity * 4 * 4 + 4096).unwrap();
    drop(file);

    let store = CollectionStore::open(&store_dir).expect("open must tolerate the extra tail");
    assert_eq!(store.meta().slot_capacity, capacity);
    assert_eq!(store.count(), 1);
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
        let on_disk = Meta::load(&store.data_dir()).unwrap();
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

/// The bulk of a checkpoint (payload pages, tracker, `keys.bin`,
/// `vectors.bin`, tombstone table) must run *without* the store write lock:
/// holding it across those fsyncs freezes readers as well as writers. Phase 2
/// then takes the lock only for the water marks, which are microseconds.
#[test]
fn test_checkpoint_flush_runs_without_the_write_lock() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_ckpt_lock");
    let store = CollectionStore::create(&store_dir, "col_ckpt_lock", &config(4)).unwrap();
    for id in 1..=8 {
        upsert_logged(&store, &point(id, 4));
    }
    let before = store.count();

    let mut probed = false;
    store
        .checkpoint_with_probe(|| {
            probed = true;
            assert!(
                store.write_lock_free(),
                "phase 1 must not hold the store write lock"
            );
            // Reads keep working while the flush is in progress: this would
            // deadlock if the write lock were held.
            assert_eq!(store.count(), before);
            assert_eq!(store.scroll(100, None, None, None).unwrap().0.len(), 8);
        })
        .unwrap();
    assert!(probed, "the probe must run inside the checkpoint");

    // The checkpoint is still a full one: the log is empty and the state
    // reloads from the data files.
    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), 8);
}

#[test]
fn test_id_map_handles_both_id_kinds() {
    let mut map: IdMap = IdMap::with_capacity(4);
    let num = PointId::Num(7);
    let uuid = PointId::Uuid("550e8400-e29b-41d4-a716-446655440000".to_string());

    assert!(map.get(&num).is_none());
    map.insert(&num, 3);
    map.insert(&uuid, 4);
    assert_eq!(map.get(&num), Some(3));
    assert_eq!(map.get(&uuid), Some(4));
    assert!(map.contains_key(&PointId::Num(7)));
    assert_eq!(map.len(), 2);

    // Numeric and string forms of the same id must not collide: `PointId`
    // normalizes "7" to `Num(7)`, so a uuid key is a genuinely different id.
    assert!(map.get(&PointId::from("7")).is_some());
    assert!(map.get(&PointId::from("not-a-number")).is_none());

    assert_eq!(map.insert(&num, 9), Some(3));
    assert_eq!(map.get(&num), Some(9));
    assert_eq!(map.remove(&num), Some(9));
    assert_eq!(map.len(), 1);
    map.clear();
    assert_eq!(map.len(), 0);
}

/// A wide-vector collection must not pay for a 25 MiB first segment: the
/// segment slot count is derived from the dimension, and `vectors.bin` starts
/// out at the resulting size.
#[test]
fn test_wide_vectors_get_smaller_segments() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_wide");
    let store = CollectionStore::create(&store_dir, "col_wide", &config(768)).unwrap();
    let meta = store.meta();
    assert_eq!(meta.segment_slots, 1365);
    // One segment, sized by the target rather than by a fixed slot count.
    let expected = meta.slot_capacity * 768 * 4;
    assert_eq!(
        std::fs::metadata(store.data_dir().join("vectors.bin"))
            .unwrap()
            .len(),
        expected
    );
    assert!(
        expected <= 8 * 1024 * 1024,
        "first segment is {expected} bytes"
    );

    // A narrow collection keeps the historical 8192-slot segment.
    let narrow_dir = dir.path().join("col_narrow");
    let narrow = CollectionStore::create(&narrow_dir, "col_narrow", &config(8)).unwrap();
    assert_eq!(
        narrow.meta().segment_slots,
        super::meta::SEGMENT_SLOTS_DEFAULT
    );
}

/// The generation manifest is the commit point: a staging directory left by
/// a crashed compaction is never adopted on open.
#[test]
fn test_unpublished_generation_is_ignored() {
    let tmp = tempfile::tempdir().unwrap();
    let store_dir = tmp.path().join("col_gen");
    {
        let store = CollectionStore::create(&store_dir, "col_gen", &config(4)).unwrap();
        upsert_logged(&store, &point(7, 4));
        store.checkpoint().unwrap();
    }
    let published = data_dir(&store_dir);
    let orphan = store_dir.join("generations/.g000000000000000000002-staging");
    std::fs::create_dir_all(&orphan).unwrap();
    std::fs::write(orphan.join("vectors.bin"), b"partial").unwrap();

    let store = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(store.data_dir(), published);
    assert!(store.get(&PointId::from(7u64)).unwrap().is_some());
}

fn data_dir(root: &std::path::Path) -> std::path::PathBuf {
    super::Generation::open(root).unwrap().path()
}

/// A checkpointed collection restores its directory from the dense snapshot;
/// tampering with `keys.bin` must not change the ids the reopened store sees.
#[test]
fn test_dense_snapshot_restores_id_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let store_dir = tmp.path().join("col_snapshot");
    let store = CollectionStore::create(&store_dir, "col_snapshot", &config(4)).unwrap();
    let payload = [("color", serde_json::json!("red"))]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    upsert_logged(
        &store,
        &VectorPoint::new("1111", vec![1.0, 0.0, 0.0, 0.0]).with_payload(payload),
    );
    upsert_logged(&store, &VectorPoint::new("2222", vec![0.0, 1.0, 0.0, 0.0]));
    delete_logged(&store, &PointId::from("2222"));
    store.checkpoint().unwrap();

    let data = data_dir(&store_dir);
    let snapshot = DenseSnapshot::read(&data).unwrap().unwrap();
    assert_eq!(
        snapshot.generation,
        super::Generation::open(&store_dir).unwrap().id()
    );
    assert_eq!(snapshot.lsn, store.meta().last_applied_txn);
    assert_eq!(snapshot.points.len(), 2);
    assert_eq!(
        snapshot.points.iter().filter(|point| point.deleted).count(),
        1
    );

    let keys = data.join("keys.bin");
    let mut bytes = std::fs::read(&keys).unwrap();
    let match_at = bytes
        .windows(4)
        .position(|window| window == b"1111")
        .expect("live key is stored in keys.bin");
    bytes[match_at..match_at + 4].copy_from_slice(b"9999");
    std::fs::write(&keys, bytes).unwrap();
    drop(store);

    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), 1);
    let point = reopened.get(&PointId::from("1111")).unwrap().unwrap();
    assert_eq!(
        point.payload.unwrap().get("color"),
        Some(&serde_json::json!("red"))
    );
    assert!(reopened.get(&PointId::from("9999")).unwrap().is_none());
}

/// A snapshot ahead of the manifest belongs to a checkpoint that never
/// committed. Open must ignore it and rebuild from files plus WAL.
#[test]
fn test_unpublished_dense_snapshot_is_ignored() {
    let tmp = tempfile::tempdir().unwrap();
    let store_dir = tmp.path().join("col_snapshot_ahead");
    let store = CollectionStore::create(&store_dir, "col_snapshot_ahead", &config(4)).unwrap();
    upsert_logged(&store, &point(7, 4));
    store.checkpoint().unwrap();
    drop(store);

    let data = data_dir(&store_dir);
    let mut snapshot = DenseSnapshot::read(&data).unwrap().unwrap();
    snapshot.lsn += 1;
    DenseSnapshot::write(&data, &snapshot).unwrap();

    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), 1);
    assert!(reopened.get(&PointId::from(7u64)).unwrap().is_some());
}

/// Out-of-order segments are treated as stale per id: a resurrected old
/// upsert cannot overwrite a newer state, nor revive a deleted id.
#[test]
fn test_point_versions_drop_stale_replay_records() {
    let tmp = tempfile::tempdir().unwrap();
    let store_dir = tmp.path().join("col_versions");
    let store = CollectionStore::create(&store_dir, "col_versions", &config(4)).unwrap();
    let newer = VectorPoint::new(42, vec![5.0; 4]);
    let older = VectorPoint::new(42, vec![3.0; 4]);
    let resurrected = VectorPoint::new(43, vec![7.0; 4]);
    let wal = store.wal();
    wal.append(&WalTxn {
        txn_id: 5,
        ops: vec![WalRecord::Upsert {
            point: WalPoint::from_point(&newer).unwrap(),
        }],
    })
    .unwrap();
    wal.append(&WalTxn {
        txn_id: 9,
        ops: vec![WalRecord::Delete {
            point_id: "42".to_string(),
        }],
    })
    .unwrap();
    wal.append(&WalTxn {
        txn_id: 7,
        ops: vec![WalRecord::Upsert {
            point: WalPoint::from_point(&older).unwrap(),
        }],
    })
    .unwrap();
    wal.append(&WalTxn {
        txn_id: 8,
        ops: vec![WalRecord::Upsert {
            point: WalPoint::from_point(&resurrected).unwrap(),
        }],
    })
    .unwrap();
    drop(store);

    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert!(reopened.get(&PointId::from(42u64)).unwrap().is_none());
    let point = reopened.get(&PointId::from(43u64)).unwrap().unwrap();
    assert_eq!(point.vector(), &[7.0; 4]);
    assert_eq!(reopened.meta().last_applied_txn, 9);
}

/// Checkpoint advances the generation lsn with the data files, and compaction
/// publishes a new generation while keeping the WAL truncated.
#[test]
fn test_generation_advances_with_checkpoint_and_compaction() {
    let tmp = tempfile::tempdir().unwrap();
    let store_dir = tmp.path().join("col_gen2");
    let store = CollectionStore::create(&store_dir, "col_gen2", &config(4)).unwrap();
    for id in 1..=4 {
        upsert_logged(&store, &point(id, 4));
    }
    delete_logged(&store, &PointId::from(2u64));
    store.checkpoint().unwrap();
    let generation = super::Generation::open(&store_dir).unwrap();
    assert_eq!(generation.lsn(), store.meta().last_applied_txn);
    let before = store.data_dir();

    assert_eq!(store.compact().unwrap(), 3);
    let after = store.data_dir();
    assert_ne!(before, after);
    let generation = super::Generation::open(&store_dir).unwrap();
    assert_eq!(generation.lsn(), store.meta().last_applied_txn);
    let snapshot = DenseSnapshot::read(&after).unwrap().unwrap();
    assert_eq!(snapshot.generation, generation.id());
    assert_eq!(snapshot.lsn, generation.lsn());
    assert_eq!(snapshot.points.len(), 4);
    assert_eq!(snapshot.tombstones.len(), 0);
    assert!(store.get(&PointId::from(2u64)).unwrap().is_none());
    drop(store);

    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.data_dir(), after);
    assert_eq!(reopened.count(), 3);
}
