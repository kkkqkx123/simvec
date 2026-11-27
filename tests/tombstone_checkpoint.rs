//! Regression: a logical delete must survive a WAL checkpoint.
//!
//! Tombstones used to live only in memory: `checkpoint()` persisted `meta.bin`
//! and truncated the WAL while `open()` rebuilt the bitmap from zeros, so a
//! deleted point came back after the next restart (and `count()` disagreed
//! with `scroll()`). These tests pin the durable contract: the table is
//! written to `tombstones.bin` before the log is cleared and read back on open.

use simvec::storage::{CollectionStore, WalPoint, WalRecord};
use simvec::types::{CollectionConfig, DistanceMetric, PointId, VectorPoint};

fn point(id: u64) -> VectorPoint {
    VectorPoint::new(PointId::Num(id), vec![1.0, 0.0, 0.0])
}

fn upsert(store: &CollectionStore, id: u64) {
    store
        .apply_ops(&[WalRecord::Upsert {
            point: WalPoint::from_point(&point(id)).unwrap(),
        }])
        .expect("upsert");
}

fn delete(store: &CollectionStore, id: u64) {
    store
        .apply_ops(&[WalRecord::Delete {
            point_id: PointId::Num(id).to_string(),
        }])
        .expect("delete");
}

fn new_store(dir: &std::path::Path) -> std::sync::Arc<CollectionStore> {
    let cfg = CollectionConfig::new(3, DistanceMetric::Dot);
    CollectionStore::create(dir.join("probe"), "probe", &cfg).expect("create")
}

#[test]
fn deleted_point_stays_deleted_after_checkpoint_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let collection_dir = dir.path().join("probe");
    let data_dir;
    {
        let store = new_store(dir.path());
        for id in 1..=3 {
            upsert(&store, id);
        }
        delete(&store, 2);
        assert_eq!(store.count(), 2, "live count right after delete");
        store.checkpoint().expect("checkpoint");
        data_dir = store.data_dir();
        assert!(data_dir.join("tombstones.bin").exists());
        drop(store);
    }

    let store = CollectionStore::open(&collection_dir).expect("reopen");
    assert_eq!(store.data_dir(), data_dir);
    assert!(
        store.get(&PointId::Num(2)).expect("get").is_none(),
        "deleted point came back after checkpoint + reopen"
    );
    assert!(store.get(&PointId::Num(1)).expect("get").is_some());
    assert_eq!(store.count(), 2);
    assert_eq!(store.meta().live_count, 2);
    assert_eq!(store.meta().tombstone_count, 1);

    let (points, _next) = store
        .scroll(100, None, None, None)
        .expect("scroll after reopen");
    assert_eq!(points.len(), 2, "scroll must agree with count()");
    assert!(points.iter().all(|p| p.id != PointId::Num(2)));
}

#[test]
fn tombstone_cycle_repeats_across_restarts() {
    let dir = tempfile::tempdir().unwrap();
    let collection_dir = dir.path().join("probe");
    {
        let store = new_store(dir.path());
        for id in 1..=4 {
            upsert(&store, id);
        }
        delete(&store, 1);
        store.checkpoint().expect("checkpoint");
    }
    {
        let store = CollectionStore::open(&collection_dir).expect("reopen #1");
        assert!(store.get(&PointId::Num(1)).expect("get").is_none());
        delete(&store, 3);
        store.checkpoint().expect("second checkpoint");
    }
    let store = CollectionStore::open(&collection_dir).expect("reopen #2");
    assert!(
        store.get(&PointId::Num(1)).expect("get").is_none(),
        "first deletion lost by the second checkpoint"
    );
    assert!(
        store.get(&PointId::Num(3)).expect("get").is_none(),
        "second deletion lost"
    );
    assert_eq!(store.count(), 2);
    assert_eq!(store.meta().tombstone_count, 2);
}

#[test]
fn payload_indexes_skip_tombstoned_slots_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let collection_dir = dir.path().join("probe");
    {
        let store = new_store(dir.path());
        for id in 1..=3 {
            upsert(&store, id);
        }
        delete(&store, 2);
        store.checkpoint().expect("checkpoint");
    }
    let store = CollectionStore::open(&collection_dir).expect("reopen");
    // Reserved id 2 must not reappear as a live slot after the rebuild pass.
    let (points, _) = store.scroll(100, None, None, None).expect("scroll");
    assert_eq!(points.len(), 2);
    assert_eq!(store.count(), 2);
}

#[test]
fn reinserting_a_deleted_id_after_restart_becomes_visible() {
    let dir = tempfile::tempdir().unwrap();
    let collection_dir = dir.path().join("probe");
    {
        let store = new_store(dir.path());
        for id in 1..=3 {
            upsert(&store, id);
        }
        delete(&store, 2);
        store.checkpoint().expect("checkpoint");
    }
    let store = CollectionStore::open(&collection_dir).expect("reopen");
    assert!(store.get(&PointId::Num(2)).expect("get").is_none());
    upsert(&store, 2);
    assert!(
        store.get(&PointId::Num(2)).expect("get").is_some(),
        "a re-inserted id must not land in its old tombstoned slot"
    );
    assert_eq!(store.count(), 3);
    let (points, _) = store.scroll(100, None, None, None).expect("scroll");
    assert_eq!(points.len(), 3);
}
