//! Storage integration tests: slot allocation, compaction, WAL-backed crash
//! recovery.

use std::collections::HashMap;

use sim_vec::storage::CollectionStore;
use sim_vec::storage::{Wal, WalRecord, WalTxn};
use sim_vec::types::{CollectionConfig, DistanceMetric, PointId, VectorPoint};

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

/// Batch upsert through the WAL-backed apply path: one WAL append and one
/// meta.bin save for the whole batch instead of per-point saves.
fn upsert_batch(store: &CollectionStore, ids: impl IntoIterator<Item = u64>, dim: usize) {
    let ops: Vec<WalRecord> = ids
        .into_iter()
        .map(|id| WalRecord::Upsert {
            point: sim_vec::storage::WalPoint::from_point(&point(id, dim)).unwrap(),
        })
        .collect();
    store.apply_ops(&ops).unwrap();
}

#[test]
fn test_apply_txn_roundtrip_and_replay() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store = CollectionStore::create(&store_dir, "col", &config(4)).unwrap();

    store
        .apply_txn(&WalTxn {
            txn_id: 1,
            ops: vec![
                WalRecord::Upsert {
                    point: sim_vec::storage::WalPoint::from_point(&point(1, 4)).unwrap(),
                },
                WalRecord::Upsert {
                    point: sim_vec::storage::WalPoint::from_point(&point(2, 4)).unwrap(),
                },
            ],
        })
        .unwrap();
    store
        .apply_txn(&WalTxn {
            txn_id: 2,
            ops: vec![WalRecord::Delete {
                point_id: "1".to_string(),
            }],
        })
        .unwrap();

    assert_eq!(store.count(), 1);
    assert_eq!(store.meta().last_applied_txn, 2);
    assert!(store.get(&PointId::Num(1)).unwrap().is_none());
    assert!(store.get(&PointId::Num(2)).unwrap().is_some());

    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), 1);
    assert!(reopened.get(&PointId::Num(1)).unwrap().is_none());
    assert_eq!(
        reopened.get(&PointId::Num(2)).unwrap().unwrap().vector,
        point(2, 4).vector
    );
    assert_eq!(reopened.meta().last_applied_txn, 2);
}

#[test]
fn test_wal_written_memory_not_applied_recovers() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store = CollectionStore::create(&store_dir, "col", &config(4)).unwrap();
    store.upsert(&point(1, 4)).unwrap();
    drop(store);

    // Simulate a crash between "WAL fsync" and "apply to memory": append
    // transactions directly to wal.bin, bypassing the in-memory apply.
    let wal = Wal::open_or_create(&store_dir.join("wal.bin")).unwrap();
    wal.append(&WalTxn {
        txn_id: 1,
        ops: vec![WalRecord::Upsert {
            point: sim_vec::storage::WalPoint::from_point(&point(2, 4)).unwrap(),
        }],
    })
    .unwrap();
    wal.append(&WalTxn {
        txn_id: 2,
        ops: vec![WalRecord::Delete {
            point_id: "1".to_string(),
        }],
    })
    .unwrap();
    drop(wal);

    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), 1, "replay must apply the pending WAL");
    assert!(reopened.get(&PointId::Num(1)).unwrap().is_none());
    assert!(reopened.get(&PointId::Num(2)).unwrap().is_some());
}

#[test]
fn test_replay_is_idempotent_for_duplicate_txns() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store = CollectionStore::create(&store_dir, "col", &config(4)).unwrap();

    let txn = WalTxn {
        txn_id: 1,
        ops: vec![WalRecord::Upsert {
            point: sim_vec::storage::WalPoint::from_point(&point(1, 4)).unwrap(),
        }],
    };
    // Same txn applied twice (crash + coordinator retry semantics).
    store.apply_txn(&txn).unwrap();
    store.apply_txn(&txn).unwrap();
    assert_eq!(store.count(), 1, "re-applying an upsert must not duplicate");

    let del = WalTxn {
        txn_id: 2,
        ops: vec![WalRecord::Delete {
            point_id: "1".to_string(),
        }],
    };
    store.apply_txn(&del).unwrap();
    store.apply_txn(&del).unwrap();
    assert_eq!(store.count(), 0, "re-applying a delete is a no-op");
}

#[test]
fn test_apply_txn_rejects_invalid_before_wal_append() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store = CollectionStore::create(&store_dir, "col", &config(4)).unwrap();

    let bad = VectorPoint::new(1u64, vec![1.0, 2.0]); // wrong dim
    let err = store
        .apply_txn(&WalTxn {
            txn_id: 1,
            ops: vec![WalRecord::Upsert {
                point: sim_vec::storage::WalPoint::from_point(&bad).unwrap(),
            }],
        })
        .unwrap_err();
    assert!(matches!(
        err,
        sim_vec::VectorSearchError::InvalidVectorDimension {
            expected: 4,
            actual: 2
        }
    ));
    // Nothing was logged or applied.
    assert_eq!(store.count(), 0);
    let wal = Wal::open_or_create(&store_dir.join("wal.bin")).unwrap();
    let last = wal.replay(|_| Ok(())).unwrap();
    assert_eq!(last, 0);
}

#[test]
fn test_compact_rebuilds_and_search_stays_correct() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store = CollectionStore::create(&store_dir, "col", &config(4)).unwrap();

    upsert_batch(&store, 0..20, 4);
    // Delete some points (below the 20% threshold: 4/20 = 20%, threshold is
    // strictly greater, so no auto-compaction).
    for i in [3u64, 7, 11, 15] {
        assert!(store.delete(&PointId::Num(i)).unwrap());
    }
    assert_eq!(store.count(), 16);
    let meta = store.meta();
    assert_eq!(meta.tombstone_count, 4);

    let live = store.compact().unwrap();
    assert_eq!(live, 16);
    let meta = store.meta();
    assert_eq!(meta.next_slot, 16);
    assert_eq!(meta.tombstone_count, 0);

    // All surviving points are readable with correct vectors.
    for i in 0..20u64 {
        let got = store.get(&PointId::Num(i)).unwrap();
        if [3, 7, 11, 15].contains(&i) {
            assert!(got.is_none(), "deleted point {i} must stay gone");
        } else {
            assert_eq!(got.unwrap().vector, point(i, 4).vector, "point {i}");
        }
    }

    // Reopen: compacted layout must be consistent on disk.
    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), 16);
    for i in 0..20u64 {
        let got = reopened.get(&PointId::Num(i)).unwrap();
        if [3, 7, 11, 15].contains(&i) {
            assert!(got.is_none());
        } else {
            assert_eq!(got.unwrap().vector, point(i, 4).vector);
        }
    }
}

#[test]
fn test_compact_with_payloads_and_no_auto_trigger() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store = CollectionStore::create(&store_dir, "col", &config(2)).unwrap();

    let mut payload = HashMap::new();
    payload.insert("color".to_string(), serde_json::json!("red"));
    let mut p = point(1, 2);
    p.payload = Some(payload.clone());
    store.upsert(&p).unwrap();
    let mut p2 = point(2, 2);
    let mut payload2 = HashMap::new();
    payload2.insert("color".to_string(), serde_json::json!("blue"));
    p2.payload = Some(payload2.clone());
    store.upsert(&p2).unwrap();

    assert!(store.delete(&PointId::Num(1)).unwrap());
    // The tombstone is visible immediately and reclaimed by an explicit
    // compaction (the engine schedules the same work in the background).
    let meta = store.meta();
    assert_eq!(meta.tombstone_count, 1);
    store.compact().unwrap();
    assert_eq!(store.meta().tombstone_count, 0);

    let got = store.get(&PointId::Num(2)).unwrap().unwrap();
    assert_eq!(got.payload, Some(payload2.clone()));

    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    let got = reopened.get(&PointId::Num(2)).unwrap().unwrap();
    assert_eq!(got.payload, Some(payload2));
    assert!(reopened.get(&PointId::Num(1)).unwrap().is_none());
}

#[test]
fn test_compact_checkpoint_truncates_wal() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store = CollectionStore::create(&store_dir, "col", &config(4)).unwrap();
    upsert_batch(&store, 0..6, 4);
    for i in 0..2u64 {
        assert!(store.delete(&PointId::Num(i)).unwrap());
    }
    store.compact().unwrap();
    // A compaction checkpoint truncates the WAL.
    let wal = Wal::open_or_create(&store_dir.join("wal.bin")).unwrap();
    let last = wal.replay(|_| Ok(())).unwrap();
    assert_eq!(last, 0, "WAL must be empty after compaction");
    assert_eq!(store.count(), 4);
}

#[test]
fn test_grow_after_compact() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store = CollectionStore::create(&store_dir, "col", &config(4)).unwrap();
    upsert_batch(&store, 0..40, 4);
    for i in 0..20u64 {
        assert!(store.delete(&PointId::Num(i)).unwrap());
    }
    assert_eq!(store.count(), 20);

    // Continue inserting well past the compacted capacity.
    upsert_batch(&store, 40..120, 4);
    assert_eq!(store.count(), 100);
    // A couple of individual upserts keep single-write growth covered.
    store.upsert(&point(200, 4)).unwrap();
    assert_eq!(store.count(), 101);
    for i in 40..120u64 {
        assert_eq!(
            store.get(&PointId::Num(i)).unwrap().unwrap().vector,
            point(i, 4).vector,
            "point {i}"
        );
    }
}

#[test]
fn compaction_under_concurrent_writes_stays_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store =
        std::sync::Arc::new(CollectionStore::create(&store_dir, "col", &config(4)).unwrap());

    // Writer: each round appends 8 fresh ids then deletes the first 4 of the
    // same round (one WAL-backed batch per phase), so tombstones keep
    // triggering compactions while appends race them. Final live set is
    // deterministic: ids with `id % 8 >= 4`.
    const ROUNDS: u64 = 15;
    let writer = {
        let store = std::sync::Arc::clone(&store);
        std::thread::spawn(move || {
            for r in 0..ROUNDS {
                upsert_batch(&store, r * 8..r * 8 + 8, 4);
                for j in 0..4u64 {
                    assert!(store.delete(&PointId::Num(r * 8 + j)).unwrap());
                }
            }
        })
    };

    // Concurrent compactions: both the lock-free retry path and the
    // contended fallback must preserve visibility and data integrity.
    for _ in 0..4 {
        std::thread::sleep(std::time::Duration::from_millis(5));
        store.compact().unwrap();
    }
    writer.join().unwrap();

    let expected: Vec<u64> = (0..ROUNDS * 8).filter(|id| id % 8 >= 4).collect();
    assert_eq!(store.count(), expected.len() as u64);
    for id in &expected {
        assert!(
            store.get(&PointId::Num(*id)).unwrap().is_some(),
            "live point {id} lost"
        );
        assert_eq!(
            store.get(&PointId::Num(*id)).unwrap().unwrap().vector,
            point(*id, 4).vector,
            "point {id} corrupted by a racing compaction"
        );
    }
    for r in 0..ROUNDS {
        assert!(store.get(&PointId::Num(r * 8)).unwrap().is_none());
    }

    // One final compaction on the quiesced store must be a clean no-op-ish
    // pass that keeps every remaining point intact.
    let live = store.compact().unwrap();
    assert_eq!(live, expected.len() as u64);
    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), expected.len() as u64);
}
