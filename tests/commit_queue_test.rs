//! Commit-queue tests through the public API: the ordering invariant (log
//! order ≡ apply order) and the slot bookkeeping under concurrent writers.
//!
//! The ordering invariant is what makes a reopened collection rebuild the
//! exact slot numbering the live store used. Replay walks the log in order
//! and hands out slots from `next_slot`; if the live store ever applied a
//! transaction in a different order, the same id would land on a different
//! slot after a restart and the directory files would disagree with the id
//! map. This test commits from several threads at once and compares the live
//! slots with the replayed ones point by point.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use simvec::storage::{CollectionStore, WalPoint, WalRecord};
use simvec::types::{CollectionConfig, DistanceMetric, PointId, VectorPoint};

const DIM: usize = 4;

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

fn upsert_op(point: &VectorPoint) -> WalRecord {
    WalRecord::Upsert {
        point: WalPoint::from_point(point).unwrap(),
    }
}

/// Concurrent writers, then a reopen: every id must keep the slot, the vector
/// and the payload it had before the "crash".
#[test]
fn concurrent_writes_reopen_with_identical_slot_numbering() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    const WRITERS: u64 = 4;
    const PER_WRITER: u64 = 40;

    let live: HashMap<PointId, (u32, VectorPoint)> = {
        let store = CollectionStore::create(&store_dir, "col", &config(DIM)).unwrap();
        let start = Arc::new(std::sync::Barrier::new(WRITERS as usize));
        let mut handles = Vec::new();
        for writer in 0..WRITERS {
            let store = Arc::clone(&store);
            let start = Arc::clone(&start);
            handles.push(thread::spawn(move || {
                // One transaction per point: the interleaving of the four
                // writers decides the log order, not the batching.
                for i in 0..PER_WRITER {
                    let id = writer * 1_000 + i;
                    start.wait();
                    let mut point = point(id, DIM);
                    point.payload = Some(HashMap::from([(
                        "writer".to_string(),
                        serde_json::json!(writer),
                    )]));
                    store.apply_ops(&[upsert_op(&point)]).unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        let mut live = HashMap::new();
        for writer in 0..WRITERS {
            for i in 0..PER_WRITER {
                let id = PointId::Num(writer * 1_000 + i);
                let slot = store.slot_of(&id).expect("slot assigned");
                let point = store.get(&id).unwrap().expect("point applied");
                live.insert(id, (slot, point));
            }
        }
        live
    };

    // Reopen without a checkpoint: everything must come back from the log.
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), live.len() as u64);
    for (id, (live_slot, point)) in &live {
        assert_eq!(
            reopened.slot_of(id),
            Some(*live_slot),
            "id {id} must replay onto the same slot"
        );
        let replayed = reopened
            .get(id)
            .unwrap()
            .unwrap_or_else(|| panic!("id {id} must come back from the log"));
        assert_eq!(&replayed.vector, &point.vector, "id {id} vector drifted");
        assert_eq!(&replayed.payload, &point.payload, "id {id} payload drifted");
    }
}

/// Upserts of many distinct new ids from concurrent writers must agree with
/// the slot accounting: no slot is handed out twice and none is skipped.
#[test]
fn concurrent_writers_never_share_a_slot() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(dir.path().join("col"), "col", &config(DIM)).unwrap();
    let store = Arc::new(store);
    const WRITERS: u64 = 4;
    const PER_WRITER: u64 = 25;

    let seen = Arc::new(Mutex::new(HashMap::<u32, u64>::new()));
    let counter = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::new();
    for _ in 0..WRITERS {
        let store = Arc::clone(&store);
        let seen = Arc::clone(&seen);
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..PER_WRITER {
                let id = counter.fetch_add(1, Ordering::Relaxed);
                store.apply_ops(&[upsert_op(&point(id, DIM))]).unwrap();
                let slot = store.slot_of(&PointId::Num(id)).expect("slot");
                let mut seen = seen.lock().unwrap();
                assert!(
                    seen.insert(slot, id).is_none(),
                    "slot {slot} handed out twice"
                );
            }
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }

    let total = (WRITERS * PER_WRITER) as usize;
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), total, "every id must own a distinct slot");
    let mut slots: Vec<u32> = seen.keys().copied().collect();
    slots.sort_unstable();
    assert_eq!(slots, (0..total as u32).collect::<Vec<_>>());
    assert_eq!(store.count(), total as u64);
}

/// A single writer still commits one transaction per fsync, so the group size
/// stays 1 when there is nothing to group.
#[test]
fn single_writer_flushes_one_transaction_at_a_time() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(dir.path().join("col"), "col", &config(DIM)).unwrap();
    for id in 0..3u64 {
        store.apply_ops(&[upsert_op(&point(id, DIM))]).unwrap();
    }
    let metrics = store.metrics().snapshot();
    assert_eq!(metrics.txns_applied, 3);
    assert_eq!(metrics.wal_flushes, 3);
    assert_eq!(metrics.wal_txns_flushed, 3);
    assert_eq!(metrics.apply_txn.count, 3);
    assert!(metrics.apply_queue_depth <= 3);
    assert!(metrics.apply_wait_nanos > 0);
}
