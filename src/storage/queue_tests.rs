use super::*;
use crate::storage::WalPoint;
use crate::types::{CollectionConfig, DistanceMetric, PointId, VectorPoint};

const DIM: usize = 4;

fn config() -> CollectionConfig {
    CollectionConfig::new(DIM, DistanceMetric::Cosine)
}

fn point(id: u64) -> VectorPoint {
    VectorPoint::new(
        id,
        (0..DIM)
            .map(|i| ((id as usize * 31 + i) % 100) as f32 / 100.0)
            .collect(),
    )
}

fn upsert_op(point: &VectorPoint) -> WalRecord {
    WalRecord::Upsert {
        point: WalPoint::from_point(point).unwrap(),
    }
}

fn store(dir: &std::path::Path) -> Arc<CollectionStore> {
    CollectionStore::create(dir, "col", &config()).unwrap()
}

/// Seed one transaction so the queue worker exists. The worker is created
/// lazily by the first transaction, and the hold can only park a worker
/// that already runs.
fn spawn_worker(store: &CollectionStore) {
    store.apply_ops(&[upsert_op(&point(9_999))]).unwrap();
}

#[test]
fn waiter_times_out_instead_of_hanging() {
    let completion = Arc::new(Completion::new());
    let err = completion
        .wait(Duration::from_millis(50))
        .expect_err("an unsettled ticket must time out");
    assert!(matches!(err, VectorSearchError::Internal(_)));
    // Settling afterwards still returns the real outcome.
    completion.settle(Ok(()));
    assert!(completion.wait(Duration::from_millis(50)).is_ok());
}

#[test]
fn wait_returns_the_settled_outcome_immediately() {
    let completion = Arc::new(Completion::new());
    completion.settle(Err(VectorSearchError::Internal("boom".to_string())));
    let err = completion
        .wait(Duration::from_secs(30))
        .expect_err("settled failure is reported");
    assert!(err.to_string().contains("boom"));
}

#[test]
fn poison_fails_waiters_before_the_timeout() {
    let queue = Arc::new(CommitQueue::new());
    let completion = Arc::new(Completion::new());
    {
        let mut state = queue.state.lock();
        state.waiters.push(Arc::clone(&completion));
        state.in_flight = 1;
    }
    queue.poison("injected");
    let err = completion
        .wait(Duration::from_secs(30))
        .expect_err("poisoned queue fails its waiters");
    assert!(err.to_string().contains("injected"));
    assert!(queue.in_flight() == 0);
}

/// Two maintenance critical sections can overlap — an explicit
/// `compact_collection` call racing the maintenance worker's checkpoint
/// for the same collection. The first one finishing must not open the
/// gate on the second: writers entering there would append into a log the
/// surviving section is about to truncate.
#[test]
fn overlapping_maintenance_sections_keep_the_gate_closed() {
    let queue = Arc::new(CommitQueue::new());
    let first = queue.quiesce();
    assert!(queue.quiescing(), "the first section closed the gate");
    let second = queue.quiesce();

    drop(first);
    assert!(
        queue.quiescing(),
        "the surviving section still owns the gate"
    );

    drop(second);
    assert!(!queue.quiescing(), "the gate reopens with the last section");
    // Writers flow again once every section is gone.
    let permit = queue.begin_write();
    drop(permit);
}

/// Real threads, no parking: the guard against a regression that moves the
/// WAL append back inside the store write lock.
///
/// The parked test above proves the flush path batches a backlog, but says
/// nothing about whether writers can pile up at all — parking the worker
/// fills the queue regardless of what the write path looks like. If
/// enqueueing were serialized again, the backlog could never exceed one
/// transaction and the observed group size would collapse to 1, which is
/// what this test detects.
#[test]
fn real_writers_group_a_flush_without_being_parked() {
    let dir = tempfile::tempdir().unwrap();
    let store = store(&dir.path().join("col"));
    let before = store.metrics().snapshot();

    const WRITERS: usize = 8;
    const PER_WRITER: u64 = 15;
    let start = Arc::new(std::sync::Barrier::new(WRITERS));
    let mut handles = Vec::new();
    for writer in 0..WRITERS as u64 {
        let store = Arc::clone(&store);
        let start = Arc::clone(&start);
        handles.push(std::thread::spawn(move || {
            start.wait();
            for i in 0..PER_WRITER {
                let id = writer * 1_000 + i;
                store.apply_ops(&[upsert_op(&point(id))]).unwrap();
            }
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }

    let metrics = store.metrics().snapshot();
    let txns = metrics.wal_txns_flushed - before.wal_txns_flushed;
    let flushes = metrics.wal_flushes - before.wal_flushes;
    assert_eq!(txns, WRITERS as u64 * PER_WRITER, "every txn was logged");
    assert_eq!(store.count(), txns);
    assert!(
        flushes < txns,
        "concurrent writers must share fsyncs, saw {flushes} flushes for {txns} txns"
    );
}

/// Group commit is the point of the queue: with the backlog filled before
/// the worker wakes, the whole backlog must share one WAL fsync.
#[test]
fn concurrent_transactions_share_one_group_commit() {
    let dir = tempfile::tempdir().unwrap();
    let store = store(&dir.path().join("col"));
    spawn_worker(&store);
    let before = store.metrics().snapshot();

    // Park the worker before it can take a batch, so the writers below
    // fill the queue first.
    store.queue.hold_flush();
    while !store.queue.worker_parked() {
        std::thread::yield_now();
    }

    const TXNS: u64 = 6;
    let mut handles = Vec::new();
    for txn in 0..TXNS {
        let store = Arc::clone(&store);
        handles.push(std::thread::spawn(move || {
            store.apply_ops(&[upsert_op(&point(txn))]).unwrap();
        }));
    }
    while store.queue.enqueued_total() < TXNS + 1 {
        std::thread::yield_now();
    }

    store.queue.release_flush();
    for handle in handles {
        handle.join().unwrap();
    }

    let metrics = store.metrics().snapshot();
    assert_eq!(
        metrics.wal_flushes - before.wal_flushes,
        1,
        "one backlog must flush as one batch"
    );
    assert_eq!(metrics.wal_txns_flushed - before.wal_txns_flushed, TXNS);
    assert!(metrics.apply_wait_nanos > before.apply_wait_nanos);
    assert_eq!(store.count(), TXNS + 1);
}

/// A checkpoint that arrives while transactions are still queued must wait
/// for them, then truncate the log without losing any of them.
#[test]
fn checkpoint_with_a_non_empty_queue_keeps_every_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store = store(&store_dir);
    spawn_worker(&store);

    store.queue.hold_flush();
    while !store.queue.worker_parked() {
        std::thread::yield_now();
    }

    const TXNS: u64 = 4;
    let mut handles = Vec::new();
    for txn in 0..TXNS {
        let store = Arc::clone(&store);
        handles.push(std::thread::spawn(move || {
            store.apply_ops(&[upsert_op(&point(txn))]).unwrap();
        }));
    }
    while store.queue.enqueued_total() < TXNS + 1 {
        std::thread::yield_now();
    }

    // The checkpoint blocks on the queue; only the release lets it run.
    let checker = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || store.checkpoint().unwrap())
    };
    while !store.queue.quiescing() {
        std::thread::yield_now();
    }
    store.queue.release_flush();
    let live = checker.join().unwrap();
    for handle in handles {
        handle.join().unwrap();
    }
    assert_eq!(live, TXNS + 1, "checkpoint must keep every live point");

    // The log was truncated after every queued transaction was applied, so
    // reopening must still see all of them.
    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), TXNS + 1);
    for txn in 0..TXNS {
        assert!(
            reopened.get(&PointId::Num(txn)).unwrap().is_some(),
            "txn {txn} must survive the checkpoint"
        );
    }
}

/// Same hazard for compaction, which also rewrites every file and
/// truncates the log.
#[test]
fn compaction_with_a_non_empty_queue_keeps_every_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store = store(&store_dir);

    for id in 0..12u64 {
        store.apply_ops(&[upsert_op(&point(id))]).unwrap();
    }
    for id in 0..6u64 {
        store
            .apply_ops(&[WalRecord::Delete {
                point_id: id.to_string(),
            }])
            .unwrap();
    }
    assert!(store.needs_compaction());

    store.queue.hold_flush();
    while !store.queue.worker_parked() {
        std::thread::yield_now();
    }

    const TXNS: u64 = 3;
    let mut handles = Vec::new();
    for txn in 0..TXNS {
        let store = Arc::clone(&store);
        handles.push(std::thread::spawn(move || {
            store.apply_ops(&[upsert_op(&point(100 + txn))]).unwrap();
        }));
    }
    while store.queue.enqueued_total() < TXNS {
        std::thread::yield_now();
    }

    let compactor = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || store.compact().unwrap())
    };
    while !store.queue.quiescing() {
        std::thread::yield_now();
    }
    store.queue.release_flush();
    let live = compactor.join().expect("compaction must not fail");
    for handle in handles {
        handle.join().unwrap();
    }
    assert_eq!(live, 6 + TXNS, "compaction must keep every live point");

    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), 6 + TXNS);
    for id in 0..6u64 {
        assert!(reopened.get(&PointId::Num(id)).unwrap().is_none());
    }
    for id in 6..12u64 {
        assert!(reopened.get(&PointId::Num(id)).unwrap().is_some());
    }
    for txn in 0..TXNS {
        assert!(
            reopened.get(&PointId::Num(100 + txn)).unwrap().is_some(),
            "txn {txn} must survive the compaction"
        );
    }
}

/// A batch whose WAL append fails must fail every transaction in it,
/// apply none of it, and leave the collection open and writable.
#[test]
fn failed_wal_append_applies_nothing_and_keeps_the_collection_usable() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store = store(&store_dir);
    store
        .apply_ops(&[upsert_op(&point(1)), upsert_op(&point(2))])
        .unwrap();
    assert_eq!(store.count(), 2);

    store.wal().fail_next_append_for_test();
    let err = store
        .apply_ops(&[upsert_op(&point(3))])
        .expect_err("a failed group commit must surface to the writer");
    assert!(
        err.to_string().contains("test-injected"),
        "unexpected error: {err}"
    );
    assert_eq!(store.count(), 2, "an un-durable batch must not be applied");
    assert!(store.get(&PointId::Num(3)).unwrap().is_none());

    // A failed flush is transient: the next transaction must still work,
    // and the collection must reopen with only the durable data.
    store
        .apply_ops(&[upsert_op(&point(4))])
        .expect("the queue survives a failed flush");
    assert_eq!(store.count(), 3);

    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), 3);
    assert!(reopened.get(&PointId::Num(3)).unwrap().is_none());
    assert!(reopened.get(&PointId::Num(4)).unwrap().is_some());
}

/// A worker that panics must fail its waiter immediately instead of
/// leaving it blocked, must refuse later writes, and must not disturb
/// reads.
#[test]
fn panicking_worker_fails_waiters_and_refuses_later_writes() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    let store = store(&store_dir);
    store.apply_ops(&[upsert_op(&point(1))]).unwrap();

    store.queue.panic_before_next_flush();
    let err = store
        .apply_ops(&[upsert_op(&point(2))])
        .expect_err("a dead applier must fail its waiters, not hang them");
    assert!(err.to_string().contains("panicked"), "unexpected: {err}");

    // Reads keep working on whatever was applied.
    assert_eq!(store.count(), 1);
    assert!(store.get(&PointId::Num(1)).unwrap().is_some());

    // The queue is poisoned: later writes fail fast instead of queueing
    // behind an applier that no longer exists.
    let err = store
        .apply_ops(&[upsert_op(&point(3))])
        .expect_err("a poisoned queue must refuse writes");
    assert!(
        err.to_string().contains("commit queue failed"),
        "unexpected: {err}"
    );

    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(reopened.count(), 1);
    assert!(reopened.get(&PointId::Num(1)).unwrap().is_some());
}
