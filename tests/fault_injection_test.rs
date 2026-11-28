//! Crash-window matrix: an armed `SIMVEC_FAULT` point must abort the process
//! and the collection must reopen consistently afterwards.
//!
//! Every window is exercised in a child process (this test binary re-invoked
//! with `--exact crash_child --ignored`), because the point aborts. The parent
//! only asserts on the reopened state, which is exactly what a real crash
//! leaves behind:
//!
//! | point | window | expected after reopen |
//! | --- | --- | --- |
//! | `wal_after_append` | log durable, not applied | every record replayed |
//! | `apply_before_ack` | applied, caller still waiting | every record replayed once |
//! | `checkpoint_after_flush` | files fsynced, tracker not published | counts unchanged |
//! | `checkpoint_after_tracker_publish` | tracker visible, manifest behind | counts unchanged |
//! | `wal_before_truncate` | checkpoint ready to cut the log | counts unchanged |
//! | `compaction_after_publish` | new generation published, not switched | new layout loads |
//!
//! The point names are the contract of `src/storage/fault.rs`; the table there
//! lists the same strings.

use std::path::Path;
use std::process::Command;

use simvec::storage::{CollectionStore, WalPoint, WalRecord};
use simvec::types::{CollectionConfig, DistanceMetric, PointId, VectorPoint};

const CHILD_DIR: &str = "SIMVEC_FAULT_TEST_DIR";
const CHILD_POINT: &str = "SIMVEC_FAULT_TEST_POINT";
const CHILD_TEST: &str = "crash_child";

fn config() -> CollectionConfig {
    CollectionConfig::new(4, DistanceMetric::Cosine)
}

fn point(id: u64) -> VectorPoint {
    VectorPoint::new(id, vec![id as f32, 1.0, 0.0, 0.0])
}

fn upsert(id: u64) -> WalRecord {
    WalRecord::Upsert {
        point: WalPoint::from_point(&point(id)).unwrap(),
    }
}

/// Child entry point: run the workload for the armed point, which aborts the
/// moment the write path reaches it.
///
/// It is `#[ignore]` so a plain `cargo test` never runs it in this process —
/// reaching the end without a crash means the parent's expectation failed.
#[test]
#[ignore = "spawned by the parent test as a crash child"]
fn crash_child() {
    let (Ok(dir), Ok(name)) = (std::env::var(CHILD_DIR), std::env::var(CHILD_POINT)) else {
        // Not the child of a crash test: nothing to do.
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let store = CollectionStore::create(&dir, "col", &config()).unwrap();
    match name.as_str() {
        "wal_after_append" | "apply_before_ack" => {
            store.apply_txn(&[upsert(1)]).unwrap();
        }
        "checkpoint_after_flush" | "checkpoint_after_tracker_publish" | "wal_before_truncate" => {
            for id in 1..=3 {
                store.apply_txn(&[upsert(id)]).unwrap();
            }
            store.checkpoint().unwrap();
        }
        "compaction_after_publish" => {
            for id in 1..=5 {
                store.apply_txn(&[upsert(id)]).unwrap();
            }
            for id in [2u64, 4] {
                store
                    .apply_txn(&[WalRecord::Delete {
                        point_id: PointId::from(id).to_string(),
                    }])
                    .unwrap();
            }
            store.compact().unwrap();
        }
        other => panic!("unknown crash window {other}"),
    }
    // A clean return means the point never fired, which the parent treats as a
    // failure — but only after the process is gone, so fail loudly here too.
    panic!("crash window {name} was never reached");
}

/// Run the child with `point` armed and require it to die from `SIGABRT`.
fn crash_at(point: &str, dir: &Path) {
    let exe = std::env::current_exe().expect("test binary path");
    let output = Command::new(exe)
        .args(["--exact", CHILD_TEST, "--ignored", "--nocapture"])
        .env(CHILD_DIR, dir)
        .env(CHILD_POINT, point)
        .env("SIMVEC_FAULT", point)
        .output()
        .expect("spawn crash child");
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(
            output.status.signal(),
            Some(6),
            "child for {point} must abort: status={:?} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    #[cfg(not(unix))]
    assert!(
        !output.status.success(),
        "child for {point} must not exit cleanly"
    );
}

fn assert_live_points(store: &CollectionStore, live: &[u64], dead: &[u64]) {
    assert_eq!(
        store.count(),
        live.len() as u64,
        "live count after the crash window"
    );
    for id in live {
        let got = store
            .get(&PointId::from(*id))
            .unwrap()
            .unwrap_or_else(|| panic!("point {id} lost"));
        assert_eq!(got.vector(), point(*id).vector(), "point {id} drifted");
    }
    for id in dead {
        assert!(
            store.get(&PointId::from(*id)).unwrap().is_none(),
            "point {id} must stay deleted"
        );
    }
}

#[test]
fn a_record_durable_but_not_applied_is_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    crash_at("wal_after_append", &store_dir);
    let store = CollectionStore::open(&store_dir).unwrap();
    assert_live_points(&store, &[1], &[]);
    // Replaying twice must not double count.
    drop(store);
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert_live_points(&reopened, &[1], &[]);
}

#[test]
fn an_applied_but_unacknowledged_record_survives() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    crash_at("apply_before_ack", &store_dir);
    let store = CollectionStore::open(&store_dir).unwrap();
    assert_live_points(&store, &[1], &[]);
}

#[test]
fn a_checkpoint_crash_before_the_tracker_publish_replays_the_log() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    crash_at("checkpoint_after_flush", &store_dir);
    let store = CollectionStore::open(&store_dir).unwrap();
    assert_live_points(&store, &[1, 2, 3], &[]);
}

#[test]
fn a_tracker_published_ahead_of_the_manifest_is_tolerated() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    crash_at("checkpoint_after_tracker_publish", &store_dir);
    let store = CollectionStore::open(&store_dir).unwrap();
    assert_live_points(&store, &[1, 2, 3], &[]);
}

#[test]
fn a_crash_inside_the_truncate_keeps_the_collection_readable() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    crash_at("wal_before_truncate", &store_dir);
    let store = CollectionStore::open(&store_dir).unwrap();
    assert_live_points(&store, &[1, 2, 3], &[]);
}

#[test]
fn a_published_generation_loads_after_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");
    crash_at("compaction_after_publish", &store_dir);
    let store = CollectionStore::open(&store_dir).unwrap();
    assert_live_points(&store, &[1, 3, 5], &[2, 4]);
    // The published generation is the compacted one, not the one the child
    // was still serving.
    assert_eq!(store.meta().tombstone_count, 0);
}
