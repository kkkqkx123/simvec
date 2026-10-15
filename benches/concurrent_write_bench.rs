//! Concurrent-writer scaling report for the commit queue.
//!
//! Report-only, one pass per writer count, no criterion and no assertions —
//! same shape as `alloc_stats_bench`, because the object of study is the
//! shape of the numbers rather than a regression baseline. Concurrency in the
//! mutation path comes from the commit queue: one writer queues a
//! transaction, the queue worker flushes the whole backlog with a single WAL
//! fsync, and applies it in log order. What this bench reports is whether
//! that grouping actually happens, and how much of the single-writer cost it
//! can overlap away.
//!
//! Three observables per configuration:
//!
//! - `txn/flush` and `group` — the group size. `wal_txns_flushed /
//!   wal_flushes` above 1 means concurrent writers really shared one fsync;
//!   a single writer always reads 1.00 because there is nothing to group.
//! - `wait/txn` — `apply_wait_nanos` per transaction: how long a writer
//!   blocked between enqueueing and its transaction being applied.
//! - `apply/txn` — the apply critical section alone, which is the part the
//!   single applier serializes. Compare it against the wall time per
//!   transaction; the difference is the fsync that concurrency can hide.
//!
//! The ceiling follows from that last comparison:
//! `speedup <= (fsync + apply) / apply`. A workload whose per-transaction
//! apply cost dwarfs the fsync (payload-heavy writes, or a background index
//! build eating the cores) approaches 1.0 no matter how the queue is shaped;
//! that is the documented floor of a single-mmap store, not a queue defect.
//!
//! fsync census: run this bench under strace and divide by the transaction
//! count. strace inflates wall times by roughly an order of magnitude on
//! multi-threaded runs, so only the census is meaningful.
//!
//! ```text
//! strace -f -c -e trace=fsync,fdatasync \
//!     cargo bench -p simvec --bench concurrent_write_bench
//! ```
//!
//! Run: `cargo bench -p simvec --bench concurrent_write_bench`; `QUICK=1`
//! shrinks the transaction count for a seconds-long smoke run.
//!
//! NOTE: the scratch index lives under `../target/bench-vector` because the
//! default system temp dir is tmpfs on this box. An fsync against tmpfs costs
//! nothing, which silently turns every number below into apply-only and makes
//! the fsync share unmeasurable.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use simvec::storage::{CollectionStore, WalPoint, WalRecord};
use simvec::{CollectionConfig, DistanceMetric, LocalVectorEngine, Payload, VectorPoint};

const DIM: usize = 128;

/// Points per transaction: the microbatch the wrapper layer and the
/// orchestrator use, so the fsync share matches the production shape.
const BATCH: usize = 1000;

/// Full-gear transaction count per writer configuration. `QUICK=1` shrinks it
/// for a seconds-long smoke run; the gear is read directly rather than through
/// `quick.rs` because this is a report-only binary and that module's
/// `benchmark_group` is criterion-only.
const FULL_TXNS: usize = 10;
const QUICK_TXNS: usize = 4;

fn txns() -> usize {
    if std::env::var("QUICK").is_ok() {
        QUICK_TXNS
    } else {
        FULL_TXNS
    }
}

/// Writer counts swept per run. Fewer than the available transactions only
/// adds scheduler noise, so the small gear drops the idle ones.
fn writer_counts() -> Vec<usize> {
    [1, 2, 4, 8].into_iter().filter(|w| *w <= txns()).collect()
}

fn point(id: usize) -> VectorPoint {
    VectorPoint::new(
        format!("pt_{id:06}"),
        (0..DIM)
            .map(|d| ((id as f32 + d as f32) * 0.001).sin())
            .collect(),
    )
}

/// Same point carrying a payload, which is the wrapper layer's shape.
/// Payloads add PayloadStore writes per point and so grow the apply half of
/// the ratio above.
fn point_with_payload(id: usize) -> VectorPoint {
    point(id).with_payload(Payload::from([
        (
            "src".to_string(),
            serde_json::json!(format!("mod_{id:06}.rs")),
        ),
        ("group".to_string(), serde_json::json!("proj-1")),
        ("epoch".to_string(), serde_json::json!(1)),
    ]))
}

fn upsert_ops(start: usize) -> Vec<WalRecord> {
    (start..start + BATCH)
        .map(|i| WalRecord::Upsert {
            point: WalPoint::from_point(&point(i)).unwrap(),
        })
        .collect()
}

/// Metric deltas one configuration produces.
#[derive(Default, Clone, Copy)]
struct Report {
    total_ms: f64,
    txns: u64,
    flushes: u64,
    wait_per_txn_us: f64,
    apply_per_txn_us: f64,
}

impl Report {
    fn pts_per_sec(&self) -> f64 {
        (self.txns * BATCH as u64) as f64 / (self.total_ms / 1000.0)
    }

    fn group(&self) -> f64 {
        self.txns as f64 / self.flushes.max(1) as f64
    }
}

/// Snapshot the counters this bench reads.
fn snap(metrics: &simvec::metrics::MetricsSnapshot) -> (u64, u64, u64, u64) {
    (
        metrics.wal_flushes,
        metrics.wal_txns_flushed,
        metrics.apply_wait_nanos,
        metrics.apply_txn.total_nanos,
    )
}

/// Store path: the commit queue and nothing else. No engine, no maintenance
/// thread, no index build, so the queue's own effect is isolated. Payload-free
/// on purpose: the engine modes below are where that cost belongs.
fn run_store(dir: &PathBuf, writer_count: usize, total: usize) -> Report {
    let store = CollectionStore::create(
        dir,
        "col",
        &CollectionConfig::new(DIM, DistanceMetric::Cosine),
    )
    .unwrap();
    let store = Arc::new(store);
    let before = store.metrics().snapshot();
    let (f0, t0, w0, a0) = snap(&before);

    let start = Arc::new(std::sync::Barrier::new(writer_count));
    let mut handles = Vec::new();
    for w in 0..writer_count {
        let store = Arc::clone(&store);
        let start = Arc::clone(&start);
        handles.push(std::thread::spawn(move || {
            // The barrier puts every writer inside the queue at once: without
            // it the first one commits and returns before the last starts.
            start.wait();
            for t in 0..total {
                // Round-robin, exactly like the wrapper bench's concurrent
                // phase, so the transaction count is identical per writer.
                if t % writer_count != w {
                    continue;
                }
                store.apply_ops(&upsert_ops(t * BATCH)).unwrap();
            }
        }));
    }
    let clock = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let total_ms = clock.elapsed().as_secs_f64() * 1000.0;

    let after = store.metrics().snapshot();
    let (f1, t1, w1, a1) = snap(&after);
    let txns = t1 - t0;
    drop(store);

    Report {
        total_ms,
        txns,
        flushes: f1 - f0,
        wait_per_txn_us: (w1 - w0) as f64 / 1000.0 / txns.max(1) as f64,
        apply_per_txn_us: (a1 - a0) as f64 / 1000.0 / txns.max(1) as f64,
    }
}

/// Engine path: the wrapper layer's shape. Adds the maintenance thread, the
/// HNSW index build and the post-mutation repair, all of which compete for
/// the same cores the overlapping was supposed to free.
fn run_engine(dir: &PathBuf, writer_count: usize, total: usize, with_payload: bool) -> Report {
    let engine = LocalVectorEngine::open(dir).unwrap();
    engine
        .create_collection("col", &CollectionConfig::new(DIM, DistanceMetric::Cosine))
        .unwrap();
    let before = engine.collection_metrics("col").unwrap();
    let (f0, t0, w0, a0) = snap(&before);
    let engine = Arc::new(engine);

    let start = Arc::new(std::sync::Barrier::new(writer_count));
    let mut handles = Vec::new();
    for w in 0..writer_count {
        let engine = Arc::clone(&engine);
        let start = Arc::clone(&start);
        handles.push(std::thread::spawn(move || {
            start.wait();
            for t in 0..total {
                if t % writer_count != w {
                    continue;
                }
                let points: Vec<VectorPoint> = (t * BATCH..t * BATCH + BATCH)
                    .map(|id| {
                        if with_payload {
                            point_with_payload(id)
                        } else {
                            point(id)
                        }
                    })
                    .collect();
                engine.upsert_batch("col", &points).unwrap();
            }
        }));
    }
    let clock = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    let total_ms = clock.elapsed().as_secs_f64() * 1000.0;

    let after = engine.collection_metrics("col").unwrap();
    let (f1, t1, w1, a1) = snap(&after);
    let txns = t1 - t0;
    drop(engine);

    Report {
        total_ms,
        txns,
        flushes: f1 - f0,
        wait_per_txn_us: (w1 - w0) as f64 / 1000.0 / txns.max(1) as f64,
        apply_per_txn_us: (a1 - a0) as f64 / 1000.0 / txns.max(1) as f64,
    }
}

/// Owned scratch directory under the crate's `target/` tree, removed on drop.
/// See the module note on why the system temp dir cannot be used here.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../target/bench-vector")
            .join(format!("write-{label}-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create scratch dir");
        Self(path)
    }

    fn path(&self) -> &PathBuf {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn main() {
    let total = txns();
    println!(
        "concurrent_write bench (dim={DIM}, batch={BATCH}, txns={})",
        total
    );
    println!(
        "{:<13} {:>7} {:>9} {:>10} {:>11} {:>7} {:>9} {:>11}",
        "mode", "writers", "total_ms", "pts/s", "txn/flush", "group", "wait/txn", "apply/txn"
    );

    for mode in ["store", "engine", "engine+pl"] {
        // Payloads are a wrapper-layer cost, so they only shape the engine
        // modes; the store path stays payload-free to keep the queue isolated.
        let with_payload = mode != "store";
        for writer_count in writer_counts() {
            // A writer with no transaction to commit would only add
            // scheduler noise, so the small tiers run with fewer writers.
            if writer_count > total {
                continue;
            }
            let scratch = ScratchDir::new(&format!("{mode}-w{writer_count}"));
            // `CollectionStore::create` refuses an existing directory, so the
            // store path gets a fresh subdirectory inside the scratch root.
            let store_dir = scratch.path().join("col");
            let report = if mode == "store" {
                run_store(&store_dir, writer_count, total)
            } else {
                run_engine(scratch.path(), writer_count, total, with_payload)
            };
            // The store or engine inside the run must be gone before the
            // scratch directory is removed.
            drop(scratch);
            println!(
                "{:<13} {:>7} {:>9.1} {:>10.0} {:>4}/{:<6} {:>7.2} {:>8.0}us {:>8.0}us",
                mode,
                writer_count,
                report.total_ms,
                report.pts_per_sec(),
                report.txns,
                report.flushes,
                report.group(),
                report.wait_per_txn_us,
                report.apply_per_txn_us
            );
        }
    }
}
