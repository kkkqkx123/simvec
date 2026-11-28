//! Compaction I/O versus garbage volume — the metric N1b is judged by.
//!
//! The question is not "how fast is compaction" but "how much does it write".
//! A whole-generation rewrite writes every live byte no matter how little is
//! garbage; a segment-level compaction writes only the selected segments. So
//! the measured quantity is **bytes written by the new generation**, counted
//! without instrumenting the crate:
//!
//! ```text
//! bytes_written = Σ{ length of every file in the new generation whose inode
//!                    does not appear in the old generation }
//! ```
//!
//! A file that was reused (hard linked) keeps its inode and therefore counts
//! zero, which is exactly the property segment-level compaction must deliver.
//!
//! Expected shape per milestone:
//!
//! - **Before N1b**: `bytes_written ≈ live bytes`, flat in the garbage column.
//!   That is the baseline recorded here, not a regression.
//! - **After N1b**: `bytes_written ≈ garbage-segment bytes`, i.e. it grows with
//!   the garbage column and stays flat in the collection size column.
//!
//! Run: `QUICK=1 cargo bench --bench compaction_io_vs_garbage` for the
//! seconds-long gear, without `QUICK` for the promotion-scale gear. Numbers
//! from the two gears (and from different machines) are never comparable.

#[path = "quick.rs"]
#[allow(dead_code)]
mod quick;

use std::path::Path;
use std::time::{Duration, Instant};

use simvec::storage::{CollectionStore, WalPoint, WalRecord};
use simvec::types::{CollectionConfig, DistanceMetric, PointId, VectorPoint};

const DIM: usize = 64;
const SEGMENT_SLOTS: u32 = 1_024;

fn point(id: u64) -> VectorPoint {
    VectorPoint::new(
        id,
        (0..DIM)
            .map(|i| ((id as usize * 17 + i * 7) % 251) as f32 / 251.0)
            .collect(),
    )
}

/// Every file under `dir`, as `(inode, length)`.
fn inode_index(dir: &Path) -> Vec<(u64, u64)> {
    let mut entries = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(read) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in read.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.is_dir() {
                stack.push(path);
            } else if let Ok(file) = std::fs::metadata(&path) {
                use std::os::unix::fs::MetadataExt;
                entries.push((file.ino(), file.len()));
            }
        }
    }
    entries
}

/// Bytes the new generation had to write: files whose inode is new.
fn bytes_written(before: &[(u64, u64)], after: &[(u64, u64)]) -> u64 {
    let old: std::collections::HashSet<u64> = before.iter().map(|(ino, _)| *ino).collect();
    after
        .iter()
        .filter(|(ino, _)| !old.contains(ino))
        .map(|(_, len)| *len)
        .sum()
}

struct Row {
    garbage: u64,
    live_bytes: u64,
    written: u64,
    elapsed: Duration,
}

/// Build `live` points, tombstone `garbage` of them in the **first** segment,
/// then compact and account for the bytes the new generation wrote.
fn run_case(live: u64, garbage: u64) -> Row {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = CollectionConfig::new(DIM, DistanceMetric::Euclid);
    let store = CollectionStore::create_with_segment_slots(
        dir.path().join("bench"),
        "bench",
        &config,
        SEGMENT_SLOTS,
    )
    .expect("create");

    // One transaction per point keeps the WAL/apply path on the same route the
    // engine uses, without batching effects skewing the fixture build.
    for id in 1..=live {
        store
            .apply_ops(&[WalRecord::Upsert {
                point: WalPoint::from_point(&point(id)).expect("wal point"),
            }])
            .expect("upsert");
    }
    // Garbage concentrated at the head: those slots live in complete segments,
    // so a segment-level compaction can leave every other segment alone.
    for id in 1..=garbage {
        store
            .apply_ops(&[WalRecord::Delete {
                point_id: PointId::from(id).to_string(),
            }])
            .expect("delete");
    }
    store.checkpoint().expect("checkpoint");

    let before = inode_index(&store.data_dir());
    let live_bytes: u64 = before.iter().map(|(_, len)| *len).sum();
    let started = Instant::now();
    let remaining = store.compact().expect("compact");
    let elapsed = started.elapsed();
    assert_eq!(remaining, live - garbage);
    let after = inode_index(&store.data_dir());
    let written = bytes_written(&before, &after);

    Row {
        garbage,
        live_bytes,
        written,
        elapsed,
    }
}

fn main() {
    let live = quick::scale(8_000, 2_000) as u64;
    // Three garbage volumes on the same collection size: the whole point is to
    // see whether the write cost follows the garbage column.
    let ratios = [0.1, 0.3, 0.5];

    println!(
        "\ncompaction I/O vs garbage (live={live}, dim={DIM}, segment_slots={SEGMENT_SLOTS})"
    );
    println!(
        "{:>8} {:>10} {:>12} {:>12} {:>9} {:>10}",
        "garbage", "live_bytes", "bytes_written", "write/live", "ms", "MB/s"
    );
    for ratio in ratios {
        let garbage = (live as f64 * ratio) as u64;
        let row = run_case(live, garbage);
        let write_ratio = row.written as f64 / row.live_bytes.max(1) as f64;
        let mbps = (row.written as f64 / 1e6) / row.elapsed.as_secs_f64().max(1e-9);
        println!(
            "{:>8} {:>12} {:>12} {:>12.2} {:>9.1} {:>10.1}",
            row.garbage,
            row.live_bytes,
            row.written,
            write_ratio,
            row.elapsed.as_secs_f64() * 1e3,
            mbps
        );
    }
    println!(
        "\npre-N1b baseline: bytes_written tracks live_bytes (whole-generation rewrite).\n\
         after N1b it must track the garbage column and stay flat in live_bytes.\n"
    );
}
