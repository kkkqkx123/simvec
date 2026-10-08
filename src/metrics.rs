//! Operational metrics for the local vector engine.
//!
//! Self-contained and wait-free: counters and latency histograms are plain
//! atomics, so recording never blocks the search or write paths. The crate
//! intentionally does not depend on a monitoring stack; [`MetricsSnapshot`]
//! is serializable so an embedder (sync / server layer) can
//! forward values into its own observability layer.
//!
//! Instrumented paths, per collection:
//! - mutations: applied transactions, upserted/deleted points, apply latency
//!   (dominated by the WAL fsync inside the store write lock)
//! - searches: exact/IVF/HNSW path split, filtered queries, latency, and the
//!   accuracy-fallback retries that signal recall degradation under filters
//! - index lifecycle: build counts and latencies per tier, load-time
//!   fallbacks to exact scan
//! - compaction: commits, race retries, contended write-lock fallbacks,
//!   latency
//! - lock contention: adjacency/list write-lock acquisition counts and wait
//!   times (compiled in only with the `lock-metrics` feature, off by
//!   default), plus the version-double-read reloads observed on the HNSW
//!   search path

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::{RwLock, RwLockWriteGuard};
use serde::Serialize;

/// Number of power-of-two buckets in [`PowerLatencyHistogram`]. Bucket `i`
/// covers `[2^i, 2^(i+1))` nanoseconds; 64 buckets cover every `u64`.
const LATENCY_BUCKETS: usize = 64;

/// Wait-free latency histogram with power-of-two nanosecond buckets.
///
/// Named apart from the exact `LatencyHistogram` in observability: this one
/// trades percentile precision (factor-of-two bucket bounds) for wait-free
/// recording on hot search paths.
///
/// Any duration fits without dynamic allocation. Percentile estimates are
/// reported as the upper bound of the bucket holding the target rank, i.e.
/// within a factor of two of the true value.
#[derive(Debug)]
pub struct PowerLatencyHistogram {
    count: AtomicU64,
    total_nanos: AtomicU64,
    buckets: [AtomicU64; LATENCY_BUCKETS],
}

impl Default for PowerLatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl PowerLatencyHistogram {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_nanos: AtomicU64::new(0),
            buckets: [const { AtomicU64::new(0) }; LATENCY_BUCKETS],
        }
    }

    /// Record one observation.
    fn record(&self, nanos: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_nanos.fetch_add(nanos, Ordering::Relaxed);
        let idx = if nanos == 0 {
            0
        } else {
            63 - nanos.leading_zeros() as usize
        };
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
    }
    fn percentile(&self, total: u64, q: f64) -> Option<u64> {
        if total == 0 {
            return None;
        }
        let target = ((total as f64) * q).ceil() as u64;
        let mut cum = 0u64;
        for (i, bucket) in self.buckets.iter().enumerate() {
            cum += bucket.load(Ordering::Relaxed);
            if cum >= target {
                return Some(if i + 1 >= LATENCY_BUCKETS {
                    u64::MAX
                } else {
                    1u64 << (i + 1)
                });
            }
        }
        Some(u64::MAX)
    }

    fn summary(&self) -> LatencySummary {
        let count = self.count.load(Ordering::Relaxed);
        let total_nanos = self.total_nanos.load(Ordering::Relaxed);
        LatencySummary {
            count,
            total_nanos,
            p50_nanos: self.percentile(count, 0.50),
            p95_nanos: self.percentile(count, 0.95),
            p99_nanos: self.percentile(count, 0.99),
        }
    }
}

/// Aggregate view of one latency histogram.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct LatencySummary {
    pub count: u64,
    pub total_nanos: u64,
    pub p50_nanos: Option<u64>,
    pub p95_nanos: Option<u64>,
    pub p99_nanos: Option<u64>,
}

impl LatencySummary {
    /// Mean latency in nanoseconds, or `None` when nothing was recorded.
    pub fn avg_nanos(&self) -> Option<u64> {
        (self.count != 0).then(|| self.total_nanos / self.count)
    }
}

/// Search execution path taken for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchPath {
    Exact,
    Ivf,
    Hnsw,
    Quantized,
}

/// Accuracy fallback triggered because a filter left too few results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchRetry {
    /// IVF widened the probe width (`nprobe` doubling).
    NprobeDoubling,
    /// HNSW resumed from candidates discarded by a previous pass.
    IterativeExpansion,
    /// HNSW doubled `ef` as the last resort.
    EfDoubling,
}

/// ANN index tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexTier {
    Hnsw,
    Ivf,
}

/// Acquire a write guard on an adjacency/list lock while measuring the
/// wait time, when lock metrics are compiled in and a recorder is attached.
///
/// Instrumentation points: HNSW adjacency writes (`insert`, `link_neighbor`,
/// `repair`) and IVF list membership writes (`assign_slot`).
#[cfg(feature = "lock-metrics")]
pub(crate) fn timed_write_lock<'a, T>(
    recorder: Option<&'a Metrics>,
    lock: &'a RwLock<T>,
) -> RwLockWriteGuard<'a, T> {
    let started = std::time::Instant::now();
    let guard = lock.write();
    if let Some(metrics) = recorder {
        metrics.record_lock_wait(started.elapsed());
    }
    guard
}

#[cfg(not(feature = "lock-metrics"))]
pub(crate) fn timed_write_lock<'a, T>(
    recorder: Option<&'a Metrics>,
    lock: &'a RwLock<T>,
) -> RwLockWriteGuard<'a, T> {
    let _ = recorder;
    lock.write()
}

/// Cumulative per-collection operational metrics.
///
/// All recording methods are wait-free and safe to call concurrently.
#[derive(Debug, Default)]
pub struct Metrics {
    txns_applied: AtomicU64,
    points_upserted: AtomicU64,
    points_deleted: AtomicU64,
    apply_txn_latency: PowerLatencyHistogram,

    search_total: AtomicU64,
    search_exact: AtomicU64,
    search_ivf: AtomicU64,
    search_hnsw: AtomicU64,
    search_quantized: AtomicU64,
    search_filtered: AtomicU64,
    /// Searches that failed at the engine boundary (validation, corruption).
    ///
    /// Counted in the engine wrappers rather than the store so the op-type
    /// attribution (search vs upsert vs delete) is exact.
    search_errors: AtomicU64,
    search_nprobe_retries: AtomicU64,
    search_iterative_expansions: AtomicU64,
    search_ef_retries: AtomicU64,
    /// Adjacency reads where the version double-read protocol detected a
    /// concurrent mutation and reloaded the neighborhood (HNSW only).
    search_version_reloads: AtomicU64,
    search_latency: PowerLatencyHistogram,

    /// Adjacency/list write-lock acquisitions. Only incremented with the
    /// `lock-metrics` feature enabled.
    adjacency_write_locks: AtomicU64,
    /// Cumulative wait time of those acquisitions, in nanoseconds.
    adjacency_lock_wait_nanos: AtomicU64,

    upsert_errors: AtomicU64,
    delete_errors: AtomicU64,

    hnsw_builds: AtomicU64,
    ivf_builds: AtomicU64,
    index_load_fallbacks: AtomicU64,
    hnsw_build_latency: PowerLatencyHistogram,
    ivf_build_latency: PowerLatencyHistogram,

    compactions: AtomicU64,
    compaction_race_retries: AtomicU64,
    compaction_contended: AtomicU64,
    compaction_latency: PowerLatencyHistogram,
}

impl Metrics {
    /// Record one successfully applied WAL transaction.
    pub fn record_apply_txn(&self, points_upserted: u64, points_deleted: u64, elapsed: Duration) {
        self.txns_applied.fetch_add(1, Ordering::Relaxed);
        self.points_upserted
            .fetch_add(points_upserted, Ordering::Relaxed);
        self.points_deleted
            .fetch_add(points_deleted, Ordering::Relaxed);
        self.apply_txn_latency.record(elapsed.as_nanos() as u64);
    }

    /// Record one completed search with the path it took.
    pub fn record_search(&self, path: SearchPath, filtered: bool, elapsed: Duration) {
        self.search_total.fetch_add(1, Ordering::Relaxed);
        match path {
            SearchPath::Exact => &self.search_exact,
            SearchPath::Ivf => &self.search_ivf,
            SearchPath::Hnsw => &self.search_hnsw,
            SearchPath::Quantized => &self.search_quantized,
        }
        .fetch_add(1, Ordering::Relaxed);
        if filtered {
            self.search_filtered.fetch_add(1, Ordering::Relaxed);
        }
        self.search_latency.record(elapsed.as_nanos() as u64);
    }

    /// Record an accuracy fallback triggered during a filtered search.
    pub fn record_search_retry(&self, retry: SearchRetry) {
        let counter = match retry {
            SearchRetry::NprobeDoubling => &self.search_nprobe_retries,
            SearchRetry::IterativeExpansion => &self.search_iterative_expansions,
            SearchRetry::EfDoubling => &self.search_ef_retries,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one adjacency read where the version double-read protocol
    /// detected a concurrent mutation and reloaded the neighborhood.
    pub fn record_version_reload(&self) {
        self.search_version_reloads.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one sampled adjacency/list write-lock acquisition with the
    /// time spent waiting for it.
    ///
    /// A no-op unless the `lock-metrics` feature is compiled in; the
    /// feature keeps the hot mutation paths free of timing overhead by
    /// default.
    pub fn record_lock_wait(&self, waited: Duration) {
        #[cfg(feature = "lock-metrics")]
        {
            self.adjacency_write_locks.fetch_add(1, Ordering::Relaxed);
            self.adjacency_lock_wait_nanos
                .fetch_add(waited.as_nanos() as u64, Ordering::Relaxed);
        }
        #[cfg(not(feature = "lock-metrics"))]
        {
            let _ = waited;
        }
    }

    /// Record a failed search (engine boundary).
    pub fn record_search_error(&self) {
        self.search_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a failed upsert (engine boundary).
    pub fn record_upsert_error(&self) {
        self.upsert_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a failed delete (engine boundary).
    pub fn record_delete_error(&self) {
        self.delete_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successful ANN index build of `tier`.
    pub fn record_index_build(&self, tier: IndexTier, elapsed: Duration) {
        match tier {
            IndexTier::Hnsw => {
                self.hnsw_builds.fetch_add(1, Ordering::Relaxed);
                self.hnsw_build_latency.record(elapsed.as_nanos() as u64);
            }
            IndexTier::Ivf => {
                self.ivf_builds.fetch_add(1, Ordering::Relaxed);
                self.ivf_build_latency.record(elapsed.as_nanos() as u64);
            }
        }
    }

    /// Record that a persisted index failed validation on open and the
    /// collection fell back to exact scan.
    pub fn record_index_load_fallback(&self) {
        self.index_load_fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one committed compaction.
    pub fn record_compaction(&self, elapsed: Duration) {
        self.compactions.fetch_add(1, Ordering::Relaxed);
        self.compaction_latency.record(elapsed.as_nanos() as u64);
    }

    /// Record a discarded compaction commit attempt (a concurrent write
    /// raced the temp-file rewrite phase).
    pub fn record_compaction_race_retry(&self) {
        self.compaction_race_retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that compaction had to rewrite files while holding the store
    /// write lock after exhausting its lock-free attempts.
    pub fn record_compaction_contended(&self) {
        self.compaction_contended.fetch_add(1, Ordering::Relaxed);
    }

    /// Take a point-in-time snapshot. Counters may be slightly skewed
    /// relative to each other under concurrent recording.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            txns_applied: self.txns_applied.load(Ordering::Relaxed),
            points_upserted: self.points_upserted.load(Ordering::Relaxed),
            points_deleted: self.points_deleted.load(Ordering::Relaxed),
            apply_txn: self.apply_txn_latency.summary(),

            search_total: self.search_total.load(Ordering::Relaxed),
            search_exact: self.search_exact.load(Ordering::Relaxed),
            search_ivf: self.search_ivf.load(Ordering::Relaxed),
            search_hnsw: self.search_hnsw.load(Ordering::Relaxed),
            search_quantized: self.search_quantized.load(Ordering::Relaxed),
            search_filtered: self.search_filtered.load(Ordering::Relaxed),
            search_errors: self.search_errors.load(Ordering::Relaxed),
            search_nprobe_retries: self.search_nprobe_retries.load(Ordering::Relaxed),
            search_iterative_expansions: self.search_iterative_expansions.load(Ordering::Relaxed),
            search_ef_retries: self.search_ef_retries.load(Ordering::Relaxed),
            search_version_reloads: self.search_version_reloads.load(Ordering::Relaxed),
            search: self.search_latency.summary(),

            adjacency_write_locks: self.adjacency_write_locks.load(Ordering::Relaxed),
            adjacency_lock_wait_nanos: self.adjacency_lock_wait_nanos.load(Ordering::Relaxed),

            upsert_errors: self.upsert_errors.load(Ordering::Relaxed),
            delete_errors: self.delete_errors.load(Ordering::Relaxed),

            hnsw_builds: self.hnsw_builds.load(Ordering::Relaxed),
            ivf_builds: self.ivf_builds.load(Ordering::Relaxed),
            index_load_fallbacks: self.index_load_fallbacks.load(Ordering::Relaxed),
            hnsw_build: self.hnsw_build_latency.summary(),
            ivf_build: self.ivf_build_latency.summary(),

            compactions: self.compactions.load(Ordering::Relaxed),
            compaction_race_retries: self.compaction_race_retries.load(Ordering::Relaxed),
            compaction_contended: self.compaction_contended.load(Ordering::Relaxed),
            compaction: self.compaction_latency.summary(),
        }
    }
}

/// Serializable snapshot of [`Metrics`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct MetricsSnapshot {
    // Mutations.
    pub txns_applied: u64,
    pub points_upserted: u64,
    pub points_deleted: u64,
    /// Apply latency includes the WAL fsync inside the store write lock.
    pub apply_txn: LatencySummary,

    // Searches.
    pub search_total: u64,
    pub search_exact: u64,
    pub search_ivf: u64,
    pub search_hnsw: u64,
    pub search_quantized: u64,
    pub search_filtered: u64,
    /// Searches that failed at the engine boundary.
    pub search_errors: u64,
    /// Filtered searches where IVF doubled its probe width.
    pub search_nprobe_retries: u64,
    /// Filtered searches where HNSW resumed from discarded candidates.
    pub search_iterative_expansions: u64,
    /// Filtered searches where HNSW doubled `ef`.
    pub search_ef_retries: u64,
    /// HNSW adjacency reads where the version double-read detected a
    /// concurrent mutation and reloaded the neighborhood.
    pub search_version_reloads: u64,
    pub search: LatencySummary,

    /// Adjacency/list write-lock acquisitions; only grows with the
    /// `lock-metrics` feature enabled (stays 0 otherwise).
    pub adjacency_write_locks: u64,
    /// Cumulative wait time of those acquisitions, in nanoseconds.
    pub adjacency_lock_wait_nanos: u64,

    /// Upserts that failed at the engine boundary (validation, WAL, IO).
    pub upsert_errors: u64,
    /// Deletes that failed at the engine boundary.
    pub delete_errors: u64,

    // Index lifecycle.
    pub hnsw_builds: u64,
    pub ivf_builds: u64,
    /// Persisted indexes rejected on open, degrading to exact scan.
    pub index_load_fallbacks: u64,
    pub hnsw_build: LatencySummary,
    pub ivf_build: LatencySummary,

    // Compaction.
    pub compactions: u64,
    pub compaction_race_retries: u64,
    pub compaction_contended: u64,
    pub compaction: LatencySummary,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_percentiles_within_factor_of_two() {
        let h = PowerLatencyHistogram::new();
        for ns in [100u64, 200, 300] {
            h.record(ns);
        }
        let s = h.summary();
        assert_eq!(s.count, 3);
        assert_eq!(s.total_nanos, 600);
        assert_eq!(s.avg_nanos(), Some(200));
        // Median is 200ns; bucket upper bounds are powers of two.
        assert_eq!(s.p50_nanos, Some(256));
        assert_eq!(s.p99_nanos, Some(512));
    }

    #[test]
    fn histogram_empty_summary() {
        let h = PowerLatencyHistogram::new();
        assert_eq!(h.summary(), LatencySummary::default());
    }

    #[test]
    fn histogram_extreme_durations_stay_in_range() {
        let h = PowerLatencyHistogram::new();
        h.record(0);
        h.record(u64::MAX);
        let s = h.summary();
        // Rank 1 is the 0ns sample in bucket 0 ([1, 2)); the max-latency
        // sample saturates the last bucket.
        assert_eq!(s.p50_nanos, Some(2));
        assert_eq!(s.p99_nanos, Some(u64::MAX));
    }

    #[test]
    fn metrics_record_and_snapshot() {
        let m = Metrics::default();
        m.record_apply_txn(3, 1, Duration::from_micros(10));
        m.record_search(SearchPath::Hnsw, true, Duration::from_micros(20));
        m.record_search(SearchPath::Exact, false, Duration::from_micros(5));
        m.record_search_retry(SearchRetry::IterativeExpansion);
        m.record_version_reload();
        m.record_index_build(IndexTier::Ivf, Duration::from_millis(4));
        m.record_index_load_fallback();
        m.record_compaction(Duration::from_millis(8));
        m.record_compaction_race_retry();
        m.record_search_error();
        m.record_upsert_error();
        m.record_delete_error();

        let s = m.snapshot();
        assert_eq!(s.txns_applied, 1);
        assert_eq!(s.points_upserted, 3);
        assert_eq!(s.points_deleted, 1);
        assert_eq!(s.search_total, 2);
        assert_eq!(s.search_hnsw, 1);
        assert_eq!(s.search_exact, 1);
        assert_eq!(s.search_filtered, 1);
        assert_eq!(s.search_iterative_expansions, 1);
        assert_eq!(s.search_version_reloads, 1);
        // Lock metrics stay zero unless the `lock-metrics` feature records
        // them; the fields must exist in every build for downstream samplers.
        assert_eq!(s.adjacency_write_locks, 0);
        assert_eq!(s.adjacency_lock_wait_nanos, 0);
        assert_eq!(s.ivf_builds, 1);
        assert_eq!(s.index_load_fallbacks, 1);
        assert_eq!(s.compactions, 1);
        assert_eq!(s.compaction_race_retries, 1);
        assert_eq!(s.search_errors, 1);
        assert_eq!(s.upsert_errors, 1);
        assert_eq!(s.delete_errors, 1);

        // Snapshot is JSON serializable for downstream forwarding.
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"search_total\":2"));
    }
}
