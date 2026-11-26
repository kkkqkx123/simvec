use serde::{Deserialize, Serialize};

/// Remote storage-engine hints carried alongside a collection request.
///
/// These map onto the remote service's per-collection storage layout
/// (segment optimizer, write-ahead log, strict mode). The local engine
/// persists them so a config round-trip stays lossless but never acts on
/// them: local storage layout is governed by [`StorageConfig`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct StorageHints {
    /// Segment optimizer hints.
    #[serde(default)]
    pub optimizers: Option<OptimizersHints>,
    /// Write-ahead log hints.
    #[serde(default)]
    pub wal: Option<WalHints>,
    /// Whether the remote backend enforces strict-mode query limits.
    #[serde(default)]
    pub strict_mode_enabled: Option<bool>,
}

/// Segment-optimizer hints, mirroring the remote service's diff message.
///
/// Every field is optional so a caller can override one knob without
/// restating the server defaults.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct OptimizersHints {
    /// Deleted-vector ratio that triggers segment optimization.
    #[serde(default)]
    pub deleted_threshold: Option<f64>,
    /// Minimum vector count before vacuuming a segment.
    #[serde(default)]
    pub vacuum_min_vector_number: Option<u64>,
    /// Preferred segment count; `Some(0)` means "let the server decide".
    #[serde(default)]
    pub default_segment_number: Option<u64>,
    /// Upper bound on a segment's vector count before it is split.
    #[serde(default)]
    pub max_segment_size: Option<u64>,
    /// Vector count above which a segment is memory-mapped instead of kept
    /// in RAM.
    #[serde(default)]
    pub memmap_threshold: Option<u64>,
    /// Vector count above which a segment gets an HNSW index.
    #[serde(default)]
    pub indexing_threshold: Option<u64>,
    /// Seconds between forced segment flushes.
    #[serde(default)]
    pub flush_interval_sec: Option<u64>,
    /// Optimizer threads; `Some(0)` means the server picks.
    #[serde(default)]
    pub max_optimization_threads: Option<u64>,
}

/// Write-ahead log hints.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct WalHints {
    /// WAL capacity in megabytes.
    #[serde(default)]
    pub wal_capacity_mb: Option<u64>,
    /// How many segments to keep ahead of the current one.
    #[serde(default)]
    pub wal_segments_ahead: Option<u64>,
}
