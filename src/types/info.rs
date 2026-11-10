use serde::{Deserialize, Serialize};

use super::config::CollectionConfig;

/// Published index tier backing a collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexKind {
    ExactScan,
    Ivf,
    Hnsw,
}

/// Index state exposed through [`CollectionInfo`].
///
/// Kind-agnostic: IVF reports list/nprobe fields, HNSW reports graph
/// parameters; fields that do not apply to the active kind stay zeroed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    pub index_kind: IndexKind,
    /// IVF only: number of clusters.
    pub lists: u32,
    /// IVF only: default nprobe.
    pub nprobe_default: usize,
    /// HNSW only: maximum connections per layer (> 0).
    pub m: usize,
    /// HNSW only: build-time candidate list size.
    pub ef_construct: usize,
    /// HNSW only: default layer-0 search width.
    pub ef_search_default: usize,
    pub built_at_live_count: u64,
    /// HNSW only: overwrite upserts observed since the graph was built (or
    /// reloaded). Combined with `built_at_live_count` this yields the stale
    /// position ratio consumed by `HnswConfig::stale_rebuild_ratio`.
    pub stale_overwrite_count: u64,
    /// HNSW only: combined staleness ratio `max(count_ratio, delta_ratio)`.
    /// When present, preferred over raw `stale_overwrite_count / built_at_live_count`.
    pub stale_ratio: Option<f64>,
    /// IVF only: last measured cluster drift ratio.
    pub last_drift_ratio: Option<f64>,
    /// Whether an ANN index build is currently in flight.
    #[serde(default)]
    pub building: bool,
    /// In-flight build progress: slots incorporated so far.
    #[serde(default)]
    pub build_inserted: u64,
    /// In-flight build progress: total slots the running build targets.
    #[serde(default)]
    pub build_points_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub vector_count: u64,
    pub indexed_vector_count: u64,
    pub points_count: u64,
    pub segments_count: u64,
    pub config: CollectionConfig,
    pub status: CollectionStatus,
    pub index: Option<IndexInfo>,
    /// Estimated resident index footprint in bytes (vectors, payload pages,
    /// pointer table, directory files, ANN graph and the current WAL), so
    /// callers can judge whether a collection fits the host before writing.
    pub memory_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollectionStatus {
    Green,
    Yellow,
    Red,
    Grey,
}

/// Metadata describing a registered vector index.
///
/// Shared between the local engine and the qdrant client so the sync layer can
/// track logical indexes without depending on the transport-specific crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub name: String,
    pub config: CollectionConfig,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub vector_count: u64,
    pub index_name: Option<String>,
}

impl IndexMetadata {
    pub fn new(name: String, config: CollectionConfig) -> Self {
        Self {
            name,
            config,
            created_at: chrono::Utc::now(),
            vector_count: 0,
            index_name: None,
        }
    }

    pub fn with_index_name(name: String, config: CollectionConfig, index_name: String) -> Self {
        Self {
            name,
            config,
            created_at: chrono::Utc::now(),
            vector_count: 0,
            index_name: Some(index_name),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub is_healthy: bool,
    pub engine_name: String,
    pub engine_version: String,
    pub message: Option<String>,
}

impl HealthStatus {
    pub fn healthy(engine_name: impl Into<String>, engine_version: impl Into<String>) -> Self {
        Self {
            is_healthy: true,
            engine_name: engine_name.into(),
            engine_version: engine_version.into(),
            message: None,
        }
    }

    pub fn unhealthy(
        engine_name: impl Into<String>,
        engine_version: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            is_healthy: false,
            engine_name: engine_name.into(),
            engine_version: engine_version.into(),
            message: Some(message.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct UpsertResult {
    pub operation_id: Option<u64>,
    pub status: UpsertStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpsertStatus {
    Completed,
    Acknowledged,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResult {
    pub operation_id: Option<u64>,
    /// Best-effort estimate of the number of deleted points.
    ///
    /// The local engine reports exact counts. Remote backends report the
    /// requested batch size once the server acknowledges completion, and a
    /// status-only estimate (1/0) for filter deletes; callers must not rely
    /// on this value for accounting.
    pub deleted_count: u64,
}
