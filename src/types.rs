//! Shared vector types.
//!
//! Migrated from `vector-client` so that the local engine and the qdrant
//! client share the exact same type surface. Kept as a single module for now.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub use crate::filter_cond::{
    ConditionType, FilterCondition, GeoBoundingBox, GeoPoint, GeoRadius, MinShouldCondition,
    PayloadSchemaType, PayloadSelector, PayloadValue, PointId, RangeCondition,
    ValuesCountCondition, VectorFilter,
};
pub type Payload = std::collections::HashMap<String, serde_json::Value>;

use crate::error::{Result, VectorSearchError};

pub type CollectionName = String;

/// Default for `with_payload` when a query leaves it unset.
///
/// Both backends must agree on this: the local engine and the Qdrant
/// transports all resolve `None` through this constant so an explicit
/// `None` cannot fork behavior between engines.
pub const DEFAULT_WITH_PAYLOAD: bool = true;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, utoipa::ToSchema)]
pub enum DistanceMetric {
    #[default]
    Cosine,
    Euclid,
    Dot,
    /// Manhattan distance (`Σ|a-b|`). Supported by both the local engine
    /// and the remote Qdrant backend.
    Manhattan,
}

impl DistanceMetric {
    /// Whether the remote qdrant engine natively supports this metric.
    pub fn is_supported_by_qdrant(&self) -> bool {
        matches!(
            self,
            Self::Cosine | Self::Euclid | Self::Dot | Self::Manhattan
        )
    }

    /// Inverse of [`DistanceMetric::is_supported_by_qdrant`].
    #[doc(hidden)]
    pub fn requires_custom_implementation(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum connections per layer above the ground layer; the ground
    /// layer uses `2 * m`. Mirrors pgvector's `m` reloption.
    pub m: usize,
    /// Build-time candidate list size. Must satisfy `ef_construct >=
    /// max(2*m, 4)` (the same floor pgvector enforces); checked by
    /// [`HnswConfig::validate`] at collection creation, config update and
    /// index build time.
    pub ef_construct: usize,
    /// Live-point threshold above which an exact-scan collection is promoted
    /// to a published graph. `None` = engine default.
    pub full_scan_threshold: Option<usize>,
    /// Build parallelism for local graph builds. `None`/0 = sequential
    /// insertion on the global rayon pool; 1 = sequential on a dedicated
    /// single-thread pool; n >= 2 = n workers insert disjoint slot subsets
    /// into the shared graph concurrently (nondeterministic topology, same
    /// recall invariants).
    pub max_indexing_threads: Option<usize>,
    /// Qdrant-compat field, reserved and not wired locally: the local engine
    /// keeps hot indexes fully in memory and never spills them to disk.
    #[serde(default)]
    pub on_disk: Option<bool>,
    /// Qdrant-compat field, reserved and not wired locally: per-layer
    /// connection caps are derived from `m` (ground layer `2 * m`).
    #[serde(default)]
    pub payload_m: Option<usize>,
    /// Default `ef` for layer-0 graph search when a query leaves
    /// `SearchMode::KNN.ef_search` unset. Local engine only; remote backends
    /// apply their server-side default.
    #[serde(default = "default_hnsw_ef_search")]
    pub ef_search: usize,
    /// Staleness rebuild trigger: when the ratio of overwrite upserts since
    /// the last build to the built-at live count exceeds this value, the
    /// maintenance sweep schedules a rebuild (overwrite upserts keep their
    /// stale graph position, which slowly erodes recall). `None` disables
    /// staleness-triggered rebuilds.
    #[serde(default)]
    pub stale_rebuild_ratio: Option<f64>,
    /// Cap on iterative-scan expansion rounds when a filtered search comes
    /// up short. `None` = engine default. Baked into the published graph at
    /// build/reload time like the other search parameters.
    #[serde(default)]
    pub iterative_max_rounds: Option<usize>,
    /// Cumulative cap on distinct nodes visited across iterative-scan
    /// rounds; the scan stops once the budget is exhausted. `None` =
    /// unbounded.
    #[serde(default)]
    pub max_scan_tuples: Option<u64>,
}

fn default_hnsw_ef_search() -> usize {
    40
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construct: 100,
            full_scan_threshold: None,
            max_indexing_threads: None,
            on_disk: None,
            payload_m: None,
            ef_search: 40,
            stale_rebuild_ratio: None,
            iterative_max_rounds: None,
            max_scan_tuples: None,
        }
    }
}

impl HnswConfig {
    pub fn new(m: usize, ef_construct: usize) -> Self {
        Self {
            m,
            ef_construct,
            ..Self::default()
        }
    }

    /// Validate the graph parameters. Mirrors pgvector's CREATE INDEX
    /// reloption checks: `ef_construct` must be at least `max(2*m, 4)`
    /// so the beam search can never be narrower than one node's full
    /// neighborhood.
    pub fn validate(&self) -> Result<()> {
        let min_ef = self.m.saturating_mul(2).max(4);
        if self.ef_construct < min_ef {
            return Err(VectorSearchError::InvalidConfig(format!(
                "hnsw ef_construct {} must be >= max(2*m, 4) = {min_ef} (m = {})",
                self.ef_construct, self.m
            )));
        }
        if self.iterative_max_rounds.is_some_and(|r| r == 0) {
            return Err(VectorSearchError::InvalidConfig(
                "hnsw iterative_max_rounds must be >= 1 when set".to_string(),
            ));
        }
        if self.max_scan_tuples.is_some_and(|t| t == 0) {
            return Err(VectorSearchError::InvalidConfig(
                "hnsw max_scan_tuples must be >= 1 when set".to_string(),
            ));
        }
        Ok(())
    }

    pub fn with_full_scan_threshold(mut self, threshold: usize) -> Self {
        self.full_scan_threshold = Some(threshold);
        self
    }

    pub fn with_max_indexing_threads(mut self, threads: usize) -> Self {
        self.max_indexing_threads = Some(threads);
        self
    }

    pub fn with_on_disk(mut self, on_disk: bool) -> Self {
        self.on_disk = Some(on_disk);
        self
    }

    pub fn with_payload_m(mut self, payload_m: usize) -> Self {
        self.payload_m = Some(payload_m);
        self
    }

    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    pub fn with_stale_rebuild_ratio(mut self, ratio: f64) -> Self {
        self.stale_rebuild_ratio = Some(ratio);
        self
    }

    pub fn with_iterative_max_rounds(mut self, rounds: usize) -> Self {
        self.iterative_max_rounds = Some(rounds);
        self
    }

    pub fn with_max_scan_tuples(mut self, tuples: u64) -> Self {
        self.max_scan_tuples = Some(tuples);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IndexType {
    /// Default local ANN tier. Mirrors Qdrant's primary index so the local
    /// engine and remote backends expose the same knobs (`m`,
    /// `ef_construct`, per-query `ef_search`) and comparable behavior.
    #[default]
    HNSW,
    /// Exact scan only; no ANN structure is built or consulted.
    FLAT,
    /// IVFFlat alternative tier. Kept as an explicit opt-in; clustering
    /// semantics need bulk training and rebuild maintenance that suit
    /// large batch-oriented collections better than OLTP workloads.
    IVF,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionRatio {
    X4,
    X8,
    X16,
    X32,
    X64,
}

impl CompressionRatio {
    pub fn as_u32(self) -> u32 {
        match self {
            CompressionRatio::X4 => 4,
            CompressionRatio::X8 => 8,
            CompressionRatio::X16 => 16,
            CompressionRatio::X32 => 32,
            CompressionRatio::X64 => 64,
        }
    }

    /// Number of PQ subquantizers `M` for `dim`.
    ///
    /// PQ encodes each vector into `M` bytes (256 centroids per subspace).
    /// Compression ratio is `dim*4 / M`, so `M = dim*4 / ratio`.
    /// The result is clamped to at least 1 and at most `dim`, then adjusted
    /// down to the nearest divisor of `dim` to keep `dim % M == 0` where
    /// possible. For indivisible dims fallback to `M=1` ensures scalar tail
    /// handling elsewhere still works.
    pub fn pq_m(self, dim: usize) -> usize {
        let ratio = self.as_u32() as usize;
        if dim == 0 || ratio == 0 {
            return 1;
        }
        let mut m = (dim * 4).div_ceil(ratio);
        m = m.clamp(1, dim);
        while m > 1 && !dim.is_multiple_of(m) {
            m -= 1;
        }
        if !dim.is_multiple_of(m) {
            1
        } else {
            m
        }
    }

    pub fn sub_dim(self, dim: usize) -> usize {
        let m = self.pq_m(dim);
        dim / m.max(1)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantizationType {
    Scalar {
        quantile: Option<f32>,
        always_ram: Option<bool>,
    },
    Product {
        compression: CompressionRatio,
        always_ram: Option<bool>,
    },
    Binary {
        always_ram: Option<bool>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct QuantizationConfig {
    pub enabled: bool,
    pub quant_type: Option<QuantizationType>,
}

impl QuantizationConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            quant_type: None,
        }
    }

    pub fn scalar(quantile: f32) -> Self {
        Self {
            enabled: true,
            quant_type: Some(QuantizationType::Scalar {
                quantile: Some(quantile),
                always_ram: Some(true),
            }),
        }
    }

    pub fn product(compression: CompressionRatio) -> Self {
        Self {
            enabled: true,
            quant_type: Some(QuantizationType::Product {
                compression,
                always_ram: Some(true),
            }),
        }
    }

    pub fn binary() -> Self {
        Self {
            enabled: true,
            quant_type: Some(QuantizationType::Binary {
                always_ram: Some(true),
            }),
        }
    }

    pub fn with_always_ram(mut self, always_ram: bool) -> Self {
        if let Some(ref mut qt) = self.quant_type {
            match qt {
                QuantizationType::Scalar { always_ram: ar, .. } => *ar = Some(always_ram),
                QuantizationType::Product { always_ram: ar, .. } => *ar = Some(always_ram),
                QuantizationType::Binary { always_ram: ar } => *ar = Some(always_ram),
            }
        }
        self
    }

    pub fn always_ram(&self) -> bool {
        match &self.quant_type {
            Some(QuantizationType::Scalar { always_ram, .. })
            | Some(QuantizationType::Product { always_ram, .. })
            | Some(QuantizationType::Binary { always_ram }) => always_ram.unwrap_or(true),
            None => true,
        }
    }

    pub fn quantile_or_default(&self) -> f32 {
        match &self.quant_type {
            Some(QuantizationType::Scalar { quantile, .. }) => quantile.unwrap_or(0.99),
            _ => 0.99,
        }
    }

    pub fn validate(&self, dim: usize) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let Some(qt) = &self.quant_type else {
            return Err(VectorSearchError::InvalidConfig(
                "quantization enabled but quant_type is None".to_string(),
            ));
        };
        match qt {
            QuantizationType::Scalar { quantile, .. } => {
                if let Some(q) = quantile {
                    if !q.is_finite() || *q <= 0.0 || *q > 1.0 {
                        return Err(VectorSearchError::InvalidConfig(format!(
                            "scalar quantization quantile {q} must be in (0,1]"
                        )));
                    }
                }
            }
            QuantizationType::Product { compression, .. } => {
                let m = compression.pq_m(dim);
                if m == 0 || !dim.is_multiple_of(m) {
                    return Err(VectorSearchError::InvalidConfig(format!(
                        "product quantization dim {dim} not divisible by M={m} for compression {compression:?}"
                    )));
                }
            }
            QuantizationType::Binary { .. } => {}
        }
        Ok(())
    }

    pub fn quant_bytes_per_vector(&self, dim: usize) -> usize {
        if !self.enabled {
            return 0;
        }
        match &self.quant_type {
            Some(QuantizationType::Scalar { .. }) => dim,
            Some(QuantizationType::Binary { .. }) => dim.div_ceil(8),
            Some(QuantizationType::Product { compression, .. }) => compression.pq_m(dim),
            None => 0,
        }
    }

    pub fn is_product(&self) -> bool {
        matches!(self.quant_type, Some(QuantizationType::Product { .. }))
    }
}

/// IVF index configuration. All thresholds are evaluated per collection.
///
/// The IVF index is a derived structure: it can be built, dropped and rebuilt
/// at any time without affecting correctness (the exact scan stays the
/// source of truth). Every field has a conservative default so that automatic
/// promotion never happens unless explicitly enabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfConfig {
    /// Number of clusters; `None` = auto (`sqrt(live)` clamped to [1, 4096]).
    pub lists: Option<u32>,
    /// Minimum live points before an index is built.
    pub min_build_points: u64,
    /// Training sample cap and drift-check sample cap.
    pub sample_limit: usize,
    /// k-means iteration cap.
    pub kmeans_max_iter: u32,
    /// Drift ratio above which a rebuild is scheduled.
    pub drift_threshold: f64,
    /// Upserts accumulated before the next drift check.
    pub drift_check_interval: u64,
    /// Default nprobe when the query does not set one.
    pub default_nprobe: usize,
    /// Whether automatic index promotion is allowed.
    pub auto_promotion: bool,
    /// Upper bound for multi-round probe widening during filtered searches.
    /// The probe width still never exceeds the list count. `None` = capped
    /// by the list count only (historical single-doubling bound).
    #[serde(default)]
    pub max_probes: Option<usize>,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            lists: None,
            min_build_points: 100_000,
            sample_limit: 65_536,
            kmeans_max_iter: 64,
            drift_threshold: 0.10,
            drift_check_interval: 25_000,
            default_nprobe: 8,
            // Off until benchmarks justify turning it on.
            auto_promotion: false,
            max_probes: None,
        }
    }
}

impl IvfConfig {
    /// Validate the probe-widening bound. Kept separate from the HNSW
    /// validator so each tier's creation path checks only its own knobs.
    pub fn validate(&self) -> Result<()> {
        if self.max_probes.is_some_and(|p| p == 0) {
            return Err(VectorSearchError::InvalidConfig(
                "ivf max_probes must be >= 1 when set".to_string(),
            ));
        }
        Ok(())
    }

    /// Number of lists to train for `live` live points.
    pub fn effective_lists(&self, live: u64) -> u32 {
        match self.lists {
            Some(k) => k.max(1),
            None => ((live as f64).sqrt().round() as u32).clamp(1, 4096),
        }
    }

    /// Clamp nprobe to the number of lists and keep at least one probe.
    pub fn clamp_nprobe(&self, nprobe: Option<usize>, lists: usize) -> usize {
        let requested = nprobe.unwrap_or(self.default_nprobe);
        requested.clamp(1, lists.max(1))
    }

    /// Ceiling for multi-round probe widening: `max_probes` clamped to the
    /// list count (and at least one probe).
    pub fn effective_max_probes(&self, lists: usize) -> usize {
        let cap = self.max_probes.unwrap_or(lists);
        cap.clamp(1, lists.max(1))
    }

    pub fn with_max_probes(mut self, max_probes: usize) -> Self {
        self.max_probes = Some(max_probes);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub vector_size: usize,
    pub distance: DistanceMetric,
    pub index_type: Option<IndexType>,
    pub hnsw_config: Option<HnswConfig>,
    pub quantization_config: Option<QuantizationConfig>,
    pub replication_factor: Option<usize>,
    pub write_consistency_factor: Option<usize>,
    pub on_disk_payload: Option<bool>,
    pub shard_number: Option<usize>,
    pub ivf_config: Option<IvfConfig>,
}

impl CollectionConfig {
    pub fn new(vector_size: usize, distance: DistanceMetric) -> Self {
        Self {
            vector_size,
            distance,
            index_type: None,
            hnsw_config: None,
            quantization_config: None,
            replication_factor: None,
            write_consistency_factor: None,
            on_disk_payload: None,
            shard_number: None,
            ivf_config: None,
        }
    }

    pub fn with_ivf(mut self, ivf_config: IvfConfig) -> Self {
        self.index_type = Some(IndexType::IVF);
        self.ivf_config = Some(ivf_config);
        self
    }

    pub fn with_index_type(mut self, index_type: IndexType) -> Self {
        self.index_type = Some(index_type);
        self
    }

    pub fn with_hnsw(mut self, hnsw_config: HnswConfig) -> Self {
        self.index_type = Some(IndexType::HNSW);
        self.hnsw_config = Some(hnsw_config);
        self
    }

    pub fn with_quantization(mut self, quantization_config: QuantizationConfig) -> Self {
        self.quantization_config = Some(quantization_config);
        self
    }

    pub fn with_shard_number(mut self, shard_number: usize) -> Self {
        self.shard_number = Some(shard_number);
        self
    }

    pub fn with_on_disk_payload(mut self, on_disk_payload: bool) -> Self {
        self.on_disk_payload = Some(on_disk_payload);
        self
    }
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self::new(1536, DistanceMetric::Cosine)
    }
}

/// Index state exposed through [`CollectionInfo`].
///
/// Kind-agnostic: IVF reports list/nprobe fields, HNSW reports graph
/// parameters; fields that do not apply to the active kind stay zeroed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    /// 0 = exact scan, 1 = IVFFlat, 2 = HNSW.
    pub index_kind: u8,
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
pub struct VectorPoint {
    pub id: PointId,
    pub vector: Vec<f32>,
    pub payload: Option<Payload>,
}

impl VectorPoint {
    pub fn new(id: impl Into<PointId>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
            payload: None,
        }
    }

    pub fn with_payload(mut self, payload: Payload) -> Self {
        self.payload = Some(payload);
        self
    }

    pub fn with_payload_kv(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        let payload = self.payload.get_or_insert_with(HashMap::new);
        payload.insert(key.into(), value);
        self
    }

    pub fn dimension(&self) -> usize {
        self.vector.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorPoints {
    pub points: Vec<VectorPoint>,
}

impl VectorPoints {
    pub fn new(points: Vec<VectorPoint>) -> Self {
        Self { points }
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }
}

impl From<Vec<VectorPoint>> for VectorPoints {
    fn from(points: Vec<VectorPoint>) -> Self {
        Self::new(points)
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMode {
    TopK(usize),
    /// HNSW-style kNN search; `ef_search` is the only recall knob remote
    /// (Qdrant) backends honor — `nprobe` is ignored there.
    KNN {
        k: usize,
        ef_search: Option<usize>,
    },
    Range {
        radius: f32,
        max_results: Option<usize>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub vector: Vec<f32>,
    pub limit: usize,
    pub offset: Option<usize>,
    /// Similarity lower bound: results with `score >= score_threshold` are
    /// kept. Scores are "higher is better" on every backend.
    pub score_threshold: Option<f32>,
    pub filter: Option<VectorFilter>,
    pub with_payload: Option<bool>,
    pub with_vector: Option<bool>,
    /// Number of IVF cells probed per search. Consumed only when the local
    /// engine published an IVF index; HNSW and remote backends ignore it.
    /// For HNSW recall control use `SearchMode::KNN.ef_search` instead.
    pub nprobe: Option<usize>,
    pub search_mode: Option<SearchMode>,
    /// Returned payload field projection (include / exclude lists).
    /// `None` returns the whole payload. Applied after the payload read,
    /// so it only trims serialization cost, never filtering semantics.
    pub payload_selector: Option<PayloadSelector>,
}

impl SearchQuery {
    pub fn new(vector: Vec<f32>, limit: usize) -> Self {
        Self {
            vector,
            limit,
            offset: None,
            score_threshold: None,
            filter: None,
            with_payload: Some(true),
            with_vector: None,
            nprobe: None,
            search_mode: None,
            payload_selector: None,
        }
    }

    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn with_score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }

    pub fn with_filter(mut self, filter: VectorFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_payload(mut self, with_payload: bool) -> Self {
        self.with_payload = Some(with_payload);
        self
    }

    pub fn with_vector(mut self, with_vector: bool) -> Self {
        self.with_vector = Some(with_vector);
        self
    }

    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = Some(nprobe);
        self
    }

    /// Restrict the returned payload to the given fields (include list).
    pub fn with_payload_include(mut self, fields: Vec<String>) -> Self {
        let mut sel = self.payload_selector.take().unwrap_or_default();
        sel.include = Some(fields);
        self.payload_selector = Some(sel);
        self
    }

    /// Set an explicit payload projection for the search results.
    pub fn with_payload_selector(mut self, selector: PayloadSelector) -> Self {
        self.payload_selector = Some(selector);
        self
    }

    /// Remove the given fields from the returned payload (exclude list).
    pub fn with_payload_exclude(mut self, fields: Vec<String>) -> Self {
        let mut sel = self.payload_selector.take().unwrap_or_default();
        sel.exclude = Some(fields);
        self.payload_selector = Some(sel);
        self
    }

    pub fn effective_limit(&self) -> usize {
        match &self.search_mode {
            Some(SearchMode::Range {
                max_results: Some(max),
                ..
            }) => *max,
            Some(SearchMode::TopK(k)) => *k,
            Some(SearchMode::KNN { k, .. }) => *k,
            _ => self.limit,
        }
    }

    pub fn hnsw_ef(&self) -> Option<usize> {
        match &self.search_mode {
            Some(SearchMode::KNN { ef_search, .. }) => *ef_search,
            _ => None,
        }
    }

    pub fn score_threshold(&self) -> Option<f32> {
        match &self.search_mode {
            Some(SearchMode::Range { radius, .. }) => Some(*radius),
            _ => None,
        }
    }

    pub fn with_search_mode(mut self, mode: SearchMode) -> Self {
        self.search_mode = Some(mode);
        self
    }

    pub fn with_knn(mut self, k: usize, ef_search: Option<usize>) -> Self {
        self.search_mode = Some(SearchMode::KNN { k, ef_search });
        self.limit = k;
        self
    }

    pub fn with_range(mut self, radius: f32, max_results: Option<usize>) -> Self {
        self.search_mode = Some(SearchMode::Range {
            radius,
            max_results,
        });
        self.score_threshold = Some(radius);
        if let Some(max) = max_results {
            self.limit = max;
        }
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: PointId,
    pub score: f32,
    pub payload: Option<Payload>,
    pub vector: Option<Vec<f32>>,
}

impl SearchResult {
    pub fn new(id: impl Into<PointId>, score: f32) -> Self {
        Self {
            id: id.into(),
            score,
            payload: None,
            vector: None,
        }
    }

    pub fn with_payload(mut self, payload: Payload) -> Self {
        self.payload = Some(payload);
        self
    }

    pub fn with_vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(vector);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    pub results: Vec<SearchResult>,
    pub total: Option<u64>,
}

impl SearchResults {
    pub fn new(results: Vec<SearchResult>) -> Self {
        let total = Some(results.len() as u64);
        Self { results, total }
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    pub fn len(&self) -> usize {
        self.results.len()
    }
}

impl From<Vec<SearchResult>> for SearchResults {
    fn from(results: Vec<SearchResult>) -> Self {
        Self::new(results)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSearchQuery {
    pub queries: Vec<SearchQuery>,
}

impl BatchSearchQuery {
    pub fn new(queries: Vec<SearchQuery>) -> Self {
        Self { queries }
    }
}

impl From<Vec<SearchQuery>> for BatchSearchQuery {
    fn from(queries: Vec<SearchQuery>) -> Self {
        Self::new(queries)
    }
}

/// Validate that a distance metric is supported by all backends.
///
/// This is the canonical validation used by both the API layer and
/// sync layer to reject unsupported metrics at index-creation time.
pub fn validate_distance_metric(distance: DistanceMetric) -> std::result::Result<(), String> {
    if matches!(
        distance,
        DistanceMetric::Cosine
            | DistanceMetric::Euclid
            | DistanceMetric::Dot
            | DistanceMetric::Manhattan
    ) {
        Ok(())
    } else {
        Err(format!(
            "distance metric {distance:?} is not supported; supported metrics: Cosine, Euclid, Dot, Manhattan"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_id_from_u64() {
        let id: PointId = 42u64.into();
        assert_eq!(id, PointId::Num(42));
    }

    #[test]
    fn test_point_id_from_string_numeric() {
        let id = PointId::from("123");
        assert_eq!(id, PointId::Num(123));
    }

    #[test]
    fn test_point_id_from_string_uuid() {
        let id = PointId::from("uuid-abc");
        assert_eq!(id, PointId::Uuid("uuid-abc".into()));
    }

    #[test]
    fn test_point_id_from_str() {
        let id = PointId::from("456");
        assert_eq!(id, PointId::Num(456));
    }

    #[test]
    fn test_point_id_from_str_uuid() {
        let id = PointId::from("550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(
            id,
            PointId::Uuid("550e8400-e29b-41d4-a716-446655440000".into())
        );
    }

    #[test]
    fn test_point_id_display_num() {
        let id = PointId::Num(42);
        assert_eq!(format!("{}", id), "42");
    }

    #[test]
    fn test_point_id_display_uuid() {
        let id = PointId::Uuid("abc".into());
        assert_eq!(format!("{}", id), "abc");
    }

    #[test]
    fn test_point_id_serialize_deserialize() {
        let id = PointId::Num(42);
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: PointId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn test_point_id_serialize_deserialize_uuid() {
        let id = PointId::Uuid("test-uuid".into());
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: PointId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn test_distance_metric_supported() {
        assert!(DistanceMetric::Cosine.is_supported_by_qdrant());
        assert!(DistanceMetric::Euclid.is_supported_by_qdrant());
        assert!(DistanceMetric::Dot.is_supported_by_qdrant());
        assert!(DistanceMetric::Manhattan.is_supported_by_qdrant());
    }

    #[test]
    fn test_distance_metric_custom_implementation() {
        assert!(!DistanceMetric::Cosine.requires_custom_implementation());
        assert!(!DistanceMetric::Euclid.requires_custom_implementation());
        assert!(!DistanceMetric::Dot.requires_custom_implementation());
        assert!(!DistanceMetric::Manhattan.requires_custom_implementation());
    }

    #[test]
    fn test_hnsw_config_default() {
        let cfg = HnswConfig::default();
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.ef_construct, 100);
        assert!(cfg.full_scan_threshold.is_none());
    }

    #[test]
    fn test_hnsw_config_new() {
        let cfg = HnswConfig::new(32, 200);
        assert_eq!(cfg.m, 32);
        assert_eq!(cfg.ef_construct, 200);
    }

    #[test]
    fn test_hnsw_config_builder() {
        let cfg = HnswConfig::new(16, 100)
            .with_full_scan_threshold(10000)
            .with_max_indexing_threads(4)
            .with_on_disk(true)
            .with_payload_m(8);
        assert_eq!(cfg.full_scan_threshold, Some(10000));
        assert_eq!(cfg.max_indexing_threads, Some(4));
        assert_eq!(cfg.on_disk, Some(true));
        assert_eq!(cfg.payload_m, Some(8));
    }

    #[test]
    fn test_hnsw_config_validate() {
        assert!(HnswConfig::default().validate().is_ok());
        // Boundary: ef_construct == max(2m, 4) is accepted.
        assert!(HnswConfig::new(8, 16).validate().is_ok());
        assert!(HnswConfig::new(2, 4).validate().is_ok());
        assert!(HnswConfig::new(1, 4).validate().is_ok());

        let err = HnswConfig::new(16, 20).validate().unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidConfig(_)));
        assert!(HnswConfig::new(2, 3).validate().is_err());
    }

    #[test]
    fn test_quantization_config_disabled() {
        let cfg = QuantizationConfig::disabled();
        assert!(!cfg.enabled);
        assert!(cfg.quant_type.is_none());
    }

    #[test]
    fn test_quantization_config_scalar() {
        let cfg = QuantizationConfig::scalar(0.99);
        assert!(cfg.enabled);
        match cfg.quant_type {
            Some(QuantizationType::Scalar { .. }) => {}
            _ => panic!("expected Scalar"),
        }
    }

    #[test]
    fn test_quantization_config_product() {
        let cfg = QuantizationConfig::product(CompressionRatio::X8);
        assert!(cfg.enabled);
        match cfg.quant_type {
            Some(QuantizationType::Product { compression, .. }) => {
                assert!(matches!(compression, CompressionRatio::X8));
            }
            _ => panic!("expected Product"),
        }
    }

    #[test]
    fn test_quantization_config_binary() {
        let cfg = QuantizationConfig::binary();
        assert!(cfg.enabled);
        match cfg.quant_type {
            Some(QuantizationType::Binary { .. }) => {}
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn test_quantization_config_with_always_ram() {
        let cfg = QuantizationConfig::scalar(0.5).with_always_ram(false);
        match cfg.quant_type {
            Some(QuantizationType::Scalar { always_ram, .. }) => {
                assert_eq!(always_ram, Some(false));
            }
            _ => panic!("expected Scalar"),
        }
    }

    #[test]
    fn test_collection_config_new() {
        let cfg = CollectionConfig::new(768, DistanceMetric::Cosine);
        assert_eq!(cfg.vector_size, 768);
        assert_eq!(cfg.distance, DistanceMetric::Cosine);
    }

    #[test]
    fn test_collection_config_with_index_type() {
        let cfg =
            CollectionConfig::new(384, DistanceMetric::Euclid).with_index_type(IndexType::FLAT);
        assert_eq!(cfg.index_type, Some(IndexType::FLAT));
    }

    #[test]
    fn test_collection_config_with_hnsw() {
        let hnsw = HnswConfig::new(32, 200);
        let cfg = CollectionConfig::new(128, DistanceMetric::Dot).with_hnsw(hnsw);
        assert_eq!(cfg.index_type, Some(IndexType::HNSW));
        assert!(cfg.hnsw_config.is_some());
    }

    #[test]
    fn test_collection_config_with_quantization() {
        let q = QuantizationConfig::scalar(0.99);
        let cfg = CollectionConfig::new(1536, DistanceMetric::Cosine).with_quantization(q);
        assert!(cfg.quantization_config.is_some());
    }

    #[test]
    fn test_collection_config_with_shard_number() {
        let cfg = CollectionConfig::new(768, DistanceMetric::Cosine).with_shard_number(2);
        assert_eq!(cfg.shard_number, Some(2));
    }

    #[test]
    fn test_collection_config_with_on_disk_payload() {
        let cfg = CollectionConfig::new(768, DistanceMetric::Cosine).with_on_disk_payload(true);
        assert_eq!(cfg.on_disk_payload, Some(true));
    }

    #[test]
    fn test_collection_config_default() {
        let cfg = CollectionConfig::default();
        assert_eq!(cfg.vector_size, 1536);
        assert_eq!(cfg.distance, DistanceMetric::Cosine);
    }

    #[test]
    fn test_payload_schema_type_as_str() {
        assert_eq!(PayloadSchemaType::Keyword.as_str(), "keyword");
        assert_eq!(PayloadSchemaType::Integer.as_str(), "integer");
        assert_eq!(PayloadSchemaType::Float.as_str(), "float");
        assert_eq!(PayloadSchemaType::Text.as_str(), "text");
        assert_eq!(PayloadSchemaType::Bool.as_str(), "bool");
        assert_eq!(PayloadSchemaType::Geo.as_str(), "geo");
        assert_eq!(PayloadSchemaType::Datetime.as_str(), "datetime");
    }

    #[test]
    fn test_health_status_healthy() {
        let h = HealthStatus::healthy("test-engine", "1.0");
        assert!(h.is_healthy);
        assert_eq!(h.engine_name, "test-engine");
        assert_eq!(h.engine_version, "1.0");
        assert!(h.message.is_none());
    }

    #[test]
    fn test_health_status_unhealthy() {
        let h = HealthStatus::unhealthy("test-engine", "1.0", "not ready");
        assert!(!h.is_healthy);
        assert_eq!(h.message, Some("not ready".to_string()));
    }

    #[test]
    fn test_index_type_default() {
        assert_eq!(IndexType::default(), IndexType::HNSW);
    }

    #[test]
    fn test_distance_metric_default() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
    }

    #[test]
    fn test_collection_status_debug() {
        assert_eq!(format!("{:?}", CollectionStatus::Green), "Green");
        assert_eq!(format!("{:?}", CollectionStatus::Yellow), "Yellow");
        assert_eq!(format!("{:?}", CollectionStatus::Red), "Red");
        assert_eq!(format!("{:?}", CollectionStatus::Grey), "Grey");
    }

    #[test]
    fn test_geo_point_new() {
        let p = GeoPoint::new(1.0, 2.0);
        assert!((p.lat - 1.0).abs() < f64::EPSILON);
        assert!((p.lon - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_geo_radius_new() {
        let center = GeoPoint::new(0.0, 0.0);
        let r = GeoRadius::new(center, 100.0);
        assert!((r.radius - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_geo_bounding_box_new() {
        let tl = GeoPoint::new(1.0, 2.0);
        let br = GeoPoint::new(3.0, 4.0);
        let bbox = GeoBoundingBox::new(tl, br);
        assert!((bbox.top_left.lat - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_values_count_condition_default() {
        let v = ValuesCountCondition::default();
        assert!(v.gt.is_none());
        assert!(v.gte.is_none());
        assert!(v.lt.is_none());
        assert!(v.lte.is_none());
    }

    #[test]
    fn test_values_count_condition_builder() {
        let v = ValuesCountCondition::new().gt(5).lt(10);
        assert_eq!(v.gt, Some(5));
        assert_eq!(v.lt, Some(10));
        assert!(v.gte.is_none());
        assert!(v.lte.is_none());
    }

    #[test]
    fn test_range_condition_default() {
        let r = RangeCondition::default();
        assert!(r.gt.is_none());
        assert!(r.gte.is_none());
        assert!(r.lt.is_none());
        assert!(r.lte.is_none());
    }

    #[test]
    fn test_range_condition_builder() {
        let r = RangeCondition::new().gte(1.5).lt(10.0);
        assert_eq!(r.gte, Some(1.5));
        assert_eq!(r.lt, Some(10.0));
        assert!(r.gt.is_none());
        assert!(r.lte.is_none());
    }

    #[test]
    fn test_vector_filter_default() {
        let f = VectorFilter::default();
        assert!(f.must.is_none());
        assert!(f.must_not.is_none());
        assert!(f.should.is_none());
        assert!(f.min_should.is_none());
    }

    #[test]
    fn test_vector_filter_must() {
        let f = VectorFilter::new()
            .must(FilterCondition::match_value("color", "red"))
            .must(FilterCondition::match_value("size", "large"));
        assert_eq!(f.must.as_ref().map(|v| v.len()), Some(2));
    }

    #[test]
    fn test_vector_filter_must_not() {
        let f = VectorFilter::new().must_not(FilterCondition::is_null("deleted"));
        assert_eq!(f.must_not.as_ref().map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_vector_filter_should() {
        let f = VectorFilter::new()
            .should(FilterCondition::match_value("tag", "a"))
            .should(FilterCondition::match_value("tag", "b"));
        assert_eq!(f.should.as_ref().map(|v| v.len()), Some(2));
    }

    #[test]
    fn test_filter_condition_match_value() {
        let c = FilterCondition::match_value("color", "blue");
        assert_eq!(c.field, "color");
        match c.condition {
            ConditionType::Match { value } => assert_eq!(value, "blue"),
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn test_filter_condition_match_any() {
        let values = vec![serde_json::json!("a"), serde_json::json!("b")];
        let c = FilterCondition::match_any("tags", values);
        assert_eq!(c.field, "tags");
    }

    #[test]
    fn test_filter_condition_range() {
        let range = RangeCondition::new().gt(10.0);
        let c = FilterCondition::range("price", range);
        assert_eq!(c.field, "price");
    }

    #[test]
    fn test_filter_condition_is_empty() {
        let c = FilterCondition::is_empty("description");
        assert_eq!(c.field, "description");
        assert!(matches!(c.condition, ConditionType::IsEmpty));
    }

    #[test]
    fn test_filter_condition_is_null() {
        let c = FilterCondition::is_null("deleted_at");
        assert!(matches!(c.condition, ConditionType::IsNull));
    }

    #[test]
    fn test_filter_condition_has_id() {
        let ids = vec!["1".to_string(), "2".to_string()];
        let c = FilterCondition::has_id(ids);
        assert_eq!(c.field, "_id");
    }

    #[test]
    fn test_filter_condition_geo_radius() {
        let center = GeoPoint::new(1.0, 2.0);
        let radius = GeoRadius::new(center, 500.0);
        let c = FilterCondition::geo_radius("location", radius);
        assert_eq!(c.field, "location");
    }

    #[test]
    fn test_filter_condition_geo_bounding_box() {
        let tl = GeoPoint::new(1.0, 2.0);
        let br = GeoPoint::new(3.0, 4.0);
        let bbox = GeoBoundingBox::new(tl, br);
        let c = FilterCondition::geo_bounding_box("location", bbox);
        assert!(matches!(c.condition, ConditionType::GeoBoundingBox(_)));
    }

    #[test]
    fn test_filter_condition_values_count() {
        let count = ValuesCountCondition::new().gt(2);
        let c = FilterCondition::values_count("tags", count);
        assert!(matches!(c.condition, ConditionType::ValuesCount(_)));
    }

    #[test]
    fn test_filter_condition_contains() {
        let c = FilterCondition::contains("title", "rust");
        assert_eq!(c.field, "title");
        match c.condition {
            ConditionType::Contains { value } => assert_eq!(value, "rust"),
            _ => panic!("expected Contains"),
        }
    }

    #[test]
    fn test_payload_selector_include() {
        let sel = PayloadSelector::include(vec!["a".into(), "b".into()]);
        assert_eq!(sel.include.as_ref().map(|v| v.len()), Some(2));
        assert!(sel.exclude.is_none());
    }

    #[test]
    fn test_payload_selector_exclude() {
        let sel = PayloadSelector::exclude(vec!["c".into()]);
        assert_eq!(sel.exclude.as_ref().map(|v| v.len()), Some(1));
        assert!(sel.include.is_none());
    }

    #[test]
    fn test_payload_selector_all() {
        let sel = PayloadSelector::all();
        assert!(sel.include.is_none());
        assert!(sel.exclude.is_none());
    }

    #[test]
    fn test_min_should_condition() {
        let condition = FilterCondition::match_value("field", "val");
        let ms = MinShouldCondition {
            conditions: vec![condition],
            min_count: 1,
        };
        assert_eq!(ms.min_count, 1);
        assert_eq!(ms.conditions.len(), 1);
    }

    #[test]
    fn test_vector_point_new() {
        let p = VectorPoint::new(42u64, vec![1.0, 2.0, 3.0]);
        assert_eq!(p.id, PointId::Num(42));
        assert_eq!(p.vector, vec![1.0, 2.0, 3.0]);
        assert!(p.payload.is_none());
    }

    #[test]
    fn test_vector_point_with_payload() {
        let mut payload = HashMap::new();
        payload.insert("key".into(), serde_json::json!("val"));
        let p = VectorPoint::new("1", vec![1.0]).with_payload(payload);
        assert!(p.payload.is_some());
    }

    #[test]
    fn test_vector_point_with_payload_kv() {
        let p = VectorPoint::new(1u64, vec![1.0])
            .with_payload_kv("color", serde_json::json!("red"))
            .with_payload_kv("size", serde_json::json!(42));
        let payload = p.payload.expect("payload expected");
        assert_eq!(payload.get("color").and_then(|v| v.as_str()), Some("red"));
        assert_eq!(payload.get("size").and_then(|v| v.as_i64()), Some(42));
    }

    #[test]
    fn test_vector_point_dimension() {
        let p = VectorPoint::new(1u64, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(p.dimension(), 4);
    }

    #[test]
    fn test_vector_point_dimension_empty() {
        let p = VectorPoint::new(1u64, vec![]);
        assert_eq!(p.dimension(), 0);
    }

    #[test]
    fn test_vector_points_new() {
        let points = vec![VectorPoint::new(1u64, vec![1.0])];
        let vp = VectorPoints::new(points);
        assert_eq!(vp.len(), 1);
        assert!(!vp.is_empty());
    }

    #[test]
    fn test_vector_points_empty() {
        let vp = VectorPoints::new(vec![]);
        assert!(vp.is_empty());
        assert_eq!(vp.len(), 0);
    }

    #[test]
    fn test_vector_points_from() {
        let points = vec![VectorPoint::new(1u64, vec![1.0])];
        let vp: VectorPoints = points.into();
        assert_eq!(vp.len(), 1);
    }

    #[test]
    fn test_upsert_result() {
        let r = UpsertResult {
            operation_id: Some(123),
            status: UpsertStatus::Completed,
        };
        assert_eq!(r.operation_id, Some(123));
        assert_eq!(r.status, UpsertStatus::Completed);
    }

    #[test]
    fn test_delete_result() {
        let r = DeleteResult {
            operation_id: None,
            deleted_count: 5,
        };
        assert!(r.operation_id.is_none());
        assert_eq!(r.deleted_count, 5);
    }

    #[test]
    fn test_upsert_status_debug() {
        assert_eq!(format!("{:?}", UpsertStatus::Completed), "Completed");
        assert_eq!(format!("{:?}", UpsertStatus::Acknowledged), "Acknowledged");
    }

    #[test]
    fn test_search_query_new() {
        let q = SearchQuery::new(vec![1.0, 2.0], 10);
        assert_eq!(q.vector, vec![1.0, 2.0]);
        assert_eq!(q.limit, 10);
        assert_eq!(q.with_payload, Some(true));
        assert!(q.with_vector.is_none());
    }

    #[test]
    fn test_search_query_with_offset() {
        let q = SearchQuery::new(vec![1.0], 10).with_offset(5);
        assert_eq!(q.offset, Some(5));
    }

    #[test]
    fn test_search_query_with_score_threshold() {
        let q = SearchQuery::new(vec![1.0], 10).with_score_threshold(0.5);
        assert_eq!(q.score_threshold, Some(0.5));
    }

    #[test]
    fn test_search_query_with_payload() {
        let q = SearchQuery::new(vec![1.0], 10).with_payload(false);
        assert_eq!(q.with_payload, Some(false));
    }

    #[test]
    fn test_search_query_with_vector() {
        let q = SearchQuery::new(vec![1.0], 10).with_vector(true);
        assert_eq!(q.with_vector, Some(true));
    }

    #[test]
    fn test_search_query_with_nprobe() {
        let q = SearchQuery::new(vec![1.0], 10).with_nprobe(64);
        assert_eq!(q.nprobe, Some(64));
    }

    #[test]
    fn test_search_query_effective_limit_default() {
        let q = SearchQuery::new(vec![1.0], 10);
        assert_eq!(q.effective_limit(), 10);
    }

    #[test]
    fn test_search_query_effective_limit_topk() {
        let q = SearchQuery::new(vec![1.0], 10).with_search_mode(SearchMode::TopK(5));
        assert_eq!(q.effective_limit(), 5);
    }

    #[test]
    fn test_search_query_effective_limit_knn() {
        let q = SearchQuery::new(vec![1.0], 10).with_search_mode(SearchMode::KNN {
            k: 20,
            ef_search: Some(100),
        });
        assert_eq!(q.effective_limit(), 20);
    }

    #[test]
    fn test_search_query_effective_limit_range() {
        let q = SearchQuery::new(vec![1.0], 10).with_search_mode(SearchMode::Range {
            radius: 0.5,
            max_results: Some(30),
        });
        assert_eq!(q.effective_limit(), 30);
    }

    #[test]
    fn test_search_query_hnsw_ef_default() {
        let q = SearchQuery::new(vec![1.0], 10);
        assert!(q.hnsw_ef().is_none());
    }

    #[test]
    fn test_search_query_hnsw_ef_knn() {
        let q = SearchQuery::new(vec![1.0], 10).with_knn(5, Some(128));
        assert_eq!(q.hnsw_ef(), Some(128));
    }

    #[test]
    fn test_search_query_hnsw_ef_topk() {
        let q = SearchQuery::new(vec![1.0], 10).with_search_mode(SearchMode::TopK(5));
        assert!(q.hnsw_ef().is_none());
    }

    #[test]
    fn test_search_query_with_knn_sets_limit() {
        let q = SearchQuery::new(vec![1.0], 10).with_knn(42, None);
        assert_eq!(q.limit, 42);
        assert!(matches!(q.search_mode, Some(SearchMode::KNN { k: 42, .. })));
    }

    #[test]
    fn test_search_query_with_range_sets_limit() {
        let q = SearchQuery::new(vec![1.0], 10).with_range(0.3, Some(25));
        assert_eq!(q.limit, 25);
        assert_eq!(q.score_threshold, Some(0.3));
    }

    #[test]
    fn test_search_query_with_range_no_max_keeps_limit() {
        let q = SearchQuery::new(vec![1.0], 10).with_range(0.3, None);
        assert_eq!(q.limit, 10);
    }

    #[test]
    fn test_search_result_new() {
        let r = SearchResult::new(42u64, 0.95);
        assert_eq!(r.id, PointId::Num(42));
        assert!((r.score - 0.95).abs() < f32::EPSILON);
        assert!(r.payload.is_none());
        assert!(r.vector.is_none());
    }

    #[test]
    fn test_search_result_with_payload() {
        let mut payload = HashMap::new();
        payload.insert("key".into(), serde_json::json!("val"));
        let r = SearchResult::new("1", 0.5).with_payload(payload);
        assert!(r.payload.is_some());
    }

    #[test]
    fn test_search_result_with_vector() {
        let r = SearchResult::new("id", 0.1).with_vector(vec![1.0, 2.0]);
        assert_eq!(r.vector, Some(vec![1.0, 2.0]));
    }

    #[test]
    fn test_search_results_empty() {
        let r = SearchResults::new(vec![]);
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn test_search_results_non_empty() {
        let results = vec![SearchResult::new(1u64, 0.9), SearchResult::new(2u64, 0.8)];
        let r = SearchResults::new(results);
        assert!(!r.is_empty());
        assert_eq!(r.len(), 2);
        assert_eq!(r.total, Some(2));
    }

    #[test]
    fn test_search_results_from_vec() {
        let results = vec![SearchResult::new(1u64, 0.9)];
        let r: SearchResults = results.into();
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_batch_search_query_new() {
        let q = SearchQuery::new(vec![1.0], 5);
        let batch = BatchSearchQuery::new(vec![q]);
        assert_eq!(batch.queries.len(), 1);
    }

    #[test]
    fn test_batch_search_query_from() {
        let q = SearchQuery::new(vec![1.0], 5);
        let batch: BatchSearchQuery = vec![q].into();
        assert_eq!(batch.queries.len(), 1);
    }

    #[test]
    fn test_search_mode_topk() {
        let mode = SearchMode::TopK(10);
        match mode {
            SearchMode::TopK(k) => assert_eq!(k, 10),
            _ => panic!("expected TopK"),
        }
    }

    #[test]
    fn test_search_mode_knn() {
        let mode = SearchMode::KNN {
            k: 20,
            ef_search: Some(200),
        };
        match mode {
            SearchMode::KNN { k, ef_search } => {
                assert_eq!(k, 20);
                assert_eq!(ef_search, Some(200));
            }
            _ => panic!("expected KNN"),
        }
    }

    #[test]
    fn test_search_mode_range() {
        let mode = SearchMode::Range {
            radius: 0.7,
            max_results: Some(50),
        };
        match mode {
            SearchMode::Range {
                radius,
                max_results,
            } => {
                assert!((radius - 0.7).abs() < f32::EPSILON);
                assert_eq!(max_results, Some(50));
            }
            _ => panic!("expected Range"),
        }
    }

    #[test]
    fn test_search_query_score_threshold_range_mode() {
        let q = SearchQuery::new(vec![1.0], 10).with_range(0.5, None);
        assert_eq!(q.score_threshold(), Some(0.5));
    }

    #[test]
    fn test_search_query_score_threshold_default() {
        let q = SearchQuery::new(vec![1.0], 10);
        assert!(q.score_threshold().is_none());
    }

    #[test]
    fn test_scan_limit_config_defaults() {
        let hnsw = HnswConfig::default();
        assert_eq!(hnsw.iterative_max_rounds, None);
        assert_eq!(hnsw.max_scan_tuples, None);
        let ivf = IvfConfig::default();
        assert_eq!(ivf.max_probes, None);
        // Missing fields fall back to engine defaults on deserialization.
        let hnsw: HnswConfig = serde_json::from_str(r#"{"m": 16, "ef_construct": 100}"#).unwrap();
        assert_eq!(hnsw.iterative_max_rounds, None);
        assert_eq!(hnsw.max_scan_tuples, None);
        assert_eq!(hnsw.ef_search, 40);
        let ivf: IvfConfig = serde_json::from_str(
            r#"{
                "min_build_points": 1,
                "sample_limit": 10,
                "kmeans_max_iter": 1,
                "drift_threshold": 0.1,
                "drift_check_interval": 1,
                "default_nprobe": 4,
                "auto_promotion": false
            }"#,
        )
        .unwrap();
        assert_eq!(ivf.max_probes, None);
        assert_eq!(ivf.default_nprobe, 4);
    }

    #[test]
    fn test_hnsw_config_validate_scan_limits() {
        let cfg = HnswConfig::new(16, 100)
            .with_iterative_max_rounds(1)
            .with_max_scan_tuples(1);
        assert!(cfg.validate().is_ok());

        let err = HnswConfig::new(16, 100)
            .with_iterative_max_rounds(0)
            .validate()
            .unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidConfig(_)));

        let err = HnswConfig::new(16, 100)
            .with_max_scan_tuples(0)
            .validate()
            .unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidConfig(_)));
    }

    #[test]
    fn test_ivf_config_validate_and_effective_max_probes() {
        assert!(IvfConfig::default().validate().is_ok());
        assert!(IvfConfig::default().with_max_probes(1).validate().is_ok());
        let err = IvfConfig::default()
            .with_max_probes(0)
            .validate()
            .unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidConfig(_)));

        // The cap never exceeds the list count and never drops below one.
        let with_cap = |max_probes| IvfConfig {
            max_probes,
            ..IvfConfig::default()
        };
        assert_eq!(with_cap(Some(2)).effective_max_probes(8), 2);
        assert_eq!(with_cap(Some(16)).effective_max_probes(8), 8);
        assert_eq!(with_cap(None).effective_max_probes(8), 8);
        assert_eq!(with_cap(None).effective_max_probes(0), 1);
    }
}
