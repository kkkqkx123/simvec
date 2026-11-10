use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};

pub type CollectionName = String;

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub quant_type: QuantizationType,
}

impl QuantizationConfig {
    pub fn scalar(quantile: f32) -> Self {
        Self {
            quant_type: QuantizationType::Scalar {
                quantile: Some(quantile),
                always_ram: Some(true),
            },
        }
    }

    pub fn product(compression: CompressionRatio) -> Self {
        Self {
            quant_type: QuantizationType::Product {
                compression,
                always_ram: Some(true),
            },
        }
    }

    pub fn binary() -> Self {
        Self {
            quant_type: QuantizationType::Binary {
                always_ram: Some(true),
            },
        }
    }

    pub fn with_always_ram(mut self, always_ram: bool) -> Self {
        match &mut self.quant_type {
            QuantizationType::Scalar { always_ram: ar, .. }
            | QuantizationType::Product { always_ram: ar, .. }
            | QuantizationType::Binary { always_ram: ar } => *ar = Some(always_ram),
        }
        self
    }

    pub fn always_ram(&self) -> bool {
        match &self.quant_type {
            QuantizationType::Scalar { always_ram, .. }
            | QuantizationType::Product { always_ram, .. }
            | QuantizationType::Binary { always_ram } => always_ram.unwrap_or(true),
        }
    }

    pub fn quantile_or_default(&self) -> f32 {
        match &self.quant_type {
            QuantizationType::Scalar { quantile, .. } => quantile.unwrap_or(0.99),
            _ => 0.99,
        }
    }

    pub fn validate(&self, dim: usize) -> Result<()> {
        match &self.quant_type {
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
        match &self.quant_type {
            QuantizationType::Scalar { .. } => dim,
            QuantizationType::Binary { .. } => dim.div_ceil(8),
            QuantizationType::Product { compression, .. } => compression.pq_m(dim),
        }
    }

    pub fn is_product(&self) -> bool {
        matches!(self.quant_type, QuantizationType::Product { .. })
    }
}

/// IVF index configuration. All thresholds are evaluated per collection.
///
/// The IVF index is a derived structure: it can be built, dropped and rebuilt
/// at any time without affecting correctness (the exact scan stays the
/// source of truth). Every field has a conservative default so that automatic
/// promotion never happens unless explicitly enabled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

/// Index tier selection for a collection.
///
/// The three variants are mutually exclusive: selecting one tier resets
/// any configuration belonging to another tier, so cross-tier residue is
/// unrepresentable. `Ivf(None)` means the IVF tier is selected but not
/// yet trained, which behaves as exact scan until a build publishes lists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndexConfig {
    Flat,
    Hnsw(HnswConfig),
    Ivf(Option<IvfConfig>),
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self::Hnsw(HnswConfig::default())
    }
}

impl IndexConfig {
    pub fn index_type(&self) -> IndexType {
        match self {
            Self::Flat => IndexType::FLAT,
            Self::Hnsw(_) => IndexType::HNSW,
            Self::Ivf(_) => IndexType::IVF,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub vector_size: usize,
    pub distance: DistanceMetric,
    pub index: IndexConfig,
    pub quantization_config: Option<QuantizationConfig>,
    pub replication_factor: Option<usize>,
    pub write_consistency_factor: Option<usize>,
    pub on_disk_payload: Option<bool>,
    pub shard_number: Option<usize>,
    /// Optional admission cap on live points. When set, an upsert that would
    /// grow the live set past this limit fails with
    /// [`VectorSearchError::CapacityExceeded`](crate::error::VectorSearchError::CapacityExceeded)
    /// instead of silently allocating unbounded index memory. `None` = no cap.
    #[serde(default)]
    pub max_points: Option<u64>,
}

impl CollectionConfig {
    pub fn new(vector_size: usize, distance: DistanceMetric) -> Self {
        Self {
            vector_size,
            distance,
            index: IndexConfig::default(),
            quantization_config: None,
            replication_factor: None,
            write_consistency_factor: None,
            on_disk_payload: None,
            shard_number: None,
            max_points: None,
        }
    }

    pub fn with_ivf(mut self, ivf_config: IvfConfig) -> Self {
        self.index = IndexConfig::Ivf(Some(ivf_config));
        self
    }

    pub fn with_index_type(mut self, index_type: IndexType) -> Self {
        self.index = match index_type {
            IndexType::HNSW => IndexConfig::Hnsw(HnswConfig::default()),
            IndexType::IVF => IndexConfig::Ivf(None),
            IndexType::FLAT => IndexConfig::Flat,
        };
        self
    }

    pub fn with_hnsw(mut self, hnsw_config: HnswConfig) -> Self {
        self.index = IndexConfig::Hnsw(hnsw_config);
        self
    }

    pub fn index_type(&self) -> IndexType {
        self.index.index_type()
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

    /// Set the optional live-point admission cap.
    pub fn with_max_points(mut self, max_points: u64) -> Self {
        self.max_points = Some(max_points);
        self
    }
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self::new(1536, DistanceMetric::Cosine)
    }
}
