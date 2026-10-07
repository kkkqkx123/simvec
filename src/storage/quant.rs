//! Quantized vector storage (`quant.bin` + `quant_meta.bin`).
//!
//! Layout per collection:
//! - `quant.bin`dense row-major quantized codes, segmented mmap like
//!   `vectors.bin`. Bytes per vector depends on the quantization type:
//!   Scalar: `dim` bytes (u8), Binary: `(dim+7)/8` bytes, Product: `M` bytes.
//! - `quant_meta.bin` holds quantization parameters: scalar `min/max/scale` or
//!   product codebook `M*256*subdim f32`. Persisted with `[magic 4][version 2B][crc 4B][postcard]`.
//!
//! Vectors stay authoritative in `vectors.bin` (f32). `quant.bin` is a derived
//! structure that can be rebuilt from `vectors.bin` at any time, exactly like
//! the ANN indexes. `QuantStore` is `None` when quantization is disabled.
//! Incremental writes quantize on the fly; product quantization requires a
//! trained codebook first, otherwise upserts fall back to no quant (and
//! search uses exact path until the codebook is built).

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;
use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};

use crate::error::{Result, VectorSearchError};
use crate::types::{DistanceMetric, QuantizationConfig, QuantizationType};

const QUANT_MAGIC: [u8; 4] = *b"VQMT";
const QUANT_VERSION: u16 = 1;
const QUANT_FILE: &str = "quant.bin";
const QUANT_META_FILE: &str = "quant_meta.bin";

/// Serialized quantization metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct QuantMeta {
    pub dim: usize,
    pub distance: DistanceMetric,
    pub config: QuantizationConfig,
    /// Scalar global range.
    pub scalar_min: Option<f32>,
    pub scalar_max: Option<f32>,
    pub scalar_scale: Option<f32>,
    /// Product codebook flat `M*256*subdim f32` in row-major `[m][256][subdim]`.
    pub codebook: Option<Vec<f32>>,
    pub pq_m: Option<usize>,
    pub pq_subdim: Option<usize>,
    /// Slot capacity this quant build covers.
    pub slot_capacity: u64,
    /// Whether the codebook / scales are fully trained.
    pub ready: bool,
}

impl QuantMeta {
    fn new(dim: usize, distance: DistanceMetric, config: QuantizationConfig) -> Self {
        Self {
            dim,
            distance,
            config,
            scalar_min: None,
            scalar_max: None,
            scalar_scale: None,
            codebook: None,
            pq_m: None,
            pq_subdim: None,
            slot_capacity: 0,
            ready: false,
        }
    }
}

/// Segmented mmap for quantized codes.
pub struct QuantStore {
    dir: PathBuf,
    path: PathBuf,
    meta_path: PathBuf,
    dim: usize,
    config: QuantizationConfig,
    segment_slots: u32,
    bytes_per_vector: usize,
    file: parking_lot::Mutex<File>,
    segments: ArcSwap<Vec<Arc<Mmap>>>,
    meta: parking_lot::RwLock<QuantMeta>,
}

impl QuantStore {
    /// Create fresh quant files for `config`.
    pub fn create(
        dir: &Path,
        dim: usize,
        distance: DistanceMetric,
        config: &QuantizationConfig,
        segment_slots: u32,
        slot_capacity: u64,
    ) -> Result<Self> {
        config.validate(dim)?;
        let bytes_per_vector = config.quant_bytes_per_vector(dim);
        let path = dir.join(QUANT_FILE);
        let meta_path = dir.join(QUANT_META_FILE);

        let bytes_per_segment = segment_bytes(bytes_per_vector, segment_slots);
        let total = slot_capacity as usize * bytes_per_vector;

        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        if total > 0 {
            file.set_len(total as u64)?;
            file.sync_all()?;
        }

        let segments = if total == 0 {
            Vec::new()
        } else {
            map_all_segments(&file, bytes_per_segment, total as u64)?
        };

        let mut meta = QuantMeta::new(dim, distance, config.clone());
        meta.slot_capacity = slot_capacity;
        // Scalar/binary ready immediately (global scale or trivial bits).
        // Product needs training; mark not ready until build_quantization runs.
        meta.ready = !matches!(config.quant_type, Some(QuantizationType::Product { .. }));
        // For scalar, init dummy scale (will be recomputed on build).
        if matches!(config.quant_type, Some(QuantizationType::Scalar { .. })) {
            meta.scalar_min = Some(0.0);
            meta.scalar_max = Some(1.0);
            meta.scalar_scale = Some(1.0 / 255.0);
        }

        save_meta(&meta_path, &meta)?;

        Ok(Self {
            dir: dir.to_path_buf(),
            path,
            meta_path,
            dim,
            config: config.clone(),
            segment_slots,
            bytes_per_vector,
            file: parking_lot::Mutex::new(file),
            segments: ArcSwap::from(Arc::new(segments)),
            meta: parking_lot::RwLock::new(meta),
        })
    }

    pub fn open(
        dir: &Path,
        dim: usize,
        distance: DistanceMetric,
        segment_slots: u32,
        slot_capacity: u64,
    ) -> Result<Option<Self>> {
        let path = dir.join(QUANT_FILE);
        let meta_path = dir.join(QUANT_META_FILE);
        if !path.exists() || !meta_path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&meta_path)?;
        let meta = match read_tagged::<QuantMeta>(&bytes, &QUANT_MAGIC, QUANT_VERSION) {
            Ok(m) => m,
            Err(_) => {
                // Corrupt meta: discard both files, caller falls back to exact.
                let _ = std::fs::remove_file(&path);
                let _ = std::fs::remove_file(&meta_path);
                return Ok(None);
            }
        };
        // Validate consistency with collection metadata.
        if meta.dim != dim || meta.distance != distance {
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(&meta_path);
            return Ok(None);
        }
        if meta.config.quant_bytes_per_vector(dim) == 0 {
            return Ok(None);
        }
        let bytes_per_vector = meta.config.quant_bytes_per_vector(dim);
        let expected = slot_capacity as usize * bytes_per_vector;
        let file = File::options().read(true).write(true).open(&path)?;
        let actual = file.metadata()?.len() as usize;
        // Allow file shorter than capacity if built before grow; grow on demand.
        // If file longer than expected, mismatch => discard.
        if actual > expected {
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(&meta_path);
            return Ok(None);
        }
        // If actual < expected, the file was built for smaller capacity; we
        // will grow it to current capacity and zero-fill.
        if actual < expected {
            file.set_len(expected as u64)?;
            file.sync_all()?;
        }

        let bytes_per_segment = segment_bytes(bytes_per_vector, segment_slots);
        let segments = if expected == 0 {
            Vec::new()
        } else {
            map_all_segments(&file, bytes_per_segment, expected as u64)?
        };

        let config = meta.config.clone();
        Ok(Some(Self {
            dir: dir.to_path_buf(),
            path,
            meta_path,
            dim,
            config,
            segment_slots,
            bytes_per_vector,
            file: parking_lot::Mutex::new(file),
            segments: ArcSwap::from(Arc::new(segments)),
            meta: parking_lot::RwLock::new(meta),
        }))
    }

    pub fn bytes_per_vector(&self) -> usize {
        self.bytes_per_vector
    }

    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }

    pub fn is_ready(&self) -> bool {
        self.meta.read().ready
    }

    pub fn snapshot(&self) -> arc_swap::Guard<Arc<Vec<Arc<Mmap>>>> {
        self.segments.load()
    }

    pub fn meta_snapshot(&self) -> QuantMeta {
        self.meta.read().clone()
    }

    /// Grow quant file to at least `target_slots`.
    pub fn grow_to(&self, target_slots: u64) -> Result<()> {
        let current = self.meta.read().slot_capacity;
        if target_slots <= current {
            return Ok(());
        }
        let new_capacity = target_slots;
        let bytes_per_vector = self.bytes_per_vector;
        let total = new_capacity as usize * bytes_per_vector;
        let file = self.file.lock();
        file.set_len(total as u64)?;
        file.sync_all()?;
        let bytes_per_segment = segment_bytes(bytes_per_vector, self.segment_slots);
        let segments = map_all_segments(&file, bytes_per_segment, total as u64)?;
        self.segments.store(Arc::new(segments));
        {
            let mut meta = self.meta.write();
            meta.slot_capacity = new_capacity;
            save_meta(&self.meta_path, &meta)?;
        }
        Ok(())
    }

    /// Write quantized code for a slot. Caller ensures slot < capacity.
    pub fn write_slot(&self, slot: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VectorSearchError::InvalidVectorDimension {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        let code = self.encode(vector)?;
        if code.len() != self.bytes_per_vector {
            return Err(VectorSearchError::Internal(format!(
                "quant code length {} != expected {}",
                code.len(),
                self.bytes_per_vector
            )));
        }
        let offset = slot as usize * self.bytes_per_vector;
        write_at(&self.file.lock(), &code, offset as u64)?;
        Ok(())
    }

    /// Read quantized code for a slot from snapshot.
    pub fn read_slot(
        snapshot: &[Arc<Mmap>],
        slot: u64,
        segment_slots: u32,
        bytes_per_vector: usize,
    ) -> Option<&[u8]> {
        if bytes_per_vector == 0 {
            return None;
        }
        let seg_idx = (slot / segment_slots as u64) as usize;
        let in_seg = slot % segment_slots as u64;
        let seg = snapshot.get(seg_idx)?;
        let offset = in_seg as usize * bytes_per_vector;
        let end = offset + bytes_per_vector;
        if end > seg.len() {
            return None;
        }
        Some(&seg[offset..end])
    }

    /// Encode a vector into its quantized byte representation using current meta.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let meta = self.meta.read();
        match &self.config.quant_type {
            Some(QuantizationType::Scalar { .. }) => {
                let min = meta.scalar_min.unwrap_or(0.0);
                let max = meta.scalar_max.unwrap_or(1.0);
                let scale = meta.scalar_scale.unwrap_or((max - min) / 255.0);
                let inv_scale = if scale.abs() < 1e-9 { 1.0 } else { 1.0 / scale };
                let mut out = Vec::with_capacity(self.dim);
                for &v in vector {
                    let clamped = v.clamp(min, max);
                    let q = ((clamped - min) * inv_scale).round().clamp(0.0, 255.0) as u8;
                    out.push(q);
                }
                Ok(out)
            }
            Some(QuantizationType::Binary { .. }) => {
                let mut out = vec![0u8; self.bytes_per_vector];
                for (i, &v) in vector.iter().enumerate() {
                    if v > 0.0 {
                        out[i / 8] |= 1 << (i % 8);
                    }
                }
                Ok(out)
            }
            Some(QuantizationType::Product { compression, .. }) => {
                if !meta.ready {
                    return Err(VectorSearchError::Internal(
                        "product quantization codebook not ready".to_string(),
                    ));
                }
                let codebook = meta.codebook.as_ref().ok_or_else(|| {
                    VectorSearchError::CorruptData("missing product codebook".to_string())
                })?;
                let m = meta.pq_m.unwrap_or(compression.pq_m(self.dim));
                let subdim = meta.pq_subdim.unwrap_or(self.dim / m.max(1));
                let mut out = Vec::with_capacity(m);
                for s in 0..m {
                    let start = s * subdim;
                    let end = start + subdim;
                    let sub = &vector[start..end];
                    // Find nearest centroid in codebook[s*256:(s+1)*256]
                    let base = s * 256 * subdim;
                    let mut best = 0u8;
                    let mut best_dist = f32::INFINITY;
                    for k in 0..256 {
                        let centroid = &codebook[base + k * subdim..base + (k + 1) * subdim];
                        let d = squared_l2(sub, centroid);
                        if d < best_dist {
                            best_dist = d;
                            best = k as u8;
                        }
                    }
                    out.push(best);
                }
                Ok(out)
            }
            None => Err(VectorSearchError::Internal(
                "quantization not configured".to_string(),
            )),
        }
    }

    /// Quantized distance between query f32 and stored code.
    /// Returns internal distance (smaller = nearer) consistent with `distance`.
    pub fn distance_quantized(&self, query: &[f32], code: &[u8], metric: DistanceMetric) -> f32 {
        let meta = self.meta.read();
        match &self.config.quant_type {
            Some(QuantizationType::Scalar { .. }) => {
                // Dequantize code to approx f32 and compute true distance.
                let min = meta.scalar_min.unwrap_or(0.0);
                let scale = meta.scalar_scale.unwrap_or(1.0 / 255.0);
                // Avoid per-slot allocation by computing on the fly in naive loop.
                // We'll materialize dequantized vector into stack small vec.
                // For performance we inline distance computation.
                match metric {
                    DistanceMetric::Euclid => {
                        let mut sum = 0.0f32;
                        for (i, &q) in query.iter().enumerate() {
                            let approx = code[i] as f32 * scale + min;
                            let d = q - approx;
                            sum += d * d;
                        }
                        sum
                    }
                    DistanceMetric::Dot => {
                        let mut dot = 0.0f32;
                        for (i, &q) in query.iter().enumerate() {
                            let approx = code[i] as f32 * scale + min;
                            dot += q * approx;
                        }
                        -dot
                    }
                    DistanceMetric::Cosine => {
                        let mut dot = 0.0f32;
                        let mut norm_q = 0.0f32;
                        let mut norm_c = 0.0f32;
                        for (i, &q) in query.iter().enumerate() {
                            let approx = code[i] as f32 * scale + min;
                            dot += q * approx;
                            norm_q += q * q;
                            norm_c += approx * approx;
                        }
                        let denom = (norm_q * norm_c).sqrt();
                        if denom == 0.0 {
                            return 1.0;
                        }
                        1.0 - (dot / denom).clamp(-1.0, 1.0)
                    }
                    DistanceMetric::Manhattan => {
                        let mut sum = 0.0f32;
                        for (i, &q) in query.iter().enumerate() {
                            let approx = code[i] as f32 * scale + min;
                            sum += (q - approx).abs();
                        }
                        sum
                    }
                }
            }
            Some(QuantizationType::Binary { .. }) => {
                // Hamming distance; for all metrics lower hamming = nearer.
                // Compute popcount of xor between query bits and stored bits.
                let qbits = encode_binary(query);
                let mut ham = 0u32;
                for (a, b) in qbits.iter().zip(code.iter()) {
                    ham += (a ^ b).count_ones();
                }
                ham as f32
            }
            Some(QuantizationType::Product { .. }) => {
                if let (Some(cb), Some(&m), Some(&subdim)) = (
                    meta.codebook.as_ref(),
                    meta.pq_m.as_ref(),
                    meta.pq_subdim.as_ref(),
                ) {
                    match metric {
                        DistanceMetric::Euclid => {
                            // ADC: precompute distance tables per subspace.
                            // For Euclid we use squared L2.
                            let mut total = 0.0f32;
                            for (s, &code_byte) in code.iter().enumerate().take(m) {
                                let start = s * subdim;
                                let qsub = &query[start..start + subdim];
                                let base = s * 256 * subdim;
                                let c = code_byte as usize;
                                let centroid = &cb[base + c * subdim..base + (c + 1) * subdim];
                                total += squared_l2(qsub, centroid);
                            }
                            total
                        }
                        DistanceMetric::Dot => {
                            // Approximate inner product via codebook.
                            let mut dot = 0.0f32;
                            for (s, &code_byte) in code.iter().enumerate().take(m) {
                                let start = s * subdim;
                                let qsub = &query[start..start + subdim];
                                let base = s * 256 * subdim;
                                let c = code_byte as usize;
                                let centroid = &cb[base + c * subdim..base + (c + 1) * subdim];
                                for (a, b) in qsub.iter().zip(centroid.iter()) {
                                    dot += a * b;
                                }
                            }
                            -dot
                        }
                        DistanceMetric::Cosine => {
                            // Norm on the fly for both query and reconstructed approx vector.
                            // Reconstruct approx vector.
                            let mut dot = 0.0f32;
                            let mut norm_q = 0.0f32;
                            let mut norm_c = 0.0f32;
                            for (s, &code_byte) in code.iter().enumerate().take(m) {
                                let start = s * subdim;
                                let qsub = &query[start..start + subdim];
                                let base = s * 256 * subdim;
                                let c = code_byte as usize;
                                let centroid = &cb[base + c * subdim..base + (c + 1) * subdim];
                                for (a, b) in qsub.iter().zip(centroid.iter()) {
                                    dot += a * b;
                                    norm_q += a * a;
                                    norm_c += b * b;
                                }
                            }
                            let denom = (norm_q * norm_c).sqrt();
                            if denom == 0.0 {
                                return 1.0;
                            }
                            1.0 - (dot / denom).clamp(-1.0, 1.0)
                        }
                        DistanceMetric::Manhattan => {
                            let mut sum = 0.0f32;
                            for (s, &code_byte) in code.iter().enumerate().take(m) {
                                let start = s * subdim;
                                let qsub = &query[start..start + subdim];
                                let base = s * 256 * subdim;
                                let c = code_byte as usize;
                                let centroid = &cb[base + c * subdim..base + (c + 1) * subdim];
                                for (a, b) in qsub.iter().zip(centroid.iter()) {
                                    sum += (a - b).abs();
                                }
                            }
                            sum
                        }
                    }
                } else {
                    f32::INFINITY
                }
            }
            None => f32::INFINITY,
        }
    }

    /// Rebuild quantization from live vectors.
    /// Called off the store lock under build_mutex, so slot numbers are stable.
    pub fn rebuild(
        &self,
        vectors: &[Arc<Mmap>],
        segment_slots: u32,
        live_slots: &[u32],
        dim: usize,
    ) -> Result<()> {
        match &self.config.quant_type {
            Some(QuantizationType::Scalar { quantile, .. }) => {
                let quantile = quantile.unwrap_or(0.99);
                // Gather all float values across live vectors for quantile clipping.
                let mut values: Vec<f32> = Vec::with_capacity(live_slots.len() * dim);
                for &slot in live_slots {
                    if let Some(v) = crate::storage::vectors::Vectors::read_slot(
                        vectors,
                        slot as u64,
                        segment_slots,
                        dim,
                    ) {
                        values.extend_from_slice(v);
                    }
                }
                let (min, max) = if values.is_empty() {
                    (0.0, 1.0)
                } else {
                    // Compute quantile range.
                    if (quantile - 1.0).abs() < 1e-6 || quantile >= 1.0 {
                        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        (min, max)
                    } else {
                        values.sort_by(|a, b| a.total_cmp(b));
                        let low_idx = ((1.0 - quantile) / 2.0 * values.len() as f32) as usize;
                        let high_idx = ((1.0 + quantile) / 2.0 * values.len() as f32) as usize;
                        let high_idx = high_idx.min(values.len() - 1);
                        (values[low_idx], values[high_idx])
                    }
                };
                let range = max - min;
                let scale = if range.abs() < 1e-9 {
                    1.0
                } else {
                    range / 255.0
                };
                {
                    let mut meta = self.meta.write();
                    meta.scalar_min = Some(min);
                    meta.scalar_max = Some(max);
                    meta.scalar_scale = Some(scale);
                    meta.ready = true;
                    save_meta(&self.meta_path, &meta)?;
                }
                // Encode all live slots into quant.bin
                for &slot in live_slots {
                    if let Some(v) = crate::storage::vectors::Vectors::read_slot(
                        vectors,
                        slot as u64,
                        segment_slots,
                        dim,
                    ) {
                        self.write_slot(slot as u64, v)?;
                    }
                }
                // For non-live slots up to capacity, zero fill is already present.
                let file = self.file.lock();
                file.sync_all()?;
                Ok(())
            }
            Some(QuantizationType::Binary { .. }) => {
                {
                    let mut meta = self.meta.write();
                    meta.ready = true;
                    save_meta(&self.meta_path, &meta)?;
                }
                for &slot in live_slots {
                    if let Some(v) = crate::storage::vectors::Vectors::read_slot(
                        vectors,
                        slot as u64,
                        segment_slots,
                        dim,
                    ) {
                        self.write_slot(slot as u64, v)?;
                    }
                }
                self.file.lock().sync_all()?;
                Ok(())
            }
            Some(QuantizationType::Product { compression, .. }) => {
                let m = compression.pq_m(dim);
                let subdim = dim / m.max(1);
                // Gather vectors per subspace for k-means training.
                // For each subspace, train 256 centroids.
                let mut codebook: Vec<f32> = Vec::with_capacity(m * 256 * subdim);
                for s in 0..m {
                    let start = s * subdim;
                    let end = start + subdim;
                    let sample: Vec<&[f32]> = live_slots
                        .iter()
                        .filter_map(|&slot| {
                            crate::storage::vectors::Vectors::read_slot(
                                vectors,
                                slot as u64,
                                segment_slots,
                                dim,
                            )
                            .map(|v| &v[start..end])
                        })
                        .collect();
                    if sample.is_empty() {
                        // No data: fill with zeros
                        codebook.resize(codebook.len() + 256 * subdim, 0.0);
                        continue;
                    }
                    // If sample < 256, duplicate to reach 256 points for kmeans (k=256)
                    // But train caps k to sample len.
                    let opts = crate::index::kmeans::KmeansOptions {
                        k: 256,
                        max_iter: 20,
                        seed: 0x9E37_79B9_7F4A_7C15u64.wrapping_add(s as u64 * 0x9E3779B9),
                    };
                    let result =
                        crate::index::kmeans::train(DistanceMetric::Euclid, &sample, &opts)?;
                    // result.centroids len may be <256 if sample small; pad.
                    for c in result.centroids {
                        codebook.extend_from_slice(&c);
                    }
                    let have = result_centroids_len(&sample, 256);
                    // pad remaining
                    let remaining = 256usize.saturating_sub(have);
                    codebook.resize(codebook.len() + remaining * subdim, 0.0);
                    debug_assert_eq!(codebook.len(), (s + 1) * 256 * subdim);
                }

                {
                    let mut meta = self.meta.write();
                    meta.codebook = Some(codebook.clone());
                    meta.pq_m = Some(m);
                    meta.pq_subdim = Some(subdim);
                    meta.ready = true;
                    save_meta(&self.meta_path, &meta)?;
                }
                // Encode live vectors
                for &slot in live_slots {
                    if let Some(v) = crate::storage::vectors::Vectors::read_slot(
                        vectors,
                        slot as u64,
                        segment_slots,
                        dim,
                    ) {
                        self.write_slot(slot as u64, v)?;
                    }
                }
                self.file.lock().sync_all()?;
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// Atomically replace quant files after compaction renumbering.
    pub fn replace_from(&self, tmp_path: &Path, new_capacity: u64, live_map: &[u32]) -> Result<()> {
        // `tmp_path` is a temp quant.bin with new capacity.
        std::fs::rename(tmp_path, &self.path)?;
        open_dir(self.dir.as_path())?.sync_all()?;

        let file = File::options().read(true).write(true).open(&self.path)?;
        let bytes_per_vector = self.bytes_per_vector;
        let total = new_capacity as usize * bytes_per_vector;
        let bytes_per_segment = segment_bytes(bytes_per_vector, self.segment_slots);
        let segments = if total == 0 {
            Vec::new()
        } else {
            map_all_segments(&file, bytes_per_segment, total as u64)?
        };
        *self.file.lock() = file;
        self.segments.store(Arc::new(segments));
        {
            let mut meta = self.meta.write();
            // Update slot_capacity, but keep scale/codebook ready.
            meta.slot_capacity = new_capacity;
            // For product, codebook slots mapping is already handled by rewriting quant.bin
            // via compaction's vector mapping (code already remapped).
            save_meta(&self.meta_path, &meta)?;
        }
        // live_map not needed beyond caller handling; quant file already rewritten.
        let _ = live_map;
        Ok(())
    }

    /// Write compaction temp quant file.
    pub fn write_compacted_file(
        &self,
        tmp_path: &Path,
        new_capacity: u64,
        old_vectors: &[Arc<Mmap>],
        map: &[u32],
    ) -> Result<()> {
        let total = new_capacity as usize * self.bytes_per_vector;
        #[allow(unused_mut)]
        let mut file = File::create(tmp_path)?;
        if total > 0 {
            file.set_len(total as u64)?;
            file.sync_all()?;
        }

        if self.bytes_per_vector == 0 {
            file.sync_all()?;
            return Ok(());
        }

        // For each old slot that is live, copy its quantized bytes to new slot offset.
        let old_snap = self.snapshot();
        for (old_slot, &new_slot) in map.iter().enumerate() {
            if new_slot == u32::MAX {
                continue;
            }
            // If quant not ready (product before build), encode from vectors instead of copying
            // old quant bytes. But copying old bytes is fine if ready.
            if self.is_ready() {
                if let Some(code) = Self::read_slot(
                    &old_snap,
                    old_slot as u64,
                    self.segment_slots,
                    self.bytes_per_vector,
                ) {
                    let dst = new_slot as usize * self.bytes_per_vector;
                    write_at(&file, code, dst as u64)?;
                    continue;
                }
            }
            // Fallback: encode from vector if available and ready after potential rebuild of meta?
            // For product not ready, skip (remains zero).
            if let Some(v) = crate::storage::vectors::Vectors::read_slot(
                old_vectors,
                old_slot as u64,
                self.segment_slots,
                self.dim,
            ) {
                // Only encode if ready or scalar/binary (always ready)
                if self.is_ready() {
                    if let Ok(code) = self.encode(v) {
                        let dst = new_slot as usize * self.bytes_per_vector;
                        write_at(&file, &code, dst as u64)?;
                    }
                }
            }
        }
        file.sync_all()?;
        Ok(())
    }
}

fn result_centroids_len(sample: &[&[f32]], requested: usize) -> usize {
    std::cmp::min(requested, sample.len())
}

fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn encode_binary(vector: &[f32]) -> Vec<u8> {
    encode_binary_for_query(vector)
}

pub(crate) fn encode_binary_for_query(vector: &[f32]) -> Vec<u8> {
    let bytes = vector.len().div_ceil(8);
    let mut out = vec![0u8; bytes];
    for (i, &v) in vector.iter().enumerate() {
        if v > 0.0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

fn segment_bytes(bytes_per_vector: usize, segment_slots: u32) -> u64 {
    bytes_per_vector as u64 * segment_slots as u64
}

fn open_dir(dir: &Path) -> Result<File> {
    Ok(File::open(dir)?)
}

fn map_all_segments(file: &File, segment_bytes: u64, total: u64) -> Result<Vec<Arc<Mmap>>> {
    let mut out = Vec::new();
    let mut offset = 0u64;
    while offset < total {
        let len = segment_bytes.min(total - offset);
        let mmap = unsafe {
            MmapOptions::new()
                .offset(offset)
                .len(len as usize)
                .map(file)
        }?;
        out.push(Arc::new(mmap));
        offset += len;
    }
    Ok(out)
}

#[cfg(unix)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}
#[cfg(not(unix))]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut guard = file;
    guard.seek(SeekFrom::Start(offset))?;
    guard.write_all(buf)
}

fn save_meta(path: &Path, meta: &QuantMeta) -> Result<()> {
    let bytes = postcard::to_stdvec(meta)?;
    let crc = crc32fast::hash(&bytes);
    let tmp = path.with_extension("tmp");
    let mut file = File::create(&tmp)?;
    file.write_all(&QUANT_MAGIC)?;
    file.write_all(&QUANT_VERSION.to_le_bytes())?;
    file.write_all(&crc.to_le_bytes())?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    std::fs::rename(&tmp, path)?;
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

fn read_tagged<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    magic: &[u8; 4],
    version: u16,
) -> Result<T> {
    if bytes.len() < 10 || &bytes[..4] != magic {
        return Err(VectorSearchError::CorruptData(
            "bad quant magic".to_string(),
        ));
    }
    let stored = u16::from_le_bytes([bytes[4], bytes[5]]);
    if stored != version {
        return Err(VectorSearchError::CorruptData(format!(
            "unsupported quant version {stored}"
        )));
    }
    let expected_crc = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
    let payload = &bytes[10..];
    if crc32fast::hash(payload) != expected_crc {
        return Err(VectorSearchError::CorruptData(
            "quant crc mismatch".to_string(),
        ));
    }
    Ok(postcard::from_bytes(payload)?)
}

impl std::fmt::Debug for QuantStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantStore")
            .field("dim", &self.dim)
            .field("config", &self.config)
            .field("bytes_per_vector", &self.bytes_per_vector)
            .finish()
    }
}
