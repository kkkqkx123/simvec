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

mod codec;
mod rebuild;

pub(crate) use self::codec::encode_binary_for_query;

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
        meta.ready = !matches!(config.quant_type, QuantizationType::Product { .. });
        // For scalar, init dummy scale (will be recomputed on build).
        if matches!(config.quant_type, QuantizationType::Scalar { .. }) {
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
