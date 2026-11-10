use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::error::Result;
use crate::types::{DistanceMetric, QuantizationType};

use super::{map_all_segments, open_dir, save_meta, segment_bytes, write_at, QuantStore};

impl QuantStore {
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
            QuantizationType::Scalar { quantile, .. } => {
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
            QuantizationType::Binary { .. } => {
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
            QuantizationType::Product { compression, .. } => {
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

pub(super) fn result_centroids_len(sample: &[&[f32]], requested: usize) -> usize {
    std::cmp::min(requested, sample.len())
}
