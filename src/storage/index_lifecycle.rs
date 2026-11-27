//! ANN index lifecycle for a collection: rehydration on open, build/publish,
//! drop, drift measurement and introspection.
//!
//! Publication protocol (shared by both tiers):
//! - Heavy build work runs off the store lock under the `build_mutex`,
//!   so compaction cannot interleave and slot numbers stay stable.
//! - While a build is in flight (`building` flag) and no index is published,
//!   freshly written slots are recorded in `pending`.
//! - Publication happens under the store write lock: slots appended during
//!   the build are adopted and `pending` is drained *before* the index
//!   becomes visible, so approximate search never misses a point.
//! Only completed segments enter the graph; slots in the partial tail segment
//! stay in `pending` and are exact-scanned until a rebuild promotes the
//! completed segment. A full tail segment's worth of pending slots schedules
//! that rebuild, bounding both graph churn and exact-scan work.
//!
//! - The published index is stored before clearing `building`: an upsert that
//!   loaded `index == None` and then observed `building == false` would
//!   record its slot nowhere; with this order it still lands in `pending`,
//!   which every publication path drains.

use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::Arc;
use std::time::Instant;

use super::segment_hnsw::{self, SegmentHnsw};
use super::segment_ivf::{self, SegmentIvf};
use super::segments;
use super::vectors::Vectors;
use super::CollectionStore;
use crate::error::Result;
use crate::index::{persist, HnswIndex, IndexBuildParams, IvfIndex};
use crate::metrics::IndexTier;
use crate::types::{HnswConfig, IndexConfig, IndexInfo, IndexKind, IvfConfig};

impl CollectionStore {
    /// Whether a pending backlog of `pending` slots warrants a full HNSW
    /// rebuild of a published graph with `nodes` nodes. Draining and
    /// rebuilding cost the same per node, so a rebuild only wins when a large
    /// share of the graph is absent; smaller backlogs are handled by the
    /// incremental drain.
    pub(crate) fn pending_warrants_rebuild(pending: usize, nodes: usize) -> bool {
        if pending == 0 {
            return false;
        }
        if nodes == 0 {
            return true;
        }
        pending * 2 >= nodes
    }

    /// Rebuild one completed segment's HNSW graph from its live rows.
    ///
    /// Local node ids are `0..end-start`; vectors are read from the global
    /// slot range so the graph file stays independent of the generation's
    /// slot numbering.
    pub(super) fn rebuild_segment_hnsw(
        &self,
        start: u32,
        end: u32,
        name: &str,
        dim: usize,
        metric: crate::types::DistanceMetric,
        segment_slots: u32,
        config: &HnswConfig,
    ) -> Result<Option<SegmentHnsw>> {
        let files = self.data_arc();
        let tombstones = files.tombstones.load();
        let vsnap = files.vectors.snapshot();
        let local: Vec<u32> = (start..end)
            .filter(|slot| !tombstones.bit(*slot as usize))
            .map(|slot| slot - start)
            .collect();
        if local.is_empty() {
            return Ok(None);
        }
        let built = HnswIndex::build(
            &IndexBuildParams::new(name, dim, metric, &local, &vsnap, segment_slots)
                .with_slot_base(start),
            config,
        )?;
        let mut built = built;
        built.set_metrics(self.metrics_arc());
        persist::save_hnsw_at(
            &segment_hnsw::file_path(&files.dir(), start),
            &built.to_persisted(),
        )?;
        let quant = {
            let guard = files.quant.read();
            match guard.as_ref() {
                Some(q) if q.is_ready() => q.write_segment_file(
                    &super::quant::segment_file_path(&files.dir(), start),
                    &vsnap,
                    segment_slots,
                    start,
                    end - start,
                )?,
                _ => false,
            }
        };
        let quant = if quant {
            let guard = files.quant.read();
            guard.as_ref().map(|q| {
                super::quant::QuantStore::load_segment_file(
                    &super::quant::segment_file_path(&files.dir(), start),
                    segment_slots,
                    q.bytes_per_vector(),
                )
            })
        } else {
            None
        };
        Ok(Some(SegmentHnsw {
            start,
            end,
            index: Arc::new(built),
            quant: quant.transpose()?.flatten(),
        }))
    }

    /// Supply the IVF configuration after open. Collections persist their
    /// effective config, but the engine-provided
    /// runtime configuration takes precedence. Applies both to future builds
    /// and to any published index, so a rehydrated index immediately honors
    /// the runtime config.
    pub fn set_ivf_config(&self, config: IvfConfig) -> Result<()> {
        self.ensure_not_poisoned()?;
        let files = self.data_arc();
        config.validate()?;
        let mut inner = self.inner.write();
        match &mut inner.meta.index {
            IndexConfig::Ivf(slot) => {
                *slot = Some(config.clone());
                inner.ivf_config = Some(config.clone());
                inner.meta.save(&self.data_dir())?;
            }
            _ => {
                return Err(crate::error::VectorSearchError::InvalidConfig(
                    "ivf config requires the IVF tier".to_string(),
                ));
            }
        }
        drop(inner);
        for segment in files.ivf_segments.load().iter() {
            segment.index.set_config(config.clone());
        }
        Ok(())
    }

    /// Supply the HNSW configuration after open. Unlike the IVF counterpart
    /// this persists to `meta.bin` immediately: the published graph bakes its
    /// parameters in at build time, so changes only affect the next rebuild
    /// and memory must stay consistent with disk.
    pub fn set_hnsw_config(&self, config: HnswConfig) -> Result<()> {
        self.ensure_not_poisoned()?;
        config.validate()?;
        let mut inner = self.inner.write();
        match &mut inner.meta.index {
            IndexConfig::Hnsw(slot) => {
                *slot = config;
                inner.meta.save(&self.data_dir())
            }
            _ => Err(crate::error::VectorSearchError::InvalidConfig(
                "hnsw config requires the HNSW tier".to_string(),
            )),
        }
    }

    /// Rehydrate the persisted ANN tier matching the collection's index type
    /// if present and consistent with the live metadata; otherwise fall back
    /// to exact scan. Slots appended after the build (WAL replay) land in
    /// `pending` so approximate searches still see them.
    pub(super) fn load_index(&self) -> Result<()> {
        match self.inner.read().meta.index {
            IndexConfig::Ivf(_) => self.load_ivf(),
            IndexConfig::Hnsw(_) => self.load_hnsw(),
            IndexConfig::Flat => Ok(()),
        }
    }

    fn load_ivf(&self) -> Result<()> {
        let files = self.data_arc();
        let (dim, distance, next_slot, segment_slots) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.next_slot,
                inner.meta.segment_slots,
            )
        };
        let config = self.inner.read().ivf_config.clone().unwrap_or_default();
        let boundary = segments::immutable_end(next_slot, segment_slots);
        let step = segment_slots.max(1);
        let mut loaded = Vec::new();
        let mut missing = false;
        let mut uncovered: Vec<u32> = Vec::new();
        for start in (0..boundary).step_by(step as usize) {
            let end = (start + step).min(boundary);
            let Some(data) = persist::load_at(&segment_ivf::file_path(&files.dir(), start))? else {
                missing = true;
                uncovered.extend(start..end);
                continue;
            };
            let covered = data.slot_list.len() as u32;
            if data.dim != dim || data.distance != distance || covered == 0 || covered > end - start
            {
                missing = true;
                uncovered.extend(start..end);
                continue;
            }
            if covered < end - start {
                missing = true;
                uncovered.extend(start + covered..end);
            }
            let mut index = IvfIndex::from_persisted(data, config.clone());
            index.set_metrics(self.metrics_arc());
            loaded.push(SegmentIvf {
                start,
                end: start + covered,
                index: Arc::new(index),
            });
        }
        {
            let tombstones = files.tombstones.load();
            let mut p = self.pending.lock();
            p.extend(
                uncovered
                    .into_iter()
                    .chain(boundary..next_slot as u32)
                    .filter(|slot| !tombstones.bit(*slot as usize)),
            );
        }
        if missing {
            self.metrics.record_index_load_fallback();
        }
        files.ivf_segments.store(Arc::new(loaded));
        Ok(())
    }

    /// Build one IVF file per completed dense segment. Each segment owns its
    /// own centroids and segment-local assignments.
    fn rebuild_segment_ivf(
        &self,
        start: u32,
        end: u32,
        name: &str,
        dim: usize,
        metric: crate::types::DistanceMetric,
        segment_slots: u32,
        config: &IvfConfig,
    ) -> Result<Option<SegmentIvf>> {
        let files = self.data_arc();
        let tombstones = files.tombstones.load();
        let vsnap = files.vectors.snapshot();
        let local: Vec<u32> = (start..end)
            .filter(|slot| !tombstones.bit(*slot as usize))
            .map(|slot| slot - start)
            .collect();
        if local.is_empty() {
            return Ok(None);
        }
        let mut index = IvfIndex::build(
            &IndexBuildParams::new(name, dim, metric, &local, &vsnap, segment_slots)
                .with_slot_base(start),
            config,
        )?;
        index.set_metrics(self.metrics_arc());
        persist::save_at(
            &segment_ivf::file_path(&files.dir(), start),
            &index.to_persisted(),
        )?;
        Ok(Some(SegmentIvf {
            start,
            end,
            index: Arc::new(index),
        }))
    }

    fn load_hnsw(&self) -> Result<()> {
        let files = self.data_arc();
        let (dim, distance, next_slot, segment_slots) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.next_slot,
                inner.meta.segment_slots,
            )
        };
        let config = match &self.inner.read().meta.index {
            IndexConfig::Hnsw(cfg) => cfg.clone(),
            _ => HnswConfig::default(),
        };
        let boundary = segments::immutable_end(next_slot, segment_slots);
        let mut loaded = Vec::new();
        let mut missing = false;
        let mut uncovered: Vec<u32> = Vec::new();
        let step = segment_slots.max(1);
        for start in (0..boundary).step_by(step as usize) {
            let end = (start + step).min(boundary);
            let Some(data) = persist::load_hnsw_at(&segment_hnsw::file_path(&files.dir(), start))?
            else {
                missing = true;
                uncovered.extend(start..end);
                continue;
            };
            let covered = data.nodes.len() as u32;
            if data.dim != dim || data.distance != distance || covered == 0 || covered > end - start
            {
                missing = true;
                uncovered.extend(start..end);
                continue;
            }
            if covered < end - start {
                // A graph published before later writes can legitimately be
                // shorter than the current segment. Keep it and exact-scan
                // the rest instead of discarding the whole segment.
                missing = true;
                uncovered.extend(start + covered..end);
            }
            match HnswIndex::from_persisted(data, &config) {
                Ok(mut index) => {
                    index.set_metrics(self.metrics_arc());
                    let quant = {
                        let guard = files.quant.read();
                        match guard.as_ref() {
                            Some(q) => super::quant::QuantStore::load_segment_file(
                                &super::quant::segment_file_path(&files.dir(), start),
                                segment_slots,
                                q.bytes_per_vector(),
                            )?,
                            None => None,
                        }
                    };
                    loaded.push(SegmentHnsw {
                        start,
                        end: start + covered,
                        index: Arc::new(index),
                        quant,
                    });
                }
                Err(_) => {
                    missing = true;
                    uncovered.extend(start..end);
                }
            }
        }
        files.hnsw_segments.store(Arc::new(loaded));

        {
            let tombstones = files.tombstones.load();
            let mut p = self.pending.lock();
            p.extend(uncovered);
            for slot in boundary..next_slot as u32 {
                if !tombstones.bit(slot as usize) {
                    p.push(slot);
                }
            }
            if missing || p.len() >= segment_slots.max(1) as usize {
                self.needs_rebuild.store(true, AtomicOrdering::Relaxed);
            }
        }
        if missing {
            self.metrics.record_index_load_fallback();
        }
        Ok(())
    }

    /// Route a freshly written slot into the published index (nearest IVF
    /// list / pending for HNSW) or, when no index is published but one
    /// is being built, into the pending set so approximate search never
    /// misses it. Runs under the store write lock; `segment_slots` is passed
    /// in because `inner` is already mutably borrowed by the caller.
    ///
    /// For published HNSW the slot is pushed to `pending` rather than
    /// inserted inline into the graph, so the upsert critical section stays
    /// O(1) instead of O(ef_construct) graph search.  Pending slots are
    /// scored by brute force in the segmented HNSW path for correct visibility, and
    /// drained into the graph by the maintenance worker on each tick.
    pub(super) fn register_slot(&self, slot: u32, vector: &[f32], segment_slots: u32) {
        // Writes to the partial tail segment stay out of the frozen index.
        // They are exact-scanned through `pending` until a rebuild promotes
        // the completed segment, which bounds graph churn per segment.
        let boundary = segments::immutable_end(slot as u64 + 1, segment_slots);
        if (slot as u64) >= boundary as u64 {
            let mut pending = self.pending.lock();
            pending.push(slot);
            if pending.len() >= segment_slots.max(1) as usize {
                self.needs_rebuild.store(true, AtomicOrdering::Relaxed);
            }
            return;
        }

        let files = self.data_arc();
        let segmented = files.hnsw_segments.load();
        if !segmented.is_empty() {
            // Frozen segments are immutable; only rows past the last built
            // segment need the exact-scan fallback.
            let covered = segmented
                .iter()
                .map(|segment| segment.end)
                .max()
                .unwrap_or(0);
            if slot < covered {
                // Overwrites keep their graph position but must still be
                // counted so staleness-based rebuilding can see them.
                let vsnap = files.vectors.snapshot();
                if let Some(target) = segmented.iter().find(|s| slot >= s.start && slot < s.end) {
                    target.index.insert(
                        slot - target.start,
                        vector,
                        &vsnap,
                        segment_slots,
                        target.start,
                    );
                }
                return;
            }
            {
                self.pending.lock().push(slot);
                // Before the first segment fills, the single growing graph
                // still takes incremental inserts. Pending remains authoritative
                // for search, but the concurrent adjacency edits keep the
                // version-based torn-read recovery path exercised.
                if (slot as u64) < segment_slots as u64 {
                    let vsnap = files.vectors.snapshot();
                    let mut graphs = (**segmented).clone();
                    if let Some(target) = graphs.iter_mut().find(|s| s.start == 0) {
                        target.index.insert(
                            slot - target.start,
                            vector,
                            &vsnap,
                            segment_slots,
                            target.start,
                        );
                        target.end = slot + 1;
                        files.hnsw_segments.store(Arc::new(graphs));
                    }
                }
            }
            return;
        }
        let ivf_segments = files.ivf_segments.load();
        if !ivf_segments.is_empty() {
            let covered = ivf_segments
                .iter()
                .map(|segment| segment.end)
                .max()
                .unwrap_or(0);
            if slot < covered {
                if let Some(target) = ivf_segments
                    .iter()
                    .find(|s| slot >= s.start && slot < s.end)
                {
                    target.index.assign_slot(slot - target.start, vector);
                    target.index.note_upsert();
                }
                return;
            }
            self.pending.lock().push(slot);
            if (slot as u64) < segment_slots as u64 {
                let mut segments = (**ivf_segments).clone();
                if let Some(target) = segments.iter_mut().find(|s| s.start == 0) {
                    target.index.assign_slot(slot, vector);
                    target.index.note_upsert();
                    target.end = slot + 1;
                    files.ivf_segments.store(Arc::new(segments));
                }
            }
            return;
        }
        // No global published index exists: segment graphs are immutable, so
        // a slot that missed every segment above lands in `pending` (exact
        // scan) when a build is in flight, exactly like the appendable tail.
        if self.building.load(AtomicOrdering::Relaxed) {
            self.pending.lock().push(slot);
        }
    }

    /// Build and publish the configured ANN tier from the current live set.
    /// Dispatches on the collection's `index`; returns whether a usable
    /// index is now published. When quantization is configured this also
    /// ensures `quant.bin` is trained before the index is built, under the
    /// same `build_mutex` so slot numbers stay stable.
    pub fn build_index(&self) -> Result<bool> {
        self.ensure_not_poisoned()?;
        // Train quantization first if needed (especially Product which needs a
        // codebook before HNSW/IVF can be usefully built). Scalar/Binary are
        // idempotent after incremental writes; rebuilding still refreshes the
        // global scale range.
        {
            let needs = {
                let inner = self.inner.read();
                inner.meta.quantization_config.as_ref().is_some()
            };
            let ready = self.has_quantization();
            if needs && !ready {
                // Product quantization not yet trained, or scalar missing.
                if let Err(e) = self.build_quantization() {
                    tracing::warn!(error = %e, "pre-index quant build failed; index builds on exact vectors");
                }
            } else if needs {
                // For scalar, periodically refresh the scale even when ready
                // if live set has grown significantly. For now trigger on
                // every build to keep the range accurate without extra cost
                // for small collections; larger collections grow via background
                // sweeps. Keep product rebuild lazy (already ready) to avoid
                // repeated k-means.
                let qc = self.inner.read().meta.quantization_config.clone().unwrap();
                if matches!(qc.quant_type, crate::types::QuantizationType::Scalar { .. }) {
                    if let Err(e) = self.build_quantization() {
                        tracing::warn!(error = %e, "scalar quant refresh failed; serving existing codes");
                    }
                }
            }
        }
        let index = self.inner.read().meta.index.clone();
        match index {
            IndexConfig::Hnsw(config) => self.build_hnsw_index(config),
            IndexConfig::Ivf(Some(config)) => self.build_ivf_index(config),
            IndexConfig::Ivf(None) | IndexConfig::Flat => Ok(false),
        }
    }

    /// Build one HNSW file per completed segment.
    ///
    /// Each graph is independent: a rebuild of one segment never touches the
    /// others, and the appendable tail is left to exact scan.
    fn build_hnsw_index(&self, config: HnswConfig) -> Result<bool> {
        let _compact_guard = self.compact_mutex.lock();
        let _guard = self.build_mutex.lock();

        let (dim, metric, segment_slots, next_slot, name) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.segment_slots,
                inner.meta.next_slot,
                inner.meta.collection.clone(),
            )
        };
        let boundary = segments::immutable_end(next_slot, segment_slots);
        if boundary == 0 {
            return Ok(false);
        }
        self.building.store(true, AtomicOrdering::Relaxed);
        let build_started = Instant::now();
        let step = segment_slots.max(1);
        let tombstones = self.data_arc().tombstones.load();
        let total = (0..boundary)
            .filter(|slot| !tombstones.bit(*slot as usize))
            .count();
        drop(tombstones);
        self.build_points_total
            .store(total as u64, AtomicOrdering::Relaxed);
        self.build_inserted.store(0, AtomicOrdering::Relaxed);

        let mut built = Vec::with_capacity((boundary / step) as usize + 1);
        for start in (0..boundary).step_by(step as usize) {
            let end = (start + step).min(boundary);
            let live = {
                let tombstones = self.data_arc().tombstones.load();
                (start..end)
                    .filter(|slot| !tombstones.bit(*slot as usize))
                    .count()
            };
            if let Some(segment) =
                self.rebuild_segment_hnsw(start, end, &name, dim, metric, segment_slots, &config)?
            {
                built.push(segment);
            }
            self.build_inserted
                .fetch_add(live as u64, AtomicOrdering::Relaxed);
        }
        let published = !built.is_empty();
        self.data_arc().hnsw_segments.store(Arc::new(built));
        self.building.store(false, AtomicOrdering::Relaxed);
        self.needs_rebuild.store(
            self.pending_len() >= segment_slots.max(1) as usize,
            AtomicOrdering::Relaxed,
        );
        if published {
            self.metrics
                .record_index_build(IndexTier::Hnsw, build_started.elapsed());
            tracing::info!(collection = %name, points = total, "HNSW segment graphs published");
        }
        Ok(published)
    }

    /// Build and publish an IVF index from the current live set.
    ///
    /// Heavy work (sampling + k-means) runs off the store lock under the
    /// `build_mutex`, so compaction cannot interleave and slot numbers
    /// stay stable. Publication happens atomically under the store write
    /// lock: slots appended during the build are adopted, pending slots are
    /// drained, and only afterwards does the index become visible, so probe
    /// search never misses a point.
    fn build_ivf_index(&self, config: IvfConfig) -> Result<bool> {
        let files = self.data_arc();
        let _compact_guard = self.compact_mutex.lock();
        let _guard = self.build_mutex.lock();

        let (dim, metric, segment_slots, next_slot, name) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.segment_slots,
                inner.meta.next_slot,
                inner.meta.collection.clone(),
            )
        };
        let boundary = segments::immutable_end(next_slot, segment_slots);
        if boundary == 0 {
            return Ok(false);
        }
        let step = segment_slots.max(1);
        let tombstones = files.tombstones.load();
        let total = (0..boundary)
            .filter(|slot| !tombstones.bit(*slot as usize))
            .count();
        drop(tombstones);
        if (total as u64) < config.min_build_points.max(1) {
            return Ok(false);
        }

        self.building.store(true, AtomicOrdering::Relaxed);
        self.build_points_total
            .store(total as u64, AtomicOrdering::Relaxed);
        self.build_inserted.store(0, AtomicOrdering::Relaxed);
        let build_started = Instant::now();
        let mut built = Vec::new();
        for start in (0..boundary).step_by(step as usize) {
            let end = (start + step).min(boundary);
            let live = {
                let tombstones = files.tombstones.load();
                (start..end)
                    .filter(|slot| !tombstones.bit(*slot as usize))
                    .count()
            };
            if let Some(segment) =
                self.rebuild_segment_ivf(start, end, &name, dim, metric, segment_slots, &config)?
            {
                built.push(segment);
            }
            self.build_inserted
                .fetch_add(live as u64, AtomicOrdering::Relaxed);
        }
        let published = !built.is_empty();
        files.ivf_segments.store(Arc::new(built));
        self.build_inserted
            .store(total as u64, AtomicOrdering::Relaxed);
        self.building.store(false, AtomicOrdering::Relaxed);
        self.needs_rebuild
            .store(self.pending_len() >= step as usize, AtomicOrdering::Relaxed);
        if published {
            self.metrics
                .record_index_build(IndexTier::Ivf, build_started.elapsed());
            tracing::info!(collection = %name, points = total, "IVF segment indexes published");
        }
        Ok(published)
    }

    /// Drop the published index and return to exact scan.
    pub fn drop_index(&self) -> Result<()> {
        let files = self.data_arc();
        let _guard = self.compact_mutex.lock();
        files.hnsw_segments.store(Arc::new(Vec::new()));
        files.ivf_segments.store(Arc::new(Vec::new()));
        if let Ok(entries) = std::fs::read_dir(files.dir()) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if segment_hnsw::is_segment_file(&name)
                    || segment_ivf::is_segment_file(&name)
                    || super::quant::is_segment_file(&name)
                {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        self.pending.lock().clear();
        self.building.store(false, AtomicOrdering::Relaxed);
        self.needs_rebuild.store(false, AtomicOrdering::Relaxed);
        Ok(())
    }

    /// Drain pending slots into the published HNSW index in batch.
    ///
    /// Called by the maintenance worker on each idle tick.  Acquires only the
    /// `build_mutex` (not the store write lock) so concurrent upserts and
    /// searches are unblocked; new slots that arrive during the drain are
    /// left for the next tick.  If no HNSW index is published or pending
    /// is empty this is a no-op.
    pub(crate) fn drain_pending_hnsw(&self) {
        let ivf_config = match &self.inner.read().meta.index {
            IndexConfig::Ivf(Some(config)) => Some(config.clone()),
            _ => None,
        };
        if let Some(config) = ivf_config {
            self.drain_pending_ivf(config);
            return;
        }
        let files = self.data_arc();
        let (dim, metric, segment_slots, next_slot, name, config) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.segment_slots,
                inner.meta.next_slot,
                inner.meta.collection.clone(),
                match &inner.meta.index {
                    IndexConfig::Hnsw(cfg) => cfg.clone(),
                    _ => HnswConfig::default(),
                },
            )
        };
        let boundary = segments::immutable_end(next_slot, segment_slots);
        let pending: Vec<u32> = {
            let mut p = self.pending.lock();
            if p.is_empty() {
                return;
            }
            std::mem::take(&mut *p)
        };
        let step = segment_slots.max(1);
        let _guard = self.build_mutex.lock();
        let mut graphs = (**files.hnsw_segments.load()).clone();
        let vsnap = files.vectors.snapshot();
        let mut rebuild_starts: Vec<u32> = Vec::new();
        let mut retained_on_failure: Vec<u32> = Vec::new();

        for &slot in &pending {
            if (slot as u64) >= boundary as u64 {
                continue;
            }
            let local_segment = slot / step * step;
            if local_segment == 0 && slot < step {
                // The first segment is still growing: incremental adjacency
                // edits keep the graph live under concurrent reads.
                if let Some(target) = graphs.iter_mut().find(|s| s.start == 0) {
                    if let Some(vector) =
                        Vectors::read_slot(&vsnap, slot as u64, segment_slots, dim)
                    {
                        target.index.insert(
                            slot - target.start,
                            vector,
                            &vsnap,
                            segment_slots,
                            target.start,
                        );
                    }
                    target.end = target.end.max(slot + 1);
                    continue;
                }
            }
            if !rebuild_starts.contains(&local_segment) {
                rebuild_starts.push(local_segment);
            }
        }

        for start in rebuild_starts {
            let end = (start + step).min(boundary);
            match self.rebuild_segment_hnsw(start, end, &name, dim, metric, segment_slots, &config)
            {
                Ok(Some(segment)) => {
                    if let Some(existing) = graphs.iter_mut().find(|s| s.start == start) {
                        *existing = segment;
                    } else {
                        graphs.push(segment);
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "HNSW segment rebuild failed");
                    retained_on_failure.extend(
                        pending
                            .iter()
                            .copied()
                            .filter(|slot| *slot >= start && *slot < end),
                    );
                }
            }
        }
        graphs.sort_by_key(|segment| segment.start);
        files.hnsw_segments.store(Arc::new(graphs));

        let retained: Vec<u32> = pending
            .into_iter()
            .filter(|slot| (*slot as u64) >= boundary as u64 || retained_on_failure.contains(slot))
            .collect();
        let retained_len = retained.len();
        self.pending.lock().extend(retained);
        self.needs_rebuild
            .store(retained_len >= step as usize, AtomicOrdering::Relaxed);
        tracing::debug!(
            collection = %name,
            retained_len,
            "HNSW appendable tail retained for exact scan"
        );
    }

    /// Incorporate pending slots into segment-local IVF indexes. Completed
    /// segments are rebuilt; the still-growing first segment keeps taking
    /// incremental assignments, matching the HNSW drain's visibility rules.
    fn drain_pending_ivf(&self, config: IvfConfig) {
        let files = self.data_arc();
        let (dim, metric, segment_slots, next_slot, name) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.segment_slots,
                inner.meta.next_slot,
                inner.meta.collection.clone(),
            )
        };
        let boundary = segments::immutable_end(next_slot, segment_slots);
        let pending: Vec<u32> = {
            let mut p = self.pending.lock();
            if p.is_empty() {
                return;
            }
            std::mem::take(&mut *p)
        };
        let step = segment_slots.max(1);
        let _guard = self.build_mutex.lock();
        let mut segments = (**files.ivf_segments.load()).clone();
        let mut rebuild_starts: Vec<u32> = Vec::new();
        let mut failed: Vec<u32> = Vec::new();

        for &slot in &pending {
            if (slot as u64) >= boundary as u64 {
                continue;
            }
            let local_segment = slot / step * step;
            if local_segment == 0 && slot < step {
                if let Some(target) = segments.iter_mut().find(|s| s.start == 0) {
                    if let Some(v) = Vectors::read_slot(
                        &files.vectors.snapshot(),
                        slot as u64,
                        segment_slots,
                        dim,
                    ) {
                        target.index.assign_slot(slot - target.start, v);
                        target.index.note_upsert();
                    }
                    target.end = target.end.max(slot + 1);
                    continue;
                }
            }
            if !rebuild_starts.contains(&local_segment) {
                rebuild_starts.push(local_segment);
            }
        }
        for start in rebuild_starts {
            let end = (start + step).min(boundary);
            match self.rebuild_segment_ivf(start, end, &name, dim, metric, segment_slots, &config) {
                Ok(Some(segment)) => {
                    if let Some(existing) = segments.iter_mut().find(|s| s.start == start) {
                        *existing = segment;
                    } else {
                        segments.push(segment);
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "IVF segment rebuild failed");
                    failed.extend(
                        pending
                            .iter()
                            .copied()
                            .filter(|slot| *slot >= start && *slot < end),
                    );
                }
            }
        }
        segments.sort_by_key(|segment| segment.start);
        files.ivf_segments.store(Arc::new(segments));
        let retained: Vec<u32> = pending
            .into_iter()
            .filter(|slot| (*slot as u64) >= boundary as u64 || failed.contains(slot))
            .collect();
        let retained_len = retained.len();
        self.pending.lock().extend(retained);
        self.needs_rebuild
            .store(retained_len >= step as usize, AtomicOrdering::Relaxed);
    }

    /// Whether an index is published.
    pub fn has_index(&self) -> bool {
        let files = self.data_arc();
        !files.hnsw_segments.load().is_empty() || !files.ivf_segments.load().is_empty()
    }

    /// Number of slots waiting in the pending queue (not yet incorporated
    /// into the published index). Used by the sweep to schedule rebuilds
    /// when the index is published but has pending gaps.
    pub(crate) fn pending_len(&self) -> usize {
        self.pending.lock().len()
    }

    /// Swap-and-reset the post-compaction rebuild flag.
    pub(crate) fn take_needs_rebuild(&self) -> bool {
        self.needs_rebuild.swap(false, AtomicOrdering::Relaxed)
    }

    pub(crate) fn ivf_config_opt(&self) -> Option<IvfConfig> {
        let inner = self.inner.read();
        if !matches!(inner.meta.index, IndexConfig::Ivf(_)) {
            return None;
        }
        inner.ivf_config.clone()
    }

    /// Effective HNSW configuration, or `None` when the collection is not an
    /// HNSW collection.
    pub(crate) fn hnsw_config_opt(&self) -> Option<HnswConfig> {
        let inner = self.inner.read();
        match &inner.meta.index {
            IndexConfig::Hnsw(cfg) => Some(cfg.clone()),
            _ => None,
        }
    }

    /// Measure the current drift ratio of the published IVF index.
    pub(crate) fn measure_drift(&self, index: &IvfIndex) -> f64 {
        let files = self.data_arc();
        let (next_slot, segment_slots) = {
            let inner = self.inner.read();
            (inner.meta.next_slot as u32, inner.meta.segment_slots)
        };
        let samples = IvfIndex::sample_plan(index.config().sample_limit, next_slot);
        if samples.is_empty() {
            return 0.0;
        }
        let tombstones = files.tombstones.load();
        let vsnap = files.vectors.snapshot();
        index.drift_ratio(&samples, &tombstones, &vsnap, segment_slots, 0)
    }

    /// Published IVF index plus its configuration, for the maintenance
    /// worker's drift checks. HNSW needs no equivalent: it has no global
    /// training state to drift away from.
    pub(crate) fn ivf_state(&self) -> Option<(Arc<IvfIndex>, IvfConfig)> {
        let files = self.data_arc();
        let segments = files.ivf_segments.load();
        let Some(segment) = segments.first() else {
            return None;
        };
        let config = self.inner.read().ivf_config.clone()?;
        Some((Arc::clone(&segment.index), config))
    }

    pub(crate) fn record_drift(&self, ratio: f64) {
        self.inner.write().last_drift_ratio = Some(ratio);
    }

    /// Current index state for [`CollectionInfo`](crate::types::CollectionInfo).
    pub fn index_info(&self) -> Option<IndexInfo> {
        let files = self.data_arc();
        let (building, build_inserted, build_points_total) = self.build_progress();
        let hnsw_segments = files.hnsw_segments.load();
        if !hnsw_segments.is_empty() {
            let nodes: u64 = hnsw_segments
                .iter()
                .map(|segment| segment.index.built_at_live_count())
                .sum();
            let overwrites: u64 = hnsw_segments
                .iter()
                .map(|segment| segment.index.overwrites_since_build())
                .sum();
            let stale = hnsw_segments
                .iter()
                .map(|segment| segment.index.stale_ratio())
                .fold(0.0f64, f64::max);
            let first = &hnsw_segments[0].index;
            return Some(IndexInfo {
                index_kind: IndexKind::Hnsw,
                lists: 0,
                nprobe_default: 0,
                m: first.m(),
                ef_construct: first.ef_construct(),
                ef_search_default: first.default_ef(),
                built_at_live_count: nodes,
                stale_overwrite_count: overwrites,
                stale_ratio: Some(stale),
                last_drift_ratio: None,
                building,
                build_inserted,
                build_points_total,
            });
        }
        let ivf_segments = files.ivf_segments.load();
        if !ivf_segments.is_empty() {
            let first = &ivf_segments[0].index;
            let lists: usize = ivf_segments
                .iter()
                .map(|segment| segment.index.list_count())
                .sum();
            return Some(IndexInfo {
                index_kind: IndexKind::Ivf,
                lists: lists as u32,
                nprobe_default: first.default_nprobe(),
                m: 0,
                ef_construct: 0,
                ef_search_default: 0,
                built_at_live_count: ivf_segments
                    .iter()
                    .map(|segment| segment.index.built_at_live_count())
                    .sum(),
                stale_overwrite_count: 0,
                stale_ratio: None,
                last_drift_ratio: self.inner.read().last_drift_ratio,
                building,
                build_inserted,
                build_points_total,
            });
        }
        // Nothing published: report a placeholder when an ANN tier is
        // configured, `None` for permanent exact-scan collections.
        let inner = self.inner.read();
        if matches!(inner.meta.index, IndexConfig::Flat) {
            None
        } else {
            Some(IndexInfo {
                index_kind: IndexKind::ExactScan,
                lists: 0,
                nprobe_default: 0,
                m: 0,
                ef_construct: 0,
                ef_search_default: 0,
                built_at_live_count: 0,
                stale_overwrite_count: 0,
                stale_ratio: None,
                last_drift_ratio: None,
                building,
                build_inserted,
                build_points_total,
            })
        }
    }
}
