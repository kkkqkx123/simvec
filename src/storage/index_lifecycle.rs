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
//! - The published index is stored before clearing `building`: an upsert that
//!   loaded `index == None` and then observed `building == false` would
//!   record its slot nowhere; with this order it still lands in `pending`,
//!   which every publication path drains.

use std::collections::HashSet;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::Arc;
use std::time::Instant;

use super::vectors::Vectors;
use super::CollectionStore;
use crate::error::Result;
use crate::index::{persist, HnswIndex, IndexBuildParams, IvfIndex};
use crate::metrics::IndexTier;
use crate::types::{HnswConfig, IndexInfo, IndexType, IvfConfig};

/// A published ANN index; `None` = exact scan. Both tiers are derived
/// structures over `vectors.bin` and swap in atomically.
pub(super) enum PublishedIndex {
    Ivf(Arc<IvfIndex>),
    Hnsw(Arc<HnswIndex>),
}

impl CollectionStore {
    /// Supply the IVF configuration after open. Collections persist their
    /// effective config, but the engine-provided
    /// runtime configuration takes precedence. Applies both to future builds
    /// and to any published index, so a rehydrated index immediately honors
    /// the runtime config.
    pub fn set_ivf_config(&self, config: IvfConfig) {
        self.inner.write().ivf_config = Some(config.clone());
        if let Some(PublishedIndex::Ivf(index)) = self.index.load().as_ref() {
            index.set_config(config);
        }
    }

    /// Supply the HNSW configuration after open. Unlike the IVF counterpart
    /// this persists to `meta.bin` immediately: the published graph bakes its
    /// parameters in at build time, so changes only affect the next rebuild
    /// and memory must stay consistent with disk.
    pub fn set_hnsw_config(&self, config: HnswConfig) -> Result<()> {
        config.validate()?;
        let mut inner = self.inner.write();
        inner.meta.hnsw_config = Some(config);
        inner.meta.save(&self.dir)
    }

    /// Rehydrate the persisted ANN tier matching the collection's index type
    /// if present and consistent with the live metadata; otherwise fall back
    /// to exact scan. Slots appended after the build (WAL replay) land in
    /// `pending` so approximate searches still see them.
    pub(super) fn load_index(&self) -> Result<()> {
        match self.inner.read().meta.index_type {
            IndexType::IVF => self.load_ivf(),
            IndexType::HNSW => self.load_hnsw(),
            IndexType::FLAT => Ok(()),
        }
    }

    fn load_ivf(&self) -> Result<()> {
        let index_path = self.dir.join("index.bin");
        let existed = index_path.exists();
        let Some(data) = persist::load(&self.dir)? else {
            if existed {
                tracing::info!("vector index.bin corrupt; falling back to exact scan");
                self.metrics.record_index_load_fallback();
            }
            return Ok(());
        };
        let (dim, distance, next_slot, capacity) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.next_slot,
                inner.meta.slot_capacity as usize,
            )
        };
        if !data.valid_for(dim, distance, next_slot) {
            tracing::info!("vector index.bin inconsistent with meta; falling back to exact scan");
            self.metrics.record_index_load_fallback();
            return Ok(());
        }
        let config = self.inner.read().ivf_config.clone().unwrap_or_default();
        let covered = data.slot_list.len();
        let mut index = IvfIndex::from_persisted(data, config);
        index.set_metrics(self.metrics_arc());
        let index = Arc::new(index);
        {
            let tombstones = self.tombstones.load();
            let mut p = (**self.pending.load()).clone();
            for slot in covered..next_slot as usize {
                if slot < capacity && !tombstones.bit(slot) {
                    p.push(slot as u32);
                }
            }
            self.pending.store(Arc::new(p));
        }
        self.index.store(Arc::new(Some(PublishedIndex::Ivf(index))));
        Ok(())
    }

    fn load_hnsw(&self) -> Result<()> {
        let hnsw_path = self.dir.join("hnsw.bin");
        let existed = hnsw_path.exists();
        let Some(data) = persist::load_hnsw(&self.dir)? else {
            if existed {
                tracing::info!("vector hnsw.bin corrupt; falling back to exact scan");
                self.metrics.record_index_load_fallback();
            }
            return Ok(());
        };
        let (dim, distance, next_slot, capacity) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.next_slot,
                inner.meta.slot_capacity as usize,
            )
        };
        if !data.valid_for(dim, distance, next_slot) {
            tracing::info!("vector hnsw.bin inconsistent with meta; falling back to exact scan");
            self.metrics.record_index_load_fallback();
            return Ok(());
        }
        let config = self
            .inner
            .read()
            .meta
            .hnsw_config
            .clone()
            .unwrap_or_default();
        let covered = data.nodes.len();
        let index = match HnswIndex::from_persisted(data, &config) {
            Ok(mut index) => {
                index.set_metrics(self.metrics_arc());
                Arc::new(index)
            }
            Err(e) => {
                tracing::info!(
                    error = %e,
                    "vector hnsw.bin failed structural validation; falling back to exact scan"
                );
                self.metrics.record_index_load_fallback();
                return Ok(());
            }
        };
        {
            let tombstones = self.tombstones.load();
            let mut p = (**self.pending.load()).clone();
            for slot in covered..next_slot as usize {
                if slot < capacity && !tombstones.bit(slot) {
                    p.push(slot as u32);
                }
            }
            if !p.is_empty() {
                self.needs_rebuild.store(true, AtomicOrdering::Relaxed);
            }
            self.pending.store(Arc::new(p));
        }
        self.index
            .store(Arc::new(Some(PublishedIndex::Hnsw(index))));
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
    /// scored by brute force in `search_hnsw` for correct visibility, and
    /// drained into the graph by the maintenance worker on each tick.
    pub(super) fn register_slot(&self, slot: u32, vector: &[f32], _segment_slots: u32) {
        let published = self.index.load();
        match published.as_ref() {
            Some(PublishedIndex::Ivf(index)) => {
                index.assign_slot(slot, vector);
                index.note_upsert();
                return;
            }
            Some(PublishedIndex::Hnsw(_)) => {
                // Defer graph insertion to the maintenance worker; the
                // brute-force pending path in search_hnsw guarantees
                // correct visibility in the interim.
                let mut p = (**self.pending.load()).clone();
                p.push(slot);
                self.pending.store(Arc::new(p));
                return;
            }
            None => {}
        }
        drop(published);
        if self.building.load(AtomicOrdering::Relaxed) {
            let mut p = (**self.pending.load()).clone();
            p.push(slot);
            self.pending.store(Arc::new(p));
        }
    }

    /// Discard every persisted index artifact (both tiers).
    pub(super) fn discard_index_files(&self) {
        persist::discard(&self.dir.join("index.bin"));
        persist::discard(&self.dir.join("hnsw.bin"));
    }

    /// Build and publish the configured ANN tier from the current live set.
    /// Dispatches on the collection's `index_type`; returns whether a usable
    /// index is now published. When quantization is configured this also
    /// ensures `quant.bin` is trained before the index is built, under the
    /// same `build_mutex` so slot numbers stay stable.
    pub fn build_index(&self) -> Result<bool> {
        // Train quantization first if needed (especially Product which needs a
        // codebook before HNSW/IVF can be usefully built). Scalar/Binary are
        // idempotent after incremental writes; rebuilding still refreshes the
        // global scale range.
        {
            let needs = {
                let inner = self.inner.read();
                inner
                    .meta
                    .quantization_config
                    .as_ref()
                    .is_some_and(|qc| qc.enabled && qc.quant_type.is_some())
            };
            let ready = self.has_quantization();
            if needs && !ready {
                // Product quantization not yet trained, or scalar missing.
                let _ = self.build_quantization();
            } else if needs {
                // For scalar, periodically refresh the scale even when ready
                // if live set has grown significantly. For now trigger on
                // every build to keep the range accurate without extra cost
                // for small collections; larger collections grow via background
                // sweeps. Keep product rebuild lazy (already ready) to avoid
                // repeated k-means.
                let qc = self.inner.read().meta.quantization_config.clone().unwrap();
                if matches!(
                    qc.quant_type,
                    Some(crate::types::QuantizationType::Scalar { .. })
                ) {
                    let _ = self.build_quantization();
                }
            }
        }
        let inner = self.inner.read();
        let kind = inner.meta.index_type;
        let ivf_config = inner.ivf_config.clone();
        let hnsw_config = inner.meta.hnsw_config.clone();
        drop(inner);
        match kind {
            IndexType::HNSW => match hnsw_config {
                Some(config) => self.build_hnsw_index(config),
                None => Ok(false),
            },
            IndexType::IVF => match ivf_config {
                Some(config) => self.build_ivf_index(config),
                None => Ok(false),
            },
            IndexType::FLAT => Ok(false),
        }
    }

    /// Build and publish an HNSW index from the current live set.
    ///
    /// Heavy work runs off the store lock under the `build_mutex`, so
    /// compaction cannot interleave and slot numbers stay stable. Slots
    /// appended while building are adopted through incremental inserts under
    /// the store write lock before publication, so approximate search never
    /// misses a point.
    fn build_hnsw_index(&self, config: HnswConfig) -> Result<bool> {
        let _compact_guard = self.compact_mutex.lock();
        let _guard = self.build_mutex.lock();

        let (dim, metric, segment_slots, snapshot_next_slot, name) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.segment_slots,
                inner.meta.next_slot,
                inner.meta.collection.clone(),
            )
        };
        let live: Vec<u32> = {
            let tombstones = self.tombstones.load();
            (0..snapshot_next_slot as u32)
                .filter(|&s| !tombstones.bit(s as usize))
                .collect()
        };
        if live.is_empty() {
            return Ok(false);
        }

        self.building.store(true, AtomicOrdering::Relaxed);
        self.build_points_total
            .store(live.len() as u64, AtomicOrdering::Relaxed);
        self.build_inserted.store(0, AtomicOrdering::Relaxed);
        tracing::info!(
            collection = %name,
            points = live.len(),
            "building HNSW index"
        );
        let build_started = Instant::now();
        let built = {
            let vsnap = self.vectors.snapshot();
            HnswIndex::build(
                &IndexBuildParams::new(&name, dim, metric, &live, &vsnap, segment_slots)
                    .with_progress(&self.build_inserted),
                &config,
            )
        };

        let index = match built {
            Ok(mut index) => {
                index.set_metrics(self.metrics_arc());
                Arc::new(index)
            }
            Err(e) => {
                self.building.store(false, AtomicOrdering::Relaxed);
                tracing::warn!(collection = %name, error = %e, "HNSW build failed");
                return Err(e);
            }
        };

        // Adopt everything that raced the build: slots appended past the
        // snapshot and pending slots recorded while no index was published.
        // Holds the store write lock (acquired before the pending lock,
        // matching `register_slot`'s order), so concurrent upserts either ran
        // before (and are adopted here) or run after (and see the published
        // graph directly).
        //
        // Slots appended during the build sit in both the appended range and
        // `pending`; the range pass wins and the pending drain skips them, so
        // each node is inserted exactly once (a repeated insert would be
        // miscounted as an overwrite-staleness event).
        {
            let inner = self.inner.write();
            let mut p = (**self.pending.load()).clone();
            let tombstones = self.tombstones.load();
            let vsnap = self.vectors.snapshot();
            let mut adopted: HashSet<u32> = HashSet::new();
            for slot in snapshot_next_slot..inner.meta.next_slot {
                let slot = slot as u32;
                if tombstones.bit(slot as usize) {
                    continue;
                }
                if let Some(v) = Vectors::read_slot(&vsnap, slot as u64, segment_slots, dim) {
                    index.insert(slot, v, &vsnap, segment_slots);
                    adopted.insert(slot);
                }
            }
            for slot in p.drain(..) {
                if adopted.contains(&slot) || tombstones.bit(slot as usize) {
                    continue;
                }
                if let Some(v) = Vectors::read_slot(&vsnap, slot as u64, segment_slots, dim) {
                    index.insert(slot, v, &vsnap, segment_slots);
                }
            }
            self.pending.store(Arc::new(p));
        }
        let persisted = index.to_persisted();
        // Publish before clearing the building flag: an upsert that loaded
        // `index == None` and then observed `building == false` would record
        // its slot nowhere and go missing from approximate searches. With
        // this order such an upsert still sees `building == true` and lands
        // in `pending`, which every published-index adoption path drains.
        self.index
            .store(Arc::new(Some(PublishedIndex::Hnsw(index))));
        self.build_inserted.store(
            self.build_points_total.load(AtomicOrdering::Relaxed),
            AtomicOrdering::Relaxed,
        );
        self.building.store(false, AtomicOrdering::Relaxed);
        self.needs_rebuild.store(false, AtomicOrdering::Relaxed);

        if let Err(e) = persist::save_hnsw(&self.dir, &persisted) {
            tracing::warn!(collection = %name, error = %e, "hnsw.bin save failed");
        }
        self.metrics
            .record_index_build(IndexTier::Hnsw, build_started.elapsed());
        tracing::info!(
            collection = %name,
            nodes = persisted.nodes.len(),
            "HNSW index published"
        );
        Ok(true)
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
        let _compact_guard = self.compact_mutex.lock();
        let _guard = self.build_mutex.lock();

        let (dim, metric, segment_slots, snapshot_next_slot, name) = {
            let inner = self.inner.read();
            (
                inner.meta.vector_size,
                inner.meta.distance,
                inner.meta.segment_slots,
                inner.meta.next_slot,
                inner.meta.collection.clone(),
            )
        };
        let live: Vec<u32> = {
            let tombstones = self.tombstones.load();
            (0..snapshot_next_slot as u32)
                .filter(|&s| !tombstones.bit(s as usize))
                .collect()
        };
        if (live.len() as u64) < config.min_build_points.max(1) || live.is_empty() {
            return Ok(false);
        }

        self.building.store(true, AtomicOrdering::Relaxed);
        self.build_points_total
            .store(live.len() as u64, AtomicOrdering::Relaxed);
        self.build_inserted.store(0, AtomicOrdering::Relaxed);
        tracing::info!(
            collection = %name,
            points = live.len(),
            "building IVF index"
        );

        let build_started = Instant::now();
        let built = {
            let vsnap = self.vectors.snapshot();
            IvfIndex::build(
                &IndexBuildParams::new(&name, dim, metric, &live, &vsnap, segment_slots)
                    .with_progress(&self.build_inserted),
                &config,
            )
        };

        let index = match built {
            Ok(mut index) => {
                index.set_metrics(self.metrics_arc());
                Arc::new(index)
            }
            Err(e) => {
                self.building.store(false, AtomicOrdering::Relaxed);
                tracing::warn!(collection = %name, error = %e, "IVF build failed");
                return Err(e);
            }
        };

        // Publish atomically with respect to writers: everything below holds
        // the store write lock (acquired before the pending lock, matching
        // `register_slot`'s order) so concurrent upserts either ran before
        // (and their slots are adopted/drained here) or run after (and see
        // the published index directly).
        let persisted = {
            let inner = self.inner.write();
            let mut p = (**self.pending.load()).clone();
            let tombstones = self.tombstones.load();
            let vsnap = self.vectors.snapshot();
            index.adopt_range(
                snapshot_next_slot as u32,
                inner.meta.next_slot as u32,
                &tombstones,
                &vsnap,
                segment_slots,
            );
            for slot in p.drain(..) {
                if !tombstones.bit(slot as usize) {
                    if let Some(v) = Vectors::read_slot(&vsnap, slot as u64, segment_slots, dim) {
                        index.assign_slot(slot, v);
                    }
                }
            }
            self.pending.store(Arc::new(p));
            index.to_persisted()
        };
        // Publish before clearing the building flag: an upsert that loaded
        // `index == None` and then observed `building == false` would record
        // its slot nowhere and go missing from probe searches. With this
        // order such an upsert still sees `building == true` and lands in
        // `pending`, which every probe search scans unconditionally.
        self.index.store(Arc::new(Some(PublishedIndex::Ivf(index))));
        self.build_inserted.store(
            self.build_points_total.load(AtomicOrdering::Relaxed),
            AtomicOrdering::Relaxed,
        );
        self.building.store(false, AtomicOrdering::Relaxed);
        self.needs_rebuild.store(false, AtomicOrdering::Relaxed);

        if let Err(e) = persist::save(&self.dir, &persisted) {
            tracing::warn!(collection = %name, error = %e, "index.bin save failed");
        }
        self.metrics
            .record_index_build(IndexTier::Ivf, build_started.elapsed());
        tracing::info!(
            collection = %name,
            lists = persisted.lists,
            "IVF index published"
        );
        Ok(true)
    }

    /// Drop the published index and return to exact scan.
    pub fn drop_index(&self) -> Result<()> {
        let _guard = self.compact_mutex.lock();
        self.index.store(Arc::new(None));
        self.pending.store(Arc::new(Vec::new()));
        self.building.store(false, AtomicOrdering::Relaxed);
        self.needs_rebuild.store(false, AtomicOrdering::Relaxed);
        self.discard_index_files();
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
        let published = self.index.load();
        let Some(PublishedIndex::Hnsw(index)) = published.as_ref() else {
            return;
        };
        let pending: Vec<u32> = {
            let p = self.pending.load();
            if p.is_empty() {
                return;
            }
            // Take a snapshot and clear; new arrivals during drain go to
            // the next tick.
            let taken = (**p).clone();
            self.pending.store(Arc::new(Vec::new()));
            taken
        };
        // Hold build_mutex only — excludes compaction and build, so the
        // graph structure is stable.  The store write lock is NOT held,
        // so upserts and searches continue concurrently.
        let _guard = self.build_mutex.lock();
        let (dim, segment_slots) = {
            let inner = self.inner.read();
            (inner.meta.vector_size, inner.meta.segment_slots)
        };
        let tombstones = self.tombstones.load();
        let vsnap = self.vectors.snapshot();
        let mut drained = 0u64;
        for slot in pending {
            if tombstones.bit(slot as usize) {
                continue;
            }
            if let Some(v) = Vectors::read_slot(&vsnap, slot as u64, segment_slots, dim) {
                index.insert(slot, v, &vsnap, segment_slots);
                drained += 1;
            }
        }
        if drained > 0 {
            tracing::debug!(
                collection = %self.inner.read().meta.collection,
                drained,
                "HNSW pending slots drained"
            );
        }
    }

    /// Whether an index is published.
    pub fn has_index(&self) -> bool {
        self.index.load().is_some()
    }

    /// Number of slots waiting in the pending queue (not yet incorporated
    /// into the published index). Used by the sweep to schedule rebuilds
    /// when the index is published but has pending gaps.
    pub(crate) fn pending_len(&self) -> usize {
        self.pending.load().len()
    }

    /// Swap-and-reset the post-compaction rebuild flag.
    pub(crate) fn take_needs_rebuild(&self) -> bool {
        self.needs_rebuild.swap(false, AtomicOrdering::Relaxed)
    }

    pub(crate) fn ivf_config_opt(&self) -> Option<IvfConfig> {
        self.inner.read().ivf_config.clone()
    }

    /// Effective HNSW configuration, or `None` when the collection is not an
    /// HNSW collection.
    pub(crate) fn hnsw_config_opt(&self) -> Option<HnswConfig> {
        let inner = self.inner.read();
        if inner.meta.index_type != IndexType::HNSW {
            return None;
        }
        inner.meta.hnsw_config.clone()
    }

    /// Measure the current drift ratio of the published IVF index.
    pub(crate) fn measure_drift(&self, index: &IvfIndex) -> f64 {
        let (next_slot, segment_slots) = {
            let inner = self.inner.read();
            (inner.meta.next_slot as u32, inner.meta.segment_slots)
        };
        let samples = IvfIndex::sample_plan(index.config().sample_limit, next_slot);
        if samples.is_empty() {
            return 0.0;
        }
        let tombstones = self.tombstones.load();
        let vsnap = self.vectors.snapshot();
        index.drift_ratio(&samples, &tombstones, &vsnap, segment_slots)
    }

    /// Published IVF index plus its configuration, for the maintenance
    /// worker's drift checks. HNSW needs no equivalent: it has no global
    /// training state to drift away from.
    pub(crate) fn ivf_state(&self) -> Option<(Arc<IvfIndex>, IvfConfig)> {
        let published = self.index.load();
        let Some(PublishedIndex::Ivf(index)) = published.as_ref() else {
            return None;
        };
        let config = self.inner.read().ivf_config.clone()?;
        Some((Arc::clone(index), config))
    }

    pub(crate) fn record_drift(&self, ratio: f64) {
        self.inner.write().last_drift_ratio = Some(ratio);
    }

    /// Current index state for [`CollectionInfo`](crate::types::CollectionInfo).
    pub fn index_info(&self) -> Option<IndexInfo> {
        let (building, build_inserted, build_points_total) = self.build_progress();
        let published = self.index.load();
        match published.as_ref() {
            Some(PublishedIndex::Ivf(index)) => Some(IndexInfo {
                index_kind: 1,
                lists: index.list_count() as u32,
                nprobe_default: index.default_nprobe(),
                m: 0,
                ef_construct: 0,
                ef_search_default: 0,
                built_at_live_count: index.built_at_live_count(),
                stale_overwrite_count: 0,
                stale_ratio: None,
                last_drift_ratio: self.inner.read().last_drift_ratio,
                building,
                build_inserted,
                build_points_total,
            }),
            Some(PublishedIndex::Hnsw(index)) => Some(IndexInfo {
                index_kind: 2,
                lists: 0,
                nprobe_default: 0,
                m: index.m(),
                ef_construct: index.ef_construct(),
                ef_search_default: index.default_ef(),
                built_at_live_count: index.built_at_live_count(),
                stale_overwrite_count: index.overwrites_since_build(),
                stale_ratio: Some(index.stale_ratio()),
                last_drift_ratio: None,
                building,
                build_inserted,
                build_points_total,
            }),
            None => {
                // Nothing published: report a placeholder when an ANN tier is
                // configured, `None` for permanent exact-scan collections.
                let inner = self.inner.read();
                if inner.meta.index_type == IndexType::FLAT {
                    None
                } else {
                    Some(IndexInfo {
                        index_kind: 0,
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
    }
}
