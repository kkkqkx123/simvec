use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::{Builder as ThreadBuilder, JoinHandle};
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};

use super::*;
use super::{LocalVectorEngine, MaintenanceJob, TxnOp};
use crate::error::{Result, VectorSearchError};
use crate::metrics::MetricsSnapshot;
use crate::storage::{CollectionStore, WalPoint, WalRecord};
use crate::types::{
    CollectionConfig, CollectionInfo, CollectionStatus, HnswConfig, IndexConfig, IvfConfig,
    Payload, PayloadSchemaType, PointId, SearchQuery, SearchResult, VectorFilter, VectorPoint,
};

impl LocalVectorEngine {
    /// Build and publish the IVF index of a collection synchronously.
    /// Returns whether a usable index is now published. This is also the
    /// entry point used by the maintenance worker.
    pub fn build_index(&self, collection: &str) -> Result<bool> {
        let store = self.store(collection)?;
        store.build_index()
    }

    /// Build or rebuild quantized storage for a collection.
    ///
    /// Scalar quantization refreshes the global min/max/scale; Binary is a
    /// no-op (bits are already per-vector); Product retrains `M` codebooks
    /// (256 centroids per subspace via k-means) and recodes every live vector.
    /// Returns whether quantization is now ready for search.
    pub fn build_quantization(&self, collection: &str) -> Result<bool> {
        let store = self.store(collection)?;
        store.build_quantization()
    }

    /// Whether quantization is active and ready for the collection.
    pub fn has_quantization(&self, collection: &str) -> bool {
        self.store(collection).is_ok_and(|s| s.has_quantization())
    }

    /// Drop the published IVF/HNSW index structures only; the collection data
    /// remains and queries fall back to exact scan. This is not a data clear.
    /// To delete points use collection-level deletion; to delete the whole
    /// collection and its directory use [`Engine::delete_collection`].
    pub fn drop_index(&self, collection: &str) -> Result<()> {
        self.store(collection)?.drop_index()
    }

    /// Alias making the non-destructive semantics explicit.
    pub fn drop_published_index(&self, collection: &str) -> Result<()> {
        self.drop_index(collection)
    }

    /// Whether an IVF index is published for the collection.
    pub fn has_index(&self, collection: &str) -> bool {
        self.store(collection).is_ok_and(|s| s.has_index())
    }

    /// Manually compact a collection (remove tombstoned slots).
    pub fn compact_collection(&self, collection: &str) -> Result<u64> {
        self.store(collection)?.compact()
    }

    /// Drain slots waiting in the HNSW pending queue into the published
    /// graph immediately, instead of waiting for the next maintenance tick.
    /// No-op when no HNSW index is published or nothing is pending.
    pub fn drain_pending(&self, collection: &str) -> Result<()> {
        self.store(collection)?.drain_pending_hnsw();
        Ok(())
    }

    /// Operational metrics snapshot for a collection.
    pub fn collection_metrics(&self, collection: &str) -> Result<MetricsSnapshot> {
        Ok(self.store(collection)?.metrics().snapshot())
    }

    /// Schedule a background compaction for `collection` once its tombstone
    /// ratio crosses the threshold. Deduplicated through the in-flight set so
    /// bursts of mutations do not flood the maintenance queue.
    pub(super) fn maybe_schedule_compaction(&self, collection: &str, store: &CollectionStore) {
        let Some(jobs) = self.jobs.as_ref() else {
            return;
        };
        if !store.needs_compaction() {
            return;
        }
        let key = format!("compact:{collection}");
        let mut guard = self.in_flight.lock();
        if guard.contains(&key) {
            return;
        }
        guard.insert(key.clone());
        if jobs
            .send(MaintenanceJob::Compact(collection.to_string()))
            .is_err()
        {
            guard.remove(&key);
        }
    }

    /// Schedule a background WAL checkpoint once the log outgrows its byte
    /// threshold. Insert-heavy workloads never cross the tombstone ratio, so
    /// this is the only trigger that bounds log growth there.
    pub(super) fn maybe_schedule_checkpoint(&self, collection: &str, store: &CollectionStore) {
        let Some(jobs) = self.jobs.as_ref() else {
            return;
        };
        if !store.needs_checkpoint() {
            return;
        }
        let key = format!("checkpoint:{collection}");
        let mut guard = self.in_flight.lock();
        if guard.contains(&key) {
            return;
        }
        guard.insert(key.clone());
        if jobs
            .send(MaintenanceJob::Checkpoint(collection.to_string()))
            .is_err()
        {
            guard.remove(&key);
        }
    }

    /// Ask the maintenance worker to drain a large HNSW pending backlog. The
    /// drain is incremental and off the caller's thread; the previous
    /// synchronous guardrail stalled a single batch for seconds.
    pub(super) fn maybe_schedule_drain(&self, collection: &str, store: &CollectionStore) {
        let Some(jobs) = self.jobs.as_ref() else {
            return;
        };
        if store.pending_len() < PENDING_DRAIN_ENQUEUE {
            return;
        }
        let key = format!("drain:{collection}");
        let mut guard = self.in_flight.lock();
        if guard.contains(&key) {
            return;
        }
        guard.insert(key.clone());
        if jobs
            .send(MaintenanceJob::Drain(collection.to_string()))
            .is_err()
        {
            guard.remove(&key);
        }
    }

    /// Post-mutation hook, run after the store locks are released: keep the
    /// pending backlog bounded by scheduling a background drain, schedule a
    /// compaction or WAL checkpoint when either threshold is crossed, and
    /// incrementally repair the HNSW graph if it has stale references to
    /// tombstoned slots.
    pub(super) fn after_mutation(&self, collection: &str, store: &Arc<CollectionStore>) {
        self.maybe_schedule_drain(collection, store);
        // Incrementally repair HNSW graph references to tombstoned slots.
        // This avoids a full rebuild when tombstones accumulate; the repair
        // is local and idempotent.
        if let Err(e) = store.repair_hnsw() {
            tracing::warn!(
                collection = %collection,
                error = %e,
                "HNSW graph repair failed; will rebuild on next compaction"
            );
        }
        self.maybe_schedule_compaction(collection, store);
        self.maybe_schedule_checkpoint(collection, store);
    }

    /// Run one maintenance sweep immediately: post-compaction rebuilds,
    /// drift-triggered rebuilds and exact-scan-to-IVF promotion. Normally
    /// driven by the worker's idle tick; also useful for administrative
    /// tooling and tests.
    pub fn run_maintenance_sweep(&self) {
        let Some(jobs) = self.jobs.as_ref() else {
            return;
        };
        maintenance_sweep(&self.collections, jobs, &self.in_flight);
    }

    /// Enqueue an index build unless one is already scheduled/running for the
    /// collection.
    pub(super) fn schedule_build(
        name: &str,
        jobs: &Sender<MaintenanceJob>,
        in_flight: &Mutex<HashSet<String>>,
    ) {
        let mut guard = in_flight.lock();
        if guard.contains(name) {
            return;
        }
        guard.insert(name.to_string());
        if jobs.send(MaintenanceJob::Build(name.to_string())).is_err() {
            guard.remove(name);
        }
    }
}

pub(super) fn maintenance_loop(
    rx: Receiver<MaintenanceJob>,
    collections: Arc<RwLock<HashMap<String, Arc<CollectionStore>>>>,
    jobs: Sender<MaintenanceJob>,
    in_flight: Arc<Mutex<HashSet<String>>>,
) {
    loop {
        match rx.recv_timeout(MAINTENANCE_TICK) {
            Ok(MaintenanceJob::Build(name)) => {
                let store = collections.read().get(&name).cloned();
                if let Some(store) = store {
                    match store.build_index() {
                        Ok(published) => {
                            tracing::debug!(collection = %name, published, "maintenance build done")
                        }
                        Err(e) => tracing::warn!(
                            collection = %name,
                            error = %e,
                            "maintenance build failed"
                        ),
                    }
                }
                in_flight.lock().remove(&name);
            }
            Ok(MaintenanceJob::Compact(name)) => {
                let store = collections.read().get(&name).cloned();
                if let Some(store) = store {
                    match store.compact() {
                        Ok(live) => tracing::debug!(
                            collection = %name,
                            live,
                            "maintenance compaction done"
                        ),
                        Err(e) => tracing::warn!(
                            collection = %name,
                            error = %e,
                            "maintenance compaction failed"
                        ),
                    }
                }
                in_flight.lock().remove(&format!("compact:{name}"));
            }
            Ok(MaintenanceJob::Checkpoint(name)) => {
                let store = collections.read().get(&name).cloned();
                if let Some(store) = store {
                    match store.checkpoint() {
                        Ok(live) => tracing::debug!(
                            collection = %name,
                            live,
                            "maintenance wal checkpoint done"
                        ),
                        Err(e) => tracing::warn!(
                            collection = %name,
                            error = %e,
                            "maintenance wal checkpoint failed"
                        ),
                    }
                }
                in_flight.lock().remove(&format!("checkpoint:{name}"));
            }
            Ok(MaintenanceJob::Drain(name)) => {
                let store = collections.read().get(&name).cloned();
                if let Some(store) = store {
                    store.drain_pending_hnsw();
                }
                in_flight.lock().remove(&format!("drain:{name}"));
            }
            Err(RecvTimeoutError::Timeout) => {
                // Drain pending HNSW slots into the graph before the
                // sweep, so that freshly inserted points are incorporated
                // into the graph structure (the sweep may then decide
                // whether a full rebuild is warranted).
                {
                    let stores = collections.read().clone();
                    for store in stores.values() {
                        if matches!(store.meta().index, IndexConfig::Hnsw(_)) {
                            store.drain_pending_hnsw();
                        }
                    }
                }
                maintenance_sweep(&collections, &jobs, &in_flight);
            }
            Err(RecvTimeoutError::Disconnected) => break,
            Ok(MaintenanceJob::Shutdown) => break,
        }
    }
}

/// One periodic pass over all collections.
pub(super) fn maintenance_sweep(
    collections: &Arc<RwLock<HashMap<String, Arc<CollectionStore>>>>,
    jobs: &Sender<MaintenanceJob>,
    in_flight: &Arc<Mutex<HashSet<String>>>,
) {
    let stores = collections.read().clone();
    for (name, store) in stores {
        // Scratch, backup, and quarantine handles are never maintained:
        // only serving collections get builds and compactions.
        if name.contains(".rebuild-") || name.contains(".old-") || name.contains(".suspect-") {
            continue;
        }
        // Compaction invalidated a previously published index: restore it
        // regardless of promotion switches.
        if store.take_needs_rebuild() {
            LocalVectorEngine::schedule_build(&name, jobs, in_flight);
            continue;
        }

        let meta = store.meta();
        match meta.index {
            IndexConfig::Flat => continue,
            IndexConfig::Ivf(_) => sweep_ivf(&name, &store, jobs, in_flight),
            IndexConfig::Hnsw(_) => sweep_hnsw(&name, &store, jobs, in_flight),
        }
    }
}

/// IVF maintenance: drift-triggered rebuilds and opt-in exact-scan
/// promotion.
pub(super) fn sweep_ivf(
    name: &str,
    store: &Arc<CollectionStore>,
    jobs: &Sender<MaintenanceJob>,
    in_flight: &Arc<Mutex<HashSet<String>>>,
) {
    let Some(config) = store.ivf_config_opt() else {
        return;
    };

    match store.ivf_state() {
        Some((index, _)) => {
            // Drift maintenance keeps any published index fresh; it does
            // not depend on the auto-promotion switch.
            if !index.should_check_drift() {
                return;
            }
            let ratio = store.measure_drift(&index);
            store.record_drift(ratio);
            tracing::debug!(collection = %name, drift = ratio, "drift check");
            if ratio > config.drift_threshold {
                tracing::info!(
                    collection = %name,
                    drift = ratio,
                    threshold = config.drift_threshold,
                    "drift threshold exceeded; scheduling rebuild"
                );
                LocalVectorEngine::schedule_build(name, jobs, in_flight);
            }
        }
        None => {
            // Promotion check: build once the collection is large enough
            // and automatic promotion is enabled.
            if config.auto_promotion && store.count() >= config.min_build_points.max(1) {
                LocalVectorEngine::schedule_build(name, jobs, in_flight);
            }
        }
    }
}

/// HNSW maintenance: promote from exact scan once the collection outgrows
/// `full_scan_threshold` (Qdrant semantics: automatic, no separate switch);
/// keep a published graph fresh by scheduling rebuilds while pending slots
/// remain, or when overwrite-driven staleness crosses
/// `HnswConfig::stale_rebuild_ratio`. Overwrite upserts keep the node's old
/// graph position and only erode recall slowly, so the ratio is an optional
/// trigger rather than a correctness requirement.
pub(super) fn sweep_hnsw(
    name: &str,
    store: &Arc<CollectionStore>,
    jobs: &Sender<MaintenanceJob>,
    in_flight: &Arc<Mutex<HashSet<String>>>,
) {
    let Some(config) = store.hnsw_config_opt() else {
        return;
    };
    if store.has_index() {
        // Index is published but slots are not incorporated into the graph
        // yet (fresh writes routed to pending, or gaps found on open). The
        // maintenance tick drains pending before sweeping, so whatever is
        // still queued here exceeded one drain: rebuild only when the backlog
        // is a large share of the graph, otherwise leave it to the next drain.
        let pending = store.pending_len();
        if pending > 0 {
            let nodes = store
                .index_info()
                .map_or(0, |info| info.built_at_live_count as usize);
            if CollectionStore::pending_warrants_rebuild(pending, nodes) {
                tracing::info!(
                    collection = %name,
                    pending,
                    nodes,
                    "HNSW pending backlog warrants rebuild"
                );
                LocalVectorEngine::schedule_build(name, jobs, in_flight);
                return;
            }
        }
        // Staleness upkeep: overwrite upserts since the build, relative to
        // the live count at build time. Prefer the combined `stale_ratio`
        // (count + distance delta) when available.
        if let (Some(threshold), Some(info)) = (config.stale_rebuild_ratio, store.index_info()) {
            if info.built_at_live_count > 0 {
                let ratio = info
                    .stale_ratio
                    .unwrap_or(info.stale_overwrite_count as f64 / info.built_at_live_count as f64);
                if ratio > threshold {
                    tracing::info!(
                        collection = %name,
                        stale = info.stale_overwrite_count,
                        built_at = info.built_at_live_count,
                        ratio,
                        threshold,
                        "stale overwrite ratio exceeded; scheduling HNSW rebuild"
                    );
                    LocalVectorEngine::schedule_build(name, jobs, in_flight);
                }
            }
        }
        return;
    }
    let threshold = config
        .full_scan_threshold
        .unwrap_or(HNSW_PROMOTION_DEFAULT)
        .max(1);
    if store.count() >= threshold as u64 {
        tracing::info!(
            collection = %name,
            count = store.count(),
            threshold,
            "full scan threshold reached; scheduling HNSW build"
        );
        LocalVectorEngine::schedule_build(name, jobs, in_flight);
    }
}
