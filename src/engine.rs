//! Local vector engine.
//!
//! Owns one collection directory per collection under a root directory and
//! keeps an in-memory registry of opened [`CollectionStore`]s. This is the
//! transport-independent engine surface; the graphdb-sync coordinator wraps it
//! in an async shell (`VectorBackend::Local`).
//!
//! Every mutation is WAL-backed (see [`CollectionStore::apply_txn`]):
//! append + fsync before applying to memory, so crash recovery replays
//! idempotently. Coordinated graph transactions land in [`LocalVectorEngine::apply_txn`].

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::{Builder as ThreadBuilder, JoinHandle};
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};

use crate::error::{Result, VectorSearchError};
use crate::metrics::MetricsSnapshot;
use crate::storage::{CollectionStore, WalPoint, WalRecord, WalTxn};
use crate::types::{
    CollectionConfig, CollectionInfo, CollectionStatus, HnswConfig, IndexType, IvfConfig, Payload,
    PayloadSchemaType, PointId, SearchQuery, SearchResult, VectorFilter, VectorPoint,
};

/// A single operation of a coordinated transaction, grouped per collection.
#[derive(Debug, Clone)]
pub enum TxnOp {
    /// Upsert a point into a collection.
    Upsert {
        collection: String,
        point: VectorPoint,
    },
    /// Delete a point by id from a collection.
    Delete {
        collection: String,
        point_id: String,
    },
}

/// Maintenance work scheduled for the background worker thread.
#[derive(Debug)]
enum MaintenanceJob {
    /// Build (or rebuild) the IVF index of a collection.
    Build(String),
    /// Physically reclaim tombstoned slots in a collection.
    Compact(String),
    /// Stop the worker (sent on engine drop).
    Shutdown,
}

/// How often the idle maintenance worker runs drift / promotion sweeps.
const MAINTENANCE_TICK: Duration = Duration::from_secs(30);

/// Default live-point threshold above which HNSW collections are promoted
/// from exact scan to the published graph (Qdrant's `full_scan_threshold`
/// default) when the config leaves it unset.
const HNSW_PROMOTION_DEFAULT: usize = 10_000;

/// Pending-slot backlog at which mutations synchronously drain the HNSW
/// graph instead of waiting for the next maintenance tick. Purely a
/// guardrail against extreme write storms leaving the graph far behind; the
/// brute-force pending path in search keeps results correct either way.
const PENDING_DRAIN_GUARDRAIL: usize = 65_536;

/// The built-in (local) vector engine.
///
/// All operations are synchronous; the graphdb-sync coordinator serializes
/// access through an async shell. Collection names must be valid path segments
/// (enforced by [`CollectionStore::create`]).
///
/// A background worker thread performs IVF scheduling: index builds,
/// post-compaction rebuilds and periodic drift checks. It holds only shared
/// handles to the collection map, so dropping the engine shuts it down.
pub struct LocalVectorEngine {
    root_dir: PathBuf,
    collections: Arc<RwLock<HashMap<String, Arc<CollectionStore>>>>,
    /// Applied to IVF collections created without an explicit IVF config.
    default_ivf: RwLock<Option<IvfConfig>>,
    /// Applied to HNSW collections created without an explicit HNSW config.
    default_hnsw: RwLock<Option<HnswConfig>>,
    /// Applied to collections created without an explicit quantization config.
    default_quantization: RwLock<Option<crate::types::QuantizationConfig>>,
    jobs: Option<Sender<MaintenanceJob>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    /// Collections with a build already queued or running.
    in_flight: Arc<Mutex<HashSet<String>>>,
}

impl std::fmt::Debug for LocalVectorEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalVectorEngine")
            .field("root_dir", &self.root_dir)
            .field("collection_count", &self.collections.read().len())
            .finish()
    }
}

impl Drop for LocalVectorEngine {
    fn drop(&mut self) {
        if let Some(jobs) = self.jobs.take() {
            let _ = jobs.send(MaintenanceJob::Shutdown);
        }
        if let Some(handle) = self.worker.lock().take() {
            let _ = handle.join();
        }
    }
}

impl LocalVectorEngine {
    /// Open (or create) the engine root directory, loading every existing
    /// collection. Returns [`VectorSearchError::InvalidConfig`] if a
    /// collection directory is corrupt.
    ///
    /// Rebuild scratch (`*.rebuild-*`), publish backups (`*.old-*`), and
    /// quarantined (`.suspect-*`) directories are never loaded: they are
    /// invisible to search and only touched by publish, reverse recovery,
    /// and startup reconciliation.
    pub fn open(root_dir: impl AsRef<Path>) -> Result<Self> {
        let root_dir = root_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&root_dir)?;

        tracing::info!(
            "vector distance kernel: {}",
            crate::distance::kernel::selected()
        );

        let mut collections = HashMap::new();
        for entry in std::fs::read_dir(&root_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if !path.join("meta.bin").exists() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.contains(".rebuild-") || name.contains(".old-") || name.contains(".suspect-") {
                continue;
            }
            let store = Arc::new(CollectionStore::open(&path)?);
            collections.insert(name, store);
        }

        Ok(Self::assemble(root_dir, collections))
    }

    fn assemble(root_dir: PathBuf, collections: HashMap<String, Arc<CollectionStore>>) -> Self {
        let (tx, rx) = mpsc::channel::<MaintenanceJob>();
        let collections = Arc::new(RwLock::new(collections));
        let in_flight = Arc::new(Mutex::new(HashSet::new()));

        let worker_collections = Arc::clone(&collections);
        let worker_in_flight = Arc::clone(&in_flight);
        let worker_jobs = tx.clone();
        let handle = ThreadBuilder::new()
            .name("vector-maintenance".to_string())
            .spawn(move || {
                maintenance_loop(rx, worker_collections, worker_jobs, worker_in_flight);
            })
            .expect("spawn vector-maintenance thread");

        Self {
            root_dir,
            collections,
            default_ivf: RwLock::new(None),
            default_hnsw: RwLock::new(None),
            default_quantization: RwLock::new(None),
            jobs: Some(tx),
            worker: Mutex::new(Some(handle)),
            in_flight,
        }
    }

    /// Root directory backing this engine.
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// Names of all loaded live collections. Rebuild scratch
    /// (`*.rebuild-*`), publish backups (`*.old-*`), and quarantined
    /// (`.suspect-*`) handles are tracked separately and never appear here,
    /// so metrics and maintenance sweeps only see serving data.
    pub fn collection_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .collections
            .read()
            .keys()
            .filter(|name| {
                !name.contains(".rebuild-")
                    && !name.contains(".old-")
                    && !name.contains(".suspect-")
            })
            .cloned()
            .collect();
        names.sort_unstable();
        names
    }

    /// Operational metrics snapshot for a collection.
    pub fn collection_metrics(&self, collection: &str) -> Result<MetricsSnapshot> {
        Ok(self.store(collection)?.metrics().snapshot())
    }

    /// Create a collection. Fails with
    /// [`VectorSearchError::CollectionAlreadyExists`] if it already exists.
    pub fn create_collection(&self, name: &str, config: &CollectionConfig) -> Result<()> {
        let mut effective = config.clone();
        match effective.index_type.unwrap_or(IndexType::HNSW) {
            IndexType::HNSW => {
                if effective.hnsw_config.is_none() {
                    if let Some(default) = &*self.default_hnsw.read() {
                        effective.hnsw_config = Some(default.clone());
                    }
                }
                effective.ivf_config = None;
            }
            IndexType::IVF => {
                if effective.ivf_config.is_none() {
                    if let Some(default) = &*self.default_ivf.read() {
                        effective.ivf_config = Some(default.clone());
                    }
                }
                effective.hnsw_config = None;
            }
            IndexType::FLAT => {}
        }
        // Apply default quantization when caller did not specify one (per-collection
        // TOML defaults or global quantization settings).
        if effective.quantization_config.is_none() {
            if let Some(default) = &*self.default_quantization.read() {
                effective.quantization_config = Some(default.clone());
            }
        }
        let dir = self.root_dir.join(name);
        let store = Arc::new(CollectionStore::create(&dir, name, &effective)?);
        self.collections.write().insert(name.to_string(), store);
        Ok(())
    }

    /// Default IVF configuration for IVF collections created without one.
    pub fn set_default_ivf_config(&self, config: IvfConfig) {
        *self.default_ivf.write() = Some(config.clone());
        for store in self.collections.read().values() {
            store.set_ivf_config(config.clone());
        }
    }

    /// Default HNSW configuration for HNSW collections created without one.
    pub fn set_default_hnsw_config(&self, config: HnswConfig) {
        *self.default_hnsw.write() = Some(config.clone());
        for store in self.collections.read().values() {
            let _ = store.set_hnsw_config(config.clone());
        }
    }

    /// Default quantization configuration for collections created without one.
    pub fn set_default_quantization_config(&self, config: crate::types::QuantizationConfig) {
        *self.default_quantization.write() = Some(config.clone());
        for store in self.collections.read().values() {
            let _ = store.set_quantization_config(config.clone());
        }
    }

    /// Current default HNSW configuration injected into HNSW collections
    /// created without one. `None` means the store-level `HnswConfig::default`
    /// applies.
    pub fn default_hnsw_config(&self) -> Option<crate::types::HnswConfig> {
        self.default_hnsw.read().clone()
    }

    /// Current default IVF configuration injected into IVF collections
    /// created without one.
    pub fn default_ivf_config(&self) -> Option<crate::types::IvfConfig> {
        self.default_ivf.read().clone()
    }

    /// Current default quantization configuration injected into collections
    /// created without one.
    pub fn default_quantization_config(&self) -> Option<crate::types::QuantizationConfig> {
        self.default_quantization.read().clone()
    }

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

    /// Drop a collection and delete its directory. Destructive: all points and
    /// index files are removed. Contrast with [`Engine::drop_index`], which
    /// only discards the published graph and keeps the data for exact scan.
    ///
    /// Unloaded directories (rebuild temps, publish backups skipped at open)
    /// are removed from the filesystem directly, so recovery can delete
    /// scratch state without opening it first.
    pub fn delete_collection(&self, name: &str) -> Result<()> {
        let dir = {
            let mut collections = self.collections.write();
            collections
                .remove(name)
                .map(|store| store.dir().to_path_buf())
                .unwrap_or_else(|| self.root_dir.join(name))
        };
        if !dir.exists() {
            return Err(VectorSearchError::CollectionNotFound(name.to_string()));
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// Names of all physical collections on disk, including loaded ones plus
    /// unloaded rebuild scratch (`*.rebuild-*`), publish backups (`*.old-*`),
    /// and quarantined (`.suspect-*`) directories. Startup recovery uses this
    /// to find orphaned temps without adopting them as logical indexes.
    pub fn list_collections(&self) -> Vec<String> {
        let mut names: std::collections::HashSet<String> =
            self.collections.read().keys().cloned().collect();
        if let Ok(entries) = std::fs::read_dir(&self.root_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() || !path.join("meta.bin").exists() {
                    continue;
                }
                names.insert(entry.file_name().to_string_lossy().into_owned());
            }
        }
        let mut names: Vec<String> = names.into_iter().collect();
        names.sort_unstable();
        names
    }

    /// Atomically promote a temp rebuild collection over its live collection.
    ///
    /// Rename dance inside the engine root directory (same filesystem):
    /// unload both handles, rename `live -> live+backup_suffix` (best
    /// effort when live is missing, e.g. interrupted publish healing),
    /// rename `temp -> live`, then reopen the live handle. The previous live
    /// directory is retained as one backup generation; older `.old-*`
    /// backups for the same live name are pruned. On any failure the live
    /// handle is restored when its directory still exists, so live stays
    /// servable and the temp residue is left for startup recovery.
    pub fn promote_temp_collection(
        &self,
        live: &str,
        temp: &str,
        backup_suffix: &str,
    ) -> Result<()> {
        let live_dir = self.root_dir.join(live);
        let temp_dir = self.root_dir.join(temp);
        if !temp_dir.is_dir() {
            return Err(VectorSearchError::CollectionNotFound(temp.to_string()));
        }
        let backup_name = format!("{live}{backup_suffix}");
        let backup_dir = self.root_dir.join(&backup_name);
        let live_store = self.collections.write().remove(live);
        let temp_store = self.collections.write().remove(temp);
        let _ = temp_store;
        let rename_result: Result<()> = (|| {
            if live_dir.exists() {
                if backup_dir.exists() {
                    std::fs::remove_dir_all(&backup_dir)?;
                }
                std::fs::rename(&live_dir, &backup_dir)?;
            }
            std::fs::rename(&temp_dir, &live_dir)?;
            Ok(())
        })();
        if let Err(error) = rename_result {
            if live_dir.exists() {
                if let Ok(store) = CollectionStore::open(&live_dir) {
                    self.collections
                        .write()
                        .insert(live.to_string(), Arc::new(store));
                } else if let Some(store) = live_store {
                    self.collections.write().insert(live.to_string(), store);
                }
            } else if backup_dir.exists() {
                let _ = std::fs::rename(&backup_dir, &live_dir);
                if let Ok(store) = CollectionStore::open(&live_dir) {
                    self.collections
                        .write()
                        .insert(live.to_string(), Arc::new(store));
                } else if let Some(store) = live_store {
                    self.collections.write().insert(live.to_string(), store);
                }
            } else if let Some(store) = live_store {
                self.collections.write().insert(live.to_string(), store);
            }
            return Err(error);
        }
        let store = CollectionStore::open(&live_dir)?;
        self.collections
            .write()
            .insert(live.to_string(), Arc::new(store));
        // Keep the backup handle loaded for inspection (`count`, reverse
        // recovery) without exposing it to search: nothing routes to
        // `.old-*` names and `collection_names` hides them.
        if backup_dir.exists() {
            if let Ok(backup_store) = CollectionStore::open(&backup_dir) {
                self.collections
                    .write()
                    .insert(backup_name.clone(), Arc::new(backup_store));
            }
        }
        self.prune_old_backups(live, Some(&backup_name));
        Ok(())
    }

    /// Reverse recovery after a bad publish: repoint `live` at its newest
    /// `.old-*` backup. The suspect live directory is renamed aside as
    /// `.suspect-<millis>` for inspection. Returns the backup name restored.
    pub fn restore_promote_backup(&self, live: &str) -> Result<String> {
        let backups = Self::list_old_backups(&self.root_dir, live);
        let Some(backup_name) = backups.into_iter().next() else {
            return Err(VectorSearchError::CollectionNotFound(format!(
                "no promote backup for collection {live}"
            )));
        };
        let live_dir = self.root_dir.join(live);
        let backup_dir = self.root_dir.join(&backup_name);
        self.collections.write().remove(live);
        self.collections.write().remove(&backup_name);
        if live_dir.exists() {
            let suspect = format!(
                "{live}.suspect-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            );
            let _ = std::fs::rename(&live_dir, self.root_dir.join(suspect));
        }
        std::fs::rename(&backup_dir, &live_dir)?;
        let store = CollectionStore::open(&live_dir)?;
        self.collections
            .write()
            .insert(live.to_string(), Arc::new(store));
        Ok(backup_name)
    }

    /// Temp rebuild collections (`*.rebuild-*`) currently present.
    pub fn list_temp_collections(&self) -> Vec<String> {
        self.list_collections()
            .into_iter()
            .filter(|name| name.contains(".rebuild-"))
            .collect()
    }

    /// Backup generations `<live>.old-*` for one live collection, newest
    /// generation first (unparseable suffixes sort last).
    pub fn list_old_backups_for(&self, live: &str) -> Vec<String> {
        Self::list_old_backups(&self.root_dir, live)
    }

    fn list_old_backups(root: &Path, live: &str) -> Vec<String> {
        let prefix = format!("{live}.old-");
        let mut backups: Vec<(u64, String)> = Vec::new();
        let Ok(entries) = std::fs::read_dir(root) else {
            return Vec::new();
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(suffix) = name.strip_prefix(&prefix) else {
                continue;
            };
            backups.push((suffix.parse::<u64>().unwrap_or(0), name));
        }
        backups.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
        backups.into_iter().map(|(_, name)| name).collect()
    }

    fn prune_old_backups(&self, live: &str, keep: Option<&str>) {
        for backup in Self::list_old_backups(&self.root_dir, live) {
            if keep.is_some_and(|keep| keep == backup) {
                continue;
            }
            self.collections.write().remove(&backup);
            if let Err(error) = std::fs::remove_dir_all(self.root_dir.join(&backup)) {
                tracing::warn!("failed to prune vector promote backup {backup}: {error}");
            }
        }
    }

    /// Whether a collection exists.
    pub fn collection_exists(&self, name: &str) -> bool {
        self.collections.read().contains_key(name)
    }

    /// Collection configuration, or `None` if the collection does not exist.
    ///
    /// Dimension, distance, effective index type, quantization and the effective
    /// ANN configurations are read back from the persisted metadata; remote-only
    /// replication fields are still excluded.
    pub fn collection_config(&self, name: &str) -> Result<Option<CollectionConfig>> {
        let collections = self.collections.read();
        Ok(collections.get(name).map(|store| {
            let meta = store.meta();
            let mut config = CollectionConfig::new(meta.vector_size, meta.distance);
            config.index_type = Some(meta.index_type);
            config.hnsw_config = meta.hnsw_config.clone();
            config.ivf_config = meta.ivf_config.clone();
            config.quantization_config = meta.quantization_config.clone();
            config
        }))
    }

    /// Detailed collection info.
    pub fn collection_info(&self, name: &str) -> Result<CollectionInfo> {
        let collections = self.collections.read();
        let store = collections
            .get(name)
            .ok_or_else(|| VectorSearchError::CollectionNotFound(name.to_string()))?;
        let meta = store.meta();
        let live = store.count();
        let segments = (meta.next_slot as usize).div_ceil(meta.segment_slots as usize);
        let mut config = CollectionConfig::new(meta.vector_size, meta.distance);
        config.index_type = Some(meta.index_type);
        config.hnsw_config = meta.hnsw_config.clone();
        config.ivf_config = meta.ivf_config.clone();
        config.quantization_config = meta.quantization_config.clone();
        Ok(CollectionInfo {
            name: meta.collection.clone(),
            vector_count: live,
            indexed_vector_count: live,
            points_count: live,
            segments_count: segments as u64,
            config,
            status: CollectionStatus::Green,
            index: store.index_info(),
        })
    }

    /// Upsert a point (WAL-backed).
    pub fn upsert(&self, collection: &str, point: VectorPoint) -> Result<()> {
        let store = self.store(collection)?;
        let result = store.apply_ops(&[WalRecord::Upsert {
            point: WalPoint::from_point(&point)?,
        }]);
        if result.is_err() {
            store.metrics().record_upsert_error();
        }
        result?;
        self.after_mutation(collection, &store);
        Ok(())
    }

    /// Upsert a batch of points (WAL-backed, single transaction).
    pub fn upsert_batch(&self, collection: &str, points: &[VectorPoint]) -> Result<()> {
        let store = self.store(collection)?;
        let ops: Result<Vec<_>> = points
            .iter()
            .map(|p| {
                Ok(WalRecord::Upsert {
                    point: WalPoint::from_point(p)?,
                })
            })
            .collect();
        let ops = match ops {
            Ok(ops) => ops,
            Err(e) => {
                store.metrics().record_upsert_error();
                return Err(e);
            }
        };
        let result = store.apply_ops(&ops);
        if result.is_err() {
            store.metrics().record_upsert_error();
        }
        result?;
        self.after_mutation(collection, &store);
        Ok(())
    }

    /// Delete a point by id (WAL-backed). Deleting a missing id is a no-op.
    pub fn delete(&self, collection: &str, point_id: &str) -> Result<()> {
        let store = self.store(collection)?;
        let result = store.apply_ops(&[WalRecord::Delete {
            point_id: point_id.to_string(),
        }]);
        if result.is_err() {
            store.metrics().record_delete_error();
        }
        result?;
        self.after_mutation(collection, &store);
        Ok(())
    }

    /// Delete a batch of points (WAL-backed, single transaction).
    pub fn delete_batch(&self, collection: &str, point_ids: &[String]) -> Result<()> {
        let store = self.store(collection)?;
        let result = store.apply_ops(&[WalRecord::DeleteBatch {
            point_ids: point_ids.to_vec(),
        }]);
        if result.is_err() {
            store.metrics().record_delete_error();
        }
        result?;
        self.after_mutation(collection, &store);
        Ok(())
    }

    /// Delete every point matching `filter`. Returns the number deleted.
    pub fn delete_by_filter(&self, collection: &str, filter: &VectorFilter) -> Result<u64> {
        let store = self.store(collection)?;
        let deleted = match store.delete_by_filter(filter) {
            Ok(deleted) => deleted,
            Err(e) => {
                store.metrics().record_delete_error();
                return Err(e);
            }
        };
        self.after_mutation(collection, &store);
        Ok(deleted)
    }

    /// Exact full-scan search.
    ///
    /// Output scores follow the crate-wide contract: higher is better
    /// (cosine similarity, inner product, and `1/(1+distance)` for euclid).
    /// Remote Qdrant engines normalize their raw Euclid distances to the
    /// same contract at the client boundary.
    pub fn search(&self, collection: &str, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let store = self.store(collection)?;
        let results = store.search(query);
        if results.is_err() {
            store.metrics().record_search_error();
        }
        results
    }

    /// Fetch a point by id.
    pub fn get(&self, collection: &str, point_id: &str) -> Result<Option<VectorPoint>> {
        let store = self.store(collection)?;
        store.get(&PointId::from(point_id.to_string()))
    }

    /// Number of live points in a collection.
    pub fn count(&self, collection: &str) -> Result<u64> {
        Ok(self.store(collection)?.count())
    }

    /// Replace the payload for a single point. The point must exist and not
    /// be tombstoned. The entire payload map is replaced atomically.
    pub fn set_payload(&self, collection: &str, point_id: &str, payload: Payload) -> Result<()> {
        let store = self.store(collection)?;
        store.set_payload(&PointId::from(point_id.to_string()), payload)
    }

    /// Remove specific keys from a point's payload. The remaining keys are
    /// preserved. If the point has no payload this is a no-op.
    pub fn delete_payload(
        &self,
        collection: &str,
        point_id: &str,
        keys: Vec<String>,
    ) -> Result<()> {
        let store = self.store(collection)?;
        store.delete_payload_keys(&PointId::from(point_id.to_string()), keys)
    }

    /// Set a single field on a point's payload (merge semantics). A missing
    /// payload is created containing just this field; all other keys are
    /// preserved. Applied atomically via a WAL-backed transaction.
    pub fn set_payload_field(
        &self,
        collection: &str,
        point_id: &str,
        key: String,
        value: serde_json::Value,
    ) -> Result<()> {
        self.set_payload_fields(collection, point_id, Payload::from([(key, value)]))
    }

    /// Merge the given fields into a point's payload within one WAL-backed
    /// transaction: keys in `fields` overwrite their previous values while
    /// all other keys are preserved. A missing payload is created.
    pub fn set_payload_fields(
        &self,
        collection: &str,
        point_id: &str,
        fields: Payload,
    ) -> Result<()> {
        let store = self.store(collection)?;
        store.set_payload_fields(&PointId::from(point_id.to_string()), fields)
    }

    /// Create a payload field index on a collection. The index is populated
    /// synchronously and its definition persisted with the collection.
    pub fn create_payload_index(
        &self,
        collection: &str,
        field: &str,
        schema: PayloadSchemaType,
    ) -> Result<()> {
        self.store(collection)?.create_payload_index(field, schema)
    }

    /// Drop the payload field index on `field`. Returns whether it existed.
    pub fn delete_payload_index(&self, collection: &str, field: &str) -> Result<bool> {
        self.store(collection)?.delete_payload_index(field)
    }

    /// All declared payload indexes as `(field, schema_type)` pairs.
    pub fn list_payload_indexes(
        &self,
        collection: &str,
    ) -> Result<Vec<(String, PayloadSchemaType)>> {
        Ok(self.store(collection)?.list_payload_indexes())
    }

    /// Paginated scan over live points in slot order.
    ///
    /// Returns up to `limit` points starting after `offset` (the last
    /// point_id from the previous page).
    pub fn scroll(
        &self,
        collection: &str,
        limit: usize,
        offset: Option<&str>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<String>)> {
        let store = self.store(collection)?;
        store.scroll(limit, offset, with_payload, with_vector)
    }

    /// Commit protocol entry for coordinated transactions.
    ///
    /// Ops are grouped by collection and each collection receives one WAL
    /// transaction carrying `txn_id`. Collections are processed in
    /// lexicographic order so a crash always advances the per-collection water
    /// marks in a deterministic order. Replay is idempotent, so graph WAL
    /// replay may re-apply the same txn id without double-applying data.
    pub fn apply_txn(&self, txn_id: u64, ops: Vec<TxnOp>) -> Result<()> {
        let mut by_collection: HashMap<String, Vec<WalRecord>> = HashMap::new();
        for op in ops {
            match op {
                TxnOp::Upsert { collection, point } => by_collection
                    .entry(collection)
                    .or_default()
                    .push(WalRecord::Upsert {
                        point: WalPoint::from_point(&point)?,
                    }),
                TxnOp::Delete {
                    collection,
                    point_id,
                } => by_collection
                    .entry(collection)
                    .or_default()
                    .push(WalRecord::Delete { point_id }),
            }
        }

        let mut names: Vec<String> = by_collection.keys().cloned().collect();
        names.sort_unstable();
        for name in names {
            let ops = by_collection.remove(&name).expect("key present");
            let store = self.store(&name)?;
            store.apply_txn(&WalTxn { txn_id, ops })?;
            self.maybe_schedule_compaction(&name, &store);
        }
        Ok(())
    }

    fn store(&self, collection: &str) -> Result<Arc<CollectionStore>> {
        self.collections
            .read()
            .get(collection)
            .cloned()
            .ok_or_else(|| VectorSearchError::CollectionNotFound(collection.to_string()))
    }

    /// Schedule a background compaction for `collection` once its tombstone
    /// ratio crosses the threshold. Deduplicated through the in-flight set so
    /// bursts of mutations do not flood the maintenance queue.
    fn maybe_schedule_compaction(&self, collection: &str, store: &CollectionStore) {
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

    /// Post-mutation hook, run after the store locks are released: drain an
    /// extreme pending backlog synchronously (guardrail), schedule a
    /// background compaction when warranted, and incrementally repair the
    /// HNSW graph if it has stale references to tombstoned slots.
    fn after_mutation(&self, collection: &str, store: &Arc<CollectionStore>) {
        if store.pending_len() >= PENDING_DRAIN_GUARDRAIL {
            store.drain_pending_hnsw();
        }
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
    fn schedule_build(
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

/// Background maintenance loop: executes scheduled builds and, on each idle
/// tick, sweeps collections for post-compaction rebuilds, drift-triggered
/// rebuilds and exact-scan-to-IVF promotion.
fn maintenance_loop(
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
            Err(RecvTimeoutError::Timeout) => {
                // Drain pending HNSW slots into the graph before the
                // sweep, so that freshly inserted points are incorporated
                // into the graph structure (the sweep may then decide
                // whether a full rebuild is warranted).
                {
                    let stores = collections.read().clone();
                    for store in stores.values() {
                        if let IndexType::HNSW = store.meta().index_type {
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
fn maintenance_sweep(
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
        match meta.index_type {
            IndexType::FLAT => continue,
            IndexType::IVF => sweep_ivf(&name, &store, jobs, in_flight),
            IndexType::HNSW => sweep_hnsw(&name, &store, jobs, in_flight),
        }
    }
}

/// IVF maintenance: drift-triggered rebuilds and opt-in exact-scan
/// promotion.
fn sweep_ivf(
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
fn sweep_hnsw(
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
        // yet (fresh writes routed to pending, or gaps found on open).
        if store.pending_len() > 0 {
            tracing::debug!(
                collection = %name,
                pending = store.pending_len(),
                "HNSW index published with pending slots; scheduling rebuild"
            );
            LocalVectorEngine::schedule_build(name, jobs, in_flight);
            return;
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

// ---------------------------------------------------------------------------
// Unified async VectorEngine trait
// ---------------------------------------------------------------------------
//
// Both the local (in-process) engine and the remote Qdrant client implement
// this trait. Callers in `graphdb-sync` talk exclusively through this
// interface via a trait object (`Arc<dyn VectorEngine>`), eliminating the
// previous enum-dispatch boilerplate.
//
// The local engine is synchronous; its implementation wraps each call in
// `tokio::task::spawn_blocking` so that blocking I/O / CPU work never
// starves the Tokio worker pool.

use std::pin::Pin;

use futures::Stream;

use crate::error::{EngineResult, VectorEngineError};
use crate::types::{HealthStatus, IndexMetadata};

/// Asynchronous vector engine abstraction.
///
/// Implementors must be `Send + Sync + Debug`.  The local engine runs
/// synchronously through `block_in_place`; remote engines execute
/// directly against the network.
#[async_trait]
pub trait VectorEngine: Send + Sync + std::fmt::Debug {
    /// Human-readable engine name (used for logging and health checks).
    fn name(&self) -> &str;

    /// Engine version string.
    fn version(&self) -> &str;

    /// Whether this is the built-in local engine.
    fn is_local(&self) -> bool {
        false
    }

    /// Whether the engine is currently unavailable (e.g. disabled Qdrant).
    fn is_disabled(&self) -> bool {
        false
    }

    /// Downcast support: returns `self` as `&dyn Any`.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Liveness / readiness probe.
    async fn health_check(&self) -> EngineResult<HealthStatus>;

    // ---- collection management ----

    /// Create a collection.  Fails if it already exists.
    async fn create_collection(&self, name: &str, config: &CollectionConfig) -> EngineResult<()>;

    /// Drop a collection and its persisted data.
    async fn delete_collection(&self, name: &str) -> EngineResult<()>;

    /// Whether a collection with this name exists.
    fn collection_exists(&self, name: &str) -> bool;

    /// Collection metadata, or `None` when not found.
    fn get_index_metadata(&self, name: &str) -> Option<IndexMetadata>;

    /// Create a payload field index for filter acceleration.
    async fn create_payload_index(
        &self,
        collection: &str,
        field: &str,
        schema: PayloadSchemaType,
    ) -> EngineResult<()>;

    /// Remove a payload field index.
    async fn delete_payload_index(&self, collection: &str, field: &str) -> EngineResult<()>;

    /// All declared payload indexes as `(field, schema_type)` pairs.
    async fn list_payload_indexes(
        &self,
        collection: &str,
    ) -> EngineResult<Vec<(String, PayloadSchemaType)>>;

    // ---- mutations ----

    /// Upsert a single point.
    async fn upsert(&self, collection: &str, point: VectorPoint) -> EngineResult<()>;

    /// Upsert a batch of points in a single transaction.
    async fn upsert_batch(&self, collection: &str, points: Vec<VectorPoint>) -> EngineResult<()>;

    /// Delete a point by id.
    async fn delete(&self, collection: &str, point_id: &str) -> EngineResult<()>;

    /// Delete a batch of points by id.
    async fn delete_batch(&self, collection: &str, point_ids: &[&str]) -> EngineResult<()>;

    /// Delete every point matching the given filter.
    async fn delete_by_filter(&self, collection: &str, filter: VectorFilter) -> EngineResult<()>;

    /// Atomically replace the full payload of each listed point.
    async fn set_payload(
        &self,
        collection: &str,
        point_ids: Vec<String>,
        payload: Payload,
    ) -> EngineResult<()>;

    /// Merge the given fields into each point's payload (other keys preserved).
    async fn set_payload_fields(
        &self,
        collection: &str,
        point_ids: Vec<String>,
        fields: Payload,
    ) -> EngineResult<()>;

    /// Remove the given keys from each point's payload.
    async fn delete_payload(
        &self,
        collection: &str,
        point_ids: Vec<String>,
        keys: Vec<String>,
    ) -> EngineResult<()>;

    /// Paginated scan over live points in slot order.
    async fn scroll(
        &self,
        collection: &str,
        limit: usize,
        offset: Option<&str>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> EngineResult<(Vec<VectorPoint>, Option<String>)>;

    // ---- reads ----

    /// Full ANN / exact-scan search.
    async fn search(&self, collection: &str, query: SearchQuery)
        -> EngineResult<Vec<SearchResult>>;

    /// Fetch a single point by id.
    async fn get(&self, collection: &str, point_id: &str) -> EngineResult<Option<VectorPoint>>;

    /// Number of live points in a collection.
    async fn count(&self, collection: &str) -> EngineResult<u64>;

    /// Streaming search: yields results one by one as a stream.
    ///
    /// Default implementation executes a full search and streams the
    /// buffered results. Remote engines may override with a true gRPC
    /// streaming implementation. The stream is boxed to keep the trait
    /// object-safe.
    async fn search_stream(
        &self,
        collection: &str,
        query: SearchQuery,
    ) -> EngineResult<Pin<Box<dyn Stream<Item = EngineResult<SearchResult>> + Send>>> {
        let results = self.search(collection, query).await?;
        let stream = futures::stream::iter(results.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }

    /// Streaming scroll over a collection.
    ///
    /// Default implementation pages through `scroll` synchronously and
    /// yields points one by one. Engines with native streaming may
    /// override for efficiency.
    async fn scroll_stream(
        &self,
        collection: &str,
        batch_size: usize,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> EngineResult<Pin<Box<dyn Stream<Item = EngineResult<VectorPoint>> + Send>>> {
        let mut offset: Option<String> = None;
        let collection = collection.to_string();
        let mut all_points: Vec<VectorPoint> = Vec::new();
        loop {
            let (points, next) = self
                .scroll(
                    &collection,
                    batch_size,
                    offset.as_deref(),
                    with_payload,
                    with_vector,
                )
                .await?;
            let is_last = next.is_none();
            all_points.extend(points);
            offset = next;
            if is_last || offset.is_none() {
                break;
            }
        }
        let stream = futures::stream::iter(all_points.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }
}

// ---- LocalVectorEngine implementation ---------------------------------------

#[async_trait]
impl VectorEngine for LocalVectorEngine {
    fn name(&self) -> &str {
        "simvec"
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn is_local(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_check(&self) -> EngineResult<HealthStatus> {
        Ok(HealthStatus::healthy(self.name(), self.version()))
    }

    async fn create_collection(&self, name: &str, config: &CollectionConfig) -> EngineResult<()> {
        let name = name.to_string();
        let config = config.clone();
        tokio::task::block_in_place(|| self.create_collection(&name, &config))
            .map_err(VectorEngineError::from)
    }

    async fn delete_collection(&self, name: &str) -> EngineResult<()> {
        let name = name.to_string();
        tokio::task::block_in_place(|| self.delete_collection(&name))
            .map_err(VectorEngineError::from)
    }

    fn collection_exists(&self, name: &str) -> bool {
        self.collections.read().contains_key(name)
    }

    fn get_index_metadata(&self, name: &str) -> Option<IndexMetadata> {
        let collections = self.collections.read();
        let store = collections.get(name)?;
        let meta = store.meta();
        let mut config = CollectionConfig::new(meta.vector_size, meta.distance);
        if let Some(hnsw) = &meta.hnsw_config {
            config = config
                .with_index_type(IndexType::HNSW)
                .with_hnsw(hnsw.clone());
        } else if let Some(ivf) = &meta.ivf_config {
            config = config.with_index_type(IndexType::IVF).with_ivf(ivf.clone());
        }
        if meta.quantization_config.is_some() {
            config.quantization_config = meta.quantization_config.clone();
        }
        Some(IndexMetadata {
            name: name.to_string(),
            config,
            created_at: chrono::Utc::now(),
            vector_count: store.count(),
            index_name: None,
        })
    }

    async fn create_payload_index(
        &self,
        collection: &str,
        field: &str,
        schema: PayloadSchemaType,
    ) -> EngineResult<()> {
        let collection = collection.to_string();
        let field = field.to_string();
        tokio::task::block_in_place(|| {
            self.store(&collection)
                .and_then(|s| s.create_payload_index(&field, schema))
        })
        .map_err(VectorEngineError::from)
    }

    async fn delete_payload_index(&self, collection: &str, field: &str) -> EngineResult<()> {
        let collection = collection.to_string();
        let field = field.to_string();
        tokio::task::block_in_place(|| {
            self.store(&collection)
                .and_then(|s| s.delete_payload_index(&field))
                .map(|_| ())
        })
        .map_err(VectorEngineError::from)
    }

    async fn list_payload_indexes(
        &self,
        collection: &str,
    ) -> EngineResult<Vec<(String, PayloadSchemaType)>> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.store(&collection).map(|s| s.list_payload_indexes()))
            .map_err(VectorEngineError::from)
    }

    async fn upsert(&self, collection: &str, point: VectorPoint) -> EngineResult<()> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.upsert(&collection, point))
            .map_err(VectorEngineError::from)
    }

    async fn upsert_batch(&self, collection: &str, points: Vec<VectorPoint>) -> EngineResult<()> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.upsert_batch(&collection, &points))
            .map_err(VectorEngineError::from)
    }

    async fn delete(&self, collection: &str, point_id: &str) -> EngineResult<()> {
        let collection = collection.to_string();
        let point_id = point_id.to_string();
        tokio::task::block_in_place(|| self.delete(&collection, &point_id))
            .map_err(VectorEngineError::from)
    }

    async fn delete_batch(&self, collection: &str, point_ids: &[&str]) -> EngineResult<()> {
        let collection = collection.to_string();
        let ids: Vec<String> = point_ids.iter().map(|s| s.to_string()).collect();
        tokio::task::block_in_place(|| self.delete_batch(&collection, &ids))
            .map_err(VectorEngineError::from)
    }

    async fn delete_by_filter(&self, collection: &str, filter: VectorFilter) -> EngineResult<()> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.delete_by_filter(&collection, &filter).map(|_| ()))
            .map_err(VectorEngineError::from)
    }

    async fn set_payload(
        &self,
        collection: &str,
        point_ids: Vec<String>,
        payload: Payload,
    ) -> EngineResult<()> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| {
            for id in &point_ids {
                self.set_payload(&collection, id, payload.clone())?;
            }
            Ok::<(), VectorSearchError>(())
        })
        .map_err(VectorEngineError::from)
    }

    async fn set_payload_fields(
        &self,
        collection: &str,
        point_ids: Vec<String>,
        fields: Payload,
    ) -> EngineResult<()> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| {
            for id in &point_ids {
                self.set_payload_fields(&collection, id, fields.clone())?;
            }
            Ok::<(), VectorSearchError>(())
        })
        .map_err(VectorEngineError::from)
    }

    async fn delete_payload(
        &self,
        collection: &str,
        point_ids: Vec<String>,
        keys: Vec<String>,
    ) -> EngineResult<()> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| {
            for id in &point_ids {
                self.delete_payload(&collection, id, keys.clone())?;
            }
            Ok::<(), VectorSearchError>(())
        })
        .map_err(VectorEngineError::from)
    }

    async fn scroll(
        &self,
        collection: &str,
        limit: usize,
        offset: Option<&str>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> EngineResult<(Vec<VectorPoint>, Option<String>)> {
        let collection = collection.to_string();
        let offset = offset.map(String::from);
        tokio::task::block_in_place(|| {
            self.scroll(
                &collection,
                limit,
                offset.as_deref(),
                with_payload,
                with_vector,
            )
        })
        .map_err(VectorEngineError::from)
    }

    async fn search(
        &self,
        collection: &str,
        query: SearchQuery,
    ) -> EngineResult<Vec<SearchResult>> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.search(&collection, &query))
            .map_err(VectorEngineError::from)
    }

    async fn get(&self, collection: &str, point_id: &str) -> EngineResult<Option<VectorPoint>> {
        let collection = collection.to_string();
        let point_id = point_id.to_string();
        tokio::task::block_in_place(|| self.get(&collection, &point_id))
            .map_err(VectorEngineError::from)
    }

    async fn count(&self, collection: &str) -> EngineResult<u64> {
        let collection = collection.to_string();
        tokio::task::block_in_place(|| self.count(&collection)).map_err(VectorEngineError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DistanceMetric, FilterCondition, Payload, VectorFilter};

    fn config(dim: usize) -> CollectionConfig {
        CollectionConfig::new(dim, DistanceMetric::Cosine)
    }

    fn point(id: u64, dim: usize) -> VectorPoint {
        VectorPoint::new(
            id,
            (0..dim)
                .map(|i| ((id as usize * 31 + i) % 100) as f32 / 100.0)
                .collect(),
        )
    }

    fn point_with_color(id: u64, dim: usize, color: &str) -> VectorPoint {
        let mut payload: Payload = HashMap::new();
        payload.insert("color".to_string(), serde_json::json!(color));
        VectorPoint::new(id, (0..dim).map(|_| 0.5).collect()).with_payload(payload)
    }

    fn engine() -> LocalVectorEngine {
        let dir = tempfile::tempdir().unwrap();
        LocalVectorEngine::open(dir.path().join("vec")).unwrap()
    }

    #[test]
    fn test_deletes_schedule_background_compaction() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        let engine = LocalVectorEngine::open(&root).unwrap();
        engine.create_collection("col", &config(4)).unwrap();
        for i in 0..10u64 {
            engine.upsert("col", point(i, 4)).unwrap();
        }
        // 3/10 live-slot ratio crosses the 20% threshold: the mutation path
        // must enqueue a background compaction without blocking.
        engine.delete("col", "0").unwrap();
        engine.delete("col", "1").unwrap();
        engine.delete("col", "2").unwrap();

        let store = engine.store("col").unwrap();
        assert!(store.needs_compaction());

        // The maintenance worker drains the queue promptly; poll briefly so
        // the test does not depend on exact thread scheduling.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while store.meta().tombstone_count > 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(store.meta().tombstone_count, 0, "background compaction ran");
        assert_eq!(store.count(), 7);
    }

    #[test]
    fn test_create_open_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        {
            let engine = LocalVectorEngine::open(&root).unwrap();
            engine.create_collection("col_a", &config(4)).unwrap();
            engine.upsert("col_a", point(1, 4)).unwrap();
            engine.upsert("col_a", point(2, 4)).unwrap();
            assert_eq!(engine.count("col_a").unwrap(), 2);
        }
        let reopened = LocalVectorEngine::open(&root).unwrap();
        assert!(reopened.collection_exists("col_a"));
        assert_eq!(reopened.count("col_a").unwrap(), 2);
        let got = reopened.get("col_a", "1").unwrap().unwrap();
        assert_eq!(got.vector, point(1, 4).vector);
        assert!(reopened.get("col_a", "99").unwrap().is_none());
    }

    #[test]
    fn test_create_duplicate_fails() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        let err = engine.create_collection("col", &config(4)).unwrap_err();
        assert!(matches!(err, VectorSearchError::CollectionAlreadyExists(_)));
    }

    #[test]
    fn test_invalid_collection_name() {
        let engine = engine();
        let err = engine.create_collection("a/b", &config(4)).unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidCollectionName(_)));
        let err = engine.create_collection("", &config(4)).unwrap_err();
        assert!(matches!(err, VectorSearchError::InvalidCollectionName(_)));
    }

    #[test]
    fn test_delete_collection() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        {
            let engine = LocalVectorEngine::open(&root).unwrap();
            engine.create_collection("col", &config(4)).unwrap();
            engine.upsert("col", point(1, 4)).unwrap();
            engine.delete_collection("col").unwrap();
            assert!(!engine.collection_exists("col"));
        }
        let reopened = LocalVectorEngine::open(&root).unwrap();
        assert!(!reopened.collection_exists("col"));
    }

    #[test]
    fn test_delete_collection_missing() {
        let engine = engine();
        let err = engine.delete_collection("nope").unwrap_err();
        assert!(matches!(err, VectorSearchError::CollectionNotFound(_)));
    }

    #[test]
    fn test_promote_temp_collection_swaps_live_and_keeps_backup() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        let engine = LocalVectorEngine::open(&root).unwrap();
        engine.create_collection("live", &config(4)).unwrap();
        engine.upsert("live", point(1, 4)).unwrap();
        engine
            .create_collection("live.rebuild-2", &config(4))
            .unwrap();
        engine.upsert("live.rebuild-2", point(2, 4)).unwrap();

        engine
            .promote_temp_collection("live", "live.rebuild-2", ".old-2")
            .unwrap();

        assert_eq!(engine.count("live").unwrap(), 1);
        let got = engine.get("live", "2").unwrap().unwrap();
        assert_eq!(got.vector, point(2, 4).vector);
        assert!(engine.get("live", "1").unwrap().is_none());
        assert_eq!(engine.count("live.old-2").unwrap(), 1);
        // Scratch and backups serve no reads through discovery names.
        assert!(!engine
            .collection_names()
            .contains(&"live.old-2".to_string()));
        assert!(engine
            .list_collections()
            .contains(&"live.old-2".to_string()));
        assert!(!engine
            .list_temp_collections()
            .contains(&"live.rebuild-2".to_string()));

        // Second publish prunes the older backup, keeping exactly one.
        engine
            .create_collection("live.rebuild-3", &config(4))
            .unwrap();
        engine.upsert("live.rebuild-3", point(3, 4)).unwrap();
        engine
            .promote_temp_collection("live", "live.rebuild-3", ".old-3")
            .unwrap();
        assert_eq!(engine.count("live").unwrap(), 1);
        assert!(engine
            .list_collections()
            .contains(&"live.old-3".to_string()));
        assert!(!engine
            .list_collections()
            .contains(&"live.old-2".to_string()));

        // Restarted engines never adopt scratch or backups as serving data.
        drop(engine);
        let reopened = LocalVectorEngine::open(&root).unwrap();
        assert_eq!(reopened.count("live").unwrap(), 1);
        assert!(!reopened
            .collection_names()
            .contains(&"live.old-3".to_string()));
        assert!(reopened
            .list_collections()
            .contains(&"live.old-3".to_string()));
    }

    #[test]
    fn test_promote_temp_collection_missing_temp_fails_without_touching_live() {
        let engine = engine();
        engine.create_collection("live", &config(4)).unwrap();
        engine.upsert("live", point(1, 4)).unwrap();
        let err = engine
            .promote_temp_collection("live", "live.rebuild-9", ".old-9")
            .unwrap_err();
        assert!(matches!(err, VectorSearchError::CollectionNotFound(_)));
        assert_eq!(engine.count("live").unwrap(), 1);
    }

    #[test]
    fn test_restore_promote_backup_reverses_bad_publish() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        let engine = LocalVectorEngine::open(&root).unwrap();
        engine.create_collection("live", &config(4)).unwrap();
        engine.upsert("live", point(1, 4)).unwrap();
        engine
            .create_collection("live.rebuild-2", &config(4))
            .unwrap();
        engine.upsert("live.rebuild-2", point(2, 4)).unwrap();
        engine
            .promote_temp_collection("live", "live.rebuild-2", ".old-2")
            .unwrap();

        let restored = engine.restore_promote_backup("live").unwrap();
        assert_eq!(restored, "live.old-2");
        assert_eq!(engine.count("live").unwrap(), 1);
        let got = engine.get("live", "1").unwrap().unwrap();
        assert_eq!(got.vector, point(1, 4).vector);

        let err = engine.restore_promote_backup("live").unwrap_err();
        assert!(matches!(err, VectorSearchError::CollectionNotFound(_)));
    }

    #[test]
    fn test_open_skips_scratch_and_backup_directories() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        {
            let engine = LocalVectorEngine::open(&root).unwrap();
            engine.create_collection("live", &config(4)).unwrap();
            engine.upsert("live", point(1, 4)).unwrap();
            engine
                .create_collection("live.rebuild-2", &config(4))
                .unwrap();
            engine
                .promote_temp_collection("live", "live.rebuild-2", ".old-2")
                .unwrap();
        }
        let reopened = LocalVectorEngine::open(&root).unwrap();
        assert!(reopened.collection_exists("live"));
        assert!(!reopened
            .collection_names()
            .contains(&"live.old-2".to_string()));
        // Scratch residue is deletable without being opened first.
        reopened.delete_collection("live.old-2").unwrap();
        assert!(!reopened
            .list_collections()
            .contains(&"live.old-2".to_string()));
    }

    #[test]
    fn test_collection_config_and_info() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        let cfg = engine.collection_config("col").unwrap().unwrap();
        assert_eq!(cfg.vector_size, 4);
        assert_eq!(cfg.distance, DistanceMetric::Cosine);
        assert!(engine.collection_config("nope").unwrap().is_none());

        engine.upsert("col", point(1, 4)).unwrap();
        let info = engine.collection_info("col").unwrap();
        assert_eq!(info.name, "col");
        assert_eq!(info.points_count, 1);
        assert_eq!(info.vector_count, 1);
        assert_eq!(info.segments_count, 1);
        assert_eq!(info.status, CollectionStatus::Green);
    }

    #[test]
    fn test_search_returns_topk() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        engine.upsert("col", point(1, 4)).unwrap();
        engine.upsert("col", point(2, 4)).unwrap();
        engine.upsert("col", point(3, 4)).unwrap();

        let results = engine
            .search("col", &SearchQuery::new(vec![0.0; 4], 2).with_payload(true))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            results[0].score >= results[1].score,
            "results sorted descending: {:?}",
            results
        );
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        engine.upsert("col", point(1, 4)).unwrap();
        let err = engine
            .search("col", &SearchQuery::new(vec![1.0, 2.0], 1))
            .unwrap_err();
        assert!(matches!(
            err,
            VectorSearchError::InvalidVectorDimension {
                expected: 4,
                actual: 2
            }
        ));
    }

    #[test]
    fn test_delete_by_id() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        engine.upsert("col", point(1, 4)).unwrap();
        engine.upsert("col", point(2, 4)).unwrap();
        engine.delete("col", "1").unwrap();
        assert_eq!(engine.count("col").unwrap(), 1);
        assert!(engine.get("col", "1").unwrap().is_none());
        engine.delete("col", "99").unwrap();
        assert_eq!(engine.count("col").unwrap(), 1);
    }

    #[test]
    fn test_delete_by_filter() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        engine.upsert("col", point_with_color(1, 4, "red")).unwrap();
        engine
            .upsert("col", point_with_color(2, 4, "blue"))
            .unwrap();
        engine.upsert("col", point_with_color(3, 4, "red")).unwrap();
        assert_eq!(engine.count("col").unwrap(), 3);

        let filter = VectorFilter::new().must(FilterCondition::match_value("color", "red"));
        let deleted = engine.delete_by_filter("col", &filter).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(engine.count("col").unwrap(), 1);
        assert!(engine.get("col", "2").unwrap().is_some());
        assert!(engine.get("col", "1").unwrap().is_none());
        assert!(engine.get("col", "3").unwrap().is_none());
    }

    #[test]
    fn test_delete_by_filter_no_match() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        engine.upsert("col", point(1, 4)).unwrap();
        let filter = VectorFilter::new().must(FilterCondition::match_value("color", "red"));
        assert_eq!(engine.delete_by_filter("col", &filter).unwrap(), 0);
        assert_eq!(engine.count("col").unwrap(), 1);
    }

    #[test]
    fn test_apply_txn_multi_collection() {
        let engine = engine();
        engine.create_collection("col_b", &config(4)).unwrap();
        engine.create_collection("col_a", &config(4)).unwrap();

        let ops = vec![
            TxnOp::Upsert {
                collection: "col_b".to_string(),
                point: point(1, 4),
            },
            TxnOp::Upsert {
                collection: "col_a".to_string(),
                point: point(2, 4),
            },
            TxnOp::Delete {
                collection: "col_a".to_string(),
                point_id: "42".to_string(),
            },
        ];
        engine.apply_txn(7, ops).unwrap();

        assert_eq!(engine.count("col_a").unwrap(), 1);
        assert_eq!(engine.count("col_b").unwrap(), 1);
        assert!(engine.get("col_b", "1").unwrap().is_some());

        // Replaying the same txn id is idempotent.
        engine
            .apply_txn(
                7,
                vec![TxnOp::Upsert {
                    collection: "col_a".to_string(),
                    point: point(2, 4),
                }],
            )
            .unwrap();
        assert_eq!(engine.count("col_a").unwrap(), 1);
    }

    #[test]
    fn test_apply_txn_unknown_collection() {
        let engine = engine();
        let err = engine
            .apply_txn(
                1,
                vec![TxnOp::Upsert {
                    collection: "nope".to_string(),
                    point: point(1, 4),
                }],
            )
            .unwrap_err();
        assert!(matches!(err, VectorSearchError::CollectionNotFound(_)));
    }

    #[test]
    fn test_apply_txn_validates_dimension() {
        let engine = engine();
        engine.create_collection("col", &config(4)).unwrap();
        let err = engine
            .apply_txn(
                1,
                vec![TxnOp::Upsert {
                    collection: "col".to_string(),
                    point: VectorPoint::new(1u64, vec![1.0, 2.0]),
                }],
            )
            .unwrap_err();
        assert!(matches!(
            err,
            VectorSearchError::InvalidVectorDimension {
                expected: 4,
                actual: 2
            }
        ));
        assert_eq!(engine.count("col").unwrap(), 0);
    }

    #[test]
    fn test_wal_recovery_after_mutation() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vec");
        {
            let engine = LocalVectorEngine::open(&root).unwrap();
            engine.create_collection("col", &config(4)).unwrap();
            engine
                .apply_txn(
                    3,
                    vec![TxnOp::Upsert {
                        collection: "col".to_string(),
                        point: point(1, 4),
                    }],
                )
                .unwrap();
            engine.upsert("col", point(2, 4)).unwrap();
            engine.delete("col", "2").unwrap();
        }
        let reopened = LocalVectorEngine::open(&root).unwrap();
        assert_eq!(reopened.count("col").unwrap(), 1);
        assert!(reopened.get("col", "1").unwrap().is_some());
        assert!(reopened.get("col", "2").unwrap().is_none());
    }
}
