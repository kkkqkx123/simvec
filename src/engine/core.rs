use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::{Builder as ThreadBuilder, JoinHandle};
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};

use super::maintenance::maintenance_loop;
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
            let store = CollectionStore::open(&path)?;
            collections.insert(name, store);
        }

        Ok(Self::assemble(root_dir, collections))
    }

    pub(super) fn assemble(
        root_dir: PathBuf,
        collections: HashMap<String, Arc<CollectionStore>>,
    ) -> Self {
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

    /// Create a collection. Fails with
    /// [`VectorSearchError::CollectionAlreadyExists`] if it already exists.
    pub fn create_collection(&self, name: &str, config: &CollectionConfig) -> Result<()> {
        let mut effective = config.clone();
        match &mut effective.index {
            IndexConfig::Hnsw(cfg) => {
                if *cfg == HnswConfig::default() {
                    if let Some(default) = &*self.default_hnsw.read() {
                        *cfg = default.clone();
                    }
                }
            }
            IndexConfig::Ivf(opt) => {
                if opt.is_none() {
                    if let Some(default) = &*self.default_ivf.read() {
                        *opt = Some(default.clone());
                    }
                }
            }
            IndexConfig::Flat => {}
        }
        // Apply default quantization when caller did not specify one (per-collection
        // TOML defaults or global quantization settings).
        if effective.quantization_config.is_none() {
            if let Some(default) = &*self.default_quantization.read() {
                effective.quantization_config = Some(default.clone());
            }
        }
        let dir = self.root_dir.join(name);
        let store = CollectionStore::create(&dir, name, &effective)?;
        self.collections.write().insert(name.to_string(), store);
        Ok(())
    }

    /// Drop a collection and delete its directory. Destructive: all points and
    /// index files are removed. Contrast with [`Engine::drop_index`], which
    /// only discards the published graph and keeps the data for exact scan.
    ///
    /// Unloaded directories (rebuild temps, publish backups skipped at open)
    /// are removed from the filesystem directly, so recovery can delete
    /// scratch state without opening it first.
    ///
    /// The entry is unregistered before the directory is removed, then this
    /// waits (bounded) for already-queued maintenance jobs of the collection
    /// to finish: the worker holds its own handle and could otherwise still
    /// be writing derived files while the directory is torn down.
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
        self.wait_for_collection_jobs(name);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// Block until background maintenance jobs already queued for `name`
    /// finish, so directory teardown never races derived-file writes.
    /// Bounded: a stuck worker only delays the removal, and any residual
    /// failure still surfaces as an error instead of being hidden.
    pub(super) fn wait_for_collection_jobs(&self, name: &str) {
        let keys = [
            name.to_string(),
            format!("compact:{name}"),
            format!("checkpoint:{name}"),
            format!("drain:{name}"),
        ];
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            {
                let guard = self.in_flight.lock();
                if keys.iter().all(|key| !guard.contains(key)) {
                    return;
                }
            }
            if std::time::Instant::now() >= deadline {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
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
            config.index = meta.index.clone();
            config.quantization_config = meta.quantization_config.clone();
            config.max_points = meta.max_points;
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
        config.index = meta.index.clone();
        config.quantization_config = meta.quantization_config.clone();
        config.max_points = meta.max_points;
        Ok(CollectionInfo {
            name: meta.collection.clone(),
            vector_count: live,
            indexed_vector_count: live,
            points_count: live,
            segments_count: segments as u64,
            config,
            status: CollectionStatus::Green,
            index: store.index_info(),
            memory_bytes: store.memory_estimate(),
        })
    }

    pub(super) fn store(&self, collection: &str) -> Result<Arc<CollectionStore>> {
        self.collections
            .read()
            .get(collection)
            .cloned()
            .ok_or_else(|| VectorSearchError::CollectionNotFound(collection.to_string()))
    }
}
