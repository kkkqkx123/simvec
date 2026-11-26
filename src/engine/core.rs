use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self};
use std::sync::Arc;
use std::thread::Builder as ThreadBuilder;

use parking_lot::{Mutex, RwLock};

use super::maintenance::maintenance_loop;
use super::{LocalVectorEngine, MaintenanceJob, SpaceHandle};
use crate::error::{Result, VectorSearchError};
use crate::storage::CollectionStore;
use crate::types::{
    CollectionConfig, CollectionInfo, CollectionStatus, HnswConfig, IndexConfig, StorageConfig,
};

impl LocalVectorEngine {
    /// Open (or create) the engine root directory, loading every existing
    /// collection.
    ///
    /// A collection that fails to open (corrupt `meta.bin`, checksum failure,
    /// incomplete directory) is renamed to `<name>.suspect-<unix_ts>` and
    /// skipped instead of failing the whole engine: one damaged collection must
    /// not take every other collection offline. Only a failure to enumerate
    /// the root directory itself is fatal.
    ///
    /// Rebuild scratch (`*.rebuild-*`), publish backups (`*.old-*`), and
    /// quarantined (`.suspect-*`) directories are never loaded: they are
    /// invisible to search and only touched by publish, reverse recovery,
    /// and startup reconciliation.
    pub fn open(root_dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_storage(root_dir, StorageConfig::default())
    }

    /// Open (or create) the engine root directory with explicit engine-level
    /// storage tuning. Every collection loaded from disk adopts `storage`.
    pub fn open_with_storage(root_dir: impl AsRef<Path>, storage: StorageConfig) -> Result<Self> {
        storage.validate()?;
        let root_dir = root_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&root_dir)?;

        tracing::info!(
            "vector distance kernel: {}",
            crate::distance::kernel::selected()
        );

        let mut stores: HashMap<String, SpaceHandle> = HashMap::new();
        for entry in std::fs::read_dir(&root_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if !path.join("meta.bin").exists()
                && !path.join("kind.bin").exists()
                && !path.join("named.bin").exists()
            {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.contains(".rebuild-") || name.contains(".old-") || name.contains(".suspect-") {
                continue;
            }
            let opened = if path.join("kind.bin").exists() {
                crate::kind_store::KindStore::open(&path, &storage).map(SpaceHandle::Kind)
            } else if path.join("named.bin").exists() {
                crate::named_store::NamedStore::open(&path, &storage).map(SpaceHandle::Named)
            } else {
                CollectionStore::open_with_storage(&path, &storage).map(SpaceHandle::Dense)
            };
            match opened {
                Ok(store) => {
                    stores.insert(name, store);
                }
                Err(err) => {
                    let quarantined = quarantine(&path, &name);
                    tracing::error!(
                        collection = %name,
                        error = %err,
                        quarantined = ?quarantined,
                        "collection failed to open; quarantined and skipped"
                    );
                }
            }
        }

        Ok(Self::assemble(root_dir, stores, storage))
    }

    pub(super) fn assemble(
        root_dir: PathBuf,
        stores: HashMap<String, SpaceHandle>,
        storage: StorageConfig,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<MaintenanceJob>();
        let stores = Arc::new(RwLock::new(stores));
        let in_flight = Arc::new(Mutex::new(HashSet::new()));

        let worker_stores = Arc::clone(&stores);
        let worker_in_flight = Arc::clone(&in_flight);
        let worker_jobs = tx.clone();
        let handle = ThreadBuilder::new()
            .name("vector-maintenance".to_string())
            .spawn(move || {
                maintenance_loop(rx, worker_stores, worker_jobs, worker_in_flight);
            })
            .expect("spawn vector-maintenance thread");

        Self {
            root_dir,
            stores,
            storage,
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
        let mut names: Vec<String> = self.stores.read().keys().cloned().collect();
        names.retain(|name| {
            !name.contains(".rebuild-")
                && !name.contains(".old-")
                && !name.contains(".suspect-")
        });
        names.sort_unstable();
        names
    }

    /// Create a collection. Fails with
    /// [`VectorSearchError::CollectionAlreadyExists`] if it already exists.
    pub fn create_collection(&self, name: &str, config: &CollectionConfig) -> Result<()> {
        if self.collection_exists(name) {
            return Err(VectorSearchError::CollectionAlreadyExists(name.to_string()));
        }
        let mut config = config.clone();
        config.normalize();
        // Named layouts are exact-scan only. Validate before applying engine
        // defaults so a requested ANN tier or quantization is never silently
        // replaced by a default.
        if config.is_named_layout() {
            config.validate_kind()?;
            let spaces = config.vectors.clone();
            let dir = self.root_dir.join(name);
            let store = crate::named_store::NamedStore::create(
                &dir,
                name,
                &spaces,
                config.max_points,
                &self.storage,
            )?;
            self.stores
                .write()
                .insert(name.to_string(), SpaceHandle::Named(store));
            return Ok(());
        }
        let mut effective = config;
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
        if effective.kind != crate::types::VectorKind::Dense {
            // Kind collections never use ANN tiers or quantization; coerce
            // the default dense knobs instead of rejecting configs built
            // with the default `CollectionConfig::new`.
            effective.index = IndexConfig::Flat;
            effective.quantization_config = None;
            let store =
                crate::kind_store::KindStore::create(&dir, name, &effective, &self.storage)?;
            self.stores
                .write()
                .insert(name.to_string(), SpaceHandle::Kind(store));
            return Ok(());
        }
        let store = CollectionStore::create_with_storage(&dir, name, &effective, &self.storage)?;
        self.stores
            .write()
            .insert(name.to_string(), SpaceHandle::Dense(store));
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
            let mut stores = self.stores.write();
            match stores.remove(name) {
                Some(handle) => handle.dir().to_path_buf(),
                None => self.root_dir.join(name),
            }
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
            self.stores.read().keys().cloned().collect();
        if let Ok(entries) = std::fs::read_dir(&self.root_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir()
                    || !(path.join("meta.bin").exists()
                        || path.join("kind.bin").exists()
                        || path.join("named.bin").exists())
                {
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
        self.stores.read().contains_key(name)
    }

    /// The named store behind `collection`, when it is a named-vectors
    /// collection.
    /// Resolve the single handle for a collection. This is the one registry
    /// lookup every read/write entry point starts from.
    pub(super) fn handle(&self, collection: &str) -> Option<SpaceHandle> {
        self.stores.read().get(collection).cloned()
    }

    /// Require a collection handle, returning the standard not-found error.
    /// This is the single lookup every read/write entry point starts from.
    pub(super) fn require_handle(&self, collection: &str) -> Result<SpaceHandle> {
        self.handle(collection)
            .ok_or_else(|| VectorSearchError::CollectionNotFound(collection.to_string()))
    }

    pub fn named_store(
        &self,
        collection: &str,
    ) -> Result<Option<Arc<crate::named_store::NamedStore>>> {
        Ok(self.handle(collection).and_then(|h| h.as_named().cloned()))
    }

    /// The kind store behind `collection`, when it is a Sparse/Multi
    /// collection.
    pub fn kind_store(
        &self,
        collection: &str,
    ) -> Result<Option<Arc<crate::kind_store::KindStore>>> {
        Ok(self.handle(collection).and_then(|h| h.as_kind().cloned()))
    }

    /// Collection configuration, or `None` if the collection does not exist.
    ///
    /// Dimension, distance, effective index type, quantization and the effective
    /// ANN configurations are read back from the persisted metadata; remote-only
    /// replication fields are still excluded.
    pub fn collection_config(&self, name: &str) -> Result<Option<CollectionConfig>> {
        if let Some(store) = self.named_store(name)? {
            // Named layouts keep the single-space fields at their defaults;
            // callers must read every parameter from `vectors` instead.
            let mut config = CollectionConfig::new(0, crate::types::DistanceMetric::Cosine);
            config.vectors = store.spaces();
            config.index = IndexConfig::Flat;
            config.max_points = store.max_points();
            return Ok(Some(config));
        }
        if let Some(store) = self.kind_store(name)? {
            let mut config = CollectionConfig::new(store.vector_size(), store.distance());
            config.kind = store.kind();
            match store.kind() {
                crate::types::VectorKind::Sparse => {}
                crate::types::VectorKind::Multi => {
                    config.multi_config = Some(crate::types::MultiConfig::new(store.max_vectors_per_point()));
                }
                crate::types::VectorKind::Dense => {}
            }
            config.index = IndexConfig::Flat;
            config.max_points = store.max_points();
            return Ok(Some(config));
        }
        let Some(store) = self.handle(name).and_then(|h| h.as_dense().cloned()) else {
            return Ok(None);
        };
        let meta = store.meta();
        let mut config = CollectionConfig::new(meta.vector_size, meta.distance);
        config.index = meta.index.clone();
        config.quantization_config = meta.quantization_config.clone();
        config.max_points = meta.max_points;
        config.vectors_on_disk = meta.vectors_on_disk;
        config.storage_hints = meta.storage_hints.clone();
        Ok(Some(config))
    }

    /// Detailed collection info.
    pub fn collection_info(&self, name: &str) -> Result<CollectionInfo> {
        if let Some(store) = self.named_store(name)? {
            let live = store.count();
            // Single-space fields carry no meaning for named layouts; the
            // authoritative layout is the per-space map.
            let mut config = CollectionConfig::new(0, crate::types::DistanceMetric::Cosine);
            config.vectors = store.spaces();
            config.index = IndexConfig::Flat;
            config.max_points = store.max_points();
            return Ok(CollectionInfo {
                name: name.to_string(),
                vector_count: live,
                indexed_vector_count: live,
                points_count: live,
                segments_count: 0,
                config,
                status: CollectionStatus::Green,
                index: None,
                memory_bytes: store.memory_bytes(),
            });
        }
        if let Some(store) = self.kind_store(name)? {
            let live = store.count();
            let mut config = CollectionConfig::new(store.vector_size(), store.distance());
            config.kind = store.kind();
            if store.kind() == crate::types::VectorKind::Multi {
                config.multi_config = Some(crate::types::MultiConfig::new(store.max_vectors_per_point()));
            }
            config.index = IndexConfig::Flat;
            config.max_points = store.max_points();
            return Ok(CollectionInfo {
                name: name.to_string(),
                vector_count: live,
                indexed_vector_count: live,
                points_count: live,
                segments_count: 0,
                config,
                status: CollectionStatus::Green,
                index: None,
                memory_bytes: store.memory_bytes(),
            });
        }
        let store = self
            .handle(name)
            .and_then(|h| h.as_dense().cloned())
            .ok_or_else(|| VectorSearchError::CollectionNotFound(name.to_string()))?;
        let meta = store.meta();
        let live = store.count();
        let segments = (meta.next_slot as usize).div_ceil(meta.segment_slots as usize);
        let mut config = CollectionConfig::new(meta.vector_size, meta.distance);
        config.index = meta.index.clone();
        config.quantization_config = meta.quantization_config.clone();
        config.max_points = meta.max_points;
        config.vectors_on_disk = meta.vectors_on_disk;
        config.storage_hints = meta.storage_hints.clone();
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
        self.handle(collection)
            .and_then(|h| h.as_dense().cloned())
            .ok_or_else(|| VectorSearchError::CollectionNotFound(collection.to_string()))
    }
}

/// Move an unopenable collection directory out of the load path.
///
/// Renaming keeps the bytes around for offline recovery while making the
/// directory invisible to the loader (`*.suspect-*` is skipped). Returns the
/// new path on success.
fn quarantine(path: &Path, name: &str) -> Option<PathBuf> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let target = path.with_file_name(format!("{name}.suspect-{stamp}"));
    match std::fs::rename(path, &target) {
        Ok(()) => Some(target),
        Err(err) => {
            tracing::error!(
                collection = %name,
                error = %err,
                "failed to quarantine unopenable collection; skipping in place"
            );
            None
        }
    }
}
