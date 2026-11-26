//! Local vector engine.
//!
//! Owns one collection directory per collection under a root directory and
//! keeps an in-memory registry of opened [`CollectionStore`]s. This is the
//! transport-independent engine surface; the sync coordinator wraps it
//! in an async shell (`VectorBackend::Local`).
//!
//! Every mutation is WAL-backed (see [`CollectionStore::apply_txn`]):
//! append + fsync before applying to memory, so crash recovery replays
//! idempotently. Coordinated graph transactions land in [`LocalVectorEngine::apply_txn`].

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::{Mutex, RwLock};

use crate::storage::CollectionStore;
use crate::types::{HnswConfig, IvfConfig, PointId, StorageConfig, VectorPoint};

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
        point_id: PointId,
    },
}

/// Maintenance work scheduled for the background worker thread.
#[derive(Debug)]
enum MaintenanceJob {
    /// Build (or rebuild) the IVF index of a collection.
    Build(String),
    /// Physically reclaim tombstoned slots in a collection.
    Compact(String),
    /// Flush derived files, persist the payload store and truncate the WAL.
    Checkpoint(String),
    /// Drain the HNSW pending queue into the published graph.
    Drain(String),
    /// Stop the worker (sent on engine drop).
    Shutdown,
}

/// How often the idle maintenance worker runs drift / promotion sweeps.
const MAINTENANCE_TICK: Duration = Duration::from_secs(30);

/// Default live-point threshold above which HNSW collections are promoted
/// from exact scan to the published graph (Qdrant's `full_scan_threshold`
/// default) when the config leaves it unset.
const HNSW_PROMOTION_DEFAULT: usize = 10_000;

/// Pending backlog at which mutations ask the maintenance worker to drain
/// the HNSW graph instead of draining inline. The drain is incremental and
/// runs off the caller's thread, so a write storm never stalls a single batch
/// for a multi-second synchronous graph insert.
const PENDING_DRAIN_ENQUEUE: usize = 4096;

pub mod api;
mod core;
mod data;
mod defaults;
mod maintenance;
mod promote;

pub use self::api::VectorEngine;

/// The built-in (local) vector engine.
///
/// All operations are synchronous; the sync coordinator serializes
/// access through an async shell. Collection names must be valid path segments
/// (enforced by [`CollectionStore::create`]).
///
/// A background worker thread performs IVF scheduling: index builds,
/// post-compaction rebuilds and periodic drift checks. It holds only shared
/// handles to the collection map, so dropping the engine shuts it down.
/// A handle to one opened collection, regardless of its storage engine.
///
/// All three store kinds register here under a single collection name, so a
/// read or write dispatch resolves the collection once and matches a single
/// enum instead of probing three independent maps.
#[derive(Clone)]
pub enum SpaceHandle {
    Dense(Arc<CollectionStore>),
    Kind(Arc<crate::kind_store::KindStore>),
    Named(Arc<crate::named_store::NamedStore>),
}

impl SpaceHandle {
    /// The dense store, when this handle is a dense collection. Maintenance
    /// and graph promotion are dense-only and use this accessor.
    pub fn as_dense(&self) -> Option<&Arc<CollectionStore>> {
        match self {
            Self::Dense(store) => Some(store),
            _ => None,
        }
    }

    pub fn as_named(&self) -> Option<&Arc<crate::named_store::NamedStore>> {
        match self {
            Self::Named(store) => Some(store),
            _ => None,
        }
    }

    pub fn as_kind(&self) -> Option<&Arc<crate::kind_store::KindStore>> {
        match self {
            Self::Kind(store) => Some(store),
            _ => None,
        }
    }

    /// Directory backing the collection, independent of store kind.
    pub fn dir(&self) -> &std::path::Path {
        match self {
            Self::Dense(store) => store.dir(),
            Self::Kind(store) => store.dir(),
            Self::Named(store) => store.dir(),
        }
    }
}

impl std::fmt::Debug for SpaceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(_) => f.write_str("SpaceHandle::Dense"),
            Self::Kind(_) => f.write_str("SpaceHandle::Kind"),
            Self::Named(_) => f.write_str("SpaceHandle::Named"),
        }
    }
}

/// The single collection registry. A name maps to exactly one handle.
pub type StoreRegistry = Arc<RwLock<HashMap<String, SpaceHandle>>>;

pub struct LocalVectorEngine {
    root_dir: PathBuf,
    /// Single registry of every opened collection, keyed by name.
    stores: StoreRegistry,
    /// Engine-level storage tuning applied to every collection this engine
    /// opens or creates.
    storage: StorageConfig,
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
            .field("collection_count", &self.stores.read().len())
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

/// Background maintenance loop: executes scheduled builds and, on each idle
/// tick, sweeps collections for post-compaction rebuilds, drift-triggered
/// rebuilds and exact-scan-to-IVF promotion.
#[cfg(test)]
#[path = "engine/tests.rs"]
mod tests;
