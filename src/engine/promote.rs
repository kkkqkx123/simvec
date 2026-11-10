use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::{Builder as ThreadBuilder, JoinHandle};
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};

use super::{LocalVectorEngine, MaintenanceJob, TxnOp};
use super::*;
use crate::error::{Result, VectorSearchError};
use crate::metrics::MetricsSnapshot;
use crate::storage::{CollectionStore, WalPoint, WalRecord};
use crate::types::{CollectionConfig, CollectionInfo, CollectionStatus, HnswConfig, IndexConfig, IvfConfig, Payload, PayloadSchemaType, PointId, SearchQuery, SearchResult, VectorFilter, VectorPoint};

impl LocalVectorEngine {
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
                    self.collections.write().insert(live.to_string(), store);
                } else if let Some(store) = live_store {
                    self.collections.write().insert(live.to_string(), store);
                }
            } else if backup_dir.exists() {
                let _ = std::fs::rename(&backup_dir, &live_dir);
                if let Ok(store) = CollectionStore::open(&live_dir) {
                    self.collections.write().insert(live.to_string(), store);
                } else if let Some(store) = live_store {
                    self.collections.write().insert(live.to_string(), store);
                }
            } else if let Some(store) = live_store {
                self.collections.write().insert(live.to_string(), store);
            }
            return Err(error);
        }
        let store = CollectionStore::open(&live_dir)?;
        self.collections.write().insert(live.to_string(), store);
        // Keep the backup handle loaded for inspection (`count`, reverse
        // recovery) without exposing it to search: nothing routes to
        // `.old-*` names and `collection_names` hides them.
        if backup_dir.exists() {
            if let Ok(backup_store) = CollectionStore::open(&backup_dir) {
                self.collections
                    .write()
                    .insert(backup_name.clone(), backup_store);
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
        self.collections.write().insert(live.to_string(), store);
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

    pub(super) fn list_old_backups(root: &Path, live: &str) -> Vec<String> {
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

    pub(super) fn prune_old_backups(&self, live: &str, keep: Option<&str>) {
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
}

