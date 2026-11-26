use std::collections::HashMap;

use crate::error::{Result, VectorSearchError};
use crate::types::{CollectionConfig, CollectionInfo, CollectionStatus, DistanceMetric};

use super::{validate_name, MemoryCollection, MemoryEngine};

impl MemoryEngine {
    /// Names of all collections, sorted.
    pub fn collection_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .collections
            .read()
            .map(|c| c.keys().cloned().collect())
            .unwrap_or_default();
        names.sort_unstable();
        names
    }

    /// Create a collection. Fails when it already exists.
    pub fn create_collection(&self, name: &str, config: &CollectionConfig) -> Result<()> {
        validate_name(name)?;
        if config.vector_size == 0 {
            return Err(VectorSearchError::InvalidVectorDimension {
                expected: 1,
                actual: 0,
            });
        }
        if !matches!(
            config.distance,
            DistanceMetric::Cosine
                | DistanceMetric::Euclid
                | DistanceMetric::Dot
                | DistanceMetric::Manhattan
        ) {
            return Err(VectorSearchError::UnsupportedMetric(config.distance));
        }
        let mut guard = self
            .collections
            .write()
            .map_err(|_| VectorSearchError::Internal("memory engine lock poisoned".to_string()))?;
        if guard.contains_key(name) {
            return Err(VectorSearchError::CollectionAlreadyExists(name.to_string()));
        }
        guard.insert(
            name.to_string(),
            MemoryCollection {
                config: config.clone(),
                points: HashMap::new(),
                payload_indexes: Vec::new(),
            },
        );
        self.bump_revision();
        Ok(())
    }

    /// Drop a collection and all its points.
    pub fn delete_collection(&self, name: &str) -> Result<()> {
        let mut guard = self
            .collections
            .write()
            .map_err(|_| VectorSearchError::Internal("memory engine lock poisoned".to_string()))?;
        if guard.remove(name).is_none() {
            return Err(VectorSearchError::CollectionNotFound(name.to_string()));
        }
        self.bump_revision();
        Ok(())
    }

    /// Whether a collection exists.
    pub fn collection_exists(&self, name: &str) -> bool {
        self.collections
            .read()
            .map(|c| c.contains_key(name))
            .unwrap_or(false)
    }

    /// Stored configuration for a collection.
    pub fn collection_config(&self, name: &str) -> Result<Option<CollectionConfig>> {
        self.with_collection(name, |c| Ok(Some(c.config.clone())))
    }

    /// Detailed collection info with exact live counts.
    pub fn collection_info(&self, name: &str) -> Result<CollectionInfo> {
        self.with_collection(name, |c| {
            let live = c.live_count();
            Ok(CollectionInfo {
                name: name.to_string(),
                vector_count: live,
                indexed_vector_count: live,
                points_count: live,
                segments_count: 1,
                config: c.config.clone(),
                status: CollectionStatus::Green,
                index: None,
                memory_bytes: live * c.config.vector_size as u64 * 4,
            })
        })
    }
}
