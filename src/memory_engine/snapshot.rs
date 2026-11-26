use std::collections::HashMap;

use crate::error::{Result, VectorSearchError};
use crate::types::VectorPoint;

use super::{
    validate_name, validate_point, MemoryCollection, MemoryEngine, MemorySnapshot,
    MemorySnapshotCollection,
};

fn snapshot_entry(name: &str, collection: &MemoryCollection) -> MemorySnapshotCollection {
    let mut points: Vec<VectorPoint> = collection.points.values().cloned().collect();
    points.sort_by_key(|a| a.id.to_string());
    MemorySnapshotCollection {
        name: name.to_string(),
        config: collection.config.clone(),
        points,
        payload_indexes: collection.payload_indexes.clone(),
    }
}

fn staged_collection(
    entry: MemorySnapshotCollection,
    staged: &HashMap<String, MemoryCollection>,
) -> Result<(String, MemoryCollection)> {
    validate_name(&entry.name)?;
    if entry.config.vector_size == 0 {
        return Err(VectorSearchError::InvalidVectorDimension {
            expected: 1,
            actual: 0,
        });
    }
    for point in &entry.points {
        validate_point(&entry.config, point)?;
    }
    if staged.contains_key(&entry.name) {
        return Err(VectorSearchError::CollectionAlreadyExists(entry.name));
    }
    if let Some(cap) = entry.config.max_points {
        if entry.points.len() as u64 > cap {
            return Err(VectorSearchError::CapacityExceeded {
                collection: entry.name,
                max_points: cap,
                requested: entry.points.len() as u64,
            });
        }
    }
    let mut points = HashMap::with_capacity(entry.points.len());
    for point in entry.points {
        points.insert(point.id.clone(), point);
    }
    Ok((
        entry.name,
        MemoryCollection {
            config: entry.config,
            points,
            payload_indexes: entry.payload_indexes,
        },
    ))
}

impl MemoryEngine {
    /// Export the whole engine as a versioned snapshot value.
    pub fn export_snapshot(&self) -> Result<MemorySnapshot> {
        let guard = self
            .collections
            .read()
            .map_err(|_| VectorSearchError::Internal("memory engine lock poisoned".to_string()))?;
        let mut collections: Vec<MemorySnapshotCollection> = Vec::with_capacity(guard.len());
        for (name, collection) in guard.iter() {
            collections.push(snapshot_entry(name, collection));
        }
        collections.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(MemorySnapshot {
            version: 1,
            collections,
        })
    }

    /// Export one collection as a snapshot entry.
    pub fn export_collection(&self, name: &str) -> Result<MemorySnapshotCollection> {
        self.with_collection(name, |c| Ok(snapshot_entry(name, c)))
    }

    /// Insert or replace one collection from a snapshot entry.
    pub fn import_collection(&self, entry: MemorySnapshotCollection) -> Result<()> {
        let (name, collection) = staged_collection(entry, &HashMap::new())?;
        let mut guard = self
            .collections
            .write()
            .map_err(|_| VectorSearchError::Internal("memory engine lock poisoned".to_string()))?;
        guard.insert(name, collection);
        self.bump_revision();
        Ok(())
    }

    /// Replace the whole engine from a snapshot after validating every entry.
    pub fn import_snapshot(&self, snapshot: MemorySnapshot) -> Result<()> {
        if snapshot.version != 1 {
            return Err(VectorSearchError::InvalidConfig(format!(
                "unsupported memory snapshot version {}",
                snapshot.version
            )));
        }
        let mut staged: HashMap<String, MemoryCollection> = HashMap::new();
        for entry in snapshot.collections {
            let (name, collection) = staged_collection(entry, &staged)?;
            staged.insert(name, collection);
        }
        let mut guard = self
            .collections
            .write()
            .map_err(|_| VectorSearchError::Internal("memory engine lock poisoned".to_string()))?;
        *guard = staged;
        self.bump_revision();
        Ok(())
    }
}
