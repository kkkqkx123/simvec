//! `NamedStore` — named-vectors collections.
//!
//! A named collection declares a map of vector spaces (dense or sparse, each
//! with its own dimension and distance metric). Points carry one vector per
//! declared space and every query targets exactly one space by name. This
//! module mirrors [`crate::kind_store::KindStore`]: an in-memory exact-scan
//! engine with variable-length payloads and WAL-backed durability, rebuilt
//! by full replay on open. No ANN indexes, no quantization.

#[cfg(test)]
#[path = "named_store_tests.rs"]
mod tests;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Result, VectorSearchError};
use crate::filter;
use crate::kind_store::sparse_score;
use crate::storage::{Wal, WalRecord, WalTxn};
use crate::types::{
    Payload, PointId, SearchQuery, SearchResult, VectorData, VectorFilter, VectorKind, VectorPoint,
    VectorSpaceConfig,
};

/// One live point of a named collection: a full vector per declared space.
#[derive(Debug, Clone)]
struct NamedPoint {
    id: PointId,
    vectors: BTreeMap<String, VectorData>,
    payload: Option<Payload>,
}

/// In-memory mutable state of a named collection.
#[derive(Debug, Default)]
struct Inner {
    /// Slot-indexed live points; `None` marks a tombstoned slot.
    points: Vec<Option<NamedPoint>>,
    /// Point id -> live slot directory.
    reverse: HashMap<String, u32>,
    /// Per-space sparse inverted index: space -> term -> (slot, weight).
    /// Dense spaces have no entry.
    inverted: HashMap<String, HashMap<u32, Vec<(u32, f32)>>>,
    next_lsn: u64,
    live_count: u64,
}

impl Inner {
    fn tombstone_slot(&mut self, slot: u32, sparse_spaces: &[&str]) {
        let Some(point) = self.points[slot as usize].take() else {
            return;
        };
        self.reverse.remove(point.id.to_string().as_str());
        for space in sparse_spaces {
            if let Some(VectorData::Sparse { indices, .. }) = point.vectors.get(*space) {
                let space_index = self.inverted.get_mut(*space);
                if let Some(space_index) = space_index {
                    for index in indices {
                        if let Some(postings) = space_index.get_mut(index) {
                            postings.retain(|(s, _)| *s != slot);
                            if postings.is_empty() {
                                space_index.remove(index);
                            }
                        }
                    }
                }
            }
        }
        self.live_count -= 1;
    }
}

/// Persisted per-collection metadata (`named.bin`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct NamedMeta {
    collection: String,
    /// Declared spaces keyed by name, in stable (BTreeMap) order.
    spaces: BTreeMap<String, NamedSpaceMeta>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct NamedSpaceMeta {
    kind: VectorKind,
    vector_size: usize,
    distance: crate::types::DistanceMetric,
}

/// A single opened named-vectors collection.
pub struct NamedStore {
    dir: PathBuf,
    spaces: BTreeMap<String, NamedSpaceMeta>,
    inner: Mutex<Inner>,
    wal: Wal,
    /// Engine-level storage tuning this collection was opened with.
    storage: crate::types::StorageConfig,
}

impl std::fmt::Debug for NamedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedStore")
            .field("dir", &self.dir)
            .field("spaces", &self.spaces.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl NamedStore {
    /// Create a new named collection directory (`named.bin` marker + WAL).
    pub fn create(
        dir: &Path,
        name: &str,
        spaces: &BTreeMap<String, VectorSpaceConfig>,
        storage: &crate::types::StorageConfig,
    ) -> Result<Arc<Self>> {
        std::fs::create_dir_all(dir)?;
        let meta = NamedMeta {
            collection: name.to_string(),
            spaces: spaces
                .iter()
                .map(|(name, config)| {
                    (
                        name.clone(),
                        NamedSpaceMeta {
                            kind: config.kind,
                            vector_size: config.vector_size,
                            distance: config.distance,
                        },
                    )
                })
                .collect(),
        };
        let bytes = postcard::to_stdvec(&meta)?;
        std::fs::write(dir.join("named.bin"), &bytes)?;
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        Ok(Arc::new(Self {
            dir: dir.to_path_buf(),
            spaces: meta.spaces,
            inner: Mutex::new(Inner::default()),
            wal,
            storage: storage.clone(),
        }))
    }

    /// Open an existing named collection, rebuilding the in-memory state by
    /// replaying the WAL.
    pub fn open(dir: &Path, storage: &crate::types::StorageConfig) -> Result<Arc<Self>> {
        let bytes = std::fs::read(dir.join("named.bin"))?;
        let meta: NamedMeta = postcard::from_bytes(&bytes)?;
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        let store = Arc::new(Self {
            dir: dir.to_path_buf(),
            spaces: meta.spaces,
            inner: Mutex::new(Inner::default()),
            wal,
            storage: storage.clone(),
        });
        let next_lsn = store.wal.replay(|txn| store.replay_txn(txn))?;
        store.inner.lock().next_lsn = next_lsn.saturating_add(1);
        Ok(store)
    }

    fn replay_txn(&self, txn: &WalTxn) -> Result<()> {
        let mut inner = self.inner.lock();
        for record in &txn.ops {
            match record {
                WalRecord::UpsertNamed { id, vectors, payload } => {
                    self.apply_upsert(&mut inner, id.clone(), vectors.clone(), payload.clone())?;
                }
                WalRecord::Delete { point_id } => {
                    if let Some(slot) = inner.reverse.get(point_id).copied() {
                        inner.tombstone_slot(slot, &self.sparse_space_names());
                    }
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for point_id in point_ids {
                        if let Some(slot) = inner.reverse.get(point_id).copied() {
                            inner.tombstone_slot(slot, &self.sparse_space_names());
                        }
                    }
                }
                WalRecord::SetPayload { slot, payload } => {
                    if let Some(Some(point)) = inner.points.get_mut(*slot as usize) {
                        point.payload = payload
                            .as_ref()
                            .map(|p| serde_json::from_str(p))
                            .transpose()?;
                    }
                }
                WalRecord::DeletePayloadKeys { slot, keys } => {
                    if let Some(Some(point)) = inner.points.get_mut(*slot as usize) {
                        if let Some(payload) = &mut point.payload {
                            for key in keys {
                                payload.remove(key);
                            }
                        }
                    }
                }
                WalRecord::SetPayloadField { slot, key, value } => {
                    if let Some(Some(point)) = inner.points.get_mut(*slot as usize) {
                        let payload = point.payload.get_or_insert_with(Default::default);
                        let value: serde_json::Value = serde_json::from_str(value)?;
                        payload.insert(key.clone(), value);
                    }
                }
                WalRecord::Compact => {}
                WalRecord::Upsert { .. } | WalRecord::UpsertKind { .. } | WalRecord::Quantize
                | WalRecord::DropCollection => {
                    return Err(VectorSearchError::CorruptData(format!(
                        "unexpected WAL record in named collection {}",
                        self.dir.display()
                    )));
                }
            }
        }
        Ok(())
    }

    fn apply_upsert(
        &self,
        inner: &mut Inner,
        id: String,
        vectors: BTreeMap<String, VectorData>,
        payload: Option<String>,
    ) -> Result<()> {
        let payload: Option<Payload> = payload
            .as_ref()
            .map(|p| serde_json::from_str(p))
            .transpose()?;
        self.validate_vectors(&vectors)?;
        // Idempotent overwrite: a replayed or repeated upsert of the same id
        // replaces the stored point in place (the old slot is tombstoned and
        // its postings dropped).
        if let Some(&slot) = inner.reverse.get(&id) {
            inner.tombstone_slot(slot, &self.sparse_space_names());
        }
        let slot = inner.points.len() as u32;
        for (space, data) in &vectors {
            if let VectorData::Sparse { indices, values } = data {
                let space_index = inner.inverted.entry(space.clone()).or_default();
                for (index, value) in indices.iter().zip(values.iter()) {
                    space_index.entry(*index).or_default().push((slot, *value));
                }
            }
        }
        inner.reverse.insert(id.clone(), slot);
        inner.points.push(Some(NamedPoint {
            id: PointId::from(id),
            vectors,
            payload,
        }));
        inner.live_count += 1;
        Ok(())
    }

    fn validate_vectors(&self, vectors: &BTreeMap<String, VectorData>) -> Result<()> {
        if vectors.len() != self.spaces.len() {
            return Err(VectorSearchError::InvalidConfig(format!(
                "named point carries {} spaces but the collection declares {}",
                vectors.len(),
                self.spaces.len()
            )));
        }
        for (name, data) in vectors {
            let Some(space) = self.spaces.get(name) else {
                return Err(VectorSearchError::InvalidConfig(format!(
                    "unknown named-vector space {name}"
                )));
            };
            if data.kind() != space.kind {
                return Err(VectorSearchError::InvalidConfig(format!(
                    "space {name} kind is {}, payload kind is {}",
                    space.kind.as_name(),
                    data.kind().as_name()
                )));
            }
            data.validate(space.vector_size, 1)?;
        }
        Ok(())
    }

    fn sparse_space_names(&self) -> Vec<&str> {
        self.spaces
            .iter()
            .filter(|(_, space)| space.kind == VectorKind::Sparse)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

impl NamedStore {
    /// Declared named-vector spaces.
    pub fn spaces(&self) -> BTreeMap<String, VectorSpaceConfig> {
        self.spaces
            .iter()
            .map(|(name, meta)| {
                (
                    name.clone(),
                    VectorSpaceConfig {
                        kind: meta.kind,
                        vector_size: meta.vector_size,
                        distance: meta.distance,
                    },
                )
            })
            .collect()
    }

    /// Directory backing this collection.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Engine-level storage tuning.
    pub fn storage_config(&self) -> &crate::types::StorageConfig {
        &self.storage
    }

    /// Number of live points.
    pub fn count(&self) -> u64 {
        self.inner.lock().live_count
    }

    /// Number of live points matching `filter`. Pure read.
    pub fn count_by_filter(&self, filter: &VectorFilter) -> Result<u64> {
        let inner = self.inner.lock();
        let mut count = 0u64;
        for (slot, point) in inner.points.iter().enumerate().filter_map(|(i, p)| {
            p.is_some().then_some((i, p.as_ref().expect("live point")))
        }) {
            let _ = slot;
            if filter::matches(filter, &point.id, point.payload.as_ref())? {
                count += 1;
            }
        }
        Ok(count)
    }

    /// WAL-backed upsert of a batch of named points. Each point must carry
    /// the full declared space set; the upsert replaces the point's whole
    /// named map.
    pub fn upsert_batch(&self, points: &[VectorPoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let mut records = Vec::with_capacity(points.len());
        for point in points {
            records.push(named_upsert_record(point)?);
        }
        self.commit(records)
    }

    /// WAL-backed delete by id. Deleting a missing id is a no-op.
    pub fn delete(&self, point_id: &PointId) -> Result<()> {
        self.delete_batch(std::slice::from_ref(point_id))
    }

    /// WAL-backed delete of a batch of ids.
    pub fn delete_batch(&self, point_ids: &[PointId]) -> Result<()> {
        if point_ids.is_empty() {
            return Ok(());
        }
        let records = point_ids
            .iter()
            .map(|id| WalRecord::Delete {
                point_id: id.to_string(),
            })
            .collect();
        self.commit(records)
    }

    /// WAL-backed delete of every point matching `filter`.
    pub fn delete_by_filter(&self, filter: &VectorFilter) -> Result<u64> {
        let mut records = Vec::new();
        {
            let inner = self.inner.lock();
            for point in inner.points.iter().flatten() {
                if filter::matches(filter, &point.id, point.payload.as_ref())? {
                    records.push(WalRecord::Delete {
                        point_id: point.id.to_string(),
                    });
                }
            }
        }
        let deleted = records.len() as u64;
        if deleted > 0 {
            self.commit(records)?;
        }
        Ok(deleted)
    }

    /// Apply already-encoded WAL records without re-logging them. Used by
    /// the engine's coordinated-transaction path.
    pub fn apply_records(&self, records: &[WalRecord]) -> Result<()> {
        let mut inner = self.inner.lock();
        for record in records {
            match record {
                WalRecord::UpsertNamed { id, vectors, payload } => {
                    self.apply_upsert(&mut inner, id.clone(), vectors.clone(), payload.clone())?;
                }
                WalRecord::Delete { point_id } => {
                    if let Some(slot) = inner.reverse.get(point_id).copied() {
                        inner.tombstone_slot(slot, &self.sparse_space_names());
                    }
                }
                _ => {
                    return Err(VectorSearchError::Internal(
                        "named store transaction received an unsupported record".to_string(),
                    ));
                }
            }
        }
        inner.next_lsn += 1;
        Ok(())
    }

    /// Commit `records` to the WAL and apply them in order.
    fn commit(&self, records: Vec<WalRecord>) -> Result<()> {
        let mut inner = self.inner.lock();
        let txn_id = inner.next_lsn;
        inner.next_lsn += 1;
        self.wal.append(&WalTxn {
            txn_id,
            ops: records.clone(),
        })?;
        for record in records {
            match record {
                WalRecord::UpsertNamed { id, vectors, payload } => {
                    self.apply_upsert(&mut inner, id, vectors, payload)?;
                }
                WalRecord::Delete { point_id } => {
                    if let Some(slot) = inner.reverse.get(&point_id).copied() {
                        inner.tombstone_slot(slot, &self.sparse_space_names());
                    }
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for point_id in point_ids {
                        if let Some(slot) = inner.reverse.get(&point_id).copied() {
                            inner.tombstone_slot(slot, &self.sparse_space_names());
                        }
                    }
                }
                WalRecord::SetPayload { slot, payload } => {
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        point.payload = payload
                            .as_ref()
                            .map(|p| serde_json::from_str(p))
                            .transpose()?;
                    }
                }
                WalRecord::DeletePayloadKeys { slot, keys } => {
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        if let Some(payload) = &mut point.payload {
                            for key in keys {
                                payload.remove(&key);
                            }
                        }
                    }
                }
                WalRecord::SetPayloadField { slot, key, value } => {
                    if let Some(Some(point)) = inner.points.get_mut(slot as usize) {
                        let payload = point.payload.get_or_insert_with(Default::default);
                        let value: serde_json::Value = serde_json::from_str(&value)?;
                        payload.insert(key, value);
                    }
                }
                WalRecord::Upsert { .. } | WalRecord::UpsertKind { .. } | WalRecord::Compact
                | WalRecord::Quantize | WalRecord::DropCollection => {
                    return Err(VectorSearchError::Internal(
                        "named store commit received an unsupported record".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl NamedStore {
    /// Exact-scan top-k search over one named space.
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let target = query.target.as_deref().ok_or_else(|| {
            VectorSearchError::InvalidConfig(
                "named collection queries must declare a target space".to_string(),
            )
        })?;
        let data = query.named_vector.as_ref().ok_or_else(|| {
            VectorSearchError::InvalidConfig(
                "named collection queries must carry a named query vector".to_string(),
            )
        })?;
        if !query.vector.is_empty() || query.data.is_some() {
            return Err(VectorSearchError::InvalidConfig(
                "named collection queries must not carry single-space payloads".to_string(),
            ));
        }
        if query.quantization.is_some() || query.nprobe.is_some() {
            return Err(VectorSearchError::InvalidConfig(
                "quantization and nprobe controls only apply to dense collections".to_string(),
            ));
        }
        let space = self.spaces.get(target).ok_or_else(|| {
            VectorSearchError::InvalidConfig(format!("unknown named-vector space {target}"))
        })?;
        if data.kind() != space.kind {
            return Err(VectorSearchError::InvalidConfig(format!(
                "space {target} kind is {}, query kind is {}",
                space.kind.as_name(),
                data.kind().as_name()
            )));
        }
        data.validate(space.vector_size, 1)?;

        let inner = self.inner.lock();
        let limit = query.effective_limit();
        let offset = query.offset.unwrap_or(0);
        if limit == 0 {
            return Ok(Vec::new());
        }

        // Candidate set: inverted postings for sparse spaces, every live
        // slot for dense spaces.
        let candidates: Vec<u32> = match data {
            VectorData::Sparse { indices, .. } => {
                inner
                    .inverted
                    .get(target)
                    .and_then(|space_index| {
                        sparse_postings(space_index, indices, inner.points.len())
                    })
                    .unwrap_or_default()
            }
            _ => (0..inner.points.len() as u32)
                .filter(|slot| inner.points[*slot as usize].is_some())
                .collect(),
        };

        let metric = space.distance;
        let mut scored: Vec<(u32, f32)> = Vec::new();
        let mut skipped = 0usize;
        for slot in candidates {
            let Some(point) = inner.points[slot as usize].as_ref() else {
                continue;
            };
            if let Some(filter) = &query.filter {
                if !filter::matches(filter, &point.id, point.payload.as_ref())? {
                    continue;
                }
            }
            let stored = point
                .vectors
                .get(target)
                .expect("validated point carries every space");
            let score = score_pair(data, stored, metric);
            if let Some(threshold) = query.score_threshold {
                if score < threshold {
                    continue;
                }
            }
            if skipped < offset {
                skipped += 1;
                continue;
            }
            scored.push((slot, score));
        }

        scored.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(limit);

        Ok(scored
            .into_iter()
            .filter_map(|(slot, score)| {
                let point = inner.points[slot as usize].as_ref()?;
                Some(SearchResult::new(point.id.clone(), score).with_payload(
                    match (&point.payload, query.with_payload.unwrap_or(true)) {
                        (Some(payload), true) => payload.clone(),
                        _ => Default::default(),
                    },
                ))
            })
            .collect())
    }

    /// Fetch a point by id.
    pub fn get(&self, point_id: &PointId) -> Result<Option<VectorPoint>> {
        let inner = self.inner.lock();
        let Some(&slot) = inner.reverse.get(&point_id.to_string()) else {
            return Ok(None);
        };
        Ok(inner.points[slot as usize].as_ref().map(|point| VectorPoint {
            id: point.id.clone(),
            vector: Vec::new(),
            data: None,
            named: point.vectors.clone(),
            payload: point.payload.clone(),
        }))
    }

    /// Replace the payload for a single point.
    pub fn set_payload(&self, point_id: &PointId, payload: Payload) -> Result<()> {
        let slot = self.require_slot(point_id)?;
        self.commit(vec![WalRecord::SetPayload {
            slot,
            payload: Some(serde_json::to_string(&payload)?),
        }])
    }

    /// Remove specific keys from a point's payload.
    pub fn delete_payload_keys(&self, point_id: &PointId, keys: &[String]) -> Result<()> {
        let slot = self.require_slot(point_id)?;
        self.commit(vec![WalRecord::DeletePayloadKeys {
            slot,
            keys: keys.to_vec(),
        }])
    }

    /// Merge the given fields into a point's payload.
    pub fn set_payload_fields(&self, point_id: &PointId, fields: Payload) -> Result<()> {
        let slot = self.require_slot(point_id)?;
        let mut records = Vec::with_capacity(fields.len());
        for (key, value) in fields {
            records.push(WalRecord::SetPayloadField {
                slot,
                key,
                value: serde_json::to_string(&value)?,
            });
        }
        self.commit(records)
    }

    /// Paginated scan over live points in slot order.
    pub fn scroll(
        &self,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        self.scroll_impl(limit, offset, with_payload, with_vector, None)
    }

    /// Paginated scan over live points matching `filter`, in slot order.
    /// `limit` counts only matching points; the cursor is the last match.
    pub fn scroll_by_filter(
        &self,
        filter: &VectorFilter,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        self.scroll_impl(limit, offset, with_payload, with_vector, Some(filter))
    }

    fn scroll_impl(
        &self,
        limit: usize,
        offset: Option<&PointId>,
        with_payload: Option<bool>,
        with_vector: Option<bool>,
        filter: Option<&VectorFilter>,
    ) -> Result<(Vec<VectorPoint>, Option<PointId>)> {
        let inner = self.inner.lock();
        let start_slot = match offset {
            Some(cursor) => inner
                .reverse
                .get(&cursor.to_string())
                .map(|slot| slot + 1)
                .unwrap_or(inner.points.len() as u32),
            None => 0,
        };
        let with_payload = with_payload.unwrap_or(true);
        let with_vector = with_vector.unwrap_or(false);
        let mut points = Vec::new();
        let mut cursor = None;
        for slot in start_slot as usize..inner.points.len() {
            let Some(point) = inner.points[slot].as_ref() else {
                continue;
            };
            if let Some(filter) = filter {
                if !filter::matches(filter, &point.id, point.payload.as_ref())? {
                    continue;
                }
            }
            points.push(VectorPoint {
                id: point.id.clone(),
                vector: Vec::new(),
                data: None,
                named: if with_vector {
                    point.vectors.clone()
                } else {
                    Default::default()
                },
                payload: with_payload.then(|| point.payload.clone()).flatten(),
            });
            cursor = Some(point.id.clone());
            if points.len() >= limit {
                break;
            }
        }
        Ok((points, cursor))
    }

    fn require_slot(&self, point_id: &PointId) -> Result<u32> {
        let inner = self.inner.lock();
        inner
            .reverse
            .get(&point_id.to_string())
            .copied()
            .ok_or_else(|| {
                VectorSearchError::Internal(format!("point {point_id} not found"))
            })
    }
}

/// Union the postings of `indices`, deduplicated by slot.
fn sparse_postings(
    space_index: &HashMap<u32, Vec<(u32, f32)>>,
    indices: &[u32],
    _live: usize,
) -> Option<Vec<u32>> {
    let mut candidates = Vec::new();
    for index in indices {
        if let Some(postings) = space_index.get(index) {
            for (slot, _) in postings {
                candidates.push(*slot);
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    Some(candidates)
}

/// Score a (query, stored) payload pair within one named space.
///
/// Scores follow the crate-wide higher-is-better contract: raw dot products
/// for Cosine/Dot and `1/(1 + distance)` for Euclid/Manhattan.
fn score_pair(query: &VectorData, stored: &VectorData, metric: crate::types::DistanceMetric) -> f32 {
    match (query, stored) {
        (
            VectorData::Sparse {
                indices: q_idx,
                values: q_val,
            },
            VectorData::Sparse {
                indices: d_idx,
                values: d_val,
            },
        ) => sparse_score(q_idx, q_val, d_idx, d_val, metric),
        (VectorData::Dense(q), VectorData::Dense(d)) => dense_similarity(q, d, metric),
        _ => 0.0,
    }
}

/// Dense similarity within a named space: the shared SIMD distance kernel
/// remapped onto the higher-is-better contract.
fn dense_similarity(query: &[f32], stored: &[f32], metric: crate::types::DistanceMetric) -> f32 {
    let distance = crate::distance::distance(metric, query, stored);
    match metric {
        crate::types::DistanceMetric::Cosine => 1.0 - distance,
        crate::types::DistanceMetric::Dot => -distance,
        crate::types::DistanceMetric::Euclid | crate::types::DistanceMetric::Manhattan => {
            1.0 / (1.0 + distance)
        }
    }
}

/// Build the WAL upsert record for a named point.
pub fn named_upsert_record(point: &VectorPoint) -> Result<WalRecord> {
    if point.named.is_empty() {
        return Err(VectorSearchError::InvalidConfig(
            "named collection upserts must carry named-space vectors".to_string(),
        ));
    }
    if !point.vector.is_empty() || point.data.is_some() {
        return Err(VectorSearchError::InvalidConfig(
            "named collection upserts must not carry single-space payloads".to_string(),
        ));
    }
    Ok(WalRecord::UpsertNamed {
        id: point.id.to_string(),
        vectors: point.named.clone(),
        payload: point
            .payload
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?,
    })
}
