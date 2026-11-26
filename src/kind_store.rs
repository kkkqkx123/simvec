//! `KindStore` — sparse and multi-vector collections.
//!
//! Dense collections run through [`crate::storage::CollectionStore`] with its
//! fixed-slot `vectors.bin` and the HNSW/IVF/quantization pipeline. Sparse
//! and Multi (late-interaction) payloads do not fit that pipeline: their data
//! is variable length and neither kind uses ANN indexes or quantization.
//! This module hosts a self-contained exact-scan engine for those kinds:
//!
//! - in-memory slot table with id -> slot directory;
//! - inverted postings per sparse term for candidate collection;
//! - weighted dot / Euclid / Manhattan scoring for Sparse and exact MaxSim
//!   for Multi;
//! - WAL-backed durability by replaying the shared `wal.bin` record format
//!   (`WalRecord::UpsertKind` carries the non-dense payload).
//!
//! The store is a reference implementation on purpose: it keeps every live
//! point in memory and rebuilds by full WAL replay, mirroring the memory
//! backend's role in `vector-contracts`.

#[cfg(test)]
#[path = "kind_store_tests.rs"]
mod tests;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Result, VectorSearchError};
use crate::filter;
use crate::storage::{Wal, WalRecord, WalTxn};
use crate::types::{
    Payload, PointId, SearchQuery, SearchResult, VectorData, VectorFilter, VectorKind,
    VectorPoint,
};

/// One live point of a Sparse/Multi collection.
#[derive(Debug, Clone)]
struct KindPoint {
    id: PointId,
    data: VectorData,
    payload: Option<Payload>,
}

/// In-memory mutable state of a kind collection.
#[derive(Debug, Default)]
struct Inner {
    /// Slot-indexed live points; `None` marks a tombstoned slot.
    points: Vec<Option<KindPoint>>,
    /// Point id -> live slot directory.
    reverse: HashMap<String, u32>,
    /// Sparse-only inverted index: term -> (slot, weight) postings. Empty for
    /// Multi collections.
    inverted: HashMap<u32, Vec<(u32, f32)>>,
    next_lsn: u64,
    live_count: u64,
}

impl Inner {
    fn tombstone_slot(&mut self, slot: u32) {
        let Some(point) = self.points[slot as usize].take() else {
            return;
        };
        self.reverse.remove(point.id.to_string().as_str());
        if let VectorData::Sparse { indices, .. } = &point.data {
            for index in indices {
                if let Some(postings) = self.inverted.get_mut(index) {
                    postings.retain(|(s, _)| *s != slot);
                    if postings.is_empty() {
                        self.inverted.remove(index);
                    }
                }
            }
        }
        self.live_count -= 1;
    }

    /// Candidate slots holding at least one of the query terms.
    fn sparse_candidates(&self, indices: &[u32]) -> Vec<u32> {
        let mut candidates = Vec::new();
        for index in indices {
            if let Some(postings) = self.inverted.get(index) {
                for (slot, _) in postings {
                    candidates.push(*slot);
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }
}

/// A single opened Sparse or Multi collection.
pub struct KindStore {
    dir: PathBuf,
    kind: VectorKind,
    /// Dense dimension / sparse vocabulary bound / multi sub-vector dimension.
    dim: usize,
    /// Per-point sub-vector cap for Multi collections; `0` for Sparse.
    max_vectors_per_point: usize,
    distance: crate::types::DistanceMetric,
    inner: Mutex<Inner>,
    wal: Wal,
    /// Engine-level storage tuning this collection was opened with.
    storage: crate::types::StorageConfig,
}

impl std::fmt::Debug for KindStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KindStore")
            .field("dir", &self.dir)
            .field("kind", &self.kind)
            .field("vector_size", &self.dim)
            .finish()
    }
}

impl KindStore {
    /// Create a new kind collection directory (`kind.bin` marker + WAL).
    pub fn create(dir: &Path, name: &str, config: &crate::types::CollectionConfig, storage: &crate::types::StorageConfig) -> Result<Arc<Self>> {
        config.validate_kind()?;
        std::fs::create_dir_all(dir)?;
        let meta = KindMeta {
            collection: name.to_string(),
            kind: config.kind,
            vector_size: config.vector_size,
            distance: config.distance,
            max_vectors_per_point: config.multi_config.map_or(0, |c| c.max_vectors_per_point),
        };
        let bytes = postcard::to_stdvec(&meta)?;
        std::fs::write(dir.join("kind.bin"), &bytes)?;
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        Ok(Arc::new(Self {
            dir: dir.to_path_buf(),
            kind: config.kind,
            dim: config.vector_size,
            max_vectors_per_point: meta.max_vectors_per_point,
            distance: config.distance,
            inner: Mutex::new(Inner::default()),
            wal,
            storage: storage.clone(),
        }))
    }

    /// Open an existing kind collection, rebuilding the in-memory state by
    /// replaying the WAL.
    pub fn open(dir: &Path, storage: &crate::types::StorageConfig) -> Result<Arc<Self>> {
        let bytes = std::fs::read(dir.join("kind.bin"))?;
        let meta: KindMeta = postcard::from_bytes(&bytes)?;
        let wal = Wal::open_or_create(&dir.join("wal.bin"))?;
        let store = Arc::new(Self {
            dir: dir.to_path_buf(),
            kind: meta.kind,
            dim: meta.vector_size,
            max_vectors_per_point: meta.max_vectors_per_point,
            distance: meta.distance,
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
                WalRecord::UpsertKind { id, data, payload } => {
                    self.apply_upsert(&mut inner, id.clone(), data.clone(), payload.clone())?;
                }
                WalRecord::Delete { point_id } => {
                    if let Some(slot) = inner.reverse.get(point_id).copied() {
                        inner.tombstone_slot(slot);
                    }
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for point_id in point_ids {
                        if let Some(slot) = inner.reverse.get(point_id).copied() {
                            inner.tombstone_slot(slot);
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
                WalRecord::Upsert { .. } | WalRecord::Quantize | WalRecord::DropCollection => {
                    return Err(VectorSearchError::CorruptData(format!(
                        "unexpected WAL record in kind collection {}",
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
        data: VectorData,
        payload: Option<String>,
    ) -> Result<()> {
        let payload: Option<Payload> = payload
            .as_ref()
            .map(|p| serde_json::from_str(p))
            .transpose()?;
        self.validate_data(&data)?;
        // Idempotent overwrite: a replayed or repeated upsert of the same id
        // replaces the stored point in place (the old slot is tombstoned and
        // its postings dropped).
        if let Some(&slot) = inner.reverse.get(&id) {
            inner.tombstone_slot(slot);
        }
        let slot = inner.points.len() as u32;
        if let VectorData::Sparse { indices, values } = &data {
            for (index, value) in indices.iter().zip(values.iter()) {
                inner.inverted.entry(*index).or_default().push((slot, *value));
            }
        }
        inner.reverse.insert(id.clone(), slot);
        inner.points.push(Some(KindPoint {
            id: PointId::from(id),
            data,
            payload,
        }));
        inner.live_count += 1;
        Ok(())
    }

    fn validate_data(&self, data: &VectorData) -> Result<()> {
        if data.kind() != self.kind {
            return Err(VectorSearchError::InvalidConfig(format!(
                "collection kind is {}, payload kind is {}",
                self.kind.as_name(),
                data.kind().as_name()
            )));
        }
        data.validate(self.dim, self.max_vectors_per_point.max(1))
    }
}

/// Persisted per-collection metadata (`kind.bin`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct KindMeta {
    collection: String,
    kind: VectorKind,
    vector_size: usize,
    distance: crate::types::DistanceMetric,
    /// Sub-vector cap for Multi collections; `0` for Sparse.
    max_vectors_per_point: usize,
}

impl KindStore {
    /// Collection kind served by this store.
    pub fn kind(&self) -> VectorKind {
        self.kind
    }

    /// Declared `vector_size` (vocabulary bound / sub-vector dimension).
    pub fn vector_size(&self) -> usize {
        self.dim
    }

    /// Declared distance metric.
    pub fn distance(&self) -> crate::types::DistanceMetric {
        self.distance
    }

    /// Per-point sub-vector cap for Multi collections (`0` for Sparse).
    pub fn max_vectors_per_point(&self) -> usize {
        self.max_vectors_per_point
    }

    /// Directory backing this collection.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Engine-level storage tuning (unused today; kept for symmetry with the
    /// dense store's checkpoint schedule).
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
        for slot in inner.points.iter().enumerate().filter_map(|(i, p)| p.is_some().then_some(i)) {
            let point = inner.points[slot].as_ref().expect("live point");
            if filter::matches(filter, &point.id, point.payload.as_ref())? {
                count += 1;
            }
        }
        Ok(count)
    }

    /// WAL-backed upsert of a batch of points.
    pub fn upsert_batch(&self, points: &[VectorPoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let mut records = Vec::with_capacity(points.len());
        for point in points {
            records.push(kind_upsert_record(point)?);
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
    /// the engine's coordinated-transaction path, which receives the
    /// records pre-split per collection.
    pub fn apply_records(&self, records: &[WalRecord]) -> Result<()> {
        let mut inner = self.inner.lock();
        for record in records {
            match record {
                WalRecord::UpsertKind { id, data, payload } => {
                    self.apply_upsert(&mut inner, id.clone(), data.clone(), payload.clone())?;
                }
                WalRecord::Delete { point_id } => {
                    if let Some(slot) = inner.reverse.get(point_id).copied() {
                        inner.tombstone_slot(slot);
                    }
                }
                _ => {
                    return Err(VectorSearchError::Internal(
                        "kind store transaction received an unsupported record".to_string(),
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
                WalRecord::UpsertKind { id, data, payload } => {
                    self.apply_upsert(&mut inner, id, data, payload)?;
                }
                WalRecord::Delete { point_id } => {
                    if let Some(slot) = inner.reverse.get(&point_id).copied() {
                        inner.tombstone_slot(slot);
                    }
                }
                WalRecord::DeleteBatch { point_ids } => {
                    for point_id in point_ids {
                        if let Some(slot) = inner.reverse.get(&point_id).copied() {
                            inner.tombstone_slot(slot);
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
                WalRecord::Upsert { .. } | WalRecord::Compact | WalRecord::Quantize | WalRecord::DropCollection => {
                    return Err(VectorSearchError::Internal(
                        "kind store commit received an unsupported record".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Exact-scan top-k search over the collection.
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let data = query.data.as_ref().ok_or_else(|| {
            VectorSearchError::InvalidConfig(
                "kind collection queries must carry a non-dense payload".to_string(),
            )
        })?;
        if !query.vector.is_empty() {
            return Err(VectorSearchError::InvalidConfig(
                "kind collection queries must not also carry a dense vector".to_string(),
            ));
        }
        if query.quantization.is_some() || query.nprobe.is_some() {
            return Err(VectorSearchError::InvalidConfig(
                "quantization and nprobe controls only apply to dense collections".to_string(),
            ));
        }
        self.validate_data(data)?;

        let inner = self.inner.lock();
        let limit = query.effective_limit();
        let offset = query.offset.unwrap_or(0);
        if limit == 0 {
            return Ok(Vec::new());
        }

        // Candidate set: inverted postings for sparse, every live slot for
        // multi (there is no coarse structure to consult in the latter case).
        let candidates: Vec<u32> = match data {
            VectorData::Sparse { indices, .. } => inner.sparse_candidates(indices),
            _ => (0..inner.points.len() as u32)
                .filter(|slot| inner.points[*slot as usize].is_some())
                .collect(),
        };

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
            let score = score_pair(data, &point.data, self.distance);
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

        let with_vector = query.with_vector.unwrap_or(false);
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
            .map(|result| {
                if with_vector {
                    // The stored data payload is returned through the point's
                    // `data` field; SearchResult carries dense vectors only,
                    // so non-dense results stay in the store.
                    result
                } else {
                    result
                }
            })
            .collect())
    }

    /// Fetch a point by id.
    pub fn get(&self, point_id: &PointId) -> Result<Option<VectorPoint>> {
        let inner = self.inner.lock();
        let Some(&slot) = inner.reverse.get(&point_id.to_string()) else {
            return Ok(None);
        };
        Ok(inner.points[slot as usize]
            .as_ref()
            .map(|point| VectorPoint {
                id: point.id.clone(),
                vector: Vec::new(),
                data: Some(point.data.clone()),
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
            if with_vector {
                let data = point.data.clone();
                let _ = data;
            }
            points.push(VectorPoint {
                id: point.id.clone(),
                vector: Vec::new(),
                data: with_vector.then(|| point.data.clone()),
                payload: with_payload.then(|| point.payload.clone()).flatten(),
            });
            cursor = Some(point.id.clone());
            if points.len() >= limit {
                break;
            }
        }
        Ok((points, cursor))
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
            if !filter::matches(filter, &point.id, point.payload.as_ref())? {
                continue;
            }
            points.push(VectorPoint {
                id: point.id.clone(),
                vector: Vec::new(),
                data: with_vector.then(|| point.data.clone()),
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
            .ok_or_else(|| VectorSearchError::Internal(format!(
                "point {point_id} not found"
            )))
    }
}

/// Build the WAL upsert record for a non-dense point.
pub fn kind_upsert_record(point: &VectorPoint) -> Result<WalRecord> {
    let data = point
        .data
        .as_ref()
        .ok_or_else(|| VectorSearchError::InvalidConfig(
            "kind collection upserts must carry a non-dense payload".to_string(),
        ))?;
    if !point.vector.is_empty() {
        return Err(VectorSearchError::InvalidConfig(
            "kind collection upserts must not also carry a dense vector".to_string(),
        ));
    }
    Ok(WalRecord::UpsertKind {
        id: point.id.to_string(),
        data: data.clone(),
        payload: point
            .payload
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?,
    })
}

/// Score a (query, stored) payload pair under the collection metric.
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
        (VectorData::Multi(q), VectorData::Multi(d)) => max_sim(q, d, metric),
        _ => 0.0,
    }
}

/// Sparse similarity: merge two strictly-increasing index lists.
///
/// Cosine/Dot accumulate the weighted dot product over shared terms;
/// Euclid/Manhattan accumulate squared/absolute gaps over the union and
/// normalize to the higher-is-better contract.
fn sparse_score(
    q_idx: &[u32],
    q_val: &[f32],
    d_idx: &[u32],
    d_val: &[f32],
    metric: crate::types::DistanceMetric,
) -> f32 {
    use crate::types::DistanceMetric::{Cosine, Dot, Euclid, Manhattan};
    let mut qi = 0usize;
    let mut di = 0usize;
    let mut acc = 0f32;
    while qi < q_idx.len() && di < d_idx.len() {
        match q_idx[qi].cmp(&d_idx[di]) {
            std::cmp::Ordering::Less => {
                if matches!(metric, Euclid | Manhattan) {
                    acc += match metric {
                        Euclid => q_val[qi] * q_val[qi],
                        _ => q_val[qi].abs(),
                    };
                }
                qi += 1;
            }
            std::cmp::Ordering::Greater => {
                if matches!(metric, Euclid | Manhattan) {
                    acc += match metric {
                        Euclid => d_val[di] * d_val[di],
                        _ => d_val[di].abs(),
                    };
                }
                di += 1;
            }
            std::cmp::Ordering::Equal => {
                let gap = q_val[qi] - d_val[di];
                acc += match metric {
                    Cosine | Dot => q_val[qi] * d_val[di],
                    Euclid => gap * gap,
                    Manhattan => gap.abs(),
                };
                qi += 1;
                di += 1;
            }
        }
    }
    for (i, value) in q_val.iter().enumerate().skip(qi) {
        if matches!(metric, Euclid | Manhattan) {
            acc += match metric {
                Euclid => value * value,
                _ => value.abs(),
            };
        }
        let _ = i;
    }
    for value in d_val.iter().skip(di) {
        if matches!(metric, Euclid | Manhattan) {
            acc += match metric {
                Euclid => value * value,
                _ => value.abs(),
            };
        }
    }
    match metric {
        Euclid | Manhattan => 1.0 / (1.0 + acc),
        Cosine | Dot => acc,
    }
}

/// Exact MaxSim: for every query sub-vector take the best-matching stored
/// sub-vector and sum the per-query bests. Inner dense operations reuse the
/// shared SIMD kernel dispatch.
fn max_sim(query: &[Vec<f32>], stored: &[Vec<f32>], metric: crate::types::DistanceMetric) -> f32 {
    let similarity = |q: &[f32], d: &[f32]| -> f32 {
        let distance = crate::distance::distance(metric, q, d);
        match metric {
            crate::types::DistanceMetric::Cosine => 1.0 - distance,
            crate::types::DistanceMetric::Dot => -distance,
            crate::types::DistanceMetric::Euclid | crate::types::DistanceMetric::Manhattan => {
                1.0 / (1.0 + distance)
            }
        }
    };
    let mut total = 0f32;
    for q in query {
        let Some(best) = stored
            .iter()
            .map(|d| similarity(q, d))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        else {
            return 0.0;
        };
        total += best;
    }
    total
}

