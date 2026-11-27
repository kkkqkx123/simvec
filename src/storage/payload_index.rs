//! Per-field payload indexes.
//!
//! The manager owns one [`FieldIndex`] per indexed field. Supported
//! acceleration:
//! - `MapIndex`   — equality (`Match` / `MatchAny`) over keyword, boolean
//!   and string-shaped fields;
//! - `NumericIndex` — equality plus range (`gt/gte/lt/lte`) over numeric
//!   fields via a sorted value array with binary search.
//!
//! Index definitions persist as `payload_indexes.json` in the collection
//! directory; index contents are derived structures rebuilt from the payload
//! storage on open (and after compaction), so they never need their own WAL.
//!
//! Construction is synchronous: creating an index populates it under the
//! store write lock before any reader can plan against it, which preserves
//! the conservative pre-filter contract (a planned mask never excludes a
//! slot the full filter would accept).

mod geo_index;
mod map_index;
mod numeric_index;
mod shape_index;
mod text_index;

use std::collections::BTreeMap;
use std::path::Path;

use bitvec::prelude::*;
use serde::{Deserialize, Serialize};

pub(crate) use geo_index::GeoIndex;
pub(crate) use map_index::MapIndex;
pub(crate) use numeric_index::NumericIndex;
pub(crate) use shape_index::{Shape, ShapeIndex};
pub(crate) use text_index::{TextExport, TextIndex};

use crate::error::{Result, VectorSearchError};
use crate::types::{ConditionType, FilterCondition, Payload, PayloadSchemaType, VectorFilter};

/// A persisted payload index definition (`payload_indexes.json`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct PayloadIndexDef {
    pub field: String,
    pub schema: PayloadSchemaType,
}

/// Concrete per-field index implementation.
#[derive(Debug, Clone)]
pub(crate) enum FieldIndex {
    Map(MapIndex),
    Numeric(NumericIndex),
    Geo(GeoIndex),
    Text(TextIndex),
}

impl FieldIndex {
    /// Build an empty index of the kind implied by `schema`.
    pub(crate) fn new(field: &str, schema: PayloadSchemaType, capacity: usize) -> Self {
        match schema {
            PayloadSchemaType::Integer | PayloadSchemaType::Float => {
                Self::Numeric(NumericIndex::new(field, capacity))
            }
            PayloadSchemaType::Geo => Self::Geo(GeoIndex::new(capacity)),
            PayloadSchemaType::Text => Self::Text(TextIndex::new(capacity)),
            PayloadSchemaType::Keyword | PayloadSchemaType::Bool | PayloadSchemaType::Datetime => {
                Self::Map(MapIndex::new(field, capacity))
            }
        }
    }

    pub(crate) fn resize(&mut self, capacity: usize) {
        match self {
            Self::Map(i) => i.resize(capacity),
            Self::Numeric(i) => i.resize(capacity),
            Self::Geo(i) => i.resize(capacity),
            Self::Text(i) => i.resize(capacity),
        }
    }

    /// Register a slot's current field value (`None` unregisters).
    pub(crate) fn register_slot(&mut self, slot: u32, value: Option<&serde_json::Value>) {
        match self {
            Self::Map(i) => i.register_slot(slot, value),
            Self::Numeric(i) => i.register_slot(slot, value),
            Self::Geo(i) => i.register_slot(slot, value),
            Self::Text(i) => i.register_slot(slot, value),
        }
    }

    /// Mask of slots satisfying `condition`, or `None` when the condition
    /// cannot be accelerated by this index kind.
    pub(crate) fn bits(&self, condition: &ConditionType) -> Option<BitVec> {
        match self {
            Self::Map(i) => i.bits(condition),
            Self::Numeric(i) => i.bits(condition),
            Self::Geo(i) => i.bits(condition),
            Self::Text(i) => i.bits(condition),
        }
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Map(_) => "map",
            Self::Numeric(_) => "numeric",
            Self::Geo(_) => "geo",
            Self::Text(_) => "text",
        }
    }

    fn export(&self) -> PersistedFieldIndex {
        match self {
            Self::Map(index) => {
                let mut postings: Vec<_> = index
                    .postings()
                    .iter()
                    .map(|(key, bits)| (key.clone(), pack_bits(bits)))
                    .collect();
                postings.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                PersistedFieldIndex::Map(postings)
            }
            Self::Numeric(index) => PersistedFieldIndex::Numeric {
                values: index.values().to_vec(),
                slots: index.slots().to_vec(),
            },
            Self::Geo(index) => PersistedFieldIndex::Geo(index.export()),
            Self::Text(index) => PersistedFieldIndex::Text(index.export()),
        }
    }

    fn restore(&mut self, data: PersistedFieldIndex, capacity: usize) -> bool {
        match (self, data) {
            (Self::Map(index), PersistedFieldIndex::Map(postings)) => {
                index.restore(
                    postings
                        .into_iter()
                        .map(|(key, words)| (key, unpack_bits(&words, capacity)))
                        .collect(),
                    capacity,
                );
                true
            }
            (
                Self::Numeric(index),
                PersistedFieldIndex::Numeric { values, slots },
            ) if values.len() == slots.len() => {
                index.restore(values, slots, capacity);
                true
            }
            (Self::Geo(index), PersistedFieldIndex::Geo(points)) => {
                index.restore(points);
                index.resize(capacity);
                true
            }
            (Self::Text(index), PersistedFieldIndex::Text(export)) => {
                index.restore(export);
                index.resize(capacity);
                true
            }
            _ => false,
        }
    }
}

/// A managed index: the requested schema plus its concrete implementation.
#[derive(Debug, Clone)]
struct ManagedIndex {
    schema: PayloadSchemaType,
    index: FieldIndex,
    shape: ShapeIndex,
}

/// Checkpointed contents of every declared field index.
///
/// The envelope carries the generation and its lsn so an index file written
/// before a manifest advance is discarded instead of describing a view that
/// was never published. A mismatch only loses acceleration: open rebuilds
/// from payload bytes and WAL replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PersistedIndexes {
    pub version: u32,
    pub generation: u64,
    pub lsn: u64,
    pub capacity: usize,
    pub fields: Vec<PersistedField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PersistedField {
    pub field: String,
    pub schema: PayloadSchemaType,
    pub index: PersistedFieldIndex,
    pub shape: Vec<(u32, Shape)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum PersistedFieldIndex {
    Map(Vec<(super::payload_key::Key, Vec<u64>)>),
    Numeric { values: Vec<f64>, slots: Vec<u32> },
    Geo(Vec<(u32, crate::types::GeoPoint)>),
    Text(TextExport),
}

const INDEX_FILE: &str = "field_indexes.bin";
const PERSISTED_VERSION: u32 = 1;

fn pack_bits(bits: &BitVec) -> Vec<u64> {
    let mut words = vec![0u64; bits.len().div_ceil(64)];
    for (i, bit) in bits.iter().by_vals().enumerate() {
        if bit {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    words
}

fn unpack_bits(words: &[u64], capacity: usize) -> BitVec {
    let mut bits = bitvec![0; capacity.max(1)];
    for (word, &bits64) in words.iter().enumerate() {
        for bit in 0..64 {
            let slot = word * 64 + bit;
            if slot >= bits.len() {
                return bits;
            }
            if bits64 & (1u64 << bit) != 0 {
                bits.set(slot, true);
            }
        }
    }
    bits
}

/// One filter condition that stays a post-filter after planning.
///
/// The search path reassembles a reduced [`VectorFilter`] from these plus
/// the untouched `must_not` / `should` / `min_should` clauses.
pub(crate) struct FilterPlan {
    /// Candidate mask from accelerated `must` conditions; `None` when no
    /// condition could be accelerated.
    pub pre_mask: Option<BitVec>,
    /// Must conditions not covered by any index.
    pub remaining_must: Vec<FilterCondition>,
    /// Accelerated conditions, formatted `<field>(<kind>)` for explain /
    /// metrics.
    pub indexes_used: Vec<String>,
}

/// Manager for every payload index declared on one collection.
#[derive(Debug, Default, Clone)]
pub(crate) struct PayloadIndexManager {
    indexes: BTreeMap<String, ManagedIndex>,
    capacity: usize,
}

impl PayloadIndexManager {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    /// All persisted definitions, ordered by field name.
    pub(crate) fn defs(&self) -> Vec<PayloadIndexDef> {
        self.indexes
            .iter()
            .map(|(field, m)| PayloadIndexDef {
                field: field.clone(),
                schema: m.schema,
            })
            .collect()
    }

    /// Declare a new index (empty). Fails when the field is already indexed.
    pub(crate) fn declare(
        &mut self,
        field: &str,
        schema: PayloadSchemaType,
        capacity: usize,
    ) -> Result<()> {
        if self.indexes.contains_key(field) {
            return Err(VectorSearchError::InvalidConfig(format!(
                "payload index already exists on field '{field}'"
            )));
        }
        self.capacity = self.capacity.max(capacity);
        self.indexes.insert(
            field.to_string(),
            ManagedIndex {
                schema,
                index: FieldIndex::new(field, schema, capacity),
                shape: ShapeIndex::new(capacity),
            },
        );
        Ok(())
    }

    /// Drop the index on `field`. Returns whether it existed.
    pub(crate) fn delete(&mut self, field: &str) -> bool {
        self.indexes.remove(field).is_some()
    }

    /// Adopt a new slot capacity (masks are rebuilt lazily; only stored
    /// lengths move).
    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = self.capacity.max(capacity);
        for m in self.indexes.values_mut() {
            m.index.resize(capacity);
            m.shape.resize(capacity);
        }
    }

    /// Re-register a slot across all indexes based on its current payload.
    pub(crate) fn register_slot(&mut self, slot: u32, payload: Option<&Payload>) {
        for (field, m) in self.indexes.iter_mut() {
            let value = payload.and_then(|p| p.get(field));
            m.index.register_slot(slot, value);
            m.shape.register_slot(slot, value);
        }
    }

    /// Remove every registration of a slot across all indexes.
    pub(crate) fn unregister_slot(&mut self, slot: u32) {
        for m in self.indexes.values_mut() {
            m.index.register_slot(slot, None);
            m.shape.register_slot(slot, None);
        }
    }

    /// Rebuild all indexes from scratch — used on open and compaction commit
    /// where slots are remapped wholesale.
    pub(crate) fn rebuild(
        &mut self,
        capacity: usize,
        mut payload: impl FnMut(u32) -> Option<Payload>,
        slots: impl Iterator<Item = u32>,
    ) {
        for m in self.indexes.values_mut() {
            m.index.resize(capacity);
            m.shape.resize(capacity);
        }
        for slot in slots {
            let p = payload(slot);
            self.register_slot(slot, p.as_ref());
        }
    }

    /// Resolve one condition against its declared index, sized to the live
    /// slot capacity. `None` means the field or predicate cannot narrow the
    /// candidate set and must stay a post-filter.
    fn condition_mask(&self, cond: &FilterCondition, capacity: usize) -> Option<BitVec> {
        let m = self.indexes.get(&cond.field)?;
        // Shape predicates are available for every schema, including fields
        // whose typed index cannot answer the condition itself.
        if let Some(mask) = m.shape.bits(&cond.condition) {
            let mut mask = mask;
            if mask.len() != capacity {
                mask.resize(capacity, false);
            }
            return Some(mask);
        }
        let mut mask = m.index.bits(&cond.condition)?;
        if mask.len() != capacity {
            mask.resize(capacity, false);
        }
        Some(mask)
    }

    /// Complement of an exact condition mask. Index masks are exact over the
    /// registered slots, so a slot outside `mask` really does satisfy the
    /// negated predicate.
    fn invert_mask(mask: &BitVec, capacity: usize) -> BitVec {
        let mut inverted = BitVec::repeat(true, capacity);
        for (mut dst, src) in inverted.iter_mut().zip(mask.iter()) {
            if *src {
                *dst = false;
            }
        }
        inverted
    }

    /// Mask for a group where at least `min_count` conditions must match.
    /// The group is only accelerated when every condition has an index; an
    /// unindexed clause could otherwise satisfy the count outside the mask
    /// and a matching slot would be lost.
    fn should_mask(
        &self,
        conditions: &[FilterCondition],
        min_count: usize,
        capacity: usize,
    ) -> Option<BitVec> {
        if conditions.is_empty() {
            return None;
        }
        if min_count == 0 {
            return Some(BitVec::repeat(true, capacity));
        }
        if min_count > conditions.len() {
            return Some(BitVec::repeat(false, capacity));
        }
        let mut masks = Vec::with_capacity(conditions.len());
        for cond in conditions {
            masks.push(self.condition_mask(cond, capacity)?);
        }
        if min_count == 1 {
            let mut acc = masks.remove(0);
            for mask in masks {
                acc |= mask;
            }
            return Some(acc);
        }
        let mut counts = vec![0u32; capacity];
        for mask in masks {
            for (slot, matched) in mask.iter().enumerate() {
                if *matched {
                    counts[slot] = counts[slot].saturating_add(1);
                }
            }
        }
        let mut acc = BitVec::repeat(false, capacity);
        for (slot, count) in counts.iter().enumerate() {
            if *count as usize >= min_count {
                acc.set(slot, true);
            }
        }
        Some(acc)
    }

    /// Split a filter into an acceleratable pre-filter mask and the must
    /// conditions left for post-filtering.
    ///
    /// Every group narrows the same candidate mask with set algebra:
    /// `must` ANDs indexed condition masks, `must_not` ANDs their complements,
    /// `should` ORs them, and `min_should` keeps slots that satisfy at least
    /// `min_count` indexed conditions. A group is skipped when any of its
    /// clauses has no usable index, so the mask can never exclude a matching
    /// slot; the full filter still runs on every candidate.
    ///
    /// Returns `pre_mask = None` when nothing was accelerated, meaning the
    /// whole filter runs post-filter exactly like before this manager
    /// existed.
    pub(crate) fn plan_filter(&self, filter: &VectorFilter, capacity: usize) -> FilterPlan {
        let mut plan = FilterPlan {
            pre_mask: None,
            remaining_must: Vec::new(),
            indexes_used: Vec::new(),
        };
        let mut acc: Option<BitVec> = None;
        if let Some(must) = filter.must.as_ref() {
            for cond in must {
                let Some(m) = self.indexes.get(&cond.field) else {
                    plan.remaining_must.push(cond.clone());
                    continue;
                };
                let Some(cond_mask) = self.condition_mask(cond, capacity) else {
                    plan.remaining_must.push(cond.clone());
                    continue;
                };
                plan.indexes_used
                    .push(format!("{}({})", cond.field, m.index.kind()));
                acc = Some(match acc {
                    Some(prev) => prev & cond_mask,
                    None => cond_mask,
                });
            }
        }
        if let Some(must_not) = filter.must_not.as_ref() {
            for cond in must_not {
                let Some(cond_mask) = self.condition_mask(cond, capacity) else {
                    continue;
                };
                let kind = self
                    .indexes
                    .get(&cond.field)
                    .map(|m| m.index.kind())
                    .unwrap_or("map");
                plan.indexes_used
                    .push(format!("{}({}:must_not)", cond.field, kind));
                let inverted = Self::invert_mask(&cond_mask, capacity);
                acc = Some(match acc {
                    Some(prev) => prev & inverted,
                    None => inverted,
                });
            }
        }
        if let Some(should) = filter.should.as_ref() {
            if let Some(union) = self.should_mask(should, 1, capacity) {
                plan.indexes_used.push("should".to_string());
                acc = Some(match acc {
                    Some(prev) => prev & union,
                    None => union,
                });
            }
        }
        if let Some(min_should) = filter.min_should.as_ref() {
            if let Some(counted) =
                self.should_mask(&min_should.conditions, min_should.min_count, capacity)
            {
                plan.indexes_used.push("min_should".to_string());
                acc = Some(match acc {
                    Some(prev) => prev & counted,
                    None => counted,
                });
            }
        }
        plan.pre_mask = acc;
        plan
    }

    /// Serialize the in-memory posting structures for a generation checkpoint.
    pub(crate) fn export(&self, generation: u64, lsn: u64) -> PersistedIndexes {
        PersistedIndexes {
            version: PERSISTED_VERSION,
            generation,
            lsn,
            capacity: self.capacity,
            fields: self
                .indexes
                .iter()
                .map(|(field, m)| PersistedField {
                    field: field.clone(),
                    schema: m.schema,
                    index: m.index.export(),
                    shape: m.shape.export(),
                })
                .collect(),
        }
    }

    /// Adopt a checkpointed index file when it describes exactly this
    /// generation, lsn and definition set. Any mismatch leaves the manager
    /// untouched so open can rebuild from payloads and WAL replay.
    pub(crate) fn restore(&mut self, persisted: PersistedIndexes, generation: u64, lsn: u64) -> bool {
        if persisted.version != PERSISTED_VERSION
            || persisted.generation != generation
            || persisted.lsn != lsn
            || persisted.fields.len() != self.indexes.len()
        {
            return false;
        }
        let mut restored = BTreeMap::new();
        for field in persisted.fields {
            let Some(current) = self.indexes.get(&field.field) else {
                return false;
            };
            if current.schema != field.schema {
                return false;
            }
            let mut index = FieldIndex::new(&field.field, field.schema, persisted.capacity);
            if !index.restore(field.index, persisted.capacity) {
                return false;
            }
            let mut shape = ShapeIndex::new(persisted.capacity);
            shape.restore(field.shape, persisted.capacity);
            restored.insert(
                field.field,
                ManagedIndex {
                    schema: field.schema,
                    index,
                    shape,
                },
            );
        }
        if restored.keys().ne(self.indexes.keys()) {
            return false;
        }
        self.indexes = restored;
        self.capacity = self.capacity.max(persisted.capacity);
        true
    }

    /// Persist index contents into the published generation directory.
    pub(crate) fn save(&self, dir: &Path, generation: u64, lsn: u64) -> Result<()> {
        let data = self.export(generation, lsn);
        crate::storage::snapshot::write(&dir.join(INDEX_FILE), &data)
    }

    /// Read a checkpointed index file, if one was published for this view.
    pub(crate) fn load(dir: &Path, generation: u64, lsn: u64) -> Option<PersistedIndexes> {
        match crate::storage::snapshot::read::<PersistedIndexes>(&dir.join(INDEX_FILE)) {
            Ok(Some(data)) if data.generation == generation && data.lsn == lsn => Some(data),
            _ => None,
        }
    }

    /// Load persisted definitions from a collection directory. A missing or
    /// unreadable file means "no definitions" (the file is a pure marker).
    pub(crate) fn load_defs(dir: &Path) -> Vec<PayloadIndexDef> {
        let path = dir.join("payload_indexes.json");
        let Ok(text) = std::fs::read_to_string(path) else {
            return Vec::new();
        };
        serde_json::from_str(&text).unwrap_or_default()
    }

    /// Persist current definitions atomically (tmp file + rename).
    pub(crate) fn save_defs(&self, dir: &Path) -> Result<()> {
        let defs = self.defs();
        let json = serde_json::to_string_pretty(&defs)?;
        let tmp = dir.join("payload_indexes.json.tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(tmp, dir.join("payload_indexes.json"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        GeoPoint, GeoRadius, MinShouldCondition, RangeCondition, ValuesCountCondition,
    };
    use serde_json::json;

    fn manager() -> PayloadIndexManager {
        let mut m = PayloadIndexManager::new();
        m.declare("tag", PayloadSchemaType::Keyword, 4).unwrap();
        m.declare("size", PayloadSchemaType::Integer, 4).unwrap();
        for (slot, tag, size) in [(0, "red", 1), (1, "blue", 2), (2, "red", 2)] {
            let mut payload = Payload::new();
            payload.insert("tag".to_string(), json!(tag));
            payload.insert("size".to_string(), json!(size));
            m.register_slot(slot, Some(&payload));
        }
        m
    }

    #[test]
    fn must_not_excludes_indexed_matches() {
        let m = manager();
        let filter = VectorFilter::new().must_not(FilterCondition::match_value("tag", "red"));
        let mask = m.plan_filter(&filter, 4).pre_mask.expect("accelerated");
        assert!(!mask[0] && mask[1] && !mask[2] && mask[3]);
    }

    #[test]
    fn should_unions_indexed_conditions() {
        let m = manager();
        let mut filter = VectorFilter::new();
        filter = filter.should(FilterCondition::match_value("tag", "red"));
        filter = filter.should(FilterCondition::match_value("tag", "blue"));
        let mask = m.plan_filter(&filter, 4).pre_mask.expect("accelerated");
        assert!(mask[0] && mask[1] && mask[2] && !mask[3]);
    }

    #[test]
    fn min_should_keeps_slots_with_enough_matches() {
        let m = manager();
        let filter = VectorFilter {
            must: None,
            must_not: None,
            should: None,
            min_should: Some(MinShouldCondition {
                conditions: vec![
                    FilterCondition::match_value("tag", "red"),
                    FilterCondition::match_value("size", "2"),
                ],
                min_count: 2,
            }),
        };
        let mask = m.plan_filter(&filter, 4).pre_mask.expect("accelerated");
        assert!(!mask[0] && !mask[1] && mask[2] && !mask[3]);
    }

    #[test]
    fn unindexed_should_clause_disables_narrowing() {
        let m = manager();
        let mut filter = VectorFilter::new();
        filter = filter.should(FilterCondition::match_value("tag", "red"));
        filter = filter.should(FilterCondition::match_value("ghost", "x"));
        assert!(m.plan_filter(&filter, 4).pre_mask.is_none());
    }

    #[test]
    fn checkpointed_postings_roundtrip() {
        let mut m = manager();
        m.register_slot(
            3,
            Some(&Payload::from([
                ("tag".to_string(), json!("green")),
                ("size".to_string(), json!(7)),
            ])),
        );
        let saved = m.export(11, 21);
        let mut opened = PayloadIndexManager::new();
        opened.declare("tag", PayloadSchemaType::Keyword, 4).unwrap();
        opened.declare("size", PayloadSchemaType::Integer, 4).unwrap();
        assert!(opened.restore(saved.clone(), 11, 21));
        let filter = VectorFilter::new()
            .must(FilterCondition::match_value("tag", "red"))
            .must(FilterCondition::range("size", RangeCondition::new().gte(2.0)));
        assert_eq!(
            m.plan_filter(&filter, 4).pre_mask,
            opened.plan_filter(&filter, 4).pre_mask
        );

        // A file from another generation / lsn must not be adopted.
        let mut stale = PayloadIndexManager::new();
        stale.declare("tag", PayloadSchemaType::Keyword, 4).unwrap();
        stale.declare("size", PayloadSchemaType::Integer, 4).unwrap();
        assert!(!stale.restore(saved, 12, 21));
    }

    #[test]
    fn text_geo_and_shape_indexes_answer_mixed_filters() {
        let mut m = PayloadIndexManager::new();
        m.declare("title", PayloadSchemaType::Text, 4).unwrap();
        m.declare("location", PayloadSchemaType::Geo, 4)
            .unwrap();
        m.declare("active", PayloadSchemaType::Bool, 4).unwrap();
        m.declare("tags", PayloadSchemaType::Keyword, 4).unwrap();
        for (slot, title, lat, active, tags) in [
            (0, "hello world", 48.8566, true, vec!["hot"]),
            (1, "rust vector", 45.7640, false, vec!["cold", "new"]),
            (2, "hello rust", 48.8600, true, vec!["hot"]),
        ] {
            let payload = Payload::from([
                ("title".to_string(), json!(title)),
                (
                    "location".to_string(),
                    json!({ "lat": lat, "lon": 2.3522 }),
                ),
                ("active".to_string(), json!(active)),
                ("tags".to_string(), json!(tags)),
            ]);
            m.register_slot(slot, Some(&payload));
        }
        let filter = VectorFilter::new()
            .must(FilterCondition::match_value("title", "hello"))
            .must(FilterCondition::geo_radius(
                "location",
                GeoRadius::new(GeoPoint::new(48.8566, 2.3522), 100.0),
            ))
            .must(FilterCondition::match_value("active", "true"))
            .must(FilterCondition::values_count(
                "tags",
                ValuesCountCondition::new().gte(1),
            ));
        let mask = m.plan_filter(&filter, 4).pre_mask.expect("all indexed");
        assert!(mask[0] && !mask[1] && !mask[2] && !mask[3]);

        let is_null = m
            .plan_filter(
                &VectorFilter::new().must(FilterCondition::is_null("tags")),
                4,
            )
            .pre_mask
            .expect("shape indexed");
        assert!(!is_null[0] && !is_null[1] && !is_null[2] && is_null[3]);
        assert!(m
            .plan_filter(
                &VectorFilter::new().must(FilterCondition::is_null("missing")),
                4,
            )
            .pre_mask
            .is_none());
    }
}
