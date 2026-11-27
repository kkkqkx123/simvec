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

mod map_index;
mod numeric_index;

use std::collections::BTreeMap;
use std::path::Path;

use bitvec::prelude::*;

pub(crate) use map_index::MapIndex;
pub(crate) use numeric_index::NumericIndex;

use crate::error::{Result, VectorSearchError};
use crate::types::{ConditionType, FilterCondition, Payload, PayloadSchemaType, VectorFilter};

/// A persisted payload index definition (`payload_indexes.json`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct PayloadIndexDef {
    pub field: String,
    pub schema: PayloadSchemaType,
}

/// Concrete per-field index implementation.
#[derive(Debug)]
pub(crate) enum FieldIndex {
    Map(MapIndex),
    Numeric(NumericIndex),
}

impl FieldIndex {
    /// Build an empty index of the kind implied by `schema`.
    pub(crate) fn new(field: &str, schema: PayloadSchemaType, capacity: usize) -> Self {
        match schema {
            PayloadSchemaType::Integer | PayloadSchemaType::Float => {
                Self::Numeric(NumericIndex::new(field, capacity))
            }
            _ => Self::Map(MapIndex::new(field, capacity)),
        }
    }

    pub(crate) fn resize(&mut self, capacity: usize) {
        match self {
            Self::Map(i) => i.resize(capacity),
            Self::Numeric(i) => i.resize(capacity),
        }
    }

    /// Register a slot's current field value (`None` unregisters).
    pub(crate) fn register_slot(&mut self, slot: u32, value: Option<&serde_json::Value>) {
        match self {
            Self::Map(i) => i.register_slot(slot, value),
            Self::Numeric(i) => i.register_slot(slot, value),
        }
    }

    /// Mask of slots satisfying `condition`, or `None` when the condition
    /// cannot be accelerated by this index kind.
    pub(crate) fn bits(&self, condition: &ConditionType) -> Option<BitVec> {
        match self {
            Self::Map(i) => i.bits(condition),
            Self::Numeric(i) => i.bits(condition),
        }
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Map(_) => "map",
            Self::Numeric(_) => "numeric",
        }
    }
}

/// A managed index: the requested schema plus its concrete implementation.
#[derive(Debug)]
struct ManagedIndex {
    schema: PayloadSchemaType,
    index: FieldIndex,
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
#[derive(Debug, Default)]
pub(crate) struct PayloadIndexManager {
    indexes: BTreeMap<String, ManagedIndex>,
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
        self.indexes.insert(
            field.to_string(),
            ManagedIndex {
                schema,
                index: FieldIndex::new(field, schema, capacity),
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
        for m in self.indexes.values_mut() {
            m.index.resize(capacity);
        }
    }

    /// Re-register a slot across all indexes based on its current payload.
    pub(crate) fn register_slot(&mut self, slot: u32, payload: Option<&Payload>) {
        for (field, m) in self.indexes.iter_mut() {
            let value = payload.and_then(|p| p.get(field));
            m.index.register_slot(slot, value);
        }
    }

    /// Remove every registration of a slot across all indexes.
    pub(crate) fn unregister_slot(&mut self, slot: u32) {
        for m in self.indexes.values_mut() {
            m.index.register_slot(slot, None);
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
    use crate::types::MinShouldCondition;
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
}
