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

    /// Split a filter into an acceleratable pre-filter mask and the must
    /// conditions left for post-filtering.
    ///
    /// Planning semantics:
    /// - every `must` condition whose field carries an index able to resolve
    ///   it contributes to the AND-ed `pre_mask`;
    /// - conditions without such an index land in `remaining_must`;
    /// - `must_not`, `should`, `min_should` are preserved verbatim inside
    ///   `remaining_must` handling (they stay post-filters), while any
    ///   accelerated `must` still narrows candidates soundly because the
    ///   post-filter re-evaluates everything that was not accelerated.
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
        let Some(must) = filter.must.as_ref() else {
            return plan;
        };
        let mut acc: Option<BitVec> = None;
        for cond in must {
            let Some(m) = self.indexes.get(&cond.field) else {
                plan.remaining_must.push(cond.clone());
                continue;
            };
            let Some(mut cond_mask) = m.index.bits(&cond.condition) else {
                plan.remaining_must.push(cond.clone());
                continue;
            };
            if cond_mask.len() != capacity {
                cond_mask.resize(capacity, false);
            }
            plan.indexes_used
                .push(format!("{}({})", cond.field, m.index.kind()));
            acc = Some(match acc {
                Some(prev) => prev & cond_mask,
                None => cond_mask,
            });
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
