//! Equality payload index for one field.
//!
//! Maintains a value → slot-bitmap posting list per normalized key plus a
//! reverse view (`slot → keys`) so updates only touch what changed. Masks
//! are conservative with respect to the *current* payload state: every
//! returned bit belongs to a slot whose indexed value satisfies the
//! condition, so combining per-field masks never widens the result.

use std::collections::HashMap;

use bitvec::prelude::*;

use super::super::payload_key::Key;
use crate::types::ConditionType;

/// Equal-value index over one payload field (keyword / boolean / numeric
/// stringified equality and `MatchAny` containment).
#[derive(Debug)]
pub(crate) struct MapIndex {
    postings: HashMap<Key, BitVec>,
    slot_entries: HashMap<u32, Vec<Key>>,
    capacity: usize,
}

impl MapIndex {
    pub(crate) fn new(_field: &str, capacity: usize) -> Self {
        Self {
            postings: HashMap::new(),
            slot_entries: HashMap::new(),
            capacity: capacity.max(1),
        }
    }

    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
    }

    /// Register a slot's current value. Arrays register each scalar element,
    /// matching `MatchAny` containment semantics. Repeated registration
    /// replaces the previous entry.
    pub(crate) fn register_slot(&mut self, slot: u32, value: Option<&serde_json::Value>) {
        self.unregister_slot(slot);
        let Some(value) = value else {
            return;
        };
        let mut keys = Self::keys_for(value);
        keys.sort_unstable();
        keys.dedup();
        if keys.is_empty() {
            return;
        }
        for key in &keys {
            let bv = self
                .postings
                .entry(key.clone())
                .or_insert_with(|| bitvec![0; self.capacity]);
            if (slot as usize) >= bv.len() {
                bv.resize(self.capacity, false);
            }
            bv.set(slot as usize, true);
        }
        self.slot_entries.insert(slot, keys);
    }

    fn unregister_slot(&mut self, slot: u32) {
        if let Some(keys) = self.slot_entries.remove(&slot) {
            for key in keys {
                if let Some(bv) = self.postings.get_mut(&key) {
                    if (slot as usize) < bv.len() {
                        bv.set(slot as usize, false);
                    }
                }
            }
        }
    }

    /// Mask of slots satisfying `condition`, or `None` when this index kind
    /// cannot resolve it. An empty mask is legitimate: no live slot holds a
    /// value matching the condition.
    pub(crate) fn bits(&self, condition: &ConditionType) -> Option<BitVec> {
        let keys: Vec<Key> = match condition {
            ConditionType::Match { value } => Key::for_match(value),
            ConditionType::MatchAny { values } => values.iter().flat_map(Key::for_value).collect(),
            _ => return None,
        };
        if keys.is_empty() {
            return None;
        }
        let mut mask = BitVec::repeat(false, self.capacity);
        for key in &keys {
            if let Some(bv) = self.postings.get(key) {
                mask |= bv.as_bitslice();
            }
        }
        Some(mask)
    }

    fn keys_for(value: &serde_json::Value) -> Vec<Key> {
        match value {
            serde_json::Value::Array(items) => items.iter().flat_map(Key::for_value).collect(),
            v => Key::for_value(v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn new_index(capacity: usize) -> MapIndex {
        MapIndex::new("color", capacity)
    }

    #[test]
    fn register_and_query_equality() {
        let mut idx = new_index(10);
        idx.register_slot(0, Some(&json!("red")));
        idx.register_slot(1, Some(&json!("blue")));
        idx.register_slot(2, Some(&json!("red")));

        let mask = idx
            .bits(&ConditionType::Match {
                value: "red".to_string(),
            })
            .expect("indexed");
        assert!(mask[0]);
        assert!(!mask[1]);
        assert!(mask[2]);

        // Numeric-string normalization: `42` numeric matches `"42"` query.
        let mut nidx = MapIndex::new("size", 4);
        nidx.register_slot(1, Some(&json!(42)));
        let m = nidx
            .bits(&ConditionType::Match {
                value: "42".to_string(),
            })
            .expect("indexed");
        assert!(m[1]);

        // Never-seen values yield an empty mask (correct empty candidate set).
        let missing = nidx
            .bits(&ConditionType::Match {
                value: "999".to_string(),
            })
            .expect("acceleratable");
        assert!(missing.not_any());
    }

    #[test]
    fn reregistration_replaces_entries() {
        let mut idx = new_index(4);
        idx.register_slot(0, Some(&json!("old")));
        idx.register_slot(0, Some(&json!("new")));
        assert!(
            !idx.bits(&ConditionType::Match {
                value: "old".to_string()
            })
            .unwrap()[0]
        );
        assert!(
            idx.bits(&ConditionType::Match {
                value: "new".to_string()
            })
            .unwrap()[0]
        );
        // Unregister clears everything.
        idx.register_slot(0, None);
        assert!(idx
            .bits(&ConditionType::Match {
                value: "new".to_string()
            })
            .unwrap()
            .not_any());
    }

    #[test]
    fn arrays_register_elements() {
        let mut idx = new_index(6);
        idx.register_slot(3, Some(&json!(["a", "b"])));
        for want in ["a", "b"] {
            assert!(
                idx.bits(&ConditionType::Match {
                    value: want.to_string()
                })
                .unwrap()[3]
            );
        }
        let any = idx
            .bits(&ConditionType::MatchAny {
                values: vec![json!("b"), json!("c")],
            })
            .unwrap();
        assert!(any[3]);
    }

    #[test]
    fn unsupported_conditions_return_none() {
        let mut idx = new_index(4);
        idx.register_slot(0, Some(&json!("x")));
        assert!(idx
            .bits(&ConditionType::Range(Default::default()))
            .is_none());
        assert!(idx
            .bits(&ConditionType::Contains { value: "x".into() })
            .is_none());
    }
}
