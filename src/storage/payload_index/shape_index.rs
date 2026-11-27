//! Value-shape index shared by every indexed payload field.
//!
//! `IsEmpty`, `IsNull` and `ValuesCount` depend on the JSON shape rather than
//! the declared field type, so all managed fields carry this index alongside
//! their typed posting structure. The masks are exact over registered slots;
//! missing fields are treated the same way as the full filter evaluator.

use std::collections::HashMap;

use bitvec::prelude::*;
use serde::{Deserialize, Serialize};

use crate::filter::eval_uint_range;
use crate::types::{ConditionType, ValuesCountCondition};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Shape {
    pub is_null: bool,
    pub is_empty: bool,
    pub count: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct ShapeIndex {
    values: HashMap<u32, Shape>,
    capacity: usize,
}

impl ShapeIndex {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            values: HashMap::new(),
            capacity: capacity.max(1),
        }
    }

    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
    }

    pub(crate) fn register_slot(&mut self, slot: u32, value: Option<&serde_json::Value>) {
        let Some(value) = value else {
            self.values.remove(&slot);
            return;
        };
        let shape = match value {
            serde_json::Value::Null => Shape {
                is_null: true,
                is_empty: false,
                count: 0,
            },
            serde_json::Value::Array(items) => Shape {
                is_null: false,
                is_empty: items.is_empty(),
                count: items.len() as u32,
            },
            serde_json::Value::Object(map) => Shape {
                is_null: false,
                is_empty: map.is_empty(),
                count: map.len() as u32,
            },
            serde_json::Value::String(text) => Shape {
                is_null: false,
                is_empty: text.is_empty(),
                count: 1,
            },
            _ => Shape {
                is_null: false,
                is_empty: false,
                count: 1,
            },
        };
        self.values.insert(slot, shape);
    }

    pub(crate) fn bits(&self, condition: &ConditionType) -> Option<BitVec> {
        let mut mask = BitVec::repeat(false, self.capacity);
        match condition {
            ConditionType::IsEmpty => {
                for (&slot, shape) in &self.values {
                    if shape.is_empty && (slot as usize) < mask.len() {
                        mask.set(slot as usize, true);
                    }
                }
            }
            ConditionType::IsNull => {
                mask.fill(true);
                for (&slot, shape) in &self.values {
                    if !shape.is_null && (slot as usize) < mask.len() {
                        mask.set(slot as usize, false);
                    }
                }
            }
            ConditionType::ValuesCount(count) => {
                for (&slot, shape) in &self.values {
                    if matches_count(shape.count, count) && (slot as usize) < mask.len() {
                        mask.set(slot as usize, true);
                    }
                }
            }
            _ => return None,
        }
        Some(mask)
    }

    pub(crate) fn export(&self) -> Vec<(u32, Shape)> {
        let mut out: Vec<_> = self.values.iter().map(|(&slot, s)| (slot, *s)).collect();
        out.sort_unstable_by_key(|(slot, _)| *slot);
        out
    }

    pub(crate) fn restore(&mut self, values: Vec<(u32, Shape)>, capacity: usize) {
        self.capacity = capacity.max(1);
        self.values = values.into_iter().collect();
    }
}

fn matches_count(count: u32, range: &ValuesCountCondition) -> bool {
    eval_uint_range(
        count as u64,
        range.gt,
        range.gte,
        range.lt,
        range.lte,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn shape_masks_match_filter_semantics() {
        let mut index = ShapeIndex::new(5);
        index.register_slot(0, Some(&json!([])));
        index.register_slot(1, Some(&json!(null)));
        index.register_slot(2, Some(&json!(["a", "b"])));
        // Slot 3 is absent from the field.

        assert!(index.bits(&ConditionType::IsEmpty).unwrap()[0]);
        assert!(index.bits(&ConditionType::IsNull).unwrap()[1]);
        assert!(index.bits(&ConditionType::IsNull).unwrap()[3]);
        assert!(!index.bits(&ConditionType::IsNull).unwrap()[2]);
        let range = ValuesCountCondition::new().gte(2).lt(3);
        let mask = index.bits(&ConditionType::ValuesCount(range)).unwrap();
        assert!(mask[2] && !mask[0] && !mask[1] && !mask[3]);
    }
}
