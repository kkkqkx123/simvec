//! Numeric range/equality payload index for one field.
//!
//! Stores `(value, slot)` pairs sorted by value; range scans translate to
//! two binary searches and a slice of slots, equality to an exact binary
//! search. The full filter evaluator compares JSON numbers as `f64`
//! (`filter::eval_range`), so this index uses the same representation and
//! is exact for the same inputs.

use std::collections::HashMap;

use bitvec::prelude::*;

use crate::types::ConditionType;

/// Range index over one numeric payload field.
#[derive(Debug)]
pub(crate) struct NumericIndex {
    /// Ascending values (total order via `f64::total_cmp`; zero normalized).
    sorted: Vec<f64>,
    /// Slot holding the corresponding value in `sorted`.
    slots: Vec<u32>,
    /// slot → current value (reverse view for O(log n) updates).
    reverse: HashMap<u32, f64>,
    capacity: usize,
}

impl NumericIndex {
    pub(crate) fn new(_field: &str, capacity: usize) -> Self {
        Self {
            sorted: Vec::new(),
            slots: Vec::new(),
            reverse: HashMap::new(),
            capacity: capacity.max(1),
        }
    }

    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
    }

    /// Register a slot's current value. Non-finite numbers are skipped —
    /// they can never satisfy a comparison under the filter evaluator's
    /// `f64` semantics either.
    pub(crate) fn register_slot(&mut self, slot: u32, value: Option<&serde_json::Value>) {
        if let Some(old) = self.reverse.get(&slot).copied() {
            self.remove_value(slot, old);
        }
        let Some(v) = value.and_then(|v| v.as_f64()) else {
            // `None` or non-numeric: ensure the slot is gone either way.
            return;
        };
        let Some(v) = normalize(v) else {
            return;
        };
        let pos = self
            .sorted
            .partition_point(|&x| x.total_cmp(&v) == std::cmp::Ordering::Less);
        self.sorted.insert(pos, v);
        self.slots.insert(pos, slot);
        self.reverse.insert(slot, v);
    }

    fn remove_value(&mut self, slot: u32, old: f64) {
        let lo = self
            .sorted
            .partition_point(|&x| x.total_cmp(&old) == std::cmp::Ordering::Less);
        let len = self.sorted.len();
        let mut i = lo;
        while i < len && self.sorted[i] == old {
            if self.slots[i] == slot {
                self.sorted.remove(i);
                self.slots.remove(i);
                break;
            }
            i += 1;
        }
        self.reverse.remove(&slot);
    }

    fn set_bit(&self, mask: &mut BitVec, slot: u32) {
        if (slot as usize) < mask.len() {
            mask.set(slot as usize, true);
        }
    }

    fn eq_mask(&self, target: f64) -> Option<BitVec> {
        let t = normalize(target)?;
        let mut mask = BitVec::repeat(false, self.capacity);
        let pos = self
            .sorted
            .binary_search_by(|probe| probe.total_cmp(&t))
            .ok()?;
        self.set_bit(&mut mask, self.slots[pos]);
        Some(mask)
    }

    pub(crate) fn bits(&self, condition: &ConditionType) -> Option<BitVec> {
        match condition {
            ConditionType::Range(rc) => {
                let lo = if let Some(gte) = rc.gte.and_then(normalize) {
                    self.sorted.partition_point(|&x| x < gte)
                } else if let Some(gt) = rc.gt.and_then(normalize) {
                    self.sorted.partition_point(|&x| x <= gt)
                } else {
                    0
                };
                let hi = if let Some(lte) = rc.lte.and_then(normalize) {
                    self.sorted.partition_point(|&x| x <= lte)
                } else if let Some(lt) = rc.lt.and_then(normalize) {
                    self.sorted.partition_point(|&x| x < lt)
                } else {
                    self.sorted.len()
                };
                let hi = hi.min(self.sorted.len());
                let mut mask = BitVec::repeat(false, self.capacity);
                for &slot in &self.slots[lo..hi] {
                    self.set_bit(&mut mask, slot);
                }
                Some(mask)
            }
            ConditionType::Match { value } => {
                let t = parse_number(value)?;
                let Some(mask) = self.eq_mask(t) else {
                    // A valid number that simply matches nothing still
                    // accelerates the condition with an empty mask.
                    return Some(BitVec::repeat(false, self.capacity));
                };
                Some(mask)
            }
            ConditionType::MatchAny { values } => {
                // Numeric JSON values participate directly; string values
                // participate through their numeric parse (mirrors the
                // stringified match contract).
                let targets: Vec<Option<f64>> = values
                    .iter()
                    .map(|v| match v {
                        serde_json::Value::Number(_) => normalize(v.as_f64()?),
                        serde_json::Value::String(s) => parse_number(s).and_then(normalize),
                        _ => None,
                    })
                    .collect();
                let mut mask = BitVec::repeat(false, self.capacity);
                let mut any = false;
                for t in targets.into_iter().flatten() {
                    any = true;
                    if let Ok(pos) = self.sorted.binary_search_by(|probe| probe.total_cmp(&t)) {
                        self.set_bit(&mut mask, self.slots[pos]);
                    }
                }
                if !any {
                    return None;
                }
                Some(mask)
            }
            _ => None,
        }
    }
}

/// Normalize `-0.0` so equal values share one key under `total_cmp`.
fn normalize(v: f64) -> Option<f64> {
    if !v.is_finite() {
        return None;
    }
    Some(if v == 0.0 { 0.0 } else { v })
}

fn parse_number(value: &str) -> Option<f64> {
    match value.parse::<f64>() {
        Ok(n) if n.is_finite() => Some(n),
        _ => None,
    }
}
