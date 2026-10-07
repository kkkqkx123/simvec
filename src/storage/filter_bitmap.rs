//! Pre-filter bitmap for equality payload filters.
//!
//! Ann search explores a candidate neighbourhood that can be much larger
//! than the subset matching a selective filter.  `FilterBitmap` maintains,
//! for each `(field, value)` pair seen in any payload, a bit vector over
//! slots so a search can restrict traversal to the union / intersection of
//! matching slots instead of discovering mismatches and discarding them
//! after the fact.
//!
//! Only equality (`Match` / `MatchAny`) conditions can be accelerated:
//! range, geo, nested and other condition types fall back to the plain
//! post-filter in `filter::matches`.  The bitmap is a *conservative*
//! pre-filter — it may admit more slots than the filter would accept
//! (false positives are removed by the post-filter) but never excludes a
//! matching slot, so it cannot change results.
//!
//! Keys are normalized per the filter semantics:
//! - string payload values key on the string itself;
//! - numbers key on their numeric value (`42` and `42.0` share a key) so
//!   `MatchAny` typed comparisons agree; the `Match` string comparison is
//!   served by also keying the query's numeric string form (`"42"` →
//!   number key) while non-integral floats like `42.5` keep a distinct key;
//! - booleans key on the boolean;
//! - null, objects and nested objects are not indexed (they never satisfy
//!   an equality condition).
//!
//! Mutations run under the store write lock; reads of the mask run under
//! the store read lock, so the mask is coherent with the slot snapshots
//! taken in the same critical section.

use std::collections::HashMap;

use bitvec::prelude::*;

use super::payload_key::{collect_keys, Key};
use crate::types::{ConditionType, Payload, VectorFilter};

#[derive(Debug, Default)]
pub(crate) struct FilterBitmap {
    map: HashMap<(String, Key), BitVec>,
    slot_entries: HashMap<u32, Vec<(String, Key)>>,
    capacity: usize,
}

impl FilterBitmap {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            slot_entries: HashMap::new(),
            capacity: capacity.max(1),
        }
    }

    /// Adopt a new slot capacity; existing bit vectors keep their length
    /// (queries only read bits below `capacity`).
    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
    }

    /// Register every indexable `(field, value)` pair of `payload` for
    /// `slot`, replacing whatever the slot was registered under before.
    pub(crate) fn register_slot(&mut self, slot: u32, payload: Option<&Payload>) {
        self.unregister_slot(slot);
        let Some(payload) = payload else {
            return;
        };
        let mut entries: Vec<(String, Key)> = Vec::new();
        for (field, value) in payload {
            collect_keys(field, value, &mut entries);
        }
        for entry in &entries {
            let bv = self
                .map
                .entry(entry.clone())
                .or_insert_with(|| bitvec![0; self.capacity]);
            if (slot as usize) >= bv.len() {
                bv.resize(self.capacity, false);
            }
            bv.set(slot as usize, true);
        }
        if !entries.is_empty() {
            self.slot_entries.insert(slot, entries);
        }
    }

    /// Remove every registration of `slot` (payload overwrite or delete).
    pub(crate) fn unregister_slot(&mut self, slot: u32) {
        if let Some(entries) = self.slot_entries.remove(&slot) {
            for (field, key) in entries {
                if let Some(bv) = self.map.get_mut(&(field, key)) {
                    if (slot as usize) < bv.len() {
                        bv.set(slot as usize, false);
                    }
                }
            }
        }
    }

    /// Build the candidate mask for a filter, or `None` when the filter
    /// cannot be accelerated (any condition more complex than equality).
    ///
    /// The mask is a conjunction over `must` conditions; each condition's
    /// bitmap is the disjunction of its accepted value bitmaps.  The mask
    /// must not exclude a matching slot, so conditions that cannot be
    /// fully resolved fall back to `None` and the search uses plain
    /// post-filtering.
    pub(crate) fn build_mask(&self, filter: &VectorFilter) -> Option<BitVec> {
        let must = filter.must.as_ref()?;
        if filter.must_not.is_some() || filter.should.is_some() || filter.min_should.is_some() {
            return None;
        }
        if must.is_empty() {
            return None;
        }
        let len = self.capacity;
        let mut mask: Option<BitVec> = None;
        for condition in must {
            let keys: Vec<Key> = match &condition.condition {
                ConditionType::Match { value } => Key::for_match(value),
                ConditionType::MatchAny { values } => {
                    values.iter().flat_map(Key::for_value).collect()
                }
                _ => return None,
            };
            if keys.is_empty() {
                return None;
            }
            let mut cond_mask = BitVec::repeat(false, len);
            for key in keys {
                if let Some(bv) = self.map.get(&(condition.field.clone(), key)) {
                    cond_mask |= bv.as_bitslice();
                }
            }
            mask = Some(match mask {
                Some(acc) => acc & cond_mask,
                None => cond_mask,
            });
        }
        mask
    }

    /// Rebuild the index from payload blobs (open path and compaction).
    pub(crate) fn rebuild(
        &mut self,
        capacity: usize,
        mut payload: impl FnMut(u32) -> Option<Payload>,
        slots: impl Iterator<Item = u32>,
    ) {
        self.map.clear();
        self.slot_entries.clear();
        self.capacity = capacity.max(1);
        for slot in slots {
            if let Some(p) = payload(slot) {
                self.register_slot(slot, Some(&p));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FilterCondition;
    use serde_json::json;

    fn payload(kv: &[(&str, serde_json::Value)]) -> Payload {
        kv.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn test_register_and_build_mask() {
        let mut bm = FilterBitmap::with_capacity(10);
        let p0 = payload(&[("color", json!("red"))]);
        let p1 = payload(&[("color", json!("blue"))]);
        let p2 = payload(&[("color", json!("red")), ("type", json!("A"))]);
        let p3 = payload(&[]);
        bm.register_slot(0, Some(&p0));
        bm.register_slot(1, Some(&p1));
        bm.register_slot(2, Some(&p2));
        bm.register_slot(3, Some(&p3));

        let f_red = VectorFilter::new().must(FilterCondition::match_value("color", "red"));
        let mask = bm.build_mask(&f_red).unwrap();
        assert!(mask[0]);
        assert!(!mask[1]);
        assert!(mask[2]);
        assert!(!mask[3]);
    }

    #[test]
    fn test_match_any_mask() {
        let mut bm = FilterBitmap::with_capacity(10);
        bm.register_slot(0, Some(&payload(&[("color", json!("red"))])));
        bm.register_slot(1, Some(&payload(&[("color", json!("blue"))])));
        let f = VectorFilter::new().must(FilterCondition::match_any(
            "color",
            vec![json!("red"), json!("blue")],
        ));
        let mask = bm.build_mask(&f).unwrap();
        assert!(mask[0]);
        assert!(mask[1]);
    }

    #[test]
    fn test_overwrite_updates_mask() {
        let mut bm = FilterBitmap::with_capacity(10);
        bm.register_slot(0, Some(&payload(&[("color", json!("red"))])));
        let mask0 = bm
            .build_mask(&VectorFilter::new().must(FilterCondition::match_value("color", "red")))
            .unwrap();
        assert!(mask0[0]);

        bm.register_slot(0, Some(&payload(&[("color", json!("blue"))])));
        let mask1 = bm
            .build_mask(&VectorFilter::new().must(FilterCondition::match_value("color", "blue")))
            .unwrap();
        assert!(mask1[0]);
        let mask_red = bm
            .build_mask(&VectorFilter::new().must(FilterCondition::match_value("color", "red")))
            .unwrap();
        assert!(!mask_red[0]);
    }

    #[test]
    fn test_delete_clears_mask() {
        let mut bm = FilterBitmap::with_capacity(10);
        bm.register_slot(0, Some(&payload(&[("color", json!("red"))])));
        bm.unregister_slot(0);
        let mask = bm
            .build_mask(&VectorFilter::new().must(FilterCondition::match_value("color", "red")))
            .unwrap();
        assert!(!mask[0]);
    }

    #[test]
    fn test_must_not_falls_back_to_none() {
        let mut bm = FilterBitmap::with_capacity(10);
        bm.register_slot(0, Some(&payload(&[("color", json!("red"))])));
        let f = VectorFilter::new()
            .must(FilterCondition::match_value("color", "red"))
            .must_not(FilterCondition::match_value("color", "red"));
        assert!(bm.build_mask(&f).is_none());
    }

    #[test]
    fn test_non_indexed_condition_falls_back() {
        let mut bm = FilterBitmap::with_capacity(10);
        bm.register_slot(0, Some(&payload(&[("price", json!(10))])));
        let f = VectorFilter::new().must(FilterCondition::range(
            "price",
            crate::types::RangeCondition::new().gt(5.0),
        ));
        assert!(bm.build_mask(&f).is_none());
    }

    #[test]
    fn test_rebuild_from_payloads() {
        let mut bm = FilterBitmap::with_capacity(5);
        bm.register_slot(0, Some(&payload(&[("color", json!("red"))])));
        bm.register_slot(2, Some(&payload(&[("color", json!("blue"))])));
        bm.rebuild(
            5,
            |s| {
                Some(payload(&[(
                    "color",
                    if s == 2 { json!("red") } else { json!("blue") },
                )]))
            },
            0..3,
        );
        let f = VectorFilter::new().must(FilterCondition::match_value("color", "red"));
        let mask = bm.build_mask(&f).unwrap();
        assert!(!mask[0]);
        assert!(mask[2]);
    }

    #[test]
    fn test_numeric_keys_normalized() {
        let mut bm = FilterBitmap::with_capacity(10);
        bm.register_slot(0, Some(&payload(&[("n", json!(42))])));
        bm.register_slot(1, Some(&payload(&[("n", json!(42.0))])));
        bm.register_slot(2, Some(&payload(&[("n", json!(42.5))])));
        let f = VectorFilter::new().must(FilterCondition::match_value("n", "42"));
        let mask = bm.build_mask(&f).unwrap();
        // Both 42 and 42.0 match "42" stringify; 42.5 does not.
        assert!(mask[0]);
        assert!(mask[1]);
        assert!(!mask[2]);
    }
}
