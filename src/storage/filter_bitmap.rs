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
//!
//! ## Cost control
//!
//! One full-width bitmap per distinct value is what makes this structure
//! unusable on high-cardinality fields: at a million slots every value costs
//! 125 KiB, so ten thousand distinct values would be 1.25 GiB of resident
//! bitmaps — for a *pre-filter* whose only job is to save a scan. Two rules
//! bound it:
//!
//! 1. **Growth is chunked.** A bitmap starts empty and grows in
//!    [`GROWTH_CHUNK_BITS`] steps up to the slot being written, so a value
//!    that only appears in early slots costs kilobytes, not a full-width
//!    vector.
//! 2. **Per-field budget with a safe fallback.** Every field has a bitmap
//!    budget ([`Self::field_budget_bits`]). A field that outgrows it is
//!    dropped from the pre-filter and recorded in `unindexed`; `build_mask`
//!    then answers `None` for any filter touching that field, so the query
//!    falls back to the plain post-filter. Dropping is *sound* only because
//!    `build_mask` refuses the field: a missing bitmap would otherwise read
//!    as "no slot has this value" and silently exclude every match.

use std::collections::{HashMap, HashSet};

use bitvec::prelude::*;

use super::payload_key::{collect_keys, Key};
use crate::types::{ConditionType, Payload, VectorFilter};

/// Bits per growth step of one value's bitmap (512 bytes).
const GROWTH_CHUNK_BITS: usize = 4096;

/// Floor for a field's bitmap budget, so small collections keep every field
/// indexed no matter how the slots are spread.
const FIELD_BUDGET_FLOOR_BYTES: usize = 64 * 1024;

/// A field's budget as a multiple of one full-width bitmap
/// (`capacity / 8` bytes): how many distinct values the field may hold if
/// every one of them were spread across the whole slot range.
const FIELD_BUDGET_FULL_WIDTHS: usize = 32;

#[derive(Debug, Default)]
pub(crate) struct FilterBitmap {
    map: HashMap<(String, Key), BitVec>,
    slot_entries: HashMap<u32, Vec<(String, Key)>>,
    /// Per-field bitmap cost: distinct values and allocated bits.
    fields: HashMap<String, FieldCost>,
    /// Fields that outgrew their budget and are served by post-filtering.
    ///
    /// Monotonic for the lifetime of this instance: the bitmaps were dropped,
    /// so re-enabling the field would need data that no longer exists. A
    /// rebuild (open, compaction) starts from a clean slate.
    unindexed: HashSet<String>,
    capacity: usize,
    field_budget_bits: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct FieldCost {
    values: usize,
    bits: usize,
}

impl FilterBitmap {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            map: HashMap::new(),
            slot_entries: HashMap::new(),
            fields: HashMap::new(),
            unindexed: HashSet::new(),
            capacity,
            field_budget_bits: Self::field_budget_bits(capacity),
        }
    }

    /// Adopt a new slot capacity; existing bit vectors keep their length
    /// (queries only read below `capacity`, and every read is clamped to the
    /// shorter of the two).
    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
        self.field_budget_bits = Self::field_budget_bits(self.capacity);
    }

    /// Bitmap budget for one field: enough for
    /// [`FIELD_BUDGET_FULL_WIDTHS`] values spread over the whole slot range,
    /// never below [`FIELD_BUDGET_FLOOR_BYTES`].
    fn field_budget_bits(capacity: usize) -> usize {
        (capacity * FIELD_BUDGET_FULL_WIDTHS).max(FIELD_BUDGET_FLOOR_BYTES * 8)
    }

    /// Fields currently served by post-filtering because they outgrew their
    /// bitmap budget. Test-only observation hook.
    #[cfg(test)]
    pub(crate) fn unindexed_fields(&self) -> impl Iterator<Item = &str> {
        self.unindexed.iter().map(|f| f.as_str())
    }

    /// Bytes the equality bitmaps currently hold. Test-only.
    #[cfg(test)]
    pub(crate) fn allocated_bytes(&self) -> usize {
        self.fields.values().map(|c| c.bits).sum::<usize>() / 8
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
        let needed = slot as usize + 1;
        let mut registered: Vec<(String, Key)> = Vec::with_capacity(entries.len());
        for entry in entries {
            let (field, key) = entry;
            if self.unindexed.contains(&field) {
                continue;
            }
            let (grew, was_empty) = {
                let bv = self.map.entry((field.clone(), key.clone())).or_default();
                let was_empty = bv.is_empty();
                let grew = grow_bits(bv, needed, self.capacity);
                bv.set(slot as usize, true);
                (grew, was_empty)
            };
            let cost = self.fields.entry(field.clone()).or_default();
            if was_empty {
                cost.values += 1;
            }
            cost.bits += grew;
            let over = cost.bits > self.field_budget_bits;
            if over {
                self.drop_field(&field);
                continue;
            }
            registered.push((field, key));
        }
        if !registered.is_empty() {
            self.slot_entries.insert(slot, registered);
        }
    }

    /// Drop every bitmap of `field` and mark it unindexed.
    ///
    /// The caller (`register_slot`) has just observed that the field
    /// outgrew its budget. Any query on this field now falls back to the
    /// post-filter, which is the only way to stay sound once the data is
    /// gone: a missing bitmap would read as "no slot matches".
    fn drop_field(&mut self, field: &str) {
        self.map.retain(|(f, _), _| f != field);
        self.slot_entries.retain(|_, entries| {
            entries.retain(|(f, _)| f != field);
            !entries.is_empty()
        });
        self.fields.remove(field);
        self.unindexed.insert(field.to_string());
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
            // A field without bitmaps cannot answer "who has this value":
            // an all-zero condition mask would exclude every match, so the
            // whole filter falls back to post-filtering instead.
            if self.unindexed.contains(&condition.field) {
                return None;
            }
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
                    // Bitmaps grow lazily, so one may be shorter than the
                    // slot range; only the prefix that exists can hold bits.
                    let take = bv.len().min(len);
                    cond_mask[..take] |= &bv[..take];
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
        self.fields.clear();
        self.unindexed.clear();
        self.capacity = capacity.max(1);
        self.field_budget_bits = Self::field_budget_bits(self.capacity);
        for slot in slots {
            if let Some(p) = payload(slot) {
                self.register_slot(slot, Some(&p));
            }
        }
    }
}

/// Grow `bits` to cover `needed` slots, in whole chunks, never past
/// `capacity`. Returns the number of bits added.
fn grow_bits(bits: &mut BitVec, needed: usize, capacity: usize) -> usize {
    if bits.len() >= needed {
        return 0;
    }
    let mut target = needed.next_multiple_of(GROWTH_CHUNK_BITS);
    if target > capacity {
        target = capacity.max(needed);
    }
    let before = bits.len();
    bits.resize(target, false);
    bits.len() - before
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

    /// A high-cardinality field must not grow without bound: it is dropped
    /// once it passes its budget, and every filter touching it falls back to
    /// post-filtering instead of answering "nothing matches".
    #[test]
    fn test_high_cardinality_field_is_dropped_not_unbounded() {
        let capacity = 1 << 20; // 1M slots: one full-width bitmap is 128 KiB.
        let mut bm = FilterBitmap::with_capacity(capacity);
        for slot in 0..40_000u32 {
            bm.register_slot(slot, Some(&payload(&[("uid", json!(slot))])));
        }

        assert!(
            bm.unindexed_fields().any(|f| f == "uid"),
            "the field must be dropped once it outgrows its budget"
        );
        assert_eq!(
            bm.allocated_bytes(),
            0,
            "dropping the field must release its bitmaps"
        );
        let filter = VectorFilter::new().must(FilterCondition::match_value("uid", "1"));
        assert!(
            bm.build_mask(&filter).is_none(),
            "a dropped field must fall back to post-filtering, not report an empty mask"
        );
    }

    /// A low-cardinality field keeps working, and its bitmaps stay far below
    /// one full-width vector per value thanks to chunked growth.
    #[test]
    fn test_low_cardinality_field_stays_indexed() {
        let capacity = 1 << 20;
        let mut bm = FilterBitmap::with_capacity(capacity);
        for slot in 0..10_000u32 {
            let color = if slot % 2 == 0 { "red" } else { "blue" };
            bm.register_slot(slot, Some(&payload(&[("color", json!(color))])));
        }

        assert!(bm.unindexed_fields().next().is_none());
        // Two bitmaps covering 10k slots each: ~2.5 KiB, not 2 * 128 KiB.
        assert!(
            bm.allocated_bytes() <= 4096,
            "chunked growth should keep {} bytes small",
            bm.allocated_bytes()
        );

        let mask = bm
            .build_mask(&VectorFilter::new().must(FilterCondition::match_value("color", "red")))
            .unwrap();
        assert_eq!(mask.len(), capacity);
        assert!(mask[0] && !mask[1] && mask[9_998] && !mask[9_999]);
    }

    /// A filter that mixes an indexed field with a dropped one must fall back
    /// as a whole: a partially-resolved conjunction would be wrong.
    #[test]
    fn test_filter_mixing_indexed_and_dropped_fields_falls_back() {
        let capacity = 1 << 20;
        let mut bm = FilterBitmap::with_capacity(capacity);
        for slot in 0..40_000u32 {
            bm.register_slot(
                slot,
                Some(&payload(&[
                    ("uid", json!(slot)),
                    ("color", json!(if slot % 2 == 0 { "red" } else { "blue" })),
                ])),
            );
        }
        assert!(bm.unindexed_fields().any(|f| f == "uid"));

        let mixed = VectorFilter::new()
            .must(FilterCondition::match_value("color", "red"))
            .must(FilterCondition::match_value("uid", "1"));
        assert!(bm.build_mask(&mixed).is_none());

        // The still-indexed field alone keeps working.
        let ok = VectorFilter::new().must(FilterCondition::match_value("color", "red"));
        let mask = bm.build_mask(&ok).unwrap();
        assert!(mask[0] && !mask[1]);
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
