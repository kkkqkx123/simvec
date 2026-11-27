//! Geo-point payload index for radius and bounding-box predicates.
//!
//! The index stores the same `(lat, lon)` values the full filter evaluator
//! reads and reuses its distance/bounds helpers, so a planned mask is exact
//! whenever the field holds a geo object.

use std::collections::HashMap;

use bitvec::prelude::*;

use crate::filter::{geo_point, haversine_meters};
use crate::types::{ConditionType, GeoPoint};

#[derive(Debug, Clone)]
pub(crate) struct GeoIndex {
    points: HashMap<u32, GeoPoint>,
    capacity: usize,
}

impl GeoIndex {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            points: HashMap::new(),
            capacity: capacity.max(1),
        }
    }

    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
    }

    pub(crate) fn register_slot(&mut self, slot: u32, value: Option<&serde_json::Value>) {
        match value.and_then(|value| geo_point(Some(value))) {
            Some(point) => {
                self.points.insert(slot, point);
            }
            None => {
                self.points.remove(&slot);
            }
        }
    }

    pub(crate) fn bits(&self, condition: &ConditionType) -> Option<BitVec> {
        let mut mask = BitVec::repeat(false, self.capacity);
        match condition {
            ConditionType::GeoRadius(radius) => {
                for (&slot, point) in &self.points {
                    if haversine_meters(*point, radius.center) <= radius.radius
                        && (slot as usize) < mask.len()
                    {
                        mask.set(slot as usize, true);
                    }
                }
            }
            ConditionType::GeoBoundingBox(bbox) => {
                for (&slot, point) in &self.points {
                    let in_lat =
                        point.lat <= bbox.top_left.lat && point.lat >= bbox.bottom_right.lat;
                    let in_lon =
                        point.lon >= bbox.top_left.lon && point.lon <= bbox.bottom_right.lon;
                    if in_lat && in_lon && (slot as usize) < mask.len() {
                        mask.set(slot as usize, true);
                    }
                }
            }
            _ => return None,
        }
        Some(mask)
    }

    pub(crate) fn export(&self) -> Vec<(u32, GeoPoint)> {
        let mut out: Vec<_> = self.points.iter().map(|(&slot, p)| (slot, *p)).collect();
        out.sort_unstable_by_key(|(slot, _)| *slot);
        out
    }

    pub(crate) fn restore(&mut self, points: Vec<(u32, GeoPoint)>) {
        self.points = points.into_iter().collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GeoBoundingBox, GeoRadius};
    use serde_json::json;

    #[test]
    fn radius_and_box_masks_are_exact() {
        let mut index = GeoIndex::new(3);
        index.register_slot(0, Some(&json!({"lat": 48.8566, "lon": 2.3522})));
        index.register_slot(1, Some(&json!({"lat": 45.7640, "lon": 4.8357})));
        let paris = GeoPoint::new(48.8566, 2.3522);
        let near = index
            .bits(&ConditionType::GeoRadius(GeoRadius::new(paris, 1000.0)))
            .unwrap();
        assert!(near[0] && !near[1]);
        let boxed = index
            .bits(&ConditionType::GeoBoundingBox(GeoBoundingBox::new(
                GeoPoint::new(49.0, 2.0),
                GeoPoint::new(48.0, 3.0),
            )))
            .unwrap();
        assert!(boxed[0] && !boxed[1]);
    }
}
