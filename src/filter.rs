//! `VectorFilter` evaluation against point payloads (post-filter,
//! Qdrant-compatible semantics).
//!
//! Semantics:
//! - `must`: every condition must match;
//! - `must_not`: no condition may match;
//! - `should`: at least one condition must match (or `min_should` count);
//! - `min_should` overrides `should` when both are present.

use std::cmp::Ordering;

use crate::error::Result;
use crate::types::{
    ConditionType, FilterCondition, MinShouldCondition, Payload, PointId, VectorFilter,
};

/// Evaluate a whole filter for a point.
pub fn matches(filter: &VectorFilter, id: &PointId, payload: Option<&Payload>) -> Result<bool> {
    if let Some(must) = &filter.must {
        for condition in must {
            if !eval_condition(condition, id, payload)? {
                return Ok(false);
            }
        }
    }
    if let Some(must_not) = &filter.must_not {
        for condition in must_not {
            if eval_condition(condition, id, payload)? {
                return Ok(false);
            }
        }
    }
    match (&filter.should, &filter.min_should) {
        (Some(should), Some(min)) => {
            let matched = should
                .iter()
                .filter(|c| eval_condition(c, id, payload).unwrap_or(false))
                .count();
            Ok(matched >= min.min_count)
        }
        (Some(should), None) => {
            for condition in should {
                if eval_condition(condition, id, payload)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        (None, Some(min)) => eval_min_should(min, id, payload),
        (None, None) => Ok(true),
    }
}

fn eval_min_should(
    min: &MinShouldCondition,
    id: &PointId,
    payload: Option<&Payload>,
) -> Result<bool> {
    let matched = min
        .conditions
        .iter()
        .filter(|c| eval_condition(c, id, payload).unwrap_or(false))
        .count();
    Ok(matched >= min.min_count)
}

fn eval_condition(
    condition: &FilterCondition,
    id: &PointId,
    payload: Option<&Payload>,
) -> Result<bool> {
    match &condition.condition {
        ConditionType::Match { value } => {
            Ok(get_field(payload, &condition.field).is_some_and(|v| value_matches_any(v, value)))
        }
        ConditionType::MatchAny { values } => Ok(get_field(payload, &condition.field)
            .is_some_and(|v| values.iter().any(|expected| value_equals_any(v, expected)))),
        ConditionType::Range(range) => {
            Ok(get_field(payload, &condition.field).is_some_and(|v| eval_range(v, range)))
        }
        ConditionType::IsEmpty => {
            Ok(get_field(payload, &condition.field).is_some_and(is_empty_value))
        }
        ConditionType::IsNull => {
            Ok(get_field(payload, &condition.field).is_none_or(|v| v.is_null()))
        }
        ConditionType::HasId { ids } => {
            Ok(ids.iter().any(|candidate| candidate == &id.to_string()))
        }
        ConditionType::Nested { filter } => {
            let Some(Value::Array(elements)) = get_field(payload, &condition.field) else {
                return Ok(false);
            };
            for element in elements {
                let Value::Object(map) = element else {
                    continue;
                };
                let nested_payload: Payload =
                    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                if matches(filter, id, Some(&nested_payload))? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        ConditionType::GeoRadius(geo) => {
            let Some(point) = geo_point(get_field(payload, &condition.field)) else {
                return Ok(false);
            };
            Ok(haversine_meters(point, geo.center) <= geo.radius)
        }
        ConditionType::GeoBoundingBox(bbox) => {
            let Some(point) = geo_point(get_field(payload, &condition.field)) else {
                return Ok(false);
            };
            let in_lat = point.lat <= bbox.top_left.lat && point.lat >= bbox.bottom_right.lat;
            let in_lon = point.lon >= bbox.top_left.lon && point.lon <= bbox.bottom_right.lon;
            Ok(in_lat && in_lon)
        }
        ConditionType::ValuesCount(count) => {
            let Some(v) = get_field(payload, &condition.field) else {
                return Ok(false);
            };
            let n = match v {
                Value::Array(items) => items.len() as u64,
                Value::Object(map) => map.len() as u64,
                Value::Null => 0,
                _ => 1,
            };
            Ok(eval_uint_range(n, count.gt, count.gte, count.lt, count.lte))
        }
        ConditionType::Contains { value } => {
            let Some(v) = get_field(payload, &condition.field) else {
                return Ok(false);
            };
            Ok(match v {
                Value::Array(items) => items.iter().any(|item| value_equals(item, value)),
                Value::String(s) => s.contains(value.as_str()),
                _ => false,
            })
        }
    }
}

fn eval_range(v: &serde_json::Value, range: &crate::types::RangeCondition) -> bool {
    let Some(n) = v.as_f64() else {
        return false;
    };
    eval_f64_range(n, range.gt, range.gte, range.lt, range.lte)
}

fn eval_f64_range(
    n: f64,
    gt: Option<f64>,
    gte: Option<f64>,
    lt: Option<f64>,
    lte: Option<f64>,
) -> bool {
    // NaN must fail the check: `partial_cmp` yields None for NaN, so only a
    // Some(Ordering) match passes.
    if let Some(gt) = gt {
        if n.partial_cmp(&gt) != Some(Ordering::Greater) {
            return false;
        }
    }
    if let Some(gte) = gte {
        if n.partial_cmp(&gte) == Some(Ordering::Less) {
            return false;
        }
    }
    if let Some(lt) = lt {
        if n.partial_cmp(&lt) != Some(Ordering::Less) {
            return false;
        }
    }
    if let Some(lte) = lte {
        if n.partial_cmp(&lte) == Some(Ordering::Greater) {
            return false;
        }
    }
    true
}

fn eval_uint_range(
    n: u64,
    gt: Option<u64>,
    gte: Option<u64>,
    lt: Option<u64>,
    lte: Option<u64>,
) -> bool {
    if let Some(gt) = gt {
        if !(n > gt) {
            return false;
        }
    }
    if let Some(gte) = gte {
        if !(n >= gte) {
            return false;
        }
    }
    if let Some(lt) = lt {
        if !(n < lt) {
            return false;
        }
    }
    if let Some(lte) = lte {
        if !(n <= lte) {
            return false;
        }
    }
    true
}

fn get_field<'a>(payload: Option<&'a Payload>, field: &str) -> Option<&'a serde_json::Value> {
    payload.and_then(|p| p.get(field))
}

/// Qdrant `match` semantics: for array fields any element may match, scalars
/// match by string equality for strings and by stringified value for
/// numbers/bools.
fn value_matches_any(v: &serde_json::Value, expected: &str) -> bool {
    match v {
        Value::Array(items) => items.iter().any(|item| value_equals(item, expected)),
        _ => value_equals(v, expected),
    }
}

fn value_equals_any(v: &serde_json::Value, expected: &serde_json::Value) -> bool {
    match v {
        Value::Array(items) => items.iter().any(|item| json_equals(item, expected)),
        _ => json_equals(v, expected),
    }
}

fn value_equals(v: &serde_json::Value, expected: &str) -> bool {
    match v {
        Value::String(s) => s == expected,
        Value::Number(n) => n.to_string() == expected,
        Value::Bool(b) => b.to_string() == expected,
        _ => false,
    }
}

fn json_equals(v: &serde_json::Value, expected: &serde_json::Value) -> bool {
    v == expected
}

fn is_empty_value(v: &serde_json::Value) -> bool {
    match v {
        Value::Array(items) => items.is_empty(),
        Value::Object(map) => map.is_empty(),
        Value::String(s) => s.is_empty(),
        _ => false,
    }
}

fn geo_point(v: Option<&serde_json::Value>) -> Option<crate::types::GeoPoint> {
    let Value::Object(map) = v? else {
        return None;
    };
    let lat = map.get("lat")?.as_f64()?;
    let lon = map.get("lon")?.as_f64()?;
    Some(crate::types::GeoPoint::new(lat, lon))
}

/// Great-circle distance in meters (haversine, Earth radius 6371.0088 km as
/// used by Qdrant).
fn haversine_meters(a: crate::types::GeoPoint, b: crate::types::GeoPoint) -> f64 {
    const EARTH_RADIUS_M: f64 = 6_371_008.8;
    let d_lat = (b.lat - a.lat).to_radians();
    let d_lon = (b.lon - a.lon).to_radians();
    let sin_lat = (d_lat / 2.0).sin();
    let sin_lon = (d_lon / 2.0).sin();
    let h =
        sin_lat * sin_lat + a.lat.to_radians().cos() * b.lat.to_radians().cos() * sin_lon * sin_lon;
    2.0 * EARTH_RADIUS_M * h.sqrt().asin()
}

use serde_json::Value;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GeoBoundingBox, GeoPoint, GeoRadius, RangeCondition, ValuesCountCondition};
    use serde_json::json;

    fn payload(kv: &[(&str, serde_json::Value)]) -> Payload {
        kv.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    fn id() -> PointId {
        PointId::Num(42)
    }

    fn eval(filter: &VectorFilter, p: &Payload) -> bool {
        matches(filter, &id(), Some(p)).unwrap()
    }

    #[test]
    fn test_match() {
        let p = payload(&[("color", json!("red")), ("size", json!(42))]);
        assert!(eval(
            &VectorFilter::new().must(FilterCondition::match_value("color", "red")),
            &p
        ));
        assert!(!eval(
            &VectorFilter::new().must(FilterCondition::match_value("color", "blue")),
            &p
        ));
        // numeric field matched by stringified value
        assert!(eval(
            &VectorFilter::new().must(FilterCondition::match_value("size", "42")),
            &p
        ));
        // absent field
        assert!(!eval(
            &VectorFilter::new().must(FilterCondition::match_value("nope", "x")),
            &p
        ));
    }

    #[test]
    fn test_match_any() {
        let p = payload(&[("tags", json!(["a", "b"]))]);
        let f = VectorFilter::new().must(FilterCondition::match_any(
            "tags",
            vec![json!("b"), json!("z")],
        ));
        assert!(eval(&f, &p));
        let f = VectorFilter::new().must(FilterCondition::match_any("tags", vec![json!("z")]));
        assert!(!eval(&f, &p));
    }

    #[test]
    fn test_range() {
        let p = payload(&[("price", json!(12.5))]);
        let f = VectorFilter::new().must(FilterCondition::range(
            "price",
            RangeCondition::new().gte(10.0).lt(20.0),
        ));
        assert!(eval(&f, &p));
        let f = VectorFilter::new().must(FilterCondition::range(
            "price",
            RangeCondition::new().gt(12.5),
        ));
        assert!(!eval(&f, &p));
        let f = VectorFilter::new().must(FilterCondition::range(
            "price",
            RangeCondition::new().lte(12.5),
        ));
        assert!(eval(&f, &p));
        // non-numeric field never matches a range
        let p2 = payload(&[("price", json!("expensive"))]);
        assert!(!eval(&f, &p2));
    }

    #[test]
    fn test_is_empty_is_null() {
        let p = payload(&[("tags", json!([])), ("nick", json!("bob"))]);
        assert!(eval(
            &VectorFilter::new().must(FilterCondition::is_empty("tags")),
            &p
        ));
        assert!(!eval(
            &VectorFilter::new().must(FilterCondition::is_empty("nick")),
            &p
        ));
        assert!(eval(
            &VectorFilter::new().must(FilterCondition::is_null("missing")),
            &p
        ));
        assert!(!eval(
            &VectorFilter::new().must(FilterCondition::is_null("nick")),
            &p
        ));
        // null value counts as null
        let p2 = payload(&[("x", Value::Null)]);
        assert!(eval(
            &VectorFilter::new().must(FilterCondition::is_null("x")),
            &p2
        ));
    }

    #[test]
    fn test_has_id() {
        let p = payload(&[]);
        let f = VectorFilter::new().must(FilterCondition::has_id(vec!["7".into(), "42".into()]));
        assert!(eval(&f, &p));
        let f = VectorFilter::new().must(FilterCondition::has_id(vec!["7".into()]));
        assert!(!eval(&f, &p));
    }

    #[test]
    fn test_nested() {
        let p = payload(&[(
            "addresses",
            json!([
                {"city": "paris", "zip": 75001},
                {"city": "lyon", "zip": 69000}
            ]),
        )]);
        let inner = VectorFilter::new().must(FilterCondition::match_value("city", "lyon"));
        let f = VectorFilter::new().must(FilterCondition::new(
            "addresses",
            ConditionType::Nested {
                filter: Box::new(inner),
            },
        ));
        assert!(eval(&f, &p));

        let inner = VectorFilter::new().must(FilterCondition::match_value("city", "berlin"));
        let f = VectorFilter::new().must(FilterCondition::new(
            "addresses",
            ConditionType::Nested {
                filter: Box::new(inner),
            },
        ));
        assert!(!eval(&f, &p));
    }

    #[test]
    fn test_geo_radius() {
        let p = payload(&[("location", json!({"lat": 48.8566, "lon": 2.3522}))]);
        let f = VectorFilter::new().must(FilterCondition::geo_radius(
            "location",
            GeoRadius::new(GeoPoint::new(48.8566, 2.3522), 1000.0),
        ));
        assert!(eval(&f, &p));
        // ~350 km away (paris -> lyon)
        let f = VectorFilter::new().must(FilterCondition::geo_radius(
            "location",
            GeoRadius::new(GeoPoint::new(45.7640, 4.8357), 100_000.0),
        ));
        assert!(!eval(&f, &p));
    }

    #[test]
    fn test_geo_bounding_box() {
        let p = payload(&[("location", json!({"lat": 48.8566, "lon": 2.3522}))]);
        let f = VectorFilter::new().must(FilterCondition::geo_bounding_box(
            "location",
            GeoBoundingBox::new(GeoPoint::new(49.0, 2.0), GeoPoint::new(48.0, 3.0)),
        ));
        assert!(eval(&f, &p));
        let f = VectorFilter::new().must(FilterCondition::geo_bounding_box(
            "location",
            GeoBoundingBox::new(GeoPoint::new(49.0, 2.5), GeoPoint::new(48.0, 3.0)),
        ));
        assert!(!eval(&f, &p));
    }

    #[test]
    fn test_values_count() {
        let p = payload(&[("tags", json!(["a", "b", "c"]))]);
        let f = VectorFilter::new().must(FilterCondition::values_count(
            "tags",
            ValuesCountCondition::new().gte(2).lt(4),
        ));
        assert!(eval(&f, &p));
        let f = VectorFilter::new().must(FilterCondition::values_count(
            "tags",
            ValuesCountCondition::new().gt(3),
        ));
        assert!(!eval(&f, &p));
        // scalar counts as 1
        let p2 = payload(&[("tags", json!("single"))]);
        let f = VectorFilter::new().must(FilterCondition::values_count(
            "tags",
            ValuesCountCondition::new().gte(1),
        ));
        assert!(eval(&f, &p2));
    }

    #[test]
    fn test_contains() {
        let p = payload(&[("tags", json!(["x", "y"])), ("desc", json!("hello world"))]);
        assert!(eval(
            &VectorFilter::new().must(FilterCondition::contains("tags", "y")),
            &p
        ));
        assert!(eval(
            &VectorFilter::new().must(FilterCondition::contains("desc", "lo wo")),
            &p
        ));
        assert!(!eval(
            &VectorFilter::new().must(FilterCondition::contains("tags", "z")),
            &p
        ));
    }

    #[test]
    fn test_combinations() {
        let p = payload(&[
            ("color", json!("red")),
            ("size", json!(42)),
            ("tags", json!(["hot"])),
        ]);
        // must + must_not
        let f = VectorFilter::new()
            .must(FilterCondition::match_value("color", "red"))
            .must_not(FilterCondition::match_value("size", "1"));
        assert!(eval(&f, &p));
        let f = VectorFilter::new()
            .must(FilterCondition::match_value("color", "red"))
            .must_not(FilterCondition::match_value("size", "42"));
        assert!(!eval(&f, &p));

        // should: at least one
        let f = VectorFilter::new()
            .should(FilterCondition::match_value("color", "blue"))
            .should(FilterCondition::match_value("color", "red"));
        assert!(eval(&f, &p));
        let f = VectorFilter::new()
            .should(FilterCondition::match_value("color", "blue"))
            .should(FilterCondition::match_value("color", "green"));
        assert!(!eval(&f, &p));

        // min_should
        let f = VectorFilter {
            must: None,
            must_not: None,
            should: Some(vec![
                FilterCondition::match_value("color", "red"),
                FilterCondition::match_value("size", "42"),
                FilterCondition::match_value("tags", "nope"),
            ]),
            min_should: Some(MinShouldCondition {
                conditions: vec![
                    FilterCondition::match_value("color", "red"),
                    FilterCondition::match_value("size", "42"),
                    FilterCondition::match_value("tags", "nope"),
                ],
                min_count: 2,
            }),
        };
        assert!(eval(&f, &p));

        // empty filter matches everything
        assert!(matches(&VectorFilter::new(), &id(), None).unwrap());
        assert!(matches(&VectorFilter::new(), &id(), Some(&p)).unwrap());
    }

    #[test]
    fn test_empty_payload_none() {
        // Filter with only must_not on absent payload
        let f = VectorFilter::new().must_not(FilterCondition::match_value("color", "red"));
        assert!(matches(&f, &id(), None).unwrap());
        let f = VectorFilter::new().must(FilterCondition::match_value("color", "red"));
        assert!(!matches(&f, &id(), None).unwrap());
    }
}
