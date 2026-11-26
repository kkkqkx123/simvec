use std::collections::HashMap;

use crate::types::*;

#[test]
fn test_point_id_from_u64() {
    let id: PointId = 42u64.into();
    assert_eq!(id, PointId::Num(42));
}

#[test]
fn test_point_id_from_string_numeric() {
    let id = PointId::from("123");
    assert_eq!(id, PointId::Num(123));
}

#[test]
fn test_point_id_from_string_uuid() {
    let id = PointId::from("uuid-abc");
    assert_eq!(id, PointId::Uuid("uuid-abc".into()));
}

#[test]
fn test_point_id_from_str() {
    let id = PointId::from("456");
    assert_eq!(id, PointId::Num(456));
}

#[test]
fn test_point_id_from_str_uuid() {
    let id = PointId::from("550e8400-e29b-41d4-a716-446655440000");
    assert_eq!(
        id,
        PointId::Uuid("550e8400-e29b-41d4-a716-446655440000".into())
    );
}

#[test]
fn test_point_id_display_num() {
    let id = PointId::Num(42);
    assert_eq!(format!("{}", id), "42");
}

#[test]
fn test_point_id_display_uuid() {
    let id = PointId::Uuid("abc".into());
    assert_eq!(format!("{}", id), "abc");
}

#[test]
fn test_point_id_serialize_deserialize() {
    let id = PointId::Num(42);
    let json = serde_json::to_string(&id).unwrap();
    let deserialized: PointId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, deserialized);
}

#[test]
fn test_point_id_serialize_deserialize_uuid() {
    let id = PointId::Uuid("test-uuid".into());
    let json = serde_json::to_string(&id).unwrap();
    let deserialized: PointId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, deserialized);
}

#[test]
fn test_vector_point_new() {
    let p = VectorPoint::new(42u64, vec![1.0, 2.0, 3.0]);
    assert_eq!(p.id, PointId::Num(42));
    assert_eq!(p.vector, vec![1.0, 2.0, 3.0]);
    assert!(p.payload.is_none());
}

#[test]
fn test_vector_point_with_payload() {
    let mut payload = HashMap::new();
    payload.insert("key".into(), serde_json::json!("val"));
    let p = VectorPoint::new("1", vec![1.0]).with_payload(payload);
    assert!(p.payload.is_some());
}

#[test]
fn test_vector_point_with_payload_kv() {
    let p = VectorPoint::new(1u64, vec![1.0])
        .with_payload_kv("color", serde_json::json!("red"))
        .with_payload_kv("size", serde_json::json!(42));
    let payload = p.payload.expect("payload expected");
    assert_eq!(payload.get("color").and_then(|v| v.as_str()), Some("red"));
    assert_eq!(payload.get("size").and_then(|v| v.as_i64()), Some(42));
}

#[test]
fn test_vector_point_dimension() {
    let p = VectorPoint::new(1u64, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(p.dimension(), 4);
}

#[test]
fn test_vector_point_dimension_empty() {
    let p = VectorPoint::new(1u64, vec![]);
    assert_eq!(p.dimension(), 0);
}

#[test]
fn test_vector_points_new() {
    let points = vec![VectorPoint::new(1u64, vec![1.0])];
    let vp = VectorPoints::new(points);
    assert_eq!(vp.len(), 1);
    assert!(!vp.is_empty());
}

#[test]
fn test_vector_points_empty() {
    let vp = VectorPoints::new(vec![]);
    assert!(vp.is_empty());
    assert_eq!(vp.len(), 0);
}

#[test]
fn test_vector_points_from() {
    let points = vec![VectorPoint::new(1u64, vec![1.0])];
    let vp: VectorPoints = points.into();
    assert_eq!(vp.len(), 1);
}
