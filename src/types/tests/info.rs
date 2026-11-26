use crate::types::*;

#[test]
fn test_payload_schema_type_as_str() {
    assert_eq!(PayloadSchemaType::Keyword.as_str(), "keyword");
    assert_eq!(PayloadSchemaType::Integer.as_str(), "integer");
    assert_eq!(PayloadSchemaType::Float.as_str(), "float");
    assert_eq!(PayloadSchemaType::Text.as_str(), "text");
    assert_eq!(PayloadSchemaType::Bool.as_str(), "bool");
    assert_eq!(PayloadSchemaType::Geo.as_str(), "geo");
    assert_eq!(PayloadSchemaType::Datetime.as_str(), "datetime");
}

#[test]
fn test_health_status_healthy() {
    let h = HealthStatus::healthy("test-engine", "1.0");
    assert!(h.is_healthy);
    assert_eq!(h.engine_name, "test-engine");
    assert_eq!(h.engine_version, "1.0");
    assert!(h.message.is_none());
}

#[test]
fn test_health_status_unhealthy() {
    let h = HealthStatus::unhealthy("test-engine", "1.0", "not ready");
    assert!(!h.is_healthy);
    assert_eq!(h.message, Some("not ready".to_string()));
}

#[test]
fn test_collection_status_debug() {
    assert_eq!(format!("{:?}", CollectionStatus::Green), "Green");
    assert_eq!(format!("{:?}", CollectionStatus::Yellow), "Yellow");
    assert_eq!(format!("{:?}", CollectionStatus::Red), "Red");
    assert_eq!(format!("{:?}", CollectionStatus::Grey), "Grey");
}

#[test]
fn test_upsert_result() {
    let r = UpsertResult {
        operation_id: Some(123),
        status: UpsertStatus::Completed,
    };
    assert_eq!(r.operation_id, Some(123));
    assert_eq!(r.status, UpsertStatus::Completed);
}

#[test]
fn test_delete_result() {
    let r = DeleteResult {
        operation_id: None,
        deleted_count: 5,
    };
    assert!(r.operation_id.is_none());
    assert_eq!(r.deleted_count, 5);
}

#[test]
fn test_upsert_status_debug() {
    assert_eq!(format!("{:?}", UpsertStatus::Completed), "Completed");
    assert_eq!(format!("{:?}", UpsertStatus::Acknowledged), "Acknowledged");
}
