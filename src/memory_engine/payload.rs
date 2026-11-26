use std::collections::HashMap;

use crate::error::{Result, VectorSearchError};
use crate::types::{Payload, PayloadSchemaType, PointId};

use super::MemoryEngine;

impl MemoryEngine {
    /// Replace the payload for a single point.
    pub fn set_payload(
        &self,
        collection: &str,
        point_id: &PointId,
        payload: Payload,
    ) -> Result<()> {
        self.with_collection_mut(collection, |c| {
            let point = c
                .points
                .get_mut(point_id)
                .ok_or_else(|| VectorSearchError::InvalidPointId(point_id.to_string()))?;
            point.payload = Some(payload);
            self.bump_revision();
            Ok(())
        })
    }

    /// Set a single payload field, preserving all other keys.
    pub fn set_payload_field(
        &self,
        collection: &str,
        point_id: &PointId,
        key: String,
        value: serde_json::Value,
    ) -> Result<()> {
        self.set_payload_fields(collection, point_id, Payload::from([(key, value)]))
    }

    /// Merge fields into a point payload.
    pub fn set_payload_fields(
        &self,
        collection: &str,
        point_id: &PointId,
        fields: Payload,
    ) -> Result<()> {
        if fields.is_empty() {
            return Ok(());
        }
        self.with_collection_mut(collection, |c| {
            let point = c
                .points
                .get_mut(point_id)
                .ok_or_else(|| VectorSearchError::InvalidPointId(point_id.to_string()))?;
            let payload = point.payload.get_or_insert_with(HashMap::new);
            for (k, v) in fields {
                payload.insert(k, v);
            }
            self.bump_revision();
            Ok(())
        })
    }

    /// Remove keys from a point payload.
    pub fn delete_payload_keys(
        &self,
        collection: &str,
        point_id: &PointId,
        keys: Vec<String>,
    ) -> Result<()> {
        self.with_collection_mut(collection, |c| {
            let point = c
                .points
                .get_mut(point_id)
                .ok_or_else(|| VectorSearchError::InvalidPointId(point_id.to_string()))?;
            if let Some(payload) = point.payload.as_mut() {
                let mut removed = false;
                for k in keys {
                    if payload.remove(&k).is_some() {
                        removed = true;
                    }
                }
                if removed {
                    self.bump_revision();
                }
            }
            Ok(())
        })
    }

    /// Declare a payload field index. Memory form tracks the declaration
    /// only; searches stay exact scan.
    pub fn create_payload_index(
        &self,
        collection: &str,
        field: &str,
        schema: PayloadSchemaType,
    ) -> Result<()> {
        self.with_collection_mut(collection, |c| {
            if !c.payload_indexes.iter().any(|(f, _)| f == field) {
                c.payload_indexes.push((field.to_string(), schema));
                self.bump_revision();
            }
            Ok(())
        })
    }

    /// Drop a payload field index. Returns whether it existed.
    pub fn delete_payload_index(&self, collection: &str, field: &str) -> Result<bool> {
        self.with_collection_mut(collection, |c| {
            let before = c.payload_indexes.len();
            c.payload_indexes.retain(|(f, _)| f != field);
            let removed = c.payload_indexes.len() != before;
            if removed {
                self.bump_revision();
            }
            Ok(removed)
        })
    }

    /// All declared payload indexes.
    pub fn list_payload_indexes(
        &self,
        collection: &str,
    ) -> Result<Vec<(String, PayloadSchemaType)>> {
        self.with_collection(collection, |c| Ok(c.payload_indexes.clone()))
    }
}
