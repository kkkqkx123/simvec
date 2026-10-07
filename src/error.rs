//! Vector search engine error types.

use crate::types::DistanceMetric;

/// Errors produced by the local vector search engine.
#[derive(Debug, thiserror::Error)]
pub enum VectorSearchError {
    #[error("collection not found: {0}")]
    CollectionNotFound(String),
    #[error("collection already exists: {0}")]
    CollectionAlreadyExists(String),
    #[error("collection incomplete (missing {file}): {dir}", dir = dir.display(), file = file)]
    CollectionIncomplete {
        dir: std::path::PathBuf,
        file: String,
    },
    #[error("invalid collection name: {0}")]
    InvalidCollectionName(String),
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid vector dimension: expected {expected}, got {actual}")]
    InvalidVectorDimension { expected: usize, actual: usize },
    #[error("invalid point id: {0}")]
    InvalidPointId(String),
    #[error("non-finite vector element at index {0}")]
    NonFiniteElement(usize),
    #[error("metric not supported by local engine: {0:?}")]
    UnsupportedMetric(DistanceMetric),
    #[error("filter error: {0}")]
    Filter(String),
    #[error("corrupt data: {0}")]
    CorruptData(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] postcard::Error),
    #[error("json serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<VectorSearchError> for VectorEngineError {
    fn from(err: VectorSearchError) -> Self {
        VectorEngineError::Local(err.to_string())
    }
}

/// Unified error type for the `VectorEngine` trait. Wraps both local
/// (`VectorSearchError`) and remote (client) errors under one type so that
/// the trait object can return a single error without depending on any
/// specific backend crate.
#[derive(Debug, thiserror::Error)]
pub enum VectorEngineError {
    #[error("local vector engine error: {0}")]
    Local(String),
    #[error("remote vector engine error: {0}")]
    Remote(String),
    #[error("vector engine internal error: {0}")]
    Internal(String),
    #[error("operation not supported: {0}")]
    NotSupported(String),
}

/// Result type alias for the vector search engine.
pub type Result<T> = std::result::Result<T, VectorSearchError>;

/// Result type alias for the unified `VectorEngine` trait.
pub type EngineResult<T> = std::result::Result<T, VectorEngineError>;
