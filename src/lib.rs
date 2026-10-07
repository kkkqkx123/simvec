//! Local vector search engine.
//!
//! Exact-scan storage and search. Leaf crate: does not depend on any
//! graphdb crate and does not pull in the qdrant networking stack.

pub mod distance;
pub mod engine;
pub mod error;
pub mod filter;
pub mod filter_cond;
mod index;
pub mod metrics;
pub mod storage;
pub mod types;

pub use engine::{LocalVectorEngine, TxnOp, VectorEngine};
pub use error::VectorEngineError;
pub use error::{EngineResult, Result, VectorSearchError};
pub use metrics::{IndexTier, Metrics, MetricsSnapshot, SearchPath, SearchRetry};
pub use types::*;
