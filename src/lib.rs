//! Local vector search engine.
//!
//! HNSW-first ANN search with exact-scan fallback. Leaf crate: does not depend on any
//! host project crate and does not pull in the qdrant networking stack.

pub mod distance;
#[cfg(not(target_arch = "wasm32"))]
pub mod engine;
pub mod error;
pub mod filter;
pub mod filter_cond;
#[cfg(not(target_arch = "wasm32"))]
mod index;
#[cfg(feature = "wasm")]
pub mod memory_engine;
pub mod metrics;
#[cfg(not(target_arch = "wasm32"))]
pub mod storage;
pub mod types;
#[cfg(feature = "wasm")]
pub mod wasm_api;

#[cfg(not(target_arch = "wasm32"))]
pub use engine::{LocalVectorEngine, TxnOp, VectorEngine};
pub use error::VectorEngineError;
pub use error::{EngineResult, Result, VectorSearchError};
pub use metrics::{IndexTier, Metrics, MetricsSnapshot, SearchPath, SearchRetry};
pub use types::*;

#[cfg(all(feature = "c_api", not(target_arch = "wasm32")))]
pub mod c_api;
