//! C ABI surface of the local vector engine.
//!
//! The module is only compiled with the `c_api` feature so default builds
//! keep a plain Rust library. Every entry point returns an integer status
//! code, borrows all inputs and hands newly allocated outputs to the caller
//! for release with `simvec_free_string`. Complex values cross the boundary
//! as JSON text reusing the existing serde shapes.

pub mod collections;
pub mod engine;
pub mod error;
pub mod maintenance;
pub mod read;
pub mod types;
pub mod write;

pub use error::simvec_error_code_t;
pub use types::simvec_engine_t;

#[cfg(test)]
mod tests;
