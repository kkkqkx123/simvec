//! Portable SIMD distance kernels (evaluation branch).
//!
//! The original plan envisioned `std::simd` (`core::simd::Simd<f32, N>`) as
//! the portable abstraction, but `portable_simd` remains an unstable library
//! feature (`#86656`) on stable Rust (1.97.1 in CI), so the `std::simd` path
//! does not compile on stable. Until it stabilizes (or the `wide` crate is
//! adopted) this module delegates to `naive` unconditionally, regardless of
//! the `simd_portable` feature: the feature only decides whether
//! `Kernel::Portable` exists in `kernel.rs` and takes part in dispatch
//! ranking. The delegation is exact (error 0 within `1e-4`) and the
//! `Kernel::Portable` dispatch remains usable for differential bench wiring.
//!
//! Future activation (when `portable_simd` stabilizes or via the `wide`
//! crate):
//! ```ignore
//! use std::simd::Simd;
//! const LANES: usize = 8;
//! let mut acc = Simd::<f32, LANES>::splat(0.0);
//! for chunk in a.chunks_exact(LANES) {
//!     let av = Simd::from_slice(chunk);
//!     let bv = Simd::from_slice(&b[...]);
//!     acc += (av - bv) * (av - bv);
//! }
//! let mut sum = acc.reduce_sum();
//! ```
//! or `wide::f32x8` as a stable alternative. The `bench/vector_scan_bench`
//! comparison is gated on this module's `distance` API so swapping the body
//! later does not touch call sites.

use crate::types::DistanceMetric;
use crate::distance::naive;

/// Delegates to naive; replace with `std::simd` or `wide` when the
/// toolchain allows it.
pub fn distance_l2(a: &[f32], b: &[f32]) -> f32 {
    naive::distance_l2(a, b)
}

/// Delegates to naive; replace with `std::simd` or `wide` when the
/// toolchain allows it.
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    naive::inner_product(a, b)
}

/// Delegates to naive; replace with `std::simd` or `wide` when the
/// toolchain allows it.
pub fn distance_cosine(a: &[f32], b: &[f32]) -> f32 {
    naive::distance_cosine(a, b)
}

/// Delegates to naive; replace with `std::simd` or `wide` when the
/// toolchain allows it.
pub fn distance_l1(a: &[f32], b: &[f32]) -> f32 {
    naive::distance_l1(a, b)
}

/// Delegates to naive; replace with `std::simd` or `wide` when the
/// toolchain allows it.
pub fn distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    naive::distance(metric, a, b)
}
