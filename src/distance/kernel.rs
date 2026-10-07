//! SIMD kernel selection.
//!
//! Mirror of tantivy's per-instruction-set dispatch
//! (`crates/tantivy/bitpacker/src/filter_vec/mod.rs`): an enum of kernel
//! variants (cfg-gated per architecture), a preferred-order list per
//! architecture, a single cached best-available selection, and a tiny
//! dispatch. Call sites go through [`distance`](super::distance) and never
//! change when a new architecture or instruction set is added (adding one is
//! one enum variant + one `is_available` arm + one `distance` arm + one
//! `IMPLS` entry).
//!
//! Runtime dispatch model:
//! - Baseline compile is `x86-64` / `aarch64` generic (`.cargo/config.toml`
//!   no longer forces `x86-64-v3`). Every kernel's `is_available` is checked
//!   at startup and `best_available` picks the highest-ranked available one.
//! - `x86_64` ranking: `Avx512 (avx512f) > Avx2+FMA > Portable? > Naive`.
//! - `aarch64` ranking: `Neon > Portable? > Naive`.
//! - `Portable` is `std::simd` (8-wide) behind the `simd_portable` feature
//!   and is always available when enabled (it degrades to scalar autovec if
//!   no SIMD target is present). It is evaluated against hand-written kernels
//!   within `1e-4` in bench before becoming a rated tier.
//! - Release binaries for specific hardware may still be built with
//!   `RUSTFLAGS="-C target-cpu=native"` for extra autovectorization; the
//!   runtime dispatch remains the correctness gate.

use std::fmt;
#[cfg(test)]
use std::sync::Mutex;
use std::sync::OnceLock;

use crate::types::DistanceMetric;

#[cfg(target_arch = "x86_64")]
use super::avx2;
#[cfg(target_arch = "x86_64")]
use super::avx512;
use super::naive;
#[cfg(target_arch = "aarch64")]
use super::neon;
#[cfg(feature = "simd_portable")]
use super::portable;

/// A distance kernel implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Scalar baseline; always available, always correct.
    Naive,
    /// AVX2 + FMA kernels (x86-64 only).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512F kernels (x86-64 only). Requires `avx512f`; `avx512bw/vl/vnni`
    /// are supersets and also satisfy this gate but are not separately ranked
    /// (the register width is the same 16 x f32 ZMM).
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// NEON kernels (aarch64 only). NEON is mandatory on ARMv8 baseline so
    /// this is always available on aarch64, but we keep the runtime check
    /// for symmetry and future SVE gating.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// Portable SIMD via `std::simd` (feature-gated, any arch).
    /// Currently delegates to `naive` on stable Rust; placeholder for
    /// `portable_simd` stabilization or `wide` crate adoption.
    #[cfg(feature = "simd_portable")]
    Portable,
}

impl Kernel {
    /// Whether this kernel can run on the current CPU.
    pub fn is_available(self) -> bool {
        match self {
            Kernel::Naive => true,
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => std::arch::is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(feature = "simd_portable")]
            Kernel::Portable => true,
        }
    }

    /// Internal distance between two vectors (smaller = nearer).
    #[inline]
    pub fn distance(self, metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Kernel::Naive => naive::distance(metric, a, b),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => unsafe { avx2::distance(metric, a, b) },
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { avx512::distance(metric, a, b) },
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => unsafe { neon::distance(metric, a, b) },
            #[cfg(feature = "simd_portable")]
            Kernel::Portable => portable::distance(metric, a, b),
        }
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kernel::Naive => write!(f, "naive"),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => write!(f, "avx2"),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => write!(f, "avx512"),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => write!(f, "neon"),
            #[cfg(feature = "simd_portable")]
            Kernel::Portable => write!(f, "portable"),
        }
    }
}

// Preferred kernel order per architecture (mirrors tantivy's `IMPLS`).
// The ordering is documented at the top of the file; each cfg expands to
// the appropriate slice length so `best_available` stays trivial.

#[cfg(all(target_arch = "x86_64", feature = "simd_portable"))]
const IMPLS: [Kernel; 4] = [
    Kernel::Avx512,
    Kernel::Avx2,
    Kernel::Portable,
    Kernel::Naive,
];
#[cfg(all(target_arch = "x86_64", not(feature = "simd_portable")))]
const IMPLS: [Kernel; 3] = [Kernel::Avx512, Kernel::Avx2, Kernel::Naive];

#[cfg(all(target_arch = "aarch64", feature = "simd_portable"))]
const IMPLS: [Kernel; 3] = [Kernel::Neon, Kernel::Portable, Kernel::Naive];
#[cfg(all(target_arch = "aarch64", not(feature = "simd_portable")))]
const IMPLS: [Kernel; 2] = [Kernel::Neon, Kernel::Naive];

#[cfg(all(
    not(target_arch = "x86_64"),
    not(target_arch = "aarch64"),
    feature = "simd_portable"
))]
const IMPLS: [Kernel; 2] = [Kernel::Portable, Kernel::Naive];
#[cfg(all(
    not(target_arch = "x86_64"),
    not(target_arch = "aarch64"),
    not(feature = "simd_portable")
))]
const IMPLS: [Kernel; 1] = [Kernel::Naive];

fn best_available() -> Kernel {
    IMPLS
        .into_iter()
        .find(|k| k.is_available())
        .unwrap_or(Kernel::Naive)
}

static SELECTED: OnceLock<Kernel> = OnceLock::new();

/// Test-only override; wins over the cached detection so differential tests
/// can pin a specific kernel deterministically regardless of test order.
#[cfg(test)]
static OVERRIDE: Mutex<Option<Kernel>> = Mutex::new(None);

/// The kernel active for this process (first call detects and caches).
pub fn selected() -> Kernel {
    #[cfg(test)]
    {
        let guard = OVERRIDE.lock().expect("kernel override lock poisoned");
        if let Some(k) = *guard {
            return k;
        }
    }
    *SELECTED.get_or_init(best_available)
}

/// Pin the active kernel for differential debugging; tests only.
#[cfg(test)]
pub fn force_for_test(kernel: Kernel) {
    *OVERRIDE.lock().expect("kernel override lock poisoned") = Some(kernel);
}

/// Clear the test override; subsequent `selected()` calls use the cached
/// `SELECTED` value (or initialize it on first call). Does **not** reset
/// the `OnceLock` cache. Only available in tests.
#[cfg(test)]
pub fn clear_override_for_test() {
    *OVERRIDE.lock().expect("kernel override lock poisoned") = None;
}

/// All kernels compiled in this binary (in preference order).
pub fn all_kernels() -> Vec<Kernel> {
    IMPLS.to_vec()
}
