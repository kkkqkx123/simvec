//! SIMD kernel selection.
//!
//! Mirror of tantivy's per-instruction-set dispatch
//! (`crates/tantivy/bitpacker/src/filter_vec/mod.rs`): an enum of kernel
//! variants (cfg-gated per architecture), a preferred-order list per
//! architecture, a single cached best-available selection, and a tiny
//! dispatch. Call sites go through [`distance`](super::distance) and never
//! change when a new architecture or instruction set is added (adding one is
//! one entry in the [`define_kernels!`] invocation below).
//!
//! Runtime dispatch model:
//! - Baseline compile is `x86-64` / `aarch64` generic (`.cargo/config.toml`
//!   no longer forces `x86-64-v3`). Every kernel's `is_available` is checked
//!   at startup and `best_available` picks the highest-ranked available one.
//! - `x86_64` ranking: `Avx2+FMA > Avx512 (f+dq+vl) > Portable? > Naive`.
//!   AVX-512 only wins past ~1024 dims (<5% at 1536, slower at 384/768 from
//!   masking and frequency costs), so AVX2 is the default and AVX-512 stays
//!   available for explicit selection and kernel-vs-kernel benchmarks.
//! - `aarch64` ranking: `Neon > Portable? > Naive`.
//! - `Portable` is `std::simd` (8-wide) behind the `simd_portable` feature
//!   and is always available when enabled (it degrades to scalar autovec if
//!   no SIMD target is present). It is evaluated against hand-written kernels
//!   within `1e-4` in bench before becoming a rated tier.
//! - Release binaries for specific hardware may still be built with
//!   `RUSTFLAGS="-C target-cpu=native"` for extra autovectorization; the
//!   runtime dispatch remains the correctness gate.
//!
//! All cfg gating is declared ONCE per kernel in the `define_kernels!`
//! table below; the enum, availability probe, dispatch, `Display`, and the
//! preference-ordered `IMPLS` table are generated from it, so a variant can
//! never drift out of `IMPLS` and array lengths are derived, not hand-counted.

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

/// Generates the [`Kernel`] enum, its availability probe, its dispatch,
/// `Display`, and the preference-ordered `IMPLS` table from a single
/// declaration list. Each entry declares its own `cfg` condition exactly
/// once; every generated item reuses it, keeping the five previously
/// hand-maintained cfg sites in sync by construction.
macro_rules! define_kernels {
    (
        $(
            $(#[$vmeta:meta])*
            $variant:ident {
                when = $cfg:meta,
                name = $display:literal,
                available = $avail:expr,
                call = $call:expr $(,)?
            }
        )+
    ) => {
        /// A distance kernel implementation.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Kernel {
            $(
                $(#[$vmeta])*
                #[cfg($cfg)]
                $variant,
            )+
        }

        impl Kernel {
            /// Whether this kernel can run on the current CPU.
            pub fn is_available(self) -> bool {
                match self {
                    $(
                        #[cfg($cfg)]
                        Kernel::$variant => $avail,
                    )+
                }
            }

            /// Internal distance between two vectors (smaller = nearer).
            ///
            /// # Safety contract
            ///
            /// For SIMD variants this must only be called after
            /// [`Kernel::is_available`] returned `true` for the variant
            /// (see `selected()`); the method itself does not re-check.
            #[inline]
            pub fn distance(self, metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
                match self {
                    $(
                        #[cfg($cfg)]
                        Kernel::$variant => ($call)(metric, a, b),
                    )+
                }
            }
        }

        impl fmt::Display for Kernel {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $(
                        #[cfg($cfg)]
                        Kernel::$variant => write!(f, $display),
                    )+
                }
            }
        }

        /// All kernels compiled in this binary (in preference order).
        pub fn all_kernels() -> Vec<Kernel> {
            IMPLS.to_vec()
        }
    };
}

define_kernels! {
    /// AVX2 + FMA kernels (x86-64 only).
    Avx2 {
        when = target_arch = "x86_64",
        name = "avx2",
        available = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"),
        call = |metric, a, b| unsafe { avx2::distance(metric, a, b) },
    }
    /// AVX-512F+DQ+VL kernels (x86-64 only). The gate requires `avx512f`,
    /// `avx512dq` and `avx512vl` because `horizontal_sum512` uses
    /// `_mm512_extractf32x8_ps` (a 256-bit EVEX extract, AVX-512DQ+VL);
    /// AVX-512F-only CPUs (early Skylake-X f-only SKUs) do not guarantee it.
    /// `avx512bw/vnni/bf16` are supersets and also satisfy this gate but are
    /// not separately ranked (the register width is the same 16 x f32 ZMM).
    Avx512 {
        when = target_arch = "x86_64",
        name = "avx512",
        available = {
            // extractf32x8_ps (used by horizontal_sum512) needs
            // AVX-512DQ + VL on top of AVX-512F; gating all three avoids
            // SIGILL on AVX-512F-only CPUs.
            std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512dq")
                && std::arch::is_x86_feature_detected!("avx512vl")
        },
        call = |metric, a, b| unsafe { avx512::distance(metric, a, b) },
    }
    /// NEON kernels (aarch64 only). NEON is mandatory on ARMv8 baseline so
    /// this is always available on aarch64, but we keep the runtime check
    /// for symmetry and future SVE gating.
    Neon {
        when = target_arch = "aarch64",
        name = "neon",
        available = std::arch::is_aarch64_feature_detected!("neon"),
        call = |metric, a, b| unsafe { neon::distance(metric, a, b) },
    }
    /// Portable SIMD via `std::simd` (feature-gated, any arch).
    /// Currently delegates to `naive` on stable Rust; placeholder for
    /// `portable_simd` stabilization or `wide` crate adoption.
    Portable {
        when = feature = "simd_portable",
        name = "portable",
        available = true,
        call = |metric, a, b| portable::distance(metric, a, b),
    }
    /// Scalar baseline; always available, always correct.
    Naive {
        when = all(),
        name = "naive",
        available = true,
        call = |metric, a, b| naive::distance(metric, a, b),
    }
}

/// Preference-ordered kernel table, generated in declaration order of
/// `define_kernels!` (Avx2 > Avx512 > Neon > Portable? > Naive; per-arch
/// cfgs drop the inapplicable entries).
const IMPLS: &[Kernel] = &[
    #[cfg(target_arch = "x86_64")]
    Kernel::Avx2,
    #[cfg(target_arch = "x86_64")]
    Kernel::Avx512,
    #[cfg(target_arch = "aarch64")]
    Kernel::Neon,
    #[cfg(feature = "simd_portable")]
    Kernel::Portable,
    Kernel::Naive,
];

fn best_available() -> Kernel {
    IMPLS
        .iter()
        .copied()
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
