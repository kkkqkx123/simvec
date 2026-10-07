//! NEON distance kernels (aarch64 only).
//!
//! Mirrors `avx2.rs`/`avx512.rs` but with 128-bit NEON registers (4 x f32).
//! Selection via `kernel::best_available`: `Neon > Naive` on `aarch64`.
//! On non-aarch64 hosts this module compiles to a thin wrapper over `naive`
//! so the crate remains buildable everywhere.
//!
//! Each kernel uses dual accumulators (2 x float32x4) processing 8 f32 per
//! iteration to hide the 4-cycle FMA latency. The two accumulators are
//! independent so the CPU can issue back-to-back FMA instructions. A scalar
//! tail handles the remaining 0-7 elements. Prefetch is emitted via inline
//! asm (`prfm pldl1keep`) since `std::arch::aarch64::_prefetch` is unstable.

use crate::types::DistanceMetric;

#[cfg(target_arch = "aarch64")]
mod imp {
    use super::DistanceMetric;
    use std::arch::aarch64::*;

    /// Prefetch `ptr` into L1 cache for read (PLDL1KEEP).
    ///
    /// Inline asm is used because `std::arch::aarch64::_prefetch` requires
    /// the unstable `stdarch_aarch64_prefetch` feature (#117217). The
    /// `prfm pldl1keep` instruction is a pure hint and safe on all ARMv8 CPUs.
    #[inline(always)]
    unsafe fn prefetch_l1(ptr: *const f32) {
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, readonly),
            );
        }
    }

    /// Squared Euclidean distance (NEON, dual-accumulator, 8-wide).
    ///
    /// # Safety
    /// Caller must guarantee `a.len() == b.len()` and NEON is available
    /// (always true on aarch64 baseline).
    #[target_feature(enable = "neon")]
    pub unsafe fn distance_l2(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut i = 0;
            let len = a.len();
            while i + 8 <= len {
                if i + 32 < len {
                    prefetch_l1(a.as_ptr().add(i + 32));
                    prefetch_l1(b.as_ptr().add(i + 32));
                }
                let av0 = vld1q_f32(a.as_ptr().add(i));
                let bv0 = vld1q_f32(b.as_ptr().add(i));
                let av1 = vld1q_f32(a.as_ptr().add(i + 4));
                let bv1 = vld1q_f32(b.as_ptr().add(i + 4));
                let d0 = vsubq_f32(av0, bv0);
                let d1 = vsubq_f32(av1, bv1);
                acc0 = vfmaq_f32(acc0, d0, d0);
                acc1 = vfmaq_f32(acc1, d1, d1);
                i += 8;
            }
            let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
            for j in i..len {
                let d = a[j] - b[j];
                sum += d * d;
            }
            sum
        }
    }

    /// Dot product (NEON, dual-accumulator, 8-wide).
    #[target_feature(enable = "neon")]
    pub unsafe fn inner_product(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut i = 0;
            let len = a.len();
            while i + 8 <= len {
                if i + 32 < len {
                    prefetch_l1(a.as_ptr().add(i + 32));
                    prefetch_l1(b.as_ptr().add(i + 32));
                }
                let av0 = vld1q_f32(a.as_ptr().add(i));
                let bv0 = vld1q_f32(b.as_ptr().add(i));
                let av1 = vld1q_f32(a.as_ptr().add(i + 4));
                let bv1 = vld1q_f32(b.as_ptr().add(i + 4));
                acc0 = vfmaq_f32(acc0, av0, bv0);
                acc1 = vfmaq_f32(acc1, av1, bv1);
                i += 8;
            }
            let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
            for j in i..len {
                sum += a[j] * b[j];
            }
            sum
        }
    }

    /// Cosine distance (NEON, dual-accumulator, 8-wide, single pass).
    #[target_feature(enable = "neon")]
    pub unsafe fn distance_cosine(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let mut acc_dot0 = vdupq_n_f32(0.0);
            let mut acc_na0 = vdupq_n_f32(0.0);
            let mut acc_nb0 = vdupq_n_f32(0.0);
            let mut acc_dot1 = vdupq_n_f32(0.0);
            let mut acc_na1 = vdupq_n_f32(0.0);
            let mut acc_nb1 = vdupq_n_f32(0.0);
            let mut i = 0;
            let len = a.len();
            while i + 8 <= len {
                if i + 32 < len {
                    prefetch_l1(a.as_ptr().add(i + 32));
                    prefetch_l1(b.as_ptr().add(i + 32));
                }
                let av0 = vld1q_f32(a.as_ptr().add(i));
                let bv0 = vld1q_f32(b.as_ptr().add(i));
                let av1 = vld1q_f32(a.as_ptr().add(i + 4));
                let bv1 = vld1q_f32(b.as_ptr().add(i + 4));
                acc_dot0 = vfmaq_f32(acc_dot0, av0, bv0);
                acc_na0 = vfmaq_f32(acc_na0, av0, av0);
                acc_nb0 = vfmaq_f32(acc_nb0, bv0, bv0);
                acc_dot1 = vfmaq_f32(acc_dot1, av1, bv1);
                acc_na1 = vfmaq_f32(acc_na1, av1, av1);
                acc_nb1 = vfmaq_f32(acc_nb1, bv1, bv1);
                i += 8;
            }
            let mut dot = vaddvq_f32(vaddq_f32(acc_dot0, acc_dot1));
            let mut norm_a = vaddvq_f32(vaddq_f32(acc_na0, acc_na1));
            let mut norm_b = vaddvq_f32(vaddq_f32(acc_nb0, acc_nb1));
            for j in i..len {
                dot += a[j] * b[j];
                norm_a += a[j] * a[j];
                norm_b += b[j] * b[j];
            }
            let denom = (norm_a * norm_b).sqrt();
            if denom == 0.0 {
                return 1.0;
            }
            1.0 - (dot / denom).clamp(-1.0, 1.0)
        }
    }

    /// Manhattan distance (NEON, dual-accumulator, 8-wide).
    #[target_feature(enable = "neon")]
    pub unsafe fn distance_l1(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut i = 0;
            let len = a.len();
            while i + 8 <= len {
                if i + 32 < len {
                    prefetch_l1(a.as_ptr().add(i + 32));
                    prefetch_l1(b.as_ptr().add(i + 32));
                }
                let av0 = vld1q_f32(a.as_ptr().add(i));
                let bv0 = vld1q_f32(b.as_ptr().add(i));
                let av1 = vld1q_f32(a.as_ptr().add(i + 4));
                let bv1 = vld1q_f32(b.as_ptr().add(i + 4));
                let d0 = vsubq_f32(av0, bv0);
                let d1 = vsubq_f32(av1, bv1);
                acc0 = vaddq_f32(acc0, vabsq_f32(d0));
                acc1 = vaddq_f32(acc1, vabsq_f32(d1));
                i += 8;
            }
            let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
            for j in i..len {
                sum += (a[j] - b[j]).abs();
            }
            sum
        }
    }

    /// Dispatch for a metric (NEON).
    #[target_feature(enable = "neon")]
    pub unsafe fn distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
        match metric {
            DistanceMetric::Euclid => distance_l2(a, b),
            DistanceMetric::Dot => -inner_product(a, b),
            DistanceMetric::Cosine => distance_cosine(a, b),
            DistanceMetric::Manhattan => distance_l1(a, b),
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use imp::{distance, distance_cosine, distance_l1, distance_l2, inner_product};

#[cfg(not(target_arch = "aarch64"))]
mod imp_fallback {
    use super::DistanceMetric;
    use crate::distance::naive;

    /// Fallback: on non-aarch64 hosts NEON is unavailable; delegate to naive.
    /// Marked unsafe to match the aarch64 signature so `kernel.rs` can call
    /// uniformly without cfg at the call site (availability is guarded there).
    ///
    /// # Safety
    ///
    /// Delegates to a safe naive implementation; safe to call on any platform.
    pub unsafe fn distance_l2(a: &[f32], b: &[f32]) -> f32 {
        naive::distance_l2(a, b)
    }
    /// # Safety
    ///
    /// Delegates to a safe naive implementation; safe to call on any platform.
    pub unsafe fn inner_product(a: &[f32], b: &[f32]) -> f32 {
        naive::inner_product(a, b)
    }
    /// # Safety
    ///
    /// Delegates to a safe naive implementation; safe to call on any platform.
    pub unsafe fn distance_cosine(a: &[f32], b: &[f32]) -> f32 {
        naive::distance_cosine(a, b)
    }
    /// # Safety
    ///
    /// Delegates to a safe naive implementation; safe to call on any platform.
    pub unsafe fn distance_l1(a: &[f32], b: &[f32]) -> f32 {
        naive::distance_l1(a, b)
    }
    /// # Safety
    ///
    /// Delegates to a safe naive implementation; safe to call on any platform.
    pub unsafe fn distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
        naive::distance(metric, a, b)
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub use imp_fallback::{distance, distance_cosine, distance_l1, distance_l2, inner_product};
