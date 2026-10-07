//! AVX2 distance kernels (x86-64 only).
//!
//! Selection via `kernel::best_available` (runtime dispatch on the baseline
//! `x86-64` build). The `x86-64-v3` compile-time mode is now opt-in via
//! `RUSTFLAGS="-C target-cpu=x86-64-v3"` for specialized deployments; the
//! runtime `is_x86_feature_detected!` checks in `Kernel::Avx2` are the
//! correctness gate on portable binaries.
//!
//! Each kernel uses dual accumulators (2 x __m256) processing 16 f32 per
//! iteration to hide the 4-cycle FMA latency. The two accumulators are
//! independent so the CPU can issue back-to-back `_mm256_fmadd_ps` without
//! stalling. An 8-wide tail chunk avoids a pure scalar fallback when dim is
//! not a multiple of 16. Loads are unaligned (`_mm256_loadu_ps`), so any
//! alignment works. Loops include `_mm_prefetch` 4 YMM ahead (256 bytes) to
//! hide memory latency on `dim=1536` scans.

use crate::types::DistanceMetric;

use std::arch::x86_64::*;

/// Horizontally reduce a YMM lane sum to a scalar.
///
/// # Safety
///
/// Requires AVX at the call site (`_mm256_extractf128_ps`/`_mm256_castps256_ps128`
/// are AVX intrinsics); all callers are `#[target_feature(enable = "avx2,fma")]`
/// functions.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum = _mm_add_ps(hi, lo);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

/// Squared Euclidean distance (dual-accumulator, 16-wide main loop).
///
/// # Safety
///
/// The caller must ensure `a` and `b` have the same length, and that the
/// current CPU supports AVX2 + FMA (checked by `is_x86_feature_detected!`
/// before dispatch).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn distance_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= a.len() {
        if i + 32 < a.len() {
            _mm_prefetch(a.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);
        }
        let av0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let av1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let bv1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let d0 = _mm256_sub_ps(av0, bv0);
        let d1 = _mm256_sub_ps(av1, bv1);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        i += 16;
    }
    // Handle remaining 8-wide chunk with YMM.
    let mut sum = horizontal_sum(_mm256_add_ps(acc0, acc1));
    if i + 8 <= a.len() {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        let d = _mm256_sub_ps(av, bv);
        sum += horizontal_sum(_mm256_mul_ps(d, d));
        i += 8;
    }
    for j in i..a.len() {
        let d = a[j] - b[j];
        sum += d * d;
    }
    sum
}

/// Dot product (dual-accumulator, 16-wide main loop).
///
/// # Safety
///
/// The caller must ensure `a` and `b` have the same length, and that the
/// current CPU supports AVX2 + FMA (checked by `is_x86_feature_detected!`
/// before dispatch).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= a.len() {
        if i + 32 < a.len() {
            _mm_prefetch(a.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);
        }
        let av0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let av1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let bv1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
        acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
        i += 16;
    }
    let mut sum = horizontal_sum(_mm256_add_ps(acc0, acc1));
    if i + 8 <= a.len() {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        sum += horizontal_sum(_mm256_mul_ps(av, bv));
        i += 8;
    }
    for j in i..a.len() {
        sum += a[j] * b[j];
    }
    sum
}

/// Cosine distance: dual-accumulator, 16-wide main loop accumulating
/// dot + both norms in a single pass.
///
/// # Safety
///
/// The caller must ensure `a` and `b` have the same length, and that the
/// current CPU supports AVX2 + FMA (checked by `is_x86_feature_detected!`
/// before dispatch).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn distance_cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut acc_dot0 = _mm256_setzero_ps();
    let mut acc_na0 = _mm256_setzero_ps();
    let mut acc_nb0 = _mm256_setzero_ps();
    let mut acc_dot1 = _mm256_setzero_ps();
    let mut acc_na1 = _mm256_setzero_ps();
    let mut acc_nb1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= a.len() {
        if i + 32 < a.len() {
            _mm_prefetch(a.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);
        }
        let av0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let av1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let bv1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc_dot0 = _mm256_fmadd_ps(av0, bv0, acc_dot0);
        acc_na0 = _mm256_fmadd_ps(av0, av0, acc_na0);
        acc_nb0 = _mm256_fmadd_ps(bv0, bv0, acc_nb0);
        acc_dot1 = _mm256_fmadd_ps(av1, bv1, acc_dot1);
        acc_na1 = _mm256_fmadd_ps(av1, av1, acc_na1);
        acc_nb1 = _mm256_fmadd_ps(bv1, bv1, acc_nb1);
        i += 16;
    }
    let mut dot = horizontal_sum(_mm256_add_ps(acc_dot0, acc_dot1));
    let mut norm_a = horizontal_sum(_mm256_add_ps(acc_na0, acc_na1));
    let mut norm_b = horizontal_sum(_mm256_add_ps(acc_nb0, acc_nb1));
    if i + 8 <= a.len() {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        dot += horizontal_sum(_mm256_mul_ps(av, bv));
        norm_a += horizontal_sum(_mm256_mul_ps(av, av));
        norm_b += horizontal_sum(_mm256_mul_ps(bv, bv));
        i += 8;
    }
    for j in i..a.len() {
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

/// Manhattan distance (dual-accumulator, 16-wide main loop).
///
/// # Safety
///
/// The caller must ensure `a` and `b` have the same length, and that the
/// current CPU supports AVX2 (checked by `is_x86_feature_detected!` before dispatch).
#[target_feature(enable = "avx2")]
pub unsafe fn distance_l1(a: &[f32], b: &[f32]) -> f32 {
    let sign = _mm256_set1_ps(-0.0f32);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= a.len() {
        if i + 32 < a.len() {
            _mm_prefetch(a.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b.as_ptr().add(i + 32) as *const i8, _MM_HINT_T0);
        }
        let av0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let av1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let bv1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let d0 = _mm256_sub_ps(av0, bv0);
        let d1 = _mm256_sub_ps(av1, bv1);
        acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign, d0));
        acc1 = _mm256_add_ps(acc1, _mm256_andnot_ps(sign, d1));
        i += 16;
    }
    let mut sum = horizontal_sum(_mm256_add_ps(acc0, acc1));
    if i + 8 <= a.len() {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        let d = _mm256_sub_ps(av, bv);
        sum += horizontal_sum(_mm256_andnot_ps(sign, d));
        i += 8;
    }
    for j in i..a.len() {
        sum += (a[j] - b[j]).abs();
    }
    sum
}

/// Dispatch for a metric.
///
/// # Safety
/// Must only be called with AVX2+FMA verified at runtime.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Euclid => distance_l2(a, b),
        DistanceMetric::Dot => -inner_product(a, b),
        DistanceMetric::Cosine => distance_cosine(a, b),
        DistanceMetric::Manhattan => distance_l1(a, b),
    }
}
