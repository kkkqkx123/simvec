//! AVX-512 distance kernels (x86-64 only).
//!
//! Extends `avx2.rs` with ZMM (16 x f32) registers. Selection is via
//! `kernel::best_available`: `Avx512 > Avx2 > Naive` on `x86_64`. The runtime
//! check is `avx512f` (the common subset of AVX-512); `avx512bw/vl/vnni/bf16`
//! variants are accepted as supersets but not required for correctness.
//! Fallback remains the same scalar tail as AVX2.
//!
//! Main loops use a single ZMM accumulator (16 f32 per iteration) unlike the
//! dual-accumulator AVX2 kernels: the FMA dependency chain on AVX-512 has
//! 4-cycle latency too, but the wider register amply saturates the load
//! ports, and a bench comparison is required before adding a second
//! accumulator. The 8-wide tail reduces via YMM as in `avx2.rs`.

use crate::types::DistanceMetric;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Horizontally reduce a ZMM lane sum to a scalar.
///
/// Uses two-stage reduction: ZMM -> YMM -> XMM, which keeps the exact same
/// rounding as `avx2::horizontal_sum` (the reference for bench thresholds).
/// Requires AVX-512F at the call site.
#[inline]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn horizontal_sum512(v: __m512) -> f32 {
    // Prefer the single-instruction reduce when available; fallback to
    // manual extract+add so the code works on all Rust versions.
    // `_mm512_reduce_add_ps` is AVX-512F in newer toolchains; if missing,
    // this path still compiles via the extract route below.
    // We use extract to stay compatible with stable 1.88+.
    unsafe {
        // Split ZMM into two YMM halves.
        let lo256 = _mm512_castps512_ps256(v);
        let hi256 = _mm512_extractf32x8_ps(v, 1);
        let sum256 = _mm256_add_ps(lo256, hi256);
        // Reduce YMM -> XMM (same as avx2::horizontal_sum).
        let hi128 = _mm256_extractf128_ps(sum256, 1);
        let lo128 = _mm256_castps256_ps128(sum256);
        let sum128 = _mm_add_ps(hi128, lo128);
        let sum128 = _mm_hadd_ps(sum128, sum128);
        let sum128 = _mm_hadd_ps(sum128, sum128);
        _mm_cvtss_f32(sum128)
    }
}

/// Horizontally reduce a YMM lane sum to a scalar.
///
/// Mirrors `avx2::horizontal_sum` with the same rounding. Requires AVX2
/// at the call site; shared across the 8-wide tail paths in this module.
#[inline]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn horizontal_sum256(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(hi, lo);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// Squared Euclidean distance (ZMM).
///
/// # Safety
/// Caller must ensure `a.len() == b.len()` and CPU supports AVX-512F.
#[target_feature(enable = "avx512f,avx2")]
pub unsafe fn distance_l2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        let len = a.len();
        // Prefetch distance: 4 cache lines ahead (64 bytes per line,
        // 16 f32 per ZMM = 64 bytes).
        while i + 16 <= len {
            // Prefetch a bit ahead to hide memory latency on large dims.
            if i + 64 < len {
                _mm_prefetch(a.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(b.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
            }
            let av = _mm512_loadu_ps(a.as_ptr().add(i));
            let bv = _mm512_loadu_ps(b.as_ptr().add(i));
            let d = _mm512_sub_ps(av, bv);
            acc = _mm512_fmadd_ps(d, d, acc);
            i += 16;
        }
        // Handle remaining 8-wide chunk with YMM to avoid scalar tail penalty.
        if i + 8 <= len {
            let av8 = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv8 = _mm256_loadu_ps(b.as_ptr().add(i));
            let d8 = _mm256_sub_ps(av8, bv8);
            let tail8 = horizontal_sum256(_mm256_mul_ps(d8, d8));
            let mut sum = horizontal_sum512(acc);
            sum += tail8;
            i += 8;
            for j in i..len {
                let d = a[j] - b[j];
                sum += d * d;
            }
            return sum;
        }
        let mut sum = horizontal_sum512(acc);
        for j in i..len {
            let d = a[j] - b[j];
            sum += d * d;
        }
        sum
    }
}

/// Dot product (ZMM).
///
/// # Safety
/// Same preconditions as `distance_l2`.
#[target_feature(enable = "avx512f,avx2")]
pub unsafe fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        let len = a.len();
        while i + 16 <= len {
            if i + 64 < len {
                _mm_prefetch(a.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(b.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
            }
            let av = _mm512_loadu_ps(a.as_ptr().add(i));
            let bv = _mm512_loadu_ps(b.as_ptr().add(i));
            acc = _mm512_fmadd_ps(av, bv, acc);
            i += 16;
        }
        if i + 8 <= len {
            let av8 = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv8 = _mm256_loadu_ps(b.as_ptr().add(i));
            let tail8 = horizontal_sum256(_mm256_mul_ps(av8, bv8));
            let mut sum = horizontal_sum512(acc);
            sum += tail8;
            i += 8;
            for j in i..len {
                sum += a[j] * b[j];
            }
            return sum;
        }
        let mut sum = horizontal_sum512(acc);
        for j in i..len {
            sum += a[j] * b[j];
        }
        sum
    }
}

/// Cosine distance: single loop accumulating dot + both norms (ZMM).
///
/// # Safety
/// Same preconditions as `distance_l2`.
#[target_feature(enable = "avx512f,avx2")]
pub unsafe fn distance_cosine(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let mut acc_dot = _mm512_setzero_ps();
        let mut acc_na = _mm512_setzero_ps();
        let mut acc_nb = _mm512_setzero_ps();
        let mut i = 0;
        let len = a.len();
        while i + 16 <= len {
            if i + 64 < len {
                _mm_prefetch(a.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(b.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
            }
            let av = _mm512_loadu_ps(a.as_ptr().add(i));
            let bv = _mm512_loadu_ps(b.as_ptr().add(i));
            acc_dot = _mm512_fmadd_ps(av, bv, acc_dot);
            acc_na = _mm512_fmadd_ps(av, av, acc_na);
            acc_nb = _mm512_fmadd_ps(bv, bv, acc_nb);
            i += 16;
        }
        // Fold remaining 8-wide with YMM helpers then scalar tail.
        let mut dot_extra = 0.0f32;
        let mut na_extra = 0.0f32;
        let mut nb_extra = 0.0f32;
        if i + 8 <= len {
            let av8 = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv8 = _mm256_loadu_ps(b.as_ptr().add(i));
            dot_extra += horizontal_sum256(_mm256_mul_ps(av8, bv8));
            na_extra += horizontal_sum256(_mm256_mul_ps(av8, av8));
            nb_extra += horizontal_sum256(_mm256_mul_ps(bv8, bv8));
            i += 8;
        }
        let mut dot = horizontal_sum512(acc_dot) + dot_extra;
        let mut norm_a = horizontal_sum512(acc_na) + na_extra;
        let mut norm_b = horizontal_sum512(acc_nb) + nb_extra;
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

/// Manhattan distance (ZMM).
///
/// # Safety
/// Same preconditions as `distance_l2`.
#[target_feature(enable = "avx512f,avx2")]
pub unsafe fn distance_l1(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let sign = _mm512_set1_ps(-0.0f32);
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        let len = a.len();
        while i + 16 <= len {
            if i + 64 < len {
                _mm_prefetch(a.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(b.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);
            }
            let av = _mm512_loadu_ps(a.as_ptr().add(i));
            let bv = _mm512_loadu_ps(b.as_ptr().add(i));
            let d = _mm512_sub_ps(av, bv);
            let abs = _mm512_andnot_ps(sign, d);
            acc = _mm512_add_ps(acc, abs);
            i += 16;
        }
        if i + 8 <= len {
            let sign256 = _mm256_set1_ps(-0.0f32);
            let av8 = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv8 = _mm256_loadu_ps(b.as_ptr().add(i));
            let d8 = _mm256_sub_ps(av8, bv8);
            let tail8 = horizontal_sum256(_mm256_andnot_ps(sign256, d8));
            let mut sum = horizontal_sum512(acc);
            sum += tail8;
            i += 8;
            for j in i..len {
                sum += (a[j] - b[j]).abs();
            }
            return sum;
        }
        let mut sum = horizontal_sum512(acc);
        for j in i..len {
            sum += (a[j] - b[j]).abs();
        }
        sum
    }
}

/// Dispatch for a metric (AVX-512).
///
/// # Safety
/// Must only be called with AVX-512F verified at runtime.
#[target_feature(enable = "avx512f,avx2")]
pub unsafe fn distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        DistanceMetric::Euclid => distance_l2(a, b),
        DistanceMetric::Dot => -inner_product(a, b),
        DistanceMetric::Cosine => distance_cosine(a, b),
        DistanceMetric::Manhattan => distance_l1(a, b),
    }
}
