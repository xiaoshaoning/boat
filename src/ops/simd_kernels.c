// simd_kernels.c - SIMD micro-kernel implementations
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/simd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Helper: process tail elements that don't fill a full vector
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// boat_simd_add_f32
// ---------------------------------------------------------------------------

BOAT_API void boat_simd_add_f32(const float* a, const float* b, float* dst, size_t n) {
#if BOAT_HAVE_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vd = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++)
        dst[i] = a[i] + b[i];
#elif BOAT_HAVE_NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vd = vaddq_f32(va, vb);
        vst1q_f32(dst + i, vd);
    }
    for (; i < n; i++)
        dst[i] = a[i] + b[i];
#else
    for (size_t i = 0; i < n; i++)
        dst[i] = a[i] + b[i];
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_mul_f32
// ---------------------------------------------------------------------------

BOAT_API void boat_simd_mul_f32(const float* a, const float* b, float* dst, size_t n) {
#if BOAT_HAVE_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vd = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++)
        dst[i] = a[i] * b[i];
#elif BOAT_HAVE_NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vd = vmulq_f32(va, vb);
        vst1q_f32(dst + i, vd);
    }
    for (; i < n; i++)
        dst[i] = a[i] * b[i];
#else
    for (size_t i = 0; i < n; i++)
        dst[i] = a[i] * b[i];
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_mul_scalar_f32
// ---------------------------------------------------------------------------

BOAT_API void boat_simd_mul_scalar_f32(const float* a, float scalar, float* dst, size_t n) {
#if BOAT_HAVE_AVX2
    __m256 vs = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vd = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++)
        dst[i] = a[i] * scalar;
#elif BOAT_HAVE_NEON
    float32x4_t vs = vdupq_n_f32(scalar);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vd = vmulq_f32(va, vs);
        vst1q_f32(dst + i, vd);
    }
    for (; i < n; i++)
        dst[i] = a[i] * scalar;
#else
    for (size_t i = 0; i < n; i++)
        dst[i] = a[i] * scalar;
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_relu_f32
// ---------------------------------------------------------------------------

BOAT_API void boat_simd_relu_f32(const float* a, float* dst, size_t n) {
#if BOAT_HAVE_AVX2
    __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vd = _mm256_max_ps(va, vzero);
        _mm256_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++)
        dst[i] = a[i] > 0 ? a[i] : 0.0f;
#elif BOAT_HAVE_NEON
    float32x4_t vzero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vd = vmaxq_f32(va, vzero);
        vst1q_f32(dst + i, vd);
    }
    for (; i < n; i++)
        dst[i] = a[i] > 0 ? a[i] : 0.0f;
#else
    for (size_t i = 0; i < n; i++)
        dst[i] = a[i] > 0 ? a[i] : 0.0f;
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_max_reduce_f32
// ---------------------------------------------------------------------------

BOAT_API float boat_simd_max_reduce_f32(const float* a, size_t n) {
    if (n == 0) return 0.0f;
#if BOAT_HAVE_AVX2
    if (n < 8) {
        float result = a[0];
        for (size_t i = 1; i < n; i++) {
            if (a[i] > result) result = a[i];
        }
        return result;
    }
    __m256 vmax = _mm256_loadu_ps(a);
    size_t i = 8;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        vmax = _mm256_max_ps(vmax, va);
    }
    // Horizontal max across the 256-bit vector
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    float result = _mm_cvtss_f32(m);
    for (; i < n; i++) {
        if (a[i] > result) result = a[i];
    }
    return result;
#elif BOAT_HAVE_NEON
    if (n < 4) {
        float result = a[0];
        for (size_t i = 1; i < n; i++) {
            if (a[i] > result) result = a[i];
        }
        return result;
    }
    float32x4_t vmax = vld1q_f32(a);
    size_t i = 4;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vmax = vmaxq_f32(vmax, va);
    }
    float result = vmaxvq_f32(vmax);
    for (; i < n; i++) {
        if (a[i] > result) result = a[i];
    }
    return result;
#else
    float result = a[0];
    for (size_t i = 1; i < n; i++) {
        if (a[i] > result) result = a[i];
    }
    return result;
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_min_reduce_f32
// ---------------------------------------------------------------------------

BOAT_API float boat_simd_min_reduce_f32(const float* a, size_t n) {
    if (n == 0) return 0.0f;
#if BOAT_HAVE_AVX2
    if (n < 8) {
        float result = a[0];
        for (size_t i = 1; i < n; i++) {
            if (a[i] < result) result = a[i];
        }
        return result;
    }
    __m256 vmin = _mm256_loadu_ps(a);
    size_t i = 8;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        vmin = _mm256_min_ps(vmin, va);
    }
    __m128 hi = _mm256_extractf128_ps(vmin, 1);
    __m128 lo = _mm256_castps256_ps128(vmin);
    __m128 m = _mm_min_ps(lo, hi);
    m = _mm_min_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_min_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    float result = _mm_cvtss_f32(m);
    for (; i < n; i++) {
        if (a[i] < result) result = a[i];
    }
    return result;
#elif BOAT_HAVE_NEON
    if (n < 4) {
        float result = a[0];
        for (size_t i = 1; i < n; i++) {
            if (a[i] < result) result = a[i];
        }
        return result;
    }
    float32x4_t vmin = vld1q_f32(a);
    size_t i = 4;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vmin = vminq_f32(vmin, va);
    }
    float result = vminvq_f32(vmin);
    for (; i < n; i++) {
        if (a[i] < result) result = a[i];
    }
    return result;
#else
    float result = a[0];
    for (size_t i = 1; i < n; i++) {
        if (a[i] < result) result = a[i];
    }
    return result;
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_sum_reduce_f32
// ---------------------------------------------------------------------------

BOAT_API float boat_simd_sum_reduce_f32(const float* a, size_t n) {
    if (n == 0) return 0.0f;
#if BOAT_HAVE_AVX2
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        vsum = _mm256_add_ps(vsum, va);
    }
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    for (; i < n; i++)
        result += a[i];
    return result;
#elif BOAT_HAVE_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vsum = vaddq_f32(vsum, va);
    }
    float result = vaddvq_f32(vsum);
    for (; i < n; i++)
        result += a[i];
    return result;
#else
    float result = 0.0f;
    for (size_t i = 0; i < n; i++)
        result += a[i];
    return result;
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_transpose2d_f32
// ---------------------------------------------------------------------------

#if BOAT_HAVE_AVX2
// Transpose one 8x8 block: src rows are `srow` floats apart, dst rows are
// `drow` floats apart (both read/written 8 consecutive floats at the offset).
static void transpose8x8_avx2(const float* src, float* dst, size_t srow, size_t drow) {
    __m256 r[8];
    for (int i = 0; i < 8; i++)
        r[i] = _mm256_loadu_ps(src + (size_t)i * srow);
    // Stage 1: unpack row pairs -> [r0_0,r1_0,r0_1,r1_1, r2_0,r3_0,...].
    __m256 a[8];
    for (int i = 0; i < 4; i++) {
        a[2 * i] = _mm256_unpacklo_ps(r[2 * i], r[2 * i + 1]);
        a[2 * i + 1] = _mm256_unpackhi_ps(r[2 * i], r[2 * i + 1]);
    }
    // Stage 2: combine the two 128-bit lanes of (rows 0-3) with (rows 4-7).
    __m256 v[8];
    v[0] = _mm256_permute2f128_ps(a[0], a[4], 0x20);
    v[1] = _mm256_permute2f128_ps(a[1], a[5], 0x20);
    v[2] = _mm256_permute2f128_ps(a[0], a[4], 0x31);
    v[3] = _mm256_permute2f128_ps(a[1], a[5], 0x31);
    v[4] = _mm256_permute2f128_ps(a[2], a[6], 0x20);
    v[5] = _mm256_permute2f128_ps(a[3], a[7], 0x20);
    v[6] = _mm256_permute2f128_ps(a[2], a[6], 0x31);
    v[7] = _mm256_permute2f128_ps(a[3], a[7], 0x31);
    // Stage 3: in-lane shuffle to the transposed columns.
    __m256 c[8];
    c[0] = _mm256_shuffle_ps(v[0], v[4], 0x44);
    c[1] = _mm256_shuffle_ps(v[0], v[4], 0xEE);
    c[2] = _mm256_shuffle_ps(v[1], v[5], 0x44);
    c[3] = _mm256_shuffle_ps(v[1], v[5], 0xEE);
    c[4] = _mm256_shuffle_ps(v[2], v[6], 0x44);
    c[5] = _mm256_shuffle_ps(v[2], v[6], 0xEE);
    c[6] = _mm256_shuffle_ps(v[3], v[7], 0x44);
    c[7] = _mm256_shuffle_ps(v[3], v[7], 0xEE);
    for (int i = 0; i < 8; i++)
        _mm256_storeu_ps(dst + (size_t)i * drow, c[i]);
}
#endif

#if BOAT_HAVE_SSE41 && !BOAT_HAVE_AVX2
// Transpose one 4x4 block (128-bit path).
static void transpose4x4_sse(const float* src, float* dst, size_t srow, size_t drow) {
    __m128 r[4];
    for (int i = 0; i < 4; i++)
        r[i] = _mm_loadu_ps(src + (size_t)i * srow);
    __m128 a0 = _mm_unpacklo_ps(r[0], r[1]);
    __m128 a1 = _mm_unpackhi_ps(r[0], r[1]);
    __m128 a2 = _mm_unpacklo_ps(r[2], r[3]);
    __m128 a3 = _mm_unpackhi_ps(r[2], r[3]);
    __m128 t0 = _mm_movelh_ps(a0, a2);
    __m128 t1 = _mm_movehl_ps(a2, a0);
    __m128 t2 = _mm_movelh_ps(a1, a3);
    __m128 t3 = _mm_movehl_ps(a3, a1);
    __m128 c[4] = {t0, t1, t2, t3};
    for (int i = 0; i < 4; i++)
        _mm_storeu_ps(dst + (size_t)i * drow, c[i]);
}
#elif BOAT_HAVE_NEON
static void transpose4x4_neon(const float* src, float* dst, size_t srow, size_t drow) {
    float32x4_t r[4];
    for (int i = 0; i < 4; i++)
        r[i] = vld1q_f32(src + (size_t)i * srow);
    float32x4_t a0 = vzip1q_f32(r[0], r[1]);
    float32x4_t a1 = vzip2q_f32(r[0], r[1]);
    float32x4_t a2 = vzip1q_f32(r[2], r[3]);
    float32x4_t a3 = vzip2q_f32(r[2], r[3]);
    float32x4_t b0 = vcombine_f32(vget_low_f32(a0), vget_low_f32(a2));
    float32x4_t b1 = vcombine_f32(vget_high_f32(a0), vget_high_f32(a2));
    float32x4_t b2 = vcombine_f32(vget_low_f32(a1), vget_low_f32(a3));
    float32x4_t b3 = vcombine_f32(vget_high_f32(a1), vget_high_f32(a3));
    float32x4_t c[4] = {b0, b1, b2, b3};
    for (int i = 0; i < 4; i++)
        vst1q_f32(dst + (size_t)i * drow, c[i]);
}
#endif

BOAT_API void boat_simd_transpose2d_f32(const float* src, float* dst, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return;
#if BOAT_HAVE_AVX2
    // 8x8 tiled interior.
    size_t r = 0;
    for (; r + 8 <= rows; r += 8) {
        size_t c = 0;
        for (; c + 8 <= cols; c += 8) {
            transpose8x8_avx2(src + r * cols + c, dst + c * rows + r, cols, rows);
        }
        // Right tail columns (fewer than 8).
        for (; c < cols; c++) {
            for (size_t rr = r; rr < r + 8; rr++)
                dst[c * rows + rr] = src[rr * cols + c];
        }
    }
    // Bottom tail rows.
    for (; r < rows; r++) {
        for (size_t c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
    }
#elif BOAT_HAVE_SSE41 || BOAT_HAVE_NEON
    size_t tile = 4;
    size_t r = 0;
    for (; r + tile <= rows; r += tile) {
        size_t c = 0;
        for (; c + tile <= cols; c += tile) {
#if BOAT_HAVE_SSE41
            transpose4x4_sse(src + r * cols + c, dst + c * rows + r, cols, rows);
#else
            transpose4x4_neon(src + r * cols + c, dst + c * rows + r, cols, rows);
#endif
        }
        for (; c < cols; c++) {
            for (size_t rr = r; rr < r + tile; rr++)
                dst[c * rows + rr] = src[rr * cols + c];
        }
    }
    for (; r < rows; r++) {
        for (size_t c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
    }
#else
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
    }
#endif
}

// ---------------------------------------------------------------------------
// Transcendental / activation kernels (AVX2 via a fast exp2-based exp).
// ---------------------------------------------------------------------------

#if BOAT_HAVE_AVX2
// exp(x) = 2^(x*log2(e)): split into an integer exponent and a fractional
// part evaluated with a degree-7 Horner polynomial (~1e-7 relative error).
BOAT_API __m256 boat_simd_exp256(__m256 x) {
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    x = _mm256_min_ps(x, _mm256_set1_ps(87.33654f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.33654f));
    __m256 f = _mm256_mul_ps(x, log2e);
    f = _mm256_round_ps(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 z = _mm256_fnmadd_ps(f, ln2, x);
    __m256 y = _mm256_set1_ps(1.0f / 5040.0f);
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(1.0f / 720.0f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(1.0f / 120.0f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(1.0f / 24.0f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(1.0f / 6.0f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(0.5f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(1.0f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(1.0f));
    __m256i i = _mm256_cvtps_epi32(f);
    i = _mm256_add_epi32(i, _mm256_set1_epi32(127));
    i = _mm256_slli_epi32(i, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(i));
}

// tanh(x) = 1 - 2/(exp(2x)+1), with 2x clamped to the exp range.
BOAT_API __m256 boat_simd_tanh256(__m256 x) {
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 t = _mm256_min_ps(_mm256_mul_ps(x, two), _mm256_set1_ps(87.33654f));
    t = _mm256_max_ps(t, _mm256_set1_ps(-87.33654f));
    __m256 e = boat_simd_exp256(t);
    __m256 r = _mm256_div_ps(two, _mm256_add_ps(one, e));
    return _mm256_sub_ps(one, r);
}

BOAT_API __m256 boat_simd_sigmoid256(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 e = boat_simd_exp256(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(one, _mm256_add_ps(one, e));
}
#endif

BOAT_API void boat_simd_exp_f32(const float* a, float* dst, size_t n) {
#if BOAT_HAVE_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(dst + i, boat_simd_exp256(v));
    }
    for (; i < n; i++) dst[i] = expf(a[i]);
#else
    for (size_t i = 0; i < n; i++) dst[i] = expf(a[i]);
#endif
}

BOAT_API void boat_simd_sigmoid_f32(const float* a, float* dst, size_t n) {
#if BOAT_HAVE_AVX2
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 e = boat_simd_exp256(_mm256_sub_ps(zero, _mm256_loadu_ps(a + i)));
        _mm256_storeu_ps(dst + i, _mm256_div_ps(one, _mm256_add_ps(one, e)));
    }
    for (; i < n; i++) dst[i] = 1.0f / (1.0f + expf(-a[i]));
#else
    for (size_t i = 0; i < n; i++) dst[i] = 1.0f / (1.0f + expf(-a[i]));
#endif
}

BOAT_API void boat_simd_tanh_f32(const float* a, float* dst, size_t n) {
#if BOAT_HAVE_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(dst + i, boat_simd_tanh256(v));
    }
    for (; i < n; i++) dst[i] = tanhf(a[i]);
#else
    for (size_t i = 0; i < n; i++) dst[i] = tanhf(a[i]);
#endif
}

BOAT_API void boat_simd_silu_f32(const float* a, float* dst, size_t n) {
#if BOAT_HAVE_AVX2
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(a + i);
        __m256 e = boat_simd_exp256(_mm256_sub_ps(zero, x));
        __m256 s = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(x, s));
    }
    for (; i < n; i++) dst[i] = a[i] / (1.0f + expf(-a[i]));
#else
    for (size_t i = 0; i < n; i++) dst[i] = a[i] / (1.0f + expf(-a[i]));
#endif
}

BOAT_API void boat_simd_gelu_f32(const float* a, float* dst, size_t n) {
    // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    const float k = 0.7978845608028654f;   // sqrt(2/pi)
    const float c = 0.044715f;
#if BOAT_HAVE_AVX2
    const __m256 vk = _mm256_set1_ps(k);
    const __m256 vc = _mm256_set1_ps(c);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(a + i);
        __m256 x3 = _mm256_mul_ps(x, _mm256_mul_ps(x, x));
        __m256 t = boat_simd_tanh256(_mm256_mul_ps(vk, _mm256_add_ps(x, _mm256_mul_ps(vc, x3))));
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, t))));
    }
    for (; i < n; i++) {
        float x = a[i];
        dst[i] = 0.5f * x * (1.0f + tanhf(k * (x + c * x * x * x)));
    }
#else
    for (size_t i = 0; i < n; i++) {
        float x = a[i];
        dst[i] = 0.5f * x * (1.0f + tanhf(k * (x + c * x * x * x)));
    }
#endif
}

BOAT_API void boat_simd_softmax_f32(const float* a, float* dst, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return;
    for (size_t r = 0; r < rows; r++) {
        const float* row = a + r * cols;
        float* drow = dst + r * cols;
        float mx = row[0];
        for (size_t c = 1; c < cols; c++) {
            if (row[c] > mx) mx = row[c];
        }
#if BOAT_HAVE_AVX2
        const __m256 vmx = _mm256_set1_ps(mx);
        size_t c = 0;
        __m256 vsum = _mm256_setzero_ps();
        for (; c + 8 <= cols; c += 8) {
            __m256 e = boat_simd_exp256(_mm256_sub_ps(_mm256_loadu_ps(row + c), vmx));
            _mm256_storeu_ps(drow + c, e);
            vsum = _mm256_add_ps(vsum, e);
        }
        float sum = 0.0f;
        for (size_t k = 0; k < c; k++) sum += drow[k];  // includes the vector tail
        for (; c < cols; c++) {
            drow[c] = expf(row[c] - mx);
            sum += drow[c];
        }
        for (size_t k = 0; k < cols; k++) drow[k] /= sum;
#else
        float sum = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            drow[c] = expf(row[c] - mx);
            sum += drow[c];
        }
        for (size_t c = 0; c < cols; c++) drow[c] /= sum;
#endif
    }
}

// ---------------------------------------------------------------------------
// Activation-derivative kernels (fused, dst[i] = dy[i] * f'(x[i])).
// ---------------------------------------------------------------------------

BOAT_API void boat_simd_sigmoid_backward_f32(const float* dy, const float* y, float* dst,
                                             size_t n) {
#if BOAT_HAVE_AVX2
    const __m256 one = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vd = _mm256_loadu_ps(dy + i);
        __m256 d = _mm256_mul_ps(vy, _mm256_sub_ps(one, vy));  // y*(1-y)
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(vd, d));
    }
    for (; i < n; i++) dst[i] = dy[i] * y[i] * (1.0f - y[i]);
#else
    for (size_t i = 0; i < n; i++) dst[i] = dy[i] * y[i] * (1.0f - y[i]);
#endif
}

BOAT_API void boat_simd_tanh_backward_f32(const float* dy, const float* y, float* dst, size_t n) {
#if BOAT_HAVE_AVX2
    const __m256 one = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vd = _mm256_loadu_ps(dy + i);
        __m256 d = _mm256_fnmadd_ps(vy, vy, one);  // 1 - y*y
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(vd, d));
    }
    for (; i < n; i++) dst[i] = dy[i] * (1.0f - y[i] * y[i]);
#else
    for (size_t i = 0; i < n; i++) dst[i] = dy[i] * (1.0f - y[i] * y[i]);
#endif
}

BOAT_API void boat_simd_relu_backward_f32(const float* dy, const float* x, float* dst, size_t n) {
#if BOAT_HAVE_AVX2
    const __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vd = _mm256_loadu_ps(dy + i);
        __m256 mask = _mm256_cmp_ps(vx, zero, _CMP_GT_OQ);
        _mm256_storeu_ps(dst + i, _mm256_blendv_ps(zero, vd, mask));  // dy where x>0 else 0
    }
    for (; i < n; i++) dst[i] = x[i] > 0.0f ? dy[i] : 0.0f;
#else
    for (size_t i = 0; i < n; i++) dst[i] = x[i] > 0.0f ? dy[i] : 0.0f;
#endif
}

BOAT_API void boat_simd_gelu_backward_f32(const float* dy, const float* x, float* dst, size_t n) {
    // d/dx [0.5*x*(1+tanh(a))] = 0.5*(1+tanh(a)) + 0.5*x*(1-tanh(a)^2)*k*(1+3c*x^2),
    // with a = k*(x + c*x^3), k = sqrt(2/pi), c = 0.044715.
    const float k = 0.7978845608028654f;
    const float c = 0.044715f;
#if BOAT_HAVE_AVX2
    const __m256 vk = _mm256_set1_ps(k);
    const __m256 vc = _mm256_set1_ps(c);
    const __m256 v3c = _mm256_set1_ps(3.0f * c);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 vd = _mm256_loadu_ps(dy + i);
        __m256 x2 = _mm256_mul_ps(xv, xv);
        __m256 x3 = _mm256_mul_ps(x2, xv);
        __m256 a = _mm256_mul_ps(vk, _mm256_add_ps(xv, _mm256_mul_ps(vc, x3)));
        __m256 t = boat_simd_tanh256(a);
        __m256 omt2 = _mm256_fnmadd_ps(t, t, one);        // 1 - tanh(a)^2
        __m256 d = _mm256_mul_ps(half, _mm256_add_ps(one, t));
        __m256 inner = _mm256_fmadd_ps(v3c, x2, one);      // 1 + 3c*x^2
        __m256 t2 = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(half, xv), omt2), vk);
        d = _mm256_fmadd_ps(t2, inner, d);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(vd, d));
    }
    for (; i < n; i++) {
        float xv = x[i];
        float a = k * (xv + c * xv * xv * xv);
        float t = tanhf(a);
        float d = 0.5f * (1.0f + t) +
                  0.5f * xv * (1.0f - t * t) * k * (1.0f + 3.0f * c * xv * xv);
        dst[i] = dy[i] * d;
    }
#else
    for (size_t i = 0; i < n; i++) {
        float xv = x[i];
        float a = k * (xv + c * xv * xv * xv);
        float t = tanhf(a);
        float d = 0.5f * (1.0f + t) +
                  0.5f * xv * (1.0f - t * t) * k * (1.0f + 3.0f * c * xv * xv);
        dst[i] = dy[i] * d;
    }
#endif
}

// Softmax backward: dst_i = y_i * (dy_i - sum_k dy_k*y_k) per row.
BOAT_API void boat_simd_softmax_backward_f32(const float* dy, const float* y, float* dst,
                                             size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return;
    for (size_t r = 0; r < rows; r++) {
        const float* rowy = y + r * cols;
        const float* rowd = dy + r * cols;
        float* drow = dst + r * cols;
#if BOAT_HAVE_AVX2
        size_t c = 0;
        __m256 vsum = _mm256_setzero_ps();
        for (; c + 8 <= cols; c += 8) {
            __m256 vy = _mm256_loadu_ps(rowy + c);
            vsum = _mm256_fmadd_ps(_mm256_loadu_ps(rowd + c), vy, vsum);
        }
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 s = _mm_hadd_ps(_mm_add_ps(lo, hi), _mm_add_ps(lo, hi));
        s = _mm_hadd_ps(s, s);
        float sum = _mm_cvtss_f32(s);
        for (; c < cols; c++) sum += rowd[c] * rowy[c];
        __m256 vsum_b = _mm256_set1_ps(sum);
        size_t c2 = 0;
        for (; c2 + 8 <= cols; c2 += 8) {
            __m256 vy = _mm256_loadu_ps(rowy + c2);
            __m256 vd = _mm256_loadu_ps(rowd + c2);
            _mm256_storeu_ps(drow + c2, _mm256_mul_ps(vy, _mm256_sub_ps(vd, vsum_b)));
        }
        for (; c2 < cols; c2++) drow[c2] = rowy[c2] * (rowd[c2] - sum);
#else
        float sum = 0.0f;
        for (size_t c = 0; c < cols; c++) sum += rowd[c] * rowy[c];
        for (size_t c = 0; c < cols; c++) drow[c] = rowy[c] * (rowd[c] - sum);
#endif
    }
}

// Log-softmax backward: dst_i = dy_i - exp(y_i) * sum_k dy_k per row.
BOAT_API void boat_simd_log_softmax_backward_f32(const float* dy, const float* y, float* dst,
                                                 size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return;
    for (size_t r = 0; r < rows; r++) {
        const float* rowy = y + r * cols;
        const float* rowd = dy + r * cols;
        float* drow = dst + r * cols;
#if BOAT_HAVE_AVX2
        size_t c = 0;
        __m256 vsum = _mm256_setzero_ps();
        for (; c + 8 <= cols; c += 8) {
            vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(rowd + c));
        }
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 s = _mm_hadd_ps(_mm_add_ps(lo, hi), _mm_add_ps(lo, hi));
        s = _mm_hadd_ps(s, s);
        float sum = _mm_cvtss_f32(s);
        for (; c < cols; c++) sum += rowd[c];
        __m256 vsum_b = _mm256_set1_ps(sum);
        size_t c2 = 0;
        for (; c2 + 8 <= cols; c2 += 8) {
            __m256 vy = _mm256_loadu_ps(rowy + c2);
            __m256 vd = _mm256_loadu_ps(rowd + c2);
            __m256 e = boat_simd_exp256(vy);
            _mm256_storeu_ps(drow + c2, _mm256_fnmadd_ps(e, vsum_b, vd));
        }
        for (; c2 < cols; c2++) drow[c2] = rowd[c2] - expf(rowy[c2]) * sum;
#else
        float sum = 0.0f;
        for (size_t c = 0; c < cols; c++) sum += rowd[c];
        for (size_t c = 0; c < cols; c++) drow[c] = rowd[c] - expf(rowy[c]) * sum;
#endif
    }
}

// Fused softmax + cross-entropy backward: grad = (softmax(logits) - onehot) * inv_batch.
BOAT_API void boat_simd_softmax_ce_backward_f32(const float* logits, const int64_t* labels,
                                                float* grad, size_t rows, size_t cols,
                                                float inv_batch) {
    if (rows == 0 || cols == 0) return;
    for (size_t r = 0; r < rows; r++) {
        const float* row = logits + r * cols;
        float* grow = grad + r * cols;
        float mx = row[0];
        for (size_t c = 1; c < cols; c++) {
            if (row[c] > mx) mx = row[c];
        }
        float sum = 0.0f;
#if BOAT_HAVE_AVX2
        const __m256 vmx = _mm256_set1_ps(mx);
        size_t c = 0;
        __m256 vsum = _mm256_setzero_ps();
        for (; c + 8 <= cols; c += 8) {
            __m256 e = boat_simd_exp256(_mm256_sub_ps(_mm256_loadu_ps(row + c), vmx));
            _mm256_storeu_ps(grow + c, e);
            vsum = _mm256_add_ps(vsum, e);
        }
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 s = _mm_hadd_ps(_mm_add_ps(lo, hi), _mm_add_ps(lo, hi));
        s = _mm_hadd_ps(s, s);
        sum = _mm_cvtss_f32(s);
        for (; c < cols; c++) {
            grow[c] = expf(row[c] - mx);
            sum += grow[c];
        }
        float inv_row = 1.0f / sum;
        const __m256 vscale = _mm256_set1_ps(inv_row * inv_batch);
        size_t c2 = 0;
        for (; c2 + 8 <= cols; c2 += 8) {
            _mm256_storeu_ps(grow + c2, _mm256_mul_ps(_mm256_loadu_ps(grow + c2), vscale));
        }
        for (; c2 < cols; c2++) grow[c2] *= inv_row * inv_batch;
#else
        for (size_t c = 0; c < cols; c++) {
            grow[c] = expf(row[c] - mx);
            sum += grow[c];
        }
        float inv_row = 1.0f / sum;
        for (size_t c = 0; c < cols; c++) grow[c] *= inv_row * inv_batch;
#endif
        int64_t label = labels[r];
        if (label >= 0 && label < (int64_t)cols) {
            grow[label] -= inv_batch;
        }
    }
}

// Cross-entropy backward: grad = -inv_n * target / clip(pred, epsilon).
BOAT_API void boat_simd_ce_backward_f32(const float* pred, const float* target, float* grad,
                                        size_t n, float inv_n, float epsilon) {
    const float lo = epsilon;
    const float hi = 1.0f - epsilon;
#if BOAT_HAVE_AVX2
    const __m256 vlo = _mm256_set1_ps(lo);
    const __m256 vhi = _mm256_set1_ps(hi);
    const __m256 vn = _mm256_set1_ps(-inv_n);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vp = _mm256_max_ps(_mm256_min_ps(_mm256_loadu_ps(pred + i), vhi), vlo);
        __m256 vt = _mm256_loadu_ps(target + i);
        _mm256_storeu_ps(grad + i, _mm256_mul_ps(_mm256_div_ps(vt, vp), vn));
    }
    for (; i < n; i++) {
        float p = pred[i] < lo ? lo : (pred[i] > hi ? hi : pred[i]);
        grad[i] = -inv_n * target[i] / p;
    }
#else
    for (size_t i = 0; i < n; i++) {
        float p = pred[i] < lo ? lo : (pred[i] > hi ? hi : pred[i]);
        grad[i] = -inv_n * target[i] / p;
    }
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_allclose_f32
// ---------------------------------------------------------------------------

BOAT_API bool boat_simd_allclose_f32(const float* a, const float* b, size_t n, float rtol,
                                     float atol) {
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        float max_ab = fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (!(diff <= atol + rtol * max_ab)) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// boat_simd_alloc / boat_simd_free
// ---------------------------------------------------------------------------

BOAT_API void* boat_simd_alloc(size_t size) {
    // Allocate with extra space for alignment offset
    size_t alloc_size = size + BOAT_SIMD_ALIGNMENT + sizeof(void*);
    unsigned char* raw = (unsigned char*)malloc(alloc_size);
    if (!raw) return NULL;

    // Store original pointer
    void** ptr = (void**)(raw + BOAT_SIMD_ALIGNMENT);
    // Align down to 32-byte boundary
    uintptr_t aligned = (uintptr_t)(ptr) & ~(uintptr_t)(BOAT_SIMD_ALIGNMENT - 1);
    // Store original pointer just before aligned region
    ((void**)aligned)[-1] = raw;
    return (void*)aligned;
}

BOAT_API void boat_simd_free(void* ptr) {
    if (!ptr) return;
    // Retrieve original pointer stored before aligned region
    void* raw = ((void**)ptr)[-1];
    free(raw);
}
