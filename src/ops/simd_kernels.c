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

BOAT_API void boat_simd_add_f32(const float* a, const float* b, float* dst, size_t n)
{
#if BOAT_HAVE_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vd = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++) dst[i] = a[i] + b[i];
#elif BOAT_HAVE_NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vd = vaddq_f32(va, vb);
        vst1q_f32(dst + i, vd);
    }
    for (; i < n; i++) dst[i] = a[i] + b[i];
#else
    for (size_t i = 0; i < n; i++) dst[i] = a[i] + b[i];
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_mul_f32
// ---------------------------------------------------------------------------

BOAT_API void boat_simd_mul_f32(const float* a, const float* b, float* dst, size_t n)
{
#if BOAT_HAVE_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vd = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++) dst[i] = a[i] * b[i];
#elif BOAT_HAVE_NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vd = vmulq_f32(va, vb);
        vst1q_f32(dst + i, vd);
    }
    for (; i < n; i++) dst[i] = a[i] * b[i];
#else
    for (size_t i = 0; i < n; i++) dst[i] = a[i] * b[i];
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_mul_scalar_f32
// ---------------------------------------------------------------------------

BOAT_API void boat_simd_mul_scalar_f32(const float* a, float scalar, float* dst, size_t n)
{
#if BOAT_HAVE_AVX2
    __m256 vs = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vd = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++) dst[i] = a[i] * scalar;
#elif BOAT_HAVE_NEON
    float32x4_t vs = vdupq_n_f32(scalar);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vd = vmulq_f32(va, vs);
        vst1q_f32(dst + i, vd);
    }
    for (; i < n; i++) dst[i] = a[i] * scalar;
#else
    for (size_t i = 0; i < n; i++) dst[i] = a[i] * scalar;
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_relu_f32
// ---------------------------------------------------------------------------

BOAT_API void boat_simd_relu_f32(const float* a, float* dst, size_t n)
{
#if BOAT_HAVE_AVX2
    __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vd = _mm256_max_ps(va, vzero);
        _mm256_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++) dst[i] = a[i] > 0 ? a[i] : 0.0f;
#elif BOAT_HAVE_NEON
    float32x4_t vzero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vd = vmaxq_f32(va, vzero);
        vst1q_f32(dst + i, vd);
    }
    for (; i < n; i++) dst[i] = a[i] > 0 ? a[i] : 0.0f;
#else
    for (size_t i = 0; i < n; i++) dst[i] = a[i] > 0 ? a[i] : 0.0f;
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_max_reduce_f32
// ---------------------------------------------------------------------------

BOAT_API float boat_simd_max_reduce_f32(const float* a, size_t n)
{
    if (n == 0) return 0.0f;
#if BOAT_HAVE_AVX2
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
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2,3,0,1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1,0,3,2)));
    float result = _mm_cvtss_f32(m);
    for (; i < n; i++) {
        if (a[i] > result) result = a[i];
    }
    return result;
#elif BOAT_HAVE_NEON
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
// boat_simd_sum_reduce_f32
// ---------------------------------------------------------------------------

BOAT_API float boat_simd_sum_reduce_f32(const float* a, size_t n)
{
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
    for (; i < n; i++) result += a[i];
    return result;
#elif BOAT_HAVE_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vsum = vaddq_f32(vsum, va);
    }
    float result = vaddvq_f32(vsum);
    for (; i < n; i++) result += a[i];
    return result;
#else
    float result = 0.0f;
    for (size_t i = 0; i < n; i++) result += a[i];
    return result;
#endif
}

// ---------------------------------------------------------------------------
// boat_simd_allclose_f32
// ---------------------------------------------------------------------------

BOAT_API bool boat_simd_allclose_f32(const float* a, const float* b, size_t n,
                             float rtol, float atol)
{
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

BOAT_API void* boat_simd_alloc(size_t size)
{
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

BOAT_API void boat_simd_free(void* ptr)
{
    if (!ptr) return;
    // Retrieve original pointer stored before aligned region
    void* raw = ((void**)ptr)[-1];
    free(raw);
}
