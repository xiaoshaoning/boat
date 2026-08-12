// simd.h - SIMD architecture detection and micro-kernel declarations
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_SIMD_H
#define BOAT_SIMD_H

#include "export.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Compile-time SIMD architecture detection
// ---------------------------------------------------------------------------

// Detect AVX2
#if defined(__AVX2__)
    #define BOAT_HAVE_AVX2 1
    #include <immintrin.h>
#else
    #define BOAT_HAVE_AVX2 0
#endif

// Detect SSE4.1 (fallback for older x86)
#if defined(__SSE4_1__) || defined(__SSE4_2__) || (defined(_MSC_VER) && !BOAT_HAVE_AVX2)
    #define BOAT_HAVE_SSE41 1
    #if !BOAT_HAVE_AVX2
        #include <smmintrin.h>
    #endif
#else
    #define BOAT_HAVE_SSE41 0
#endif

// Detect NEON (ARM)
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
    #define BOAT_HAVE_NEON 1
    #include <arm_neon.h>
#else
    #define BOAT_HAVE_NEON 0
#endif

// FMA detection: MSVC enables FMA with /arch:AVX2; GCC/Clang need -mfma
#if BOAT_HAVE_AVX2 && (defined(_MSC_VER) || defined(__FMA__))
    #define BOAT_HAVE_FMA 1
#else
    #define BOAT_HAVE_FMA 0
#endif

#define BOAT_SIMD_ALIGNMENT 32

// ---------------------------------------------------------------------------
// Aligned allocation helpers
// ---------------------------------------------------------------------------

BOAT_API void* boat_simd_alloc(size_t size);
BOAT_API void boat_simd_free(void* ptr);

// ---------------------------------------------------------------------------
// SIMD micro-kernel declarations (implemented in simd_kernels.c)
// ---------------------------------------------------------------------------

// Elementwise: dst[i] = a[i] + b[i]  (all float32, n elements)
BOAT_API void boat_simd_add_f32(const float* a, const float* b, float* dst, size_t n);

// Elementwise: dst[i] = a[i] * b[i]
BOAT_API void boat_simd_mul_f32(const float* a, const float* b, float* dst, size_t n);

// Elementwise: dst[i] = a[i] * scalar
BOAT_API void boat_simd_mul_scalar_f32(const float* a, float scalar, float* dst, size_t n);

// Activation: dst[i] = max(a[i], 0)
BOAT_API void boat_simd_relu_f32(const float* a, float* dst, size_t n);

// Reduction: return max(a[0..n-1])
BOAT_API float boat_simd_max_reduce_f32(const float* a, size_t n);

// Reduction: return sum(a[0..n-1])
BOAT_API float boat_simd_sum_reduce_f32(const float* a, size_t n);

// Check if two arrays are element-wise equal within tolerance
BOAT_API bool boat_simd_allclose_f32(const float* a, const float* b, size_t n,
                             float rtol, float atol);

#ifdef __cplusplus
}
#endif

#endif // BOAT_SIMD_H
