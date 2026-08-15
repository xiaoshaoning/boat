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

// Reduction: return min(a[0..n-1])
BOAT_API float boat_simd_min_reduce_f32(const float* a, size_t n);

// Reduction: return sum(a[0..n-1])
BOAT_API float boat_simd_sum_reduce_f32(const float* a, size_t n);

// 2D transpose of a contiguous [rows, cols] f32 matrix (row-major):
// dst[c * rows + r] = src[r * cols + c]. Tiled SIMD when available.
BOAT_API void boat_simd_transpose2d_f32(const float* src, float* dst, size_t rows, size_t cols);

// Transcendental / activation elementwise kernels (dst[i] = f(a[i])).
BOAT_API void boat_simd_exp_f32(const float* a, float* dst, size_t n);
BOAT_API void boat_simd_sigmoid_f32(const float* a, float* dst, size_t n);
BOAT_API void boat_simd_tanh_f32(const float* a, float* dst, size_t n);
BOAT_API void boat_simd_silu_f32(const float* a, float* dst, size_t n);
BOAT_API void boat_simd_gelu_f32(const float* a, float* dst, size_t n);

// Row-wise softmax over the last (contiguous) dim of a [rows, cols] matrix.
BOAT_API void boat_simd_softmax_f32(const float* a, float* dst, size_t rows, size_t cols);

// ---------------------------------------------------------------------------
// Activation-derivative kernels (fused: no intermediate temporaries).
// All compute dst[i] = dy[i] * d/dx f(x[i]); y holds the forward output.
// ---------------------------------------------------------------------------
BOAT_API void boat_simd_sigmoid_backward_f32(const float* dy, const float* y, float* dst,
                                             size_t n);
BOAT_API void boat_simd_tanh_backward_f32(const float* dy, const float* y, float* dst, size_t n);
BOAT_API void boat_simd_relu_backward_f32(const float* dy, const float* x, float* dst, size_t n);
BOAT_API void boat_simd_gelu_backward_f32(const float* dy, const float* x, float* dst, size_t n);

// Softmax/log-softmax Jacobian-vector products, row-wise over the last dim:
//   softmax:      dst_i = y_i * (dy_i - sum_k dy_k*y_k)
//   log-softmax:  dst_i = dy_i - exp(y_i) * sum_k dy_k
BOAT_API void boat_simd_softmax_backward_f32(const float* dy, const float* y, float* dst,
                                             size_t rows, size_t cols);
BOAT_API void boat_simd_log_softmax_backward_f32(const float* dy, const float* y, float* dst,
                                                 size_t rows, size_t cols);

// Fused loss backward kernels.
//   softmax-CE: grad = (softmax(logits) - onehot(labels)) * inv_batch
//   CE:         grad = -inv_n * target / clip(pred, epsilon)
BOAT_API void boat_simd_softmax_ce_backward_f32(const float* logits, const int64_t* labels,
                                                float* grad, size_t rows, size_t cols,
                                                float inv_batch);
BOAT_API void boat_simd_ce_backward_f32(const float* pred, const float* target, float* grad,
                                        size_t n, float inv_n, float epsilon);

// ---------------------------------------------------------------------------
// Elementwise binary / scalar kernels (dst[i] = a[i] op b[i], or a[i] op s).
// Aliasing is allowed (dst may equal a or b).
// ---------------------------------------------------------------------------
BOAT_API void boat_simd_add_f32(const float* a, const float* b, float* dst, size_t n);
BOAT_API void boat_simd_sub_f32(const float* a, const float* b, float* dst, size_t n);
BOAT_API void boat_simd_mul_f32(const float* a, const float* b, float* dst, size_t n);
BOAT_API void boat_simd_div_f32(const float* a, const float* b, float* dst, size_t n);
BOAT_API void boat_simd_add_scalar_f32(const float* a, float s, float* dst, size_t n);
BOAT_API void boat_simd_sub_scalar_f32(const float* a, float s, float* dst, size_t n);
BOAT_API void boat_simd_mul_scalar_f32(const float* a, float s, float* dst, size_t n);
BOAT_API void boat_simd_div_scalar_f32(const float* a, float s, float* dst, size_t n);
BOAT_API void boat_simd_abs_f32(const float* a, float* dst, size_t n);

// ---------------------------------------------------------------------------
// Normalization kernels (row-wise over the last, contiguous dim).
// ---------------------------------------------------------------------------
// mean[o] = sum_c a[o,c]/cols ; var[o] = sum_c a[o,c]^2/cols - mean[o]^2
BOAT_API void boat_simd_mean_var_f32(const float* a, float* mean, float* var, size_t rows,
                                     size_t cols);
// rms[o] = sqrt(sum_c a[o,c]^2/cols)
BOAT_API void boat_simd_rms_f32(const float* a, float* rms, size_t rows, size_t cols);
// out = (x - mean[o]) * inv_std[o] (*weight[c]? + bias[c]?); NULL weight/bias/mean = identity.
BOAT_API void boat_simd_norm_affine_f32(const float* x, const float* weight, const float* bias,
                                        float* out, size_t rows, size_t cols,
                                        const float* mean, const float* inv_std);
// LayerNorm backward: recomputes per-row stats internally, accumulates grad_weight/
// grad_bias (may be NULL) and writes grad_input.
BOAT_API void boat_simd_layernorm_backward_f32(const float* x, const float* dy,
                                               const float* gamma, float* dx, float* grad_weight,
                                               float* grad_bias, size_t rows, size_t cols,
                                               float eps);
// RMSNorm backward: same contract, rms-based.
BOAT_API void boat_simd_rmsnorm_backward_f32(const float* x, const float* dy,
                                             const float* gamma, float* dx, float* grad_weight,
                                             size_t rows, size_t cols, float eps);

#if BOAT_HAVE_AVX2
// 256-bit vector helpers (for consumers that need to fuse the math, e.g. the
// LSTM/GRU gate loops).
BOAT_API __m256 boat_simd_exp256(__m256 x);
BOAT_API __m256 boat_simd_sigmoid256(__m256 x);
BOAT_API __m256 boat_simd_tanh256(__m256 x);
#endif

// Check if two arrays are element-wise equal within tolerance
BOAT_API bool boat_simd_allclose_f32(const float* a, const float* b, size_t n, float rtol,
                                     float atol);

#ifdef __cplusplus
}
#endif

#endif // BOAT_SIMD_H
