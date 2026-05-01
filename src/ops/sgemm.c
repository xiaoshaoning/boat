// sgemm.c - Tiled SGEMM with packing and SIMD micro-kernel
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/sgemm.h>
#include <boat/simd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Packing routines
// ---------------------------------------------------------------------------

// Pack A panel [i..i+mc, p..p+kc] into packed buffer:
//   packed_a[k * mr + row] = A[(i+row) * K + (p + k)]  for row in 0..mr-1
// So the layout is [kc][mr] — for each k, mr consecutive values (one per row).
static void pack_a_mr(int64_t mc, int64_t kc,
                       const float* A, int64_t ldA,
                       float* packed, int64_t mr)
{
    for (int64_t i = 0; i < mc; i += mr) {
        int64_t rows = mr;
        if (i + rows > mc) rows = mc - i;

        for (int64_t k = 0; k < kc; k++) {
            for (int64_t r = 0; r < rows; r++) {
                *packed++ = A[(i + r) * ldA + k];
            }
            for (int64_t r = rows; r < mr; r++) {
                *packed++ = 0.0f; // zero-pad for remainder rows
            }
        }
    }
}

// Pack B panel [p..p+kc, j..j+nc] into packed buffer:
//   packed_b[k * nr + col] = B[(p + k) * ldB + (j + col)]  for col in 0..nr-1
// So the layout is [kc][nr] — for each k, nr consecutive values.
static void pack_b_nr(int64_t nc, int64_t kc,
                       const float* B, int64_t ldB,
                       float* packed, int64_t nr)
{
    for (int64_t j = 0; j < nc; j += nr) {
        int64_t cols = nr;
        if (j + cols > nc) cols = nc - j;

        for (int64_t k = 0; k < kc; k++) {
            for (int64_t c = 0; c < cols; c++) {
                *packed++ = B[k * ldB + (j + c)];
            }
            for (int64_t c = cols; c < nr; c++) {
                *packed++ = 0.0f; // zero-pad for remainder cols
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Micro-kernel: compute a 4x8 tile of C += A * B
//   A_packed has shape [kc][4]  (interleaved rows)
//   B_packed has shape [kc][8]  (contiguous cols)
//   C has leading dimension ldC (row stride of output matrix)
// ---------------------------------------------------------------------------

#if BOAT_HAVE_AVX2

static void micro_4x8(int64_t kc,
                       const float* A_packed, const float* B_packed,
                       float* C, int64_t ldC)
{
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();

    for (int64_t k = 0; k < kc; k++) {
        __m256 b = _mm256_loadu_ps(B_packed);
        B_packed += 8;

        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[0]), b, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[1]), b, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[2]), b, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A_packed[3]), b, c3);
        A_packed += 4;
    }

    _mm256_storeu_ps(C,          _mm256_add_ps(c0, _mm256_loadu_ps(C)));
    _mm256_storeu_ps(C + ldC,    _mm256_add_ps(c1, _mm256_loadu_ps(C + ldC)));
    _mm256_storeu_ps(C + ldC * 2, _mm256_add_ps(c2, _mm256_loadu_ps(C + ldC * 2)));
    _mm256_storeu_ps(C + ldC * 3, _mm256_add_ps(c3, _mm256_loadu_ps(C + ldC * 3)));
}

#elif BOAT_HAVE_NEON

static void micro_4x8(int64_t kc,
                       const float* A_packed, const float* B_packed,
                       float* C, int64_t ldC)
{
    float32x4_t c0l = vdupq_n_f32(0.0f), c0h = vdupq_n_f32(0.0f);
    float32x4_t c1l = vdupq_n_f32(0.0f), c1h = vdupq_n_f32(0.0f);
    float32x4_t c2l = vdupq_n_f32(0.0f), c2h = vdupq_n_f32(0.0f);
    float32x4_t c3l = vdupq_n_f32(0.0f), c3h = vdupq_n_f32(0.0f);

    for (int64_t k = 0; k < kc; k++) {
        float32x4_t bl = vld1q_f32(B_packed);
        float32x4_t bh = vld1q_f32(B_packed + 4);
        B_packed += 8;

        float32x4_t a0 = vdupq_n_f32(A_packed[0]);
        float32x4_t a1 = vdupq_n_f32(A_packed[1]);
        float32x4_t a2 = vdupq_n_f32(A_packed[2]);
        float32x4_t a3 = vdupq_n_f32(A_packed[3]);
        A_packed += 4;

        c0l = vmlaq_f32(c0l, a0, bl); c0h = vmlaq_f32(c0h, a0, bh);
        c1l = vmlaq_f32(c1l, a1, bl); c1h = vmlaq_f32(c1h, a1, bh);
        c2l = vmlaq_f32(c2l, a2, bl); c2h = vmlaq_f32(c2h, a2, bh);
        c3l = vmlaq_f32(c3l, a3, bl); c3h = vmlaq_f32(c3h, a3, bh);
    }

    float* c0p = C;
    float* c1p = C + ldC;
    float* c2p = C + ldC * 2;
    float* c3p = C + ldC * 3;

    vst1q_f32(c0p,     vaddq_f32(c0l, vld1q_f32(c0p)));
    vst1q_f32(c0p + 4, vaddq_f32(c0h, vld1q_f32(c0p + 4)));
    vst1q_f32(c1p,     vaddq_f32(c1l, vld1q_f32(c1p)));
    vst1q_f32(c1p + 4, vaddq_f32(c1h, vld1q_f32(c1p + 4)));
    vst1q_f32(c2p,     vaddq_f32(c2l, vld1q_f32(c2p)));
    vst1q_f32(c2p + 4, vaddq_f32(c2h, vld1q_f32(c2p + 4)));
    vst1q_f32(c3p,     vaddq_f32(c3l, vld1q_f32(c3p)));
    vst1q_f32(c3p + 4, vaddq_f32(c3h, vld1q_f32(c3p + 4)));
}

#else
// Scalar fallback micro-kernel

static void micro_4x8(int64_t kc,
                       const float* A_packed, const float* B_packed,
                       float* C, int64_t ldC)
{
    float c00, c01, c02, c03, c04, c05, c06, c07;
    float c10, c11, c12, c13, c14, c15, c16, c17;
    float c20, c21, c22, c23, c24, c25, c26, c27;
    float c30, c31, c32, c33, c34, c35, c36, c37;

    c00 = C[0];   c01 = C[1];   c02 = C[2];   c03 = C[3];
    c04 = C[4];   c05 = C[5];   c06 = C[6];   c07 = C[7];
    c10 = C[ldC]; c11 = C[ldC+1]; c12 = C[ldC+2]; c13 = C[ldC+3];
    c14 = C[ldC+4]; c15 = C[ldC+5]; c16 = C[ldC+6]; c17 = C[ldC+7];
    c20 = C[ldC*2]; c21 = C[ldC*2+1]; c22 = C[ldC*2+2]; c23 = C[ldC*2+3];
    c24 = C[ldC*2+4]; c25 = C[ldC*2+5]; c26 = C[ldC*2+6]; c27 = C[ldC*2+7];
    c30 = C[ldC*3]; c31 = C[ldC*3+1]; c32 = C[ldC*3+2]; c33 = C[ldC*3+3];
    c34 = C[ldC*3+4]; c35 = C[ldC*3+5]; c36 = C[ldC*3+6]; c37 = C[ldC*3+7];

    for (int64_t k = 0; k < kc; k++) {
        float a0 = A_packed[0];
        float a1 = A_packed[1];
        float a2 = A_packed[2];
        float a3 = A_packed[3];
        A_packed += 4;

        float b0 = B_packed[0]; float b1 = B_packed[1];
        float b2 = B_packed[2]; float b3 = B_packed[3];
        float b4 = B_packed[4]; float b5 = B_packed[5];
        float b6 = B_packed[6]; float b7 = B_packed[7];
        B_packed += 8;

        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
        c04 += a0 * b4; c05 += a0 * b5; c06 += a0 * b6; c07 += a0 * b7;
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
        c14 += a1 * b4; c15 += a1 * b5; c16 += a1 * b6; c17 += a1 * b7;
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
        c24 += a2 * b4; c25 += a2 * b5; c26 += a2 * b6; c27 += a2 * b7;
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        c34 += a3 * b4; c35 += a3 * b5; c36 += a3 * b6; c37 += a3 * b7;
    }

    C[0] = c00; C[1] = c01; C[2] = c02; C[3] = c03;
    C[4] = c04; C[5] = c05; C[6] = c06; C[7] = c07;
    C[ldC] = c10; C[ldC+1] = c11; C[ldC+2] = c12; C[ldC+3] = c13;
    C[ldC+4] = c14; C[ldC+5] = c15; C[ldC+6] = c16; C[ldC+7] = c17;
    C[ldC*2] = c20; C[ldC*2+1] = c21; C[ldC*2+2] = c22; C[ldC*2+3] = c23;
    C[ldC*2+4] = c24; C[ldC*2+5] = c25; C[ldC*2+6] = c26; C[ldC*2+7] = c27;
    C[ldC*3] = c30; C[ldC*3+1] = c31; C[ldC*3+2] = c32; C[ldC*3+3] = c33;
    C[ldC*3+4] = c34; C[ldC*3+5] = c35; C[ldC*3+6] = c36; C[ldC*3+7] = c37;
}

#endif

// ---------------------------------------------------------------------------
// Handle remainder rows (< MR) after the main micro-kernel tiles
// ---------------------------------------------------------------------------

static void micro_remainder_rows(int64_t mr, int64_t nc, int64_t kc,
                                  const float* A_packed, const float* B_packed,
                                  float* C, int64_t ldC)
{
    for (int64_t r = 0; r < mr; r++) {
        for (int64_t c = 0; c < nc; c++) {
            float acc = C[r * ldC + c];
            for (int64_t k = 0; k < kc; k++) {
                acc += A_packed[k * 4 + r] * B_packed[k * 8 + c];
            }
            C[r * ldC + c] = acc;
        }
    }
}

static void micro_remainder_cols(int64_t mr, int64_t nc, int64_t kc,
                                  const float* A_packed, const float* B_packed,
                                  float* C, int64_t ldC)
{
    for (int64_t r = 0; r < mr; r++) {
        for (int64_t c = 0; c < nc; c++) {
            float acc = C[r * ldC + c];
            for (int64_t k = 0; k < kc; k++) {
                acc += A_packed[k * 4 + r] * B_packed[k * 8 + c];
            }
            C[r * ldC + c] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Process an MC x NC panel, iterating over micro-tiles
// ---------------------------------------------------------------------------

static void sgemm_panel(int64_t mc, int64_t nc, int64_t kc,
                         const float* A_packed, const float* B_packed,
                         float* C, int64_t ldC)
{
    int64_t mr = BOAT_SGEMM_MR;
    int64_t nr = BOAT_SGEMM_NR;

    for (int64_t i = 0; i < mc; i += mr) {
        int64_t rows = mc - i;
        int64_t packed_a_stride = kc * mr; // rows per A panel
        const float* A_panel = A_packed + (i / mr) * packed_a_stride;

        for (int64_t j = 0; j < nc; j += nr) {
            int64_t cols = nc - j;
            int64_t packed_b_stride = kc * nr;
            const float* B_panel = B_packed + (j / nr) * packed_b_stride;

            if (rows >= mr && cols >= nr) {
                micro_4x8(kc, A_panel, B_panel, C + i * ldC + j, ldC);
            } else if (rows >= mr) {
                micro_remainder_cols(mr, cols, kc,
                                     A_panel, B_panel, C + i * ldC + j, ldC);
            } else {
                micro_remainder_rows(rows, cols < nr ? cols : nr, kc,
                                     A_panel, B_panel, C + i * ldC + j, ldC);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// boat_sgemm: C[M][N] += A[M][K] * B[K][N]  (row-major, float32)
// ---------------------------------------------------------------------------

void boat_sgemm(int64_t M, int64_t N, int64_t K,
                const float* A, const float* B, float* C)
{
    if (M <= 0 || N <= 0 || K <= 0) return;

    int64_t mc = BOAT_SGEMM_MC;
    int64_t nc = BOAT_SGEMM_NC;
    int64_t kc = BOAT_SGEMM_KC;

    // For small matrices, use kc = K to avoid unnecessary packing overhead
    if (K < kc) kc = K;

    // Allocate packed buffers (unaligned OK — loadu handles it)
    size_t a_pack_size = (size_t)(mc * kc) * sizeof(float);
    size_t b_pack_size = (size_t)(kc * nc) * sizeof(float);
    float* a_packed = (float*)malloc(a_pack_size);
    float* b_packed = (float*)malloc(b_pack_size);
    if (!a_packed || !b_packed) {
        free(a_packed);
        free(b_packed);
        return;
    }

    // Outer tiling loop
    for (int64_t i = 0; i < M; i += mc) {
        int64_t mc_eff = (M - i < mc) ? (M - i) : mc;

        for (int64_t j = 0; j < N; j += nc) {
            int64_t nc_eff = (N - j < nc) ? (N - j) : nc;

            // Clear C panel (will accumulate over k panels)
            // memset is faster than element-wise zeroing for large panels
            for (int64_t ir = 0; ir < mc_eff; ir++) {
                memset(C + (i + ir) * N + j, 0, (size_t)nc_eff * sizeof(float));
            }

            for (int64_t p = 0; p < K; p += kc) {
                int64_t kc_eff = (K - p < kc) ? (K - p) : kc;

                // Pack B panel [p..p+kc_eff, j..j+nc_eff]
                pack_b_nr(nc_eff, kc_eff, B + p * N + j, N, b_packed, BOAT_SGEMM_NR);

                // Pack A panel [i..i+mc_eff, p..p+kc_eff] and run micro-kernel
                pack_a_mr(mc_eff, kc_eff, A + i * K + p, K, a_packed, BOAT_SGEMM_MR);

                sgemm_panel(mc_eff, nc_eff, kc_eff,
                            a_packed, b_packed, C + i * N + j, N);
            }
        }
    }

    free(a_packed);
    free(b_packed);
}

// ---------------------------------------------------------------------------
// boat_sgemm_batched: batched version with configurable strides
// ---------------------------------------------------------------------------

void boat_sgemm_batched(int64_t batch,
                         int64_t M, int64_t N, int64_t K,
                         const float* A, int64_t stride_a,
                         const float* B, int64_t stride_b,
                         float* C, int64_t stride_c)
{
    if (batch <= 0 || M <= 0 || N <= 0 || K <= 0) return;

    if (stride_a == 0) stride_a = M * K;
    if (stride_b == 0) stride_b = K * N;
    if (stride_c == 0) stride_c = M * N;

    for (int64_t b = 0; b < batch; b++) {
        boat_sgemm(M, N, K,
                   A + b * stride_a,
                   B + b * stride_b,
                   C + b * stride_c);
    }
}
