// sgemm.h - Tiled SGEMM micro-kernel API
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_SGEMM_H
#define BOAT_SGEMM_H

#include <stdint.h>
#include <stddef.h>
#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Tile size constants
// ---------------------------------------------------------------------------

// Outer tiling: panels sized for L2 cache (~256KB per panel: 256*256*4)
#define BOAT_SGEMM_MC 256
#define BOAT_SGEMM_NC 256
#define BOAT_SGEMM_KC 256

// Micro-kernel shape: 4 rows x 8 columns
// Fits 16 YMM registers (4 accum + 1 broadcast + 1 load) with AVX2
#define BOAT_SGEMM_MR 4
#define BOAT_SGEMM_NR 8

// ---------------------------------------------------------------------------
// SGEMM: C[M][N] += A[M][K] * B[K][N]   (all row-major, float32)
//
// Computes: C[i][j] += sum_{k=0..K-1} A[i][k] * B[k][j]
//
// All matrices are in row-major layout:
//   A has leading dimension K, B has leading dimension N, C has leading dim N.
// ---------------------------------------------------------------------------

BOAT_API void boat_sgemm(int64_t M, int64_t N, int64_t K,
                         const float* A, const float* B, float* C);

// Batched SGEMM: compute batch independent C = A * B
// Strides can be used for transposed or strided layouts.
// If stride_a/stride_b/stride_c == 0, defaults to M*K, K*N, M*N.
BOAT_API void boat_sgemm_batched(int64_t batch,
                                  int64_t M, int64_t N, int64_t K,
                                  const float* A, int64_t stride_a,
                                  const float* B, int64_t stride_b,
                                  float* C, int64_t stride_c);

#ifdef __cplusplus
}
#endif

#endif // BOAT_SGEMM_H
