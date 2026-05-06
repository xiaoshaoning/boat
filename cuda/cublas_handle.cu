// cublas_handle.cu - cuBLAS handle manager and matmul wrapper
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <boat/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Error checking macro
// ---------------------------------------------------------------------------
#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t stat = call;                                         \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                \
        fprintf(stderr, "[cuBLAS] %s:%d: error %d\n",                  \
                __FILE__, __LINE__, (int)stat);                         \
        exit(1);                                                        \
    }                                                                   \
} while(0)

// ---------------------------------------------------------------------------
// Global cuBLAS handle (lazy initialization)
// ---------------------------------------------------------------------------
static cublasHandle_t g_cublas_handle = NULL;
static int g_cublas_initialized = 0;

extern "C" cublasHandle_t boat_cuda_get_cublas_handle(void) {
    if (!g_cublas_initialized) {
        CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
        g_cublas_initialized = 1;
    }
    return g_cublas_handle;
}

extern "C" void boat_cuda_cublas_destroy(void) {
    if (g_cublas_initialized && g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = NULL;
        g_cublas_initialized = 0;
    }
}

// ---------------------------------------------------------------------------
// cuBLAS Sgemm wrapper — row-major A[M,K] @ B[K,N] = C[M,N]
//
// cuBLAS is column-major: cublasSgemm computes C = alpha * op(A) * op(B) + beta * C
// where op(A) is MxK, op(B) is KxN, C is MxN (column-major).
//
// For row-major A[M,K], B[K,N], C[M,N], we need C^T = B^T @ A^T
// which is exactly cublasSgemm with op(A)=N, op(B)=N and swapped arguments:
//   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N)
// This computes C_colmajor(N,M) = B_colmajor(N,K) * A_colmajor(K,M) = (A_rowmajor * B_rowmajor)^T
// and since cuBLAS writes the result as C_colmajor(N,M) which is C_rowmajor(M,N) in memory,
// the result is correct.
// ---------------------------------------------------------------------------
void boat_cuda_matmul_f32_cublas(const float* A, const float* B, float* C,
                                  size_t M, size_t N, size_t K) {
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        B, (int)N,
        A, (int)K,
        &beta,
        C, (int)N));
}

// ---------------------------------------------------------------------------
// cuBLAS batched Sgemm — strided batched for grouped conv / multi-batch matmul
// ---------------------------------------------------------------------------
void boat_cuda_matmul_f32_strided_batched(const float* A, const float* B, float* C,
                                            size_t M, size_t N, size_t K,
                                            size_t batch_count,
                                            int64_t stride_A, int64_t stride_B, int64_t stride_C) {
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        B, (int)N, (long long)stride_B,
        A, (int)K, (long long)stride_A,
        &beta,
        C, (int)N, (long long)stride_C,
        (int)batch_count));
}
