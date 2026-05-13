// linear.cu - CUDA linear algebra kernels (cuBLAS wrappers)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Forward declaration
extern "C" cublasHandle_t boat_cuda_cublas_get_handle(void);

// ---------------------------------------------------------------------------
// Matmul: C = A @ B, where A[M,K], B[K,N], C[M,N]
// cuBLAS uses column-major, so we compute C^T = B^T @ A^T
// which means C(M,N) = alpha * A(M,K) * B(K,N) + beta * C(M,N)
// In cuBLAS column-major: C = alpha * B^T * A^T + beta * C
// But if we use CUBLAS_OP_N for both: C[N,M] = A[N,K] * B[K,M]
// Better to use row-major interpretation via cuBLAS:
// cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N)
// This computes C[N,M] = B[N,K] * A[K,M] which in row-major is C[M,N] = A[M,K] * B[K,N]
// =========================================================================
extern "C" void boat_cuda_matmul_f32(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K)
{
    cublasHandle_t handle = boat_cuda_cublas_get_handle();
    if (!handle) return;

    float alpha = 1.0f, beta = 0.0f;
    // row-major: C[M,N] = A[M,K] @ B[K,N]
    // cuBLAS col-major: C^T = B^T @ A^T  =>  C[N,M] = B[N,K] * A[K,M]
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        B, (int)N,
        A, (int)K,
        &beta,
        C, (int)N);
}

// Strided-batched matmul
extern "C" void boat_cuda_matmul_f32_strided_batched(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K,
    size_t batch_count,
    int64_t stride_A, int64_t stride_B, int64_t stride_C)
{
    cublasHandle_t handle = boat_cuda_cublas_get_handle();
    if (!handle) return;

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        B, (int)N, (int)stride_B,
        A, (int)K, (int)stride_A,
        &beta,
        C, (int)N, (int)stride_C,
        (int)batch_count);
}

// cuBLAS matmul (alias for clarity)
extern "C" void boat_cuda_matmul_f32_cublas(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K)
{
    boat_cuda_matmul_f32(A, B, C, M, N, K);
}
