// fp8_matmul.cu — FP8 matmul kernel (manual decode → FP32 compute)
// C[M,N] = A[M,K] @ B[K,N]  with FP8 inputs and FP32 output
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>
#include <boat/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// FP8 E4M3 → FP32 device helper
__device__ inline float fp8e4m3_to_float(unsigned char v) {
    unsigned int sign = (unsigned int)(v >> 7) << 31;
    int exp = (int)((v >> 3) & 0x0F);
    unsigned int mant = (unsigned int)(v & 0x07);
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        unsigned int biased, mant_bits;
        if (mant >= 4) { biased = 120; mant_bits = (mant - 4) << 21; }
        else if (mant >= 2) { biased = 119; mant_bits = (mant - 2) << 22; }
        else { biased = 118; mant_bits = 0; }
        unsigned int fp32_bits = sign | (biased << 23) | mant_bits;
        float result;
        memcpy(&result, &fp32_bits, sizeof(result));
        return result;
    }
    if (exp == 15 && mant == 7) {
        unsigned int inf_bits = sign | 0x7F800000u;
        float result;
        memcpy(&result, &inf_bits, sizeof(result));
        return result;
    }
    unsigned int biased = (unsigned int)(exp + 120);
    unsigned int fp32_bits = sign | (biased << 23) | (mant << 20);
    float result;
    memcpy(&result, &fp32_bits, sizeof(result));
    return result;
}

// ---------------------------------------------------------------------------
// FP8 matmul: C[M,N] = A[M,K] @ B[K,N]  — each thread computes one element
// ---------------------------------------------------------------------------
__global__ void fp8_matmul_kernel(const unsigned char* __restrict__ A,
                                   const unsigned char* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += fp8e4m3_to_float(A[row * K + k]) * fp8e4m3_to_float(B[k * N + col]);
    C[row * N + col] = sum;
}

void boat_cuda_matmul_fp8_cublas(const void* A, const void* B, float* C,
                                  int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    fp8_matmul_kernel<<<grid, block>>>((const unsigned char*)A,
                                        (const unsigned char*)B,
                                        C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
