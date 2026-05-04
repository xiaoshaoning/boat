// bf16_conversion.cu — BF16 ↔ FP32 conversion CUDA kernels
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

// ---------------------------------------------------------------------------
// FP32 → BF16 conversion
// ---------------------------------------------------------------------------
__global__ void fp32_to_bf16_kernel(const float* __restrict__ in,
                                     __nv_bfloat16* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

void boat_cuda_fp32_to_bf16(const float* in, void* out, int n) {
    const int block = 256;
    unsigned int grid = (unsigned int)((n + block - 1) / block);
    fp32_to_bf16_kernel<<<grid, block>>>(in, (__nv_bfloat16*)out, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// BF16 → FP32 conversion
// ---------------------------------------------------------------------------
__global__ void bf16_to_fp32_kernel(const __nv_bfloat16* __restrict__ in,
                                     float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

void boat_cuda_bf16_to_fp32(const void* in, float* out, int n) {
    const int block = 256;
    unsigned int grid = (unsigned int)((n + block - 1) / block);
    bf16_to_fp32_kernel<<<grid, block>>>((const __nv_bfloat16*)in, out, n);
    CUDA_CHECK(cudaGetLastError());
}
