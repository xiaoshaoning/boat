// dense.cu - Dense layer CUDA kernels
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

// ---------------------------------------------------------------------------
// Kernel 1: simple element-per-thread dense forward with fused bias
// Each thread computes one output element C[row, col] = dot(row of A, col of B) + bias
// Block: 1D (256 threads), Grid: 2D (ceil(O/256), B)
// ---------------------------------------------------------------------------
__global__ void dense_forward_f32_kernel(const float* __restrict__ input,  // [B, I]
                                          const float* __restrict__ weight, // [I, O]
                                          const float* __restrict__ bias,   // [O]
                                          float* __restrict__ output,       // [B, O]
                                          size_t B, size_t I, size_t O) {
    size_t row = blockIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B || col >= O) return;

    float sum = 0.0f;
    for (size_t k = 0; k < I; k++) {
        sum += input[row * I + k] * weight[k * O + col];
    }
    output[row * O + col] = sum + (bias ? bias[col] : 0.0f);
}

// ---------------------------------------------------------------------------
// Kernel 2: warp-level reduction dense forward
// Each warp (32 threads) collectively computes ONE output element.
// Thread t loads input[row, t + k*32] and weight[t + k*32, col].
// Partial products are accumulated then warp-reduced via __shfl_xor_sync.
// Lane 0 writes the final result.
//
// This kernel demonstrates warp-level primitives at the cost of lower
 // throughput per thread (32 threads → 1 output). Useful for moderate sizes
// where cuBLAS launch overhead dominates.
// ---------------------------------------------------------------------------
__global__ void dense_warp_f32_kernel(const float* __restrict__ input,  // [B, I]
                                       const float* __restrict__ weight, // [I, O]
                                       const float* __restrict__ bias,   // [O]
                                       float* __restrict__ output,       // [B, O]
                                       size_t B, size_t I, size_t O) {
    // Each block has N_WARPS warps, each warp = 32 threads
    // warp_id = threadIdx.x / 32, lane = threadIdx.x % 32
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x / 32;

    size_t row = blockIdx.y * warps_per_block + warp_id;
    size_t col = blockIdx.x;
    if (row >= B || col >= O) return;

    // Each thread computes a partial dot product over a strided slice of I
    float partial = 0.0f;
    for (size_t k = lane; k < I; k += 32) {
        partial += input[row * I + k] * weight[k * O + col];
    }

    // Warp-level reduction via butterfly shuffle
    // After each step, the result is duplicated — all lanes have the sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
    }

    // Lane 0 writes (all lanes now hold the same sum)
    if (lane == 0) {
        output[row * O + (blockIdx.x)] = partial + (bias ? bias[blockIdx.x] : 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: element-wise bias addition (used after cuBLAS matmul)
// output[B, O] += bias[O]
// ---------------------------------------------------------------------------
__global__ void add_bias_f32_kernel(float* __restrict__ output,
                                     const float* __restrict__ bias,
                                     size_t B, size_t O) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * O;
    if (idx >= total) return;
    size_t col = idx % O;
    output[idx] += bias[col];
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
extern "C" {

void boat_cuda_dense_forward_f32(const float* input, const float* weight,
                                  const float* bias, float* output,
                                  size_t B, size_t I, size_t O) {
    // Use the simple element-per-thread kernel (practical choice)
    // For large matrices, prefer boat_cuda_matmul_f32_cublas + boat_cuda_add_bias_f32
    const int block = 256;
    dim3 grid((O + block - 1) / block, (unsigned int)B);
    dense_forward_f32_kernel<<<grid, block>>>(input, weight, bias, output, B, I, O);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_dense_forward_warp_f32(const float* input, const float* weight,
                                       const float* bias, float* output,
                                       size_t B, size_t I, size_t O) {
    // Warp-level kernel: 4 warps per block, each warp computes one output row
    const int warps_per_block = 4;
    const int block = warps_per_block * 32;
    dim3 grid((unsigned int)O, (unsigned int)((B + warps_per_block - 1) / warps_per_block));
    dense_warp_f32_kernel<<<grid, block>>>(input, weight, bias, output, B, I, O);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_add_bias_f32(const float* input, const float* bias,
                             float* output, size_t B, size_t O) {
    // Copy input to output first if they differ
    if (input != output) {
        size_t nbytes = B * O * sizeof(float);
        CUDA_CHECK(cudaMemcpy(output, input, nbytes, cudaMemcpyDeviceToDevice));
    }
    // Add bias
    const int block = 256;
    unsigned int grid = (unsigned int)((B * O + block - 1) / block);
    add_bias_f32_kernel<<<grid, block>>>(output, bias, B, O);
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
