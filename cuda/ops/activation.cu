// activation.cu - CUDA activation operations
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>
#include <boat/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// ---------------------------------------------------------------------------
// Softmax: one block per row
// Input/Output: [outer, axis_size, inner] flattened
// ---------------------------------------------------------------------------
__global__ void softmax_f32_kernel(const float* __restrict__ a,
                                    float* __restrict__ c,
                                    int64_t outer, int64_t axis_size, int64_t inner)
{
    extern __shared__ float shared[];
    float* sdata = shared;

    int64_t row = blockIdx.x;
    int64_t o = row / inner;
    int64_t i = row % inner;

    const float* row_a = a + o * axis_size * inner + i;
    float* row_c = c + o * axis_size * inner + i;

    int tid = threadIdx.x;

    // Find max
    float max_val = -FLT_MAX;
    for (int k = tid; k < axis_size; k += blockDim.x) {
        float v = row_a[k * inner];
        if (v > max_val) max_val = v;
    }
    sdata[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int k = tid; k < axis_size; k += blockDim.x) {
        float v = expf(row_a[k * inner] - row_max);
        row_c[k * inner] = v;
        sum += v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    // Normalize
    for (int k = tid; k < axis_size; k += blockDim.x) {
        row_c[k * inner] *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// LogSoftmax
// ---------------------------------------------------------------------------
__global__ void log_softmax_f32_kernel(const float* __restrict__ a,
                                        float* __restrict__ c,
                                        int64_t outer, int64_t axis_size, int64_t inner)
{
    extern __shared__ float shared[];
    float* sdata = shared;

    int64_t row = blockIdx.x;
    int64_t o = row / inner;
    int64_t i = row % inner;

    const float* row_a = a + o * axis_size * inner + i;
    float* row_c = c + o * axis_size * inner + i;
    int tid = threadIdx.x;

    // Max
    float max_val = -FLT_MAX;
    for (int k = tid; k < axis_size; k += blockDim.x) {
        float v = row_a[k * inner];
        if (v > max_val) max_val = v;
    }
    sdata[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Sum exp(x - max)
    float sum = 0.0f;
    for (int k = tid; k < axis_size; k += blockDim.x) {
        sum += expf(row_a[k * inner] - row_max);
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float log_sum = logf(sdata[0]);
    __syncthreads();

    for (int k = tid; k < axis_size; k += blockDim.x) {
        row_c[k * inner] = row_a[k * inner] - row_max - log_sum;
    }
}

// ---------------------------------------------------------------------------
// GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ---------------------------------------------------------------------------
__global__ void gelu_f32_kernel(const float* __restrict__ a,
                                 float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x3); // sqrt(2/pi) * (x + 0.044715 * x^3)
        c[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

extern "C" {

void boat_cuda_softmax_f32(const float* a, float* c,
                            int64_t outer, int64_t axis_size, int64_t inner)
{
    int64_t rows = outer * inner;
    if (rows <= 0) return;
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)rows;
    size_t shared_mem = block * sizeof(float);
    softmax_f32_kernel<<<grid, block, shared_mem>>>(a, c, outer, axis_size, inner);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_log_softmax_f32(const float* a, float* c,
                                int64_t outer, int64_t axis_size, int64_t inner)
{
    int64_t rows = outer * inner;
    if (rows <= 0) return;
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)rows;
    size_t shared_mem = block * sizeof(float);
    log_softmax_f32_kernel<<<grid, block, shared_mem>>>(a, c, outer, axis_size, inner);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_gelu_f32(const float* a, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    gelu_f32_kernel<<<grid, block>>>(a, c, n);
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
