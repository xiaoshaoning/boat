// arithmetic.cu - CUDA arithmetic operations
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>
#include <boat/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// ---------------------------------------------------------------------------
// Math kernels (float32)
// ---------------------------------------------------------------------------
__global__ void exp_f32_kernel(const float* __restrict__ a,
                                float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = expf(a[idx]);
}

__global__ void log_f32_kernel(const float* __restrict__ a,
                                float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = logf(a[idx]);
}

__global__ void sqrt_f32_kernel(const float* __restrict__ a,
                                 float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = sqrtf(a[idx]);
}

__global__ void rsqrt_f32_kernel(const float* __restrict__ a,
                                  float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = rsqrtf(a[idx]);
}

__global__ void neg_f32_kernel(const float* __restrict__ a,
                                float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = -a[idx];
}

__global__ void abs_f32_kernel(const float* __restrict__ a,
                                float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = fabsf(a[idx]);
}

__global__ void mod_f32_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = fmodf(a[idx], b[idx]);
}

__global__ void sub_scalar_f32_kernel(const float* __restrict__ a,
                                       float scalar,
                                       float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] - scalar;
}

__global__ void div_scalar_f32_kernel(const float* __restrict__ a,
                                       float scalar,
                                       float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] / scalar;
}

__global__ void fill_f32_kernel(float* __restrict__ c, float val, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = val;
}

__global__ void scale_f32_kernel(float* __restrict__ c, float scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] *= scalar;
}

__global__ void clamp_f32_kernel(const float* __restrict__ a,
                                  float* __restrict__ c,
                                  float lo, float hi, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = a[idx];
        c[idx] = v < lo ? lo : (v > hi ? hi : v);
    }
}

extern "C" {

void boat_cuda_exp_f32(const float* a, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    exp_f32_kernel<<<grid, block>>>(a, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_log_f32(const float* a, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    log_f32_kernel<<<grid, block>>>(a, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_sqrt_f32(const float* a, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    sqrt_f32_kernel<<<grid, block>>>(a, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_rsqrt_f32(const float* a, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    rsqrt_f32_kernel<<<grid, block>>>(a, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_neg_f32(const float* a, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    neg_f32_kernel<<<grid, block>>>(a, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_abs_f32(const float* a, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    abs_f32_kernel<<<grid, block>>>(a, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_mod_f32(const float* a, const float* b, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    mod_f32_kernel<<<grid, block>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_sub_scalar_f32(const float* a, float scalar, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    sub_scalar_f32_kernel<<<grid, block>>>(a, scalar, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_div_scalar_f32(const float* a, float scalar, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    div_scalar_f32_kernel<<<grid, block>>>(a, scalar, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_fill_f32(float* c, float val, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    fill_f32_kernel<<<grid, block>>>(c, val, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_scale_f32(float* c, float scalar, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    scale_f32_kernel<<<grid, block>>>(c, scalar, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_clamp_f32(const float* a, float* c, float lo, float hi, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    clamp_f32_kernel<<<grid, block>>>(a, c, lo, hi, n);
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
