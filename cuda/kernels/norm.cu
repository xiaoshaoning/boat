// norm.cu - CUDA normalization kernels (LayerNorm, RMSNorm)
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
// LayerNorm forward: y = (x - mean) / sqrt(var + eps) * gamma + beta
// One block per row, shared memory for mean + variance reduction
// ---------------------------------------------------------------------------
__global__ void layernorm_fwd_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int64_t cols, float eps)
{
    extern __shared__ float shared[];
    float* sdata = shared;

    int tid = threadIdx.x;
    int row = blockIdx.x;

    const float* row_x = x + row * cols;

    // Phase 1: compute mean
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += row_x[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / cols;
    __syncthreads();

    // Phase 2: compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = row_x[i] - mean;
        var_sum += diff * diff;
    }
    sdata[tid] = var_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float variance = sdata[0] / cols;
    float inv_std = rsqrtf(variance + eps);
    __syncthreads();

    // Phase 3: normalize and apply gamma/beta
    for (int i = tid; i < cols; i += blockDim.x) {
        float normalized = (row_x[i] - mean) * inv_std;
        y[row * cols + i] = normalized * gamma[i] + (beta ? beta[i] : 0.0f);
    }
}

// ---------------------------------------------------------------------------
// LayerNorm backward:
//   d_x = (d_y * gamma - (sum(d_y*gamma) + mean_result * sum(d_y*gamma * x_hat)) / cols
//          - x_hat * sum(d_y * gamma * x_hat) / cols) / (sigma + eps)
// Simplified: uses two shared-memory reductions
// ---------------------------------------------------------------------------
__global__ void layernorm_bwd_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,       // not used directly, use x_hat
    const float* __restrict__ gamma,
    const float* __restrict__ d_y,
    float* __restrict__ d_x,
    float* __restrict__ d_gamma,
    float* __restrict__ d_beta,
    int64_t cols, float eps)
{
    extern __shared__ float shared[];
    float* sdata = shared;

    int tid = threadIdx.x;
    int row = blockIdx.x;

    const float* row_x = x + row * cols;
    const float* row_dy = d_y + row * cols;

    // Compute mean and inv_std for this row (reuse from forward or recompute)
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sum += row_x[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / cols;
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = row_x[i] - mean;
        var_sum += diff * diff;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / cols + eps);
    __syncthreads();

    // Reduction: sum d_y * gamma, sum d_y * gamma * x_hat
    float sum_dy_gamma = 0.0f;
    float sum_dy_gamma_xhat = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float dy = row_dy[i];
        float g = gamma[i];
        float x_hat = (row_x[i] - mean) * inv_std;
        sum_dy_gamma += dy * g;
        sum_dy_gamma_xhat += dy * g * x_hat;
    }
    sdata[tid] = sum_dy_gamma;
    if (tid + blockDim.x < blockDim.x * 2) {
        sdata[tid + blockDim.x] = sum_dy_gamma_xhat;
    }
    __syncthreads();

    // Two reductions in parallel
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + s + blockDim.x];
        }
        __syncthreads();
    }
    float ds = sdata[0];
    float dx = sdata[blockDim.x];
    __syncthreads();

    // Compute d_x
    float norm = 1.0f / cols;
    for (int i = tid; i < cols; i += blockDim.x) {
        float x_hat_i = (row_x[i] - mean) * inv_std;
        float dy_gamma = row_dy[i] * gamma[i];
        float dx_i = (dy_gamma - (ds + dx * x_hat_i) * norm) * inv_std;
        d_x[row * cols + i] = dx_i;
    }

    // Compute d_gamma, d_beta (if requested)
    if (d_gamma && d_beta && tid == 0) {
        // Note: proper reduction across rows would be needed for full batch
        // This handles single-row case; caller is responsible for multi-row reduction
        float dg = 0.0f, db = 0.0f;
        for (int i = 0; i < cols; i++) {
            float x_hat_i = (row_x[i] - mean) * inv_std;
            dg += row_dy[i] * x_hat_i;
            db += row_dy[i];
        }
        // atomicAdd for multi-row (simplified — proper impl uses separate reduction)
        // For now, just write per-row partials
        if (d_gamma) atomicAdd(&d_gamma[row], dg);
        if (d_beta)  atomicAdd(&d_beta[row],  db);
    }
}

// ---------------------------------------------------------------------------
// RMSNorm forward: y = x / sqrt(mean(x^2) + eps) * gamma
// ---------------------------------------------------------------------------
__global__ void rmsnorm_fwd_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    int64_t cols, float eps)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const float* row_x = x + row * cols;

    // Compute sum(x^2)
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum_sq += row_x[i] * row_x[i];
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / cols + eps);
    __syncthreads();

    for (int i = tid; i < cols; i += blockDim.x) {
        y[row * cols + i] = row_x[i] * rms * (gamma ? gamma[i] : 1.0f);
    }
}

// ---------------------------------------------------------------------------
// RMSNorm backward
// ---------------------------------------------------------------------------
__global__ void rmsnorm_bwd_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ d_y,
    float* __restrict__ d_x,
    int64_t cols, float eps)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const float* row_x = x + row * cols;
    const float* row_dy = d_y + row * cols;

    // sum(x^2)
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sum_sq += row_x[i] * row_x[i];
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / cols + eps);
    float rms_cube = rms * rms * rms;
    float sum_sq_global = sdata[0];
    __syncthreads();

    // sum(d_y * gamma * x)
    float sum_dy_g_x = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = gamma ? gamma[i] : 1.0f;
        sum_dy_g_x += row_dy[i] * g * row_x[i];
    }
    sdata[tid] = sum_dy_g_x;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum_dy_g_x_global = sdata[0];
    __syncthreads();

    float norm = 1.0f / cols;
    for (int i = tid; i < cols; i += blockDim.x) {
        float g = gamma ? gamma[i] : 1.0f;
        float dx_i = rms * row_dy[i] * g
                   - rms_cube * norm * row_x[i] * sum_dy_g_x_global;
        d_x[row * cols + i] = dx_i;
    }
}

extern "C" {

void boat_cuda_layernorm_forward_f32(
    const float* x, const float* gamma, const float* beta,
    float* y, int64_t rows, int64_t cols, float eps)
{
    const unsigned int block = 256;
    dim3 grid((unsigned int)rows);
    size_t shared_mem = block * sizeof(float);
    layernorm_fwd_f32_kernel<<<grid, block, shared_mem>>>(x, gamma, beta, y, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_layernorm_backward_f32(
    const float* x, const float* y, const float* gamma,
    const float* d_y,
    float* d_x, float* d_gamma, float* d_beta,
    int64_t rows, int64_t cols, float eps)
{
    const unsigned int block = 256;
    dim3 grid((unsigned int)rows);
    size_t shared_mem = block * 2 * sizeof(float);
    layernorm_bwd_f32_kernel<<<grid, block, shared_mem>>>(
        x, y, gamma, d_y, d_x, d_gamma, d_beta, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_rmsnorm_forward_f32(
    const float* x, const float* gamma,
    float* y, int64_t rows, int64_t cols, float eps)
{
    const unsigned int block = 256;
    dim3 grid((unsigned int)rows);
    size_t shared_mem = block * sizeof(float);
    rmsnorm_fwd_f32_kernel<<<grid, block, shared_mem>>>(x, gamma, y, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_rmsnorm_backward_f32(
    const float* x, const float* gamma, const float* d_y,
    float* d_x, int64_t rows, int64_t cols, float eps)
{
    const unsigned int block = 256;
    dim3 grid((unsigned int)rows);
    size_t shared_mem = block * sizeof(float);
    rmsnorm_bwd_f32_kernel<<<grid, block, shared_mem>>>(x, gamma, d_y, d_x, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
