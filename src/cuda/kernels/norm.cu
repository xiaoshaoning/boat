// norm.cu - CUDA normalization kernels
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
// x: [rows, cols], gamma/beta: [cols], y: [rows, cols]
// ---------------------------------------------------------------------------
__global__ void layernorm_forward_kernel(
    const float* x, const float* gamma, const float* beta, float* y,
    int64_t rows, int64_t cols, float eps)
{
    extern __shared__ float shared[];
    float* mean = shared;
    float* var = &shared[blockDim.x];

    int tid = threadIdx.x;
    int row = blockIdx.x;

    if (row >= rows) return;

    // Load data
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = x[row * cols + i];
        sum += val;
        sum2 += val * val;
    }
    mean[tid] = sum;
    var[tid] = sum2;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            mean[tid] += mean[tid + s];
            var[tid] += var[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float m = mean[0] / cols;
        float v = var[0] / cols - m * m;
        mean[0] = m;
        var[0] = v;
    }
    __syncthreads();

    float m = mean[0];
    float inv_std = rsqrtf(var[0] + eps);

    for (int i = tid; i < cols; i += blockDim.x) {
        int idx = row * cols + i;
        y[idx] = (x[idx] - m) * inv_std;
        if (gamma) y[idx] *= gamma[i];
        if (beta) y[idx] += beta[i];
    }
}

extern "C" void boat_cuda_layernorm_forward_f32(
    const float* x, const float* gamma, const float* beta, float* y,
    int64_t rows, int64_t cols, float eps)
{
    int block = 256;
    int grid = (int)rows;
    layernorm_forward_kernel<<<grid, block, 2 * block * sizeof(float)>>>(
        x, gamma, beta, y, rows, cols, eps);
}

// ---------------------------------------------------------------------------
// RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma
// ---------------------------------------------------------------------------
__global__ void rmsnorm_forward_kernel(
    const float* x, const float* gamma, float* y,
    int64_t rows, int64_t cols, float eps)
{
    extern __shared__ float shared[];
    float* sq_sum = shared;

    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = x[row * cols + i];
        sum += val * val;
    }
    sq_sum[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sq_sum[tid] += sq_sum[tid + s];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(sq_sum[0] / cols + eps);
    for (int i = tid; i < cols; i += blockDim.x) {
        int idx = row * cols + i;
        y[idx] = x[idx] * inv_rms;
        if (gamma) y[idx] *= gamma[i];
    }
}

extern "C" void boat_cuda_rmsnorm_forward_f32(
    const float* x, const float* gamma, float* y,
    int64_t rows, int64_t cols, float eps)
{
    int block = 256;
    int grid = (int)rows;
    rmsnorm_forward_kernel<<<grid, block, block * sizeof(float)>>>(
        x, gamma, y, rows, cols, eps);
}
