// fused.cu - Batch norm and fused conv→bn→relu CUDA kernels
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
// Kernel 1: compute mean for batchnorm
// Input:  [N, C, H, W]
// Output: mean[C] — mean per channel over N*H*W elements
//
// One block per channel, parallel reduction over N*H*W
// ---------------------------------------------------------------------------
__global__ void batchnorm_mean_f32_kernel(const float* __restrict__ input,
                                           float* __restrict__ mean,
                                           size_t N, size_t C, size_t H, size_t W) {
    size_t c = blockIdx.x;
    if (c >= C) return;

    size_t hw_stride = H * W;
    size_t spatial = N * hw_stride;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    size_t chw = c * hw_stride;

    // Each thread sums over a strided slice of the channel's data (NCHW layout)
    float sum = 0.0f;
    for (size_t i = tid; i < spatial; i += stride) {
        size_t n = i / hw_stride;
        size_t hw = i % hw_stride;
        sum += input[n * C * hw_stride + chw + hw];
    }

    // Shared memory reduction
    extern __shared__ float sdata_mean[];
    sdata_mean[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata_mean[tid] += sdata_mean[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        mean[c] = sdata_mean[0] / (float)spatial;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: compute variance and normalize (one pass)
// Input:  [N, C, H, W], mean[C], gamma[C], beta[C]
// Output: [N, C, H, W] — normalized output
//
// First computes variance using the pre-computed mean, then normalizes.
// Since we can't compute var without mean, and we need var before normalizing,
// we use shared memory: first pass computes var across threads, second pass normalizes.
// ---------------------------------------------------------------------------
__global__ void batchnorm_norm_f32_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           const float* __restrict__ mean,
                                           const float* __restrict__ var,
                                           const float* __restrict__ gamma,
                                           const float* __restrict__ beta,
                                           size_t N, size_t C, size_t H, size_t W,
                                           float eps) {
    size_t c = blockIdx.x;
    if (c >= C) return;

    size_t hw_stride = H * W;
    size_t spatial = N * hw_stride;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    size_t chw = c * hw_stride;

    float mu = mean[c];
    float inv_std = 1.0f / sqrtf(var[c] + eps);
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;

    // Normalize each element assigned to this thread (NCHW layout)
    for (size_t i = tid; i < spatial; i += stride) {
        size_t n = i / hw_stride;
        size_t hw = i % hw_stride;
        size_t idx = n * C * hw_stride + chw + hw;
        output[idx] = g * (input[idx] - mu) * inv_std + b;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: fused bn+relu (computes var then normalizes + applies relu)
//
// Since we need var which requires mean which is already pre-computed,
// this kernel only does the normalization + relu in one pass.
// The mean and var must be computed before calling this kernel.
//
// output = relu(gamma * (input - mean) / sqrt(var + eps) + beta)
// ---------------------------------------------------------------------------
__global__ void fused_bn_relu_f32_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ beta,
                                          const float* __restrict__ mean,
                                          const float* __restrict__ var,
                                          size_t N, size_t C, size_t H, size_t W,
                                          float eps) {
    size_t c = blockIdx.x;
    if (c >= C) return;

    size_t hw_stride = H * W;
    size_t spatial = N * hw_stride;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    size_t chw = c * hw_stride;

    float mu = mean[c];
    float inv_std = 1.0f / sqrtf(var[c] + eps);
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;

    for (size_t i = tid; i < spatial; i += stride) {
        size_t n = i / hw_stride;
        size_t hw = i % hw_stride;
        size_t idx = n * C * hw_stride + chw + hw;
        float x = g * (input[idx] - mu) * inv_std + b;
        output[idx] = x > 0.0f ? x : 0.0f;  // ReLU
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: compute both mean and variance in a single pass (two-stage reduction)
// Used for the standalone batchnorm forward when var is not pre-computed.
//
// One block per channel, first pass in shared memory computes sum(x) and sum(x^2),
// then mean = sum(x)/N, var = sum(x^2)/N - mean^2.
// ---------------------------------------------------------------------------
__global__ void batchnorm_mean_var_f32_kernel(const float* __restrict__ input,
                                               float* __restrict__ mean,
                                               float* __restrict__ var,
                                               size_t N, size_t C, size_t H, size_t W) {
    size_t c = blockIdx.x;
    if (c >= C) return;

    size_t hw_stride = H * W;
    size_t spatial = N * hw_stride;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    size_t chw = c * hw_stride;

    // Each thread accumulates sum and sum of squares over NCHW data
    float sum = 0.0f, sum_sq = 0.0f;
    for (size_t i = tid; i < spatial; i += stride) {
        size_t n = i / hw_stride;
        size_t hw = i % hw_stride;
        float val = input[n * C * hw_stride + chw + hw];
        sum += val;
        sum_sq += val * val;
    }

    // Shared memory reduction: interleave sum and sum_sq
    extern __shared__ float sdata_mv[];
    size_t shared_idx = tid * 2;
    size_t shared_size = blockDim.x * 2;
    if (shared_idx + 1 < shared_size) {
        sdata_mv[shared_idx] = sum;
        sdata_mv[shared_idx + 1] = sum_sq;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            size_t a = tid * 2, b = (tid + s) * 2;
            sdata_mv[a] += sdata_mv[b];
            sdata_mv[a + 1] += sdata_mv[b + 1];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = sdata_mv[0];
        float total_sum_sq = sdata_mv[1];
        float mu = total_sum / (float)spatial;
        mean[c] = mu;
        // E[X^2] - E[X]^2, with max(0) for numerical safety
        float variance = total_sum_sq / (float)spatial - mu * mu;
        var[c] = fmaxf(variance, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
extern "C" {

void boat_cuda_batchnorm_forward_f32(const float* input, float* output,
                                      const float* gamma, const float* beta,
                                      float* mean, float* var,
                                      size_t N, size_t C, size_t H, size_t W,
                                      float eps) {
    if (C == 0) return;

    const int block = 256;

    // Step 1: compute mean + var in one kernel (two-moment estimation)
    size_t smem_mv = block * 2 * sizeof(float);
    batchnorm_mean_var_f32_kernel<<<C, block, smem_mv>>>(input, mean, var, N, C, H, W);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: normalize using computed mean and var
    batchnorm_norm_f32_kernel<<<C, block>>>(input, output, mean, var,
                                             gamma, beta, N, C, H, W, eps);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_fused_bn_relu_f32(const float* input, float* output,
                                   const float* gamma, const float* beta,
                                   const float* mean, const float* var,
                                   size_t N, size_t C, size_t H, size_t W,
                                   float eps) {
    if (C == 0) return;

    const int block = 256;
    fused_bn_relu_f32_kernel<<<C, block>>>(input, output, gamma, beta,
                                            mean, var, N, C, H, W, eps);
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
