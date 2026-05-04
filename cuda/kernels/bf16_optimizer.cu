// bf16_optimizer.cu — BF16 mixed-precision optimizer CUDA kernels
// BF16 weights with FP32 optimizer state and FP32 gradients
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
// SGD update — BF16 params, FP32 gradients
// param -= lr * grad
// ---------------------------------------------------------------------------
__global__ void sgd_update_bf16_kernel(__nv_bfloat16* __restrict__ param,
                                        const float* __restrict__ grad,
                                        float lr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float p = __bfloat162float(param[idx]);
    param[idx] = __float2bfloat16(p - lr * grad[idx]);
}

void boat_cuda_sgd_update_bf16(void* param, const float* grad,
                                 float lr, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    sgd_update_bf16_kernel<<<grid, block>>>((__nv_bfloat16*)param, grad, lr, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// SGD with momentum — BF16 params, FP32 velocity, FP32 gradients
// Standard:  v = momentum * v + grad;  param -= lr * v
// Nesterov:  v = momentum * v + grad;  param -= lr * (grad + momentum * v)
// ---------------------------------------------------------------------------
__global__ void sgd_momentum_bf16_kernel(__nv_bfloat16* __restrict__ param,
                                          const float* __restrict__ grad,
                                          float* __restrict__ velocity,
                                          float lr, float momentum,
                                          bool use_nesterov, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = grad[idx];
    if (use_nesterov) {
        float v_prev = velocity[idx];
        velocity[idx] = momentum * v_prev + g;
        float p = __bfloat162float(param[idx]);
        param[idx] = __float2bfloat16(p - lr * (g + momentum * velocity[idx]));
    } else {
        velocity[idx] = momentum * velocity[idx] + g;
        float p = __bfloat162float(param[idx]);
        param[idx] = __float2bfloat16(p - lr * velocity[idx]);
    }
}

void boat_cuda_sgd_momentum_bf16(void* param, const float* grad,
                                  float* velocity, float lr, float momentum,
                                  bool use_nesterov, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    sgd_momentum_bf16_kernel<<<grid, block>>>((__nv_bfloat16*)param, grad, velocity, lr,
                                               momentum, use_nesterov, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Adam update — BF16 params, FP32 m/v state, FP32 gradients
// m = beta1 * m + (1 - beta1) * grad
// v = beta2 * v + (1 - beta2) * grad^2
// param -= lr * m_hat / (sqrt(v_hat) + eps)
// ---------------------------------------------------------------------------
__global__ void adam_update_bf16_kernel(__nv_bfloat16* __restrict__ param,
                                         const float* __restrict__ grad,
                                         float* __restrict__ m,
                                         float* __restrict__ v,
                                         float lr, float beta1, float beta2,
                                         float beta1_t, float beta2_t, float eps,
                                         size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = grad[idx];
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
    float m_hat = m[idx] / (1.0f - beta1_t);
    float v_hat = v[idx] / (1.0f - beta2_t);
    float p = __bfloat162float(param[idx]);
    param[idx] = __float2bfloat16(p - lr * m_hat / (sqrtf(v_hat) + eps));
}

void boat_cuda_adam_update_bf16(void* param, const float* grad,
                                 float* m, float* v,
                                 float lr, float beta1, float beta2,
                                 float beta1_t, float beta2_t, float eps,
                                 size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    adam_update_bf16_kernel<<<grid, block>>>((__nv_bfloat16*)param, grad, m, v,
                                              lr, beta1, beta2,
                                              beta1_t, beta2_t, eps, n);
    CUDA_CHECK(cudaGetLastError());
}
