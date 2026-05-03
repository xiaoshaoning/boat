// optimizer.cu - Optimizer CUDA kernel implementations
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>
#include <boat/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Error checking helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// ---------------------------------------------------------------------------
// SGD update kernels (float32)
// ---------------------------------------------------------------------------

// Vanilla SGD: param -= lr * grad
__global__ void sgd_update_f32_kernel(float* __restrict__ param,
                                       const float* __restrict__ grad,
                                       float lr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) param[idx] -= lr * grad[idx];
}

// SGD with momentum
__global__ void sgd_momentum_f32_kernel(float* __restrict__ param,
                                         const float* __restrict__ grad,
                                         float* __restrict__ velocity,
                                         float lr, float momentum,
                                         bool use_nesterov, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = grad[idx];
    if (use_nesterov) {
        // Nesterov: v = momentum * v + g; param -= lr * (g + momentum * v)
        float v_prev = velocity[idx];
        velocity[idx] = momentum * v_prev + g;
        param[idx] -= lr * (g + momentum * velocity[idx]);
    } else {
        // Standard: v = momentum * v + g; param -= lr * v
        velocity[idx] = momentum * velocity[idx] + g;
        param[idx] -= lr * velocity[idx];
    }
}

void boat_cuda_sgd_update_f32(float* param, const float* grad,
                               float lr, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    sgd_update_f32_kernel<<<grid, block>>>(param, grad, lr, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_sgd_momentum_f32(float* param, const float* grad,
                                 float* velocity, float lr, float momentum,
                                 bool use_nesterov, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    sgd_momentum_f32_kernel<<<grid, block>>>(param, grad, velocity, lr, momentum, use_nesterov, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Adam update kernel (float32)
// ---------------------------------------------------------------------------
__global__ void adam_update_f32_kernel(float* __restrict__ param,
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
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

void boat_cuda_adam_update_f32(float* param, const float* grad,
                                float* m, float* v,
                                float lr, float beta1, float beta2,
                                float beta1_t, float beta2_t, float eps,
                                size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    adam_update_f32_kernel<<<grid, block>>>(param, grad, m, v, lr, beta1, beta2,
                                             beta1_t, beta2_t, eps, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Loss backward kernels (float32)
// ---------------------------------------------------------------------------

// MSE backward: grad = 2 * (pred - target) / n
__global__ void mse_backward_f32_kernel(const float* __restrict__ pred,
                                         const float* __restrict__ target,
                                         float* __restrict__ grad, float scale,
                                         size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad[idx] = 2.0f * (pred[idx] - target[idx]) * scale;
}

void boat_cuda_mse_backward_f32(const float* pred, const float* target,
                                 float* grad, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    // scale = 1.0f / n
    mse_backward_f32_kernel<<<grid, block>>>(pred, target, grad, 1.0f / n, n);
    CUDA_CHECK(cudaGetLastError());
}

// Cross-entropy backward: grad = (pred - target) (softmax gradient)
__global__ void cross_entropy_backward_f32_kernel(const float* __restrict__ pred,
                                                    const float* __restrict__ target,
                                                    float* __restrict__ grad,
                                                    size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad[idx] = pred[idx] - target[idx];
}

void boat_cuda_cross_entropy_backward_f32(const float* pred, const float* target,
                                           float* grad, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    cross_entropy_backward_f32_kernel<<<grid, block>>>(pred, target, grad, n);
    CUDA_CHECK(cudaGetLastError());
}
