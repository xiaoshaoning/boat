// activation.cu - CUDA activation kernels
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Softmax: across axis_size dim (typically last dim)
// input: [outer, axis_size, inner], output: same shape
// ---------------------------------------------------------------------------
__global__ void softmax_kernel(
    const float* a, float* c,
    int64_t outer, int64_t axis_size, int64_t inner)
{
    int idx = blockIdx.x;
    if (idx >= outer * axis_size * inner) return;

    int o = idx / (axis_size * inner);
    int i = idx % inner;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_val = &shared[blockDim.x];

    // Find max
    float local_max = -INFINITY;
    for (int j = tid; j < axis_size; j += blockDim.x) {
        float val = a[o * axis_size * inner + j * inner + i];
        if (val > local_max) local_max = val;
    }
    max_val[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (max_val[tid + s] > max_val[tid]) max_val[tid] = max_val[tid + s];
        }
        __syncthreads();
    }
    __syncthreads();

    float maxv = max_val[0];

    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int j = tid; j < axis_size; j += blockDim.x) {
        local_sum += expf(a[o * axis_size * inner + j * inner + i] - maxv);
    }
    sum_val[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sum_val[tid] += sum_val[tid + s];
        __syncthreads();
    }
    float total = sum_val[0];

    // Write output
    for (int j = tid; j < axis_size; j += blockDim.x) {
        c[o * axis_size * inner + j * inner + i] = expf(a[o * axis_size * inner + j * inner + i] - maxv) / total;
    }
}

extern "C" void boat_cuda_softmax_f32(
    const float* a, float* c,
    int64_t outer, int64_t axis_size, int64_t inner)
{
    int block = 256;
    if (axis_size < 256) block = 128;
    if (axis_size < 128) block = 64;
    int grid = (int)(outer * axis_size * inner);
    softmax_kernel<<<grid, block, 2 * block * sizeof(float)>>>(a, c, outer, axis_size, inner);
}

// ---------------------------------------------------------------------------
// Sum along axis: sum over cols for each row
// ---------------------------------------------------------------------------
__global__ void sum_axis_kernel(const float* a, float* c, int64_t rows, int64_t cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) sum += a[row * cols + j];
    c[row] = sum;
}

extern "C" void boat_cuda_sum_axis_f32(const float* a, float* c,
    int64_t rows, int64_t cols) {
    sum_axis_kernel<<<rows, 256>>>(a, c, rows, cols);
}
