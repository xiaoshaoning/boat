// linear.cu - CUDA linear algebra operations
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
// 2D transpose: tiled with 16x16 shared memory
// Input:  [rows, cols] row-major
// Output: [cols, rows] row-major
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void transpose_2d_f32_kernel(const float* __restrict__ a,
                                          float* __restrict__ c,
                                          int64_t rows, int64_t cols)
{
    __shared__ float tile[TILE][TILE + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Load tile from input (row-major)
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = a[y * cols + x];
    }
    __syncthreads();

    // Store transposed tile to output
    int x_out = blockIdx.y * TILE + threadIdx.x;
    int y_out = blockIdx.x * TILE + threadIdx.y;

    if (x_out < rows && y_out < cols) {
        c[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

// ---------------------------------------------------------------------------
// N-D transpose fallback: compute indices via coordinate transform
// ---------------------------------------------------------------------------
__global__ void transpose_nd_f32_kernel(const float* __restrict__ a,
                                          float* __restrict__ c,
                                          int64_t total_elements,
                                          const int64_t* __restrict__ in_shape,
                                          const size_t* __restrict__ in_stride,
                                          const size_t* __restrict__ out_stride,
                                          int64_t ndim, int dim0, int dim1)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Compute input coordinates (row-major)
    int64_t temp = idx;
    int64_t coords[8]; // max ndim
    for (int d = ndim - 1; d >= 0; d--) {
        coords[d] = temp % in_shape[d];
        temp /= in_shape[d];
    }

    // Swap dim0 and dim1
    int64_t t = coords[dim0];
    coords[dim0] = coords[dim1];
    coords[dim1] = t;

    // Compute output index
    size_t out_idx = 0;
    for (int d = 0; d < ndim; d++) {
        out_idx += (size_t)coords[d] * out_stride[d];
    }

    c[out_idx] = a[idx];
}

// ---------------------------------------------------------------------------
// Dot product: 1D reduction kernel
// ---------------------------------------------------------------------------
__global__ void dot_f32_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ partials,
                                int64_t n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;
    float sum = 0.0f;
    if (i < n) sum += a[i] * b[i];
    if (i + blockDim.x < n) sum += a[i + blockDim.x] * b[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Reduce bias gradient along batch: sum across first dim
// Input: [B, O], Output: [1, O]
// ---------------------------------------------------------------------------
__global__ void sum_axis_f32_kernel(const float* __restrict__ a,
                                     float* __restrict__ c,
                                     int64_t rows, int64_t cols)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int col = blockIdx.x;

    if (col >= cols) return;

    float sum = 0.0f;
    for (int r = tid; r < rows; r += blockDim.x) {
        sum += a[r * cols + col];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) c[col] += sdata[0];
}

extern "C" {

void boat_cuda_transpose_f32(const float* a, float* c,
                              int64_t rows, int64_t cols)
{
    const int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    transpose_2d_f32_kernel<TILE><<<grid, block>>>(a, c, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_transpose_nd_f32(const float* a, float* c,
                                  int64_t total_elements,
                                  const int64_t* in_shape,
                                  const size_t* in_stride,
                                  const size_t* out_stride,
                                  int64_t ndim, int dim0, int dim1)
{
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((total_elements + block - 1) / block);
    // Copy shapes/ strides to device
    int64_t* d_in_shape;
    size_t* d_in_stride;
    size_t* d_out_stride;
    CUDA_CHECK(cudaMalloc(&d_in_shape, ndim * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_in_stride, ndim * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_out_stride, ndim * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_in_shape, in_shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_stride, in_stride, ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_stride, out_stride, ndim * sizeof(size_t), cudaMemcpyHostToDevice));

    transpose_nd_f32_kernel<<<grid, block>>>(
        a, c, total_elements, d_in_shape, d_in_stride, d_out_stride, ndim, dim0, dim1);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_in_shape));
    CUDA_CHECK(cudaFree(d_in_stride));
    CUDA_CHECK(cudaFree(d_out_stride));
}

float boat_cuda_dot_f32(const float* a, const float* b, int64_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block * 2 - 1) / (block * 2));
    float* d_partials;
    CUDA_CHECK(cudaMalloc(&d_partials, grid * sizeof(float)));
    dot_f32_kernel<<<grid, block, block * sizeof(float)>>>(a, b, d_partials, n);
    CUDA_CHECK(cudaGetLastError());
    float* h_partials = new float[grid];
    CUDA_CHECK(cudaMemcpy(h_partials, d_partials, grid * sizeof(float), cudaMemcpyDeviceToHost));
    float total = 0.0f;
    for (unsigned int i = 0; i < grid; i++) total += h_partials[i];
    delete[] h_partials;
    CUDA_CHECK(cudaFree(d_partials));
    return total;
}

void boat_cuda_sum_axis_f32(const float* a, float* c,
                              int64_t rows, int64_t cols)
{
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)cols;
    size_t shared_mem = block * sizeof(float);
    sum_axis_f32_kernel<<<grid, block, shared_mem>>>(a, c, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
