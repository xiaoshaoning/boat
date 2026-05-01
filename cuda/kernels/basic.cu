// basic.cu - Basic CUDA kernel implementations
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
// Element-wise kernels (float32)
// ---------------------------------------------------------------------------
__global__ void add_f32_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void sub_f32_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] - b[idx];
}

__global__ void mul_f32_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

__global__ void div_f32_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] / b[idx];
}

__global__ void relu_f32_kernel(const float* __restrict__ a,
                                 float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] > 0.0f ? a[idx] : 0.0f;
}

__global__ void sigmoid_f32_kernel(const float* __restrict__ a,
                                    float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = 1.0f / (1.0f + expf(-a[idx]));
}

__global__ void tanh_f32_kernel(const float* __restrict__ a,
                                 float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = tanhf(a[idx]);
}

__global__ void silu_f32_kernel(const float* __restrict__ a,
                                 float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] / (1.0f + expf(-a[idx]));
}

__global__ void add_scalar_f32_kernel(const float* __restrict__ a,
                                       float scalar,
                                       float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + scalar;
}

__global__ void mul_scalar_f32_kernel(const float* __restrict__ a,
                                       float scalar,
                                       float* __restrict__ c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * scalar;
}

// ---------------------------------------------------------------------------
// Reduction kernel: parallel sum (float32)
// ---------------------------------------------------------------------------
__global__ void sum_f32_kernel(const float* __restrict__ a,
                                float* __restrict__ partials,
                                size_t n) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + tid;
    float sum = 0.0f;
    if (i < n) sum += a[i];
    if (i + blockDim.x < n) sum += a[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    // Warp-level reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Tiled matrix multiply kernel (float32, MxK * KxN = MxN)
// ---------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void matmul_f32_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   size_t M, size_t N, size_t K) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ---------------------------------------------------------------------------
// Host-callable wrappers (extern "C")
// ---------------------------------------------------------------------------
extern "C" {

void* boat_cuda_malloc(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void boat_cuda_free(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void boat_cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void boat_cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void boat_cuda_memcpy_d2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

void boat_cuda_memset(void* ptr, int value, size_t size) {
    CUDA_CHECK(cudaMemset(ptr, value, size));
}

int boat_cuda_device_count(void) {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) return 0;
    return count;
}

int boat_cuda_get_device(void) {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    return dev;
}

void boat_cuda_set_device(int dev) {
    CUDA_CHECK(cudaSetDevice(dev));
}

void boat_cuda_synchronize(void) {
    CUDA_CHECK(cudaDeviceSynchronize());
}

boat_cuda_launch_config_t boat_cuda_choose_launch_config(size_t n_elements, unsigned int block_size) {
    boat_cuda_launch_config_t cfg;
    cfg.block_x = block_size;
    cfg.block_y = 1;
    cfg.block_z = 1;
    cfg.grid_x = (unsigned int)((n_elements + block_size - 1) / block_size);
    cfg.grid_y = 1;
    cfg.grid_z = 1;
    return cfg;
}

// Element-wise wrappers
void boat_cuda_add_f32(const float* a, const float* b, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    add_f32_kernel<<<grid, block>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_sub_f32(const float* a, const float* b, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    sub_f32_kernel<<<grid, block>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_mul_f32(const float* a, const float* b, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    mul_f32_kernel<<<grid, block>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_div_f32(const float* a, const float* b, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    div_f32_kernel<<<grid, block>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_relu_f32(const float* a, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    relu_f32_kernel<<<grid, block>>>(a, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_sigmoid_f32(const float* a, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    sigmoid_f32_kernel<<<grid, block>>>(a, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_tanh_f32(const float* a, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    tanh_f32_kernel<<<grid, block>>>(a, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_silu_f32(const float* a, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    silu_f32_kernel<<<grid, block>>>(a, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_add_scalar_f32(const float* a, float scalar, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    add_scalar_f32_kernel<<<grid, block>>>(a, scalar, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_mul_scalar_f32(const float* a, float scalar, float* c, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block - 1) / block);
    mul_scalar_f32_kernel<<<grid, block>>>(a, scalar, c, n);
    CUDA_CHECK(cudaGetLastError());
}

// Sum reduction
float boat_cuda_sum_f32(const float* a, size_t n) {
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((n + block * 2 - 1) / (block * 2));
    float* d_partials;
    CUDA_CHECK(cudaMalloc(&d_partials, grid * sizeof(float)));
    sum_f32_kernel<<<grid, block, block * sizeof(float)>>>(a, d_partials, n);
    CUDA_CHECK(cudaGetLastError());
    float* h_partials = new float[grid];
    CUDA_CHECK(cudaMemcpy(h_partials, d_partials, grid * sizeof(float), cudaMemcpyDeviceToHost));
    float total = 0.0f;
    for (unsigned int i = 0; i < grid; i++) total += h_partials[i];
    delete[] h_partials;
    CUDA_CHECK(cudaFree(d_partials));
    return total;
}

// Matrix multiply (uses tiled kernel, TILE_SIZE=16)
void boat_cuda_matmul_f32(const float* A, const float* B, float* C,
                          size_t M, size_t N, size_t K) {
    const int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_f32_kernel<TILE><<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// Tensor device transfer wrappers
void* boat_cuda_clone_to_device(const void* src, size_t nbytes) {
    void* dst;
    CUDA_CHECK(cudaMalloc(&dst, nbytes));
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
    return dst;
}

void boat_cuda_copy_to_device(void* dst, const void* src, size_t nbytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
}

void boat_cuda_copy_from_device(void* dst, const void* src, size_t nbytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost));
}

void boat_cuda_copy_device_to_device(void* dst, const void* src, size_t nbytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
}

} // extern "C"