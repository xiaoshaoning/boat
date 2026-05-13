// basic.cu - Basic CUDA kernels (memory, element-wise)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------
extern "C" void* boat_cuda_malloc(size_t size) {
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, size);
    return (err == cudaSuccess) ? ptr : NULL;
}

extern "C" void boat_cuda_free(void* ptr) { cudaFree(ptr); }
extern "C" void boat_cuda_memcpy_h2d(void* dst, const void* src, size_t size) { cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice); }
extern "C" void boat_cuda_memcpy_d2h(void* dst, const void* src, size_t size) { cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost); }
extern "C" void boat_cuda_memcpy_d2d(void* dst, const void* src, size_t size) { cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice); }
extern "C" void boat_cuda_memset(void* ptr, int value, size_t size) { cudaMemset(ptr, value, size); }

extern "C" int boat_cuda_device_count(void) { int c; return (cudaGetDeviceCount(&c) == cudaSuccess) ? c : 0; }
extern "C" int boat_cuda_get_device(void) { int d; return (cudaGetDevice(&d) == cudaSuccess) ? d : -1; }
extern "C" void boat_cuda_set_device(int dev) { cudaSetDevice(dev); }
extern "C" void boat_cuda_synchronize(void) { cudaDeviceSynchronize(); }

// ---------------------------------------------------------------------------
// Element-wise kernels
// ---------------------------------------------------------------------------
__global__ void add_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
__global__ void sub_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] - b[idx];
}
__global__ void mul_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}
__global__ void div_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] / b[idx];
}
__global__ void relu_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = (a[idx] > 0.0f) ? a[idx] : 0.0f;
}
__global__ void sigmoid_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = 1.0f / (1.0f + expf(-a[idx]));
}
__global__ void tanh_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = tanhf(a[idx]);
}
__global__ void silu_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] / (1.0f + expf(-a[idx]));
}
__global__ void exp_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = expf(a[idx]);
}
__global__ void log_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = logf(a[idx]);
}
__global__ void sqrt_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = sqrtf(a[idx]);
}
__global__ void rsqrt_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = rsqrtf(a[idx]);
}
__global__ void neg_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = -a[idx];
}
__global__ void abs_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = fabsf(a[idx]);
}
__global__ void add_scalar_f32_kernel(const float* a, float s, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + s;
}
__global__ void mul_scalar_f32_kernel(const float* a, float s, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * s;
}
__global__ void fill_f32_kernel(float* c, float v, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = v;
}
__global__ void scale_f32_kernel(float* c, float s, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) c[idx] *= s;
}
__global__ void gelu_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        c[idx] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

#define LAUNCH1(kernel, a, b, c, n) do { \
    int block = 256; int grid = (int)((n + block - 1) / block); \
    kernel<<<grid, block>>>(a, b, c, n); } while(0)
#define LAUNCH2(kernel, a, c, n) do { \
    int block = 256; int grid = (int)((n + block - 1) / block); \
    kernel<<<grid, block>>>(a, c, n); } while(0)
#define LAUNCH3(kernel, a, s, c, n) do { \
    int block = 256; int grid = (int)((n + block - 1) / block); \
    kernel<<<grid, block>>>(a, s, c, n); } while(0)
#define LAUNCH4(kernel, c, v, n) do { \
    int block = 256; int grid = (int)((n + block - 1) / block); \
    kernel<<<grid, block>>>(c, v, n); } while(0)
#define LAUNCH5(kernel, c, s, n) do { \
    int block = 256; int grid = (int)((n + block - 1) / block); \
    kernel<<<grid, block>>>(c, s, n); } while(0)

extern "C" void boat_cuda_add_f32(const float* a, const float* b, float* c, size_t n) { LAUNCH1(add_f32_kernel, a, b, c, n); }
extern "C" void boat_cuda_sub_f32(const float* a, const float* b, float* c, size_t n) { LAUNCH1(sub_f32_kernel, a, b, c, n); }
extern "C" void boat_cuda_mul_f32(const float* a, const float* b, float* c, size_t n) { LAUNCH1(mul_f32_kernel, a, b, c, n); }
extern "C" void boat_cuda_div_f32(const float* a, const float* b, float* c, size_t n) { LAUNCH1(div_f32_kernel, a, b, c, n); }
extern "C" void boat_cuda_relu_f32(const float* a, float* c, size_t n) { LAUNCH2(relu_f32_kernel, a, c, n); }
extern "C" void boat_cuda_sigmoid_f32(const float* a, float* c, size_t n) { LAUNCH2(sigmoid_f32_kernel, a, c, n); }
extern "C" void boat_cuda_tanh_f32(const float* a, float* c, size_t n) { LAUNCH2(tanh_f32_kernel, a, c, n); }
extern "C" void boat_cuda_silu_f32(const float* a, float* c, size_t n) { LAUNCH2(silu_f32_kernel, a, c, n); }
extern "C" void boat_cuda_exp_f32(const float* a, float* c, size_t n) { LAUNCH2(exp_f32_kernel, a, c, n); }
extern "C" void boat_cuda_log_f32(const float* a, float* c, size_t n) { LAUNCH2(log_f32_kernel, a, c, n); }
extern "C" void boat_cuda_sqrt_f32(const float* a, float* c, size_t n) { LAUNCH2(sqrt_f32_kernel, a, c, n); }
extern "C" void boat_cuda_rsqrt_f32(const float* a, float* c, size_t n) { LAUNCH2(rsqrt_f32_kernel, a, c, n); }
extern "C" void boat_cuda_neg_f32(const float* a, float* c, size_t n) { LAUNCH2(neg_f32_kernel, a, c, n); }
extern "C" void boat_cuda_abs_f32(const float* a, float* c, size_t n) { LAUNCH2(abs_f32_kernel, a, c, n); }
extern "C" void boat_cuda_add_scalar_f32(const float* a, float s, float* c, size_t n) { LAUNCH3(add_scalar_f32_kernel, a, s, c, n); }
extern "C" void boat_cuda_mul_scalar_f32(const float* a, float s, float* c, size_t n) { LAUNCH3(mul_scalar_f32_kernel, a, s, c, n); }
extern "C" void boat_cuda_fill_f32(float* c, float v, size_t n) { LAUNCH4(fill_f32_kernel, c, v, n); }
extern "C" void boat_cuda_scale_f32(float* c, float s, size_t n) { LAUNCH5(scale_f32_kernel, c, s, n); }
extern "C" void boat_cuda_gelu_f32(const float* a, float* c, size_t n) { LAUNCH2(gelu_f32_kernel, a, c, n); }

// ---------------------------------------------------------------------------
// Device transfer
// ---------------------------------------------------------------------------
extern "C" void* boat_cuda_clone_to_device(const void* src, size_t nbytes) {
    void* dst = boat_cuda_malloc(nbytes);
    if (dst) cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice);
    return dst;
}
extern "C" void boat_cuda_copy_to_device(void* dst, const void* src, size_t nbytes) { cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice); }
extern "C" void boat_cuda_copy_from_device(void* dst, const void* src, size_t nbytes) { cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost); }
extern "C" void boat_cuda_copy_device_to_device(void* dst, const void* src, size_t nbytes) { cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice); }
// Tensor-level functions are handled by memory.c

// ---------------------------------------------------------------------------
// Sum reduction
// ---------------------------------------------------------------------------
__global__ void sum_f32_kernel(const float* a, float* partial, size_t n) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? a[idx] : 0.0f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

extern "C" float boat_cuda_sum_f32(const float* a, size_t n) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    float* partial;
    cudaMalloc(&partial, grid * sizeof(float));
    sum_f32_kernel<<<grid, block, block * sizeof(float)>>>(a, partial, n);
    float* host_partial = new float[grid];
    cudaMemcpy(host_partial, partial, grid * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0.0f;
    for (int i = 0; i < grid; i++) sum += host_partial[i];
    delete[] host_partial;
    cudaFree(partial);
    return sum;
}
