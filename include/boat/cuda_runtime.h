// cuda_runtime.h - C-callable wrapper declarations for CUDA kernels
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_CUDA_RUNTIME_H
#define BOAT_CUDA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------
void* boat_cuda_malloc(size_t size);
void  boat_cuda_free(void* ptr);
void  boat_cuda_memcpy_h2d(void* dst, const void* src, size_t size);
void  boat_cuda_memcpy_d2h(void* dst, const void* src, size_t size);
void  boat_cuda_memcpy_d2d(void* dst, const void* src, size_t size);
void  boat_cuda_memset(void* ptr, int value, size_t size);
int   boat_cuda_device_count(void);
int   boat_cuda_get_device(void);
void  boat_cuda_set_device(int dev);
void  boat_cuda_synchronize(void);

// ---------------------------------------------------------------------------
// Kernel launch helpers
// ---------------------------------------------------------------------------
typedef struct {
    unsigned int grid_x, grid_y, grid_z;
    unsigned int block_x, block_y, block_z;
} boat_cuda_launch_config_t;

boat_cuda_launch_config_t boat_cuda_choose_launch_config(size_t n_elements, unsigned int block_size);

// ---------------------------------------------------------------------------
// Element-wise kernels (float32)
// ---------------------------------------------------------------------------
void boat_cuda_add_f32(const float* a, const float* b, float* c, size_t n);
void boat_cuda_sub_f32(const float* a, const float* b, float* c, size_t n);
void boat_cuda_mul_f32(const float* a, const float* b, float* c, size_t n);
void boat_cuda_div_f32(const float* a, const float* b, float* c, size_t n);
void boat_cuda_relu_f32(const float* a, float* c, size_t n);
void boat_cuda_sigmoid_f32(const float* a, float* c, size_t n);
void boat_cuda_tanh_f32(const float* a, float* c, size_t n);
void boat_cuda_silu_f32(const float* a, float* c, size_t n);
void boat_cuda_add_scalar_f32(const float* a, float scalar, float* c, size_t n);
void boat_cuda_mul_scalar_f32(const float* a, float scalar, float* c, size_t n);

// ---------------------------------------------------------------------------
// Reduction kernels (float32)
// ---------------------------------------------------------------------------
float boat_cuda_sum_f32(const float* a, size_t n);

// ---------------------------------------------------------------------------
// Matrix multiply (float32) — uses cuBLAS or a simple tiled kernel
// ---------------------------------------------------------------------------
void boat_cuda_matmul_f32(const float* A, const float* B, float* C,
                          size_t M, size_t N, size_t K);

// ---------------------------------------------------------------------------
// cuBLAS wrappers
// ---------------------------------------------------------------------------
void boat_cuda_cublas_destroy(void);
void boat_cuda_matmul_f32_cublas(const float* A, const float* B, float* C,
                                  size_t M, size_t N, size_t K);
void boat_cuda_matmul_f32_strided_batched(const float* A, const float* B, float* C,
                                           size_t M, size_t N, size_t K,
                                           size_t batch_count,
                                           int64_t stride_A, int64_t stride_B, int64_t stride_C);

// ---------------------------------------------------------------------------
// Dense layer kernels
// ---------------------------------------------------------------------------
void boat_cuda_dense_forward_f32(const float* input, const float* weight,
                                  const float* bias, float* output,
                                  size_t B, size_t I, size_t O);
void boat_cuda_dense_forward_warp_f32(const float* input, const float* weight,
                                       const float* bias, float* output,
                                       size_t B, size_t I, size_t O);
void boat_cuda_add_bias_f32(const float* input, const float* bias,
                             float* output, size_t B, size_t O);

// ---------------------------------------------------------------------------
// Conv2D kernels (implicit GEMM via im2col + cuBLAS)
// ---------------------------------------------------------------------------
void boat_cuda_conv2d_forward_f32(const float* input, const float* weight,
                                   const float* bias, float* output,
                                   size_t N, size_t C, size_t H, size_t W,
                                   size_t OC, size_t KH, size_t KW,
                                   size_t pad, size_t stride, size_t groups);

// ---------------------------------------------------------------------------
// Batch norm kernels
// ---------------------------------------------------------------------------
void boat_cuda_batchnorm_forward_f32(const float* input, float* output,
                                      const float* gamma, const float* beta,
                                      float* mean, float* var,
                                      size_t N, size_t C, size_t H, size_t W,
                                      float eps);

// ---------------------------------------------------------------------------
// Fused kernels
// ---------------------------------------------------------------------------
void boat_cuda_fused_bn_relu_f32(const float* input, float* output,
                                   const float* gamma, const float* beta,
                                   const float* mean, const float* var,
                                   size_t N, size_t C, size_t H, size_t W,
                                   float eps);

// ---------------------------------------------------------------------------
// cuDNN wrappers (only available when BOAT_WITH_CUDNN is defined)
// ---------------------------------------------------------------------------
#ifdef BOAT_WITH_CUDNN
void boat_cuda_cudnn_destroy(void);
void boat_cuda_conv2d_cudnn_forward_f32(const float* input, const float* weight,
                                          const float* bias, float* output,
                                          size_t N, size_t C, size_t H, size_t W,
                                          size_t OC, size_t KH, size_t KW,
                                          size_t pad, size_t stride, size_t groups);
void boat_cuda_batchnorm_cudnn_forward_f32(const float* input, float* output,
                                             const float* gamma, const float* beta,
                                             float* mean, float* var,
                                             size_t N, size_t C, size_t H, size_t W,
                                             float eps);
#endif

// ---------------------------------------------------------------------------
// Tensor device transfer
// ---------------------------------------------------------------------------
void* boat_cuda_clone_to_device(const void* src, size_t nbytes);
void  boat_cuda_copy_to_device(void* dst, const void* src, size_t nbytes);
void  boat_cuda_copy_from_device(void* dst, const void* src, size_t nbytes);
void  boat_cuda_copy_device_to_device(void* dst, const void* src, size_t nbytes);

#ifdef __cplusplus
}
#endif

#endif // BOAT_CUDA_RUNTIME_H