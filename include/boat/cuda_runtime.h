// cuda_runtime.h - C-callable wrapper declarations for CUDA kernels
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_CUDA_RUNTIME_H
#define BOAT_CUDA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Forward declaration needed for device transfer functions */
typedef struct boat_tensor_t boat_tensor_t;

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
void boat_cuda_relu_backward_f32(const float* x, const float* dy, float* dx, size_t n);
void boat_cuda_sigmoid_f32(const float* a, float* c, size_t n);
void boat_cuda_tanh_f32(const float* a, float* c, size_t n);
void boat_cuda_silu_f32(const float* a, float* c, size_t n);
void boat_cuda_add_scalar_f32(const float* a, float scalar, float* c, size_t n);
void boat_cuda_mul_scalar_f32(const float* a, float scalar, float* c, size_t n);

// Additional math kernels
void boat_cuda_exp_f32(const float* a, float* c, size_t n);
void boat_cuda_log_f32(const float* a, float* c, size_t n);
void boat_cuda_sqrt_f32(const float* a, float* c, size_t n);
void boat_cuda_rsqrt_f32(const float* a, float* c, size_t n);
void boat_cuda_neg_f32(const float* a, float* c, size_t n);
void boat_cuda_abs_f32(const float* a, float* c, size_t n);
void boat_cuda_mod_f32(const float* a, const float* b, float* c, size_t n);
void boat_cuda_sub_scalar_f32(const float* a, float scalar, float* c, size_t n);
void boat_cuda_div_scalar_f32(const float* a, float scalar, float* c, size_t n);
void boat_cuda_fill_f32(float* c, float val, size_t n);
void boat_cuda_scale_f32(float* c, float scalar, size_t n);
void boat_cuda_clamp_f32(const float* a, float* c, float lo, float hi, size_t n);

// ---------------------------------------------------------------------------
// Activation kernels (float32)
// ---------------------------------------------------------------------------
void boat_cuda_softmax_f32(const float* a, float* c,
                           int64_t outer, int64_t axis_size, int64_t inner);
void boat_cuda_log_softmax_f32(const float* a, float* c,
                                int64_t outer, int64_t axis_size, int64_t inner);
void boat_cuda_gelu_f32(const float* a, float* c, size_t n);

// ---------------------------------------------------------------------------
// Reduction kernels (float32)
// ---------------------------------------------------------------------------
float boat_cuda_sum_f32(const float* a, size_t n);

// ---------------------------------------------------------------------------
// Linear algebra kernels (float32)
// ---------------------------------------------------------------------------
void boat_cuda_transpose_f32(const float* a, float* c,
                              int64_t rows, int64_t cols);
void boat_cuda_transpose_nd_f32(const float* a, float* c,
                                  int64_t total_elements,
                                  const int64_t* in_shape,
                                  const size_t* in_stride,
                                  const size_t* out_stride,
                                  int64_t ndim, int dim0, int dim1);
float boat_cuda_dot_f32(const float* a, const float* b, int64_t n);
void boat_cuda_sum_axis_f32(const float* a, float* c,
                              int64_t rows, int64_t cols);

// ---------------------------------------------------------------------------
// Normalization kernels (float32)
// ---------------------------------------------------------------------------
void boat_cuda_layernorm_forward_f32(const float* x, const float* gamma,
                                      const float* beta, float* y,
                                      int64_t rows, int64_t cols, float eps);
void boat_cuda_layernorm_backward_f32(const float* x, const float* y,
                                       const float* gamma, const float* d_y,
                                       float* d_x, float* d_gamma, float* d_beta,
                                       int64_t rows, int64_t cols, float eps);
void boat_cuda_rmsnorm_forward_f32(const float* x, const float* gamma,
                                    float* y, int64_t rows, int64_t cols, float eps);
void boat_cuda_rmsnorm_backward_f32(const float* x, const float* gamma,
                                     const float* d_y, float* d_x,
                                     int64_t rows, int64_t cols, float eps);

// ---------------------------------------------------------------------------
// Pooling kernels (float32)
// ---------------------------------------------------------------------------
void boat_cuda_maxpool2d_forward_f32(const float* input, float* output,
                                      int64_t* indices,
                                      int64_t N, int64_t C, int64_t H, int64_t W,
                                      int64_t KH, int64_t KW,
                                      int64_t pad_h, int64_t pad_w,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t H_out, int64_t W_out);
void boat_cuda_maxpool2d_backward_f32(const float* grad_output,
                                       const int64_t* indices,
                                       float* grad_input,
                                       int64_t N, int64_t C, int64_t H, int64_t W,
                                       int64_t H_out, int64_t W_out);

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
void boat_cuda_conv2d_cudnn_backward_input_f32(const float* grad_output,
                                                  const float* weight, float* grad_input,
                                                  size_t N, size_t C, size_t H, size_t W,
                                                  size_t OC, size_t KH, size_t KW,
                                                  size_t pad, size_t stride, size_t groups);
void boat_cuda_conv2d_cudnn_backward_filter_f32(const float* input,
                                                  const float* grad_output,
                                                  float* grad_weight, float* grad_bias,
                                                  size_t N, size_t C, size_t H, size_t W,
                                                  size_t OC, size_t KH, size_t KW,
                                                  size_t pad, size_t stride, size_t groups);
void boat_cuda_batchnorm_cudnn_forward_f32(const float* input, float* output,
                                             const float* gamma, const float* beta,
                                             float* mean, float* var,
                                             size_t N, size_t C, size_t H, size_t W,
                                             float eps);
void boat_cuda_batchnorm_cudnn_backward_f32(const float* input,
                                              const float* grad_output,
                                              float* grad_input,
                                              const float* gamma,
                                              float* grad_gamma, float* grad_beta,
                                              const float* save_mean,
                                              const float* save_inv_var,
                                              size_t N, size_t C, size_t H, size_t W,
                                              float eps);
void boat_cuda_var_to_inv_var_f32(float* data, size_t C, float eps);
#endif

// ---------------------------------------------------------------------------
// Optimizer kernels (float32)
// ---------------------------------------------------------------------------
void boat_cuda_sgd_update_f32(float* param, const float* grad,
                               float lr, size_t n);
void boat_cuda_sgd_momentum_f32(float* param, const float* grad,
                                  float* velocity, float lr, float momentum,
                                  bool use_nesterov, size_t n);
void boat_cuda_adam_update_f32(float* param, const float* grad,
                                float* m, float* v,
                                float lr, float beta1, float beta2,
                                float beta1_t, float beta2_t, float eps,
                                size_t n);
void boat_cuda_mse_backward_f32(const float* pred, const float* target,
                                 float* grad, size_t n);
void boat_cuda_cross_entropy_backward_f32(const float* pred, const float* target,
                                           float* grad, size_t n);

// ---------------------------------------------------------------------------
// BF16 conversion kernels
// ---------------------------------------------------------------------------
void boat_cuda_fp32_to_bf16(const float* in, void* out, int n);
void boat_cuda_bf16_to_fp32(const void* in, float* out, int n);

// ---------------------------------------------------------------------------
// Optimizer kernels (BF16 params via void*, FP32 grads + FP32 state)
// ---------------------------------------------------------------------------
void boat_cuda_sgd_update_bf16(void* param, const float* grad,
                                float lr, size_t n);
void boat_cuda_sgd_momentum_bf16(void* param, const float* grad,
                                  float* velocity, float lr, float momentum,
                                  bool use_nesterov, size_t n);
void boat_cuda_adam_update_bf16(void* param, const float* grad,
                                 float* m, float* v,
                                 float lr, float beta1, float beta2,
                                 float beta1_t, float beta2_t, float eps,
                                 size_t n);

// ---------------------------------------------------------------------------
// FP8 conversion kernels (E4M3)
// ---------------------------------------------------------------------------
void boat_cuda_fp32_to_fp8(const float* in, void* out, int n);
void boat_cuda_fp8_to_fp32(const void* in, float* out, int n);

// ---------------------------------------------------------------------------
// FP8 element-wise kernels (void* for FP8 data, C-compatible)
// ---------------------------------------------------------------------------
void boat_cuda_fp8_add(const void* a, const void* b, void* out, int n);
void boat_cuda_fp8_mul(const void* a, const void* b, void* out, int n);
void boat_cuda_fp8_relu(const void* a, void* out, int n);
void boat_cuda_fp8_residual_add(void* y, const void* x, int n);

// ---------------------------------------------------------------------------
// FP8 matmul — cuBLAS tensor core via cublasGemmEx (CUDA_R_8F_E4M3)
// C[M,N] = A[M,K] @ B[K,N], FP8 inputs, FP32 output
// ---------------------------------------------------------------------------
void boat_cuda_matmul_fp8_cublas(const void* A, const void* B, float* C,
                                  int M, int N, int K);

// ---------------------------------------------------------------------------
// Tensor device transfer
// ---------------------------------------------------------------------------
void* boat_cuda_clone_to_device(const void* src, size_t nbytes);
void  boat_cuda_copy_to_device(void* dst, const void* src, size_t nbytes);
void  boat_cuda_copy_from_device(void* dst, const void* src, size_t nbytes);
void  boat_cuda_copy_device_to_device(void* dst, const void* src, size_t nbytes);
boat_tensor_t* boat_cuda_tensor_clone(const boat_tensor_t* src);
void boat_cuda_tensor_to_host(boat_tensor_t* tensor);
void boat_cuda_tensor_to_device(boat_tensor_t* tensor);

#ifdef __cplusplus
}
#endif

#endif // BOAT_CUDA_RUNTIME_H