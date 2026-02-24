// ops.h - Mathematical operations for deep learning framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_OPS_H
#define BOAT_OPS_H

#include "tensor.h"
#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Arithmetic operations
BOAT_API boat_tensor_t* boat_add(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_sub(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_mul(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_div(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_mod(const boat_tensor_t* a, const boat_tensor_t* b);

// In-place arithmetic operations
BOAT_API void boat_add_(boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API void boat_sub_(boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API void boat_mul_(boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API void boat_div_(boat_tensor_t* a, const boat_tensor_t* b);

// Scalar operations
BOAT_API boat_tensor_t* boat_add_scalar(const boat_tensor_t* a, double scalar);
BOAT_API boat_tensor_t* boat_sub_scalar(const boat_tensor_t* a, double scalar);
BOAT_API boat_tensor_t* boat_mul_scalar(const boat_tensor_t* a, double scalar);
BOAT_API boat_tensor_t* boat_div_scalar(const boat_tensor_t* a, double scalar);
BOAT_API boat_tensor_t* boat_pow_scalar(const boat_tensor_t* a, double scalar);

// In-place scalar operations
BOAT_API void boat_add_scalar_(boat_tensor_t* a, double scalar);
BOAT_API void boat_sub_scalar_(boat_tensor_t* a, double scalar);
BOAT_API void boat_mul_scalar_(boat_tensor_t* a, double scalar);
BOAT_API void boat_div_scalar_(boat_tensor_t* a, double scalar);

// Linear algebra operations
BOAT_API boat_tensor_t* boat_matmul(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_dot(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_transpose(const boat_tensor_t* a, int dim0, int dim1);
BOAT_API boat_tensor_t* boat_inverse(const boat_tensor_t* a);

// Activation functions
BOAT_API boat_tensor_t* boat_relu(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_sigmoid(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_tanh(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_softmax(const boat_tensor_t* a, int axis);
BOAT_API boat_tensor_t* boat_log_softmax(const boat_tensor_t* a, int axis);
BOAT_API boat_tensor_t* boat_gelu(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_selu(const boat_tensor_t* a);

// Reduction operations
BOAT_API boat_tensor_t* boat_sum(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim);
BOAT_API boat_tensor_t* boat_mean(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim);
BOAT_API boat_tensor_t* boat_max(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim);
BOAT_API boat_tensor_t* boat_min(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim);
BOAT_API boat_tensor_t* boat_prod(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim);
BOAT_API boat_tensor_t* boat_std(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim, bool unbiased);
BOAT_API boat_tensor_t* boat_var(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim, bool unbiased);

// Comparison operations
BOAT_API boat_tensor_t* boat_eq(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_ne(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_lt(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_le(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_gt(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_ge(const boat_tensor_t* a, const boat_tensor_t* b);

// Logical operations
BOAT_API boat_tensor_t* boat_logical_and(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_logical_or(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_logical_not(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_logical_xor(const boat_tensor_t* a, const boat_tensor_t* b);

// Element-wise mathematical functions
BOAT_API boat_tensor_t* boat_exp(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_log(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_log10(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_log2(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_sqrt(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_rsqrt(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_sin(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_cos(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_tan(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_asin(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_acos(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_atan(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_sinh(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_cosh(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_tanh(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_abs(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_ceil(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_floor(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_round(const boat_tensor_t* a);
BOAT_API boat_tensor_t* boat_trunc(const boat_tensor_t* a);

// Clamping operations
BOAT_API boat_tensor_t* boat_clamp(const boat_tensor_t* a, double min, double max);
BOAT_API boat_tensor_t* boat_clamp_min(const boat_tensor_t* a, double min);
BOAT_API boat_tensor_t* boat_clamp_max(const boat_tensor_t* a, double max);

// Broadcasting utility
BOAT_API bool boat_can_broadcast(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API boat_tensor_t* boat_broadcast_to(const boat_tensor_t* a, const int64_t* shape, size_t ndim);

#ifdef __cplusplus
}
#endif

#endif // BOAT_OPS_H