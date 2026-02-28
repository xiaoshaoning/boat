// norm.h - Normalization layers for neural networks
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#ifndef BOAT_NORM_H
#define BOAT_NORM_H

#include "../tensor.h"
#include "../export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct boat_layernorm_t boat_layernorm_t;
typedef struct boat_rmsnorm_t boat_rmsnorm_t;

// Layer normalization configuration
typedef struct {
    size_t normalized_shape;  // Size of the shape to normalize
    float eps;                // Epsilon for numerical stability
    bool elementwise_affine;  // Whether to apply learnable affine transformation
    bool use_bias;            // Whether to use bias in affine transformation
} boat_layernorm_config_t;

// RMS normalization configuration
typedef struct {
    size_t normalized_shape;  // Size of the shape to normalize
    float eps;                // Epsilon for numerical stability
    bool elementwise_affine;  // Whether to apply learnable scale
} boat_rmsnorm_config_t;

// Layer normalization functions
BOAT_API boat_layernorm_t* boat_layernorm_create(const boat_layernorm_config_t* config);
BOAT_API void boat_layernorm_free(boat_layernorm_t* norm);
BOAT_API boat_tensor_t* boat_layernorm_forward(boat_layernorm_t* norm, const boat_tensor_t* input);
BOAT_API boat_tensor_t* boat_layernorm_backward(boat_layernorm_t* norm, const boat_tensor_t* grad_output);
BOAT_API void boat_layernorm_update(boat_layernorm_t* norm, float learning_rate);

// RMS normalization functions
BOAT_API boat_rmsnorm_t* boat_rmsnorm_create(const boat_rmsnorm_config_t* config);
BOAT_API void boat_rmsnorm_free(boat_rmsnorm_t* norm);
BOAT_API boat_tensor_t* boat_rmsnorm_forward(boat_rmsnorm_t* norm, const boat_tensor_t* input);
BOAT_API boat_tensor_t* boat_rmsnorm_backward(boat_rmsnorm_t* norm, const boat_tensor_t* grad_output);
BOAT_API void boat_rmsnorm_update(boat_rmsnorm_t* norm, float learning_rate);

// Standalone normalization functions (stateless)
BOAT_API boat_tensor_t* boat_layer_norm(const boat_tensor_t* input,
                                const int64_t* normalized_shape,
                                size_t normalized_shape_len,
                                float eps);

BOAT_API boat_tensor_t* boat_rms_norm(const boat_tensor_t* input,
                              const int64_t* normalized_shape,
                              size_t normalized_shape_len,
                              float eps);

// Gradient functions for normalization
BOAT_API boat_tensor_t* boat_layer_norm_grad(const boat_tensor_t* grad_output,
                                     const boat_tensor_t* input,
                                     const boat_tensor_t* output,
                                     const int64_t* normalized_shape,
                                     size_t normalized_shape_len,
                                     float eps);

BOAT_API boat_tensor_t* boat_rms_norm_grad(const boat_tensor_t* grad_output,
                                   const boat_tensor_t* input,
                                   const boat_tensor_t* output,
                                   const int64_t* normalized_shape,
                                   size_t normalized_shape_len,
                                   float eps);

// Parameter setting for model loading
BOAT_API void boat_layernorm_set_weight(boat_layernorm_t* norm, boat_tensor_t* weight);
BOAT_API void boat_layernorm_set_bias(boat_layernorm_t* norm, boat_tensor_t* bias);

#ifdef __cplusplus
}
#endif

#endif // BOAT_NORM_H