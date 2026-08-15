// norm.c - Normalization layers implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers/norm.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <boat/simd.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef BOAT_WITH_CUDA
#include <boat/cuda_runtime.h>
#endif

// Layer normalization structure
struct boat_layernorm_t {
    boat_layernorm_config_t config;

    // Learnable parameters (if elementwise_affine is true)
    boat_tensor_t* weight; // gamma
    boat_tensor_t* bias;   // beta

    // Gradient accumulators
    boat_tensor_t* grad_weight; // dL/dgamma
    boat_tensor_t* grad_bias;   // dL/dbeta

    // Cache for backward pass
    boat_tensor_t* cache_input;
    boat_tensor_t* cache_mean;
    boat_tensor_t* cache_variance;
};

// RMS normalization structure
struct boat_rmsnorm_t {
    boat_rmsnorm_config_t config;

    // Learnable scale (if elementwise_affine is true)
    boat_tensor_t* weight; // gamma

    // Gradient accumulator
    boat_tensor_t* grad_weight; // dL/dgamma

    // Cache for backward pass
    boat_tensor_t* cache_input;
    boat_tensor_t* cache_rms;
};

// Helper function to create weight tensor
static boat_tensor_t* create_weight_tensor(size_t normalized_shape) {
    const int64_t shape[] = {(int64_t)normalized_shape};
    boat_tensor_t* weight = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    if (!weight) {
        return NULL;
    }

    // Initialize with ones
    float* data = (float*)boat_tensor_data(weight);
    size_t num_elements = boat_tensor_nelements(weight);
    for (size_t i = 0; i < num_elements; i++) {
        data[i] = 1.0f;
    }

    return weight;
}

// Helper function to create bias tensor
static boat_tensor_t* create_bias_tensor(size_t normalized_shape) {
    const int64_t shape[] = {(int64_t)normalized_shape};
    boat_tensor_t* bias = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    if (!bias) {
        return NULL;
    }

    // Initialize with zeros
    float* data = (float*)boat_tensor_data(bias);
    size_t num_elements = boat_tensor_nelements(bias);
    memset(data, 0, num_elements * sizeof(float));

    return bias;
}

BOAT_API boat_layernorm_t* boat_layernorm_create(const boat_layernorm_config_t* config) {
    if (!config || config->normalized_shape == 0) {
        return NULL;
    }

    boat_layernorm_t* norm =
        (boat_layernorm_t*)boat_malloc(sizeof(boat_layernorm_t), BOAT_DEVICE_CPU);
    if (!norm) {
        return NULL;
    }

    // Copy configuration
    memcpy(&norm->config, config, sizeof(boat_layernorm_config_t));

    // Create learnable parameters if elementwise_affine is true
    if (config->elementwise_affine) {
        norm->weight = create_weight_tensor(config->normalized_shape);
        if (!norm->weight) {
            boat_free(norm);
            return NULL;
        }

        if (config->use_bias) {
            norm->bias = create_bias_tensor(config->normalized_shape);
            if (!norm->bias) {
                boat_tensor_free(norm->weight);
                boat_free(norm);
                return NULL;
            }
        } else {
            norm->bias = NULL;
        }
    } else {
        norm->weight = NULL;
        norm->bias = NULL;
    }

    // Initialize cache pointers
    norm->cache_input = NULL;
    norm->cache_mean = NULL;
    norm->cache_variance = NULL;
    norm->grad_weight = NULL;
    norm->grad_bias = NULL;

    return norm;
}

BOAT_API void boat_layernorm_free(boat_layernorm_t* norm) {
    if (!norm) {
        return;
    }

    // Free learnable parameters
    if (norm->weight) boat_tensor_free(norm->weight);
    if (norm->bias) boat_tensor_free(norm->bias);

    // Free cache tensors
    if (norm->cache_input) boat_tensor_free(norm->cache_input);
    if (norm->cache_mean) boat_tensor_free(norm->cache_mean);
    if (norm->cache_variance) boat_tensor_free(norm->cache_variance);
    if (norm->grad_weight) boat_tensor_free(norm->grad_weight);
    if (norm->grad_bias) boat_tensor_free(norm->grad_bias);

    // Free norm structure
    boat_free(norm);
}

BOAT_API boat_rmsnorm_t* boat_rmsnorm_create(const boat_rmsnorm_config_t* config) {
    if (!config || config->normalized_shape == 0) {
        return NULL;
    }

    boat_rmsnorm_t* norm = (boat_rmsnorm_t*)boat_malloc(sizeof(boat_rmsnorm_t), BOAT_DEVICE_CPU);
    if (!norm) {
        return NULL;
    }

    // Copy configuration
    memcpy(&norm->config, config, sizeof(boat_rmsnorm_config_t));

    // Create learnable scale if elementwise_affine is true
    if (config->elementwise_affine) {
        norm->weight = create_weight_tensor(config->normalized_shape);
        if (!norm->weight) {
            boat_free(norm);
            return NULL;
        }
    } else {
        norm->weight = NULL;
    }

    // Initialize cache pointers
    norm->cache_input = NULL;
    norm->cache_rms = NULL;
    norm->grad_weight = NULL;

    return norm;
}

BOAT_API void boat_rmsnorm_free(boat_rmsnorm_t* norm) {
    if (!norm) {
        return;
    }

    // Free learnable parameters
    if (norm->weight) boat_tensor_free(norm->weight);

    // Free cache tensors
    if (norm->cache_input) boat_tensor_free(norm->cache_input);
    if (norm->cache_rms) boat_tensor_free(norm->cache_rms);
    if (norm->grad_weight) boat_tensor_free(norm->grad_weight);

    // Free norm structure
    boat_free(norm);
}

// Helper function to compute mean and variance along last dimension
static void compute_mean_variance(const float* input, size_t batch_size, size_t seq_len,
                                  size_t hidden_size, float* mean, float* variance) {
    size_t rows = batch_size * seq_len;
    boat_simd_mean_var_f32(input, mean, variance, rows, hidden_size);
}

// Helper function to compute RMS (root mean square) along last dimension
static void compute_rms(const float* input, size_t batch_size, size_t seq_len, size_t hidden_size,
                        float* rms) {
    size_t rows = batch_size * seq_len;
    boat_simd_rms_f32(input, rms, rows, hidden_size);
}

BOAT_API boat_tensor_t* boat_layernorm_forward(boat_layernorm_t* norm, const boat_tensor_t* input) {
    if (!norm || !input) {
        return NULL;
    }

    // Get input shape
    size_t ndim = boat_tensor_ndim(input);
    const int64_t* shape = boat_tensor_shape(input);

    if (ndim < 1) {
        return NULL;
    }

    // For simplicity, assume last dimension is normalized_shape
    size_t last_dim = shape[ndim - 1];
    if (last_dim != norm->config.normalized_shape) {
        return NULL;
    }

    // Calculate total elements and inner dimensions
    size_t total_elements = boat_tensor_nelements(input);
    size_t hidden_size = norm->config.normalized_shape;
    size_t outer_elements = total_elements / hidden_size;

    // Create output tensor
    boat_tensor_t* output = boat_tensor_create_like(input);
    if (!output) {
        return NULL;
    }

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(input) == BOAT_DEVICE_CUDA) {
        const float* gamma =
            norm->weight ? (const float*)boat_tensor_const_data(norm->weight) : NULL;
        const float* beta = norm->bias ? (const float*)boat_tensor_const_data(norm->bias) : NULL;

        boat_cuda_layernorm_forward_f32((const float*)boat_tensor_const_data(input), gamma, beta,
                                        (float*)boat_tensor_data(output), (int64_t)outer_elements,
                                        (int64_t)hidden_size, norm->config.eps);

        // Cache input for backward pass
        if (norm->cache_input) boat_tensor_free(norm->cache_input);
        norm->cache_input = boat_tensor_clone(input);

        return output;
    }
#endif

    // Get data pointers
    const float* input_data = (const float*)boat_tensor_const_data(input);
    float* output_data = (float*)boat_tensor_data(output);

    // Compute mean and variance for each position
    float* mean = (float*)boat_malloc(outer_elements * sizeof(float), BOAT_DEVICE_CPU);
    float* variance = (float*)boat_malloc(outer_elements * sizeof(float), BOAT_DEVICE_CPU);

    if (!mean || !variance) {
        boat_tensor_free(output);
        if (mean) boat_free(mean);
        if (variance) boat_free(variance);
        return NULL;
    }

    // For simplicity, assume 3D tensor: [batch, seq_len, hidden]
    size_t batch_size = (ndim >= 3) ? shape[0] : 1;
    size_t seq_len = (ndim >= 3) ? shape[1] : (ndim == 2 ? shape[0] : 1);

    compute_mean_variance(input_data, batch_size, seq_len, hidden_size, mean, variance);

    // Apply layer normalization
    float eps = norm->config.eps;
    const float* weight_data =
        norm->weight ? (const float*)boat_tensor_const_data(norm->weight) : NULL;
    const float* bias_data = norm->bias ? (const float*)boat_tensor_const_data(norm->bias) : NULL;

    float* inv_std = (float*)boat_malloc(outer_elements * sizeof(float), BOAT_DEVICE_CPU);
    if (!inv_std) {
        boat_free(mean);
        boat_free(variance);
        boat_tensor_free(output);
        return NULL;
    }
    for (size_t o = 0; o < outer_elements; o++) {
        inv_std[o] = 1.0f / sqrtf(variance[o] + eps);
    }
    boat_simd_norm_affine_f32(input_data, weight_data, bias_data, output_data, outer_elements,
                              hidden_size, mean, inv_std);
    boat_free(inv_std);

    // Cache input, mean, and variance for backward pass
    if (norm->cache_input) boat_tensor_free(norm->cache_input);
    if (norm->cache_mean) boat_tensor_free(norm->cache_mean);
    if (norm->cache_variance) boat_tensor_free(norm->cache_variance);
    norm->cache_input = boat_tensor_clone(input);
    const int64_t stats_shape[] = {(int64_t)outer_elements};
    norm->cache_mean = boat_tensor_create(stats_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    norm->cache_variance = boat_tensor_create(stats_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (norm->cache_mean && norm->cache_variance) {
        memcpy(boat_tensor_data(norm->cache_mean), mean, outer_elements * sizeof(float));
        memcpy(boat_tensor_data(norm->cache_variance), variance, outer_elements * sizeof(float));
    }

    // Free temporary arrays (after caching into norm)
    boat_free(mean);
    boat_free(variance);

    return output;
}

BOAT_API boat_tensor_t* boat_rmsnorm_forward(boat_rmsnorm_t* norm, const boat_tensor_t* input) {
    if (!norm || !input) {
        return NULL;
    }

    // Get input shape
    size_t ndim = boat_tensor_ndim(input);
    const int64_t* shape = boat_tensor_shape(input);

    if (ndim < 1) {
        return NULL;
    }

    // For simplicity, assume last dimension is normalized_shape
    size_t last_dim = shape[ndim - 1];
    if (last_dim != norm->config.normalized_shape) {
        return NULL;
    }

    // Calculate total elements and inner dimensions
    size_t total_elements = boat_tensor_nelements(input);
    size_t hidden_size = norm->config.normalized_shape;
    size_t outer_elements = total_elements / hidden_size;

    // Create output tensor
    boat_tensor_t* output = boat_tensor_create_like(input);
    if (!output) {
        return NULL;
    }

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(input) == BOAT_DEVICE_CUDA) {
        const float* gamma =
            norm->weight ? (const float*)boat_tensor_const_data(norm->weight) : NULL;

        boat_cuda_rmsnorm_forward_f32((const float*)boat_tensor_const_data(input), gamma,
                                      (float*)boat_tensor_data(output), (int64_t)outer_elements,
                                      (int64_t)hidden_size, norm->config.eps);

        // Cache input for backward pass
        if (norm->cache_input) boat_tensor_free(norm->cache_input);
        norm->cache_input = boat_tensor_clone(input);

        return output;
    }
#endif

    // Get data pointers
    const float* input_data = (const float*)boat_tensor_const_data(input);
    float* output_data = (float*)boat_tensor_data(output);

    // Compute RMS for each position
    float* rms = (float*)boat_malloc(outer_elements * sizeof(float), BOAT_DEVICE_CPU);
    if (!rms) {
        boat_tensor_free(output);
        return NULL;
    }

    // For simplicity, assume 3D tensor: [batch, seq_len, hidden]
    size_t batch_size = (ndim >= 3) ? shape[0] : 1;
    size_t seq_len = (ndim >= 3) ? shape[1] : (ndim == 2 ? shape[0] : 1);

    compute_rms(input_data, batch_size, seq_len, hidden_size, rms);

    // Apply RMS normalization
    float eps = norm->config.eps;
    const float* weight_data =
        norm->weight ? (const float*)boat_tensor_const_data(norm->weight) : NULL;

    float* inv_rms = (float*)boat_malloc(outer_elements * sizeof(float), BOAT_DEVICE_CPU);
    if (!inv_rms) {
        boat_free(rms);
        boat_tensor_free(output);
        return NULL;
    }
    for (size_t o = 0; o < outer_elements; o++) {
        inv_rms[o] = 1.0f / (rms[o] + eps);
    }
    boat_simd_norm_affine_f32(input_data, weight_data, NULL, output_data, outer_elements,
                              hidden_size, NULL, inv_rms);
    boat_free(inv_rms);

    // Cache input and RMS for backward pass
    if (norm->cache_input) boat_tensor_free(norm->cache_input);
    if (norm->cache_rms) boat_tensor_free(norm->cache_rms);
    norm->cache_input = boat_tensor_clone(input);
    const int64_t rms_shape[] = {(int64_t)outer_elements};
    norm->cache_rms = boat_tensor_create(rms_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (norm->cache_rms) {
        memcpy(boat_tensor_data(norm->cache_rms), rms, outer_elements * sizeof(float));
    }

    // Free temporary array (after caching into norm)
    boat_free(rms);

    return output;
}

BOAT_API boat_tensor_t* boat_layernorm_backward(boat_layernorm_t* norm,
                                                const boat_tensor_t* grad_output) {
    if (!norm || !grad_output || !norm->cache_input) {
        return NULL;
    }

    size_t hidden = norm->config.normalized_shape;
    size_t total = boat_tensor_nelements(grad_output);
    if (total == 0 || total % hidden != 0) {
        return NULL;
    }
    size_t outer = total / hidden;

    boat_tensor_t* grad_input = boat_tensor_create_like(grad_output);
    if (!grad_input) {
        return NULL;
    }

    const float* x = NULL;
    const float* dy = NULL;
    float* dx = NULL;
    float* x_host = NULL;
    float* dy_host = NULL;
    float* dx_host = NULL;
    bool on_cuda = false;

#ifdef BOAT_WITH_CUDA
    on_cuda = (boat_tensor_device(grad_output) == BOAT_DEVICE_CUDA);
#endif

    if (on_cuda) {
        // Run the host algorithm on copied data for correctness; the CUDA
        // d_gamma/d_beta kernel only supports single-row partials.
        x_host = (float*)boat_malloc(total * sizeof(float), BOAT_DEVICE_CPU);
        dy_host = (float*)boat_malloc(total * sizeof(float), BOAT_DEVICE_CPU);
        dx_host = (float*)boat_malloc(total * sizeof(float), BOAT_DEVICE_CPU);
        if (!x_host || !dy_host || !dx_host) {
            boat_free(x_host);
            boat_free(dy_host);
            boat_free(dx_host);
            boat_tensor_free(grad_input);
            return NULL;
        }
#ifdef BOAT_WITH_CUDA
        if (boat_tensor_device(norm->cache_input) == BOAT_DEVICE_CUDA) {
            boat_cuda_memcpy_d2h(x_host, boat_tensor_const_data(norm->cache_input),
                                 total * sizeof(float));
        } else {
            memcpy(x_host, boat_tensor_const_data(norm->cache_input), total * sizeof(float));
        }
        boat_cuda_memcpy_d2h(dy_host, boat_tensor_const_data(grad_output), total * sizeof(float));
#else
        memcpy(x_host, boat_tensor_const_data(norm->cache_input), total * sizeof(float));
        memcpy(dy_host, boat_tensor_const_data(grad_output), total * sizeof(float));
#endif
        x = x_host;
        dy = dy_host;
        dx = dx_host;
    } else {
        x = (const float*)boat_tensor_const_data(norm->cache_input);
        dy = (const float*)boat_tensor_const_data(grad_output);
        dx = (float*)boat_tensor_data(grad_input);
    }

    float eps = norm->config.eps;
    const float* gamma = norm->weight ? (const float*)boat_tensor_const_data(norm->weight) : NULL;

    // Accumulate d_gamma and d_beta (lazy creation)
    if (norm->weight && !norm->grad_weight) {
        norm->grad_weight = boat_tensor_create_like(norm->weight);
        if (norm->grad_weight) {
            memset(boat_tensor_data(norm->grad_weight), 0, boat_tensor_nbytes(norm->grad_weight));
        }
    }
    if (norm->bias && !norm->grad_bias) {
        norm->grad_bias = boat_tensor_create_like(norm->bias);
        if (norm->grad_bias) {
            memset(boat_tensor_data(norm->grad_bias), 0, boat_tensor_nbytes(norm->grad_bias));
        }
    }

    // Fused SIMD backward: recomputes per-row stats, accumulates grad_weight /
    // grad_bias and writes grad_input in a single pass.
    boat_simd_layernorm_backward_f32(
        x, dy, gamma, dx,
        norm->grad_weight ? (float*)boat_tensor_data(norm->grad_weight) : NULL,
        norm->grad_bias ? (float*)boat_tensor_data(norm->grad_bias) : NULL, outer, hidden, eps);

#ifdef BOAT_WITH_CUDA
    if (on_cuda) {
        boat_cuda_memcpy_h2d(boat_tensor_data(grad_input), dx, total * sizeof(float));
    }
#endif

    if (on_cuda) {
        boat_free(x_host);
        boat_free(dy_host);
        boat_free(dx_host);
    }

    return grad_input;
}

BOAT_API boat_tensor_t* boat_rmsnorm_backward(boat_rmsnorm_t* norm,
                                              const boat_tensor_t* grad_output) {
    if (!norm || !grad_output || !norm->cache_input) {
        return NULL;
    }

    size_t hidden = norm->config.normalized_shape;
    size_t total = boat_tensor_nelements(grad_output);
    if (total == 0 || total % hidden != 0) {
        return NULL;
    }
    size_t outer = total / hidden;

    boat_tensor_t* grad_input = boat_tensor_create_like(grad_output);
    if (!grad_input) {
        return NULL;
    }

    const float* x = NULL;
    const float* dy = NULL;
    float* dx = NULL;
    float* x_host = NULL;
    float* dy_host = NULL;
    float* dx_host = NULL;
    bool on_cuda = false;

#ifdef BOAT_WITH_CUDA
    on_cuda = (boat_tensor_device(grad_output) == BOAT_DEVICE_CUDA);
#endif

    if (on_cuda) {
        // Run the host algorithm on copied data for correctness.
        x_host = (float*)boat_malloc(total * sizeof(float), BOAT_DEVICE_CPU);
        dy_host = (float*)boat_malloc(total * sizeof(float), BOAT_DEVICE_CPU);
        dx_host = (float*)boat_malloc(total * sizeof(float), BOAT_DEVICE_CPU);
        if (!x_host || !dy_host || !dx_host) {
            boat_free(x_host);
            boat_free(dy_host);
            boat_free(dx_host);
            boat_tensor_free(grad_input);
            return NULL;
        }
#ifdef BOAT_WITH_CUDA
        if (boat_tensor_device(norm->cache_input) == BOAT_DEVICE_CUDA) {
            boat_cuda_memcpy_d2h(x_host, boat_tensor_const_data(norm->cache_input),
                                 total * sizeof(float));
        } else {
            memcpy(x_host, boat_tensor_const_data(norm->cache_input), total * sizeof(float));
        }
        boat_cuda_memcpy_d2h(dy_host, boat_tensor_const_data(grad_output), total * sizeof(float));
#else
        memcpy(x_host, boat_tensor_const_data(norm->cache_input), total * sizeof(float));
        memcpy(dy_host, boat_tensor_const_data(grad_output), total * sizeof(float));
#endif
        x = x_host;
        dy = dy_host;
        dx = dx_host;
    } else {
        x = (const float*)boat_tensor_const_data(norm->cache_input);
        dy = (const float*)boat_tensor_const_data(grad_output);
        dx = (float*)boat_tensor_data(grad_input);
    }

    float eps = norm->config.eps;
    const float* gamma = norm->weight ? (const float*)boat_tensor_const_data(norm->weight) : NULL;

    // Accumulate d_gamma (lazy creation)
    if (norm->weight && !norm->grad_weight) {
        norm->grad_weight = boat_tensor_create_like(norm->weight);
        if (norm->grad_weight) {
            memset(boat_tensor_data(norm->grad_weight), 0, boat_tensor_nbytes(norm->grad_weight));
        }
    }

    // Fused SIMD backward: recomputes per-row RMS, accumulates grad_weight and
    // writes grad_input in a single pass.
    boat_simd_rmsnorm_backward_f32(
        x, dy, gamma, dx,
        norm->grad_weight ? (float*)boat_tensor_data(norm->grad_weight) : NULL, outer, hidden,
        eps);

#ifdef BOAT_WITH_CUDA
    if (on_cuda) {
        boat_cuda_memcpy_h2d(boat_tensor_data(grad_input), dx, total * sizeof(float));
    }
#endif

    if (on_cuda) {
        boat_free(x_host);
        boat_free(dy_host);
        boat_free(dx_host);
    }

    return grad_input;
}

BOAT_API void boat_layernorm_update(boat_layernorm_t* norm, float learning_rate) {
    if (!norm) return;
    if (norm->grad_weight && norm->weight) {
        float* w = (float*)boat_tensor_data(norm->weight);
        float* gw = (float*)boat_tensor_data(norm->grad_weight);
        size_t n = boat_tensor_nelements(norm->weight);
        for (size_t i = 0; i < n; i++) {
            w[i] -= learning_rate * gw[i];
        }
        memset(gw, 0, n * sizeof(float));
    }
    if (norm->grad_bias && norm->bias) {
        float* b = (float*)boat_tensor_data(norm->bias);
        float* gb = (float*)boat_tensor_data(norm->grad_bias);
        size_t n = boat_tensor_nelements(norm->bias);
        for (size_t i = 0; i < n; i++) {
            b[i] -= learning_rate * gb[i];
        }
        memset(gb, 0, n * sizeof(float));
    }
}

BOAT_API void boat_rmsnorm_update(boat_rmsnorm_t* norm, float learning_rate) {
    if (!norm) return;
    if (norm->grad_weight && norm->weight) {
        float* w = (float*)boat_tensor_data(norm->weight);
        float* gw = (float*)boat_tensor_data(norm->grad_weight);
        size_t n = boat_tensor_nelements(norm->weight);
        for (size_t i = 0; i < n; i++) {
            w[i] -= learning_rate * gw[i];
        }
        memset(gw, 0, n * sizeof(float));
    }
}

// Standalone layer norm function
BOAT_API boat_tensor_t* boat_layer_norm(const boat_tensor_t* input, const int64_t* normalized_shape,
                                        size_t normalized_shape_len, float eps) {
    if (!input || !normalized_shape || normalized_shape_len == 0) return NULL;

    size_t ndim = boat_tensor_ndim(input);
    const int64_t* shape = boat_tensor_shape(input);
    int64_t D = normalized_shape[normalized_shape_len - 1];

    // Verify last dim matches
    if (shape[ndim - 1] != D) return NULL;

    size_t total = boat_tensor_nelements(input);
    size_t outer = total / (size_t)D;

    boat_tensor_t* output = boat_tensor_create_like(input);
    if (!output) return NULL;

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(input) == BOAT_DEVICE_CUDA &&
        boat_tensor_dtype(input) == BOAT_DTYPE_FLOAT32) {
        boat_cuda_layernorm_forward_f32((const float*)boat_tensor_const_data(input), NULL, NULL,
                                        (float*)boat_tensor_data(output), (int64_t)outer,
                                        (int64_t)D, eps);
        return output;
    }
#endif

    if (boat_tensor_dtype(input) != BOAT_DTYPE_FLOAT32) {
        boat_tensor_unref(output);
        return NULL;
    }

    const float* in = (const float*)boat_tensor_const_data(input);
    float* out = (float*)boat_tensor_data(output);

    float* mean = (float*)boat_malloc(outer * sizeof(float), BOAT_DEVICE_CPU);
    float* var = (float*)boat_malloc(outer * sizeof(float), BOAT_DEVICE_CPU);
    float* inv_std = (float*)boat_malloc(outer * sizeof(float), BOAT_DEVICE_CPU);
    if (!mean || !var || !inv_std) {
        boat_free(mean);
        boat_free(var);
        boat_free(inv_std);
        boat_tensor_unref(output);
        return NULL;
    }
    boat_simd_mean_var_f32(in, mean, var, outer, (size_t)D);
    for (size_t i = 0; i < outer; i++) {
        inv_std[i] = 1.0f / sqrtf(var[i] + eps);
    }
    boat_simd_norm_affine_f32(in, NULL, NULL, out, outer, (size_t)D, mean, inv_std);
    boat_free(mean);
    boat_free(var);
    boat_free(inv_std);

    return output;
}

// Standalone RMS norm function
BOAT_API boat_tensor_t* boat_rms_norm(const boat_tensor_t* input, const int64_t* normalized_shape,
                                      size_t normalized_shape_len, float eps) {
    if (!input || !normalized_shape || normalized_shape_len == 0) return NULL;

    size_t ndim = boat_tensor_ndim(input);
    const int64_t* shape = boat_tensor_shape(input);
    int64_t D = normalized_shape[normalized_shape_len - 1];

    // Verify last dim matches
    if (shape[ndim - 1] != D) return NULL;

    size_t total = boat_tensor_nelements(input);
    size_t outer = total / (size_t)D;

    boat_tensor_t* output = boat_tensor_create_like(input);
    if (!output) return NULL;

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(input) == BOAT_DEVICE_CUDA &&
        boat_tensor_dtype(input) == BOAT_DTYPE_FLOAT32) {
        boat_cuda_rmsnorm_forward_f32((const float*)boat_tensor_const_data(input), NULL,
                                      (float*)boat_tensor_data(output), (int64_t)outer, (int64_t)D,
                                      eps);
        return output;
    }
#endif

    if (boat_tensor_dtype(input) != BOAT_DTYPE_FLOAT32) {
        boat_tensor_unref(output);
        return NULL;
    }

    const float* in = (const float*)boat_tensor_const_data(input);
    float* out = (float*)boat_tensor_data(output);

    float* rms = (float*)boat_malloc(outer * sizeof(float), BOAT_DEVICE_CPU);
    float* inv_rms = (float*)boat_malloc(outer * sizeof(float), BOAT_DEVICE_CPU);
    if (!rms || !inv_rms) {
        boat_free(rms);
        boat_free(inv_rms);
        boat_tensor_unref(output);
        return NULL;
    }
    boat_simd_rms_f32(in, rms, outer, (size_t)D);
    for (size_t i = 0; i < outer; i++) {
        inv_rms[i] = 1.0f / (rms[i] + eps);
    }
    boat_simd_norm_affine_f32(in, NULL, NULL, out, outer, (size_t)D, NULL, inv_rms);
    boat_free(rms);
    boat_free(inv_rms);

    return output;
}

// Gradient functions
BOAT_API boat_tensor_t* boat_layer_norm_grad(const boat_tensor_t* grad_output,
                                             const boat_tensor_t* input,
                                             const boat_tensor_t* output,
                                             const int64_t* normalized_shape,
                                             size_t normalized_shape_len, float eps) {
    (void)grad_output;
    (void)input;
    (void)output;
    (void)normalized_shape;
    (void)normalized_shape_len;
    (void)eps;
    return NULL;
}

BOAT_API boat_tensor_t* boat_rms_norm_grad(const boat_tensor_t* grad_output,
                                           const boat_tensor_t* input, const boat_tensor_t* output,
                                           const int64_t* normalized_shape,
                                           size_t normalized_shape_len, float eps) {
    (void)grad_output;
    (void)input;
    (void)output;
    (void)normalized_shape;
    (void)normalized_shape_len;
    (void)eps;
    return NULL;
}

// Parameter setting for model loading
BOAT_API void boat_layernorm_set_weight(boat_layernorm_t* norm, boat_tensor_t* weight) {
    if (!norm || !weight) {
        return;
    }
    // Check weight shape matches normalized_shape
    const int64_t* weight_shape = boat_tensor_shape(weight);
    if (weight_shape[0] != (int64_t)norm->config.normalized_shape) {
        fprintf(stderr, "Error: Weight shape [%lld] does not match normalized_shape %zu\n",
                weight_shape[0], norm->config.normalized_shape);
        return;
    }
    // Replace weight tensor
    if (norm->weight) {
        boat_tensor_free(norm->weight);
    }
    norm->weight = weight;
    boat_tensor_ref(weight); // Increase ref count since layer now owns it
}

BOAT_API void boat_layernorm_set_bias(boat_layernorm_t* norm, boat_tensor_t* bias) {
    if (!norm || !bias) {
        return;
    }
    if (!norm->config.use_bias) {
        fprintf(stderr,
                "Warning: Layer normalization was created without bias, ignoring bias tensor\n");
        return;
    }
    // Check bias shape matches normalized_shape
    const int64_t* bias_shape = boat_tensor_shape(bias);
    if (bias_shape[0] != (int64_t)norm->config.normalized_shape) {
        fprintf(stderr, "Error: Bias shape [%lld] does not match normalized_shape %zu\n",
                bias_shape[0], norm->config.normalized_shape);
        return;
    }
    // Replace bias tensor
    if (norm->bias) {
        boat_tensor_free(norm->bias);
    }
    norm->bias = bias;
    boat_tensor_ref(bias); // Increase ref count since layer now owns it
}

// Adapter for generic norm layer interface (layers.h)
typedef boat_layernorm_t boat_norm_layer_t;

BOAT_API boat_norm_layer_t* BOAT_CALL boat_norm_layer_create(size_t normalized_shape, float eps,
                                                             bool elementwise_affine) {
    boat_layernorm_config_t config = {
        .normalized_shape = normalized_shape,
        .eps = eps,
        .elementwise_affine = elementwise_affine,
        .use_bias = elementwise_affine // Use bias if affine is enabled
    };
    return boat_layernorm_create(&config);
}

BOAT_API void BOAT_CALL boat_norm_layer_free(boat_norm_layer_t* layer) {
    boat_layernorm_free(layer);
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_norm_layer_forward(boat_norm_layer_t* layer, const boat_tensor_t* input) {
    return boat_layernorm_forward(layer, input);
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_norm_layer_backward(boat_norm_layer_t* layer, const boat_tensor_t* grad_output) {
    return boat_layernorm_backward(layer, grad_output);
}

BOAT_API BOAT_NOINLINE void BOAT_CALL boat_norm_layer_update(boat_norm_layer_t* layer,
                                                             float learning_rate) {
    boat_layernorm_update(layer, learning_rate);
}

BOAT_API boat_tensor_t* boat_layernorm_get_grad_weight(const boat_layernorm_t* norm) {
    if (!norm) return NULL;
    return norm->grad_weight;
}

BOAT_API boat_tensor_t* boat_layernorm_get_grad_bias(const boat_layernorm_t* norm) {
    if (!norm) return NULL;
    return norm->grad_bias;
}

BOAT_API boat_tensor_t* boat_rmsnorm_get_grad_weight(const boat_rmsnorm_t* norm) {
    if (!norm) return NULL;
    return norm->grad_weight;
}

BOAT_API void boat_rmsnorm_set_weight(boat_rmsnorm_t* norm, boat_tensor_t* weight) {
    if (!norm || !weight) return;

    const int64_t* ws = boat_tensor_shape(weight);
    if (ws[0] != (int64_t)norm->config.normalized_shape) {
        fprintf(stderr, "Error: RMSNorm weight shape [%lld] != normalized_shape %zu\n",
                (long long)ws[0], norm->config.normalized_shape);
        return;
    }

    if (norm->weight) boat_tensor_free(norm->weight);
    norm->weight = weight;
    boat_tensor_ref(weight);
}
