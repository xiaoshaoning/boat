// norm.c - Normalization layers implementation
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat/layers/norm.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Layer normalization structure
struct boat_layernorm_t {
    boat_layernorm_config_t config;

    // Learnable parameters (if elementwise_affine is true)
    boat_tensor_t* weight;  // gamma
    boat_tensor_t* bias;    // beta

    // Cache for backward pass
    boat_tensor_t* cache_input;
    boat_tensor_t* cache_mean;
    boat_tensor_t* cache_variance;
};

// RMS normalization structure
struct boat_rmsnorm_t {
    boat_rmsnorm_config_t config;

    // Learnable scale (if elementwise_affine is true)
    boat_tensor_t* weight;  // gamma

    // Cache for backward pass
    boat_tensor_t* cache_input;
    boat_tensor_t* cache_rms;
};

// Helper function to create weight tensor
static boat_tensor_t* create_weight_tensor(size_t normalized_shape) {
    int64_t shape[] = { (int64_t)normalized_shape };
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
    int64_t shape[] = { (int64_t)normalized_shape };
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

boat_layernorm_t* boat_layernorm_create(const boat_layernorm_config_t* config) {
    if (!config || config->normalized_shape == 0) {
        return NULL;
    }

    boat_layernorm_t* norm = (boat_layernorm_t*)boat_malloc(sizeof(boat_layernorm_t), BOAT_DEVICE_CPU);
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

    return norm;
}

void boat_layernorm_free(boat_layernorm_t* norm) {
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

    // Free norm structure
    boat_free(norm);
}

boat_rmsnorm_t* boat_rmsnorm_create(const boat_rmsnorm_config_t* config) {
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

    return norm;
}

void boat_rmsnorm_free(boat_rmsnorm_t* norm) {
    if (!norm) {
        return;
    }

    // Free learnable parameters
    if (norm->weight) boat_tensor_free(norm->weight);

    // Free cache tensors
    if (norm->cache_input) boat_tensor_free(norm->cache_input);
    if (norm->cache_rms) boat_tensor_free(norm->cache_rms);

    // Free norm structure
    boat_free(norm);
}

// Helper function to compute mean and variance along last dimension
static void compute_mean_variance(const float* input, size_t batch_size, size_t seq_len, size_t hidden_size,
                                  float* mean, float* variance) {
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            size_t offset = b * seq_len * hidden_size + s * hidden_size;

            // Compute mean
            float sum = 0.0f;
            for (size_t h = 0; h < hidden_size; h++) {
                sum += input[offset + h];
            }
            float m = sum / hidden_size;

            // Compute variance
            float var_sum = 0.0f;
            for (size_t h = 0; h < hidden_size; h++) {
                float diff = input[offset + h] - m;
                var_sum += diff * diff;
            }
            float v = var_sum / hidden_size;

            mean[b * seq_len + s] = m;
            variance[b * seq_len + s] = v;
        }
    }
}

// Helper function to compute RMS (root mean square) along last dimension
static void compute_rms(const float* input, size_t batch_size, size_t seq_len, size_t hidden_size,
                        float* rms) {
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            size_t offset = b * seq_len * hidden_size + s * hidden_size;

            // Compute sum of squares
            float sum_sq = 0.0f;
            for (size_t h = 0; h < hidden_size; h++) {
                float val = input[offset + h];
                sum_sq += val * val;
            }

            // Compute RMS
            rms[b * seq_len + s] = sqrtf(sum_sq / hidden_size);
        }
    }
}

boat_tensor_t* boat_layernorm_forward(boat_layernorm_t* norm, const boat_tensor_t* input) {
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
    const float* weight_data = norm->weight ? (const float*)boat_tensor_const_data(norm->weight) : NULL;
    const float* bias_data = norm->bias ? (const float*)boat_tensor_const_data(norm->bias) : NULL;

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            size_t offset = b * seq_len * hidden_size + s * hidden_size;
            size_t idx = b * seq_len + s;

            float m = mean[idx];
            float v = variance[idx];
            float scale = 1.0f / sqrtf(v + eps);

            for (size_t h = 0; h < hidden_size; h++) {
                float normalized = (input_data[offset + h] - m) * scale;

                // Apply affine transformation if enabled
                if (weight_data) {
                    normalized = normalized * weight_data[h];
                }
                if (bias_data) {
                    normalized = normalized + bias_data[h];
                }

                output_data[offset + h] = normalized;
            }
        }
    }

    // Free temporary arrays
    boat_free(mean);
    boat_free(variance);

    return output;
}

boat_tensor_t* boat_rmsnorm_forward(boat_rmsnorm_t* norm, const boat_tensor_t* input) {
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
    const float* weight_data = norm->weight ? (const float*)boat_tensor_const_data(norm->weight) : NULL;

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            size_t offset = b * seq_len * hidden_size + s * hidden_size;
            size_t idx = b * seq_len + s;

            float r = rms[idx];
            float scale = 1.0f / (r + eps);

            for (size_t h = 0; h < hidden_size; h++) {
                float normalized = input_data[offset + h] * scale;

                // Apply scale if enabled
                if (weight_data) {
                    normalized = normalized * weight_data[h];
                }

                output_data[offset + h] = normalized;
            }
        }
    }

    // Free temporary array
    boat_free(rms);

    return output;
}

boat_tensor_t* boat_layernorm_backward(boat_layernorm_t* norm, const boat_tensor_t* grad_output) {
    (void)norm;
    (void)grad_output;
    // TODO: Implement backward pass
    return NULL;
}

boat_tensor_t* boat_rmsnorm_backward(boat_rmsnorm_t* norm, const boat_tensor_t* grad_output) {
    (void)norm;
    (void)grad_output;
    // TODO: Implement backward pass
    return NULL;
}

void boat_layernorm_update(boat_layernorm_t* norm, float learning_rate) {
    (void)norm;
    (void)learning_rate;
    // TODO: Implement parameter update
}

void boat_rmsnorm_update(boat_rmsnorm_t* norm, float learning_rate) {
    (void)norm;
    (void)learning_rate;
    // TODO: Implement parameter update
}

// Standalone layer norm function
boat_tensor_t* boat_layer_norm(const boat_tensor_t* input,
                                const int64_t* normalized_shape,
                                size_t normalized_shape_len,
                                float eps) {
    (void)input;
    (void)normalized_shape;
    (void)normalized_shape_len;
    (void)eps;
    // TODO: Implement standalone layer norm
    return NULL;
}

// Standalone RMS norm function
boat_tensor_t* boat_rms_norm(const boat_tensor_t* input,
                              const int64_t* normalized_shape,
                              size_t normalized_shape_len,
                              float eps) {
    (void)input;
    (void)normalized_shape;
    (void)normalized_shape_len;
    (void)eps;
    // TODO: Implement standalone RMS norm
    return NULL;
}

// Gradient functions
boat_tensor_t* boat_layer_norm_grad(const boat_tensor_t* grad_output,
                                     const boat_tensor_t* input,
                                     const boat_tensor_t* output,
                                     const int64_t* normalized_shape,
                                     size_t normalized_shape_len,
                                     float eps) {
    (void)grad_output;
    (void)input;
    (void)output;
    (void)normalized_shape;
    (void)normalized_shape_len;
    (void)eps;
    return NULL;
}

boat_tensor_t* boat_rms_norm_grad(const boat_tensor_t* grad_output,
                                   const boat_tensor_t* input,
                                   const boat_tensor_t* output,
                                   const int64_t* normalized_shape,
                                   size_t normalized_shape_len,
                                   float eps) {
    (void)grad_output;
    (void)input;
    (void)output;
    (void)normalized_shape;
    (void)normalized_shape_len;
    (void)eps;
    return NULL;
}

// Parameter setting for model loading
void boat_layernorm_set_weight(boat_layernorm_t* norm, boat_tensor_t* weight) {
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

void boat_layernorm_set_bias(boat_layernorm_t* norm, boat_tensor_t* bias) {
    if (!norm || !bias) {
        return;
    }
    if (!norm->config.use_bias) {
        fprintf(stderr, "Warning: Layer normalization was created without bias, ignoring bias tensor\n");
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

BOAT_API boat_norm_layer_t* BOAT_CALL boat_norm_layer_create(size_t normalized_shape, float eps, bool elementwise_affine) {
    boat_layernorm_config_t config = {
        .normalized_shape = normalized_shape,
        .eps = eps,
        .elementwise_affine = elementwise_affine,
        .use_bias = elementwise_affine  // Use bias if affine is enabled
    };
    return boat_layernorm_create(&config);
}

BOAT_API void BOAT_CALL boat_norm_layer_free(boat_norm_layer_t* layer) {
    boat_layernorm_free(layer);
}

BOAT_NOINLINE BOAT_API boat_tensor_t* BOAT_CALL boat_norm_layer_forward(boat_norm_layer_t* layer, const boat_tensor_t* input) {
    return boat_layernorm_forward(layer, input);
}

BOAT_NOINLINE BOAT_API boat_tensor_t* BOAT_CALL boat_norm_layer_backward(boat_norm_layer_t* layer, const boat_tensor_t* grad_output) {
    return boat_layernorm_backward(layer, grad_output);
}

BOAT_NOINLINE BOAT_API void BOAT_CALL boat_norm_layer_update(boat_norm_layer_t* layer, float learning_rate) {
    boat_layernorm_update(layer, learning_rate);
}