// dense.c - Dense (fully connected) layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Debug output control
#ifndef BOAT_DEBUG
#define BOAT_DEBUG 0
#endif

#if BOAT_DEBUG
#define BOAT_DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define BOAT_DEBUG_PRINT(...) ((void)0)
#endif

// Dense layer structure
struct boat_dense_layer_t {
    size_t input_features;
    size_t output_features;
    boat_tensor_t* weight;
    boat_tensor_t* bias;
    bool use_bias;

    // Gradient accumulators for training
    boat_tensor_t* grad_weight;
    boat_tensor_t* grad_bias;

    // Cache for backward pass
    boat_tensor_t* cache_input;  // Input tensor from forward pass
};

BOAT_API boat_dense_layer_t* BOAT_CALL boat_dense_layer_create(size_t input_features, size_t output_features, bool use_bias) {
    BOAT_DEBUG_PRINT("DEBUG dense_create called: in=%zu, out=%zu, bias=%d\n", input_features, output_features, use_bias);
    boat_dense_layer_t* layer = (boat_dense_layer_t*)boat_malloc(sizeof(boat_dense_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        BOAT_DEBUG_PRINT("DEBUG dense_create: malloc failed\n");
        return NULL;
    }

    layer->input_features = input_features;
    layer->output_features = output_features;
    layer->use_bias = use_bias;
    layer->cache_input = NULL;

    // Create weight tensor
    const int64_t weight_shape[] = { (int64_t)input_features, (int64_t)output_features };
    layer->weight = boat_tensor_create(weight_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!layer->weight) {
        boat_free(layer);
        return NULL;
    }

    // Initialize weights (Xavier/Glorot initialization)
    float* weight_data = (float*)boat_tensor_data(layer->weight);
    size_t weight_elements = boat_tensor_nelements(layer->weight);
    float scale = sqrtf(2.0f / (input_features + output_features));
    for (size_t i = 0; i < weight_elements; i++) {
        weight_data[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }

    // Create bias tensor if requested
    const int64_t bias_shape[] = { (int64_t)output_features };
    if (use_bias) {
        layer->bias = boat_tensor_create(bias_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!layer->bias) {
            boat_tensor_free(layer->weight);
            boat_free(layer);
            return NULL;
        }

        // Initialize bias to zeros
        float* bias_data = (float*)boat_tensor_data(layer->bias);
        size_t bias_elements = boat_tensor_nelements(layer->bias);
        memset(bias_data, 0, bias_elements * sizeof(float));
    } else {
        layer->bias = NULL;
    }

    // Create gradient weight tensor with same shape as weight
    layer->grad_weight = boat_tensor_create(weight_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    BOAT_DEBUG_PRINT("DEBUG dense_create: grad_weight tensor at %p\n", layer->grad_weight);
    if (!layer->grad_weight) {
        boat_tensor_free(layer->weight);
        if (layer->bias) boat_tensor_free(layer->bias);
        boat_free(layer);
        return NULL;
    }
    // Initialize gradient weight with zeros
    float* grad_weight_data = (float*)boat_tensor_data(layer->grad_weight);
    size_t grad_weight_elements = boat_tensor_nelements(layer->grad_weight);
    memset(grad_weight_data, 0, grad_weight_elements * sizeof(float));

    // Create gradient bias tensor if bias is used
    if (use_bias) {
        layer->grad_bias = boat_tensor_create(bias_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!layer->grad_bias) {
            boat_tensor_free(layer->weight);
            if (layer->bias) boat_tensor_free(layer->bias);
            boat_tensor_free(layer->grad_weight);
            boat_free(layer);
            return NULL;
        }
        // Initialize gradient bias with zeros
        float* grad_bias_data = (float*)boat_tensor_data(layer->grad_bias);
        size_t grad_bias_elements = boat_tensor_nelements(layer->grad_bias);
        memset(grad_bias_data, 0, grad_bias_elements * sizeof(float));
    } else {
        layer->grad_bias = NULL;
    }

    return layer;
}

BOAT_API void BOAT_CALL boat_dense_layer_free(boat_dense_layer_t* layer) {
    if (!layer) {
        return;
    }

    if (layer->weight) boat_tensor_free(layer->weight);
    if (layer->bias) boat_tensor_free(layer->bias);
    if (layer->grad_weight) boat_tensor_free(layer->grad_weight);
    if (layer->grad_bias) boat_tensor_free(layer->grad_bias);
    if (layer->cache_input) boat_tensor_unref(layer->cache_input);
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_forward(boat_dense_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        return NULL;
    }

    // Clear previous cache if exists
    if (layer->cache_input) {
        boat_tensor_unref(layer->cache_input);
        layer->cache_input = NULL;
    }
    // Cache input tensor (increase ref count)
    layer->cache_input = (boat_tensor_t*)input;
    boat_tensor_ref(layer->cache_input);

    // Perform matrix multiplication: input @ weight
    boat_tensor_t* output = boat_matmul(input, layer->weight);
    if (!output) {
        return NULL;
    }

    // Add bias if present
    if (layer->use_bias && layer->bias) {
        // Manual bias addition with broadcasting
        // output shape: [batch, output_features], bias shape: [output_features]
        const int64_t* out_shape = boat_tensor_shape(output);
        size_t out_ndim = boat_tensor_ndim(output);
        if (out_ndim != 2) {
            boat_tensor_free(output);
            return NULL;
        }
        int64_t batch = out_shape[0];
        int64_t out_features = out_shape[1];

        // Check bias shape matches output_features
        const int64_t* bias_shape = boat_tensor_shape(layer->bias);
        if (bias_shape[0] != out_features) {
            boat_tensor_free(output);
            return NULL;
        }

        // Get data pointers
        float* out_data = (float*)boat_tensor_data(output);
        const float* bias_data = (float*)boat_tensor_data(layer->bias);

        // Add bias to each row
        for (int64_t i = 0; i < batch; i++) {
            for (int64_t j = 0; j < out_features; j++) {
                out_data[i * out_features + j] += bias_data[j];
            }
        }
    }

    return output;
}

BOAT_NOINLINE BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_backward(boat_dense_layer_t* layer, const boat_tensor_t* grad_output) {
    if (!layer || !grad_output || !layer->cache_input) {
        return NULL;
    }

    // Get cached input
    const boat_tensor_t* input = layer->cache_input;

    // Check input shape: [batch, input_features]
    // Check grad_output shape: [batch, output_features]
    const int64_t* input_shape = boat_tensor_shape(input);
    const int64_t* grad_output_shape = boat_tensor_shape(grad_output);

    if (boat_tensor_ndim(input) != 2 || boat_tensor_ndim(grad_output) != 2) {
        fprintf(stderr, "Error: Dense layer backward expects 2D tensors\n");
        return NULL;
    }

    int64_t batch = input_shape[0];
    int64_t input_features = input_shape[1];
    int64_t output_features = grad_output_shape[1];

    if (input_features != (int64_t)layer->input_features ||
        output_features != (int64_t)layer->output_features ||
        grad_output_shape[0] != batch) {
        fprintf(stderr, "Error: Dense layer backward shape mismatch\n");
        return NULL;
    }

    // Compute gradient with respect to input: grad_input = grad_output @ weight^T
    boat_tensor_t* weight_T = boat_transpose(layer->weight, 0, 1);
    if (!weight_T) {
        return NULL;
    }
    boat_tensor_t* grad_input = boat_matmul(grad_output, weight_T);
    boat_tensor_unref(weight_T);
    if (!grad_input) {
        return NULL;
    }

    // Compute gradient with respect to weight: grad_weight = input^T @ grad_output
    boat_tensor_t* input_T = boat_transpose(input, 0, 1);
    if (!input_T) {
        boat_tensor_unref(grad_input);
        return NULL;
    }

    // Create or update grad_weight tensor
    if (!layer->grad_weight) {
        layer->grad_weight = boat_tensor_create_like(layer->weight);
        if (!layer->grad_weight) {
            boat_tensor_unref(input_T);
            boat_tensor_unref(grad_input);
            return NULL;
        }
    }

    // Compute weight gradient: grad_weight = input^T @ grad_output
    boat_tensor_t* new_grad_weight = boat_matmul(input_T, grad_output);
    boat_tensor_unref(input_T);
    if (!new_grad_weight) {
        boat_tensor_unref(grad_input);
        return NULL;
    }

    // Accumulate gradient (or replace if first time)
    if (layer->grad_weight) {
        boat_add_(layer->grad_weight, new_grad_weight);
        boat_tensor_unref(new_grad_weight);
    } else {
        layer->grad_weight = new_grad_weight;
    }

    // Compute gradient with respect to bias if present
    if (layer->use_bias && layer->bias) {
        if (!layer->grad_bias) {
            layer->grad_bias = boat_tensor_create_like(layer->bias);
            if (!layer->grad_bias) {
                // grad_weight already stored, keep it
                return grad_input;
            }
        }

        // grad_bias = sum(grad_output, axis=0)
        // For simplicity, we'll compute sum manually
        const float* grad_output_data = (const float*)boat_tensor_data(grad_output);
        float* grad_bias_data = (float*)boat_tensor_data(layer->grad_bias);
        size_t bias_elements = boat_tensor_nelements(layer->grad_bias);

        // Initialize with zeros if first accumulation
        static bool first_bias_grad = true;
        if (first_bias_grad) {
            memset(grad_bias_data, 0, bias_elements * sizeof(float));
            first_bias_grad = false;
        }

        // Accumulate gradients across batch dimension
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t of = 0; of < output_features; of++) {
                grad_bias_data[of] += grad_output_data[b * output_features + of];
            }
        }
    }

    return grad_input;
}

BOAT_NOINLINE BOAT_API void BOAT_CALL boat_dense_layer_update(boat_dense_layer_t* layer, float learning_rate) {
    if (!layer) return;

    // Update weight: weight = weight - learning_rate * grad_weight
    if (layer->grad_weight && layer->weight) {
        // weight = weight - learning_rate * grad_weight
        // For simplicity, we'll do manual update
        float* weight_data = (float*)boat_tensor_data(layer->weight);
        float* grad_weight_data = (float*)boat_tensor_data(layer->grad_weight);
        size_t weight_elements = boat_tensor_nelements(layer->weight);

        for (size_t i = 0; i < weight_elements; i++) {
            weight_data[i] -= learning_rate * grad_weight_data[i];
        }

        // Clear gradient after update
        memset(grad_weight_data, 0, weight_elements * sizeof(float));
    }

    // Update bias if present
    if (layer->use_bias && layer->bias && layer->grad_bias) {
        float* bias_data = (float*)boat_tensor_data(layer->bias);
        float* grad_bias_data = (float*)boat_tensor_data(layer->grad_bias);
        size_t bias_elements = boat_tensor_nelements(layer->bias);

        for (size_t i = 0; i < bias_elements; i++) {
            bias_data[i] -= learning_rate * grad_bias_data[i];
        }

        // Clear gradient after update
        memset(grad_bias_data, 0, bias_elements * sizeof(float));
    }
}

BOAT_API void BOAT_CALL boat_dense_layer_set_weight(boat_dense_layer_t* layer, boat_tensor_t* weight) {
    if (!layer || !weight) {
        return;
    }
    // Check weight shape matches layer dimensions
    const int64_t* weight_shape = boat_tensor_shape(weight);
    if (weight_shape[0] != (int64_t)layer->input_features ||
        weight_shape[1] != (int64_t)layer->output_features) {
        fprintf(stderr, "Error: Weight shape [%lld, %lld] does not match layer dimensions [%zu, %zu]\n",
                weight_shape[0], weight_shape[1], layer->input_features, layer->output_features);
        return;
    }
    // Replace weight tensor
    if (layer->weight) {
        boat_tensor_free(layer->weight);
    }
    layer->weight = weight;
    boat_tensor_ref(weight); // Increase ref count since layer now owns it
}

BOAT_API void BOAT_CALL boat_dense_layer_set_bias(boat_dense_layer_t* layer, boat_tensor_t* bias) {
    if (!layer || !bias) {
        return;
    }
    if (!layer->use_bias) {
        fprintf(stderr, "Warning: Layer was created without bias, ignoring bias tensor\n");
        return;
    }
    // Check bias shape matches output features
    const int64_t* bias_shape = boat_tensor_shape(bias);
    if (bias_shape[0] != (int64_t)layer->output_features) {
        fprintf(stderr, "Error: Bias shape [%lld] does not match output features %zu\n",
                bias_shape[0], layer->output_features);
        return;
    }
    // Replace bias tensor
    if (layer->bias) {
        boat_tensor_free(layer->bias);
    }
    layer->bias = bias;
    boat_tensor_ref(bias);
}

// Get weight tensor
BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_get_weight(const boat_dense_layer_t* layer) {
    if (!layer) return NULL;
    return layer->weight;
}

// Get bias tensor
BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_get_bias(const boat_dense_layer_t* layer) {
    if (!layer) return NULL;
    return layer->bias;
}

// Get weight gradient tensor
BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_get_grad_weight(const boat_dense_layer_t* layer) {
    if (!layer) return NULL;
    return layer->grad_weight;
}

// Get bias gradient tensor
BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_get_grad_bias(const boat_dense_layer_t* layer) {
    if (!layer) return NULL;
    return layer->grad_bias;
}