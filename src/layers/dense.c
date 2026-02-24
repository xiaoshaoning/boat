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

// Dense layer structure
struct boat_dense_layer_t {
    size_t input_features;
    size_t output_features;
    boat_tensor_t* weight;
    boat_tensor_t* bias;
    bool use_bias;
};

BOAT_API boat_dense_layer_t* BOAT_CALL boat_dense_layer_create(size_t input_features, size_t output_features, bool use_bias) {
    boat_dense_layer_t* layer = (boat_dense_layer_t*)boat_malloc(sizeof(boat_dense_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }

    layer->input_features = input_features;
    layer->output_features = output_features;
    layer->use_bias = use_bias;

    // Create weight tensor
    int64_t weight_shape[] = { (int64_t)input_features, (int64_t)output_features };
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
    if (use_bias) {
        int64_t bias_shape[] = { (int64_t)output_features };
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

    return layer;
}

BOAT_API void BOAT_CALL boat_dense_layer_free(boat_dense_layer_t* layer) {
    if (!layer) {
        return;
    }

    if (layer->weight) boat_tensor_free(layer->weight);
    if (layer->bias) boat_tensor_free(layer->bias);
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_forward(boat_dense_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        return NULL;
    }

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
        float* bias_data = (float*)boat_tensor_data(layer->bias);

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
    (void)layer;
    (void)grad_output;
    // TODO: Implement backward pass
    return NULL;
}

BOAT_NOINLINE BOAT_API void BOAT_CALL boat_dense_layer_update(boat_dense_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // TODO: Implement weight update
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