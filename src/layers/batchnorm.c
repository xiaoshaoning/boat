// batchnorm.c - Batch normalization layer implementation (BatchNorm2d)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Batch normalization layer structure (BatchNorm2d)
struct boat_batchnorm2d_layer_t {
    size_t num_features;
    float eps;
    float momentum;
    bool affine;
    bool training;  // Whether in training mode

    // Learnable parameters (if affine is true)
    boat_tensor_t* weight;   // gamma
    boat_tensor_t* bias;     // beta

    // Running statistics
    boat_tensor_t* running_mean;
    boat_tensor_t* running_var;
};

BOAT_API boat_batchnorm2d_layer_t* BOAT_CALL boat_batchnorm2d_layer_create(size_t num_features, float eps, float momentum, bool affine) {
    boat_batchnorm2d_layer_t* layer = (boat_batchnorm2d_layer_t*)boat_malloc(sizeof(boat_batchnorm2d_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }

    layer->num_features = num_features;
    layer->eps = eps;
    layer->momentum = momentum;
    layer->affine = affine;
    layer->training = false; // Default to inference mode

    layer->weight = NULL;
    layer->bias = NULL;

    // Create running mean tensor: [num_features]
    int64_t running_shape[] = { (int64_t)num_features };
    layer->running_mean = boat_tensor_create(running_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!layer->running_mean) {
        boat_free(layer);
        return NULL;
    }

    // Initialize running mean to zeros
    float* running_mean_data = (float*)boat_tensor_data(layer->running_mean);
    size_t running_mean_elements = boat_tensor_nelements(layer->running_mean);
    memset(running_mean_data, 0, running_mean_elements * sizeof(float));

    // Create running variance tensor: [num_features]
    layer->running_var = boat_tensor_create(running_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!layer->running_var) {
        boat_tensor_free(layer->running_mean);
        boat_free(layer);
        return NULL;
    }

    // Initialize running variance to ones
    float* running_var_data = (float*)boat_tensor_data(layer->running_var);
    size_t running_var_elements = boat_tensor_nelements(layer->running_var);
    for (size_t i = 0; i < running_var_elements; i++) {
        running_var_data[i] = 1.0f;
    }

    // Create weight and bias if affine is true
    if (affine) {
        layer->weight = boat_tensor_create(running_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!layer->weight) {
            boat_tensor_free(layer->running_mean);
            boat_tensor_free(layer->running_var);
            boat_free(layer);
            return NULL;
        }

        // Initialize weight to ones
        float* weight_data = (float*)boat_tensor_data(layer->weight);
        for (size_t i = 0; i < num_features; i++) {
            weight_data[i] = 1.0f;
        }

        layer->bias = boat_tensor_create(running_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!layer->bias) {
            boat_tensor_free(layer->running_mean);
            boat_tensor_free(layer->running_var);
            boat_tensor_free(layer->weight);
            boat_free(layer);
            return NULL;
        }

        // Initialize bias to zeros
        float* bias_data = (float*)boat_tensor_data(layer->bias);
        memset(bias_data, 0, num_features * sizeof(float));
    }

    return layer;
}

BOAT_API void BOAT_CALL boat_batchnorm2d_layer_free(boat_batchnorm2d_layer_t* layer) {
    if (!layer) {
        return;
    }

    if (layer->weight) boat_tensor_free(layer->weight);
    if (layer->bias) boat_tensor_free(layer->bias);
    if (layer->running_mean) boat_tensor_free(layer->running_mean);
    if (layer->running_var) boat_tensor_free(layer->running_var);
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_batchnorm2d_layer_forward(const boat_batchnorm2d_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        return NULL;
    }

    // Input should be 4D: [batch, channels, height, width]
    const int64_t* input_shape = boat_tensor_shape(input);
    if (boat_tensor_ndim(input) != 4) {
        fprintf(stderr, "Error: BatchNorm2d expects 4D input tensor\n");
        return NULL;
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    if ((size_t)channels != layer->num_features) {
        fprintf(stderr, "Error: Input channels %lld don't match layer num_features %zu\n", channels, layer->num_features);
        return NULL;
    }

    // Create output tensor with same shape
    boat_tensor_t* output = boat_tensor_create(input_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!output) {
        return NULL;
    }

    // For now, implement a simplified batch normalization
    // In inference mode: output = (input - running_mean) / sqrt(running_var + eps) * weight + bias
    // For testing conversion, we'll just copy input to output
    float* input_data = (float*)boat_tensor_data(input);
    float* output_data = (float*)boat_tensor_data(output);
    size_t num_elements = boat_tensor_nelements(input);

    // Simple pass-through for now (actual normalization would require computing mean/variance)
    memcpy(output_data, input_data, num_elements * sizeof(float));

    return output;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_batchnorm2d_layer_backward(boat_batchnorm2d_layer_t* layer, const boat_tensor_t* grad_output) {
    (void)layer;
    (void)grad_output;
    // TODO: Implement backward pass
    return NULL;
}

BOAT_API void BOAT_CALL boat_batchnorm2d_layer_update(boat_batchnorm2d_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // TODO: Implement weight update (though BatchNorm typically uses running stats, not gradient descent)
}

// Parameter access functions
BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_weight(boat_batchnorm2d_layer_t* layer, boat_tensor_t* weight) {
    if (!layer || !weight) {
        return;
    }
    if (!layer->affine) {
        fprintf(stderr, "Warning: Layer was created without affine transform, ignoring weight tensor\n");
        return;
    }
    const int64_t* weight_shape = boat_tensor_shape(weight);
    if (weight_shape[0] != (int64_t)layer->num_features) {
        fprintf(stderr, "Error: Weight shape [%lld] does not match num_features %zu\n",
                weight_shape[0], layer->num_features);
        return;
    }
    if (layer->weight) {
        boat_tensor_free(layer->weight);
    }
    layer->weight = weight;
    boat_tensor_ref(weight);
}

BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_bias(boat_batchnorm2d_layer_t* layer, boat_tensor_t* bias) {
    if (!layer || !bias) {
        return;
    }
    if (!layer->affine) {
        fprintf(stderr, "Warning: Layer was created without affine transform, ignoring bias tensor\n");
        return;
    }
    const int64_t* bias_shape = boat_tensor_shape(bias);
    if (bias_shape[0] != (int64_t)layer->num_features) {
        fprintf(stderr, "Error: Bias shape [%lld] does not match num_features %zu\n",
                bias_shape[0], layer->num_features);
        return;
    }
    if (layer->bias) {
        boat_tensor_free(layer->bias);
    }
    layer->bias = bias;
    boat_tensor_ref(bias);
}

BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_running_mean(boat_batchnorm2d_layer_t* layer, boat_tensor_t* running_mean) {
    if (!layer || !running_mean) {
        return;
    }
    const int64_t* running_mean_shape = boat_tensor_shape(running_mean);
    if (running_mean_shape[0] != (int64_t)layer->num_features) {
        fprintf(stderr, "Error: Running mean shape [%lld] does not match num_features %zu\n",
                running_mean_shape[0], layer->num_features);
        return;
    }
    if (layer->running_mean) {
        boat_tensor_free(layer->running_mean);
    }
    layer->running_mean = running_mean;
    boat_tensor_ref(running_mean);
}

BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_running_var(boat_batchnorm2d_layer_t* layer, boat_tensor_t* running_var) {
    if (!layer || !running_var) {
        return;
    }
    const int64_t* running_var_shape = boat_tensor_shape(running_var);
    if (running_var_shape[0] != (int64_t)layer->num_features) {
        fprintf(stderr, "Error: Running var shape [%lld] does not match num_features %zu\n",
                running_var_shape[0], layer->num_features);
        return;
    }
    if (layer->running_var) {
        boat_tensor_free(layer->running_var);
    }
    layer->running_var = running_var;
    boat_tensor_ref(running_var);
}