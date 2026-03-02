// flatten.c - Flatten layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Flatten layer structure
struct boat_flatten_layer_t {
    int64_t* cached_shape;    // Shape of input tensor from forward pass
    size_t cached_ndim;       // Number of dimensions
};

BOAT_API boat_flatten_layer_t* BOAT_CALL boat_flatten_layer_create() {
    boat_flatten_layer_t* layer = (boat_flatten_layer_t*)boat_malloc(sizeof(boat_flatten_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }
    layer->cached_shape = NULL;
    layer->cached_ndim = 0;
    return layer;
}

BOAT_API void BOAT_CALL boat_flatten_layer_free(boat_flatten_layer_t* layer) {
    if (!layer) {
        return;
    }
    if (layer->cached_shape) {
        boat_free(layer->cached_shape);
    }
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_flatten_layer_forward(boat_flatten_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        return NULL;
    }

    // Get input shape
    const int64_t* input_shape = boat_tensor_shape(input);
    size_t ndim = boat_tensor_ndim(input);

    if (ndim < 2) {
        fprintf(stderr, "Error: Flatten expects at least 2D input tensor\n");
        return NULL;
    }

    // Cache input shape for backward pass
    if (layer->cached_shape) {
        boat_free(layer->cached_shape);
    }
    layer->cached_shape = (int64_t*)boat_malloc(sizeof(int64_t) * ndim, BOAT_DEVICE_CPU);
    if (!layer->cached_shape) {
        return NULL;
    }
    for (size_t i = 0; i < ndim; i++) {
        layer->cached_shape[i] = input_shape[i];
    }
    layer->cached_ndim = ndim;

    // Calculate flattened shape: [batch, product of remaining dimensions]
    int64_t batch = input_shape[0];
    int64_t features = 1;
    for (size_t i = 1; i < ndim; i++) {
        features *= input_shape[i];
    }

    const int64_t output_shape[] = { batch, features };
    return boat_tensor_reshape(input, output_shape, 2);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_flatten_layer_backward(const boat_flatten_layer_t* layer, const boat_tensor_t* grad_output) {
    if (!layer || !grad_output) {
        return NULL;
    }

    // Check if shape is cached
    if (!layer->cached_shape || layer->cached_ndim == 0) {
        fprintf(stderr, "Error: Flatten backward called without cached shape (forward not called)\n");
        return NULL;
    }

    // Reshape gradient back to original input shape
    return boat_tensor_reshape(grad_output, layer->cached_shape, layer->cached_ndim);
}

BOAT_API void BOAT_CALL boat_flatten_layer_update(boat_flatten_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // Flatten layer has no parameters to update
}