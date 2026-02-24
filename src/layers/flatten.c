// flatten.c - Flatten layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Flatten layer structure (no internal state needed)
struct boat_flatten_layer_t {
    char dummy; // MSVC requires at least one member
};

boat_flatten_layer_t* boat_flatten_layer_create() {
    boat_flatten_layer_t* layer = (boat_flatten_layer_t*)boat_malloc(sizeof(boat_flatten_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }
    return layer;
}

void boat_flatten_layer_free(boat_flatten_layer_t* layer) {
    if (!layer) {
        return;
    }
    boat_free(layer);
}

boat_tensor_t* boat_flatten_layer_forward(boat_flatten_layer_t* layer, const boat_tensor_t* input) {
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

    // Calculate flattened shape: [batch, product of remaining dimensions]
    int64_t batch = input_shape[0];
    int64_t features = 1;
    for (size_t i = 1; i < ndim; i++) {
        features *= input_shape[i];
    }

    int64_t output_shape[] = { batch, features };
    return boat_tensor_reshape(input, output_shape, 2);
}

boat_tensor_t* boat_flatten_layer_backward(boat_flatten_layer_t* layer, const boat_tensor_t* grad_output) {
    (void)layer;
    (void)grad_output;
    // TODO: Implement backward pass
    return NULL;
}

void boat_flatten_layer_update(boat_flatten_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // Flatten layer has no parameters to update
}