// relu.c - ReLU activation layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ReLU layer structure (no parameters, just operations)
struct boat_relu_layer_t {
    char dummy; // MSVC requires at least one member
};

boat_relu_layer_t* boat_relu_layer_create() {
    boat_relu_layer_t* layer = (boat_relu_layer_t*)boat_malloc(sizeof(boat_relu_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }
    return layer;
}

void boat_relu_layer_free(boat_relu_layer_t* layer) {
    if (!layer) {
        return;
    }
    boat_free(layer);
}

boat_tensor_t* boat_relu_layer_forward(boat_relu_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        return NULL;
    }
    // Apply element-wise ReLU: max(0, x)
    return boat_relu(input);
}

boat_tensor_t* boat_relu_layer_backward(boat_relu_layer_t* layer, const boat_tensor_t* grad_output) {
    (void)layer;
    (void)grad_output;
    // TODO: Implement backward pass
    return NULL;
}

void boat_relu_layer_update(boat_relu_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // ReLU has no parameters to update
}