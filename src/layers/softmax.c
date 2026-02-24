// softmax.c - Softmax activation layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Softmax layer structure
struct boat_softmax_layer_t {
    int axis;  // Dimension along which softmax is applied
};

boat_softmax_layer_t* boat_softmax_layer_create(int axis) {
    boat_softmax_layer_t* layer = (boat_softmax_layer_t*)boat_malloc(sizeof(boat_softmax_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }

    layer->axis = axis;
    return layer;
}

void boat_softmax_layer_free(boat_softmax_layer_t* layer) {
    if (!layer) {
        return;
    }
    boat_free(layer);
}

boat_tensor_t* boat_softmax_layer_forward(boat_softmax_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        return NULL;
    }

    // Use existing boat_softmax operation
    return boat_softmax(input, layer->axis);
}

boat_tensor_t* boat_softmax_layer_backward(boat_softmax_layer_t* layer, const boat_tensor_t* grad_output) {
    (void)layer;
    (void)grad_output;
    // TODO: Implement backward pass for softmax
    return NULL;
}

void boat_softmax_layer_update(boat_softmax_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // Softmax has no parameters to update
}