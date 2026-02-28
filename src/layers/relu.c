// relu.c - ReLU activation layer implementation
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>
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

// ReLU layer structure (no parameters, just operations)
struct boat_relu_layer_t {
    char dummy; // MSVC requires at least one member
};

BOAT_API boat_relu_layer_t* BOAT_CALL boat_relu_layer_create() {
    boat_relu_layer_t* layer = (boat_relu_layer_t*)boat_malloc(sizeof(boat_relu_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        BOAT_DEBUG_PRINT("DEBUG boat_relu_layer_create: allocation failed\n");
        return NULL;
    }
    BOAT_DEBUG_PRINT("DEBUG boat_relu_layer_create: returning %p\n", (void*)layer);
    return layer;
}

BOAT_API void BOAT_CALL boat_relu_layer_free(boat_relu_layer_t* layer) {
    if (!layer) {
        return;
    }
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_relu_layer_forward(boat_relu_layer_t* layer, const boat_tensor_t* input) {
    BOAT_DEBUG_PRINT("DEBUG relu_layer_forward: ENTER, layer=%p, input=%p\n", (void*)layer, (void*)input);
    if (!layer || !input) {
        BOAT_DEBUG_PRINT("DEBUG relu_layer_forward: NULL input or layer\n");
        return NULL;
    }
    BOAT_DEBUG_PRINT("DEBUG relu_layer_forward: calling boat_relu\n");
    // Apply element-wise ReLU: max(0, x)
    boat_tensor_t* result = boat_relu(input);
    fprintf(stderr, "DEBUG relu_layer_forward: boat_relu returned %p\n", (void*)result);
    return result;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_relu_layer_backward(boat_relu_layer_t* layer, const boat_tensor_t* grad_output) {
    (void)layer;

    // Simple backward pass for ReLU: grad_output * mask(where input > 0)
    // For now, just pass gradient through (will need input cache for proper implementation)
    if (!grad_output) {
        return NULL;
    }

    // Clone the gradient to avoid reference issues
    // return boat_tensor_clone(grad_output); // TODO: implement clone
    // For now, increment reference count and return same tensor
    boat_tensor_ref((boat_tensor_t*)grad_output);
    return (boat_tensor_t*)grad_output;
}

BOAT_API void BOAT_CALL boat_relu_layer_update(boat_relu_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // ReLU has no parameters to update
}