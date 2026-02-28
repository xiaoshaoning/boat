// softmax.c - Softmax activation layer implementation
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

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

BOAT_API boat_softmax_layer_t* BOAT_CALL boat_softmax_layer_create(int axis) {
    boat_softmax_layer_t* layer = (boat_softmax_layer_t*)boat_malloc(sizeof(boat_softmax_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }

    layer->axis = axis;
    return layer;
}

BOAT_API void BOAT_CALL boat_softmax_layer_free(boat_softmax_layer_t* layer) {
    if (!layer) {
        return;
    }
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_softmax_layer_forward(boat_softmax_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        return NULL;
    }

    // Use existing boat_softmax operation
    return boat_softmax(input, layer->axis);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_softmax_layer_backward(boat_softmax_layer_t* layer, const boat_tensor_t* grad_output) {
    (void)layer;

    // For softmax + cross entropy, backward gradient is just passed through
    // In our MNIST implementation, we already compute pred - one_hot(label)
    // So we just need to return the gradient output
    if (!grad_output) {
        return NULL;
    }

    // Clone the gradient to avoid reference issues
    // boat_tensor_t* clone = boat_tensor_clone(grad_output); // TODO: implement clone
    // For now, increment reference count and return same tensor
    boat_tensor_ref((boat_tensor_t*)grad_output);
    return (boat_tensor_t*)grad_output;
}

BOAT_API void BOAT_CALL boat_softmax_layer_update(boat_softmax_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // Softmax has no parameters to update
}