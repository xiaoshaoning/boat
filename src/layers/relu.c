// relu.c - ReLU activation layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <stdlib.h>
#include <string.h>

#ifdef BOAT_WITH_CUDA
#include <boat/cuda_runtime.h>
#endif

// ReLU layer structure
struct boat_relu_layer_t {
    boat_tensor_t* cache_input;  // Input from forward pass (for backward masking)
};

BOAT_API boat_relu_layer_t* BOAT_CALL boat_relu_layer_create() {
    boat_relu_layer_t* layer = (boat_relu_layer_t*)boat_malloc(sizeof(boat_relu_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }
    layer->cache_input = NULL;
    return layer;
}

BOAT_API void BOAT_CALL boat_relu_layer_free(boat_relu_layer_t* layer) {
    if (!layer) {
        return;
    }
    if (layer->cache_input) {
        boat_tensor_unref(layer->cache_input);
    }
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_relu_layer_forward(boat_relu_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[ReLULayer] NULL input or layer\n");
        return NULL;
    }

    // Clear previous cache if exists
    if (layer->cache_input) {
        boat_tensor_unref(layer->cache_input);
        layer->cache_input = NULL;
    }

    // Apply element-wise ReLU: max(0, x)
    boat_tensor_t* result = boat_relu(input);
    if (!result) {
        return NULL;
    }

    // Cache input for backward pass
    layer->cache_input = (boat_tensor_t*)input;
    boat_tensor_ref(layer->cache_input);

    return result;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_relu_layer_backward(boat_relu_layer_t* layer, const boat_tensor_t* grad_output) {
    if (!layer || !grad_output || !layer->cache_input) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[ReLULayer] NULL input, gradient, or missing cache\n");
        return NULL;
    }

    // Create gradient input tensor
    boat_tensor_t* grad_input = boat_tensor_create_like(grad_output);
    if (!grad_input) {
        return NULL;
    }

    const float* input_data = (const float*)boat_tensor_const_data(layer->cache_input);
    const float* grad_output_data = (const float*)boat_tensor_const_data(grad_output);
    float* grad_input_data = (float*)boat_tensor_data(grad_input);
    size_t n = boat_tensor_nelements(grad_output);

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(grad_output) == BOAT_DEVICE_CUDA) {
        boat_cuda_relu_backward_f32(input_data, grad_output_data, grad_input_data, n);
        return grad_input;
    }
#endif

    // CPU: grad_input[i] = (input[i] > 0) ? grad_output[i] : 0.0f
    for (size_t i = 0; i < n; i++) {
        grad_input_data[i] = (input_data[i] > 0.0f) ? grad_output_data[i] : 0.0f;
    }

    return grad_input;
}

BOAT_API void BOAT_CALL boat_relu_layer_update(boat_relu_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // ReLU has no parameters to update
}
