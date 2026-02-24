// pool.c - Pooling layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Pooling layer structure (MaxPool2d)
struct boat_pool_layer_t {
    size_t pool_size;
    size_t stride;
    size_t padding;
};

boat_pool_layer_t* boat_pool_layer_create(size_t pool_size, size_t stride, size_t padding) {
    boat_pool_layer_t* layer = (boat_pool_layer_t*)boat_malloc(sizeof(boat_pool_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }

    layer->pool_size = pool_size;
    layer->stride = stride;
    layer->padding = padding;

    return layer;
}

void boat_pool_layer_free(boat_pool_layer_t* layer) {
    if (!layer) {
        return;
    }
    boat_free(layer);
}

boat_tensor_t* boat_pool_layer_forward(boat_pool_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        return NULL;
    }

    // Input should be 4D: [batch, channels, height, width]
    const int64_t* input_shape = boat_tensor_shape(input);
    if (boat_tensor_ndim(input) != 4) {
        fprintf(stderr, "Error: MaxPool2d expects 4D input tensor\n");
        return NULL;
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    // Calculate output dimensions
    int64_t height_out = (height + 2 * layer->padding - layer->pool_size) / layer->stride + 1;
    int64_t width_out = (width + 2 * layer->padding - layer->pool_size) / layer->stride + 1;

    // Create output tensor
    int64_t output_shape[] = { batch, channels, height_out, width_out };
    boat_tensor_t* output = boat_tensor_create(output_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    if (!output) {
        return NULL;
    }

    // For now, initialize with zeros (no actual pooling)
    float* output_data = (float*)boat_tensor_data(output);
    size_t output_elements = boat_tensor_nelements(output);
    memset(output_data, 0, output_elements * sizeof(float));

    return output;
}

boat_tensor_t* boat_pool_layer_backward(boat_pool_layer_t* layer, const boat_tensor_t* grad_output) {
    (void)layer;
    (void)grad_output;
    // TODO: Implement backward pass
    return NULL;
}

void boat_pool_layer_update(boat_pool_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // Pooling layers have no parameters to update
}