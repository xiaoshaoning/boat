// pool.c - Pooling layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#define _USE_MATH_DEFINES  // For INFINITY on Windows
#include <math.h>

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

    // Cache for backward pass
    boat_tensor_t* cache_input;  // Input tensor from forward pass
    int64_t* max_indices;        // Indices of max values in input (flattened indices)
    int64_t cache_batch;
    int64_t cache_channels;
    int64_t cache_height;
    int64_t cache_width;
    int64_t cache_height_out;
    int64_t cache_width_out;
};

BOAT_API boat_pool_layer_t* BOAT_CALL boat_pool_layer_create(size_t pool_size, size_t stride, size_t padding) {
    boat_pool_layer_t* layer = (boat_pool_layer_t*)boat_malloc(sizeof(boat_pool_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }

    layer->pool_size = pool_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->cache_input = NULL;
    layer->max_indices = NULL;
    layer->cache_batch = 0;
    layer->cache_channels = 0;
    layer->cache_height = 0;
    layer->cache_width = 0;
    layer->cache_height_out = 0;
    layer->cache_width_out = 0;

    return layer;
}

BOAT_API void BOAT_CALL boat_pool_layer_free(boat_pool_layer_t* layer) {
    if (!layer) {
        return;
    }
    if (layer->cache_input) {
        boat_tensor_unref(layer->cache_input);
    }
    if (layer->max_indices) {
        boat_free(layer->max_indices);
    }
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_pool_layer_forward(boat_pool_layer_t* layer, const boat_tensor_t* input) {
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
    const int64_t output_shape[] = { batch, channels, height_out, width_out };
    boat_tensor_t* output = boat_tensor_create(output_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!output) {
        return NULL;
    }

    // Get data pointers
    const float* input_data = (const float*)boat_tensor_data(input);
    float* output_data = (float*)boat_tensor_data(output);

    // Clear previous cache if exists
    if (layer->cache_input) {
        boat_tensor_unref(layer->cache_input);
        layer->cache_input = NULL;
    }
    if (layer->max_indices) {
        boat_free(layer->max_indices);
        layer->max_indices = NULL;
    }

    // Cache input tensor (increase ref count)
    layer->cache_input = (boat_tensor_t*)input;
    boat_tensor_ref(layer->cache_input);

    // Allocate max indices array (one index per output element)
    size_t output_elements = batch * channels * height_out * width_out;
    layer->max_indices = (int64_t*)boat_malloc(output_elements * sizeof(int64_t), BOAT_DEVICE_CPU);
    if (!layer->max_indices) {
        boat_tensor_unref(output);
        boat_tensor_unref(layer->cache_input);
        layer->cache_input = NULL;
        return NULL;
    }

    // Store dimensions for backward pass
    layer->cache_batch = batch;
    layer->cache_channels = channels;
    layer->cache_height = height;
    layer->cache_width = width;
    layer->cache_height_out = height_out;
    layer->cache_width_out = width_out;

    // Perform MaxPool2d
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t c = 0; c < channels; c++) {
            for (int64_t oh = 0; oh < height_out; oh++) {
                for (int64_t ow = 0; ow < width_out; ow++) {
                    int64_t h_start = oh * layer->stride - layer->padding;
                    int64_t w_start = ow * layer->stride - layer->padding;
                    int64_t h_end = h_start + layer->pool_size;
                    int64_t w_end = w_start + layer->pool_size;

                    // Clamp to valid range
                    h_start = h_start < 0 ? 0 : h_start;
                    w_start = w_start < 0 ? 0 : w_start;
                    h_end = h_end > height ? height : h_end;
                    w_end = w_end > width ? width : w_end;

                    // Find max value in pooling window
                    float max_val = -INFINITY;
                    int64_t max_idx = -1;

                    for (int64_t ph = h_start; ph < h_end; ph++) {
                        for (int64_t pw = w_start; pw < w_end; pw++) {
                            int64_t input_idx = ((b * channels + c) * height + ph) * width + pw;
                            float val = input_data[input_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = input_idx;
                            }
                        }
                    }

                    // Store output and max index
                    int64_t output_idx = ((b * channels + c) * height_out + oh) * width_out + ow;
                    output_data[output_idx] = max_val;
                    layer->max_indices[output_idx] = max_idx;
                }
            }
        }
    }

    return output;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_pool_layer_backward(boat_pool_layer_t* layer, const boat_tensor_t* grad_output) {
    if (!layer || !grad_output || !layer->cache_input || !layer->max_indices) {
        return NULL;
    }

    // Check gradient output shape matches cached output shape
    const int64_t* grad_output_shape = boat_tensor_shape(grad_output);
    if (boat_tensor_ndim(grad_output) != 4 ||
        grad_output_shape[0] != layer->cache_batch ||
        grad_output_shape[1] != layer->cache_channels ||
        grad_output_shape[2] != layer->cache_height_out ||
        grad_output_shape[3] != layer->cache_width_out) {
        fprintf(stderr, "Error: MaxPool2d backward gradient shape mismatch\n");
        return NULL;
    }

    // Create gradient input tensor with cached input shape
    const int64_t input_shape[] = { layer->cache_batch, layer->cache_channels,
                              layer->cache_height, layer->cache_width };
    boat_tensor_t* grad_input = boat_tensor_create(input_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!grad_input) {
        return NULL;
    }

    // Initialize gradient input with zeros
    float* grad_input_data = (float*)boat_tensor_data(grad_input);
    size_t input_elements = layer->cache_batch * layer->cache_channels *
                           layer->cache_height * layer->cache_width;
    memset(grad_input_data, 0, input_elements * sizeof(float));

    // Get gradient output data
    const float* grad_output_data = (const float*)boat_tensor_data(grad_output);

    // Propagate gradients: each output gradient goes to the input position that was max
    size_t output_elements = layer->cache_batch * layer->cache_channels *
                            layer->cache_height_out * layer->cache_width_out;

    for (size_t i = 0; i < output_elements; i++) {
        int64_t max_idx = layer->max_indices[i];
        if (max_idx >= 0 && max_idx < (int64_t)input_elements) {
            grad_input_data[max_idx] += grad_output_data[i];
        }
    }

    return grad_input;
}

BOAT_API void BOAT_CALL boat_pool_layer_update(boat_pool_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // Pooling layers have no parameters to update
}