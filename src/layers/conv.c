// conv.c - Convolutional layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Debug output control
#ifndef BOAT_DEBUG
#define BOAT_DEBUG 0
#endif

#if BOAT_DEBUG
#define BOAT_DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define BOAT_DEBUG_PRINT(...) ((void)0)
#endif

// Convolutional layer structure
struct boat_conv_layer_t {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    bool use_bias;
    boat_tensor_t* weight;
    boat_tensor_t* bias;

    // Gradient accumulators for training
    boat_tensor_t* grad_weight;
    boat_tensor_t* grad_bias;

    // Cache for backward pass
    boat_tensor_t* cache_input;  // Input tensor from forward pass
    int64_t cache_input_shape[4]; // [batch, in_channels, height, width]
    int64_t cache_output_shape[4]; // [batch, out_channels, height_out, width_out]
};

// Helper function: compute gradient with respect to input
static boat_tensor_t* compute_input_gradient(const boat_conv_layer_t* layer,
                                             const boat_tensor_t* cached_input,
                                             const int64_t* input_shape,
                                             const int64_t* output_shape,
                                             const boat_tensor_t* grad_output) {
    if (!layer || !cached_input || !grad_output) {
        return NULL;
    }

    // Extract dimensions
    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t out_channels = output_shape[1];
    int64_t height_out = output_shape[2];
    int64_t width_out = output_shape[3];

    // Create gradient input tensor with same shape as input
    boat_tensor_t* grad_input = boat_tensor_create(input_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!grad_input) {
        return NULL;
    }

    // Get data pointers
    float* grad_input_data = (float*)boat_tensor_data(grad_input);
    const float* weight_data = (float*)boat_tensor_data(layer->weight);
    const float* grad_output_data = (float*)boat_tensor_data(grad_output);

    // Initialize gradient input with zeros
    size_t grad_input_elements = boat_tensor_nelements(grad_input);
    memset(grad_input_data, 0, grad_input_elements * sizeof(float));

    // For each batch
    for (int64_t b = 0; b < batch; b++) {
        // For each output channel
        for (int64_t oc = 0; oc < (int64_t)layer->out_channels; oc++) {
            // For each input channel
            for (int64_t ic = 0; ic < (int64_t)layer->in_channels; ic++) {
                // For each kernel row
                for (size_t kh = 0; kh < layer->kernel_size; kh++) {
                    // For each kernel column
                    for (size_t kw = 0; kw < layer->kernel_size; kw++) {
                        // For each output height position
                        for (int64_t oh = 0; oh < height_out; oh++) {
                            int64_t ih = oh * layer->stride - layer->padding + kh;
                            if (ih < 0 || ih >= height) continue;
                            for (int64_t ow = 0; ow < width_out; ow++) {
                                int64_t iw = ow * layer->stride - layer->padding + kw;
                                if (iw < 0 || iw >= width) continue;

                                // Compute indices
                                // For input gradient, we need to rotate weights 180 degrees
                                size_t kh_flipped = layer->kernel_size - 1 - kh;
                                size_t kw_flipped = layer->kernel_size - 1 - kw;
                                size_t weight_idx = ((oc * layer->in_channels + ic) * layer->kernel_size + kh_flipped) * layer->kernel_size + kw_flipped;
                                size_t grad_output_idx = ((b * out_channels + oc) * height_out + oh) * width_out + ow;
                                size_t grad_input_idx = ((b * in_channels + ic) * height + ih) * width + iw;

                                grad_input_data[grad_input_idx] += weight_data[weight_idx] * grad_output_data[grad_output_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

// Helper function: compute gradient with respect to weights
static boat_tensor_t* compute_weight_gradient(const boat_conv_layer_t* layer,
                                              const boat_tensor_t* cached_input,
                                              const int64_t* input_shape,
                                              const int64_t* output_shape,
                                              const boat_tensor_t* grad_output) {
    if (!layer || !cached_input || !grad_output) {
        return NULL;
    }

    // Extract dimensions
    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t out_channels = output_shape[1];
    int64_t height_out = output_shape[2];
    int64_t width_out = output_shape[3];

    // Create gradient weight tensor with same shape as weights
    const int64_t weight_shape[] = { (int64_t)layer->out_channels, (int64_t)layer->in_channels,
                                     (int64_t)layer->kernel_size, (int64_t)layer->kernel_size };
    boat_tensor_t* grad_weight = boat_tensor_create(weight_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!grad_weight) {
        return NULL;
    }

    // Get data pointers
    float* grad_weight_data = (float*)boat_tensor_data(grad_weight);
    const float* input_data = (float*)boat_tensor_data(cached_input);
    const float* grad_output_data = (float*)boat_tensor_data(grad_output);

    // Initialize gradient weight with zeros
    size_t grad_weight_elements = boat_tensor_nelements(grad_weight);
    memset(grad_weight_data, 0, grad_weight_elements * sizeof(float));

    // For each batch
    for (int64_t b = 0; b < batch; b++) {
        // For each output channel
        for (int64_t oc = 0; oc < (int64_t)layer->out_channels; oc++) {
            // For each input channel
            for (int64_t ic = 0; ic < (int64_t)layer->in_channels; ic++) {
                // For each kernel row
                for (size_t kh = 0; kh < layer->kernel_size; kh++) {
                    // For each kernel column
                    for (size_t kw = 0; kw < layer->kernel_size; kw++) {
                        // For each output height position
                        for (int64_t oh = 0; oh < height_out; oh++) {
                            int64_t ih = oh * layer->stride - layer->padding + kh;
                            if (ih < 0 || ih >= height) continue;
                            for (int64_t ow = 0; ow < width_out; ow++) {
                                int64_t iw = ow * layer->stride - layer->padding + kw;
                                if (iw < 0 || iw >= width) continue;

                                // Compute indices
                                size_t input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                                size_t grad_output_idx = ((b * out_channels + oc) * height_out + oh) * width_out + ow;
                                size_t weight_idx = ((oc * layer->in_channels + ic) * layer->kernel_size + kh) * layer->kernel_size + kw;

                                grad_weight_data[weight_idx] += input_data[input_idx] * grad_output_data[grad_output_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_weight;
}

// Helper function: compute gradient with respect to bias
static boat_tensor_t* compute_bias_gradient(const boat_conv_layer_t* layer,
                                            const int64_t* output_shape,
                                            const boat_tensor_t* grad_output) {
    if (!layer || !grad_output) {
        return NULL;
    }

    // Only compute bias gradient if bias is used
    if (!layer->use_bias) {
        return NULL;
    }

    // Extract dimensions
    int64_t batch = output_shape[0];
    int64_t out_channels = output_shape[1];
    int64_t height_out = output_shape[2];
    int64_t width_out = output_shape[3];

    // Create gradient bias tensor with same shape as bias
    const int64_t bias_shape[] = { (int64_t)layer->out_channels };
    boat_tensor_t* grad_bias = boat_tensor_create(bias_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!grad_bias) {
        return NULL;
    }

    // Get data pointers
    float* grad_bias_data = (float*)boat_tensor_data(grad_bias);
    const float* grad_output_data = (float*)boat_tensor_data(grad_output);

    // Initialize gradient bias with zeros
    memset(grad_bias_data, 0, layer->out_channels * sizeof(float));

    // Sum grad_output over batch, height, width dimensions
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t oc = 0; oc < (int64_t)layer->out_channels; oc++) {
            float sum = 0.0f;
            for (int64_t oh = 0; oh < height_out; oh++) {
                for (int64_t ow = 0; ow < width_out; ow++) {
                    size_t grad_output_idx = ((b * out_channels + oc) * height_out + oh) * width_out + ow;
                    sum += grad_output_data[grad_output_idx];
                }
            }
            grad_bias_data[oc] += sum;
        }
    }

    return grad_bias;
}

BOAT_API boat_conv_layer_t* BOAT_CALL boat_conv_layer_create(size_t in_channels, size_t out_channels,
                                           size_t kernel_size, size_t stride, size_t padding) {
    BOAT_DEBUG_PRINT("DEBUG conv_create called: in=%zu, out=%zu, k=%zu\n", in_channels, out_channels, kernel_size);
    boat_conv_layer_t* layer = (boat_conv_layer_t*)boat_malloc(sizeof(boat_conv_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        BOAT_DEBUG_PRINT("DEBUG conv_create: malloc failed\n");
        return NULL;
    }

    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->use_bias = true; // Default to using bias

    // Create weight tensor: [out_channels, in_channels, kernel_size, kernel_size]
    const int64_t weight_shape[] = { (int64_t)out_channels, (int64_t)in_channels, (int64_t)kernel_size, (int64_t)kernel_size };
    layer->weight = boat_tensor_create(weight_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!layer->weight) {
        boat_free(layer);
        return NULL;
    }

    // Initialize weights using Kaiming/He initialization for ReLU activations
    float* weight_data = (float*)boat_tensor_data(layer->weight);
    size_t weight_elements = boat_tensor_nelements(layer->weight);
    float scale = sqrtf(2.0f / (in_channels * kernel_size * kernel_size));
    for (size_t i = 0; i < weight_elements; i++) {
        weight_data[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }

    // Create bias tensor: [out_channels]
    const int64_t bias_shape[] = { (int64_t)out_channels };
    layer->bias = boat_tensor_create(bias_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!layer->bias) {
        boat_tensor_free(layer->weight);
        boat_free(layer);
        return NULL;
    }

    // Initialize bias to zeros
    float* bias_data = (float*)boat_tensor_data(layer->bias);
    size_t bias_elements = boat_tensor_nelements(layer->bias);
    memset(bias_data, 0, bias_elements * sizeof(float));

    // Create gradient accumulators with same shape as parameters
    layer->grad_weight = boat_tensor_create(weight_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    BOAT_DEBUG_PRINT("DEBUG conv create: grad_weight tensor created at %p\n", layer->grad_weight);
    if (!layer->grad_weight) {
        boat_tensor_free(layer->weight);
        boat_tensor_free(layer->bias);
        boat_free(layer);
        return NULL;
    }
    // Initialize gradient weight with zeros
    float* grad_weight_data = (float*)boat_tensor_data(layer->grad_weight);
    size_t grad_weight_elements = boat_tensor_nelements(layer->grad_weight);
    memset(grad_weight_data, 0, grad_weight_elements * sizeof(float));

    layer->grad_bias = boat_tensor_create(bias_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    BOAT_DEBUG_PRINT("DEBUG conv create: grad_bias tensor created at %p\n", layer->grad_bias);
    if (!layer->grad_bias) {
        boat_tensor_free(layer->weight);
        boat_tensor_free(layer->bias);
        boat_tensor_free(layer->grad_weight);
        boat_free(layer);
        return NULL;
    }
    // Initialize gradient bias with zeros
    float* grad_bias_data = (float*)boat_tensor_data(layer->grad_bias);
    size_t grad_bias_elements = boat_tensor_nelements(layer->grad_bias);
    memset(grad_bias_data, 0, grad_bias_elements * sizeof(float));

    layer->cache_input = NULL;
    memset(layer->cache_input_shape, 0, sizeof(layer->cache_input_shape));
    memset(layer->cache_output_shape, 0, sizeof(layer->cache_output_shape));

    return layer;
}

BOAT_API void BOAT_CALL boat_conv_layer_free(boat_conv_layer_t* layer) {
    if (!layer) {
        return;
    }

    if (layer->weight) boat_tensor_free(layer->weight);
    if (layer->bias) boat_tensor_free(layer->bias);
    if (layer->grad_weight) boat_tensor_free(layer->grad_weight);
    if (layer->grad_bias) boat_tensor_free(layer->grad_bias);
    if (layer->cache_input) boat_tensor_free(layer->cache_input);
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_forward(boat_conv_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        return NULL;
    }

    // Output shape: [batch, out_channels, height_out, width_out]
    const int64_t* input_shape = boat_tensor_shape(input);
    if (boat_tensor_ndim(input) != 4) {
        fprintf(stderr, "Error: Conv2d expects 4D input tensor\n");
        return NULL;
    }

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    if ((size_t)in_channels != layer->in_channels) {
        fprintf(stderr, "Error: Input channels %lld don't match layer in_channels %zu\n", in_channels, layer->in_channels);
        return NULL;
    }

    // Check data types
    if (boat_tensor_dtype(input) != BOAT_DTYPE_FLOAT32) {
        fprintf(stderr, "Error: Conv2d only supports FLOAT32 input tensors\n");
        return NULL;
    }
    if (boat_tensor_dtype(layer->weight) != BOAT_DTYPE_FLOAT32) {
        fprintf(stderr, "Error: Conv2d weight tensor must be FLOAT32\n");
        return NULL;
    }
    if (layer->use_bias && layer->bias && boat_tensor_dtype(layer->bias) != BOAT_DTYPE_FLOAT32) {
        fprintf(stderr, "Error: Conv2d bias tensor must be FLOAT32\n");
        return NULL;
    }

    // Calculate output dimensions
    int64_t height_out = (height + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;
    int64_t width_out = (width + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;

    // Validate output dimensions
    if (height_out <= 0 || width_out <= 0) {
        fprintf(stderr, "Error: Invalid convolution parameters - output dimensions would be non-positive\n");
        fprintf(stderr, "  height_out = %lld, width_out = %lld\n", height_out, width_out);
        fprintf(stderr, "  height = %lld, width = %lld, padding = %zu, kernel_size = %zu, stride = %zu\n",
                height, width, layer->padding, layer->kernel_size, layer->stride);
        return NULL;
    }

    // Clear old cache
    if (layer->cache_input) {
        boat_tensor_free(layer->cache_input);
        layer->cache_input = NULL;
    }

    // Cache input tensor for backward pass
    layer->cache_input = (boat_tensor_t*)input;
    boat_tensor_ref(layer->cache_input);  // Increase ref count

    // Cache input shape
    layer->cache_input_shape[0] = batch;
    layer->cache_input_shape[1] = in_channels;
    layer->cache_input_shape[2] = height;
    layer->cache_input_shape[3] = width;

    // Cache output shape
    layer->cache_output_shape[0] = batch;
    layer->cache_output_shape[1] = layer->out_channels;
    layer->cache_output_shape[2] = height_out;
    layer->cache_output_shape[3] = width_out;

    // Create output tensor
    const int64_t output_shape[] = { batch, (int64_t)layer->out_channels, height_out, width_out };
    boat_tensor_t* output = boat_tensor_create(output_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!output) {
        return NULL;
    }

    // Get data pointers
    const float* input_data = (float*)boat_tensor_data(input);
    const float* weight_data = (float*)boat_tensor_data(layer->weight);
    const float* bias_data = layer->use_bias ? (float*)boat_tensor_data(layer->bias) : NULL;
    float* output_data = (float*)boat_tensor_data(output);

    // Initialize output with zeros
    size_t output_elements = boat_tensor_nelements(output);
    memset(output_data, 0, output_elements * sizeof(float));

    // Perform convolution
    // For each sample in batch
    for (int64_t b = 0; b < batch; b++) {
        // For each output channel
        for (size_t oc = 0; oc < layer->out_channels; oc++) {
            // For each input channel
            for (size_t ic = 0; ic < layer->in_channels; ic++) {
                // For each output height position
                for (int64_t oh = 0; oh < height_out; oh++) {
                    int64_t ih_start = oh * layer->stride - layer->padding;
                    // For each output width position
                    for (int64_t ow = 0; ow < width_out; ow++) {
                        int64_t iw_start = ow * layer->stride - layer->padding;

                        // Convolve kernel with input patch
                        float sum = 0.0f;
                        for (size_t kh = 0; kh < layer->kernel_size; kh++) {
                            int64_t ih = ih_start + kh;
                            if (ih < 0 || ih >= height) continue;  // Padding handling

                            for (size_t kw = 0; kw < layer->kernel_size; kw++) {
                                int64_t iw = iw_start + kw;
                                if (iw < 0 || iw >= width) continue;  // Padding handling

                                // Compute indices
                                size_t input_idx = ((b * layer->in_channels + ic) * height + ih) * width + iw;
                                size_t weight_idx = ((oc * layer->in_channels + ic) * layer->kernel_size + kh) * layer->kernel_size + kw;

                                sum += input_data[input_idx] * weight_data[weight_idx];
                            }
                        }

                        // Accumulate to output
                        size_t output_idx = ((b * layer->out_channels + oc) * height_out + oh) * width_out + ow;
                        output_data[output_idx] += sum;
                    }
                }
            }

            // Add bias if present
            if (layer->use_bias && bias_data) {
                float bias = bias_data[oc];
                for (int64_t oh = 0; oh < height_out; oh++) {
                    for (int64_t ow = 0; ow < width_out; ow++) {
                        size_t output_idx = ((b * layer->out_channels + oc) * height_out + oh) * width_out + ow;
                        output_data[output_idx] += bias;
                    }
                }
            }
        }
    }

    return output;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_backward(boat_conv_layer_t* layer, const boat_tensor_t* grad_output) {
    if (!layer || !grad_output) {
        fprintf(stderr, "Error: conv backward: NULL input\n");
        return NULL;
    }

    // Check that cached input exists
    if (!layer->cache_input) {
        fprintf(stderr, "Error: conv backward: no cached input (forward not called or cache cleared)\n");
        return NULL;
    }

    // Verify grad_output shape matches cached output shape
    const int64_t* grad_shape = boat_tensor_shape(grad_output);
    if (grad_shape[0] != layer->cache_output_shape[0] ||
        grad_shape[1] != layer->cache_output_shape[1] ||
        grad_shape[2] != layer->cache_output_shape[2] ||
        grad_shape[3] != layer->cache_output_shape[3]) {
        fprintf(stderr, "Error: conv backward: grad_output shape [%lld, %lld, %lld, %lld] doesn't match cached output shape [%lld, %lld, %lld, %lld]\n",
                grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3],
                layer->cache_output_shape[0], layer->cache_output_shape[1],
                layer->cache_output_shape[2], layer->cache_output_shape[3]);
        return NULL;
    }

    // Compute gradients using helper functions
    boat_tensor_t* grad_input = compute_input_gradient(layer, layer->cache_input,
                                                       layer->cache_input_shape,
                                                       layer->cache_output_shape,
                                                       grad_output);
    if (!grad_input) {
        fprintf(stderr, "Error: conv backward: failed to compute input gradient\n");
        return NULL;
    }

    boat_tensor_t* grad_weight = compute_weight_gradient(layer, layer->cache_input,
                                                         layer->cache_input_shape,
                                                         layer->cache_output_shape,
                                                         grad_output);
    if (!grad_weight) {
        fprintf(stderr, "Error: conv backward: failed to compute weight gradient\n");
        boat_tensor_free(grad_input);
        return NULL;
    }

    boat_tensor_t* grad_bias = compute_bias_gradient(layer, layer->cache_output_shape, grad_output);
    // grad_bias may be NULL if bias not used, that's fine

    // Store gradients in layer (free old gradients if they exist)
    if (layer->grad_weight) {
        boat_tensor_free(layer->grad_weight);
    }
    layer->grad_weight = grad_weight;

    if (layer->grad_bias) {
        boat_tensor_free(layer->grad_bias);
    }
    layer->grad_bias = grad_bias;

    // Note: we don't free cache_input here, it will be freed in next forward pass or layer free
    // Return gradient with respect to input
    return grad_input;
}

BOAT_API void BOAT_CALL boat_conv_layer_update(boat_conv_layer_t* layer, float learning_rate) {
    if (!layer) {
        return;
    }

    // Simple SGD update: weight = weight - learning_rate * grad_weight
    if (layer->grad_weight && layer->weight) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(layer->grad_weight, learning_rate);
        if (scaled_grad) {
            boat_sub_(layer->weight, scaled_grad);  // weight -= learning_rate * grad_weight
            boat_tensor_unref(scaled_grad);
        }
    }

    if (layer->use_bias && layer->grad_bias && layer->bias) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(layer->grad_bias, learning_rate);
        if (scaled_grad) {
            boat_sub_(layer->bias, scaled_grad);    // bias -= learning_rate * grad_bias
            boat_tensor_unref(scaled_grad);
        }
    }

    // Note: we don't zero gradients after update; caller can decide to clear gradients
}

// Parameter access functions for model loading
BOAT_API void BOAT_CALL boat_conv_layer_set_weight(boat_conv_layer_t* layer, boat_tensor_t* weight) {
    if (!layer || !weight) {
        return;
    }
    // Check weight shape matches layer dimensions
    const int64_t* weight_shape = boat_tensor_shape(weight);
    if (weight_shape[0] != (int64_t)layer->out_channels ||
        weight_shape[1] != (int64_t)layer->in_channels ||
        weight_shape[2] != (int64_t)layer->kernel_size ||
        weight_shape[3] != (int64_t)layer->kernel_size) {
        fprintf(stderr, "Error: Weight shape [%lld, %lld, %lld, %lld] does not match layer dimensions [%zu, %zu, %zu, %zu]\n",
                weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3],
                layer->out_channels, layer->in_channels, layer->kernel_size, layer->kernel_size);
        return;
    }
    // Replace weight tensor
    if (layer->weight) {
        boat_tensor_free(layer->weight);
    }
    layer->weight = weight;
    boat_tensor_ref(weight); // Increase ref count since layer now owns it
}

BOAT_API void BOAT_CALL boat_conv_layer_set_bias(boat_conv_layer_t* layer, boat_tensor_t* bias) {
    if (!layer || !bias) {
        return;
    }
    if (!layer->use_bias) {
        fprintf(stderr, "Warning: Layer was created without bias, ignoring bias tensor\n");
        return;
    }
    // Check bias shape matches output channels
    const int64_t* bias_shape = boat_tensor_shape(bias);
    if (bias_shape[0] != (int64_t)layer->out_channels) {
        fprintf(stderr, "Error: Bias shape [%lld] does not match output channels %zu\n",
                bias_shape[0], layer->out_channels);
        return;
    }
    // Replace bias tensor
    if (layer->bias) {
        boat_tensor_free(layer->bias);
    }
    layer->bias = bias;
    boat_tensor_ref(bias);
}

BOAT_NOINLINE BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_get_weight(const boat_conv_layer_t* layer) {
    return layer ? layer->weight : NULL;
}

BOAT_NOINLINE BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_get_bias(const boat_conv_layer_t* layer) {
    return layer ? layer->bias : NULL;
}

BOAT_NOINLINE BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_get_grad_weight(const boat_conv_layer_t* layer) {
    BOAT_DEBUG_PRINT("DEBUG get_grad_weight: layer=%p, grad_weight=%p\n", layer, layer ? layer->grad_weight : NULL);
    return layer ? layer->grad_weight : NULL;
}

BOAT_NOINLINE BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_get_grad_bias(const boat_conv_layer_t* layer) {
    return layer ? layer->grad_bias : NULL;
}