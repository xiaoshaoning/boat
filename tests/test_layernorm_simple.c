// test_layernorm_simple.c - Simple test for layer normalization integration
#include <boat/layers/norm.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    printf("Testing layer normalization integration...\n");

    // Create layer normalization config
    boat_layernorm_config_t ln_config = {
        .normalized_shape = 768,
        .eps = 1e-5f,
        .elementwise_affine = true,
        .use_bias = true
    };

    // Create layer
    boat_layernorm_t* ln = boat_layernorm_create(&ln_config);
    if (!ln) {
        fprintf(stderr, "Failed to create layer normalization layer\n");
        return 1;
    }
    printf("Created layer normalization layer\n");

    // Create weight tensor
    int64_t weight_shape[] = {768};
    boat_tensor_t* weight = boat_tensor_create(weight_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!weight) {
        fprintf(stderr, "Failed to create weight tensor\n");
        boat_layernorm_free(ln);
        return 1;
    }

    // Initialize weight with ones
    float* weight_data = (float*)boat_tensor_data(weight);
    for (int i = 0; i < 768; i++) {
        weight_data[i] = 1.0f;
    }

    // Set weight
    boat_layernorm_set_weight(ln, weight);
    boat_tensor_unref(weight); // layer now owns weight
    printf("Set weight tensor\n");

    // Create bias tensor
    int64_t bias_shape[] = {768};
    boat_tensor_t* bias = boat_tensor_create(bias_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!bias) {
        fprintf(stderr, "Failed to create bias tensor\n");
        boat_layernorm_free(ln);
        return 1;
    }

    // Initialize bias with zeros
    float* bias_data = (float*)boat_tensor_data(bias);
    for (int i = 0; i < 768; i++) {
        bias_data[i] = 0.0f;
    }

    // Set bias
    boat_layernorm_set_bias(ln, bias);
    boat_tensor_unref(bias); // layer now owns bias
    printf("Set bias tensor\n");

    // Create input tensor: batch=2, sequence=16, hidden=768
    int64_t input_shape[] = {2, 16, 768};
    boat_tensor_t* input = boat_tensor_create(input_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!input) {
        fprintf(stderr, "Failed to create input tensor\n");
        boat_layernorm_free(ln);
        return 1;
    }

    // Initialize input with simple pattern
    float* input_data = (float*)boat_tensor_data(input);
    size_t total_elements = 2 * 16 * 768;
    for (size_t i = 0; i < total_elements; i++) {
        input_data[i] = (float)(i % 10) * 0.1f; // Simple pattern
    }

    // Forward pass
    boat_tensor_t* output = boat_layernorm_forward(ln, input);
    if (!output) {
        fprintf(stderr, "Forward pass failed\n");
        boat_tensor_unref(input);
        boat_layernorm_free(ln);
        return 1;
    }

    printf("Forward pass succeeded\n");
    printf("Output shape: [%lld, %lld, %lld]\n",
           boat_tensor_shape(output)[0],
           boat_tensor_shape(output)[1],
           boat_tensor_shape(output)[2]);

    // Check that output is not all zeros
    float* output_data = (float*)boat_tensor_data(output);
    float sum = 0.0f;
    for (size_t i = 0; i < total_elements; i++) {
        sum += output_data[i];
    }
    printf("Sum of output elements: %f\n", sum);

    if (fabsf(sum) < 1e-6f) {
        printf("WARNING: Output sum is near zero, might indicate problem\n");
    }

    // Cleanup
    boat_tensor_unref(output);
    boat_tensor_unref(input);
    boat_layernorm_free(ln);

    printf("Test completed\n");
    return 0;
}