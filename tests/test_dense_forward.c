// test_dense_forward.c - Test dense layer forward pass with bias
#include <boat/layers.h>
#include <boat/tensor.h>
#include <boat/ops.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    printf("Testing dense layer forward pass with bias...\n");

    // Create a dense layer: input_features=3, output_features=2, with bias
    boat_dense_layer_t* dense = boat_dense_layer_create(3, 2, true);
    if (!dense) {
        fprintf(stderr, "Failed to create dense layer\n");
        return 1;
    }
    printf("Created dense layer: input_features=3, output_features=2, bias=true\n");

    // Set custom weight (3x2 matrix)
    int64_t weight_shape[] = {3, 2};
    boat_tensor_t* weight = boat_tensor_create(weight_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!weight) {
        fprintf(stderr, "Failed to create weight tensor\n");
        boat_dense_layer_free(dense);
        return 1;
    }
    float* weight_data = (float*)boat_tensor_data(weight);
    // Simple weights: identity-like (but 3x2)
    weight_data[0] = 1.0f; weight_data[1] = 0.0f;  // row 0
    weight_data[2] = 0.0f; weight_data[3] = 1.0f;  // row 1
    weight_data[4] = 0.5f; weight_data[5] = 0.5f;  // row 2
    boat_dense_layer_set_weight(dense, weight);
    boat_tensor_unref(weight); // layer now owns weight

    // Set custom bias (2-element vector)
    int64_t bias_shape[] = {2};
    boat_tensor_t* bias = boat_tensor_create(bias_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!bias) {
        fprintf(stderr, "Failed to create bias tensor\n");
        boat_dense_layer_free(dense);
        return 1;
    }
    float* bias_data = (float*)boat_tensor_data(bias);
    bias_data[0] = 0.1f;
    bias_data[1] = 0.2f;
    boat_dense_layer_set_bias(dense, bias);
    boat_tensor_unref(bias); // layer now owns bias

    // Create input tensor: batch=2, input_features=3
    int64_t input_shape[] = {2, 3};
    boat_tensor_t* input = boat_tensor_create(input_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!input) {
        fprintf(stderr, "Failed to create input tensor\n");
        boat_dense_layer_free(dense);
        return 1;
    }
    float* input_data = (float*)boat_tensor_data(input);
    // Simple input: two samples
    input_data[0] = 1.0f; input_data[1] = 0.0f; input_data[2] = 0.0f; // sample 1
    input_data[3] = 0.0f; input_data[4] = 1.0f; input_data[5] = 0.0f; // sample 2

    // Forward pass
    boat_tensor_t* output = boat_dense_layer_forward(dense, input);
    if (!output) {
        fprintf(stderr, "Forward pass failed\n");
        boat_tensor_unref(input);
        boat_dense_layer_free(dense);
        return 1;
    }

    printf("Forward pass succeeded\n");
    printf("Output shape: [%lld", boat_tensor_shape(output)[0]);
    for (size_t i = 1; i < boat_tensor_ndim(output); i++) {
        printf(", %lld", boat_tensor_shape(output)[i]);
    }
    printf("]\n");

    // Expected output calculation:
    // Sample 1: [1,0,0] @ [[1,0],[0,1],[0.5,0.5]] + [0.1,0.2] = [1*1+0*0+0*0.5, 1*0+0*1+0*0.5] + [0.1,0.2] = [1.0, 0.0] + [0.1,0.2] = [1.1, 0.2]
    // Sample 2: [0,1,0] @ ... = [0*1+1*0+0*0.5, 0*0+1*1+0*0.5] + [0.1,0.2] = [0.0, 1.0] + [0.1,0.2] = [0.1, 1.2]
    float* output_data = (float*)boat_tensor_data(output);
    float expected[4] = {1.1f, 0.2f, 0.1f, 1.2f};
    float tolerance = 1e-5f;
    int errors = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(output_data[i] - expected[i]) > tolerance) {
            fprintf(stderr, "Output[%d] = %f, expected %f\n", i, output_data[i], expected[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("Output matches expected values!\n");
    } else {
        fprintf(stderr, "%d errors in output\n", errors);
    }

    // Cleanup
    boat_tensor_unref(output);
    boat_tensor_unref(input);
    boat_dense_layer_free(dense);

    printf("Test %s\n", errors == 0 ? "PASSED" : "FAILED");
    return errors == 0 ? 0 : 1;
}