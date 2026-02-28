// test_conv_forward.c - Test convolutional layer forward pass
#include <boat/layers.h>
#include <boat/tensor.h>
#include <boat/ops.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    printf("Testing convolutional layer forward pass...\n");

    // Create a convolutional layer: in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0
    boat_conv_layer_t* conv = boat_conv_layer_create(1, 1, 2, 1, 0);
    if (!conv) {
        fprintf(stderr, "Failed to create convolutional layer\n");
        return 1;
    }
    printf("Created conv layer: in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0\n");

    // Set custom weight (1x1x2x2 tensor)
    int64_t weight_shape[] = {1, 1, 2, 2};  // [out_channels, in_channels, kernel_h, kernel_w]
    boat_tensor_t* weight = boat_tensor_create(weight_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!weight) {
        fprintf(stderr, "Failed to create weight tensor\n");
        boat_conv_layer_free(conv);
        return 1;
    }
    float* weight_data = (float*)boat_tensor_data(weight);
    // Simple 2x2 kernel
    weight_data[0] = 1.0f; weight_data[1] = 0.0f;  // row 0
    weight_data[2] = 0.0f; weight_data[3] = 1.0f;  // row 1
    boat_conv_layer_set_weight(conv, weight);
    boat_tensor_unref(weight); // layer now owns weight

    // Disable bias for this test
    // Note: we need a way to disable bias, but layer is created with use_bias=true by default
    // For now, we'll just set bias to zeros

    // Create input tensor: batch=1, channels=1, height=3, width=3
    int64_t input_shape[] = {1, 1, 3, 3};
    boat_tensor_t* input = boat_tensor_create(input_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!input) {
        fprintf(stderr, "Failed to create input tensor\n");
        boat_conv_layer_free(conv);
        return 1;
    }
    float* input_data = (float*)boat_tensor_data(input);
    // Simple 3x3 image
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f;
    input_data[3] = 4.0f; input_data[4] = 5.0f; input_data[5] = 6.0f;
    input_data[6] = 7.0f; input_data[7] = 8.0f; input_data[8] = 9.0f;

    // Forward pass
    boat_tensor_t* output = boat_conv_layer_forward(conv, input);
    if (!output) {
        fprintf(stderr, "Forward pass failed\n");
        boat_tensor_unref(input);
        boat_conv_layer_free(conv);
        return 1;
    }

    printf("Forward pass succeeded\n");
    printf("Output shape: [%lld", boat_tensor_shape(output)[0]);
    for (size_t i = 1; i < boat_tensor_ndim(output); i++) {
        printf(", %lld", boat_tensor_shape(output)[i]);
    }
    printf("]\n");

    // Calculate expected output manually
    // Input: 3x3 matrix, Kernel: 2x2 matrix
    // [[1,2,3],    Kernel: [[1,0],
    //  [4,5,6],             [0,1]]
    //  [7,8,9]]
    // Output should be 2x2 (since padding=0, stride=1):
    // Position (0,0): 1*1 + 2*0 + 4*0 + 5*1 = 1 + 5 = 6
    // Position (0,1): 2*1 + 3*0 + 5*0 + 6*1 = 2 + 6 = 8
    // Position (1,0): 4*1 + 5*0 + 7*0 + 8*1 = 4 + 8 = 12
    // Position (1,1): 5*1 + 6*0 + 8*0 + 9*1 = 5 + 9 = 14
    float* output_data = (float*)boat_tensor_data(output);
    float expected[4] = {6.0f, 8.0f, 12.0f, 14.0f};
    float tolerance = 1e-5f;
    int errors = 0;

    printf("Output values:\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%d] = %f (expected %f)", i, output_data[i], expected[i]);
        if (fabsf(output_data[i] - expected[i]) > tolerance) {
            printf("  ERROR\n");
            errors++;
        } else {
            printf("  OK\n");
        }
    }

    // Test 2: With padding=1
    printf("\nTesting conv layer with padding=1...\n");
    boat_conv_layer_t* conv_pad = boat_conv_layer_create(1, 1, 2, 1, 1);
    if (!conv_pad) {
        fprintf(stderr, "Failed to create convolutional layer with padding\n");
        boat_tensor_unref(output);
        boat_tensor_unref(input);
        boat_conv_layer_free(conv);
        return 1;
    }

    // Use same weight
    boat_tensor_t* weight2 = boat_tensor_create(weight_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* weight2_data = (float*)boat_tensor_data(weight2);
    weight2_data[0] = 1.0f; weight2_data[1] = 0.0f;
    weight2_data[2] = 0.0f; weight2_data[3] = 1.0f;
    boat_conv_layer_set_weight(conv_pad, weight2);
    boat_tensor_unref(weight2);

    boat_tensor_t* output_pad = boat_conv_layer_forward(conv_pad, input);
    if (!output_pad) {
        fprintf(stderr, "Forward pass with padding failed\n");
        boat_conv_layer_free(conv_pad);
    } else {
        const int64_t* out_shape = boat_tensor_shape(output_pad);
        printf("Output shape with padding=1: [%lld, %lld, %lld, %lld]\n",
               out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
        // With padding=1, input 3x3 becomes effectively 5x5, output should be 4x4
        if (out_shape[2] == 4 && out_shape[3] == 4) {
            printf("Padding test: Output dimensions correct!\n");
        } else {
            fprintf(stderr, "Padding test: Wrong output dimensions\n");
            errors++;
        }
        boat_tensor_unref(output_pad);
    }
    boat_conv_layer_free(conv_pad);

    // Test 3: With stride=2
    printf("\nTesting conv layer with stride=2...\n");
    boat_conv_layer_t* conv_stride = boat_conv_layer_create(1, 1, 2, 2, 0);
    if (!conv_stride) {
        fprintf(stderr, "Failed to create convolutional layer with stride\n");
    } else {
        boat_tensor_t* weight3 = boat_tensor_create(weight_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        float* weight3_data = (float*)boat_tensor_data(weight3);
        weight3_data[0] = 1.0f; weight3_data[1] = 0.0f;
        weight3_data[2] = 0.0f; weight3_data[3] = 1.0f;
        boat_conv_layer_set_weight(conv_stride, weight3);
        boat_tensor_unref(weight3);

        boat_tensor_t* output_stride = boat_conv_layer_forward(conv_stride, input);
        if (!output_stride) {
            fprintf(stderr, "Forward pass with stride failed\n");
        } else {
            const int64_t* out_shape = boat_tensor_shape(output_stride);
            printf("Output shape with stride=2: [%lld, %lld, %lld, %lld]\n",
                   out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
            // With stride=2, input 3x3, output should be 1x1 (ceil((3-2+1)/2) = 1)
            if (out_shape[2] == 1 && out_shape[3] == 1) {
                printf("Stride test: Output dimensions correct!\n");
            } else {
                fprintf(stderr, "Stride test: Wrong output dimensions\n");
                errors++;
            }
            boat_tensor_unref(output_stride);
        }
        boat_conv_layer_free(conv_stride);
    }

    // Cleanup
    boat_tensor_unref(output);
    boat_tensor_unref(input);
    boat_conv_layer_free(conv);

    printf("\nTest %s\n", errors == 0 ? "PASSED" : "FAILED");
    return errors == 0 ? 0 : 1;
}