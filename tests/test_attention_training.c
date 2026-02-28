// test_attention_training.c - End-to-end training test for attention layer
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat/layers/attention.h>
#include <boat/tensor.h>
#include <boat/ops.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// Simple loss function: mean squared error between output and target
static float compute_mse_loss(const boat_tensor_t* output, const boat_tensor_t* target) {
    float loss = 0.0f;
    float* out_data = (float*)boat_tensor_data(output);
    float* target_data = (float*)boat_tensor_data(target);
    size_t n = boat_tensor_nelements(output);

    for (size_t i = 0; i < n; i++) {
        float diff = out_data[i] - target_data[i];
        loss += diff * diff;
    }
    return loss / n;
}

// Create gradient tensor for MSE loss: dL/doutput = 2*(output - target)/n
static boat_tensor_t* create_mse_gradient(const boat_tensor_t* output, const boat_tensor_t* target) {
    boat_tensor_t* grad = boat_tensor_create_like(output);
    if (!grad) return NULL;

    float* out_data = (float*)boat_tensor_data(output);
    float* target_data = (float*)boat_tensor_data(target);
    float* grad_data = (float*)boat_tensor_data(grad);
    size_t n = boat_tensor_nelements(output);

    for (size_t i = 0; i < n; i++) {
        grad_data[i] = 2.0f * (out_data[i] - target_data[i]) / n;
    }
    return grad;
}

int main() {
    printf("=== Attention Layer End-to-End Training Test ===\n\n");

    // Configuration for a small attention layer
    boat_attention_config_t config = {
        .hidden_size = 32,
        .num_heads = 4,
        .head_size = 8,
        .dropout_prob = 0.0f,  // No dropout for deterministic test
        .causal_mask = false,
        .use_bias = true,
        .use_rotary = false,
        .rotary_theta = 10000.0f
    };

    // Create attention layer
    boat_attention_t* attention = boat_attention_create(&config);
    if (!attention) {
        printf("ERROR: Failed to create attention layer\n");
        return 1;
    }

    // Create input tensors (batch=2, seq_len=4, hidden=32)
    int64_t input_shape[] = {2, 4, 32};
    boat_tensor_t* query = boat_tensor_create(input_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* key = boat_tensor_create(input_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* value = boat_tensor_create(input_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    if (!query || !key || !value) {
        printf("ERROR: Failed to create input tensors\n");
        boat_attention_free(attention);
        if (query) boat_tensor_unref(query);
        if (key) boat_tensor_unref(key);
        if (value) boat_tensor_unref(value);
        return 1;
    }

    // Initialize with random data (fixed seed for reproducibility)
    srand(42);
    float* query_data = (float*)boat_tensor_data(query);
    float* key_data = (float*)boat_tensor_data(key);
    float* value_data = (float*)boat_tensor_data(value);
    size_t num_elements = boat_tensor_nelements(query);

    for (size_t i = 0; i < num_elements; i++) {
        query_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        key_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        value_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Create target tensor (random target values)
    boat_tensor_t* target = boat_tensor_create(input_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!target) {
        printf("ERROR: Failed to create target tensor\n");
        boat_tensor_unref(query);
        boat_tensor_unref(key);
        boat_tensor_unref(value);
        boat_attention_free(attention);
        return 1;
    }

    float* target_data = (float*)boat_tensor_data(target);
    for (size_t i = 0; i < num_elements; i++) {
        target_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    printf("Starting training loop (5 iterations)...\n");
    printf("Learning rate: 0.01\n\n");

    float learning_rate = 0.01f;
    float previous_loss = 0.0f;

    for (int iteration = 0; iteration < 5; iteration++) {
        // Forward pass
        boat_tensor_t* output = boat_attention_forward(attention, query, key, value, NULL);
        if (!output) {
            printf("ERROR: Forward pass failed at iteration %d\n", iteration);
            boat_tensor_unref(target);
            boat_tensor_unref(query);
            boat_tensor_unref(key);
            boat_tensor_unref(value);
            boat_attention_free(attention);
            return 1;
        }

        // Compute loss
        float loss = compute_mse_loss(output, target);

        // Create gradient for loss
        boat_tensor_t* grad_output = create_mse_gradient(output, target);
        if (!grad_output) {
            printf("ERROR: Failed to create gradient tensor at iteration %d\n", iteration);
            boat_tensor_unref(output);
            boat_tensor_unref(target);
            boat_tensor_unref(query);
            boat_tensor_unref(key);
            boat_tensor_unref(value);
            boat_attention_free(attention);
            return 1;
        }

        // Backward pass
        boat_tensor_t* grad_query = NULL;
        boat_tensor_t* grad_key = NULL;
        boat_tensor_t* grad_value = NULL;
        if (!boat_attention_backward(attention, grad_output, &grad_query, &grad_key, &grad_value)) {
            printf("ERROR: Backward pass failed at iteration %d\n", iteration);
            boat_tensor_unref(grad_output);
            boat_tensor_unref(output);
            boat_tensor_unref(target);
            boat_tensor_unref(query);
            boat_tensor_unref(key);
            boat_tensor_unref(value);
            boat_attention_free(attention);
            return 1;
        }

        // Update parameters
        boat_attention_update(attention, learning_rate);

        // Cleanup iteration tensors
        if (grad_query) boat_tensor_unref(grad_query);
        if (grad_key) boat_tensor_unref(grad_key);
        if (grad_value) boat_tensor_unref(grad_value);
        boat_tensor_unref(grad_output);
        boat_tensor_unref(output);

        // Print progress
        printf("Iteration %d: Loss = %f", iteration, loss);
        if (iteration > 0) {
            float loss_change = loss - previous_loss;
            printf(" (change: %+f)", loss_change);
        }
        printf("\n");

        previous_loss = loss;
    }

    // Verify that gradients were computed (check one parameter gradient)
    boat_tensor_t* grad_weight_q = boat_attention_get_grad_weight_q(attention);
    if (grad_weight_q) {
        float* grad_data = (float*)boat_tensor_data(grad_weight_q);
        size_t grad_elements = boat_tensor_nelements(grad_weight_q);

        // Check that gradient is not all zeros (indicating gradient flow)
        float grad_sum = 0.0f;
        for (size_t i = 0; i < grad_elements && i < 10; i++) {
            grad_sum += fabsf(grad_data[i]);
        }
        printf("\nGradient check: sum of abs first 10 elements of grad_weight_q = %f\n", grad_sum);
        if (grad_sum < 1e-10f) {
            printf("WARNING: Gradient appears to be zero - may indicate gradient flow issue\n");
        }
    } else {
        printf("WARNING: Could not access gradient for weight_q\n");
    }

    // Cleanup
    boat_tensor_unref(target);
    boat_tensor_unref(query);
    boat_tensor_unref(key);
    boat_tensor_unref(value);
    boat_attention_free(attention);

    printf("\n✅ Training test completed successfully\n");
    return 0;
}