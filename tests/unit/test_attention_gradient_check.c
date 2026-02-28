// test_attention_gradient_check.c - Attention layer gradient checking tests
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers/attention.h>
#include <boat/tensor.h>
#include <boat/ops.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Helper function to compute numerical gradient using finite differences
static float compute_numerical_gradient_element(boat_attention_t* attention,
                                                boat_tensor_t* query,
                                                boat_tensor_t* key,
                                                boat_tensor_t* value,
                                                boat_tensor_t* param,
                                                size_t idx,
                                                float epsilon) {
    // Save original value
    float* data = (float*)boat_tensor_data(param);
    float original = data[idx];

    // Compute loss with positive perturbation
    data[idx] = original + epsilon;
    boat_tensor_t* output_plus = boat_attention_forward(attention, query, key, value, NULL);
    float loss_plus = 0.0f;
    if (output_plus) {
        // Simple loss: sum of all elements
        float* out_data = (float*)boat_tensor_data(output_plus);
        size_t n = boat_tensor_nelements(output_plus);
        for (size_t i = 0; i < n; i++) {
            loss_plus += out_data[i];
        }
        boat_tensor_unref(output_plus);
    }

    // Compute loss with negative perturbation
    data[idx] = original - epsilon;
    boat_tensor_t* output_minus = boat_attention_forward(attention, query, key, value, NULL);
    float loss_minus = 0.0f;
    if (output_minus) {
        float* out_data = (float*)boat_tensor_data(output_minus);
        size_t n = boat_tensor_nelements(output_minus);
        for (size_t i = 0; i < n; i++) {
            loss_minus += out_data[i];
        }
        boat_tensor_unref(output_minus);
    }

    // Restore original value
    data[idx] = original;

    // Compute numerical gradient: (loss_plus - loss_minus) / (2 * epsilon)
    return (loss_plus - loss_minus) / (2.0f * epsilon);
}

// Check gradient agreement with relative and absolute tolerances
static bool check_gradient_agreement(float analytical, float numerical,
                                     float rel_tol, float abs_tol) {
    float diff = fabsf(analytical - numerical);
    if (diff <= abs_tol) {
        return true;
    }
    float sum = fabsf(analytical) + fabsf(numerical);
    if (sum > 0.0f) {
        float rel_err = diff / sum;
        if (rel_err <= rel_tol) {
            return true;
        }
    }
    return false;
}

// Test gradient for a specific parameter tensor
static bool test_parameter_gradient(boat_attention_t* attention,
                                    boat_tensor_t* query,
                                    boat_tensor_t* key,
                                    boat_tensor_t* value,
                                    boat_tensor_t* param,
                                    boat_tensor_t* grad,
                                    const char* param_name,
                                    size_t max_tests) {
    if (!param || !grad) {
        printf("    %s: no parameter or gradient (skipping)\n", param_name);
        return true;
    }

    printf("    Testing %s... ", param_name);

    const float epsilon = 1e-4f;
    const float rel_tol = 2e-2f;  // 2% tolerance for gradient checking
    const float abs_tol = 1e-2f;   // Increased absolute tolerance for attention gradients

    size_t num_elements = boat_tensor_nelements(param);
    float* grad_data = (float*)boat_tensor_data(grad);

    int failures = 0;
    int tests_done = 0;

    // Test a subset of elements
    size_t step = num_elements / max_tests;
    if (step < 1) step = 1;

    for (size_t i = 0; i < num_elements && tests_done < max_tests; i += step) {
        float numerical = compute_numerical_gradient_element(attention, query, key, value, param, i, epsilon);
        float analytical = grad_data[i];

        if (!check_gradient_agreement(analytical, numerical, rel_tol, abs_tol)) {
            if (failures < 3) {
                printf("\n      mismatch at %zu: analytical=%g, numerical=%g, diff=%g",
                       i, analytical, numerical, analytical - numerical);
            }
            failures++;
        }
        tests_done++;
    }

    if (failures > 0) {
        printf("\n      FAILED: %d/%d elements\n", failures, tests_done);
        return false;
    } else {
        printf("PASSED (%d/%d elements)\n", tests_done, num_elements);
        return true;
    }
}

int main() {
    printf("=== Attention Layer Gradient Checking Tests ===\n\n");

    // Create a small attention layer for testing
    boat_attention_config_t config = {
        .hidden_size = 32,
        .num_heads = 4,
        .head_size = 8,
        .dropout_prob = 0.0f,
        .causal_mask = false,
        .use_bias = true,
        .use_rotary = false,
        .rotary_theta = 10000.0f
    };

    boat_attention_t* attention = boat_attention_create(&config);
    if (!attention) {
        printf("ERROR: Failed to create attention layer\n");
        return 1;
    }

    // Create small input tensors
    int64_t input_shape[] = {2, 4, 32}; // batch=2, seq_len=4, hidden=32
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

    // Forward pass
    boat_tensor_t* output = boat_attention_forward(attention, query, key, value, NULL);
    if (!output) {
        printf("ERROR: Forward pass failed\n");
        boat_tensor_unref(query);
        boat_tensor_unref(key);
        boat_tensor_unref(value);
        boat_attention_free(attention);
        return 1;
    }

    // Create gradient output (loss gradient = 1 for all elements)
    int64_t output_shape[3];
    size_t output_ndim = boat_tensor_ndim(output);
    const int64_t* output_shape_ptr = boat_tensor_shape(output);
    for (size_t i = 0; i < output_ndim; i++) {
        output_shape[i] = output_shape_ptr[i];
    }

    boat_tensor_t* grad_output = boat_tensor_create(output_shape, output_ndim, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!grad_output) {
        printf("ERROR: Failed to create gradient output tensor\n");
        boat_tensor_unref(output);
        boat_tensor_unref(query);
        boat_tensor_unref(key);
        boat_tensor_unref(value);
        boat_attention_free(attention);
        return 1;
    }

    // Set gradient output to 1.0
    float* grad_output_data = (float*)boat_tensor_data(grad_output);
    size_t grad_num_elements = boat_tensor_nelements(grad_output);
    for (size_t i = 0; i < grad_num_elements; i++) {
        grad_output_data[i] = 1.0f;
    }

    // Backward pass (compute analytical gradients)
    boat_tensor_t* grad_input_q = NULL;
    boat_tensor_t* grad_input_k = NULL;
    boat_tensor_t* grad_input_v = NULL;
    if (!boat_attention_backward(attention, grad_output, &grad_input_q, &grad_input_k, &grad_input_v)) {
        printf("ERROR: Backward pass failed\n");
        boat_tensor_unref(grad_output);
        boat_tensor_unref(output);
        boat_tensor_unref(query);
        boat_tensor_unref(key);
        boat_tensor_unref(value);
        boat_attention_free(attention);
        return 1;
    }
    // For compatibility with existing code, use grad_input_q as grad_input (query gradient)
    boat_tensor_t* grad_input = grad_input_q;

    // Get parameter and gradient tensors using accessor functions
    boat_tensor_t* weight_q = boat_attention_get_weight_q(attention);
    boat_tensor_t* weight_k = boat_attention_get_weight_k(attention);
    boat_tensor_t* weight_v = boat_attention_get_weight_v(attention);
    boat_tensor_t* weight_o = boat_attention_get_weight_o(attention);
    boat_tensor_t* bias_q = boat_attention_get_bias_q(attention);
    boat_tensor_t* bias_k = boat_attention_get_bias_k(attention);
    boat_tensor_t* bias_v = boat_attention_get_bias_v(attention);
    boat_tensor_t* bias_o = boat_attention_get_bias_o(attention);

    boat_tensor_t* grad_weight_q = boat_attention_get_grad_weight_q(attention);
    boat_tensor_t* grad_weight_k = boat_attention_get_grad_weight_k(attention);
    boat_tensor_t* grad_weight_v = boat_attention_get_grad_weight_v(attention);
    boat_tensor_t* grad_weight_o = boat_attention_get_grad_weight_o(attention);
    boat_tensor_t* grad_bias_q = boat_attention_get_grad_bias_q(attention);
    boat_tensor_t* grad_bias_k = boat_attention_get_grad_bias_k(attention);
    boat_tensor_t* grad_bias_v = boat_attention_get_grad_bias_v(attention);
    boat_tensor_t* grad_bias_o = boat_attention_get_grad_bias_o(attention);

    printf("Testing gradient correctness with finite differences:\n");
    printf("  (Testing up to 10 elements per parameter)\n\n");

    bool all_pass = true;
    const size_t max_tests_per_param = 10;

    // Test weight gradients
    all_pass = test_parameter_gradient(attention, query, key, value,
                                       weight_q, grad_weight_q, "W_q", max_tests_per_param) && all_pass;
    all_pass = test_parameter_gradient(attention, query, key, value,
                                       weight_k, grad_weight_k, "W_k", max_tests_per_param) && all_pass;
    all_pass = test_parameter_gradient(attention, query, key, value,
                                       weight_v, grad_weight_v, "W_v", max_tests_per_param) && all_pass;
    all_pass = test_parameter_gradient(attention, query, key, value,
                                       weight_o, grad_weight_o, "W_o", max_tests_per_param) && all_pass;

    // Test bias gradients
    if (config.use_bias) {
        all_pass = test_parameter_gradient(attention, query, key, value,
                                           bias_q, grad_bias_q, "b_q", max_tests_per_param) && all_pass;
        all_pass = test_parameter_gradient(attention, query, key, value,
                                           bias_k, grad_bias_k, "b_k", max_tests_per_param) && all_pass;
        all_pass = test_parameter_gradient(attention, query, key, value,
                                           bias_v, grad_bias_v, "b_v", max_tests_per_param) && all_pass;
        all_pass = test_parameter_gradient(attention, query, key, value,
                                           bias_o, grad_bias_o, "b_o", max_tests_per_param) && all_pass;
    }

    // Cleanup
    boat_tensor_unref(grad_input);
    boat_tensor_unref(grad_output);
    boat_tensor_unref(output);
    boat_tensor_unref(query);
    boat_tensor_unref(key);
    boat_tensor_unref(value);
    boat_attention_free(attention);

    printf("\n");
    if (all_pass) {
        printf("✅ All gradient checks PASSED\n");
        return 0;
    } else {
        printf("❌ Some gradient checks FAILED\n");
        return 1;
    }
}