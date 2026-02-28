// test_gradient_check.c - General gradient checking tests for automatic differentiation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/autodiff.h>
#include <boat/tensor.h>
#include <boat/ops.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Helper function to compute numerical gradient using finite differences
static float compute_numerical_gradient_element(
    boat_tensor_t* param,
    size_t idx,
    float epsilon,
    boat_variable_t* (*forward_func)(boat_variable_t*, boat_variable_t*),
    boat_variable_t* a,
    boat_variable_t* b
) {
    // Save original value
    float* data = (float*)boat_tensor_data(param);
    float original = data[idx];

    // Compute loss with positive perturbation
    data[idx] = original + epsilon;
    boat_variable_t* output_plus = forward_func(a, b);
    float loss_plus = 0.0f;
    if (output_plus) {
        boat_tensor_t* output_tensor = boat_variable_data(output_plus);
        float* out_data = (float*)boat_tensor_data(output_tensor);
        size_t n = boat_tensor_nelements(output_tensor);
        for (size_t i = 0; i < n; i++) {
            loss_plus += out_data[i];
        }
        boat_variable_free(output_plus);
    }

    // Compute loss with negative perturbation
    data[idx] = original - epsilon;
    boat_variable_t* output_minus = forward_func(a, b);
    float loss_minus = 0.0f;
    if (output_minus) {
        boat_tensor_t* output_tensor = boat_variable_data(output_minus);
        float* out_data = (float*)boat_tensor_data(output_tensor);
        size_t n = boat_tensor_nelements(output_tensor);
        for (size_t i = 0; i < n; i++) {
            loss_minus += out_data[i];
        }
        boat_variable_free(output_minus);
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
        return true;  // Difference less than absolute tolerance, gradients match
    }
    float sum = fabsf(analytical) + fabsf(numerical);
    if (sum > 0.0f) {
        float rel_err = diff / sum;
        if (rel_err <= rel_tol) {
            return true;  // Relative error less than tolerance, gradients match
        }
    }
    return false;  // Gradients do not match
}

// Test gradient for a simple addition operation
static bool test_addition_gradient() {
    printf("Testing addition operation gradient...\n");

    // Create two input variables
    int64_t shape[] = {2, 3};
    boat_variable_t* a = boat_variable_create_with_shape(shape, 2, BOAT_DTYPE_FLOAT32, true);
    boat_variable_t* b = boat_variable_create_with_shape(shape, 2, BOAT_DTYPE_FLOAT32, true);

    if (!a || !b) {
        printf("  Failed to create variables\n");
        return false;
    }

    // Initialize with random data
    float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
    float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
    size_t n_elements = 2 * 3;
    srand(42);
    for (size_t i = 0; i < n_elements; i++) {
        a_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        b_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Forward pass: c = a + b
    boat_variable_t* c = boat_var_add(a, b);
    if (!c) {
        printf("  Forward pass failed\n");
        boat_variable_free(a);
        boat_variable_free(b);
        return false;
    }

    // Compute loss as sum of all elements (so gradient w.r.t c is 1)
    boat_tensor_t* c_tensor = boat_variable_data(c);
    float* c_data = (float*)boat_tensor_data(c_tensor);
    float loss = 0.0f;
    for (size_t i = 0; i < n_elements; i++) {
        loss += c_data[i];
    }

    // Backward pass
    boat_variable_zero_grad(a);
    boat_variable_zero_grad(b);

    // Create gradient of loss w.r.t c (all ones)
    boat_tensor_t* grad_c = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!grad_c) {
        printf("  Failed to create gradient tensor\n");
        boat_variable_free(c);
        boat_variable_free(a);
        boat_variable_free(b);
        return false;
    }
    float* grad_c_data = (float*)boat_tensor_data(grad_c);
    for (size_t i = 0; i < n_elements; i++) {
        grad_c_data[i] = 1.0f;
    }

    printf("  Calling boat_variable_backward: c=%p, requires_grad=%d\n",
           c, c ? boat_variable_requires_grad(c) : 0);
    boat_variable_backward(c, grad_c);
    printf("  boat_variable_backward returned\n");

    // Get analytical gradients
    boat_tensor_t* grad_a_tensor = boat_variable_grad(a);
    boat_tensor_t* grad_b_tensor = boat_variable_grad(b);
    float* grad_a = grad_a_tensor ? (float*)boat_tensor_data(grad_a_tensor) : NULL;
    float* grad_b = grad_b_tensor ? (float*)boat_tensor_data(grad_b_tensor) : NULL;

    if (!grad_a || !grad_b) {
        printf("  Failed to get gradient tensors\n");
        boat_tensor_unref(grad_c);
        boat_variable_free(c);
        boat_variable_free(a);
        boat_variable_free(b);
        return false;
    }

    // Numerical gradient parameters
    const float epsilon = 1e-4f;
    const float rel_tol = 1e-3f;
    const float abs_tol = 1e-5f;

    int failures = 0;

    // Check gradients for each element
    for (size_t i = 0; i < n_elements; i++) {
        // For addition, gradient w.r.t a and b should be 1
        float analytical_a = grad_a[i];
        float analytical_b = grad_b[i];

        // Numerical gradient for a
        float numerical_a = compute_numerical_gradient_element(
            boat_variable_data(a), i, epsilon, boat_var_add, a, b);
        // Numerical gradient for b (need to perturb b)
        float numerical_b = compute_numerical_gradient_element(
            boat_variable_data(b), i, epsilon, boat_var_add, a, b);

        if (!check_gradient_agreement(analytical_a, numerical_a, rel_tol, abs_tol)) {
            printf("  Mismatch for a[%zu]: analytical=%g, numerical=%g\n", i, analytical_a, numerical_a);
            failures++;
        }
        if (!check_gradient_agreement(analytical_b, numerical_b, rel_tol, abs_tol)) {
            printf("  Mismatch for b[%zu]: analytical=%g, numerical=%g\n", i, analytical_b, numerical_b);
            failures++;
        }
    }

    // Cleanup
    boat_tensor_unref(grad_c);
    boat_variable_free(c);
    boat_variable_free(a);
    boat_variable_free(b);

    if (failures > 0) {
        printf("  FAILED: %d mismatches\n", failures);
        return false;
    } else {
        printf("  PASSED\n");
        return true;
    }
}

// Test gradient for multiplication operation
static bool test_multiplication_gradient() {
    printf("Testing multiplication operation gradient...\n");

    // Create two input variables
    int64_t shape[] = {2, 2};
    boat_variable_t* a = boat_variable_create_with_shape(shape, 2, BOAT_DTYPE_FLOAT32, true);
    boat_variable_t* b = boat_variable_create_with_shape(shape, 2, BOAT_DTYPE_FLOAT32, true);

    if (!a || !b) {
        printf("  Failed to create variables\n");
        return false;
    }

    // Initialize with random data
    float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
    float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
    size_t n_elements = 2 * 2;
    srand(43);
    for (size_t i = 0; i < n_elements; i++) {
        a_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        b_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Forward pass: c = a * b (element-wise)
    boat_variable_t* c = boat_var_mul(a, b);
    if (!c) {
        printf("  Forward pass failed\n");
        boat_variable_free(a);
        boat_variable_free(b);
        return false;
    }

    // Compute loss as sum of all elements (so gradient w.r.t c is 1)
    boat_tensor_t* c_tensor = boat_variable_data(c);
    float* c_data = (float*)boat_tensor_data(c_tensor);
    float loss = 0.0f;
    for (size_t i = 0; i < n_elements; i++) {
        loss += c_data[i];
    }

    // Backward pass
    boat_variable_zero_grad(a);
    boat_variable_zero_grad(b);

    // Create gradient of loss w.r.t c (all ones)
    boat_tensor_t* grad_c = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!grad_c) {
        printf("  Failed to create gradient tensor\n");
        boat_variable_free(c);
        boat_variable_free(a);
        boat_variable_free(b);
        return false;
    }
    float* grad_c_data = (float*)boat_tensor_data(grad_c);
    for (size_t i = 0; i < n_elements; i++) {
        grad_c_data[i] = 1.0f;
    }

    printf("  Calling boat_variable_backward: c=%p, requires_grad=%d\n",
           c, c ? boat_variable_requires_grad(c) : 0);
    boat_variable_backward(c, grad_c);
    printf("  boat_variable_backward returned\n");

    // Get analytical gradients
    boat_tensor_t* grad_a_tensor = boat_variable_grad(a);
    boat_tensor_t* grad_b_tensor = boat_variable_grad(b);
    float* grad_a = grad_a_tensor ? (float*)boat_tensor_data(grad_a_tensor) : NULL;
    float* grad_b = grad_b_tensor ? (float*)boat_tensor_data(grad_b_tensor) : NULL;

    if (!grad_a || !grad_b) {
        printf("  Failed to get gradient tensors\n");
        boat_tensor_unref(grad_c);
        boat_variable_free(c);
        boat_variable_free(a);
        boat_variable_free(b);
        return false;
    }

    // Numerical gradient parameters
    const float epsilon = 1e-4f;
    const float rel_tol = 1e-3f;
    const float abs_tol = 1e-5f;

    int failures = 0;

    // Check gradients for each element
    for (size_t i = 0; i < n_elements; i++) {
        // For multiplication, gradient w.r.t a is b, w.r.t b is a
        float analytical_a = grad_a[i];
        float analytical_b = grad_b[i];

        // Numerical gradient for a
        float numerical_a = compute_numerical_gradient_element(
            boat_variable_data(a), i, epsilon, boat_var_mul, a, b);
        // Numerical gradient for b
        float numerical_b = compute_numerical_gradient_element(
            boat_variable_data(b), i, epsilon, boat_var_mul, a, b);

        if (!check_gradient_agreement(analytical_a, numerical_a, rel_tol, abs_tol)) {
            printf("  Mismatch for a[%zu]: analytical=%g, numerical=%g\n", i, analytical_a, numerical_a);
            failures++;
        }
        if (!check_gradient_agreement(analytical_b, numerical_b, rel_tol, abs_tol)) {
            printf("  Mismatch for b[%zu]: analytical=%g, numerical=%g\n", i, analytical_b, numerical_b);
            failures++;
        }
    }

    // Cleanup
    boat_tensor_unref(grad_c);
    boat_variable_free(c);
    boat_variable_free(a);
    boat_variable_free(b);

    if (failures > 0) {
        printf("  FAILED: %d mismatches\n", failures);
        return false;
    } else {
        printf("  PASSED\n");
        return true;
    }
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0); // Disable output buffering
    fprintf(stderr, "Test starting\n");

    // Explicitly create and set autodiff context to ensure shared graph
    // This works around Windows DLL static variable issues
    boat_autodiff_context_t* ctx = boat_autodiff_context_create();
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to create autodiff context\n");
        return 1;
    }
    boat_autodiff_set_current_context(ctx);
    fprintf(stderr, "[TEST] Created autodiff context=%p\n", (void*)ctx);

    // Create a graph and associate with context
    boat_graph_t* graph = boat_graph_create_with_device(BOAT_DEVICE_CPU);
    if (!graph) {
        fprintf(stderr, "ERROR: Failed to create computation graph\n");
        boat_autodiff_context_free(ctx);
        return 1;
    }
    boat_autodiff_context_set_graph(ctx, graph);
    fprintf(stderr, "[TEST] Created computation graph=%p\n", (void*)graph);
    printf("=== General Gradient Checking Tests ===\n\n");

    bool all_pass = true;

    // Test addition gradient
    all_pass = test_addition_gradient() && all_pass;

    // Test multiplication gradient
    all_pass = test_multiplication_gradient() && all_pass;

    printf("\n");
    if (all_pass) {
        printf("✅ All gradient checks PASSED\n");
        return 0;
    } else {
        printf("❌ Some gradient checks FAILED\n");
        return 1;
    }
}