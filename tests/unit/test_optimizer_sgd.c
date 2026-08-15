// test_optimizer_sgd.c - SGD optimizer unit tests
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/optimizers.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

int main() {
    printf("Testing SGD optimizer...\n");

    // Test 1: SGD optimizer creation
    {
        boat_optimizer_t* optimizer = boat_sgd_optimizer_create(0.01f, 0.9f);
        assert(optimizer != NULL);

        // Test that invalid parameters are rejected
        boat_optimizer_t* invalid1 = boat_sgd_optimizer_create(0.0f, 0.9f);
        assert(invalid1 == NULL);

        boat_optimizer_t* invalid2 = boat_sgd_optimizer_create(0.01f, -0.1f);
        assert(invalid2 == NULL);

        boat_optimizer_t* valid_no_momentum = boat_sgd_optimizer_create(0.01f, 0.0f);
        assert(valid_no_momentum != NULL);
        boat_optimizer_free(valid_no_momentum);

        boat_optimizer_free(optimizer);
        printf("  Test 1 passed: SGD optimizer creation\n");
    }

    // Test 2: Parameter registration
    {
        boat_optimizer_t* optimizer = boat_sgd_optimizer_create(0.01f, 0.9f);
        assert(optimizer != NULL);

        int64_t shape[] = {3, 2};
        boat_tensor_t* param = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* grad = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

        assert(param != NULL);
        assert(grad != NULL);

        float* param_data = (float*)boat_tensor_data(param);
        param_data[0] = 1.0f;
        param_data[1] = 2.0f;
        param_data[2] = 3.0f;
        param_data[3] = 4.0f;
        param_data[4] = 5.0f;
        param_data[5] = 6.0f;

        float* grad_data = (float*)boat_tensor_data(grad);
        grad_data[0] = 0.1f;
        grad_data[1] = 0.2f;
        grad_data[2] = 0.3f;
        grad_data[3] = 0.4f;
        grad_data[4] = 0.5f;
        grad_data[5] = 0.6f;

        boat_optimizer_add_parameter(optimizer, param, grad);

        boat_optimizer_free(optimizer);
        boat_tensor_unref(param);
        boat_tensor_unref(grad);

        printf("  Test 2 passed: Parameter registration\n");
    }

    // Test 3: Vanilla SGD step (no momentum) — predictable update
    {
        boat_optimizer_t* optimizer = boat_sgd_optimizer_create(0.01f, 0.0f); // no momentum
        assert(optimizer != NULL);

        int64_t shape[] = {1};
        boat_tensor_t* param = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* grad = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

        assert(param != NULL);
        assert(grad != NULL);

        float* param_data = (float*)boat_tensor_data(param);
        float* grad_data = (float*)boat_tensor_data(grad);

        param_data[0] = 5.0f;
        grad_data[0] = 2.0f;

        boat_optimizer_add_parameter(optimizer, param, grad);
        boat_optimizer_step(optimizer);

        // Vanilla SGD: param -= lr * grad => 5.0 - 0.01 * 2.0 = 4.98
        assert(fabs(param_data[0] - 4.98f) < 1e-5f);

        boat_optimizer_free(optimizer);
        boat_tensor_unref(param);
        boat_tensor_unref(grad);

        printf("  Test 3 passed: Vanilla SGD step\n");
    }

    // Test 4: SGD with momentum — multiple steps
    {
        boat_optimizer_t* optimizer = boat_sgd_optimizer_create(0.1f, 0.8f);
        assert(optimizer != NULL);

        int64_t shape[] = {2};
        boat_tensor_t* param = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* grad = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

        assert(param != NULL);
        assert(grad != NULL);

        float* param_data = (float*)boat_tensor_data(param);
        float* grad_data = (float*)boat_tensor_data(grad);

        param_data[0] = 10.0f;
        param_data[1] = -5.0f;
        grad_data[0] = 1.0f;
        grad_data[1] = -0.5f;

        boat_optimizer_add_parameter(optimizer, param, grad);

        float prev_param0 = param_data[0];
        float prev_param1 = param_data[1];

        for (int i = 0; i < 5; i++) {
            boat_optimizer_step(optimizer);

            assert(fabs(param_data[0] - prev_param0) > 1e-6f || i == 0);
            assert(fabs(param_data[1] - prev_param1) > 1e-6f || i == 0);

            prev_param0 = param_data[0];
            prev_param1 = param_data[1];
        }

        // Verify momentum accelerates: param[0] with positive grad should decrease
        assert(param_data[0] < 9.0f);

        boat_optimizer_free(optimizer);
        boat_tensor_unref(param);
        boat_tensor_unref(grad);

        printf("  Test 4 passed: SGD with momentum\n");
    }

    // Test 5: SGD with Nesterov momentum
    {
        boat_optimizer_t* optimizer = boat_sgd_optimizer_create(0.1f, 0.8f);
        assert(optimizer != NULL);

        boat_sgd_set_nesterov(optimizer, 1);

        int64_t shape[] = {2};
        boat_tensor_t* param = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* grad = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

        assert(param != NULL);
        assert(grad != NULL);

        float* param_data = (float*)boat_tensor_data(param);
        float* grad_data = (float*)boat_tensor_data(grad);

        param_data[0] = 10.0f;
        param_data[1] = -5.0f;
        grad_data[0] = 1.0f;
        grad_data[1] = -0.5f;

        boat_optimizer_add_parameter(optimizer, param, grad);

        // With Nesterov: v = 0.8*v + g, param -= 0.1*(g + 0.8*v)
        // Step 1: v = 0 + 1 = 1, param[0] -= 0.1*(1 + 0.8*1) = 0.18 => 9.82
        // Step 2: v = 0.8*1 + 1 = 1.8, param[0] -= 0.1*(1 + 0.8*1.8) = 0.244 => 9.576
        // Step 3: v = 0.8*1.8 + 1 = 2.44, param[0] -= 0.1*(1 + 0.8*2.44) = 0.2952 => 9.2808

        boat_optimizer_step(optimizer);
        assert(fabs(param_data[0] - 9.82f) < 1e-4f);

        boat_optimizer_step(optimizer);
        assert(fabs(param_data[0] - 9.576f) < 1e-4f);

        boat_optimizer_step(optimizer);
        assert(fabs(param_data[0] - 9.2808f) < 1e-4f);

        // Without Nesterov, the same steps would give different values
        // Step 1: v = 1, param -= 0.1*1 = 0.1 => 9.9
        // Step 2: v = 0.8*1 + 1 = 1.8, param -= 0.1*1.8 = 0.18 => 9.72
        // So Nesterov is giving different (more aggressive) updates

        boat_optimizer_free(optimizer);
        boat_tensor_unref(param);
        boat_tensor_unref(grad);

        printf("  Test 5 passed: SGD with Nesterov momentum\n");
    }

    // Test 6: Zero gradient
    {
        boat_optimizer_t* optimizer = boat_sgd_optimizer_create(0.01f, 0.9f);
        assert(optimizer != NULL);

        int64_t shape[] = {3};
        boat_tensor_t* param = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* grad = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

        assert(param != NULL);
        assert(grad != NULL);

        float* param_data = (float*)boat_tensor_data(param);
        float* grad_data = (float*)boat_tensor_data(grad);

        param_data[0] = 1.0f;
        param_data[1] = 2.0f;
        param_data[2] = 3.0f;

        grad_data[0] = 0.5f;
        grad_data[1] = -0.5f;
        grad_data[2] = 0.2f;

        boat_optimizer_add_parameter(optimizer, param, grad);

        // Zero gradients
        boat_optimizer_zero_grad(optimizer);

        assert(fabs(grad_data[0]) < 1e-6f);
        assert(fabs(grad_data[1]) < 1e-6f);
        assert(fabs(grad_data[2]) < 1e-6f);

        // Take step — with zero gradient and momentum=0.9:
        // After zero_grad, velocity is still 0, so step with zero grad keeps velocity = 0
        // Thus param should not change
        float prev_param0 = param_data[0];
        float prev_param1 = param_data[1];
        float prev_param2 = param_data[2];

        boat_optimizer_step(optimizer);

        assert(fabs(param_data[0] - prev_param0) < 1e-6f);
        assert(fabs(param_data[1] - prev_param1) < 1e-6f);
        assert(fabs(param_data[2] - prev_param2) < 1e-6f);

        boat_optimizer_free(optimizer);
        boat_tensor_unref(param);
        boat_tensor_unref(grad);

        printf("  Test 6 passed: Zero gradient\n");
    }

    // Test 7: Multiple parameters
    {
        boat_optimizer_t* optimizer = boat_sgd_optimizer_create(0.05f, 0.5f);
        assert(optimizer != NULL);

        int64_t shape1[] = {2, 2};
        int64_t shape2[] = {3};

        boat_tensor_t* param1 = boat_tensor_create(shape1, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* grad1 = boat_tensor_create(shape1, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* param2 = boat_tensor_create(shape2, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* grad2 = boat_tensor_create(shape2, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

        assert(param1 != NULL && grad1 != NULL);
        assert(param2 != NULL && grad2 != NULL);

        float* param1_data = (float*)boat_tensor_data(param1);
        float* grad1_data = (float*)boat_tensor_data(grad1);
        float* param2_data = (float*)boat_tensor_data(param2);
        float* grad2_data = (float*)boat_tensor_data(grad2);

        for (int i = 0; i < 4; i++) {
            param1_data[i] = (float)i;
            grad1_data[i] = 0.1f * (float)(i + 1);
        }

        for (int i = 0; i < 3; i++) {
            param2_data[i] = (float)(i * 2);
            grad2_data[i] = -0.05f * (float)(i + 1);
        }

        // Register both parameters
        boat_optimizer_add_parameter(optimizer, param1, grad1);
        boat_optimizer_add_parameter(optimizer, param2, grad2);

        // Take optimization step
        boat_optimizer_step(optimizer);

        // Both parameters should have changed
        int changed1 = 0;
        int changed2 = 0;

        for (int i = 0; i < 4; i++) {
            if (fabs(param1_data[i] - (float)i) > 1e-6f) changed1 = 1;
        }

        for (int i = 0; i < 3; i++) {
            if (fabs(param2_data[i] - (float)(i * 2)) > 1e-6f) changed2 = 1;
        }

        assert(changed1 == 1);
        assert(changed2 == 1);

        boat_optimizer_free(optimizer);
        boat_tensor_unref(param1);
        boat_tensor_unref(grad1);
        boat_tensor_unref(param2);
        boat_tensor_unref(grad2);

        printf("  Test 7 passed: Multiple parameters\n");
    }

    printf("All SGD optimizer tests passed!\n");
    return 0;
}
