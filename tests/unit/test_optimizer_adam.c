// test_optimizer_adam.c - Adam optimizer unit tests
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat/optimizers.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

int main() {
    printf("Testing Adam optimizer...\n");

    // Test 1: Adam optimizer creation
    {
        boat_optimizer_t* optimizer = boat_adam_optimizer_create(0.001f, 0.9f, 0.999f, 1e-8f);
        assert(optimizer != NULL);

        // Test that invalid parameters are rejected
        boat_optimizer_t* invalid1 = boat_adam_optimizer_create(0.0f, 0.9f, 0.999f, 1e-8f);
        assert(invalid1 == NULL);

        boat_optimizer_t* invalid2 = boat_adam_optimizer_create(0.001f, 1.1f, 0.999f, 1e-8f);
        assert(invalid2 == NULL);

        boat_optimizer_t* invalid3 = boat_adam_optimizer_create(0.001f, 0.9f, 1.5f, 1e-8f);
        assert(invalid3 == NULL);

        boat_optimizer_free(optimizer);
        printf("  Test 1 passed: Adam optimizer creation\n");
    }

    // Test 2: Parameter registration
    {
        boat_optimizer_t* optimizer = boat_adam_optimizer_create(0.001f, 0.9f, 0.999f, 1e-8f);
        assert(optimizer != NULL);

        // Create a parameter tensor and gradient tensor
        int64_t shape[] = {3, 2};
        boat_tensor_t* param = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* grad = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

        assert(param != NULL);
        assert(grad != NULL);

        // Initialize parameter with some values
        float* param_data = (float*)boat_tensor_data(param);
        param_data[0] = 1.0f;
        param_data[1] = 2.0f;
        param_data[2] = 3.0f;
        param_data[3] = 4.0f;
        param_data[4] = 5.0f;
        param_data[5] = 6.0f;

        // Initialize gradient with some values
        float* grad_data = (float*)boat_tensor_data(grad);
        grad_data[0] = 0.1f;
        grad_data[1] = 0.2f;
        grad_data[2] = 0.3f;
        grad_data[3] = 0.4f;
        grad_data[4] = 0.5f;
        grad_data[5] = 0.6f;

        // Register parameter with optimizer
        boat_optimizer_add_parameter(optimizer, param, grad);

        // TODO: Verify that parameter was registered
        // Currently there's no API to check number of parameters

        boat_optimizer_free(optimizer);
        boat_tensor_unref(param);
        boat_tensor_unref(grad);

        printf("  Test 2 passed: Parameter registration\n");
    }

    // Test 3: Single optimization step
    {
        boat_optimizer_t* optimizer = boat_adam_optimizer_create(0.01f, 0.9f, 0.999f, 1e-8f);
        assert(optimizer != NULL);

        // Create a simple 1D parameter
        int64_t shape[] = {1};
        boat_tensor_t* param = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* grad = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

        assert(param != NULL);
        assert(grad != NULL);

        // Initialize parameter and gradient
        float* param_data = (float*)boat_tensor_data(param);
        float* grad_data = (float*)boat_tensor_data(grad);

        param_data[0] = 5.0f;
        grad_data[0] = 2.0f;  // Gradient points upward

        // Register parameter
        boat_optimizer_add_parameter(optimizer, param, grad);

        // Take optimization step
        boat_optimizer_step(optimizer);

        // Parameter should decrease because we subtract gradient * learning_rate
        // But Adam has momentum and adaptive learning rate, so hard to predict exact value
        // Just check that parameter changed
        assert(fabs(param_data[0] - 5.0f) > 1e-6f);

        boat_optimizer_free(optimizer);
        boat_tensor_unref(param);
        boat_tensor_unref(grad);

        printf("  Test 3 passed: Single optimization step\n");
    }

    // Test 4: Multiple optimization steps
    {
        boat_optimizer_t* optimizer = boat_adam_optimizer_create(0.1f, 0.9f, 0.999f, 1e-8f);
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

        // Take multiple steps with same gradient
        float prev_param0 = param_data[0];
        float prev_param1 = param_data[1];

        for (int i = 0; i < 5; i++) {
            boat_optimizer_step(optimizer);

            // Parameters should keep changing
            assert(fabs(param_data[0] - prev_param0) > 1e-6f || i == 0);
            assert(fabs(param_data[1] - prev_param1) > 1e-6f || i == 0);

            prev_param0 = param_data[0];
            prev_param1 = param_data[1];
        }

        boat_optimizer_free(optimizer);
        boat_tensor_unref(param);
        boat_tensor_unref(grad);

        printf("  Test 4 passed: Multiple optimization steps\n");
    }

    // Test 5: Zero gradient
    {
        boat_optimizer_t* optimizer = boat_adam_optimizer_create(0.01f, 0.9f, 0.999f, 1e-8f);
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

        // Check gradients are zero
        assert(fabs(grad_data[0]) < 1e-6f);
        assert(fabs(grad_data[1]) < 1e-6f);
        assert(fabs(grad_data[2]) < 1e-6f);

        // Take optimization step - should not change parameters much
        // (might change slightly due to epsilon in denominator)
        float prev_param0 = param_data[0];
        float prev_param1 = param_data[1];
        float prev_param2 = param_data[2];

        boat_optimizer_step(optimizer);

        // Parameters should change very little with zero gradient
        assert(fabs(param_data[0] - prev_param0) < 1e-4f);
        assert(fabs(param_data[1] - prev_param1) < 1e-4f);
        assert(fabs(param_data[2] - prev_param2) < 1e-4f);

        boat_optimizer_free(optimizer);
        boat_tensor_unref(param);
        boat_tensor_unref(grad);

        printf("  Test 5 passed: Zero gradient\n");
    }

    // Test 6: Multiple parameters
    {
        boat_optimizer_t* optimizer = boat_adam_optimizer_create(0.05f, 0.9f, 0.999f, 1e-8f);
        assert(optimizer != NULL);

        // Create two parameters with different shapes
        int64_t shape1[] = {2, 2};
        int64_t shape2[] = {3};

        boat_tensor_t* param1 = boat_tensor_create(shape1, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* grad1 = boat_tensor_create(shape1, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* param2 = boat_tensor_create(shape2, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        boat_tensor_t* grad2 = boat_tensor_create(shape2, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

        assert(param1 != NULL && grad1 != NULL);
        assert(param2 != NULL && grad2 != NULL);

        // Initialize with random values
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

        printf("  Test 6 passed: Multiple parameters\n");
    }

    printf("All Adam optimizer tests passed!\n");
    return 0;
}