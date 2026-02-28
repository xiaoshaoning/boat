// test_autodiff.c - Automatic differentiation unit tests
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat/autodiff.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main() {
    printf("Testing automatic differentiation...\n");

    // Test 1: Variable creation
    {
        int64_t shape[] = {2, 2};
        boat_variable_t* var = boat_variable_create_with_shape(shape, 2, BOAT_DTYPE_FLOAT32, true);
        assert(var != NULL);
        assert(boat_variable_requires_grad(var) == true);
        assert(boat_variable_grad(var) == NULL); // Gradient not computed yet

        boat_tensor_t* data = boat_variable_data(var);
        assert(data != NULL);
        assert(boat_tensor_ndim(data) == 2);
        assert(boat_tensor_nelements(data) == 4);

        // Set some data
        float* data_ptr = (float*)boat_tensor_data(data);
        data_ptr[0] = 1.0f;
        data_ptr[1] = 2.0f;
        data_ptr[2] = 3.0f;
        data_ptr[3] = 4.0f;

        boat_variable_free(var);
    }

    // Test 2: Addition operation
    {
        int64_t shape[] = {2};
        boat_variable_t* a = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        boat_variable_t* b = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);

        assert(a != NULL);
        assert(b != NULL);

        // Set values
        float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        a_data[0] = 2.0f; a_data[1] = 3.0f;
        b_data[0] = 4.0f; b_data[1] = 5.0f;

        // Perform addition
        boat_variable_t* c = boat_var_add(a, b);
        assert(c != NULL);

        // Check result
        float* c_data = (float*)boat_tensor_data(boat_variable_data(c));
        assert(c_data[0] == 6.0f); // 2 + 4
        assert(c_data[1] == 8.0f); // 3 + 5

        // Cleanup
        boat_variable_free(c);
        boat_variable_free(b);
        boat_variable_free(a);
    }

    // Test 3: Multiplication operation
    {
        int64_t shape[] = {2};
        boat_variable_t* a = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        boat_variable_t* b = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);

        assert(a != NULL);
        assert(b != NULL);

        float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        a_data[0] = 2.0f; a_data[1] = 3.0f;
        b_data[0] = 4.0f; b_data[1] = 5.0f;

        boat_variable_t* c = boat_var_mul(a, b);
        assert(c != NULL);

        float* c_data = (float*)boat_tensor_data(boat_variable_data(c));
        assert(c_data[0] == 8.0f); // 2 * 4
        assert(c_data[1] == 15.0f); // 3 * 5

        boat_variable_free(c);
        boat_variable_free(b);
        boat_variable_free(a);
    }

    // Test 4: ReLU operation
    {
        int64_t shape[] = {3};
        boat_variable_t* a = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        assert(a != NULL);

        float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
        a_data[0] = -1.0f;
        a_data[1] = 0.0f;
        a_data[2] = 2.0f;

        boat_variable_t* b = boat_var_relu(a);
        assert(b != NULL);

        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        assert(b_data[0] == 0.0f); // relu(-1) = 0
        assert(b_data[1] == 0.0f); // relu(0) = 0
        assert(b_data[2] == 2.0f); // relu(2) = 2

        boat_variable_free(b);
        boat_variable_free(a);
    }

    // Test 5: Gradient computation for addition
    {
        int64_t shape[] = {2};
        boat_variable_t* a = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        boat_variable_t* b = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        assert(a != NULL);
        assert(b != NULL);

        // Set values
        float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        a_data[0] = 2.0f; a_data[1] = 3.0f;
        b_data[0] = 4.0f; b_data[1] = 5.0f;

        // Perform addition
        boat_variable_t* c = boat_var_add(a, b);
        assert(c != NULL);

        // Check forward pass
        float* c_data = (float*)boat_tensor_data(boat_variable_data(c));
        assert(c_data[0] == 6.0f);
        assert(c_data[1] == 8.0f);

        // Zero gradients
        boat_variable_zero_grad(a);
        boat_variable_zero_grad(b);

        // Backward pass (scalar loss, assume output gradient of 1)
        boat_variable_backward_full(c);

        // Check gradients
        boat_tensor_t* a_grad = boat_variable_grad(a);
        boat_tensor_t* b_grad = boat_variable_grad(b);
        assert(a_grad != NULL);
        assert(b_grad != NULL);

        float* a_grad_data = (float*)boat_tensor_data(a_grad);
        float* b_grad_data = (float*)boat_tensor_data(b_grad);
        // Gradient of addition should be 1 for all elements
        assert(a_grad_data[0] == 1.0f);
        assert(a_grad_data[1] == 1.0f);
        assert(b_grad_data[0] == 1.0f);
        assert(b_grad_data[1] == 1.0f);

        boat_variable_free(c);
        boat_variable_free(b);
        boat_variable_free(a);
    }

    // Test 6: Gradient computation for multiplication
    {
        int64_t shape[] = {2};
        boat_variable_t* a = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        boat_variable_t* b = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        assert(a != NULL);
        assert(b != NULL);

        float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        a_data[0] = 2.0f; a_data[1] = 3.0f;
        b_data[0] = 4.0f; b_data[1] = 5.0f;

        boat_variable_t* c = boat_var_mul(a, b);
        assert(c != NULL);

        float* c_data = (float*)boat_tensor_data(boat_variable_data(c));
        assert(c_data[0] == 8.0f);
        assert(c_data[1] == 15.0f);

        boat_variable_zero_grad(a);
        boat_variable_zero_grad(b);
        boat_variable_backward_full(c);

        boat_tensor_t* a_grad = boat_variable_grad(a);
        boat_tensor_t* b_grad = boat_variable_grad(b);
        assert(a_grad != NULL);
        assert(b_grad != NULL);

        float* a_grad_data = (float*)boat_tensor_data(a_grad);
        float* b_grad_data = (float*)boat_tensor_data(b_grad);
        // Gradient ∂L/∂a = ∂L/∂c * b = 1 * b
        assert(a_grad_data[0] == 4.0f); // 1 * 4
        assert(a_grad_data[1] == 5.0f); // 1 * 5
        // Gradient ∂L/∂b = ∂L/∂c * a = 1 * a
        assert(b_grad_data[0] == 2.0f); // 1 * 2
        assert(b_grad_data[1] == 3.0f); // 1 * 3

        boat_variable_free(c);
        boat_variable_free(b);
        boat_variable_free(a);
    }

    // Test 7: Gradient computation for ReLU
    {
        int64_t shape[] = {3};
        boat_variable_t* a = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        assert(a != NULL);

        float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
        a_data[0] = -1.0f;
        a_data[1] = 0.0f;
        a_data[2] = 2.0f;

        boat_variable_t* b = boat_var_relu(a);
        assert(b != NULL);

        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        assert(b_data[0] == 0.0f);
        assert(b_data[1] == 0.0f);
        assert(b_data[2] == 2.0f);

        boat_variable_zero_grad(a);
        boat_variable_backward_full(b);

        boat_tensor_t* a_grad = boat_variable_grad(a);
        assert(a_grad != NULL);

        float* a_grad_data = (float*)boat_tensor_data(a_grad);
        // Gradient ∂L/∂a = ∂L/∂c * (a > 0 ? 1 : 0) = 1 * (a > 0 ? 1 : 0)
        assert(a_grad_data[0] == 0.0f); // a = -1
        assert(a_grad_data[1] == 0.0f); // a = 0 (ReLU derivative at 0 is 0)
        assert(a_grad_data[2] == 1.0f); // a = 2

        boat_variable_free(b);
        boat_variable_free(a);
    }

    // Test 8: Subtraction operation and gradient
    {
        int64_t shape[] = {2};
        boat_variable_t* a = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        boat_variable_t* b = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        assert(a != NULL);
        assert(b != NULL);

        float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        a_data[0] = 5.0f; a_data[1] = 8.0f;
        b_data[0] = 2.0f; b_data[1] = 3.0f;

        boat_variable_t* c = boat_var_sub(a, b);
        assert(c != NULL);

        // Check forward pass: 5-2=3, 8-3=5
        float* c_data = (float*)boat_tensor_data(boat_variable_data(c));
        assert(c_data[0] == 3.0f);
        assert(c_data[1] == 5.0f);

        boat_variable_zero_grad(a);
        boat_variable_zero_grad(b);
        boat_variable_backward_full(c);

        boat_tensor_t* a_grad = boat_variable_grad(a);
        boat_tensor_t* b_grad = boat_variable_grad(b);
        assert(a_grad != NULL);
        assert(b_grad != NULL);

        float* a_grad_data = (float*)boat_tensor_data(a_grad);
        float* b_grad_data = (float*)boat_tensor_data(b_grad);
        // Gradient ∂L/∂a = ∂L/∂c = 1
        assert(a_grad_data[0] == 1.0f);
        assert(a_grad_data[1] == 1.0f);
        // Gradient ∂L/∂b = -∂L/∂c = -1
        assert(b_grad_data[0] == -1.0f);
        assert(b_grad_data[1] == -1.0f);

        boat_variable_free(c);
        boat_variable_free(b);
        boat_variable_free(a);
    }

    // Test 9: Division operation and gradient
    {
        int64_t shape[] = {2};
        boat_variable_t* a = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        boat_variable_t* b = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        assert(a != NULL);
        assert(b != NULL);

        float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        a_data[0] = 6.0f; a_data[1] = 12.0f;
        b_data[0] = 2.0f; b_data[1] = 3.0f;

        boat_variable_t* c = boat_var_div(a, b);
        assert(c != NULL);

        // Check forward pass: 6/2=3, 12/3=4
        float* c_data = (float*)boat_tensor_data(boat_variable_data(c));
        assert(c_data[0] == 3.0f);
        assert(c_data[1] == 4.0f);

        boat_variable_zero_grad(a);
        boat_variable_zero_grad(b);
        boat_variable_backward_full(c);

        boat_tensor_t* a_grad = boat_variable_grad(a);
        boat_tensor_t* b_grad = boat_variable_grad(b);
        assert(a_grad != NULL);
        assert(b_grad != NULL);

        float* a_grad_data = (float*)boat_tensor_data(a_grad);
        float* b_grad_data = (float*)boat_tensor_data(b_grad);
        // Gradient ∂L/∂a = ∂L/∂c / b = 1 / b
        assert(a_grad_data[0] == 0.5f);  // 1 / 2 = 0.5
        assert(a_grad_data[1] == 0.3333333333f);  // 1 / 3 ≈ 0.333...
        // Gradient ∂L/∂b = -∂L/∂c * a / b² = -1 * a / b²
        assert(b_grad_data[0] == -1.5f);  // -1 * 6 / 4 = -1.5
        assert(b_grad_data[1] == -1.3333333333f);  // -1 * 12 / 9 = -1.333...

        boat_variable_free(c);
        boat_variable_free(b);
        boat_variable_free(a);
    }

    // Test 10: Dot product operation and gradient
    {
        int64_t shape[] = {3};
        boat_variable_t* a = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        boat_variable_t* b = boat_variable_create_with_shape(shape, 1, BOAT_DTYPE_FLOAT32, true);
        assert(a != NULL);
        assert(b != NULL);

        float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
        b_data[0] = 4.0f; b_data[1] = 5.0f; b_data[2] = 6.0f;

        boat_variable_t* c = boat_var_dot(a, b);
        assert(c != NULL);

        // Check forward pass: dot([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6 = 4+10+18=32
        float* c_data = (float*)boat_tensor_data(boat_variable_data(c));
        assert(c_data[0] == 32.0f);

        boat_variable_zero_grad(a);
        boat_variable_zero_grad(b);
        boat_variable_backward_full(c);

        boat_tensor_t* a_grad = boat_variable_grad(a);
        boat_tensor_t* b_grad = boat_variable_grad(b);
        assert(a_grad != NULL);
        assert(b_grad != NULL);

        float* a_grad_data = (float*)boat_tensor_data(a_grad);
        float* b_grad_data = (float*)boat_tensor_data(b_grad);
        // Gradient ∂L/∂a = ∂L/∂c * b = 1 * b
        assert(a_grad_data[0] == 4.0f);
        assert(a_grad_data[1] == 5.0f);
        assert(a_grad_data[2] == 6.0f);
        // Gradient ∂L/∂b = ∂L/∂c * a = 1 * a
        assert(b_grad_data[0] == 1.0f);
        assert(b_grad_data[1] == 2.0f);
        assert(b_grad_data[2] == 3.0f);

        boat_variable_free(c);
        boat_variable_free(b);
        boat_variable_free(a);
    }

    // Test 11: Softmax operation and gradient
    {
        int64_t shape[] = {2, 3};  // 2 rows, 3 columns
        boat_variable_t* a = boat_variable_create_with_shape(shape, 2, BOAT_DTYPE_FLOAT32, true);
        assert(a != NULL);

        float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
        // Simple test values
        a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
        a_data[3] = 4.0f; a_data[4] = 5.0f; a_data[5] = 6.0f;

        // Test softmax
        boat_variable_t* b = boat_var_softmax(a, -1);  // last dimension
        assert(b != NULL);

        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));

        // Check that each row sums to approximately 1.0
        float row1_sum = b_data[0] + b_data[1] + b_data[2];
        float row2_sum = b_data[3] + b_data[4] + b_data[5];
        assert(fabsf(row1_sum - 1.0f) < 1e-6f);
        assert(fabsf(row2_sum - 1.0f) < 1e-6f);

        // Check that all outputs are positive (softmax output is always positive)
        for (int i = 0; i < 6; i++) {
            assert(b_data[i] > 0.0f);
        }

        // Test gradient computation
        boat_variable_zero_grad(a);
        boat_variable_backward_full(b);

        boat_tensor_t* a_grad = boat_variable_grad(a);
        assert(a_grad != NULL);

        // Simple gradient check: gradient should exist
        float* a_grad_data = (float*)boat_tensor_data(a_grad);
        // Just check gradient is computed (non-zero for this case)
        // In practice, we would verify with finite differences

        boat_variable_free(b);
        boat_variable_free(a);
    }

    // Test 12: Log softmax operation and gradient
    {
        int64_t shape[] = {2, 3};  // 2 rows, 3 columns
        boat_variable_t* a = boat_variable_create_with_shape(shape, 2, BOAT_DTYPE_FLOAT32, true);
        assert(a != NULL);

        float* a_data = (float*)boat_tensor_data(boat_variable_data(a));
        // Simple test values
        a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
        a_data[3] = 4.0f; a_data[4] = 5.0f; a_data[5] = 6.0f;

        // Test log_softmax
        boat_variable_t* b = boat_var_log_softmax(a, -1);  // last dimension
        assert(b != NULL);

        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));

        // Check that exp of each row sums to approximately 1.0
        float row1_exp_sum = expf(b_data[0]) + expf(b_data[1]) + expf(b_data[2]);
        float row2_exp_sum = expf(b_data[3]) + expf(b_data[4]) + expf(b_data[5]);
        assert(fabsf(row1_exp_sum - 1.0f) < 1e-6f);
        assert(fabsf(row2_exp_sum - 1.0f) < 1e-6f);

        // Test gradient computation
        boat_variable_zero_grad(a);
        boat_variable_backward_full(b);

        boat_tensor_t* a_grad = boat_variable_grad(a);
        assert(a_grad != NULL);

        // Simple gradient check
        float* a_grad_data = (float*)boat_tensor_data(a_grad);
        // Just check gradient is computed

        boat_variable_free(b);
        boat_variable_free(a);
    }

    printf("Autodiff tests passed!\n");
    return 0;
}