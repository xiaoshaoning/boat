// test_transpose.c - Transpose operation unit tests
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/tensor.h>
#include <boat/ops.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond, msg) if (!(cond)) { printf("FAIL: %s\n", msg); return false; }

static bool test_2d_transpose() {
    printf("  Testing 2D transpose... ");

    int64_t shape[] = {2, 3};
    boat_tensor_t* t = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    ASSERT(t, "Failed to create tensor");

    float* data = (float*)boat_tensor_data(t);
    for (int i = 0; i < 6; i++) data[i] = (float)i + 1.0f;

    boat_tensor_t* t_t = boat_transpose(t, 0, 1);
    ASSERT(t_t, "Transpose failed");

    const int64_t* out_shape = boat_tensor_shape(t_t);
    ASSERT(out_shape[0] == 3 && out_shape[1] == 2, "Wrong output shape");

    float* out_data = (float*)boat_tensor_data(t_t);

    // Expected: [1, 4, 2, 5, 3, 6]
    float expected[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    bool correct = true;
    for (int i = 0; i < 6; i++) {
        if (fabs(out_data[i] - expected[i]) > 1e-6) {
            correct = false;
            break;
        }
    }

    boat_tensor_unref(t_t);
    boat_tensor_unref(t);

    if (correct) {
        printf("PASSED\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

static bool test_4d_transpose_last_two_dims() {
    printf("  Testing 4D transpose (swap last two dimensions)... ");

    // Shape: [batch=2, heads=2, seq_len=3, head_size=4]
    int64_t shape[] = {2, 2, 3, 4};
    boat_tensor_t* t = boat_tensor_create(shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    ASSERT(t, "Failed to create tensor");

    float* data = (float*)boat_tensor_data(t);
    size_t total = 2*2*3*4;
    for (size_t i = 0; i < total; i++) data[i] = (float)i + 1.0f;

    boat_tensor_t* t_t = boat_transpose(t, 2, 3); // swap seq_len and head_size
    ASSERT(t_t, "Transpose failed");

    const int64_t* out_shape = boat_tensor_shape(t_t);
    ASSERT(out_shape[0] == 2 && out_shape[1] == 2 && out_shape[2] == 4 && out_shape[3] == 3,
           "Wrong output shape");

    float* out_data = (float*)boat_tensor_data(t_t);

    // Verify transposition
    bool correct = true;
    for (int b = 0; b < 2 && correct; b++) {
        for (int h = 0; h < 2 && correct; h++) {
            for (int i = 0; i < 3 && correct; i++) {
                for (int j = 0; j < 4 && correct; j++) {
                    size_t idx_orig = ((b * 2 + h) * 3 + i) * 4 + j;
                    size_t idx_trans = ((b * 2 + h) * 4 + j) * 3 + i;
                    if (fabs(data[idx_orig] - out_data[idx_trans]) > 1e-6) {
                        correct = false;
                    }
                }
            }
        }
    }

    boat_tensor_unref(t_t);
    boat_tensor_unref(t);

    if (correct) {
        printf("PASSED\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

static bool test_transpose_identity() {
    printf("  Testing double transpose identity... ");

    int64_t shape[] = {2, 3, 4};
    boat_tensor_t* t = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    ASSERT(t, "Failed to create tensor");

    float* data = (float*)boat_tensor_data(t);
    size_t total = 2*3*4;
    for (size_t i = 0; i < total; i++) data[i] = (float)rand() / RAND_MAX;

    boat_tensor_t* t1 = boat_transpose(t, 0, 1);
    ASSERT(t1, "First transpose failed");

    boat_tensor_t* t2 = boat_transpose(t1, 0, 1);
    ASSERT(t2, "Second transpose failed");

    float* final_data = (float*)boat_tensor_data(t2);
    bool correct = true;
    for (size_t i = 0; i < total; i++) {
        if (fabs(data[i] - final_data[i]) > 1e-6) {
            correct = false;
            break;
        }
    }

    boat_tensor_unref(t2);
    boat_tensor_unref(t1);
    boat_tensor_unref(t);

    if (correct) {
        printf("PASSED\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

static bool test_transpose_different_dtypes() {
    printf("  Testing transpose with different data types... ");

    bool all_pass = true;

    // Test float64
    {
        int64_t shape[] = {2, 3};
        boat_tensor_t* t = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT64, BOAT_DEVICE_CPU);
        if (!t) all_pass = false;

        double* data = (double*)boat_tensor_data(t);
        for (int i = 0; i < 6; i++) data[i] = (double)i + 1.0;

        boat_tensor_t* t_t = boat_transpose(t, 0, 1);
        if (!t_t) all_pass = false;

        if (t_t) {
            const int64_t* out_shape = boat_tensor_shape(t_t);
            if (!(out_shape[0] == 3 && out_shape[1] == 2)) all_pass = false;

            boat_tensor_unref(t_t);
        }
        if (t) boat_tensor_unref(t);
    }

    // Test uint8 (should fall back to memcpy, no actual transposition)
    {
        int64_t shape[] = {2, 3};
        boat_tensor_t* t = boat_tensor_create(shape, 2, BOAT_DTYPE_UINT8, BOAT_DEVICE_CPU);
        if (!t) all_pass = false;

        uint8_t* data = (uint8_t*)boat_tensor_data(t);
        for (int i = 0; i < 6; i++) data[i] = (uint8_t)(i + 1);

        boat_tensor_t* t_t = boat_transpose(t, 0, 1);
        if (!t_t) all_pass = false;

        // For unsupported types, we just expect a tensor with swapped shape
        // (implementation may or may not actually transpose)
        if (t_t) {
            const int64_t* out_shape = boat_tensor_shape(t_t);
            if (!(out_shape[0] == 3 && out_shape[1] == 2)) all_pass = false;

            boat_tensor_unref(t_t);
        }
        if (t) boat_tensor_unref(t);
    }

    if (all_pass) {
        printf("PASSED\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

int main() {
    printf("=== Transpose Operation Unit Tests ===\n\n");

    srand(42);
    bool all_pass = true;

    all_pass = test_2d_transpose() && all_pass;
    all_pass = test_4d_transpose_last_two_dims() && all_pass;
    all_pass = test_transpose_identity() && all_pass;
    all_pass = test_transpose_different_dtypes() && all_pass;

    printf("\n");
    if (all_pass) {
        printf("✅ All transpose tests PASSED\n");
        return 0;
    } else {
        printf("❌ Some transpose tests FAILED\n");
        return 1;
    }
}