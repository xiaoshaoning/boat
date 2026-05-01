// test_tensor.c - Tensor operations unit tests
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

int main() {
    printf("Testing tensor operations...\n");

    // Test 1: Tensor creation and properties
    {
        int64_t shape[] = {2, 3, 4};
        boat_tensor_t* tensor = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        assert(tensor != NULL);
        assert(boat_tensor_ndim(tensor) == 3);
        assert(boat_tensor_nelements(tensor) == 2*3*4);
        assert(boat_tensor_dtype(tensor) == BOAT_DTYPE_FLOAT32);
        assert(boat_tensor_device(tensor) == BOAT_DEVICE_CPU);

        const int64_t* retrieved_shape = boat_tensor_shape(tensor);
        assert(retrieved_shape[0] == 2);
        assert(retrieved_shape[1] == 3);
        assert(retrieved_shape[2] == 4);

        boat_tensor_unref(tensor);
    }

    // Test 2: Tensor from data
    {
        int64_t shape[] = {3};
        float data[] = {1.0f, 2.0f, 3.0f};
        boat_tensor_t* tensor = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);
        assert(tensor != NULL);

        float* tensor_data = (float*)boat_tensor_data(tensor);
        assert(tensor_data[0] == 1.0f);
        assert(tensor_data[1] == 2.0f);
        assert(tensor_data[2] == 3.0f);

        boat_tensor_unref(tensor);
    }

    // Test 3: Tensor reference counting
    {
        int64_t shape[] = {2, 2};
        boat_tensor_t* tensor = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        assert(tensor != NULL);

        boat_tensor_ref(tensor);  // Increase ref count
        boat_tensor_unref(tensor); // Decrease ref count (should not free)
        boat_tensor_unref(tensor); // Should free now
        // Note: after this, tensor is dangling pointer, but test continues
    }

    // Test 4: Data type size and name
    {
        assert(boat_dtype_size(BOAT_DTYPE_FLOAT32) == sizeof(float));
        assert(boat_dtype_size(BOAT_DTYPE_FLOAT64) == sizeof(double));
        assert(boat_dtype_size(BOAT_DTYPE_INT32) == sizeof(int32_t));
        assert(boat_dtype_size(BOAT_DTYPE_INT64) == sizeof(int64_t));
        assert(boat_dtype_size(BOAT_DTYPE_INT8) == sizeof(int8_t));
        assert(boat_dtype_size(BOAT_DTYPE_UINT8) == sizeof(uint8_t));
        assert(boat_dtype_size(BOAT_DTYPE_BFLOAT16) == 2);

        const char* name = boat_dtype_name(BOAT_DTYPE_FLOAT32);
        assert(name != NULL);
        printf("Float32 dtype name: %s\n", name);

        // BF16 conversion roundtrip
        float orig = 3.14159f;
        uint16_t bf16 = boat_f32_to_bf16(orig);
        float back = boat_bf16_to_f32(bf16);
        float rel_err = fabsf(back - orig) / orig;
        assert(rel_err < 0.01f);
        printf("BF16 roundtrip: %f -> %f (rel_err=%f)\n", orig, back, rel_err);

        // BF16 of 1.0 and 2.0 should be exactly representable
        assert(boat_bf16_to_f32(boat_f32_to_bf16(1.0f)) == 1.0f);
        assert(boat_bf16_to_f32(boat_f32_to_bf16(2.0f)) == 2.0f);
    }

    // Test 5: Different data types
    {
        int64_t shape[] = {3};

        // Test INT32
        boat_tensor_t* int_tensor = boat_tensor_create(shape, 1, BOAT_DTYPE_INT32, BOAT_DEVICE_CPU);
        assert(int_tensor != NULL);
        int32_t* int_data = (int32_t*)boat_tensor_data(int_tensor);
        int_data[0] = 42;
        assert(int_data[0] == 42);
        boat_tensor_unref(int_tensor);

        // Test INT64
        boat_tensor_t* int64_tensor = boat_tensor_create(shape, 1, BOAT_DTYPE_INT64, BOAT_DEVICE_CPU);
        assert(int64_tensor != NULL);
        boat_tensor_unref(int64_tensor);

        // Test UINT8
        boat_tensor_t* uint8_tensor = boat_tensor_create(shape, 1, BOAT_DTYPE_UINT8, BOAT_DEVICE_CPU);
        assert(uint8_tensor != NULL);
        boat_tensor_unref(uint8_tensor);

        // Test INT8
        boat_tensor_t* int8_tensor = boat_tensor_create(shape, 1, BOAT_DTYPE_INT8, BOAT_DEVICE_CPU);
        assert(int8_tensor != NULL);
        int8_t* int8_data = (int8_t*)boat_tensor_data(int8_tensor);
        int8_data[0] = -42;
        assert(int8_data[0] == -42);
        boat_tensor_unref(int8_tensor);

        // Test BFLOAT16
        {
            boat_tensor_t* bf16_tensor = boat_tensor_create(shape, 1, BOAT_DTYPE_BFLOAT16, BOAT_DEVICE_CPU);
            assert(bf16_tensor != NULL);
            assert(boat_tensor_nbytes(bf16_tensor) == 3 * 2);
            uint16_t* bf16_data = (uint16_t*)boat_tensor_data(bf16_tensor);
            bf16_data[0] = boat_f32_to_bf16(1.5f);
            bf16_data[1] = boat_f32_to_bf16(-2.0f);
            bf16_data[2] = boat_f32_to_bf16(0.0f);
            assert(boat_bf16_to_f32(bf16_data[0]) == 1.5f);
            assert(boat_bf16_to_f32(bf16_data[1]) == -2.0f);
            assert(boat_bf16_to_f32(bf16_data[2]) == 0.0f);
            boat_tensor_unref(bf16_tensor);
        }

        // Test BOOL
        boat_tensor_t* bool_tensor = boat_tensor_create(shape, 1, BOAT_DTYPE_BOOL, BOAT_DEVICE_CPU);
        assert(bool_tensor != NULL);
        boat_tensor_unref(bool_tensor);
    }

    printf("Tensor tests passed!\n");
    return 0;
}