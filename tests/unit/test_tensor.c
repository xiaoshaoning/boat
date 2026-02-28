// test_tensor.c - Tensor operations unit tests
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

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

        const char* name = boat_dtype_name(BOAT_DTYPE_FLOAT32);
        assert(name != NULL);
        printf("Float32 dtype name: %s\n", name);
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

        // Test BOOL
        boat_tensor_t* bool_tensor = boat_tensor_create(shape, 1, BOAT_DTYPE_BOOL, BOAT_DEVICE_CPU);
        assert(bool_tensor != NULL);
        boat_tensor_unref(bool_tensor);
    }

    printf("Tensor tests passed!\n");
    return 0;
}