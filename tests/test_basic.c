// test_basic.c - Basic functionality test for Boat framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

int main() {
    printf("=== Boat Framework Basic Test ===\n");
    printf("Testing core functionality...\n");

    // Initialize memory tracking (optional)
    boat_memory_reset_stats();

    // Test 1: Tensor creation and basic properties
    {
        printf("Test 1: Tensor creation...\n");
        int64_t shape[] = {2, 3};
        boat_tensor_t* tensor = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        assert(tensor != NULL && "Failed to create tensor");
        assert(boat_tensor_ndim(tensor) == 2);
        assert(boat_tensor_nelements(tensor) == 6);
        assert(boat_tensor_dtype(tensor) == BOAT_DTYPE_FLOAT32);
        assert(boat_tensor_device(tensor) == BOAT_DEVICE_CPU);

        const int64_t* retrieved_shape = boat_tensor_shape(tensor);
        assert(retrieved_shape[0] == 2);
        assert(retrieved_shape[1] == 3);

        printf("  Shape: [%lld, %lld]\n", retrieved_shape[0], retrieved_shape[1]);
        printf("  Elements: %zu\n", boat_tensor_nelements(tensor));
        printf("  Data type: %s\n", boat_dtype_name(boat_tensor_dtype(tensor)));

        boat_tensor_unref(tensor);
        printf("  Test 1 passed!\n");
    }

    // Test 2: Tensor from data
    {
        printf("\nTest 2: Tensor from data...\n");
        int64_t shape[] = {4};
        float data[] = {1.5f, 2.5f, 3.5f, 4.5f};
        boat_tensor_t* tensor = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);
        assert(tensor != NULL && "Failed to create tensor from data");

        float* tensor_data = (float*)boat_tensor_data(tensor);
        for (int i = 0; i < 4; i++) {
            assert(tensor_data[i] == data[i]);
        }
        printf("  Data verified: [%.1f, %.1f, %.1f, %.1f]\n",
               tensor_data[0], tensor_data[1], tensor_data[2], tensor_data[3]);

        boat_tensor_unref(tensor);
        printf("  Test 2 passed!\n");
    }

    // Test 3: Reference counting
    {
        printf("\nTest 3: Reference counting...\n");
        int64_t shape[] = {3};
        boat_tensor_t* tensor = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        assert(tensor != NULL);

        // Increase reference count
        boat_tensor_ref(tensor);
        // First unref shouldn't free
        boat_tensor_unref(tensor);
        printf("  First unref completed (tensor should still exist)\n");

        // Second unref should free
        boat_tensor_unref(tensor);
        printf("  Second unref completed (tensor should be freed)\n");
        printf("  Test 3 passed!\n");
    }

    // Test 4: Data type information
    {
        printf("\nTest 4: Data type information...\n");
        assert(boat_dtype_size(BOAT_DTYPE_FLOAT32) == sizeof(float));
        assert(boat_dtype_size(BOAT_DTYPE_INT32) == sizeof(int32_t));

        const char* float32_name = boat_dtype_name(BOAT_DTYPE_FLOAT32);
        const char* int32_name = boat_dtype_name(BOAT_DTYPE_INT32);
        assert(float32_name != NULL);
        assert(int32_name != NULL);

        printf("  FLOAT32 size: %zu, name: %s\n",
               boat_dtype_size(BOAT_DTYPE_FLOAT32), float32_name);
        printf("  INT32 size: %zu, name: %s\n",
               boat_dtype_size(BOAT_DTYPE_INT32), int32_name);
        printf("  Test 4 passed!\n");
    }

    // Test 5: Multiple data types
    {
        printf("\nTest 5: Multiple data types...\n");
        int64_t shape[] = {2};

        // Test INT32
        boat_tensor_t* int_tensor = boat_tensor_create(shape, 1, BOAT_DTYPE_INT32, BOAT_DEVICE_CPU);
        assert(int_tensor != NULL);
        int32_t* int_data = (int32_t*)boat_tensor_data(int_tensor);
        int_data[0] = 42;
        int_data[1] = 100;
        assert(int_data[0] == 42);
        assert(int_data[1] == 100);
        boat_tensor_unref(int_tensor);
        printf("  INT32 tensor: [%d, %d]\n", 42, 100);

        // Test INT64
        boat_tensor_t* int64_tensor = boat_tensor_create(shape, 1, BOAT_DTYPE_INT64, BOAT_DEVICE_CPU);
        assert(int64_tensor != NULL);
        boat_tensor_unref(int64_tensor);
        printf("  INT64 tensor created successfully\n");

        printf("  Test 5 passed!\n");
    }

    // Print memory statistics
    printf("\n=== Memory Statistics ===\n");
    boat_memory_stats_t stats = boat_memory_get_stats();
    printf("Allocated bytes: %zu\n", stats.allocated_bytes);
    printf("Allocated blocks: %zu\n", stats.allocated_blocks);
    printf("Freed bytes: %zu\n", stats.freed_bytes);
    printf("Freed blocks: %zu\n", stats.freed_blocks);

    // Check for memory leaks (should have freed everything)
    if (stats.allocated_blocks == stats.freed_blocks) {
        printf("\n✓ No memory leaks detected!\n");
    } else {
        printf("\n⚠ Potential memory leak detected!\n");
        printf("  Allocated blocks: %zu, Freed blocks: %zu\n",
               stats.allocated_blocks, stats.freed_blocks);
    }

    printf("\n=== All tests passed! ===\n");
    return 0;
}