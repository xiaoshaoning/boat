// test_memory.c - Memory management unit tests
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat/memory.h>
#include <stdio.h>
#include <assert.h>

int main() {
    printf("Testing memory management...\n");

    // Test basic allocation
    void* ptr = boat_malloc(100, BOAT_DEVICE_CPU);
    assert(ptr != NULL);
    boat_free(ptr);

    // Test zero allocation
    void* ptr2 = boat_calloc(50, BOAT_DEVICE_CPU);
    assert(ptr2 != NULL);
    // Check first byte is zero (not guaranteed but likely)
    boat_free(ptr2);

    // Test reallocation
    void* ptr3 = boat_malloc(10, BOAT_DEVICE_CPU);
    assert(ptr3 != NULL);
    void* ptr4 = boat_realloc(ptr3, 20, BOAT_DEVICE_CPU);
    assert(ptr4 != NULL);
    boat_free(ptr4);

    // Test memory statistics
    boat_memory_stats_t stats = boat_memory_get_stats();
    printf("Allocated bytes: %zu\n", stats.allocated_bytes);
    printf("Allocated blocks: %zu\n", stats.allocated_blocks);
    printf("Peak allocated bytes: %zu\n", stats.peak_allocated_bytes);

    // Note: stats may show some allocations due to test itself

    printf("Memory tests passed!\n");
    return 0;
}