// memory.h - Memory management for deep learning framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_MEMORY_H
#define BOAT_MEMORY_H

#include <stddef.h>
#include <stdio.h>
#include "tensor.h"
#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Memory statistics structure
typedef struct {
    size_t allocated_bytes;
    size_t allocated_blocks;
    size_t peak_allocated_bytes;
    size_t freed_bytes;
    size_t freed_blocks;
} boat_memory_stats_t;

// Memory allocation functions
BOAT_API void* boat_memory_allocate(size_t size, boat_device_t device,
                                    const char* file, int line);
BOAT_API void* boat_memory_allocate_zero(size_t size, boat_device_t device,
                                         const char* file, int line);
BOAT_API void* boat_memory_reallocate(void* ptr, size_t new_size, boat_device_t device,
                                      const char* file, int line);
BOAT_API void boat_memory_free(void* ptr);
BOAT_API void boat_memory_free_safe(void** ptr_ptr);

// Device-specific memory allocation
BOAT_API void* boat_memory_allocate_device(size_t size, boat_device_t device,
                                           const char* file, int line);
BOAT_API void boat_memory_free_device(void* ptr, boat_device_t device);

// Aligned memory allocation
BOAT_API void* boat_memory_allocate_aligned(size_t size, size_t alignment,
                                            boat_device_t device,
                                            const char* file, int line);
BOAT_API void boat_memory_free_aligned(const void* aligned_ptr);

// Memory operations
BOAT_API void boat_memory_copy(void* dest, const void* src, size_t size,
                               boat_device_t dest_device, boat_device_t src_device);
BOAT_API void boat_memory_set(void* dest, int value, size_t size, boat_device_t device);

// Memory statistics
BOAT_API boat_memory_stats_t boat_memory_get_stats();
BOAT_API void boat_memory_reset_stats();
BOAT_API void boat_memory_print_stats(FILE* stream);

// Memory pool management
typedef struct boat_memory_pool_t boat_memory_pool_t;

BOAT_API boat_memory_pool_t* boat_memory_pool_create(size_t block_size, size_t initial_blocks);
BOAT_API void boat_memory_pool_free(boat_memory_pool_t* pool);
BOAT_API void* boat_memory_pool_alloc(boat_memory_pool_t* pool, size_t size);
BOAT_API void boat_memory_pool_free_block(boat_memory_pool_t* pool, void* block);
BOAT_API void boat_memory_pool_clear(boat_memory_pool_t* pool);
BOAT_API size_t boat_memory_pool_allocated_blocks(const boat_memory_pool_t* pool);
BOAT_API size_t boat_memory_pool_free_blocks(const boat_memory_pool_t* pool);
BOAT_API size_t boat_memory_pool_total_memory(const boat_memory_pool_t* pool);

// Arena allocator
typedef struct boat_memory_arena_t boat_memory_arena_t;

BOAT_API boat_memory_arena_t* boat_memory_arena_create(size_t initial_size);
BOAT_API void boat_memory_arena_free(boat_memory_arena_t* arena);
BOAT_API void* boat_memory_arena_alloc(boat_memory_arena_t* arena, size_t size);
BOAT_API void boat_memory_arena_reset(boat_memory_arena_t* arena);
BOAT_API size_t boat_memory_arena_used(const boat_memory_arena_t* arena);
BOAT_API size_t boat_memory_arena_capacity(const boat_memory_arena_t* arena);

// Convenience macros
#ifdef BOAT_MEMORY_DEBUG
#define boat_malloc(size, device) boat_memory_allocate(size, device, __FILE__, __LINE__)
#define boat_calloc(size, device) boat_memory_allocate_zero(size, device, __FILE__, __LINE__)
#define boat_realloc(ptr, size, device) boat_memory_reallocate(ptr, size, device, __FILE__, __LINE__)
#define boat_free(ptr) boat_memory_free(ptr)
#else
#define boat_malloc(size, device) boat_memory_allocate(size, device, NULL, 0)
#define boat_calloc(size, device) boat_memory_allocate_zero(size, device, NULL, 0)
#define boat_realloc(ptr, size, device) boat_memory_reallocate(ptr, size, device, NULL, 0)
#define boat_free(ptr) boat_memory_free(ptr)
#endif

#ifdef __cplusplus
}
#endif

#endif // BOAT_MEMORY_H