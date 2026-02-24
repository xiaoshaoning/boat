// memory.c - Memory management for deep learning framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


// Global memory statistics
static boat_memory_stats_t g_memory_stats = {0};

// Memory allocation header for tracking
typedef struct {
    size_t size;
    boat_device_t device;
    const char* file;
    int line;
} boat_memory_header_t;

// Forward declarations
void boat_memory_free(void* ptr);

// Internal helper functions
static void update_alloc_stats(size_t size) {
    g_memory_stats.allocated_bytes += size;
    g_memory_stats.allocated_blocks += 1;

    // Update peak
    size_t current_bytes = g_memory_stats.allocated_bytes;
    if (current_bytes > g_memory_stats.peak_allocated_bytes) {
        g_memory_stats.peak_allocated_bytes = current_bytes;
    }
}

static void update_free_stats(size_t size) {
    g_memory_stats.allocated_bytes -= size;
    g_memory_stats.freed_bytes += size;
    g_memory_stats.freed_blocks += 1;
}

// Public API implementation
void* boat_memory_allocate(size_t size, boat_device_t device,
                           const char* file, int line) {
    if (size == 0) {
        return NULL;
    }

    // Calculate total size with header
    size_t total_size = sizeof(boat_memory_header_t) + size;

    // Allocate memory
    boat_memory_header_t* header = malloc(total_size);
    if (!header) {
        fprintf(stderr, "Memory allocation failed: %zu bytes at %s:%d\n",
                size, file, line);
        return NULL;
    }

    // Initialize header
    header->size = size;
    header->device = device;
    header->file = file;
    header->line = line;

    // Update statistics
    update_alloc_stats(size);

    // Return pointer to user data
    return (void*)(header + 1);
}

void* boat_memory_allocate_zero(size_t size, boat_device_t device,
                                const char* file, int line) {
    void* ptr = boat_memory_allocate(size, device, file, line);
    if (ptr) {
        memset(ptr, 0, size);
    }
    return ptr;
}

void* boat_memory_reallocate(void* ptr, size_t new_size, boat_device_t device,
                             const char* file, int line) {
    if (!ptr) {
        return boat_memory_allocate(new_size, device, file, line);
    }

    if (new_size == 0) {
        boat_memory_free(ptr);
        return NULL;
    }

    // Get original header
    boat_memory_header_t* old_header = ((boat_memory_header_t*)ptr) - 1;
    size_t old_size = old_header->size;

    // Reallocate with new size
    size_t total_size = sizeof(boat_memory_header_t) + new_size;
    boat_memory_header_t* new_header = realloc(old_header, total_size);
    if (!new_header) {
        fprintf(stderr, "Memory reallocation failed: %zu bytes at %s:%d\n",
                new_size, file, line);
        return NULL;
    }

    // Update header
    new_header->size = new_size;
    new_header->device = device;
    new_header->file = file;
    new_header->line = line;

    // Update statistics
    g_memory_stats.allocated_bytes = g_memory_stats.allocated_bytes - old_size + new_size;

    // Return new pointer
    return (void*)(new_header + 1);
}

void boat_memory_free(void* ptr) {
    if (!ptr) {
        return;
    }

    // Get header
    boat_memory_header_t* header = ((boat_memory_header_t*)ptr) - 1;

    // Update statistics
    update_free_stats(header->size);

    // Free memory
    free(header);
}

void boat_memory_free_safe(void** ptr_ptr) {
    if (ptr_ptr && *ptr_ptr) {
        boat_memory_free(*ptr_ptr);
        *ptr_ptr = NULL;
    }
}

// Memory statistics functions
boat_memory_stats_t boat_memory_get_stats() {
    return g_memory_stats;
}

void boat_memory_reset_stats() {
    g_memory_stats.allocated_bytes = 0;
    g_memory_stats.allocated_blocks = 0;
    g_memory_stats.peak_allocated_bytes = 0;
    g_memory_stats.freed_bytes = 0;
    g_memory_stats.freed_blocks = 0;
}

void boat_memory_print_stats(FILE* stream) {
    boat_memory_stats_t stats = boat_memory_get_stats();

    fprintf(stream, "=== Memory Statistics ===\n");
    fprintf(stream, "Currently allocated: %zu bytes in %zu blocks\n",
            stats.allocated_bytes, stats.allocated_blocks);
    fprintf(stream, "Peak allocation: %zu bytes\n", stats.peak_allocated_bytes);
    fprintf(stream, "Total freed: %zu bytes in %zu blocks\n",
            stats.freed_bytes, stats.freed_blocks);

    if (stats.allocated_blocks > 0) {
        fprintf(stream, "Average block size: %.2f bytes\n",
                (float)stats.allocated_bytes / stats.allocated_blocks);
    }
}

// Device-specific memory allocation (stubs for now)
void* boat_memory_allocate_device(size_t size, boat_device_t device,
                                  const char* file, int line) {
    // For now, just use CPU allocation
    // TODO: Implement CUDA allocation when CUDA support is added
    return boat_memory_allocate(size, BOAT_DEVICE_CPU, file, line);
}

void boat_memory_free_device(void* ptr, boat_device_t device) {
    // For now, just use CPU deallocation
    // TODO: Implement CUDA deallocation when CUDA support is added
    (void)device; // Unused parameter
    boat_memory_free(ptr);
}

// Alignment functions
void* boat_memory_allocate_aligned(size_t size, size_t alignment,
                                   boat_device_t device,
                                   const char* file, int line) {
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }

    size_t total_size = size + alignment + sizeof(size_t);
    void* ptr = boat_memory_allocate(total_size, device, file, line);
    if (!ptr) {
        return NULL;
    }

    // Align pointer
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t aligned_addr = (addr + alignment + sizeof(size_t)) & ~(alignment - 1);

    // Store original pointer before aligned address
    void** original_ptr = (void**)(aligned_addr - sizeof(size_t));
    *original_ptr = ptr;

    // Store size for deallocation
    size_t* size_ptr = (size_t*)(aligned_addr - 2 * sizeof(size_t));
    *size_ptr = total_size;

    return (void*)aligned_addr;
}

void boat_memory_free_aligned(void* aligned_ptr) {
    if (!aligned_ptr) {
        return;
    }

    // Get original pointer
    uintptr_t addr = (uintptr_t)aligned_ptr;
    void* original_ptr = *(void**)(addr - sizeof(size_t));

    boat_memory_free(original_ptr);
}

// Memory copy functions
void boat_memory_copy(void* dest, const void* src, size_t size,
                      boat_device_t dest_device, boat_device_t src_device) {
    if (dest_device == BOAT_DEVICE_CPU && src_device == BOAT_DEVICE_CPU) {
        memcpy(dest, src, size);
    } else {
        // TODO: Implement cross-device copying when CUDA support is added
        fprintf(stderr, "Cross-device memory copy not implemented yet\n");
    }
}

void boat_memory_set(void* dest, int value, size_t size, boat_device_t device) {
    if (device == BOAT_DEVICE_CPU) {
        memset(dest, value, size);
    } else {
        // TODO: Implement device-specific memset when CUDA support is added
        fprintf(stderr, "Device-specific memset not implemented yet\n");
    }
}

// Convenience macros for allocation with file/line tracking
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

// ============================================================================
// Memory Pool Implementation
// ============================================================================

// Memory pool block structure
typedef struct boat_memory_block_t {
    struct boat_memory_block_t* next;
    bool in_use;
    size_t size;
    uint8_t data[];  // Flexible array member for actual data
} boat_memory_block_t;

// Memory pool structure
struct boat_memory_pool_t {
    size_t block_size;
    size_t total_blocks;
    size_t free_blocks;
    boat_memory_block_t* free_list;
    boat_memory_block_t* used_list;
    void* memory_area;
};

// Arena allocator structure
struct boat_memory_arena_t {
    uint8_t* memory;
    size_t capacity;
    size_t used;
    size_t peak_used;
};

boat_memory_pool_t* boat_memory_pool_create(size_t block_size, size_t initial_blocks) {
    if (block_size == 0 || initial_blocks == 0) {
        return NULL;
    }

    // Allocate pool structure
    boat_memory_pool_t* pool = (boat_memory_pool_t*)malloc(sizeof(boat_memory_pool_t));
    if (!pool) {
        return NULL;
    }

    pool->block_size = block_size;
    pool->total_blocks = initial_blocks;
    pool->free_blocks = initial_blocks;
    pool->free_list = NULL;
    pool->used_list = NULL;

    // Allocate memory area for all blocks
    size_t total_size = initial_blocks * (sizeof(boat_memory_block_t) + block_size);
    pool->memory_area = malloc(total_size);
    if (!pool->memory_area) {
        free(pool);
        return NULL;
    }

    // Initialize blocks and build free list
    uint8_t* current = (uint8_t*)pool->memory_area;
    for (size_t i = 0; i < initial_blocks; i++) {
        boat_memory_block_t* block = (boat_memory_block_t*)current;
        block->next = pool->free_list;
        block->in_use = false;
        block->size = block_size;

        pool->free_list = block;
        current += sizeof(boat_memory_block_t) + block_size;
    }

    return pool;
}

void boat_memory_pool_free(boat_memory_pool_t* pool) {
    if (!pool) {
        return;
    }

    free(pool->memory_area);
    free(pool);
}

void* boat_memory_pool_alloc(boat_memory_pool_t* pool, size_t size) {
    if (!pool || size > pool->block_size) {
        return NULL;
    }

    if (!pool->free_list) {
        // No free blocks available
        return NULL;
    }

    // Take block from free list
    boat_memory_block_t* block = pool->free_list;
    pool->free_list = block->next;
    pool->free_blocks--;

    // Add to used list
    block->next = pool->used_list;
    pool->used_list = block;
    block->in_use = true;

    return block->data;
}

void boat_memory_pool_free_block(boat_memory_pool_t* pool, void* block_ptr) {
    if (!pool || !block_ptr) {
        return;
    }

    // Get block header from data pointer
    boat_memory_block_t* block = (boat_memory_block_t*)((uint8_t*)block_ptr - sizeof(boat_memory_block_t));

    if (!block->in_use) {
        return;  // Already free
    }

    // Remove from used list
    boat_memory_block_t** prev = &pool->used_list;
    boat_memory_block_t* curr = pool->used_list;

    while (curr) {
        if (curr == block) {
            *prev = curr->next;
            break;
        }
        prev = &curr->next;
        curr = curr->next;
    }

    // Add to free list
    block->next = pool->free_list;
    pool->free_list = block;
    pool->free_blocks++;
    block->in_use = false;
}

void boat_memory_pool_clear(boat_memory_pool_t* pool) {
    if (!pool) {
        return;
    }

    // Move all blocks back to free list
    while (pool->used_list) {
        boat_memory_block_t* block = pool->used_list;
        pool->used_list = block->next;

        block->next = pool->free_list;
        pool->free_list = block;
        block->in_use = false;
    }

    pool->free_blocks = pool->total_blocks;
}

size_t boat_memory_pool_allocated_blocks(const boat_memory_pool_t* pool) {
    return pool ? pool->total_blocks - pool->free_blocks : 0;
}

size_t boat_memory_pool_free_blocks(const boat_memory_pool_t* pool) {
    return pool ? pool->free_blocks : 0;
}

size_t boat_memory_pool_total_memory(const boat_memory_pool_t* pool) {
    return pool ? pool->total_blocks * (sizeof(boat_memory_block_t) + pool->block_size) : 0;
}

// ============================================================================
// Arena Allocator Implementation
// ============================================================================

boat_memory_arena_t* boat_memory_arena_create(size_t initial_size) {
    if (initial_size == 0) {
        initial_size = 1024 * 1024;  // Default 1MB
    }

    boat_memory_arena_t* arena = (boat_memory_arena_t*)malloc(sizeof(boat_memory_arena_t));
    if (!arena) {
        return NULL;
    }

    arena->memory = (uint8_t*)malloc(initial_size);
    if (!arena->memory) {
        free(arena);
        return NULL;
    }

    arena->capacity = initial_size;
    arena->used = 0;
    arena->peak_used = 0;

    return arena;
}

void boat_memory_arena_free(boat_memory_arena_t* arena) {
    if (!arena) {
        return;
    }

    free(arena->memory);
    free(arena);
}

void* boat_memory_arena_alloc(boat_memory_arena_t* arena, size_t size) {
    if (!arena || size == 0) {
        return NULL;
    }

    // Align to 8-byte boundary
    size = (size + 7) & ~7;

    if (arena->used + size > arena->capacity) {
        // Need to grow arena
        size_t new_capacity = arena->capacity * 2;
        while (arena->used + size > new_capacity) {
            new_capacity *= 2;
        }

        uint8_t* new_memory = (uint8_t*)realloc(arena->memory, new_capacity);
        if (!new_memory) {
            return NULL;
        }

        arena->memory = new_memory;
        arena->capacity = new_capacity;
    }

    void* ptr = arena->memory + arena->used;
    arena->used += size;

    if (arena->used > arena->peak_used) {
        arena->peak_used = arena->used;
    }

    return ptr;
}

void boat_memory_arena_reset(boat_memory_arena_t* arena) {
    if (arena) {
        arena->used = 0;
    }
}

size_t boat_memory_arena_used(const boat_memory_arena_t* arena) {
    return arena ? arena->used : 0;
}

size_t boat_memory_arena_capacity(const boat_memory_arena_t* arena) {
    return arena ? arena->capacity : 0;
}