// boat.h - Main header for Boat Deep Learning Framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_H
#define BOAT_H

// Export macros
#include "boat/export.h"

// Version information
#include "boat/version.h"

// Standard headers
#include <stdbool.h>
#include <stdarg.h>

// Core tensor operations
#include "boat/tensor.h"

// Automatic differentiation
#include "boat/autodiff.h"

// Computational graph
#include "boat/graph.h"

// Mathematical operations
#include "boat/ops.h"

// Neural network layers
#include "boat/layers.h"

// Optimization algorithms
#include "boat/optimizers.h"

// Loss functions
#include "boat/loss.h"

// Model management
#include "boat/model.h"

// Data handling
#include "boat/data.h"

// Model format loaders
#ifdef BOAT_WITH_ONNX
#include "boat/format/onnx.h"
#endif

#ifdef BOAT_WITH_PYTORCH
#include "boat/format/pytorch.h"
#endif

#ifdef BOAT_WITH_TENSORFLOW
#include "boat/format/tensorflow.h"
#endif

// Utility functions
#ifdef __cplusplus
extern "C" {
#endif

// Library initialization and cleanup
BOAT_API void boat_init();
BOAT_API void boat_cleanup();

// Error handling
typedef enum {
    BOAT_SUCCESS = 0,
    BOAT_ERROR_INVALID_ARGUMENT,
    BOAT_ERROR_OUT_OF_MEMORY,
    BOAT_ERROR_INVALID_OPERATION,
    BOAT_ERROR_DEVICE,
    BOAT_ERROR_FILE_IO,
    BOAT_ERROR_FORMAT,
    BOAT_ERROR_NOT_IMPLEMENTED,
    BOAT_ERROR_UNKNOWN
} boat_error_t;

BOAT_API const char* boat_error_string(boat_error_t error);
BOAT_API boat_error_t boat_get_last_error();
BOAT_API void boat_clear_error();

// Extended error handling
BOAT_API void boat_set_error(boat_error_t error, const char* message);
BOAT_API const char* boat_get_last_error_message();
BOAT_API void boat_set_errorf(boat_error_t error, const char* format, ...);
BOAT_API bool boat_has_error();
BOAT_API void boat_reset_error();

// Memory statistics
#include "boat/memory.h"

boat_memory_stats_t boat_get_memory_stats();
void boat_reset_memory_stats();
void boat_print_memory_stats();

// Configuration
typedef struct {
    bool use_cuda;
    bool use_openmp;
    size_t thread_pool_size;
    size_t tensor_arena_size;
    bool enable_grad_checkpointing;
    bool enable_graph_optimization;
} boat_config_t;

boat_config_t boat_get_default_config();
void boat_set_config(const boat_config_t* config);
boat_config_t boat_get_current_config();

#ifdef __cplusplus
}
#endif

#endif // BOAT_H