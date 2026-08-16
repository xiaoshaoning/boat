// version.c - Version information, library init/cleanup for Boat Deep Learning Framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/version.h>
#include <stdio.h>
#include <string.h>

// Feature probe: runtime-declared capabilities so consumers linked against
// older static builds can degrade gracefully instead of assuming every
// feature exists. The names are a stable contract; add new entries when a
// capability lands.
static const char* const kBoatFeatures[] = {
    "graph-forward",   // boat_graph_forward (multi-input DAG execution)
    "forward-many",    // boat_layer_ops_t::forward_many (merge layers)
    "concat-layer",    // boat_concat_layer
    "add-layer",       // boat_add_layer
    "tanh-layer",      // boat_tanh_layer
    "sigmoid-layer",   // boat_sigmoid_layer
    "avg-pool",        // boat_pool_layer average-pooling mode
    "gru-corrected",   // standard GRU update (reset before recurrent matmul)
    "onnx-export",     // boat_onnx_save
};

BOAT_API bool boat_has_feature(const char* name) {
    if (!name) return false;
    for (size_t i = 0; i < sizeof(kBoatFeatures) / sizeof(kBoatFeatures[0]); i++) {
        if (strcmp(kBoatFeatures[i], name) == 0) return true;
    }
    return false;
}

// Library initialization
BOAT_API void boat_init(void) {
    // Currently no global state to initialize.
    // Reserved for future initialization of device backends,
    // global thread pool, logging system, etc.
}

// Library cleanup
BOAT_API void boat_cleanup(void) {
    // Currently no global state to clean up.
    // Reserved for future cleanup.
}

// Get version as string
BOAT_API const char* boat_get_version_string(void) {
    return BOAT_VERSION_STRING;
}

// Get git hash
BOAT_API const char* boat_get_git_hash(void) {
    return BOAT_GIT_HASH;
}

// Get git describe
BOAT_API const char* boat_get_git_describe(void) {
    return BOAT_GIT_DESCRIBE;
}

// Get full version string with git hash
BOAT_API const char* boat_get_version_full(void) {
    return BOAT_VERSION_FULL;
}

// Get version components
BOAT_API void boat_get_version(int* major, int* minor, int* patch) {
    if (major) *major = BOAT_VERSION_MAJOR;
    if (minor) *minor = BOAT_VERSION_MINOR;
    if (patch) *patch = BOAT_VERSION_PATCH;
}

// Get version as integer
BOAT_API unsigned int boat_get_version_int(void) {
    return BOAT_VERSION_INT;
}