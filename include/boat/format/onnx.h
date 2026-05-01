// onnx.h - ONNX model format support
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_ONNX_H
#define BOAT_ONNX_H

#include "../model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Load ONNX model from file
boat_model_t* boat_onnx_load(const char* filename);

// Save model to ONNX format
bool boat_onnx_save(const boat_model_t* model, const char* filename);

// Load ONNX model from memory buffer
boat_model_t* boat_onnx_load_from_memory(const void* data, size_t size);

// Save model to memory buffer in ONNX format
bool boat_onnx_save_to_memory(const boat_model_t* model, void** data, size_t* size);

// Check if file is a valid ONNX model
bool boat_onnx_check(const char* filename);

// Get ONNX model version information
bool boat_onnx_get_version(const char* filename, int* major, int* minor, int* patch);

// -----------------------------------------------------------------------
// ONNX Runtime: executes arbitrary ONNX graphs (including residual adds)
// -----------------------------------------------------------------------

// Opaque runtime handle
typedef struct boat_onnx_runtime_t boat_onnx_runtime_t;

// Load an ONNX model into a runtime executor.
// Supports: Conv, BatchNormalization, PRelu, Add, Gemm, Flatten.
boat_onnx_runtime_t* boat_onnx_runtime_load(const char* filename);

// Free the runtime and all associated resources.
void boat_onnx_runtime_free(boat_onnx_runtime_t* rt);

// Run inference: feed input tensor (name from ONNX graph.input[0]),
// returns a new tensor (caller owns).
boat_tensor_t* boat_onnx_runtime_run(boat_onnx_runtime_t* rt, const boat_tensor_t* input);

#ifdef __cplusplus
}
#endif

#endif // BOAT_ONNX_H