// onnxruntime.h - ONNX Runtime C inference backend
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_ONNXRUNTIME_H
#define BOAT_ONNXRUNTIME_H

#include "../tensor.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------
// Execution provider selection
// -----------------------------------------------------------------------
typedef enum {
    BOAT_ORT_CPU = 0,   // CPU execution only
    BOAT_ORT_CUDA = 1,  // CUDA GPU execution provider
    BOAT_ORT_AUTO = 2,  // Auto-select best available provider
} boat_onnxruntime_provider_t;

// -----------------------------------------------------------------------
// Opaque session handle
// -----------------------------------------------------------------------
typedef struct boat_onnxruntime_session_t boat_onnxruntime_session_t;

// -----------------------------------------------------------------------
// Session creation and destruction
// -----------------------------------------------------------------------

// Create an ONNX Runtime session from a model file.
// model_path: path to .onnx model file
// provider: execution provider selection
// Returns NULL on failure (sets boat error).
boat_onnxruntime_session_t* boat_onnxruntime_create(
    const char* model_path,
    boat_onnxruntime_provider_t provider);

// Create an ONNX Runtime session from a model buffer in memory.
// data: pointer to ONNX model bytes
// size: number of bytes
// provider: execution provider selection
// Returns NULL on failure (sets boat error).
boat_onnxruntime_session_t* boat_onnxruntime_create_from_buffer(
    const void* data, size_t size,
    boat_onnxruntime_provider_t provider);

// Destroy a session and free all associated resources.
void boat_onnxruntime_free(boat_onnxruntime_session_t* session);

// -----------------------------------------------------------------------
// Model introspection
// -----------------------------------------------------------------------

// Get the number of model inputs.
size_t boat_onnxruntime_input_count(
    const boat_onnxruntime_session_t* session);

// Get the number of model outputs.
size_t boat_onnxruntime_output_count(
    const boat_onnxruntime_session_t* session);

// Get the name of the i-th input (returns pointer to internal string,
// valid until session is freed).
const char* boat_onnxruntime_input_name(
    const boat_onnxruntime_session_t* session, size_t index);

// Get the name of the i-th output (returns pointer to internal string,
// valid until session is freed).
const char* boat_onnxruntime_output_name(
    const boat_onnxruntime_session_t* session, size_t index);

// -----------------------------------------------------------------------
// Inference
// -----------------------------------------------------------------------

// Run single-input single-output inference.
// session: the ORT session
// input: input tensor (float32, any shape matching model)
// Returns a new output tensor (caller owns via boat_tensor_unref).
// Returns NULL on failure.
boat_tensor_t* boat_onnxruntime_run(
    boat_onnxruntime_session_t* session,
    const boat_tensor_t* input);

// Run multi-input multi-output inference.
// session: the ORT session
// inputs: array of input tensors
// input_names: array of input names (must match model's expected input names)
// num_inputs: number of inputs
// num_outputs: output parameter, set to number of output tensors
// Returns array of output tensors (caller must free both the array
// and each tensor via boat_tensor_unref). Returns NULL on failure.
boat_tensor_t** boat_onnxruntime_run_multi(
    boat_onnxruntime_session_t* session,
    boat_tensor_t* const* inputs,
    const char** input_names,
    size_t num_inputs,
    size_t* num_outputs);

#ifdef __cplusplus
}
#endif

#endif // BOAT_ONNXRUNTIME_H
