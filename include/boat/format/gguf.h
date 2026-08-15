// gguf.h - GGUF model format support (llama.cpp ecosystem)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_GGUF_H
#define BOAT_GGUF_H

#include "../model.h"

#ifdef __cplusplus
extern "C" {
#endif

// GGUF magic: "GGUF" as little-endian uint32
#define GGUF_MAGIC 0x46554747u

// GGUF version supported
#define GGUF_VERSION 3

// Default tensor data alignment
#define GGUF_DEFAULT_ALIGNMENT 32

// GGUF metadata value types
typedef enum {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type_t;

// GGML tensor data types (subset relevant to GGUF)
typedef enum {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
} ggml_type_t;

// Load GGUF model from file
boat_model_t* boat_gguf_load(const char* filename);

// Check if file is a valid GGUF model
bool boat_gguf_check(const char* filename);

// Dequantize raw GGML-quantized bytes into float32 values. `ggml_type` is a
// GGML_TYPE_* value (Q4_0, Q2_K, ..., F16); `n_values` is the logical element
// count the bytes encode. Returns false on unsupported types or invalid args.
bool boat_gguf_dequantize(const uint8_t* data, size_t nbytes, int ggml_type, float* out,
                          size_t n_values);

#ifdef __cplusplus
}
#endif

#endif // BOAT_GGUF_H
