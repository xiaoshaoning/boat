// tensor.h - Tensor operations for deep learning framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_TENSOR_H
#define BOAT_TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
typedef struct boat_tensor_t boat_tensor_t;

// Data type enumeration
typedef enum {
    // Standard floating point types
    BOAT_DTYPE_FLOAT64,   // 64-bit floating point (double)
    BOAT_DTYPE_FLOAT32,   // 32-bit floating point (float)
    BOAT_DTYPE_FLOAT16,   // 16-bit floating point (half precision)

    // Custom floating point types
    BOAT_DTYPE_FLOAT8,    // 8-bit floating point (custom format)
    BOAT_DTYPE_FLOAT4,    // 4-bit floating point (custom format)

    // Integer types
    BOAT_DTYPE_INT64,     // 64-bit integer
    BOAT_DTYPE_INT32,     // 32-bit integer
    BOAT_DTYPE_UINT8,     // 8-bit unsigned integer

    // Low-bit quantization types
    BOAT_DTYPE_BITS2,     // 2-bit packed values
    BOAT_DTYPE_BITS1,     // 1-bit packed values (binary)

    // Special types
    BOAT_DTYPE_BOOL,      // boolean (1 byte per element)

    // Future types (at end for backward compat with serialization)
    BOAT_DTYPE_INT8,      // 8-bit signed integer
    BOAT_DTYPE_BFLOAT16,  // 16-bit brain floating point (BF16)

    BOAT_DTYPE_COUNT      // number of data types
} boat_dtype_t;

// Device type (CPU or GPU)
#ifndef BOAT_DEVICE_T_DEFINED
#define BOAT_DEVICE_T_DEFINED
enum boat_device_enum {
    BOAT_DEVICE_CPU,      // CPU device
    BOAT_DEVICE_CUDA,     // CUDA device (future)
    BOAT_DEVICE_COUNT     // number of device types
};
typedef enum boat_device_enum boat_device_t;
#endif

// Maximum number of tensor dimensions
#define BOAT_MAX_DIMS 8

// Tensor creation and destruction
BOAT_API boat_tensor_t* boat_tensor_create(const int64_t* shape, size_t ndim, boat_dtype_t dtype, boat_device_t device);
BOAT_API boat_tensor_t* boat_tensor_create_like(const boat_tensor_t* other);
BOAT_API boat_tensor_t* boat_tensor_from_data(const int64_t* shape, size_t ndim, boat_dtype_t dtype, const void* data);
BOAT_API void boat_tensor_free(boat_tensor_t* tensor);

// Reference counting
BOAT_API void boat_tensor_ref(boat_tensor_t* tensor);
BOAT_API void boat_tensor_unref(boat_tensor_t* tensor);

// Tensor properties
BOAT_API const int64_t* boat_tensor_shape(const boat_tensor_t* tensor);
BOAT_API size_t boat_tensor_ndim(const boat_tensor_t* tensor);
BOAT_API boat_dtype_t boat_tensor_dtype(const boat_tensor_t* tensor);
BOAT_API boat_device_t boat_tensor_device(const boat_tensor_t* tensor);
BOAT_API size_t boat_tensor_nbytes(const boat_tensor_t* tensor);
BOAT_API size_t boat_tensor_nelements(const boat_tensor_t* tensor);
BOAT_API bool boat_tensor_is_contiguous(const boat_tensor_t* tensor);

// Data access
BOAT_API void* boat_tensor_data(const boat_tensor_t* tensor);
BOAT_API const void* boat_tensor_const_data(const boat_tensor_t* tensor);

// Tensor operations
BOAT_API boat_tensor_t* boat_tensor_reshape(const boat_tensor_t* tensor, const int64_t* new_shape, size_t new_ndim);
BOAT_API boat_tensor_t* boat_tensor_transpose(const boat_tensor_t* tensor, const size_t* perm, size_t nperm);
BOAT_API boat_tensor_t* boat_tensor_slice(const boat_tensor_t* tensor, const size_t* start, const size_t* end, const size_t* step);
BOAT_API boat_tensor_t* boat_tensor_concatenate(const boat_tensor_t** tensors, size_t n_tensors, size_t axis);
BOAT_API boat_tensor_t* boat_tensor_stack(const boat_tensor_t** tensors, size_t n_tensors, size_t axis);

// Memory operations
BOAT_API boat_tensor_t* boat_tensor_to_device(const boat_tensor_t* tensor, boat_device_t dev);
BOAT_API boat_tensor_t* boat_tensor_clone(const boat_tensor_t* tensor);
BOAT_API boat_tensor_t* boat_tensor_contiguous(const boat_tensor_t* tensor);

// Utility functions
BOAT_API void boat_tensor_print(const boat_tensor_t* tensor);
BOAT_API char* boat_tensor_to_string(const boat_tensor_t* tensor);
BOAT_API bool boat_tensor_equal(const boat_tensor_t* a, const boat_tensor_t* b);
BOAT_API bool boat_tensor_allclose(const boat_tensor_t* a, const boat_tensor_t* b, float rtol, float atol);

// Data type information
BOAT_API size_t boat_dtype_size(boat_dtype_t dtype);
BOAT_API const char* boat_dtype_name(boat_dtype_t dtype);

// Quantization parameter access
BOAT_API float boat_tensor_get_scale(const boat_tensor_t* tensor);
BOAT_API int32_t boat_tensor_get_zero_point(const boat_tensor_t* tensor);
BOAT_API void boat_tensor_set_quant_params(boat_tensor_t* tensor, float scale, int32_t zero_point);

// Per-channel quantization accessors
BOAT_API bool boat_tensor_is_per_channel(const boat_tensor_t* tensor);
BOAT_API size_t boat_tensor_num_channels(const boat_tensor_t* tensor);
BOAT_API const float* boat_tensor_get_scales(const boat_tensor_t* tensor);
BOAT_API const int32_t* boat_tensor_get_zero_points(const boat_tensor_t* tensor);
BOAT_API void boat_tensor_set_per_channel_quant_params(boat_tensor_t* tensor, const float* scales, const int32_t* zero_points, size_t n_channels);

// BF16 conversion utilities
static inline uint16_t boat_f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static inline float boat_bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

#ifdef __cplusplus
}
#endif

#endif // BOAT_TENSOR_H