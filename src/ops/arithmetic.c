// arithmetic.c - Arithmetic operations for deep learning framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <boat/simd.h>
#include "../core/openmp.h"
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef BOAT_WITH_CUDA
#include <boat/cuda_runtime.h>
#endif

// Helper functions
static size_t broadcast_index(const boat_tensor_t* tensor, size_t output_idx,
                              const int64_t* output_shape, size_t output_ndim) {
    // Convert output linear index to multi-dimensional coordinates
    // then convert to input linear index based on input shape
    // Assumes tensor is contiguous and memory layout is row-major

    size_t ndim = boat_tensor_ndim(tensor);
    const int64_t* shape = boat_tensor_shape(tensor);

    // Handle scalar tensor (ndim = 0)
    if (ndim == 0) {
        return 0;  // Always index 0 for scalar
    }

    // Compute strides for output (largest shape)
    size_t output_strides[BOAT_MAX_DIMS];
    output_strides[output_ndim - 1] = 1;
    for (int i = (int)output_ndim - 2; i >= 0; i--) {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    // Compute strides for input
    size_t input_strides[BOAT_MAX_DIMS];
    input_strides[ndim - 1] = 1;
    if (ndim >= 2) {
        for (int i = (int)ndim - 2; i >= 0; i--) {
            input_strides[i] = input_strides[i + 1] * shape[i + 1];
        }
    }

    // Convert output linear index to coordinates
    size_t remaining = output_idx;
    size_t coords[BOAT_MAX_DIMS];
    for (size_t i = 0; i < output_ndim; i++) {
        coords[i] = remaining / output_strides[i];
        remaining %= output_strides[i];
    }

    // Adjust coordinates for input (broadcasting dimensions where shape == 1)
    // Input may have fewer dimensions than output
    size_t input_idx = 0;
    for (size_t i = 0; i < ndim; i++) {
        // Align from the rightmost dimension (broadcasting rule)
        size_t output_dim_idx = output_ndim - ndim + i;
        size_t coord = coords[output_dim_idx];
        // If input dimension is 1, use 0 (broadcast)
        if (shape[i] == 1) {
            coord = 0;
        }
        input_idx += coord * input_strides[i];
    }

    return input_idx;
}

static bool validate_shapes_for_broadcasting(const boat_tensor_t* a,
                                             const boat_tensor_t* b,
                                             int64_t* out_shape,
                                             size_t* out_ndim) {
    size_t a_ndim = boat_tensor_ndim(a);
    size_t b_ndim = boat_tensor_ndim(b);
    const int64_t* a_shape = boat_tensor_shape(a);
    const int64_t* b_shape = boat_tensor_shape(b);

    // Determine output ndim (max of input ndims)
    size_t max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
    *out_ndim = max_ndim;

    // Check if shapes can be broadcast
    for (size_t i = 0; i < max_ndim; i++) {
        int64_t a_dim = (i < max_ndim - a_ndim) ? 1 : a_shape[i - (max_ndim - a_ndim)];
        int64_t b_dim = (i < max_ndim - b_ndim) ? 1 : b_shape[i - (max_ndim - b_ndim)];

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            return false;
        }

        out_shape[i] = a_dim > b_dim ? a_dim : b_dim;
    }

    return true;
}

static boat_tensor_t* create_broadcasted_output(const boat_tensor_t* a,
                                                const boat_tensor_t* b,
                                                boat_dtype_t dtype) {
    int64_t out_shape[BOAT_MAX_DIMS];
    size_t out_ndim;

    if (!validate_shapes_for_broadcasting(a, b, out_shape, &out_ndim)) {
        BOAT_DEBUG_PRINT("DEBUG create_broadcasted_output: validate_shapes_for_broadcasting failed\n");
        // Debug shape info
        size_t a_ndim = boat_tensor_ndim(a);
        size_t b_ndim = boat_tensor_ndim(b);
        const int64_t* a_shape = boat_tensor_shape(a);
        const int64_t* b_shape = boat_tensor_shape(b);
        (void)a_shape; (void)b_shape;  // used only in BOAT_DEBUG_PRINT below
        BOAT_DEBUG_PRINT("  a shape: [");
        for (size_t i = 0; i < a_ndim; i++) {
            BOAT_DEBUG_PRINT("%ld", a_shape[i]);
            if (i < a_ndim - 1) BOAT_DEBUG_PRINT(", ");
        }
        BOAT_DEBUG_PRINT("]\n");
        BOAT_DEBUG_PRINT("  b shape: [");
        for (size_t i = 0; i < b_ndim; i++) {
            BOAT_DEBUG_PRINT("%ld", b_shape[i]);
            if (i < b_ndim - 1) BOAT_DEBUG_PRINT(", ");
        }
        BOAT_DEBUG_PRINT("]\n");
        return NULL;
    }

    boat_device_t device = boat_tensor_device(a);
    boat_device_t b_device = boat_tensor_device(b);
    if (device != b_device) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Device mismatch in broadcast: a=%d, b=%d\n", device, b_device);
        BOAT_DEBUG_PRINT("DEBUG create_broadcasted_output: device mismatch: a=%d, b=%d\n", device, b_device);
        return NULL;
    }

    BOAT_DEBUG_PRINT("DEBUG create_broadcasted_output: creating tensor shape=[");
    for (size_t i = 0; i < out_ndim; i++) {
        BOAT_DEBUG_PRINT("%ld", out_shape[i]);
        if (i < out_ndim - 1) BOAT_DEBUG_PRINT(", ");
    }
    BOAT_DEBUG_PRINT("], dtype=%d, device=%d\n", dtype, device);
    return boat_tensor_create(out_shape, out_ndim, dtype, device);
}

// ========== Element-wise ops with CUDA dispatch ==========

#define ELEMWISE_CUDA_FAST_PATH(op_name, cuda_func_call) \
    do { \
        if (boat_tensor_device(a) == BOAT_DEVICE_CUDA) { \
            if (dtype != BOAT_DTYPE_FLOAT32) { \
                boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] CUDA only supports float32 in boat_%s\n", #op_name); \
                boat_tensor_free(out); \
                return NULL; \
            } \
            if (boat_tensor_ndim(a) == boat_tensor_ndim(b) && \
                boat_tensor_nelements(a) == nelements && boat_tensor_nelements(b) == nelements) { \
                cuda_func_call; \
                return out; \
            } \
            /* For broadcast shapes on CUDA, need a broadcast kernel — fall through to CPU */ \
        } \
    } while(0)

#define ELEMWISE_SCALAR_CUDA_FAST_PATH(cuda_func_call) \
    do { \
        if (boat_tensor_device(a) == BOAT_DEVICE_CUDA) { \
            if (dtype != BOAT_DTYPE_FLOAT32) { \
                boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] CUDA only supports float32 in scalar op\n"); \
                boat_tensor_free(out); \
                return NULL; \
            } \
            cuda_func_call; \
            return out; \
        } \
    } while(0)

// Define arithmetic operations (explicit for CUDA support)
BOAT_API boat_tensor_t* boat_add(const boat_tensor_t* a, const boat_tensor_t* b) {
    BOAT_DEBUG_PRINT("DEBUG boat_add: called, a=%p, b=%p\n", (void*)a, (void*)b);
    if (!a || !b) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Null input in boat_add\n");
        return NULL;
    }

    boat_dtype_t dtype = boat_tensor_dtype(a);
    if (dtype != boat_tensor_dtype(b)) { return NULL; }

    boat_tensor_t* out = create_broadcasted_output(a, b, dtype);
    if (!out) { return NULL; }

    size_t nelements = boat_tensor_nelements(out);
    void* a_data = boat_tensor_data(a);
    void* b_data = boat_tensor_data(b);
    void* out_data = boat_tensor_data(out);

#ifdef BOAT_WITH_CUDA
    ELEMWISE_CUDA_FAST_PATH(add, boat_cuda_add_f32((const float*)a_data, (const float*)b_data, (float*)out_data, nelements));
#endif

    const int64_t* out_shape = boat_tensor_shape(out);
    size_t out_ndim = boat_tensor_ndim(out);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            const float* b_ptr = (const float*)b_data;
            float* out_ptr = (float*)out_data;
            if (boat_tensor_ndim(a) == boat_tensor_ndim(b) &&
                boat_tensor_nelements(a) == nelements && boat_tensor_nelements(b) == nelements &&
                boat_tensor_is_contiguous(a) && boat_tensor_is_contiguous(b)) {
                for (size_t _i = 0; _i < nelements; _i++) out_ptr[_i] = a_ptr[_i] + b_ptr[_i];
            } else {
                for (size_t i = 0; i < nelements; i++) {
                    size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                    size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                    out_ptr[i] = a_ptr[a_idx] + b_ptr[b_idx];
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            const double* b_ptr = (const double*)b_data;
            double* out_ptr = (double*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                out_ptr[i] = a_ptr[a_idx] + b_ptr[b_idx];
            }
            break;
        }
        case BOAT_DTYPE_INT32: {
            const int32_t* a_ptr = (const int32_t*)a_data;
            const int32_t* b_ptr = (const int32_t*)b_data;
            int32_t* out_ptr = (int32_t*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                out_ptr[i] = a_ptr[a_idx] + b_ptr[b_idx];
            }
            break;
        }
        case BOAT_DTYPE_INT64: {
            const int64_t* a_ptr = (const int64_t*)a_data;
            const int64_t* b_ptr = (const int64_t*)b_data;
            int64_t* out_ptr = (int64_t*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                out_ptr[i] = a_ptr[a_idx] + b_ptr[b_idx];
            }
            break;
        }
        case BOAT_DTYPE_UINT8: {
            const uint8_t* a_ptr = (const uint8_t*)a_data;
            const uint8_t* b_ptr = (const uint8_t*)b_data;
            uint8_t* out_ptr = (uint8_t*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                out_ptr[i] = a_ptr[a_idx] + b_ptr[b_idx];
            }
            break;
        }
        case BOAT_DTYPE_INT8: {
            const int8_t* a_ptr = (const int8_t*)a_data;
            const int8_t* b_ptr = (const int8_t*)b_data;
            int8_t* out_ptr = (int8_t*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                out_ptr[i] = a_ptr[a_idx] + b_ptr[b_idx];
            }
            break;
        }
        case BOAT_DTYPE_BOOL: {
            const bool* a_ptr = (const bool*)a_data;
            const bool* b_ptr = (const bool*)b_data;
            bool* out_ptr = (bool*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                out_ptr[i] = a_ptr[a_idx] + b_ptr[b_idx];
            }
            break;
        }
        case BOAT_DTYPE_FLOAT16:
            boat_set_errorf(BOAT_ERROR_NOT_IMPLEMENTED, "[Arithmetic] float16 not supported in boat_add\n");
            boat_tensor_free(out);
            return NULL;
        case BOAT_DTYPE_BFLOAT16: {
            const uint16_t* a_ptr = (const uint16_t*)a_data;
            const uint16_t* b_ptr = (const uint16_t*)b_data;
            uint16_t* out_ptr = (uint16_t*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                float av = boat_bf16_to_f32(a_ptr[a_idx]);
                float bv = boat_bf16_to_f32(b_ptr[b_idx]);
                out_ptr[i] = boat_f32_to_bf16(av + bv);
            }
            break;
        }
        case BOAT_DTYPE_FLOAT8:
        case BOAT_DTYPE_FLOAT4:
        case BOAT_DTYPE_BITS2:
        case BOAT_DTYPE_BITS1:
        default:
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Unsupported dtype in boat_add: %d\n", dtype);
            boat_tensor_free(out);
            return NULL;
    }
    return out;
}

BOAT_API boat_tensor_t* boat_sub(const boat_tensor_t* a, const boat_tensor_t* b) {
    if (!a || !b) { return NULL; }
    boat_dtype_t dtype = boat_tensor_dtype(a);
    if (dtype != boat_tensor_dtype(b)) { return NULL; }
    boat_tensor_t* out = create_broadcasted_output(a, b, dtype);
    if (!out) { return NULL; }
    size_t nelements = boat_tensor_nelements(out);
    void* a_data = boat_tensor_data(a);
    void* b_data = boat_tensor_data(b);
    void* out_data = boat_tensor_data(out);
#ifdef BOAT_WITH_CUDA
    ELEMWISE_CUDA_FAST_PATH(sub, boat_cuda_sub_f32((const float*)a_data, (const float*)b_data, (float*)out_data, nelements));
#endif
    const int64_t* out_shape = boat_tensor_shape(out);
    size_t out_ndim = boat_tensor_ndim(out);
    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            const float* b_ptr = (const float*)b_data;
            float* out_ptr = (float*)out_data;
            if (boat_tensor_ndim(a) == boat_tensor_ndim(b) && boat_tensor_nelements(a) == nelements && boat_tensor_nelements(b) == nelements && boat_tensor_is_contiguous(a) && boat_tensor_is_contiguous(b)) {
                for (size_t _i = 0; _i < nelements; _i++) out_ptr[_i] = a_ptr[_i] - b_ptr[_i];
            } else {
                for (size_t i = 0; i < nelements; i++) {
                    size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                    size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                    out_ptr[i] = a_ptr[a_idx] - b_ptr[b_idx];
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: { const double* a_ptr = (const double*)a_data; const double* b_ptr = (const double*)b_data; double* out_ptr = (double*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] - b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_INT32: { const int32_t* a_ptr = (const int32_t*)a_data; const int32_t* b_ptr = (const int32_t*)b_data; int32_t* out_ptr = (int32_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] - b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_INT64: { const int64_t* a_ptr = (const int64_t*)a_data; const int64_t* b_ptr = (const int64_t*)b_data; int64_t* out_ptr = (int64_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] - b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_UINT8: { const uint8_t* a_ptr = (const uint8_t*)a_data; const uint8_t* b_ptr = (const uint8_t*)b_data; uint8_t* out_ptr = (uint8_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] - b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_INT8: { const int8_t* a_ptr = (const int8_t*)a_data; const int8_t* b_ptr = (const int8_t*)b_data; int8_t* out_ptr = (int8_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] - b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_BOOL: { const bool* a_ptr = (const bool*)a_data; const bool* b_ptr = (const bool*)b_data; bool* out_ptr = (bool*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] - b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_BFLOAT16: { const uint16_t* a_ptr = (const uint16_t*)a_data; const uint16_t* b_ptr = (const uint16_t*)b_data; uint16_t* out_ptr = (uint16_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = boat_f32_to_bf16(boat_bf16_to_f32(a_ptr[a_idx]) - boat_bf16_to_f32(b_ptr[b_idx])); } break; }
        default: boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Unsupported dtype in boat_sub\n"); boat_tensor_free(out); return NULL;
    }
    return out;
}

BOAT_API boat_tensor_t* boat_mul(const boat_tensor_t* a, const boat_tensor_t* b) {
    if (!a || !b) { return NULL; }
    boat_dtype_t dtype = boat_tensor_dtype(a);
    if (dtype != boat_tensor_dtype(b)) { return NULL; }
    boat_tensor_t* out = create_broadcasted_output(a, b, dtype);
    if (!out) { return NULL; }
    size_t nelements = boat_tensor_nelements(out);
    void* a_data = boat_tensor_data(a);
    void* b_data = boat_tensor_data(b);
    void* out_data = boat_tensor_data(out);
#ifdef BOAT_WITH_CUDA
    ELEMWISE_CUDA_FAST_PATH(mul, boat_cuda_mul_f32((const float*)a_data, (const float*)b_data, (float*)out_data, nelements));
#endif
    const int64_t* out_shape = boat_tensor_shape(out);
    size_t out_ndim = boat_tensor_ndim(out);
    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            const float* b_ptr = (const float*)b_data;
            float* out_ptr = (float*)out_data;
            if (boat_tensor_ndim(a) == boat_tensor_ndim(b) && boat_tensor_nelements(a) == nelements && boat_tensor_nelements(b) == nelements && boat_tensor_is_contiguous(a) && boat_tensor_is_contiguous(b)) {
                for (size_t _i = 0; _i < nelements; _i++) out_ptr[_i] = a_ptr[_i] * b_ptr[_i];
            } else {
                for (size_t i = 0; i < nelements; i++) {
                    size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                    size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                    out_ptr[i] = a_ptr[a_idx] * b_ptr[b_idx];
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: { const double* a_ptr = (const double*)a_data; const double* b_ptr = (const double*)b_data; double* out_ptr = (double*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] * b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_INT32: { const int32_t* a_ptr = (const int32_t*)a_data; const int32_t* b_ptr = (const int32_t*)b_data; int32_t* out_ptr = (int32_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] * b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_INT64: { const int64_t* a_ptr = (const int64_t*)a_data; const int64_t* b_ptr = (const int64_t*)b_data; int64_t* out_ptr = (int64_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] * b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_UINT8: { const uint8_t* a_ptr = (const uint8_t*)a_data; const uint8_t* b_ptr = (const uint8_t*)b_data; uint8_t* out_ptr = (uint8_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] * b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_INT8: { const int8_t* a_ptr = (const int8_t*)a_data; const int8_t* b_ptr = (const int8_t*)b_data; int8_t* out_ptr = (int8_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] * b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_BOOL: { const bool* a_ptr = (const bool*)a_data; const bool* b_ptr = (const bool*)b_data; bool* out_ptr = (bool*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] * b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_BFLOAT16: { const uint16_t* a_ptr = (const uint16_t*)a_data; const uint16_t* b_ptr = (const uint16_t*)b_data; uint16_t* out_ptr = (uint16_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = boat_f32_to_bf16(boat_bf16_to_f32(a_ptr[a_idx]) * boat_bf16_to_f32(b_ptr[b_idx])); } break; }
        default: boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Unsupported dtype in boat_mul\n"); boat_tensor_free(out); return NULL;
    }
    return out;
}

BOAT_API boat_tensor_t* boat_div(const boat_tensor_t* a, const boat_tensor_t* b) {
    if (!a || !b) { return NULL; }
    boat_dtype_t dtype = boat_tensor_dtype(a);
    if (dtype != boat_tensor_dtype(b)) { return NULL; }
    boat_tensor_t* out = create_broadcasted_output(a, b, dtype);
    if (!out) { return NULL; }
    size_t nelements = boat_tensor_nelements(out);
    void* a_data = boat_tensor_data(a);
    void* b_data = boat_tensor_data(b);
    void* out_data = boat_tensor_data(out);
#ifdef BOAT_WITH_CUDA
    ELEMWISE_CUDA_FAST_PATH(div, boat_cuda_div_f32((const float*)a_data, (const float*)b_data, (float*)out_data, nelements));
#endif
    const int64_t* out_shape = boat_tensor_shape(out);
    size_t out_ndim = boat_tensor_ndim(out);
    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            const float* b_ptr = (const float*)b_data;
            float* out_ptr = (float*)out_data;
            if (boat_tensor_ndim(a) == boat_tensor_ndim(b) && boat_tensor_nelements(a) == nelements && boat_tensor_nelements(b) == nelements && boat_tensor_is_contiguous(a) && boat_tensor_is_contiguous(b)) {
                for (size_t _i = 0; _i < nelements; _i++) out_ptr[_i] = a_ptr[_i] / b_ptr[_i];
            } else {
                for (size_t i = 0; i < nelements; i++) {
                    size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                    size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                    out_ptr[i] = a_ptr[a_idx] / b_ptr[b_idx];
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: { const double* a_ptr = (const double*)a_data; const double* b_ptr = (const double*)b_data; double* out_ptr = (double*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] / b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_INT32: { const int32_t* a_ptr = (const int32_t*)a_data; const int32_t* b_ptr = (const int32_t*)b_data; int32_t* out_ptr = (int32_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] / b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_INT64: { const int64_t* a_ptr = (const int64_t*)a_data; const int64_t* b_ptr = (const int64_t*)b_data; int64_t* out_ptr = (int64_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] / b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_UINT8: { const uint8_t* a_ptr = (const uint8_t*)a_data; const uint8_t* b_ptr = (const uint8_t*)b_data; uint8_t* out_ptr = (uint8_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] / b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_INT8: { const int8_t* a_ptr = (const int8_t*)a_data; const int8_t* b_ptr = (const int8_t*)b_data; int8_t* out_ptr = (int8_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] / b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_BOOL: { const bool* a_ptr = (const bool*)a_data; const bool* b_ptr = (const bool*)b_data; bool* out_ptr = (bool*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = a_ptr[a_idx] / b_ptr[b_idx]; } break; }
        case BOAT_DTYPE_BFLOAT16: { const uint16_t* a_ptr = (const uint16_t*)a_data; const uint16_t* b_ptr = (const uint16_t*)b_data; uint16_t* out_ptr = (uint16_t*)out_data; for (size_t i = 0; i < nelements; i++) { size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); out_ptr[i] = boat_f32_to_bf16(boat_bf16_to_f32(a_ptr[a_idx]) / boat_bf16_to_f32(b_ptr[b_idx])); } break; }
        default: boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Unsupported dtype in boat_div\n"); boat_tensor_free(out); return NULL;
    }
    return out;
}

// Mod operation (special handling for floating point)
BOAT_API boat_tensor_t* boat_mod(const boat_tensor_t* a, const boat_tensor_t* b) {
    if (!a || !b) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Null input in boat_mod\n");
        return NULL;
    }

    boat_dtype_t dtype = boat_tensor_dtype(a);
    if (dtype != boat_tensor_dtype(b)) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Dtype mismatch in boat_mod: %d vs %d\n", dtype, boat_tensor_dtype(b));
        return NULL;
    }

    boat_tensor_t* out = create_broadcasted_output(a, b, dtype);
    if (!out) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Broadcast failed in boat_mod\n");
        return NULL;
    }

    size_t nelements = boat_tensor_nelements(out);
    const void* a_data = boat_tensor_data(a);
    const void* b_data = boat_tensor_data(b);
    void* out_data = boat_tensor_data(out);

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(a) == BOAT_DEVICE_CUDA && dtype == BOAT_DTYPE_FLOAT32) {
        if (boat_tensor_ndim(a) == boat_tensor_ndim(b) && boat_tensor_nelements(a) == nelements && boat_tensor_nelements(b) == nelements) {
            boat_cuda_mod_f32((const float*)a_data, (const float*)b_data, (float*)out_data, nelements);
            return out;
        }
    }
#endif

    /* Get output shape for broadcasting */
    const int64_t* out_shape = boat_tensor_shape(out);
    size_t out_ndim = boat_tensor_ndim(out);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            const float* b_ptr = (const float*)b_data;
            float* out_ptr = (float*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                out_ptr[i] = fmodf(a_ptr[a_idx], b_ptr[b_idx]);
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            const double* b_ptr = (const double*)b_data;
            double* out_ptr = (double*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                out_ptr[i] = fmod(a_ptr[a_idx], b_ptr[b_idx]);
            }
            break;
        }
        case BOAT_DTYPE_INT32: {
            const int32_t* a_ptr = (const int32_t*)a_data;
            const int32_t* b_ptr = (const int32_t*)b_data;
            int32_t* out_ptr = (int32_t*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                out_ptr[i] = a_ptr[a_idx] % b_ptr[b_idx];
            }
            break;
        }
        case BOAT_DTYPE_INT64: {
            const int64_t* a_ptr = (const int64_t*)a_data;
            const int64_t* b_ptr = (const int64_t*)b_data;
            int64_t* out_ptr = (int64_t*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim);
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim);
                out_ptr[i] = a_ptr[a_idx] % b_ptr[b_idx];
            }
            break;
        }
        default:
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Unsupported dtype in boat_mod\n");
            boat_tensor_free(out);
            return NULL;
    }

    return out;
}

// In-place operations
#define DEFINE_INPLACE_OP(op_name, op) \
void boat_##op_name##_(boat_tensor_t* const a, const boat_tensor_t* b) { \
    if (!a || !b) { \
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Null input in boat_%s_\n", #op_name); \
        return; \
    } \
    \
    boat_dtype_t dtype = boat_tensor_dtype(a); \
    if (dtype != boat_tensor_dtype(b)) { \
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Dtype mismatch in boat_%s_: %d vs %d\n", #op_name, dtype, boat_tensor_dtype(b)); \
        return; \
    } \
    \
    /* TODO: Implement proper broadcasting for in-place ops */ \
    size_t a_nelements = boat_tensor_nelements(a); \
    size_t b_nelements = boat_tensor_nelements(b); \
    if (a_nelements != b_nelements) { \
        return; \
    } \
    \
    void* a_data = boat_tensor_data(a); \
    void* b_data = boat_tensor_data(b); \
    \
    switch (dtype) { \
        case BOAT_DTYPE_FLOAT32: { \
            float* a_ptr = (float*)a_data; \
            const float* b_ptr = (const float*)b_data; \
            for (size_t i = 0; i < a_nelements; i++) { \
                a_ptr[i] = a_ptr[i] op b_ptr[i]; \
            } \
            break; \
        } \
        case BOAT_DTYPE_FLOAT64: { \
            double* a_ptr = (double*)a_data; \
            const double* b_ptr = (const double*)b_data; \
            for (size_t i = 0; i < a_nelements; i++) { \
                a_ptr[i] = a_ptr[i] op b_ptr[i]; \
            } \
            break; \
        } \
        case BOAT_DTYPE_BFLOAT16: { \
            uint16_t* a_ptr = (uint16_t*)a_data; \
            const uint16_t* b_ptr = (const uint16_t*)b_data; \
            for (size_t i = 0; i < a_nelements; i++) { \
                float av = boat_bf16_to_f32(a_ptr[i]); \
                float bv = boat_bf16_to_f32(b_ptr[i]); \
                a_ptr[i] = boat_f32_to_bf16(av op bv); \
            } \
            break; \
        } \
        case BOAT_DTYPE_INT32: { \
            int32_t* a_ptr = (int32_t*)a_data; \
            const int32_t* b_ptr = (const int32_t*)b_data; \
            for (size_t i = 0; i < a_nelements; i++) { \
                a_ptr[i] = a_ptr[i] op b_ptr[i]; \
            } \
            break; \
        } \
        case BOAT_DTYPE_INT64: { \
            int64_t* a_ptr = (int64_t*)a_data; \
            const int64_t* b_ptr = (const int64_t*)b_data; \
            for (size_t i = 0; i < a_nelements; i++) { \
                a_ptr[i] = a_ptr[i] op b_ptr[i]; \
            } \
            break; \
        } \
        default: \
            break; \
    } \
}

DEFINE_INPLACE_OP(add, +)
DEFINE_INPLACE_OP(sub, -)
DEFINE_INPLACE_OP(mul, *)
DEFINE_INPLACE_OP(div, /)

// Scalar operations
#define DEFINE_SCALAR_OP(op_name, op) \
boat_tensor_t* boat_##op_name##_scalar(const boat_tensor_t* a, double scalar) { \
    if (!a) { \
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Null input in boat_%s_scalar\n", #op_name); \
        return NULL; \
    } \
    \
    \
    boat_dtype_t dtype = boat_tensor_dtype(a); \
    boat_tensor_t* out = boat_tensor_create_like(a); \
    if (!out) { \
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Failed to create output in boat_%s_scalar\n", #op_name); \
        return NULL; \
    } \
    \
    size_t nelements = boat_tensor_nelements(a); \
    void* a_data = boat_tensor_data(a); \
    void* out_data = boat_tensor_data(out); \
    \
    switch (dtype) { \
        case BOAT_DTYPE_FLOAT32: { \
            float scalar_f = (float)scalar; \
            float* a_ptr = (float*)a_data; \
            float* out_ptr = (float*)out_data; \
            for (size_t i = 0; i < nelements; i++) { \
                out_ptr[i] = a_ptr[i] op scalar_f; \
            } \
            break; \
        } \
        case BOAT_DTYPE_FLOAT64: { \
            double* a_ptr = (double*)a_data; \
            double* out_ptr = (double*)out_data; \
            for (size_t i = 0; i < nelements; i++) { \
                out_ptr[i] = a_ptr[i] op scalar; \
            } \
            break; \
        } \
        case BOAT_DTYPE_BFLOAT16: { \
            float scalar_f = (float)scalar; \
            const uint16_t* a_ptr = (const uint16_t*)a_data; \
            uint16_t* out_ptr = (uint16_t*)out_data; \
            for (size_t i = 0; i < nelements; i++) { \
                float av = boat_bf16_to_f32(a_ptr[i]); \
                out_ptr[i] = boat_f32_to_bf16(av op scalar_f); \
            } \
            break; \
        } \
        case BOAT_DTYPE_INT32: { \
            int32_t scalar_i = (int32_t)scalar; \
            int32_t* a_ptr = (int32_t*)a_data; \
            int32_t* out_ptr = (int32_t*)out_data; \
            for (size_t i = 0; i < nelements; i++) { \
                out_ptr[i] = a_ptr[i] op scalar_i; \
            } \
            break; \
        } \
        case BOAT_DTYPE_INT64: { \
            int64_t scalar_i = (int64_t)scalar; \
            int64_t* a_ptr = (int64_t*)a_data; \
            int64_t* out_ptr = (int64_t*)out_data; \
            for (size_t i = 0; i < nelements; i++) { \
                out_ptr[i] = a_ptr[i] op scalar_i; \
            } \
            break; \
        } \
        default: \
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Unsupported dtype in boat_%s_scalar\n", #op_name); \
            boat_tensor_free(out); \
            return NULL; \
    } \
    \
    return out; \
}

DEFINE_SCALAR_OP(add, +)
DEFINE_SCALAR_OP(sub, -)
DEFINE_SCALAR_OP(mul, *)
DEFINE_SCALAR_OP(div, /)

BOAT_API boat_tensor_t* boat_pow_scalar(const boat_tensor_t* a, double scalar) {
    if (!a) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Null input in boat_pow_scalar\n");
        return NULL;
    }

    boat_dtype_t dtype = boat_tensor_dtype(a);
    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    size_t nelements = boat_tensor_nelements(a);
    const void* a_data = boat_tensor_data(a);
    void* out_data = boat_tensor_data(out);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            float scalar_f = (float)scalar;
            const float* a_ptr = (const float*)a_data;
            float* out_ptr = (float*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                out_ptr[i] = powf(a_ptr[i], scalar_f);
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            double* out_ptr = (double*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                out_ptr[i] = pow(a_ptr[i], scalar);
            }
            break;
        }
        case BOAT_DTYPE_BFLOAT16: {
            float scalar_f = (float)scalar;
            const uint16_t* a_ptr = (const uint16_t*)a_data;
            uint16_t* out_ptr = (uint16_t*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                float av = boat_bf16_to_f32(a_ptr[i]);
                out_ptr[i] = boat_f32_to_bf16(powf(av, scalar_f));
            }
            break;
        }
        default:
            boat_tensor_free(out);
            return NULL;
    }

    return out;
}

// Element-wise absolute value
BOAT_API boat_tensor_t* boat_abs(const boat_tensor_t* a) {
    if (!a) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Null input in boat_abs\n");
        return NULL;
    }

    boat_dtype_t dtype = boat_tensor_dtype(a);
    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    size_t n = boat_tensor_nelements(a);
    const void* a_data = boat_tensor_const_data(a);
    void* out_data = boat_tensor_data(out);

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(a) == BOAT_DEVICE_CUDA) {
        if (dtype == BOAT_DTYPE_FLOAT32) {
            boat_cuda_abs_f32((const float*)a_data, (float*)out_data, n);
            return out;
        }
    }
#endif

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* src = (const float*)a_data;
            float* dst = (float*)out_data;
            for (size_t i = 0; i < n; i++) dst[i] = fabsf(src[i]);
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* src = (const double*)a_data;
            double* dst = (double*)out_data;
            for (size_t i = 0; i < n; i++) dst[i] = fabs(src[i]);
            break;
        }
        case BOAT_DTYPE_INT32: {
            const int32_t* src = (const int32_t*)a_data;
            int32_t* dst = (int32_t*)out_data;
            for (size_t i = 0; i < n; i++) dst[i] = src[i] < 0 ? -src[i] : src[i];
            break;
        }
        case BOAT_DTYPE_INT64: {
            const int64_t* src = (const int64_t*)a_data;
            int64_t* dst = (int64_t*)out_data;
            for (size_t i = 0; i < n; i++) dst[i] = src[i] < 0 ? -src[i] : src[i];
            break;
        }
        default:
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                            "[Arithmetic] Unsupported dtype in boat_abs: %d\n", dtype);
            boat_tensor_free(out);
            return NULL;
    }

    return out;
}

// In-place scalar operations
#define DEFINE_INPLACE_SCALAR_OP(op_name, op) \
void boat_##op_name##_scalar_(boat_tensor_t* const a, double scalar) { \
    if (!a) return; \
    \
    boat_dtype_t dtype = boat_tensor_dtype(a); \
    size_t nelements = boat_tensor_nelements(a); \
    void* a_data = boat_tensor_data(a); \
    \
    switch (dtype) { \
        case BOAT_DTYPE_FLOAT32: { \
            float scalar_f = (float)scalar; \
            float* a_ptr = (float*)a_data; \
            for (size_t i = 0; i < nelements; i++) { \
                a_ptr[i] = a_ptr[i] op scalar_f; \
            } \
            break; \
        } \
        case BOAT_DTYPE_FLOAT64: { \
            double* a_ptr = (double*)a_data; \
            for (size_t i = 0; i < nelements; i++) { \
                a_ptr[i] = a_ptr[i] op scalar; \
            } \
            break; \
        } \
        case BOAT_DTYPE_BFLOAT16: { \
            float scalar_f = (float)scalar; \
            uint16_t* a_ptr = (uint16_t*)a_data; \
            for (size_t i = 0; i < nelements; i++) { \
                float av = boat_bf16_to_f32(a_ptr[i]); \
                a_ptr[i] = boat_f32_to_bf16(av op scalar_f); \
            } \
            break; \
        } \
        case BOAT_DTYPE_INT32: { \
            int32_t scalar_i = (int32_t)scalar; \
            int32_t* a_ptr = (int32_t*)a_data; \
            for (size_t i = 0; i < nelements; i++) { \
                a_ptr[i] = a_ptr[i] op scalar_i; \
            } \
            break; \
        } \
        case BOAT_DTYPE_INT64: { \
            int64_t scalar_i = (int64_t)scalar; \
            int64_t* a_ptr = (int64_t*)a_data; \
            for (size_t i = 0; i < nelements; i++) { \
                a_ptr[i] = a_ptr[i] op scalar_i; \
            } \
            break; \
        } \
        default: \
            break; \
    } \
}

DEFINE_INPLACE_SCALAR_OP(add, +)
DEFINE_INPLACE_SCALAR_OP(sub, -)
DEFINE_INPLACE_SCALAR_OP(mul, *)
DEFINE_INPLACE_SCALAR_OP(div, /)

// Broadcasting utility
BOAT_API bool boat_can_broadcast(const boat_tensor_t* a, const boat_tensor_t* b) {
    if (!a || !b) return false;

    int64_t out_shape[BOAT_MAX_DIMS];
    size_t out_ndim;
    return validate_shapes_for_broadcasting(a, b, out_shape, &out_ndim);
}

BOAT_API boat_tensor_t* boat_broadcast_to(const boat_tensor_t* a, const int64_t* shape, size_t ndim) {
    if (!a || !shape) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] Null input in boat_broadcast_to\n");
        return NULL;
    }

    // TODO: Implement actual broadcasting (this is just shape checking for now)
    // For now, create a new tensor with the target shape and copy data
    // This is inefficient and should be replaced with strided views

    boat_dtype_t dtype = boat_tensor_dtype(a);
    boat_device_t device = boat_tensor_device(a);
    boat_tensor_t* out = boat_tensor_create(shape, ndim, dtype, device);
    if (!out) return NULL;

    // Simple copy (assumes shapes are compatible)
    size_t a_nelements = boat_tensor_nelements(a);
    size_t out_nelements = boat_tensor_nelements(out);
    if (a_nelements != out_nelements) {
        boat_tensor_free(out);
        return NULL;
    }

    const void* a_data = boat_tensor_data(a);
    void* out_data = boat_tensor_data(out);
    memcpy(out_data, a_data, boat_tensor_nbytes(a));

    return out;
}

// Reduction operations
// ========== Reduction operations (sum, mean, max, min) with axis support ==========

typedef enum {
    BOAT_REDUCE_SUM,
    BOAT_REDUCE_MEAN,
    BOAT_REDUCE_MAX,
    BOAT_REDUCE_MIN
} boat_reduce_kind_t;

// Fill `reduced[d]` = true for each dimension to reduce. dims==NULL or
// n_dims==0 means full reduction. Negative dims are Python-style (from the end).
// Returns false on an out-of-range dim.
static bool normalize_reduce_dims(size_t ndim, const int64_t* dims, size_t n_dims, bool* reduced) {
    for (size_t i = 0; i < ndim; i++) reduced[i] = false;
    if (dims == NULL || n_dims == 0) {
        for (size_t i = 0; i < ndim; i++) reduced[i] = true;
        return true;
    }
    for (size_t d = 0; d < n_dims; d++) {
        int64_t dim = dims[d];
        if (dim < 0) dim += (int64_t)ndim;
        if (dim < 0 || dim >= (int64_t)ndim) return false;
        reduced[(size_t)dim] = true;
    }
    return true;
}

// Generic axis reduction for float32/float64 tensors. Returns a new tensor.
static boat_tensor_t* reduce_axis(const boat_tensor_t* a, const int64_t* dims, size_t n_dims,
                                  bool keepdim, boat_reduce_kind_t kind) {
    if (!a) return NULL;
    size_t ndim = boat_tensor_ndim(a);
    const int64_t* shape = boat_tensor_shape(a);
    boat_dtype_t dtype = boat_tensor_dtype(a);

    if (dtype != BOAT_DTYPE_FLOAT32 && dtype != BOAT_DTYPE_FLOAT64) {
        boat_set_errorf(BOAT_ERROR_NOT_IMPLEMENTED,
                        "[Arithmetic] reduction only supports float32/float64\n");
        return NULL;
    }

    if (ndim == 0) {
        // Reducing a scalar is the identity operation.
        return boat_tensor_clone(a);
    }

    bool reduced[BOAT_MAX_DIMS];
    if (!normalize_reduce_dims(ndim, dims, n_dims, reduced)) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Arithmetic] invalid reduction dim\n");
        return NULL;
    }

    // Input row-major strides.
    size_t in_stride[BOAT_MAX_DIMS];
    in_stride[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) in_stride[i] = in_stride[i + 1] * (size_t)shape[i + 1];

    // Reduced dims (ascending order) and their sizes/strides.
    size_t red_sizes[BOAT_MAX_DIMS];
    size_t red_strides[BOAT_MAX_DIMS];
    size_t n_red = 0;
    size_t red_total = 1;
    for (size_t i = 0; i < ndim; i++) {
        if (reduced[i]) {
            red_sizes[n_red] = (size_t)shape[i];
            red_strides[n_red] = in_stride[i];
            red_total *= (size_t)shape[i];
            n_red++;
        }
    }

    // Output shape and dim mapping.
    int64_t out_shape[BOAT_MAX_DIMS];
    size_t out_to_in[BOAT_MAX_DIMS];
    size_t out_ndim = 0;
    for (size_t i = 0; i < ndim; i++) {
        if (!reduced[i]) {
            out_to_in[out_ndim] = i;
            out_shape[out_ndim++] = shape[i];
        } else if (keepdim) {
            out_to_in[out_ndim] = i;
            out_shape[out_ndim++] = 1;
        }
    }

    size_t out_stride[BOAT_MAX_DIMS];
    size_t out_nelements = 1;
    if (out_ndim > 0) {
        out_stride[out_ndim - 1] = 1;
        for (int i = (int)out_ndim - 2; i >= 0; i--) out_stride[i] = out_stride[i + 1] * (size_t)out_shape[i + 1];
        for (size_t i = 0; i < out_ndim; i++) out_nelements *= (size_t)out_shape[i];
    }

    boat_tensor_t* out = boat_tensor_create(out_shape, out_ndim, dtype, boat_tensor_device(a));
    if (!out) return NULL;

    const void* in = boat_tensor_const_data(a);
    void* out_data = boat_tensor_data(out);
    const float* f_in = (const float*)in;
    const double* d_in = (const double*)in;
    float* f_out = (float*)out_data;
    double* d_out = (double*)out_data;
    bool is_f64 = (dtype == BOAT_DTYPE_FLOAT64);

    for (size_t oi = 0; oi < out_nelements; oi++) {
        // Decode output linear index to the base input offset.
        size_t rem = oi;
        size_t base_off = 0;
        for (size_t d = 0; d < out_ndim; d++) {
            size_t coord = rem / out_stride[d];
            rem %= out_stride[d];
            base_off += coord * in_stride[out_to_in[d]];
        }

        double acc = 0.0;
        bool first = true;
        for (size_t r = 0; r < red_total; r++) {
            size_t rr = r;
            size_t red_off = 0;
            for (size_t d = 0; d < n_red; d++) {
                size_t coord = rr % red_sizes[d];
                rr /= red_sizes[d];
                red_off += coord * red_strides[d];
            }
            double v = is_f64 ? d_in[base_off + red_off] : (double)f_in[base_off + red_off];
            switch (kind) {
                case BOAT_REDUCE_SUM:
                case BOAT_REDUCE_MEAN:
                    acc += v;
                    break;
                case BOAT_REDUCE_MAX:
                    if (first || v > acc) acc = v;
                    break;
                case BOAT_REDUCE_MIN:
                    if (first || v < acc) acc = v;
                    break;
            }
            first = false;
        }
        if (kind == BOAT_REDUCE_MEAN && red_total > 0) acc /= (double)red_total;

        if (is_f64) d_out[oi] = acc;
        else f_out[oi] = (float)acc;
    }

    return out;
}

BOAT_API boat_tensor_t* boat_sum(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim) {
    return reduce_axis(a, dims, n_dims, keepdim, BOAT_REDUCE_SUM);
}

BOAT_API boat_tensor_t* boat_mean(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim) {
    return reduce_axis(a, dims, n_dims, keepdim, BOAT_REDUCE_MEAN);
}

BOAT_API boat_tensor_t* boat_max(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim) {
    return reduce_axis(a, dims, n_dims, keepdim, BOAT_REDUCE_MAX);
}

BOAT_API boat_tensor_t* boat_min(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim) {
    return reduce_axis(a, dims, n_dims, keepdim, BOAT_REDUCE_MIN);
}
