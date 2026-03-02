// arithmetic.c - Arithmetic operations for deep learning framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/ops.h>
#include <boat/memory.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifndef BOAT_DEBUG
#define BOAT_DEBUG 0
#endif

#if BOAT_DEBUG
#define BOAT_DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define BOAT_DEBUG_PRINT(...) ((void)0)
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
#if BOAT_DEBUG
        fprintf(stderr, "DEBUG create_broadcasted_output: validate_shapes_for_broadcasting failed\n");
        // Debug shape info
        size_t a_ndim = boat_tensor_ndim(a);
        size_t b_ndim = boat_tensor_ndim(b);
        const int64_t* a_shape = boat_tensor_shape(a);
        const int64_t* b_shape = boat_tensor_shape(b);
        fprintf(stderr, "  a shape: [");
        for (size_t i = 0; i < a_ndim; i++) {
            fprintf(stderr, "%ld", a_shape[i]);
            if (i < a_ndim - 1) fprintf(stderr, ", ");
        }
        fprintf(stderr, "]\n");
        fprintf(stderr, "  b shape: [");
        for (size_t i = 0; i < b_ndim; i++) {
            fprintf(stderr, "%ld", b_shape[i]);
            if (i < b_ndim - 1) fprintf(stderr, ", ");
        }
        fprintf(stderr, "]\n");
#endif
        return NULL;
    }

    boat_device_t device = boat_tensor_device(a);
    boat_device_t b_device = boat_tensor_device(b);
    if (device != b_device) {
#if BOAT_DEBUG
        fprintf(stderr, "DEBUG create_broadcasted_output: device mismatch: a=%d, b=%d\n", device, b_device);
#endif
        return NULL;
    }

#if BOAT_DEBUG
    fprintf(stderr, "DEBUG create_broadcasted_output: creating tensor shape=[");
    for (size_t i = 0; i < out_ndim; i++) {
        fprintf(stderr, "%ld", out_shape[i]);
        if (i < out_ndim - 1) fprintf(stderr, ", ");
    }
    fprintf(stderr, "], dtype=%d, device=%d\n", dtype, device);
#endif
    return boat_tensor_create(out_shape, out_ndim, dtype, device);
}

// Generic element-wise operation macro with broadcasting support
#define DEFINE_ELEMENTWISE_OP(op_name, op) \
boat_tensor_t* boat_##op_name(const boat_tensor_t* a, const boat_tensor_t* b) { \
    BOAT_DEBUG_PRINT("DEBUG boat_%s: called, a=%p, b=%p\n", #op_name, (void*)a, (void*)b); \
    if (!a || !b) { \
        BOAT_DEBUG_PRINT("DEBUG boat_%s: null input\n", #op_name); \
        return NULL; \
    } \
    \
    boat_dtype_t dtype = boat_tensor_dtype(a); \
    if (dtype != boat_tensor_dtype(b)) { \
        BOAT_DEBUG_PRINT("DEBUG boat_%s: dtype mismatch: a=%d, b=%d\n", #op_name, dtype, boat_tensor_dtype(b)); \
        /* TODO: Type promotion */ \
        return NULL; \
    } \
    \
    boat_tensor_t* out = create_broadcasted_output(a, b, dtype); \
    if (!out) { \
        BOAT_DEBUG_PRINT("DEBUG boat_%s: create_broadcasted_output failed\n", #op_name); \
        return NULL; \
    } \
    \
    size_t nelements = boat_tensor_nelements(out); \
    void* a_data = boat_tensor_data(a); \
    void* b_data = boat_tensor_data(b); \
    void* out_data = boat_tensor_data(out); \
    \
    /* Get output shape for broadcasting */ \
    const int64_t* out_shape = boat_tensor_shape(out); \
    size_t out_ndim = boat_tensor_ndim(out); \
    \
    BOAT_DEBUG_PRINT("DEBUG boat_%s: computing %zu elements, shape=[", #op_name, nelements); \
    for (size_t i = 0; i < out_ndim; i++) { \
        BOAT_DEBUG_PRINT("%ld", out_shape[i]); \
        if (i < out_ndim - 1) BOAT_DEBUG_PRINT(", "); \
    } \
    BOAT_DEBUG_PRINT("], dtype=%d\n", dtype); \
    \
    switch (dtype) { \
        case BOAT_DTYPE_FLOAT32: { \
            const float* a_ptr = (const float*)a_data; \
            const float* b_ptr = (const float*)b_data; \
            float* out_ptr = (float*)out_data; \
            for (size_t i = 0; i < nelements; i++) { \
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); \
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); \
                out_ptr[i] = a_ptr[a_idx] op b_ptr[b_idx]; \
            } \
            break; \
        } \
        case BOAT_DTYPE_FLOAT64: { \
            const double* a_ptr = (const double*)a_data; \
            const double* b_ptr = (const double*)b_data; \
            double* out_ptr = (double*)out_data; \
            for (size_t i = 0; i < nelements; i++) { \
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); \
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); \
                out_ptr[i] = a_ptr[a_idx] op b_ptr[b_idx]; \
            } \
            break; \
        } \
        case BOAT_DTYPE_INT32: { \
            const int32_t* a_ptr = (const int32_t*)a_data; \
            const int32_t* b_ptr = (const int32_t*)b_data; \
            int32_t* out_ptr = (int32_t*)out_data; \
            for (size_t i = 0; i < nelements; i++) { \
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); \
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); \
                out_ptr[i] = a_ptr[a_idx] op b_ptr[b_idx]; \
            } \
            break; \
        } \
        case BOAT_DTYPE_INT64: { \
            const int64_t* a_ptr = (const int64_t*)a_data; \
            const int64_t* b_ptr = (const int64_t*)b_data; \
            int64_t* out_ptr = (int64_t*)out_data; \
            for (size_t i = 0; i < nelements; i++) { \
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); \
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); \
                out_ptr[i] = a_ptr[a_idx] op b_ptr[b_idx]; \
            } \
            break; \
        } \
        case BOAT_DTYPE_UINT8: { \
            const uint8_t* a_ptr = (const uint8_t*)a_data; \
            const uint8_t* b_ptr = (const uint8_t*)b_data; \
            uint8_t* out_ptr = (uint8_t*)out_data; \
            for (size_t i = 0; i < nelements; i++) { \
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); \
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); \
                out_ptr[i] = a_ptr[a_idx] op b_ptr[b_idx]; \
            } \
            break; \
        } \
        case BOAT_DTYPE_BOOL: { \
            const bool* a_ptr = (const bool*)a_data; \
            const bool* b_ptr = (const bool*)b_data; \
            bool* out_ptr = (bool*)out_data; \
            for (size_t i = 0; i < nelements; i++) { \
                size_t a_idx = broadcast_index(a, i, out_shape, out_ndim); \
                size_t b_idx = broadcast_index(b, i, out_shape, out_ndim); \
                out_ptr[i] = a_ptr[a_idx] op b_ptr[b_idx]; \
            } \
            break; \
        } \
        case BOAT_DTYPE_FLOAT16: { \
            /* TODO: Implement half-precision floating point */ \
            BOAT_DEBUG_PRINT("DEBUG boat_%s: float16 not supported\n", #op_name); \
            boat_tensor_free(out); \
            return NULL; \
        } \
        case BOAT_DTYPE_FLOAT8:  /* Fall through */ \
        case BOAT_DTYPE_FLOAT4:  /* Fall through */ \
        case BOAT_DTYPE_BITS2:   /* Fall through */ \
        case BOAT_DTYPE_BITS1:   /* Fall through */ \
        default: \
            BOAT_DEBUG_PRINT("DEBUG boat_%s: unsupported dtype=%d\n", #op_name, dtype); \
            boat_tensor_free(out); \
            return NULL; \
    } \
    \
    BOAT_DEBUG_PRINT("DEBUG boat_%s: success, out=%p\n", #op_name, (void*)out); \
    return out; \
}

// Define arithmetic operations
DEFINE_ELEMENTWISE_OP(add, +)
DEFINE_ELEMENTWISE_OP(sub, -)
DEFINE_ELEMENTWISE_OP(mul, *)
DEFINE_ELEMENTWISE_OP(div, /)

// Mod operation (special handling for floating point)
boat_tensor_t* boat_mod(const boat_tensor_t* a, const boat_tensor_t* b) {
    if (!a || !b) return NULL;

    boat_dtype_t dtype = boat_tensor_dtype(a);
    if (dtype != boat_tensor_dtype(b)) {
        return NULL;
    }

    boat_tensor_t* out = create_broadcasted_output(a, b, dtype);
    if (!out) return NULL;

    size_t nelements = boat_tensor_nelements(out);
    void* a_data = boat_tensor_data(a);
    void* b_data = boat_tensor_data(b);
    void* out_data = boat_tensor_data(out);

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
            boat_tensor_free(out);
            return NULL;
    }

    return out;
}

// In-place operations
#define DEFINE_INPLACE_OP(op_name, op) \
void boat_##op_name##_(boat_tensor_t* const a, const boat_tensor_t* b) { \
    if (!a || !b) return; \
    \
    boat_dtype_t dtype = boat_tensor_dtype(a); \
    if (dtype != boat_tensor_dtype(b)) { \
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
    if (!a) return NULL; \
    \
    boat_dtype_t dtype = boat_tensor_dtype(a); \
    boat_tensor_t* out = boat_tensor_create_like(a); \
    if (!out) return NULL; \
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

boat_tensor_t* boat_pow_scalar(const boat_tensor_t* a, double scalar) {
    if (!a) return NULL;

    boat_dtype_t dtype = boat_tensor_dtype(a);
    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    size_t nelements = boat_tensor_nelements(a);
    void* a_data = boat_tensor_data(a);
    void* out_data = boat_tensor_data(out);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            float scalar_f = (float)scalar;
            float* a_ptr = (float*)a_data;
            float* out_ptr = (float*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                out_ptr[i] = powf(a_ptr[i], scalar_f);
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            double* a_ptr = (double*)a_data;
            double* out_ptr = (double*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                out_ptr[i] = pow(a_ptr[i], scalar);
            }
            break;
        }
        default:
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
bool boat_can_broadcast(const boat_tensor_t* a, const boat_tensor_t* b) {
    if (!a || !b) return false;

    int64_t out_shape[BOAT_MAX_DIMS];
    size_t out_ndim;
    return validate_shapes_for_broadcasting(a, b, out_shape, &out_ndim);
}

boat_tensor_t* boat_broadcast_to(const boat_tensor_t* a, const int64_t* shape, size_t ndim) {
    if (!a || !shape) return NULL;

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
boat_tensor_t* boat_sum(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim) {
    if (!a) return NULL;

    // For now, implement only full reduction (dims == NULL, n_dims == 0)
    if (dims != NULL || n_dims != 0) {
        // TODO: Implement reduction along specific dimensions
        return NULL;
    }

    boat_dtype_t dtype = boat_tensor_dtype(a);
    size_t nelements = boat_tensor_nelements(a);
    void* data = boat_tensor_data(a);

    // Create output tensor: scalar if keepdim == false, otherwise shape with ones?
    // For simplicity, always return a scalar tensor.
    const int64_t out_shape[] = {1};
    size_t out_ndim = 1;
    if (keepdim) {
        // If keepdim is true, output shape should have same ndim with ones
        // For now, just return scalar as well
    }

    boat_tensor_t* out = boat_tensor_create(out_shape, out_ndim, dtype, boat_tensor_device(a));
    if (!out) return NULL;

    void* out_data = boat_tensor_data(out);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            float* ptr = (float*)data;
            float sum = 0.0f;
            for (size_t i = 0; i < nelements; i++) {
                sum += ptr[i];
            }
            *((float*)out_data) = sum;
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            double* ptr = (double*)data;
            double sum = 0.0;
            for (size_t i = 0; i < nelements; i++) {
                sum += ptr[i];
            }
            *((double*)out_data) = sum;
            break;
        }
        case BOAT_DTYPE_INT32: {
            int32_t* ptr = (int32_t*)data;
            int32_t sum = 0;
            for (size_t i = 0; i < nelements; i++) {
                sum += ptr[i];
            }
            *((int32_t*)out_data) = sum;
            break;
        }
        case BOAT_DTYPE_INT64: {
            int64_t* ptr = (int64_t*)data;
            int64_t sum = 0;
            for (size_t i = 0; i < nelements; i++) {
                sum += ptr[i];
            }
            *((int64_t*)out_data) = sum;
            break;
        }
        case BOAT_DTYPE_UINT8: {
            uint8_t* ptr = (uint8_t*)data;
            uint8_t sum = 0;
            for (size_t i = 0; i < nelements; i++) {
                sum += ptr[i];
            }
            *((uint8_t*)out_data) = sum;
            break;
        }
        default:
            // Unsupported dtype
            boat_tensor_free(out);
            return NULL;
    }

    return out;
}