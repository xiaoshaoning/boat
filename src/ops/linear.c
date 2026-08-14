// linear.c - Linear algebra operations for deep learning framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/ops.h>
#include <boat/memory.h>
#include <boat/sgemm.h>
#include <boat/simd.h>
#include "../core/openmp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef BOAT_WITH_CUDA
#include <boat/cuda_runtime.h>
#endif

// Matrix multiplication with batch support
// Matrix multiplication with batch support and batch-dim broadcasting.
// The last two dims of each tensor are the matrix dims; the leading dims are
// batch dims that broadcast right-aligned (e.g. [B,H,T,D] @ [D,K] -> [B,H,T,K]).
BOAT_API boat_tensor_t* boat_matmul(const boat_tensor_t* a, const boat_tensor_t* b) {
    if (!a || !b) {
        return NULL;
    }

    size_t a_ndim = boat_tensor_ndim(a);
    size_t b_ndim = boat_tensor_ndim(b);
    const int64_t* a_shape = boat_tensor_shape(a);
    const int64_t* b_shape = boat_tensor_shape(b);

    if (a_ndim < 2 || a_ndim > 4 || b_ndim < 2 || b_ndim > 4) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[Linear] matmul expects 2D..4D tensors, got a_ndim=%zu b_ndim=%zu\n",
                        a_ndim, b_ndim);
        return NULL;
    }

    // Matrix dimensions (last two dims of each tensor).
    int64_t m = a_shape[a_ndim - 2];
    int64_t k_a = a_shape[a_ndim - 1];
    int64_t k_b = b_shape[b_ndim - 2];
    int64_t n = b_shape[b_ndim - 1];
    if (k_a != k_b) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[Linear] matmul inner dim mismatch: %lld != %lld\n",
                        (long long)k_a, (long long)k_b);
        return NULL;
    }
    int64_t k = k_a;

    // Batch dimensions (all leading dims).
    size_t a_bd = a_ndim - 2;
    size_t b_bd = b_ndim - 2;
    size_t out_bd = a_bd > b_bd ? a_bd : b_bd;

    // Broadcast the batch shapes (right-aligned).
    int64_t out_shape[4];
    size_t out_batch_nelems = 1;
    for (size_t i = 0; i < out_bd; i++) {
        int64_t a_dim = (i >= out_bd - a_bd) ? a_shape[i - (out_bd - a_bd)] : 1;
        int64_t b_dim = (i >= out_bd - b_bd) ? b_shape[i - (out_bd - b_bd)] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                            "[Linear] matmul batch broadcast mismatch at dim %zu: %lld vs %lld\n",
                            i, (long long)a_dim, (long long)b_dim);
            return NULL;
        }
        out_shape[i] = a_dim > b_dim ? a_dim : b_dim;
        out_batch_nelems *= (size_t)out_shape[i];
    }
    out_shape[out_bd] = m;
    out_shape[out_bd + 1] = n;
    size_t out_ndim = out_bd + 2;

    boat_dtype_t dtype = boat_tensor_dtype(a);
    if (dtype != boat_tensor_dtype(b)) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[Linear] matmul dtype mismatch: %d != %d\n",
                        (int)dtype, (int)boat_tensor_dtype(b));
        return NULL;
    }

    boat_tensor_t* out = boat_tensor_create(out_shape, out_ndim, dtype, boat_tensor_device(a));
    if (!out) return NULL;

    const void* a_data = boat_tensor_data(a);
    const void* b_data = boat_tensor_data(b);
    void* out_data = boat_tensor_data(out);

    // Batch strides in matrix-block units (m*k for a, k*n for b, m*n for out).
    size_t a_bs[4] = {0}, b_bs[4] = {0}, out_bs[4] = {0};
    if (a_bd > 0) {
        a_bs[a_bd - 1] = 1;
        for (int i = (int)a_bd - 2; i >= 0; i--) a_bs[i] = a_bs[i + 1] * (size_t)a_shape[i + 1];
    }
    if (b_bd > 0) {
        b_bs[b_bd - 1] = 1;
        for (int i = (int)b_bd - 2; i >= 0; i--) b_bs[i] = b_bs[i + 1] * (size_t)b_shape[i + 1];
    }
    if (out_bd > 0) {
        out_bs[out_bd - 1] = 1;
        for (int i = (int)out_bd - 2; i >= 0; i--) out_bs[i] = out_bs[i + 1] * (size_t)out_shape[i + 1];
    }

    // Map an output batch linear index to the a/b batch offsets (in matrix blocks).
#define MATMUL_BATCH_OFFSETS(ob, a_off, b_off)                                              \
    do {                                                                                    \
        size_t _rem = (ob);                                                                 \
        (a_off) = 0;                                                                        \
        (b_off) = 0;                                                                        \
        for (size_t _i = 0; _i < out_bd; _i++) {                                            \
            size_t _coord = _rem / out_bs[_i];                                              \
            _rem %= out_bs[_i];                                                             \
            if (_i >= out_bd - a_bd && a_shape[_i - (out_bd - a_bd)] != 1) {                \
                (a_off) += _coord * a_bs[_i - (out_bd - a_bd)];                             \
            }                                                                               \
            if (_i >= out_bd - b_bd && b_shape[_i - (out_bd - b_bd)] != 1) {                \
                (b_off) += _coord * b_bs[_i - (out_bd - b_bd)];                             \
            }                                                                               \
        }                                                                                   \
    } while (0)

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            const float* b_ptr = (const float*)b_data;
            float* out_ptr = (float*)out_data;

            size_t out_elements = boat_tensor_nelements(out);
            boat_memory_set(out_ptr, 0, out_elements * sizeof(float), boat_tensor_device(a));

#ifdef BOAT_WITH_CUDA
            if (boat_tensor_device(a) == BOAT_DEVICE_CUDA && a_bd == b_bd) {
                if (out_batch_nelems == 1) {
                    boat_cuda_matmul_f32_cublas(a_ptr, b_ptr, out_ptr, (size_t)m, (size_t)n, (size_t)k);
                } else {
                    boat_cuda_matmul_f32_strided_batched(
                        a_ptr, b_ptr, out_ptr, (size_t)m, (size_t)n, (size_t)k,
                        (size_t)out_batch_nelems,
                        (size_t)(m * k), (size_t)(k * n), (size_t)(m * n));
                }
            } else
#endif
            {
                for (size_t ob = 0; ob < out_batch_nelems; ob++) {
                    size_t a_off, b_off;
                    MATMUL_BATCH_OFFSETS(ob, a_off, b_off);
                    boat_sgemm(m, n, k,
                               a_ptr + a_off * (size_t)(m * k),
                               b_ptr + b_off * (size_t)(k * n),
                               out_ptr + ob * (size_t)(m * n));
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            const double* b_ptr = (const double*)b_data;
            double* out_ptr = (double*)out_data;

            size_t out_elements = boat_tensor_nelements(out);
            boat_memory_set(out_ptr, 0, out_elements * sizeof(double), boat_tensor_device(a));

            for (size_t ob = 0; ob < out_batch_nelems; ob++) {
                size_t a_off, b_off;
                MATMUL_BATCH_OFFSETS(ob, a_off, b_off);
                const double* a_batch_ptr = a_ptr + a_off * (size_t)(m * k);
                const double* b_batch_ptr = b_ptr + b_off * (size_t)(k * n);
                double* out_batch_ptr = out_ptr + ob * (size_t)(m * n);

                for (int64_t i = 0; i < m; i++) {
                    for (int64_t j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int64_t l = 0; l < k; l++) {
                            sum += a_batch_ptr[i * k + l] * b_batch_ptr[l * n + j];
                        }
                        out_batch_ptr[i * n + j] = sum;
                    }
                }
            }
            break;
        }
        default:
            boat_tensor_unref(out);
            return NULL;
    }

#undef MATMUL_BATCH_OFFSETS
    return out;
}

// Dot product for 1D tensors
BOAT_API boat_tensor_t* boat_dot(const boat_tensor_t* a, const boat_tensor_t* b) {
    if (!a || !b) return NULL;

    size_t a_ndim = boat_tensor_ndim(a);
    size_t b_ndim = boat_tensor_ndim(b);

    // Support for 1D vectors only for now
    if (a_ndim != 1 || b_ndim != 1) {
        return NULL;
    }

    const int64_t* a_shape = boat_tensor_shape(a);
    const int64_t* b_shape = boat_tensor_shape(b);

    if (a_shape[0] != b_shape[0]) {
        return NULL;
    }

    int64_t n = a_shape[0];
    boat_dtype_t dtype = boat_tensor_dtype(a);
    if (dtype != boat_tensor_dtype(b)) {
        return NULL;
    }

    // Output is a scalar (0D tensor)
    const int64_t out_shape[] = {1};
    boat_tensor_t* out = boat_tensor_create(out_shape, 1, dtype, boat_tensor_device(a));
    if (!out) return NULL;

    const void* a_data = boat_tensor_data(a);
    const void* b_data = boat_tensor_data(b);
    void* out_data = boat_tensor_data(out);

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(a) == BOAT_DEVICE_CUDA && dtype == BOAT_DTYPE_FLOAT32) {
        float result = boat_cuda_dot_f32((const float*)a_data, (const float*)b_data, (int64_t)n);
        *((float*)out_data) = result;
        return out;
    }
#endif

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            const float* b_ptr = (const float*)b_data;
            float* out_ptr = (float*)out_data;

            // Use SIMD-accelerated dot product for large vectors
            float sum = 0.0f;
            if (n >= 16) {
                // Compute elementwise product first, then sum
                float* prod = (float*)malloc((size_t)n * sizeof(float));
                if (prod) {
                    boat_simd_mul_f32(a_ptr, b_ptr, prod, (size_t)n);
                    sum = boat_simd_sum_reduce_f32(prod, (size_t)n);
                    free(prod);
                } else {
                    for (int64_t i = 0; i < n; i++) sum += a_ptr[i] * b_ptr[i];
                }
            } else {
                for (int64_t i = 0; i < n; i++) sum += a_ptr[i] * b_ptr[i];
            }
            out_ptr[0] = sum;
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            const double* b_ptr = (const double*)b_data;
            double* out_ptr = (double*)out_data;
            double sum = 0.0;
            for (int64_t i = 0; i < n; i++) {
                sum += a_ptr[i] * b_ptr[i];
            }
            out_ptr[0] = sum;
            break;
        }
        default:
            boat_tensor_unref(out);
            return NULL;
    }

    return out;
}

// Transpose operation
BOAT_API boat_tensor_t* boat_transpose(const boat_tensor_t* a, int dim0, int dim1) {
    if (!a) return NULL;

    size_t ndim = boat_tensor_ndim(a);
    if ((size_t)dim0 >= ndim || (size_t)dim1 >= ndim) {
        return NULL;
    }


    // Create output tensor with swapped dimensions
    const int64_t* shape = boat_tensor_shape(a);
    int64_t* out_shape = boat_malloc(sizeof(int64_t) * ndim, BOAT_DEVICE_CPU);
    if (!out_shape) return NULL;

    for (size_t i = 0; i < ndim; i++) {
        out_shape[i] = shape[i];
    }

    // Swap dimensions
    out_shape[dim0] = shape[dim1];
    out_shape[dim1] = shape[dim0];

    boat_tensor_t* out = boat_tensor_create(out_shape, ndim, boat_tensor_dtype(a), boat_tensor_device(a));
    boat_free(out_shape);

    if (!out) return NULL;

    // Get data pointers
    const void* in_data = boat_tensor_const_data(a);
    void* out_data = boat_tensor_data(out);

    // Get output shape from the created tensor
    const int64_t* out_shape_ptr = boat_tensor_shape(out);

    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(a) == BOAT_DEVICE_CUDA && boat_tensor_dtype(a) == BOAT_DTYPE_FLOAT32) {
        if (ndim == 2) {
            // Fast 2D tiled transpose
            boat_cuda_transpose_f32((const float*)in_data, (float*)out_data,
                                     shape[0], shape[1]);
        } else {
            // N-D transpose: compute strides and dispatch
            size_t* in_stride = (size_t*)malloc(ndim * sizeof(size_t));
            size_t* out_stride = (size_t*)malloc(ndim * sizeof(size_t));
            int64_t* dev_shape = (int64_t*)malloc(ndim * sizeof(int64_t));
            if (in_stride && out_stride && dev_shape) {
                in_stride[ndim-1] = 1;
                for (int i = (int)ndim-2; i >= 0; i--)
                    in_stride[i] = in_stride[i+1] * (size_t)shape[i+1];
                out_stride[ndim-1] = 1;
                for (int i = (int)ndim-2; i >= 0; i--)
                    out_stride[i] = out_stride[i+1] * (size_t)out_shape_ptr[i+1];
                for (size_t i = 0; i < ndim; i++) dev_shape[i] = shape[i];

                boat_cuda_transpose_nd_f32((const float*)in_data, (float*)out_data,
                                            (int64_t)total_elements,
                                            dev_shape, in_stride, out_stride,
                                            (int64_t)ndim, dim0, dim1);
                free(in_stride); free(out_stride); free(dev_shape);
            }
        }
        return out;
    }
#endif

    // Handle different data types
    switch (boat_tensor_dtype(a)) {
        case BOAT_DTYPE_FLOAT32: {
            const float* in_ptr = (const float*)in_data;
            float* out_ptr = (float*)out_data;

            // Fast path: transposing the trailing two dims is a batch of 2D
            // matrix transposes -- use the tiled SIMD kernel.
            if ((dim0 == (int)ndim - 2 && dim1 == (int)ndim - 1) ||
                (dim0 == (int)ndim - 1 && dim1 == (int)ndim - 2)) {
                size_t rows = (size_t)shape[dim0];
                size_t cols = (size_t)shape[dim1];
                size_t n_mat = (rows && cols) ? total_elements / (rows * cols) : 0;
                for (size_t m = 0; m < n_mat; m++) {
                    boat_simd_transpose2d_f32(in_ptr + m * rows * cols,
                                              out_ptr + m * rows * cols,
                                              rows, cols);
                }
                return out;
            }

            // Compute strides for input and output (dynamic allocation for MSVC compatibility)
            size_t* in_stride = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
            size_t* out_stride = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
            if (!in_stride || !out_stride) {
                if (in_stride) boat_free(in_stride);
                if (out_stride) boat_free(out_stride);
                boat_tensor_unref(out);
                return NULL;
            }

            // Input strides (row-major)
            in_stride[ndim-1] = 1;
            for (int i = ndim-2; i >= 0; i--) {
                in_stride[i] = in_stride[i+1] * shape[i+1];
            }

            // Output strides (row-major with swapped shape)
            out_stride[ndim-1] = 1;
            for (int i = ndim-2; i >= 0; i--) {
                out_stride[i] = out_stride[i+1] * out_shape_ptr[i+1];
            }

            // Pre-allocate coordinate buffer (avoids per-element malloc)
            size_t* coords = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
            if (!coords) {
                boat_free(in_stride);
                boat_free(out_stride);
                boat_tensor_unref(out);
                return NULL;
            }

            // Transpose by iterating through all elements
            for (size_t idx = 0; idx < total_elements; idx++) {
                // Compute coordinates in input tensor
                size_t temp = idx;
                for (int i = ndim-1; i >= 0; i--) {
                    coords[i] = temp % (size_t)shape[i];
                    temp /= (size_t)shape[i];
                }

                // Swap the two dimensions
                size_t temp_coord = coords[dim0];
                coords[dim0] = coords[dim1];
                coords[dim1] = temp_coord;

                // Compute output index
                size_t out_idx = 0;
                for (size_t i = 0; i < ndim; i++) {
                    out_idx += coords[i] * out_stride[i];
                }

                out_ptr[out_idx] = in_ptr[idx];
            }
            boat_free(coords);
            boat_free(in_stride);
            boat_free(out_stride);
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* in_ptr = (const double*)in_data;
            double* out_ptr = (double*)out_data;

            // Compute strides for input and output (dynamic allocation for MSVC compatibility)
            size_t* in_stride = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
            size_t* out_stride = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
            if (!in_stride || !out_stride) {
                if (in_stride) boat_free(in_stride);
                if (out_stride) boat_free(out_stride);
                boat_tensor_unref(out);
                return NULL;
            }

            // Input strides (row-major)
            in_stride[ndim-1] = 1;
            for (int i = ndim-2; i >= 0; i--) {
                in_stride[i] = in_stride[i+1] * shape[i+1];
            }

            // Output strides (row-major with swapped shape)
            out_stride[ndim-1] = 1;
            for (int i = ndim-2; i >= 0; i--) {
                out_stride[i] = out_stride[i+1] * out_shape_ptr[i+1];
            }

            // Pre-allocate coordinate buffer
            size_t* coords = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
            if (!coords) {
                boat_free(in_stride);
                boat_free(out_stride);
                boat_tensor_unref(out);
                return NULL;
            }

            // Transpose by iterating through all elements
            for (size_t idx = 0; idx < total_elements; idx++) {
                // Compute coordinates in input tensor
                size_t temp = idx;
                for (int i = ndim-1; i >= 0; i--) {
                    coords[i] = temp % (size_t)shape[i];
                    temp /= (size_t)shape[i];
                }

                // Swap the two dimensions
                size_t temp_coord = coords[dim0];
                coords[dim0] = coords[dim1];
                coords[dim1] = temp_coord;

                // Compute output index
                size_t out_idx = 0;
                for (size_t i = 0; i < ndim; i++) {
                    out_idx += coords[i] * out_stride[i];
                }

                out_ptr[out_idx] = in_ptr[idx];
            }
            boat_free(coords);
            boat_free(in_stride);
            boat_free(out_stride);
            break;
        }
        default: {
            // Generic byte-wise transpose for any remaining dtype (int8/uint8,
            // int32, bf16, ...). Reuses the same row-major stride logic as the
            // float paths so the data is actually transposed, not flat-copied.
            size_t elem_size = boat_dtype_size(boat_tensor_dtype(a));
            const uint8_t* in_ptr = (const uint8_t*)in_data;
            uint8_t* out_ptr = (uint8_t*)out_data;

            size_t* in_stride = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
            size_t* out_stride = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
            if (!in_stride || !out_stride) {
                if (in_stride) boat_free(in_stride);
                if (out_stride) boat_free(out_stride);
                boat_tensor_unref(out);
                return NULL;
            }
            in_stride[ndim - 1] = 1;
            for (int i = ndim - 2; i >= 0; i--) {
                in_stride[i] = in_stride[i + 1] * (size_t)shape[i + 1];
            }
            out_stride[ndim - 1] = 1;
            for (int i = ndim - 2; i >= 0; i--) {
                out_stride[i] = out_stride[i + 1] * (size_t)out_shape_ptr[i + 1];
            }

            size_t* coords = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
            if (!coords) {
                boat_free(in_stride);
                boat_free(out_stride);
                boat_tensor_unref(out);
                return NULL;
            }

            for (size_t idx = 0; idx < total_elements; idx++) {
                size_t temp = idx;
                for (int i = ndim - 1; i >= 0; i--) {
                    coords[i] = temp % (size_t)shape[i];
                    temp /= (size_t)shape[i];
                }
                size_t tmp = coords[dim0];
                coords[dim0] = coords[dim1];
                coords[dim1] = tmp;

                size_t out_idx = 0;
                for (size_t i = 0; i < ndim; i++) {
                    out_idx += coords[i] * out_stride[i];
                }
                memcpy(out_ptr + out_idx * elem_size, in_ptr + idx * elem_size, elem_size);
            }
            boat_free(coords);
            boat_free(in_stride);
            boat_free(out_stride);
            break;
        }
    }

    return out;
}

// Invert a single n x n matrix using Gauss-Jordan elimination with partial
// pivoting on an augmented [A | I] matrix (n rows, 2n columns). Sets *OK to
// false if the matrix is singular.
#define BOAT_INVERT_SINGLE(T, AUG, N, ABSFN, OK)                                   \
    do {                                                                           \
        size_t _n = (N);                                                           \
        (OK) = true;                                                               \
        for (size_t _c = 0; _c < _n && (OK); _c++) {                               \
            size_t _p = _c;                                                        \
            T _maxv = ABSFN((AUG)[_c * (2 * _n) + _c]);                            \
            for (size_t _r = _c + 1; _r < _n; _r++) {                              \
                T _v = ABSFN((AUG)[_r * (2 * _n) + _c]);                           \
                if (_v > _maxv) { _maxv = _v; _p = _r; }                           \
            }                                                                      \
            if (_maxv < (T)1e-12) { (OK) = false; }                                \
            else {                                                                 \
                if (_p != _c) {                                                    \
                    for (size_t _j = 0; _j < 2 * _n; _j++) {                       \
                        T _t = (AUG)[_c * (2 * _n) + _j];                          \
                        (AUG)[_c * (2 * _n) + _j] = (AUG)[_p * (2 * _n) + _j];     \
                        (AUG)[_p * (2 * _n) + _j] = _t;                            \
                    }                                                              \
                }                                                                  \
                T _inv = (T)1 / (AUG)[_c * (2 * _n) + _c];                         \
                for (size_t _j = 0; _j < 2 * _n; _j++)                             \
                    (AUG)[_c * (2 * _n) + _j] *= _inv;                             \
                for (size_t _r = 0; _r < _n; _r++) {                               \
                    if (_r == _c) continue;                                        \
                    T _f = (AUG)[_r * (2 * _n) + _c];                              \
                    for (size_t _j = 0; _j < 2 * _n; _j++)                         \
                        (AUG)[_r * (2 * _n) + _j] -= _f * (AUG)[_c * (2 * _n) + _j]; \
                }                                                                  \
            }                                                                      \
        }                                                                          \
    } while (0)

// Matrix inverse: Gauss-Jordan elimination with partial pivoting.
// Supports 2D square matrices and batched matrices (leading dims are batch).
BOAT_API boat_tensor_t* boat_inverse(const boat_tensor_t* a) {
    if (!a) return NULL;
    size_t ndim = boat_tensor_ndim(a);
    const int64_t* shape = boat_tensor_shape(a);
    boat_dtype_t dtype = boat_tensor_dtype(a);

    if (ndim < 2) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Linear] inverse expects a >=2D tensor\n");
        return NULL;
    }
    if (dtype != BOAT_DTYPE_FLOAT32 && dtype != BOAT_DTYPE_FLOAT64) {
        boat_set_errorf(BOAT_ERROR_NOT_IMPLEMENTED,
                        "[Linear] inverse only supports float32/float64\n");
        return NULL;
    }

    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Linear] inverse requires square matrices\n");
        return NULL;
    }
    if (n == 0) return NULL;

    size_t batch = 1;
    for (size_t i = 0; i < ndim - 2; i++) batch *= (size_t)shape[i];

    boat_tensor_t* out = boat_tensor_create(shape, ndim, dtype, boat_tensor_device(a));
    if (!out) return NULL;

    const void* in = boat_tensor_const_data(a);
    void* out_data = boat_tensor_data(out);
    size_t nn = (size_t)n;

    if (dtype == BOAT_DTYPE_FLOAT32) {
        const float* in_f = (const float*)in;
        float* out_f = (float*)out_data;
        float* aug = (float*)malloc(2 * nn * nn * sizeof(float));
        if (!aug) { boat_tensor_unref(out); return NULL; }
        for (size_t b = 0; b < batch; b++) {
            const float* A = in_f + b * nn * nn;
            for (size_t i = 0; i < nn; i++) {
                for (size_t j = 0; j < nn; j++) {
                    aug[i * (2 * nn) + j] = A[i * nn + j];
                    aug[i * (2 * nn) + nn + j] = (i == j) ? 1.0f : 0.0f;
                }
            }
            bool ok;
            BOAT_INVERT_SINGLE(float, aug, nn, fabsf, ok);
            if (!ok) {
                free(aug);
                boat_tensor_unref(out);
                boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Linear] matrix is singular\n");
                return NULL;
            }
            float* Ainv = out_f + b * nn * nn;
            for (size_t i = 0; i < nn; i++)
                for (size_t j = 0; j < nn; j++)
                    Ainv[i * nn + j] = aug[i * (2 * nn) + nn + j];
        }
        free(aug);
    } else {
        const double* in_d = (const double*)in;
        double* out_d = (double*)out_data;
        double* aug = (double*)malloc(2 * nn * nn * sizeof(double));
        if (!aug) { boat_tensor_unref(out); return NULL; }
        for (size_t b = 0; b < batch; b++) {
            const double* A = in_d + b * nn * nn;
            for (size_t i = 0; i < nn; i++) {
                for (size_t j = 0; j < nn; j++) {
                    aug[i * (2 * nn) + j] = A[i * nn + j];
                    aug[i * (2 * nn) + nn + j] = (i == j) ? 1.0 : 0.0;
                }
            }
            bool ok;
            BOAT_INVERT_SINGLE(double, aug, nn, fabs, ok);
            if (!ok) {
                free(aug);
                boat_tensor_unref(out);
                boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Linear] matrix is singular\n");
                return NULL;
            }
            double* Ainv = out_d + b * nn * nn;
            for (size_t i = 0; i < nn; i++)
                for (size_t j = 0; j < nn; j++)
                    Ainv[i * nn + j] = aug[i * (2 * nn) + nn + j];
        }
        free(aug);
    }

    return out;
}