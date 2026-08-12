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
BOAT_API boat_tensor_t* boat_matmul(const boat_tensor_t* a, const boat_tensor_t* b) {
    if (!a || !b) {
        return NULL;
    }

    size_t a_ndim = boat_tensor_ndim(a);
    size_t b_ndim = boat_tensor_ndim(b);

    const int64_t* a_shape = boat_tensor_shape(a);
    const int64_t* b_shape = boat_tensor_shape(b);

    // Validate dimensions: support 2D, 3D, or 4D tensors
    if (a_ndim < 2 || a_ndim > 4 || b_ndim < 2 || b_ndim > 4) {
        return NULL;
    }

    // Determine batch dimensions (all dimensions except last two)
    size_t a_batch_dims = (a_ndim > 2) ? a_ndim - 2 : 0;
    size_t b_batch_dims = (b_ndim > 2) ? b_ndim - 2 : 0;

    // For now, require same number of batch dimensions
    // TODO: Support broadcasting (e.g., 3D matmul with 4D)
    if (a_batch_dims != b_batch_dims) {
        return NULL;
    }

    size_t batch_dims = a_batch_dims; // same as b_batch_dims
    int64_t batch_size = 1;

    // Check that batch dimensions match
    for (size_t i = 0; i < batch_dims; i++) {
        if (a_shape[i] != b_shape[i]) {
            return NULL;
        }
        batch_size *= a_shape[i];
    }

    // Matrix dimensions: last two dimensions of each tensor
    int64_t m, k_a, k_b, n;

    if (batch_dims > 0) {
        // a_shape indices: batch_dims, batch_dims+1
        m = a_shape[batch_dims];
        k_a = a_shape[batch_dims + 1];
        k_b = b_shape[batch_dims];
        n = b_shape[batch_dims + 1];
    } else {
        // 2D case
        m = a_shape[0];
        k_a = a_shape[1];
        k_b = b_shape[0];
        n = b_shape[1];
    }

    // Check dimension compatibility
    if (k_a != k_b) {
        return NULL;
    }

    int64_t k = k_a;

    // Determine output shape: batch dimensions + (m, n)
    size_t out_ndim = batch_dims + 2;
    int64_t out_shape[4]; // max 4 dimensions (batch_dims up to 2)

    // Copy batch dimensions
    for (size_t i = 0; i < batch_dims; i++) {
        out_shape[i] = a_shape[i];
    }
    // Add matrix dimensions
    out_shape[batch_dims] = m;
    out_shape[batch_dims + 1] = n;

    boat_dtype_t dtype = boat_tensor_dtype(a);
    if (dtype != boat_tensor_dtype(b)) {
        return NULL; // TODO: Type promotion
    }

    boat_tensor_t* out = boat_tensor_create(out_shape, out_ndim, dtype, boat_tensor_device(a));
    if (!out) return NULL;

    // Get data pointers
    const void* a_data = boat_tensor_data(a);
    const void* b_data = boat_tensor_data(b);
    void* out_data = boat_tensor_data(out);

    // Perform matrix multiplication based on data type
    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            const float* b_ptr = (const float*)b_data;
            float* out_ptr = (float*)out_data;

            // Initialize output to zero (device-aware — use cudaMemset for CUDA tensors)
            size_t out_elements = boat_tensor_nelements(out);
            boat_memory_set(out_ptr, 0, out_elements * sizeof(float), boat_tensor_device(a));

            // Compute strides for batch dimension (flattened batch dimensions)
            bool has_batch = (batch_dims > 0);
            size_t a_batch_stride = has_batch ? (m * k) : 0;
            size_t b_batch_stride = has_batch ? (k * n) : 0;
            size_t out_batch_stride = has_batch ? (m * n) : 0;

            // Dispatch to cuBLAS for CUDA tensors, CPU SGEMM otherwise
            int batch_count = (int)batch_size;
#ifdef BOAT_WITH_CUDA
            if (boat_tensor_device(a) == BOAT_DEVICE_CUDA) {
                if (batch_count == 1) {
                    boat_cuda_matmul_f32_cublas(a_ptr, b_ptr, out_ptr, (size_t)m, (size_t)n, (size_t)k);
                } else {
                    boat_cuda_matmul_f32_strided_batched(
                        a_ptr, b_ptr, out_ptr,
                        (size_t)m, (size_t)n, (size_t)k,
                        (size_t)batch_count,
                        a_batch_stride, b_batch_stride, out_batch_stride);
                }
            } else
#endif
            {
                int batch;
                BOAT_OMP_PARALLEL_FOR
                for (batch = 0; batch < batch_count; batch++) {
                    const float* a_batch_ptr = a_ptr + batch * a_batch_stride;
                    const float* b_batch_ptr = b_ptr + batch * b_batch_stride;
                    float* out_batch_ptr = out_ptr + batch * out_batch_stride;

                    boat_sgemm(m, n, k, a_batch_ptr, b_batch_ptr, out_batch_ptr);
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

            bool has_batch = (batch_dims > 0);
            size_t a_batch_stride = has_batch ? (m * k) : 0;
            size_t b_batch_stride = has_batch ? (k * n) : 0;
            size_t out_batch_stride = has_batch ? (m * n) : 0;

            for (int64_t batch = 0; batch < batch_size; batch++) {
                const double* a_batch_ptr = a_ptr + batch * a_batch_stride;
                const double* b_batch_ptr = b_ptr + batch * b_batch_stride;
                double* out_batch_ptr = out_ptr + batch * out_batch_stride;

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
            // Unsupported data type for matrix multiplication
            boat_tensor_unref(out);
            return NULL;
    }

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
    if (dim0 >= ndim || dim1 >= ndim) {
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
        default:
            // For unsupported types, fall back to memcpy (no actual transposition)
            memcpy(out_data, in_data, boat_tensor_nbytes(a));
            break;
    }

    return out;
}

// Matrix inverse (placeholder)
BOAT_API boat_tensor_t* boat_inverse(const boat_tensor_t* a) {
    (void)a;
    // TODO: Implement matrix inverse
    return NULL;
}