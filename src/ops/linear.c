// linear.c - Linear algebra operations for deep learning framework
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat/ops.h>
#include <boat/memory.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Matrix multiplication with batch support
boat_tensor_t* boat_matmul(const boat_tensor_t* a, const boat_tensor_t* b) {
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
    void* a_data = boat_tensor_data(a);
    void* b_data = boat_tensor_data(b);
    void* out_data = boat_tensor_data(out);

    // Perform matrix multiplication based on data type
    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            float* a_ptr = (float*)a_data;
            float* b_ptr = (float*)b_data;
            float* out_ptr = (float*)out_data;

            // Initialize output to zero
            size_t out_elements = boat_tensor_nelements(out);
            memset(out_ptr, 0, out_elements * sizeof(float));

            // Compute strides for batch dimension (flattened batch dimensions)
            bool has_batch = (batch_dims > 0);
            size_t a_batch_stride = has_batch ? (m * k) : 0;
            size_t b_batch_stride = has_batch ? (k * n) : 0;
            size_t out_batch_stride = has_batch ? (m * n) : 0;

            // Naive batch matrix multiplication
            for (int64_t batch = 0; batch < batch_size; batch++) {
                float* a_batch_ptr = a_ptr + batch * a_batch_stride;
                float* b_batch_ptr = b_ptr + batch * b_batch_stride;
                float* out_batch_ptr = out_ptr + batch * out_batch_stride;

                for (int64_t i = 0; i < m; i++) {
                    for (int64_t j = 0; j < n; j++) {
                        float sum = 0.0f;
                        for (int64_t l = 0; l < k; l++) {
                            sum += a_batch_ptr[i * k + l] * b_batch_ptr[l * n + j];
                        }
                        out_batch_ptr[i * n + j] = sum;
                    }
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            double* a_ptr = (double*)a_data;
            double* b_ptr = (double*)b_data;
            double* out_ptr = (double*)out_data;

            size_t out_elements = boat_tensor_nelements(out);
            memset(out_ptr, 0, out_elements * sizeof(double));

            bool has_batch = (batch_dims > 0);
            size_t a_batch_stride = has_batch ? (m * k) : 0;
            size_t b_batch_stride = has_batch ? (k * n) : 0;
            size_t out_batch_stride = has_batch ? (m * n) : 0;

            for (int64_t batch = 0; batch < batch_size; batch++) {
                double* a_batch_ptr = a_ptr + batch * a_batch_stride;
                double* b_batch_ptr = b_ptr + batch * b_batch_stride;
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
boat_tensor_t* boat_dot(const boat_tensor_t* a, const boat_tensor_t* b) {
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
    int64_t out_shape[] = {1};
    boat_tensor_t* out = boat_tensor_create(out_shape, 1, dtype, boat_tensor_device(a));
    if (!out) return NULL;

    void* a_data = boat_tensor_data(a);
    void* b_data = boat_tensor_data(b);
    void* out_data = boat_tensor_data(out);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            float* a_ptr = (float*)a_data;
            float* b_ptr = (float*)b_data;
            float* out_ptr = (float*)out_data;
            float sum = 0.0f;
            for (int64_t i = 0; i < n; i++) {
                sum += a_ptr[i] * b_ptr[i];
            }
            out_ptr[0] = sum;
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            double* a_ptr = (double*)a_data;
            double* b_ptr = (double*)b_data;
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
boat_tensor_t* boat_transpose(const boat_tensor_t* a, int dim0, int dim1) {
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

            // Transpose by iterating through all elements
            for (size_t idx = 0; idx < total_elements; idx++) {
                // Compute coordinates in input tensor
                size_t temp = idx;
                size_t* coords = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
                if (!coords) {
                    boat_free(in_stride);
                    boat_free(out_stride);
                    boat_tensor_unref(out);
                    return NULL;
                }
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
                boat_free(coords);
            }
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

            // Transpose by iterating through all elements
            for (size_t idx = 0; idx < total_elements; idx++) {
                // Compute coordinates in input tensor
                size_t temp = idx;
                size_t* coords = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
                if (!coords) {
                    boat_free(in_stride);
                    boat_free(out_stride);
                    boat_tensor_unref(out);
                    return NULL;
                }
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
                boat_free(coords);
            }
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
boat_tensor_t* boat_inverse(const boat_tensor_t* a) {
    (void)a;
    // TODO: Implement matrix inverse
    return NULL;
}