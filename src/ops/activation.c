// activation.c - Activation functions for deep learning framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/ops.h>
#include <boat/memory.h>
#include <boat/simd.h>
#include "../core/openmp.h"
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

#ifdef BOAT_WITH_CUDA
#include <boat/cuda_runtime.h>
#endif

// Helper function to compute stride for a given dimension
static size_t compute_stride(const int64_t* shape, size_t ndim, size_t axis) {
    size_t stride = 1;
    for (size_t i = axis + 1; i < ndim; i++) {
        stride *= shape[i];
    }
    return stride;
}

// Helper function to compute total number of elements up to a dimension
static size_t compute_elements_before(const int64_t* shape, size_t ndim, size_t axis) {
    size_t elements = 1;
    for (size_t i = 0; i < axis; i++) {
        elements *= shape[i];
    }
    return elements;
}

// Softmax along a specific axis
BOAT_API boat_tensor_t* boat_softmax(const boat_tensor_t* a, int axis) {
    if (!a) return NULL;

    size_t ndim = boat_tensor_ndim(a);
    const int64_t* shape = boat_tensor_shape(a);

    // Handle negative axis (Python-style)
    if (axis < 0) {
        axis += ndim;
    }
    if (axis < 0 || axis >= ndim) {
        return NULL; // Invalid axis
    }

    size_t axis_size = shape[axis];
    size_t outer_elements = compute_elements_before(shape, ndim, axis);
    size_t inner_stride = compute_stride(shape, ndim, axis);

    boat_dtype_t dtype = boat_tensor_dtype(a);
    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    size_t total_elements = boat_tensor_nelements(a);
    if (total_elements == 0) return out;

    const void* a_data = boat_tensor_data(a);
    void* out_data = boat_tensor_data(out);

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(a) == BOAT_DEVICE_CUDA && dtype == BOAT_DTYPE_FLOAT32) {
        boat_cuda_softmax_f32((const float*)boat_tensor_const_data(a),
                               (float*)boat_tensor_data(out),
                               (int64_t)outer_elements, (int64_t)axis_size, (int64_t)inner_stride);
        return out;
    }
#endif

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            float* out_ptr = (float*)out_data;

            size_t inner_str = inner_stride;
            size_t ax_sz = axis_size;
            int outer_limit = (int)outer_elements;
            int outer;
            BOAT_OMP_PARALLEL_FOR
            for (outer = 0; outer < outer_limit; outer++) {
                for (size_t inner = 0; inner < inner_str; inner++) {
                    size_t base_idx = (size_t)outer * ax_sz * inner_str + inner;

                    float max_val = a_ptr[base_idx];
                    for (size_t k = 1; k < ax_sz; k++) {
                        size_t idx = base_idx + k * inner_str;
                        float val = a_ptr[idx];
                        if (val > max_val) max_val = val;
                    }

                    float exp_sum = 0.0f;
                    for (size_t k = 0; k < ax_sz; k++) {
                        size_t idx = base_idx + k * inner_str;
                        float val = a_ptr[idx] - max_val;
                        float exp_val = expf(val);
                        out_ptr[idx] = exp_val;
                        exp_sum += exp_val;
                    }

                    if (exp_sum != 0.0f) {
                        float inv_exp_sum = 1.0f / exp_sum;
                        for (size_t k = 0; k < ax_sz; k++) {
                            size_t idx = base_idx + k * inner_str;
                            out_ptr[idx] *= inv_exp_sum;
                        }
                    }
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            double* out_ptr = (double*)out_data;

            for (size_t outer = 0; outer < outer_elements; outer++) {
                for (size_t inner = 0; inner < inner_stride; inner++) {
                    size_t base_idx = outer * axis_size * inner_stride + inner;

                    double max_val = a_ptr[base_idx];
                    for (size_t k = 1; k < axis_size; k++) {
                        size_t idx = base_idx + k * inner_stride;
                        double val = a_ptr[idx];
                        if (val > max_val) max_val = val;
                    }

                    double exp_sum = 0.0;
                    for (size_t k = 0; k < axis_size; k++) {
                        size_t idx = base_idx + k * inner_stride;
                        double val = a_ptr[idx] - max_val;
                        double exp_val = exp(val);
                        out_ptr[idx] = exp_val;
                        exp_sum += exp_val;
                    }

                    if (exp_sum != 0.0) {
                        double inv_exp_sum = 1.0 / exp_sum;
                        for (size_t k = 0; k < axis_size; k++) {
                            size_t idx = base_idx + k * inner_stride;
                            out_ptr[idx] *= inv_exp_sum;
                        }
                    }
                }
            }
            break;
        }
        default:
            // Only support float types for softmax
            boat_tensor_unref(out);
            return NULL;
    }

    return out;
}

// Log softmax along a specific axis
BOAT_API boat_tensor_t* boat_log_softmax(const boat_tensor_t* a, int axis) {
    if (!a) return NULL;

    size_t ndim = boat_tensor_ndim(a);
    const int64_t* shape = boat_tensor_shape(a);

    if (axis < 0) {
        axis += ndim;
    }
    if (axis < 0 || axis >= ndim) {
        return NULL;
    }

    size_t axis_size = shape[axis];
    size_t outer_elements = compute_elements_before(shape, ndim, axis);
    size_t inner_stride = compute_stride(shape, ndim, axis);

    boat_dtype_t dtype = boat_tensor_dtype(a);
    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    size_t total_elements = boat_tensor_nelements(a);
    if (total_elements == 0) return out;

    const void* a_data = boat_tensor_data(a);
    void* out_data = boat_tensor_data(out);

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(a) == BOAT_DEVICE_CUDA && dtype == BOAT_DTYPE_FLOAT32) {
        boat_cuda_log_softmax_f32((const float*)a_data, (float*)out_data,
                                   (int64_t)outer_elements, (int64_t)axis_size, (int64_t)inner_stride);
        return out;
    }
#endif

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            float* out_ptr = (float*)out_data;

            for (size_t outer = 0; outer < outer_elements; outer++) {
                for (size_t inner = 0; inner < inner_stride; inner++) {
                    size_t base_idx = outer * axis_size * inner_stride + inner;

                    // Find max for numerical stability
                    float max_val = a_ptr[base_idx];
                    for (size_t k = 1; k < axis_size; k++) {
                        size_t idx = base_idx + k * inner_stride;
                        float val = a_ptr[idx];
                        if (val > max_val) max_val = val;
                    }

                    // Compute exp(x - max) and sum
                    float exp_sum = 0.0f;
                    for (size_t k = 0; k < axis_size; k++) {
                        size_t idx = base_idx + k * inner_stride;
                        float val = a_ptr[idx] - max_val;
                        float exp_val = expf(val);
                        exp_sum += exp_val;
                    }

                    // Compute log_softmax: x - max - log(sum(exp(x - max)))
                    float log_exp_sum = logf(fmaxf(exp_sum, FLT_MIN));
                    for (size_t k = 0; k < axis_size; k++) {
                        size_t idx = base_idx + k * inner_stride;
                        out_ptr[idx] = a_ptr[idx] - max_val - log_exp_sum;
                    }
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            double* out_ptr = (double*)out_data;

            for (size_t outer = 0; outer < outer_elements; outer++) {
                for (size_t inner = 0; inner < inner_stride; inner++) {
                    size_t base_idx = outer * axis_size * inner_stride + inner;

                    double max_val = a_ptr[base_idx];
                    for (size_t k = 1; k < axis_size; k++) {
                        size_t idx = base_idx + k * inner_stride;
                        double val = a_ptr[idx];
                        if (val > max_val) max_val = val;
                    }

                    double exp_sum = 0.0;
                    for (size_t k = 0; k < axis_size; k++) {
                        size_t idx = base_idx + k * inner_stride;
                        double val = a_ptr[idx] - max_val;
                        double exp_val = exp(val);
                        exp_sum += exp_val;
                    }

                    double log_exp_sum = log(fmax(exp_sum, DBL_MIN));
                    for (size_t k = 0; k < axis_size; k++) {
                        size_t idx = base_idx + k * inner_stride;
                        out_ptr[idx] = a_ptr[idx] - max_val - log_exp_sum;
                    }
                }
            }
            break;
        }
        default:
            boat_tensor_unref(out);
            return NULL;
    }

    return out;
}

// Other activation functions
BOAT_API boat_tensor_t* boat_relu(const boat_tensor_t* a) {

    if (!a) {
        return NULL;
    }

    // Create output tensor
    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) {
        return NULL;
    }

    // Get tensor information
    boat_dtype_t dtype = boat_tensor_dtype(a);
    size_t total_elements = boat_tensor_nelements(a);
    const void* a_data = boat_tensor_data(a);
    void* out_data = boat_tensor_data(out);

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(a) == BOAT_DEVICE_CUDA && dtype == BOAT_DTYPE_FLOAT32 && total_elements > 0) {
        boat_cuda_relu_f32((const float*)a_data, (float*)out_data, total_elements);
        return out;
    }
#endif

    // Only implement FP32 version (MNIST uses FP32)
    if (dtype == BOAT_DTYPE_FLOAT32 && total_elements > 0) {
        const float* a_ptr = (const float*)a_data;
        float* out_ptr = (float*)out_data;
        boat_simd_relu_f32(a_ptr, out_ptr, total_elements);
        return out;
    }

    // For other data types, temporarily return a copy of the tensor
    if (total_elements > 0 && a_data && out_data) {
        size_t bytes = boat_tensor_nbytes(a);
        memcpy(out_data, a_data, bytes);
    }

    return out;
}


BOAT_API boat_tensor_t* boat_sigmoid(const boat_tensor_t* a) {
    if (!a) return NULL;

    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    boat_dtype_t dtype = boat_tensor_dtype(a);
    size_t n = boat_tensor_nelements(a);

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(a) == BOAT_DEVICE_CUDA && dtype == BOAT_DTYPE_FLOAT32) {
        boat_cuda_sigmoid_f32((const float*)boat_tensor_const_data(a),
                              (float*)boat_tensor_data(out), n);
        return out;
    }
#endif

    if (dtype == BOAT_DTYPE_FLOAT32) {
        const float* src = (const float*)boat_tensor_const_data(a);
        float* dst = (float*)boat_tensor_data(out);
        for (size_t i = 0; i < n; i++) {
            dst[i] = 1.0f / (1.0f + expf(-src[i]));
        }
        return out;
    }

    if (dtype == BOAT_DTYPE_FLOAT64) {
        const double* src = (const double*)boat_tensor_const_data(a);
        double* dst = (double*)boat_tensor_data(out);
        for (size_t i = 0; i < n; i++) {
            dst[i] = 1.0 / (1.0 + exp(-src[i]));
        }
        return out;
    }

    boat_tensor_unref(out);
    return NULL;
}

BOAT_API boat_tensor_t* boat_silu(const boat_tensor_t* a) {
    if (!a) return NULL;

    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    boat_dtype_t dtype = boat_tensor_dtype(a);
    size_t n = boat_tensor_nelements(a);

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(a) == BOAT_DEVICE_CUDA && dtype == BOAT_DTYPE_FLOAT32) {
        boat_cuda_silu_f32((const float*)boat_tensor_const_data(a),
                            (float*)boat_tensor_data(out), n);
        return out;
    }
#endif

    if (dtype == BOAT_DTYPE_FLOAT32) {
        const float* src = (const float*)boat_tensor_const_data(a);
        float* dst = (float*)boat_tensor_data(out);
        for (size_t i = 0; i < n; i++) {
            dst[i] = src[i] / (1.0f + expf(-src[i]));
        }
        return out;
    }

    if (dtype == BOAT_DTYPE_FLOAT64) {
        const double* src = (const double*)boat_tensor_const_data(a);
        double* dst = (double*)boat_tensor_data(out);
        for (size_t i = 0; i < n; i++) {
            dst[i] = src[i] / (1.0 + exp(-src[i]));
        }
        return out;
    }

    boat_tensor_unref(out);
    return NULL;
}

BOAT_API boat_tensor_t* boat_tanh(const boat_tensor_t* a) {
    if (!a) return NULL;

    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    boat_dtype_t dtype = boat_tensor_dtype(a);
    size_t n = boat_tensor_nelements(a);

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(a) == BOAT_DEVICE_CUDA && dtype == BOAT_DTYPE_FLOAT32) {
        boat_cuda_tanh_f32((const float*)boat_tensor_const_data(a),
                           (float*)boat_tensor_data(out), n);
        return out;
    }
#endif

    if (dtype == BOAT_DTYPE_FLOAT32) {
        const float* src = (const float*)boat_tensor_const_data(a);
        float* dst = (float*)boat_tensor_data(out);
        for (size_t i = 0; i < n; i++) {
            dst[i] = tanhf(src[i]);
        }
        return out;
    }

    if (dtype == BOAT_DTYPE_FLOAT64) {
        const double* src = (const double*)boat_tensor_const_data(a);
        double* dst = (double*)boat_tensor_data(out);
        for (size_t i = 0; i < n; i++) {
            dst[i] = tanh(src[i]);
        }
        return out;
    }

    boat_tensor_unref(out);
    return NULL;
}

BOAT_API boat_tensor_t* boat_gelu(const boat_tensor_t* a) {
    if (!a) return NULL;

    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    boat_dtype_t dtype = boat_tensor_dtype(a);
    size_t n = boat_tensor_nelements(a);

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(a) == BOAT_DEVICE_CUDA && dtype == BOAT_DTYPE_FLOAT32) {
        boat_cuda_gelu_f32((const float*)boat_tensor_const_data(a),
                            (float*)boat_tensor_data(out), n);
        return out;
    }
#endif

    if (dtype == BOAT_DTYPE_FLOAT32) {
        const float* src = (const float*)boat_tensor_const_data(a);
        float* dst = (float*)boat_tensor_data(out);
        for (size_t i = 0; i < n; i++) {
            float x = src[i];
            float x3 = x * x * x;
            float inner = 0.7978845608028654f * (x + 0.044715f * x3);
            dst[i] = 0.5f * x * (1.0f + tanhf(inner));
        }
        return out;
    }

    boat_tensor_unref(out);
    return NULL;
}

BOAT_API boat_tensor_t* boat_selu(const boat_tensor_t* a) {
    if (!a) return NULL;

    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    boat_dtype_t dtype = boat_tensor_dtype(a);
    size_t n = boat_tensor_nelements(a);
    const double scale_d = 1.0507009873554804934193349852946;
    const double alpha_d = 1.6732632423543772848170429916717;
    const float scale = (float)scale_d;
    const float alpha = (float)alpha_d;

    if (dtype == BOAT_DTYPE_FLOAT32) {
        const float* src = (const float*)boat_tensor_const_data(a);
        float* dst = (float*)boat_tensor_data(out);
        for (size_t i = 0; i < n; i++) {
            float x = src[i];
            dst[i] = scale * (x > 0.0f ? x : alpha * (expf(x) - 1.0f));
        }
        return out;
    }

    if (dtype == BOAT_DTYPE_FLOAT64) {
        const double* src = (const double*)boat_tensor_const_data(a);
        double* dst = (double*)boat_tensor_data(out);
        for (size_t i = 0; i < n; i++) {
            double x = src[i];
            dst[i] = scale_d * (x > 0.0 ? x : alpha_d * (exp(x) - 1.0));
        }
        return out;
    }

    boat_tensor_unref(out);
    return NULL;
}

BOAT_API boat_tensor_t* boat_sinusoidal_embedding(size_t seq_len, size_t embedding_dim, float theta) {
    if (seq_len == 0 || embedding_dim == 0 || embedding_dim % 2 != 0) {
        return NULL;
    }

    const int64_t shape[] = { (int64_t)seq_len, (int64_t)embedding_dim };
    boat_tensor_t* out = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!out) {
        return NULL;
    }

    float* data = (float*)boat_tensor_data(out);
    size_t half = embedding_dim / 2;
    float log_theta = logf(theta);

    for (size_t p = 0; p < seq_len; p++) {
        for (size_t i = 0; i < half; i++) {
            float freq = expf(-log_theta * (float)i / (float)(half - 1));
            float angle = (float)p * freq;
            data[p * embedding_dim + i] = sinf(angle);
            data[p * embedding_dim + half + i] = cosf(angle);
        }
    }

    return out;
}
