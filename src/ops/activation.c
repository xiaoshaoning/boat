// activation.c - Activation functions for deep learning framework
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat/ops.h>
#include <boat/memory.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

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

    void* a_data = boat_tensor_data(a);
    void* out_data = boat_tensor_data(out);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            float* a_ptr = (float*)a_data;
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
                        out_ptr[idx] = exp_val;
                        exp_sum += exp_val;
                    }

                    // Normalize
                    if (exp_sum != 0.0f) {
                        float inv_exp_sum = 1.0f / exp_sum;
                        for (size_t k = 0; k < axis_size; k++) {
                            size_t idx = base_idx + k * inner_stride;
                            out_ptr[idx] *= inv_exp_sum;
                        }
                    }
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            double* a_ptr = (double*)a_data;
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
boat_tensor_t* boat_log_softmax(const boat_tensor_t* a, int axis) {
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

    void* a_data = boat_tensor_data(a);
    void* out_data = boat_tensor_data(out);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            float* a_ptr = (float*)a_data;
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
                    float log_exp_sum = logf(exp_sum);
                    for (size_t k = 0; k < axis_size; k++) {
                        size_t idx = base_idx + k * inner_stride;
                        out_ptr[idx] = a_ptr[idx] - max_val - log_exp_sum;
                    }
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            double* a_ptr = (double*)a_data;
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

                    double log_exp_sum = log(exp_sum);
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

// Other activation functions (placeholders for now)
BOAT_API boat_tensor_t* boat_relu(const boat_tensor_t* a) {
    // 紧急修复：简单的ReLU实现
    // 写入日志文件确认函数被调用
    FILE* debug_log = fopen("boat_relu_debug.log", "a");
    if (debug_log) {
        fprintf(debug_log, "boat_relu called at %p with input %p\n",
                (void*)boat_relu, (void*)a);
        fclose(debug_log);
    }

    if (!a) {
        return NULL;
    }

    // 创建输出张量
    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) {
        return NULL;
    }

    // 获取张量信息
    boat_dtype_t dtype = boat_tensor_dtype(a);
    size_t total_elements = boat_tensor_nelements(a);
    void* a_data = boat_tensor_data(a);
    void* out_data = boat_tensor_data(out);


    // 只实现FP32版本（MNIST使用FP32）
    if (dtype == BOAT_DTYPE_FLOAT32 && total_elements > 0) {
        float* a_ptr = (float*)a_data;
        float* out_ptr = (float*)out_data;
        for (size_t i = 0; i < total_elements; i++) {
            out_ptr[i] = a_ptr[i] > 0.0f ? a_ptr[i] : 0.0f;
        }
        return out;
    }

    // 对于其他数据类型，暂时返回原张量副本
    if (total_elements > 0 && a_data && out_data) {
        size_t bytes = boat_tensor_nbytes(a);
        memcpy(out_data, a_data, bytes);
    }

    return out;
}


boat_tensor_t* boat_sigmoid(const boat_tensor_t* a) {
    // TODO: Implement sigmoid
    (void)a;
    return NULL;
}

boat_tensor_t* boat_tanh(const boat_tensor_t* a) {
    // TODO: Implement tanh
    (void)a;
    return NULL;
}

boat_tensor_t* boat_gelu(const boat_tensor_t* a) {
    // TODO: Implement GELU
    (void)a;
    return NULL;
}

boat_tensor_t* boat_selu(const boat_tensor_t* a) {
    // TODO: Implement SELU
    (void)a;
    return NULL;
}