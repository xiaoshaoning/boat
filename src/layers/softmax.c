// softmax.c - Softmax activation layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Softmax layer structure
struct boat_softmax_layer_t {
    int axis;                    // Dimension along which softmax is applied
    boat_tensor_t* cache_output; // Softmax output from the last forward pass
};

BOAT_API boat_softmax_layer_t* BOAT_CALL boat_softmax_layer_create(int axis) {
    boat_softmax_layer_t* layer =
        (boat_softmax_layer_t*)boat_malloc(sizeof(boat_softmax_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }

    layer->axis = axis;
    layer->cache_output = NULL;
    return layer;
}

BOAT_API void BOAT_CALL boat_softmax_layer_free(boat_softmax_layer_t* layer) {
    if (!layer) {
        return;
    }
    if (layer->cache_output) {
        boat_tensor_unref(layer->cache_output);
    }
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_softmax_layer_forward(const boat_softmax_layer_t* layer,
                                                             const boat_tensor_t* input) {
    if (!layer || !input) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[SoftmaxLayer] NULL input or layer\n");
        return NULL;
    }

    boat_tensor_t* output = boat_softmax(input, layer->axis);
    if (!output) {
        return NULL;
    }

    // Cache the softmax output for the backward pass
    boat_softmax_layer_t* self = (boat_softmax_layer_t*)layer;
    if (self->cache_output) {
        boat_tensor_unref(self->cache_output);
    }
    self->cache_output = output;
    boat_tensor_ref(output);

    return output;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_softmax_layer_backward(boat_softmax_layer_t* layer,
                                                              const boat_tensor_t* grad_output) {
    if (!layer || !grad_output || !layer->cache_output) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[SoftmaxLayer] NULL gradient output or missing forward cache\n");
        return NULL;
    }

    // Softmax Jacobian: for y = softmax(x), dL/dx_i = y_i * (dL/dy_i - sum_j(y_j * dL/dy_j))
    const boat_tensor_t* y = layer->cache_output;
    size_t ndim = boat_tensor_ndim(y);
    const int64_t* shape = boat_tensor_shape(y);

    int axis = layer->axis;
    if (axis < 0) axis += (int)ndim;
    if (axis < 0 || axis >= (int)ndim) {
        return NULL;
    }

    size_t axis_size = (size_t)shape[axis];
    size_t outer = 1;
    for (int i = 0; i < axis; i++)
        outer *= (size_t)shape[i];
    size_t inner = 1;
    for (int i = axis + 1; i < (int)ndim; i++)
        inner *= (size_t)shape[i];

    boat_tensor_t* grad = boat_tensor_create_like(y);
    if (!grad) {
        return NULL;
    }

    boat_dtype_t dtype = boat_tensor_dtype(y);
    const void* y_data = boat_tensor_const_data(y);
    const void* g_data = boat_tensor_const_data(grad_output);
    void* out_data = boat_tensor_data(grad);

    switch (dtype) {
    case BOAT_DTYPE_FLOAT32: {
        const float* y_ptr = (const float*)y_data;
        const float* g_ptr = (const float*)g_data;
        float* out_ptr = (float*)out_data;
        for (size_t o = 0; o < outer; o++) {
            for (size_t i = 0; i < inner; i++) {
                size_t base = o * axis_size * inner + i;
                float sum_y_g = 0.0f;
                for (size_t k = 0; k < axis_size; k++) {
                    sum_y_g += y_ptr[base + k * inner] * g_ptr[base + k * inner];
                }
                for (size_t k = 0; k < axis_size; k++) {
                    out_ptr[base + k * inner] =
                        y_ptr[base + k * inner] * (g_ptr[base + k * inner] - sum_y_g);
                }
            }
        }
        break;
    }
    case BOAT_DTYPE_FLOAT64: {
        const double* y_ptr = (const double*)y_data;
        const double* g_ptr = (const double*)g_data;
        double* out_ptr = (double*)out_data;
        for (size_t o = 0; o < outer; o++) {
            for (size_t i = 0; i < inner; i++) {
                size_t base = o * axis_size * inner + i;
                double sum_y_g = 0.0;
                for (size_t k = 0; k < axis_size; k++) {
                    sum_y_g += y_ptr[base + k * inner] * g_ptr[base + k * inner];
                }
                for (size_t k = 0; k < axis_size; k++) {
                    out_ptr[base + k * inner] =
                        y_ptr[base + k * inner] * (g_ptr[base + k * inner] - sum_y_g);
                }
            }
        }
        break;
    }
    default:
        boat_tensor_unref(grad);
        boat_set_errorf(BOAT_ERROR_UNKNOWN,
                        "[SoftmaxLayer] Backward only supports float32/float64\n");
        return NULL;
    }

    return grad;
}

BOAT_API void BOAT_CALL boat_softmax_layer_update(boat_softmax_layer_t* layer,
                                                  float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // Softmax has no parameters to update
}

BOAT_API int BOAT_CALL boat_softmax_layer_get_axis(const boat_softmax_layer_t* layer) {
    return layer ? layer->axis : 0;
}
