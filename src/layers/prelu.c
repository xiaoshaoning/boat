// prelu.c - PReLU (Parametric ReLU) activation layer
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <stdlib.h>
#include <string.h>

struct boat_prelu_layer_t {
    boat_tensor_t* slope;  // per-channel slope, shape [C, 1, 1]
};

BOAT_API BOAT_API boat_prelu_layer_t* BOAT_CALL boat_prelu_layer_create(size_t num_params) {
    boat_prelu_layer_t* layer = (boat_prelu_layer_t*)boat_malloc(sizeof(boat_prelu_layer_t), BOAT_DEVICE_CPU);
    if (!layer) return NULL;
    layer->slope = NULL;
    if (num_params > 0) {
        int64_t shape[] = {(int64_t)num_params, 1, 1};
        layer->slope = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!layer->slope) { boat_free(layer); return NULL; }
        // Initialize slope to 0.25 (default PReLU init)
        float* d = (float*)boat_tensor_data(layer->slope);
        for (size_t i = 0; i < num_params; i++) d[i] = 0.25f;
    }
    return layer;
}

BOAT_API BOAT_API void BOAT_CALL boat_prelu_layer_free(boat_prelu_layer_t* layer) {
    if (!layer) return;
    if (layer->slope) boat_tensor_unref(layer->slope);
    boat_free(layer);
}

BOAT_API BOAT_API void BOAT_CALL boat_prelu_layer_set_slope(boat_prelu_layer_t* layer, const boat_tensor_t* slope) {
    if (!layer || !slope) return;
    if (layer->slope) boat_tensor_unref(layer->slope);
    layer->slope = boat_tensor_create(boat_tensor_shape(slope), boat_tensor_ndim(slope),
                                       BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (layer->slope) {
        memcpy(boat_tensor_data(layer->slope), boat_tensor_const_data(slope),
               boat_tensor_nbytes(slope));
    }
}

BOAT_API BOAT_API boat_tensor_t* BOAT_CALL boat_prelu_layer_get_slope(const boat_prelu_layer_t* layer) {
    return layer ? layer->slope : NULL;
}

BOAT_API BOAT_API boat_tensor_t* BOAT_CALL boat_prelu_layer_forward(const boat_prelu_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[PReLULayer] NULL input or layer\n");
        return NULL;
    }

    boat_tensor_t* result = boat_tensor_create_like(input);
    if (!result) return NULL;

    const float* in = (const float*)boat_tensor_const_data(input);
    float* out = (float*)boat_tensor_data(result);
    size_t n = boat_tensor_nelements(input);
    size_t channels = 1;
    if (layer->slope) {
        const int64_t* sshape = boat_tensor_shape(layer->slope);
        channels = (size_t)sshape[0];
    }

    // Determine spatial size per channel
    const int64_t* ishape = boat_tensor_shape(input);
    size_t ndim = boat_tensor_ndim(input);
    size_t spatial = 1;
    // For NCHW: dims are [N, C, H, W] or [N, C, H, W, D] etc.
    // Channel dim is always 1, spatial dims are 2..ndim-1
    size_t ch_dim = (ndim > 1) ? 1 : 0;
    size_t ch_stride = 1;
    for (size_t d = ch_dim + 1; d < ndim; d++) ch_stride *= (size_t)ishape[d];
    // ch_stride is number of elements per channel (without batch)
    size_t batch_stride = (ndim > 1) ? (size_t)ishape[1] * ch_stride : n;

    const float* slope_data = layer->slope ? (const float*)boat_tensor_const_data(layer->slope) : NULL;

    if (slope_data && channels > 1) {
        // Per-channel PReLU: slope broadcasts over spatial dims
        // Input may have batch dimension: [N, C, H, W]
        for (size_t b = 0; b < n / batch_stride; b++) {
            for (size_t c = 0; c < channels && c < (size_t)ishape[1]; c++) {
                float s = slope_data[c >= channels ? channels - 1 : c];
                for (size_t sp = 0; sp < ch_stride; sp++) {
                    float x = in[b * batch_stride + c * ch_stride + sp];
                    out[b * batch_stride + c * ch_stride + sp] = (x > 0.0f) ? x : s * x;
                }
            }
        }
    } else {
        // Scalar or no slope
        float s = slope_data ? slope_data[0] : 0.25f;
        for (size_t i = 0; i < n; i++) {
            float x = in[i];
            out[i] = (x > 0.0f) ? x : s * x;
        }
    }

    return result;
}
