// prelu.c - PReLU (Parametric ReLU) activation layer
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <stdlib.h>
#include <string.h>

struct boat_prelu_layer_t {
    boat_tensor_t* slope;       // per-channel slope, shape [C, 1, 1]
    boat_tensor_t* grad_slope;  // accumulated gradient, shape [C, 1, 1]
    boat_tensor_t* cache_input; // input cached during forward (for backward)
};

BOAT_API boat_prelu_layer_t* BOAT_CALL boat_prelu_layer_create(size_t num_params) {
    boat_prelu_layer_t* layer = (boat_prelu_layer_t*)boat_malloc(sizeof(boat_prelu_layer_t), BOAT_DEVICE_CPU);
    if (!layer) return NULL;
    layer->slope = NULL;
    layer->grad_slope = NULL;
    layer->cache_input = NULL;
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

BOAT_API void BOAT_CALL boat_prelu_layer_free(boat_prelu_layer_t* layer) {
    if (!layer) return;
    if (layer->slope) boat_tensor_unref(layer->slope);
    if (layer->grad_slope) boat_tensor_unref(layer->grad_slope);
    if (layer->cache_input) boat_tensor_unref(layer->cache_input);
    boat_free(layer);
}

BOAT_API void BOAT_CALL boat_prelu_layer_set_slope(boat_prelu_layer_t* layer, const boat_tensor_t* slope) {
    if (!layer || !slope) return;
    if (layer->slope) boat_tensor_unref(layer->slope);
    layer->slope = boat_tensor_create(boat_tensor_shape(slope), boat_tensor_ndim(slope),
                                       BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (layer->slope) {
        memcpy(boat_tensor_data(layer->slope), boat_tensor_const_data(slope),
               boat_tensor_nbytes(slope));
    }
}

BOAT_API boat_tensor_t* BOAT_CALL boat_prelu_layer_get_slope(const boat_prelu_layer_t* layer) {
    return layer ? layer->slope : NULL;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_prelu_layer_get_grad_slope(const boat_prelu_layer_t* layer) {
    return layer ? layer->grad_slope : NULL;
}

// Channel index for a linear element index in an NCHW tensor.
static size_t channel_index(const int64_t* shape, size_t ndim, size_t linear) {
    size_t spatial = 1;
    for (size_t d = 2; d < ndim; d++) spatial *= (size_t)shape[d];
    size_t ch_dim = (ndim > 1) ? 1 : 0;
    size_t ch_stride = (ndim > 1) ? spatial : 1;
    if (ch_dim == 0) return 0;  // no channel dim; scalar slope
    // linear = n * (C * ch_stride) + c * ch_stride + sp
    return (linear / ch_stride) % (size_t)shape[ch_dim];
}

BOAT_API boat_tensor_t* BOAT_CALL boat_prelu_layer_forward(const boat_prelu_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[PReLULayer] NULL input or layer\n");
        return NULL;
    }
    if (boat_tensor_dtype(input) != BOAT_DTYPE_FLOAT32) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[PReLULayer] only FLOAT32 input supported\n");
        return NULL;
    }

    // Cache input for the backward pass (cast away const: caching is an
    // implementation detail, not a user-visible mutation).
    boat_prelu_layer_t* mutable_layer = (boat_prelu_layer_t*)layer;
    if (mutable_layer->cache_input) boat_tensor_unref(mutable_layer->cache_input);
    mutable_layer->cache_input = (boat_tensor_t*)input;
    boat_tensor_ref(mutable_layer->cache_input);

    boat_tensor_t* result = boat_tensor_create_like(input);
    if (!result) return NULL;

    const float* in = (const float*)boat_tensor_const_data(input);
    float* out = (float*)boat_tensor_data(result);
    size_t n = boat_tensor_nelements(input);
    const int64_t* ishape = boat_tensor_shape(input);
    size_t ndim = boat_tensor_ndim(input);

    size_t channels = 1;
    if (layer->slope) channels = (size_t)boat_tensor_shape(layer->slope)[0];
    const float* slope_data = layer->slope ? (const float*)boat_tensor_const_data(layer->slope) : NULL;

    if (slope_data && channels > 1) {
        for (size_t i = 0; i < n; i++) {
            size_t c = channel_index(ishape, ndim, i);
            if (c >= channels) c = channels - 1;
            float x = in[i];
            out[i] = (x > 0.0f) ? x : slope_data[c] * x;
        }
    } else {
        float s = slope_data ? slope_data[0] : 0.25f;
        for (size_t i = 0; i < n; i++) {
            float x = in[i];
            out[i] = (x > 0.0f) ? x : s * x;
        }
    }

    return result;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_prelu_layer_backward(boat_prelu_layer_t* layer, const boat_tensor_t* grad_output) {
    if (!layer || !grad_output) return NULL;
    if (!layer->cache_input) {
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION, "[PReLULayer] backward requires a prior forward pass\n");
        return NULL;
    }
    if (boat_tensor_dtype(grad_output) != BOAT_DTYPE_FLOAT32) return NULL;

    const boat_tensor_t* input = layer->cache_input;
    if (boat_tensor_nelements(input) != boat_tensor_nelements(grad_output)) return NULL;

    size_t n = boat_tensor_nelements(input);
    const int64_t* ishape = boat_tensor_shape(input);
    size_t ndim = boat_tensor_ndim(input);

    size_t channels = 1;
    if (layer->slope) channels = (size_t)boat_tensor_shape(layer->slope)[0];
    const float* slope_data = layer->slope ? (const float*)boat_tensor_const_data(layer->slope) : NULL;

    // grad_slope accumulator (shape [C, 1, 1]).
    if (!layer->grad_slope) {
        int64_t gshape[] = {(int64_t)channels, 1, 1};
        layer->grad_slope = boat_tensor_create(gshape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!layer->grad_slope) return NULL;
    }
    float* grad_slope = (float*)boat_tensor_data(layer->grad_slope);
    size_t gs_n = boat_tensor_nelements(layer->grad_slope);
    memset(grad_slope, 0, gs_n * sizeof(float));

    boat_tensor_t* grad_input = boat_tensor_create_like(input);
    if (!grad_input) return NULL;
    float* gi = (float*)boat_tensor_data(grad_input);

    const float* x = (const float*)boat_tensor_const_data(input);
    const float* go = (const float*)boat_tensor_const_data(grad_output);

    if (slope_data && channels > 1) {
        for (size_t i = 0; i < n; i++) {
            size_t c = channel_index(ishape, ndim, i);
            if (c >= channels) c = channels - 1;
            float s = slope_data[c];
            gi[i] = (x[i] > 0.0f) ? go[i] : s * go[i];
            if (x[i] < 0.0f) grad_slope[c] += go[i] * x[i];
        }
    } else {
        float s = slope_data ? slope_data[0] : 0.25f;
        for (size_t i = 0; i < n; i++) {
            gi[i] = (x[i] > 0.0f) ? go[i] : s * go[i];
            if (x[i] < 0.0f) grad_slope[0] += go[i] * x[i];
        }
    }

    return grad_input;
}

BOAT_API void BOAT_CALL boat_prelu_layer_update(boat_prelu_layer_t* layer, float learning_rate) {
    if (!layer || !layer->slope || !layer->grad_slope) return;
    float* s = (float*)boat_tensor_data(layer->slope);
    float* gs = (float*)boat_tensor_data(layer->grad_slope);
    size_t n = boat_tensor_nelements(layer->slope);
    for (size_t i = 0; i < n; i++) s[i] -= learning_rate * gs[i];
}
