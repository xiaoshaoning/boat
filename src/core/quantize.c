// quantize.c - Post-Training Quantization (PTQ) implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#define BOAT_BUILDING_DLL
#include <boat/quantize.h>
#include <boat/tensor.h>
#include <boat/layers.h>
#include <boat/model.h>
#include <boat.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

BOAT_API boat_quant_config_t boat_quant_config_default(void) {
    boat_quant_config_t cfg;
    cfg.quant_dtype = BOAT_DTYPE_UINT8;
    cfg.symmetric = false;
    return cfg;
}

BOAT_API void boat_compute_quant_params(float min_val, float max_val,
                                         boat_dtype_t quant_dtype,
                                         bool symmetric,
                                         float* out_scale,
                                         int32_t* out_zero_point) {
    (void)quant_dtype; // only UINT8 supported for now

    // Degenerate case: all values identical
    if (max_val - min_val < 1e-10f) {
        // Use symmetric-style centered quantization
        float abs_val = fmaxf(fabsf(min_val), 1e-10f);
        *out_scale = abs_val / 127.0f;
        *out_zero_point = 128;
        return;
    }

    if (symmetric) {
        // Symmetric UINT8: map [-abs_max, +abs_max] -> [0, 255], zp = 128
        float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
        *out_scale = abs_max / 127.0f;
        *out_zero_point = 128;
    } else {
        // Asymmetric UINT8: map [min_val, max_val] -> [0, 255]
        // zp can be negative or > 255 — the q values will still be in [0,255].
        *out_scale = (max_val - min_val) / 255.0f;
        float inv_scale = 1.0f / *out_scale;
        *out_zero_point = (int32_t)roundf(-min_val * inv_scale);
    }
}

// ---------------------------------------------------------------------------
// Tensor-level quantize / dequantize
// ---------------------------------------------------------------------------

BOAT_API boat_tensor_t* boat_quantize_tensor(const boat_tensor_t* fp32_tensor,
                                              const boat_quant_config_t* config) {
    if (!fp32_tensor || !config) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Quantize] NULL argument\n");
        return NULL;
    }
    if (boat_tensor_dtype(fp32_tensor) != BOAT_DTYPE_FLOAT32) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[Quantize] expected FLOAT32 tensor, got %s\n",
                        boat_dtype_name(boat_tensor_dtype(fp32_tensor)));
        return NULL;
    }

    const float* src = (const float*)boat_tensor_const_data(fp32_tensor);
    size_t n = boat_tensor_nelements(fp32_tensor);

    // Pass 1: find min/max
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        if (src[i] < min_val) min_val = src[i];
        if (src[i] > max_val) max_val = src[i];
    }

    // Compute scale / zero_point
    float scale;
    int32_t zero_point;
    boat_compute_quant_params(min_val, max_val, config->quant_dtype,
                              config->symmetric, &scale, &zero_point);

    // Create UINT8 output tensor with same shape
    const int64_t* shape = boat_tensor_shape(fp32_tensor);
    size_t ndim = boat_tensor_ndim(fp32_tensor);
    boat_tensor_t* quantized = boat_tensor_create(shape, ndim, BOAT_DTYPE_UINT8,
                                                   BOAT_DEVICE_CPU);
    if (!quantized) return NULL;

    // Set quantization parameters
    boat_tensor_set_quant_params(quantized, scale, zero_point);

    // Pass 2: quantize
    uint8_t* dst = (uint8_t*)boat_tensor_data(quantized);
    float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < n; i++) {
        int32_t q = (int32_t)roundf(src[i] * inv_scale) + zero_point;
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        dst[i] = (uint8_t)q;
    }

    return quantized;
}

BOAT_API boat_tensor_t* boat_dequantize_tensor(const boat_tensor_t* quantized_tensor) {
    if (!quantized_tensor) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Dequantize] NULL argument\n");
        return NULL;
    }
    if (boat_tensor_dtype(quantized_tensor) != BOAT_DTYPE_UINT8) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[Dequantize] expected UINT8 tensor, got %s\n",
                        boat_dtype_name(boat_tensor_dtype(quantized_tensor)));
        return NULL;
    }

    float scale = boat_tensor_get_scale(quantized_tensor);
    if (scale == 0.0f) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[Dequantize] tensor has scale=0 (not quantized)\n");
        return NULL;
    }
    int32_t zero_point = boat_tensor_get_zero_point(quantized_tensor);

    const uint8_t* src = (const uint8_t*)boat_tensor_const_data(quantized_tensor);
    size_t n = boat_tensor_nelements(quantized_tensor);

    const int64_t* shape = boat_tensor_shape(quantized_tensor);
    size_t ndim = boat_tensor_ndim(quantized_tensor);
    boat_tensor_t* fp32 = boat_tensor_create(shape, ndim, BOAT_DTYPE_FLOAT32,
                                              BOAT_DEVICE_CPU);
    if (!fp32) return NULL;

    float* dst = (float*)boat_tensor_data(fp32);
    for (size_t i = 0; i < n; i++) {
        dst[i] = ((int32_t)src[i] - zero_point) * scale;
    }

    return fp32;
}

// ---------------------------------------------------------------------------
// Model-level quantization helpers
// ---------------------------------------------------------------------------

static bool is_weight_quantizable(boat_layer_type_t type) {
    return type == BOAT_LAYER_TYPE_DENSE || type == BOAT_LAYER_TYPE_CONV2D;
}

static bool quantize_layer_weight(void* layer_data, boat_layer_type_t type,
                                   const boat_quant_config_t* config) {
    boat_tensor_t* weight = NULL;
    boat_tensor_t* (*get_weight)(const void*) = NULL;
    void (*set_weight)(void*, boat_tensor_t*) = NULL;

    switch (type) {
        case BOAT_LAYER_TYPE_DENSE:
            get_weight = (void*)(void*)boat_dense_layer_get_weight;
            set_weight = (void*)(void*)boat_dense_layer_set_weight;
            break;
        case BOAT_LAYER_TYPE_CONV2D:
            get_weight = (void*)(void*)boat_conv_layer_get_weight;
            set_weight = (void*)(void*)boat_conv_layer_set_weight;
            break;
        default:
            return false;
    }

    weight = get_weight(layer_data);
    if (!weight) return false;

    // Skip if already quantized
    if (boat_tensor_dtype(weight) != BOAT_DTYPE_FLOAT32) {
        return true; // already quantized or not FP32, skip
    }

    boat_tensor_t* q = boat_quantize_tensor(weight, config);
    if (!q) return false;

    set_weight(layer_data, q);
    boat_tensor_unref(q); // set_weight takes a ref, drop ours
    return true;
}

static bool dequantize_layer_weight(void* layer_data, boat_layer_type_t type) {
    boat_tensor_t* weight = NULL;
    boat_tensor_t* (*get_weight)(const void*) = NULL;
    void (*set_weight)(void*, boat_tensor_t*) = NULL;

    switch (type) {
        case BOAT_LAYER_TYPE_DENSE:
            get_weight = (void*)(void*)boat_dense_layer_get_weight;
            set_weight = (void*)(void*)boat_dense_layer_set_weight;
            break;
        case BOAT_LAYER_TYPE_CONV2D:
            get_weight = (void*)(void*)boat_conv_layer_get_weight;
            set_weight = (void*)(void*)boat_conv_layer_set_weight;
            break;
        default:
            return false;
    }

    weight = get_weight(layer_data);
    if (!weight) return false;

    // Only dequantize UINT8 tensors with scale != 0
    if (boat_tensor_dtype(weight) != BOAT_DTYPE_UINT8 ||
        boat_tensor_get_scale(weight) == 0.0f) {
        return true; // not quantized, skip
    }

    boat_tensor_t* fp32 = boat_dequantize_tensor(weight);
    if (!fp32) return false;

    set_weight(layer_data, fp32);
    boat_tensor_unref(fp32);
    return true;
}

BOAT_API bool boat_model_quantize(boat_model_t* model,
                                   const boat_quant_config_t* config) {
    if (!model || !config) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ModelQuantize] NULL argument\n");
        return false;
    }

    size_t n = boat_model_layer_count(model);
    for (size_t i = 0; i < n; i++) {
        boat_layer_t* layer = boat_model_get_layer(model, i);
        if (!layer || !is_weight_quantizable(layer->type)) continue;

        if (!quantize_layer_weight(layer->data, layer->type, config)) {
            return false;
        }
    }
    return true;
}

BOAT_API bool boat_model_dequantize(boat_model_t* model) {
    if (!model) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ModelDequantize] NULL argument\n");
        return false;
    }

    size_t n = boat_model_layer_count(model);
    for (size_t i = 0; i < n; i++) {
        boat_layer_t* layer = boat_model_get_layer(model, i);
        if (!layer || !is_weight_quantizable(layer->type)) continue;

        if (!dequantize_layer_weight(layer->data, layer->type)) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

BOAT_API boat_calibration_data_t* boat_calibration_create(size_t num_layers) {
    boat_calibration_data_t* calib =
        (boat_calibration_data_t*)boat_malloc(sizeof(boat_calibration_data_t),
                                               BOAT_DEVICE_CPU);
    if (!calib) return NULL;

    calib->num_layers = num_layers;
    calib->layer_min = (float*)boat_malloc(sizeof(float) * num_layers,
                                           BOAT_DEVICE_CPU);
    calib->layer_max = (float*)boat_malloc(sizeof(float) * num_layers,
                                           BOAT_DEVICE_CPU);
    if (!calib->layer_min || !calib->layer_max) {
        boat_free(calib->layer_min);
        boat_free(calib->layer_max);
        boat_free(calib);
        return NULL;
    }

    for (size_t i = 0; i < num_layers; i++) {
        calib->layer_min[i] = FLT_MAX;
        calib->layer_max[i] = -FLT_MAX;
    }

    return calib;
}

BOAT_API void boat_calibration_observe(boat_calibration_data_t* calib,
                                        size_t layer_index,
                                        const boat_tensor_t* activation) {
    if (!calib || !activation) return;
    if (layer_index >= calib->num_layers) return;
    if (boat_tensor_dtype(activation) != BOAT_DTYPE_FLOAT32) return;

    const float* data = (const float*)boat_tensor_const_data(activation);
    size_t n = boat_tensor_nelements(activation);
    float* layer_min = &calib->layer_min[layer_index];
    float* layer_max = &calib->layer_max[layer_index];

    for (size_t i = 0; i < n; i++) {
        if (data[i] < *layer_min) *layer_min = data[i];
        if (data[i] > *layer_max) *layer_max = data[i];
    }
}

BOAT_API bool boat_calibration_get_range(const boat_calibration_data_t* calib,
                                          size_t layer_index,
                                          float* out_min, float* out_max) {
    if (!calib || !out_min || !out_max) return false;
    if (layer_index >= calib->num_layers) return false;
    if (calib->layer_min[layer_index] == FLT_MAX) return false; // no observations

    *out_min = calib->layer_min[layer_index];
    *out_max = calib->layer_max[layer_index];
    return true;
}

BOAT_API void boat_calibration_free(boat_calibration_data_t* calib) {
    if (calib) {
        boat_free(calib->layer_min);
        boat_free(calib->layer_max);
        boat_free(calib);
    }
}
