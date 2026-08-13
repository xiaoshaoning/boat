// quantize.c - Post-Training Quantization (PTQ) implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

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

// Packed type utility functions
#include <boat/packed.h>

BOAT_API boat_quant_config_t boat_quant_config_default(void) {
    boat_quant_config_t cfg;
    cfg.quant_dtype = BOAT_DTYPE_UINT8;
    cfg.symmetric = false;
    cfg.per_channel = false;
    return cfg;
}

BOAT_API void boat_compute_quant_params(float min_val, float max_val,
                                         boat_dtype_t quant_dtype,
                                         bool symmetric,
                                         float* out_scale,
                                         int32_t* out_zero_point) {
    // Degenerate case: all values identical
    if (max_val - min_val < 1e-10f) {
        float abs_val = fmaxf(fabsf(min_val), 1e-10f);
        if (quant_dtype == BOAT_DTYPE_INT8) {
            *out_scale = abs_val / 127.0f;
            *out_zero_point = 0;
        } else if (quant_dtype == BOAT_DTYPE_BITS2) {
            *out_scale = abs_val / 3.0f;
            *out_zero_point = 0;
        } else if (quant_dtype == BOAT_DTYPE_BITS1) {
            *out_scale = abs_val / 1.0f;
            *out_zero_point = 0;
        } else if (quant_dtype == BOAT_DTYPE_FLOAT4) {
            *out_scale = 1.0f;
            *out_zero_point = 0;
        } else {
            *out_scale = abs_val / 127.0f;
            *out_zero_point = 128;
        }
        return;
    }

    if (quant_dtype == BOAT_DTYPE_BITS2) {
        // BITS2: unsigned 2-bit, range [0, 3], qmax=3, asymmetric only
        *out_scale = (max_val - min_val) / 3.0f;
        float inv_scale = 1.0f / *out_scale;
        *out_zero_point = (int32_t)roundf(-min_val * inv_scale);
        return;
    }

    if (quant_dtype == BOAT_DTYPE_BITS1) {
        // BITS1: unsigned 1-bit, range [0, 1], qmax=1, asymmetric only
        *out_scale = (max_val - min_val) / 1.0f;
        float inv_scale = 1.0f / *out_scale;
        *out_zero_point = (int32_t)roundf(-min_val * inv_scale);
        return;
    }

    if (quant_dtype == BOAT_DTYPE_FLOAT4) {
        // FLOAT4: custom 4-bit float, no affine quantization
        *out_scale = 1.0f;
        *out_zero_point = 0;
        return;
    }

    if (quant_dtype == BOAT_DTYPE_INT8) {
        if (symmetric) {
            // Symmetric INT8: map [-abs_max, +abs_max] -> [-128, 127], zp = 0
            float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
            *out_scale = abs_max / 127.0f;
            *out_zero_point = 0;
        } else {
            // Asymmetric INT8: map [min_val, max_val] -> [-128, 127]
            *out_scale = (max_val - min_val) / 255.0f;
            float inv_scale = 1.0f / *out_scale;
            *out_zero_point = -128 - (int32_t)roundf(min_val * inv_scale);
            if (*out_zero_point < -128) *out_zero_point = -128;
            if (*out_zero_point > 127) *out_zero_point = 127;
        }
    } else {
        // UINT8 (default)
        if (symmetric) {
            // Symmetric UINT8: map [-abs_max, +abs_max] -> [0, 255], zp = 128
            float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
            *out_scale = abs_max / 127.0f;
            *out_zero_point = 128;
        } else {
            // Asymmetric UINT8: map [min_val, max_val] -> [0, 255]
            *out_scale = (max_val - min_val) / 255.0f;
            float inv_scale = 1.0f / *out_scale;
            *out_zero_point = (int32_t)roundf(-min_val * inv_scale);
        }
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

    // Create quantized output tensor with the requested dtype
    const int64_t* shape = boat_tensor_shape(fp32_tensor);
    size_t ndim = boat_tensor_ndim(fp32_tensor);
    boat_tensor_t* quantized = boat_tensor_create(shape, ndim, config->quant_dtype,
                                                   BOAT_DEVICE_CPU);
    if (!quantized) return NULL;

    // Set quantization parameters
    boat_tensor_set_quant_params(quantized, scale, zero_point);

    // Pass 2: quantize
    if (config->quant_dtype == BOAT_DTYPE_FLOAT4) {
        // FLOAT4: pack directly (no affine quantization)
        uint8_t* dst = (uint8_t*)boat_tensor_data(quantized);
        boat_pack_float4(src, dst, n);
        return quantized;
    }

    float inv_scale = 1.0f / scale;
    if (config->quant_dtype == BOAT_DTYPE_BITS2) {
        // BITS2: quantize to [0, 3], then pack
        uint8_t* unpacked = (uint8_t*)boat_malloc(sizeof(uint8_t) * n, BOAT_DEVICE_CPU);
        if (!unpacked) {
            boat_tensor_unref(quantized);
            return NULL;
        }
        for (size_t i = 0; i < n; i++) {
            int32_t q = (int32_t)roundf(src[i] * inv_scale) + zero_point;
            if (q < 0) q = 0;
            if (q > 3) q = 3;
            unpacked[i] = (uint8_t)q;
        }
        uint8_t* dst = (uint8_t*)boat_tensor_data(quantized);
        boat_pack_bits2(unpacked, dst, n);
        boat_free(unpacked);
        return quantized;
    }

    if (config->quant_dtype == BOAT_DTYPE_BITS1) {
        // BITS1: quantize to [0, 1], then pack
        bool* unpacked = (bool*)boat_malloc(sizeof(bool) * n, BOAT_DEVICE_CPU);
        if (!unpacked) {
            boat_tensor_unref(quantized);
            return NULL;
        }
        for (size_t i = 0; i < n; i++) {
            int32_t q = (int32_t)roundf(src[i] * inv_scale) + zero_point;
            if (q < 0) q = 0;
            if (q > 1) q = 1;
            unpacked[i] = (bool)q;
        }
        uint8_t* dst = (uint8_t*)boat_tensor_data(quantized);
        boat_pack_bits1(unpacked, dst, n);
        boat_free(unpacked);
        return quantized;
    }

    if (config->quant_dtype == BOAT_DTYPE_INT8) {
        int8_t* dst = (int8_t*)boat_tensor_data(quantized);
        for (size_t i = 0; i < n; i++) {
            int32_t q = (int32_t)roundf(src[i] * inv_scale) + zero_point;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            dst[i] = (int8_t)q;
        }
    } else {
        uint8_t* dst = (uint8_t*)boat_tensor_data(quantized);
        for (size_t i = 0; i < n; i++) {
            int32_t q = (int32_t)roundf(src[i] * inv_scale) + zero_point;
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            dst[i] = (uint8_t)q;
        }
    }

    return quantized;
}

BOAT_API boat_tensor_t* boat_dequantize_tensor(const boat_tensor_t* quantized_tensor) {
    if (!quantized_tensor) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Dequantize] NULL argument\n");
        return NULL;
    }
    boat_dtype_t dt = boat_tensor_dtype(quantized_tensor);
    if (dt != BOAT_DTYPE_UINT8 && dt != BOAT_DTYPE_INT8 &&
        dt != BOAT_DTYPE_BITS2 && dt != BOAT_DTYPE_BITS1 &&
        dt != BOAT_DTYPE_FLOAT4) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[Dequantize] expected UINT8/INT8/BITS2/BITS1/FLOAT4 tensor, got %s\n",
                        boat_dtype_name(dt));
        return NULL;
    }

    // FLOAT4 doesn't use affine quantization
    if (dt == BOAT_DTYPE_FLOAT4) {
        size_t n = boat_tensor_nelements(quantized_tensor);
        const int64_t* shape = boat_tensor_shape(quantized_tensor);
        size_t ndim = boat_tensor_ndim(quantized_tensor);
        boat_tensor_t* fp32 = boat_tensor_create(shape, ndim, BOAT_DTYPE_FLOAT32,
                                                  BOAT_DEVICE_CPU);
        if (!fp32) return NULL;
        const uint8_t* src = (const uint8_t*)boat_tensor_const_data(quantized_tensor);
        float* dst = (float*)boat_tensor_data(fp32);
        boat_unpack_float4(src, dst, n);
        return fp32;
    }

    float scale = boat_tensor_get_scale(quantized_tensor);
    if (scale == 0.0f) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[Dequantize] tensor has scale=0 (not quantized)\n");
        return NULL;
    }
    int32_t zero_point = boat_tensor_get_zero_point(quantized_tensor);

    size_t n = boat_tensor_nelements(quantized_tensor);
    const int64_t* shape = boat_tensor_shape(quantized_tensor);
    size_t ndim = boat_tensor_ndim(quantized_tensor);
    boat_tensor_t* fp32 = boat_tensor_create(shape, ndim, BOAT_DTYPE_FLOAT32,
                                              BOAT_DEVICE_CPU);
    if (!fp32) return NULL;

    float* dst = (float*)boat_tensor_data(fp32);

    if (dt == BOAT_DTYPE_BITS2) {
        // Unpack BITS2 then apply affine dequantize
        uint8_t* unpacked = (uint8_t*)boat_malloc(sizeof(uint8_t) * n, BOAT_DEVICE_CPU);
        if (!unpacked) {
            boat_tensor_unref(fp32);
            return NULL;
        }
        const uint8_t* src = (const uint8_t*)boat_tensor_const_data(quantized_tensor);
        boat_unpack_bits2(src, unpacked, n);
        for (size_t i = 0; i < n; i++) {
            dst[i] = ((int32_t)unpacked[i] - zero_point) * scale;
        }
        boat_free(unpacked);
    } else if (dt == BOAT_DTYPE_BITS1) {
        // Unpack BITS1 then apply affine dequantize
        bool* unpacked = (bool*)boat_malloc(sizeof(bool) * n, BOAT_DEVICE_CPU);
        if (!unpacked) {
            boat_tensor_unref(fp32);
            return NULL;
        }
        const uint8_t* src = (const uint8_t*)boat_tensor_const_data(quantized_tensor);
        boat_unpack_bits1(src, unpacked, n);
        for (size_t i = 0; i < n; i++) {
            dst[i] = ((int32_t)unpacked[i] - zero_point) * scale;
        }
        boat_free(unpacked);
    } else if (dt == BOAT_DTYPE_INT8) {
        const int8_t* src = (const int8_t*)boat_tensor_const_data(quantized_tensor);
        for (size_t i = 0; i < n; i++) {
            dst[i] = ((int32_t)src[i] - zero_point) * scale;
        }
    } else {
        const uint8_t* src = (const uint8_t*)boat_tensor_const_data(quantized_tensor);
        for (size_t i = 0; i < n; i++) {
            dst[i] = ((int32_t)src[i] - zero_point) * scale;
        }
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

    boat_tensor_t* q;
    if (config->per_channel) {
        // Determine channel dimension based on layer type
        // Dense: weight shape [input_features, output_features], channel_dim = 1
        // Conv2D: weight shape [out_channels, in_channels, kH, KW], channel_dim = 0
        size_t channel_dim = (type == BOAT_LAYER_TYPE_DENSE) ? 1 : 0;
        q = boat_quantize_tensor_per_channel(weight, config, channel_dim);
    } else {
        q = boat_quantize_tensor(weight, config);
    }
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

    // Only dequantize quantized tensors with scale != 0 (FLOAT4 always has scale=1)
    boat_dtype_t wdt = boat_tensor_dtype(weight);
    bool is_quantized = (wdt == BOAT_DTYPE_UINT8 || wdt == BOAT_DTYPE_INT8 ||
                         wdt == BOAT_DTYPE_BITS2 || wdt == BOAT_DTYPE_BITS1 ||
                         wdt == BOAT_DTYPE_FLOAT4);
    if (!is_quantized) {
        return true; // not quantized, skip
    }
    if (wdt != BOAT_DTYPE_FLOAT4 && boat_tensor_get_scale(weight) == 0.0f) {
        return true; // quantized dtype but scale=0 (not actually quantized)
    }

    boat_tensor_t* fp32;
    if (boat_tensor_is_per_channel(weight)) {
        fp32 = boat_dequantize_tensor_per_channel(weight);
    } else {
        fp32 = boat_dequantize_tensor(weight);
    }
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

// ---------------------------------------------------------------------------
// Per-channel quantization
// ---------------------------------------------------------------------------

BOAT_API boat_tensor_t* boat_quantize_tensor_per_channel(
    const boat_tensor_t* fp32_tensor,
    const boat_quant_config_t* config,
    size_t channel_dim) {
    if (!fp32_tensor || !config) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[QuantizePerChannel] NULL argument\n");
        return NULL;
    }
    if (boat_tensor_dtype(fp32_tensor) != BOAT_DTYPE_FLOAT32) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[QuantizePerChannel] expected FLOAT32 tensor, got %s\n",
                        boat_dtype_name(boat_tensor_dtype(fp32_tensor)));
        return NULL;
    }

    size_t ndim = boat_tensor_ndim(fp32_tensor);
    const int64_t* shape = boat_tensor_shape(fp32_tensor);

    if (channel_dim >= ndim) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[QuantizePerChannel] channel_dim %zu out of range (ndim=%zu)\n",
                        channel_dim, ndim);
        return NULL;
    }

    size_t n_channels = (size_t)shape[channel_dim];
    size_t outer_elements = 1;
    for (size_t i = 0; i < channel_dim; i++) {
        outer_elements *= shape[i];
    }
    size_t inner_elements = 1;
    for (size_t i = channel_dim + 1; i < ndim; i++) {
        inner_elements *= shape[i];
    }
    size_t channel_stride = inner_elements;
    (void)channel_stride;
    size_t total_elements = boat_tensor_nelements(fp32_tensor);

    const float* src = (const float*)boat_tensor_const_data(fp32_tensor);

    // Compute per-channel scale/zero_point
    float* scales = (float*)boat_malloc(sizeof(float) * n_channels, BOAT_DEVICE_CPU);
    int32_t* zero_points = (int32_t*)boat_malloc(sizeof(int32_t) * n_channels, BOAT_DEVICE_CPU);
    if (!scales || !zero_points) {
        boat_free(scales);
        boat_free(zero_points);
        return NULL;
    }

    for (size_t c = 0; c < n_channels; c++) {
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        for (size_t o = 0; o < outer_elements; o++) {
            for (size_t i = 0; i < inner_elements; i++) {
                size_t idx = (o * n_channels + c) * inner_elements + i;
                float v = src[idx];
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
        }
        boat_compute_quant_params(min_val, max_val, config->quant_dtype,
                                  config->symmetric, &scales[c], &zero_points[c]);
    }

    // Create quantized output tensor
    boat_tensor_t* quantized = boat_tensor_create(shape, ndim, config->quant_dtype,
                                                   BOAT_DEVICE_CPU);
    if (!quantized) {
        boat_free(scales);
        boat_free(zero_points);
        return NULL;
    }

    // Store per-channel parameters
    boat_tensor_set_per_channel_quant_params(quantized, scales, zero_points, n_channels);
    // Also set per-tensor params to first channel's params for backward compat
    boat_tensor_set_quant_params(quantized, scales[0], zero_points[0]);

    // Quantize each channel
    if (config->quant_dtype == BOAT_DTYPE_FLOAT4) {
        // FLOAT4: pack each channel's data separately
        uint8_t* dst = (uint8_t*)boat_tensor_data(quantized);
        // For FLOAT4, we pack element by element, so we can just pack the whole tensor
        // FLOAT4 packing is element-wise, not affected by per-channel structure
        // But we still want per-channel scales for potential dequantization flexibility
        for (size_t c = 0; c < n_channels; c++) {
            // Copy each channel's float data, then pack it
            // Since boat_pack_float4 works on the full tensor, we need to
            // do it per-channel for correctness if channels have different scale
            // Actually FLOAT4 doesn't use scale, it's a direct float conversion
            // So packing is fine regardless
        }
        // Pack entire tensor at once (FLOAT4 is a direct format, no affine step)
        boat_pack_float4(src, dst, total_elements);
    } else {
        void* dst = boat_tensor_data(quantized);

        if (config->quant_dtype == BOAT_DTYPE_BITS2) {
            // BITS2: quantize to [0,3] then pack
            uint8_t* unpacked = (uint8_t*)boat_malloc(sizeof(uint8_t) * total_elements, BOAT_DEVICE_CPU);
            if (!unpacked) {
                boat_tensor_unref(quantized);
                boat_free(scales);
                boat_free(zero_points);
                return NULL;
            }
            for (size_t c = 0; c < n_channels; c++) {
                float inv_scale = 1.0f / scales[c];
                int32_t zp = zero_points[c];
                for (size_t o = 0; o < outer_elements; o++) {
                    for (size_t i = 0; i < inner_elements; i++) {
                        size_t idx = (o * n_channels + c) * inner_elements + i;
                        int32_t q = (int32_t)roundf(src[idx] * inv_scale) + zp;
                        if (q < 0) q = 0;
                        if (q > 3) q = 3;
                        unpacked[idx] = (uint8_t)q;
                    }
                }
            }
            boat_pack_bits2(unpacked, (uint8_t*)dst, total_elements);
            boat_free(unpacked);
        } else if (config->quant_dtype == BOAT_DTYPE_BITS1) {
            // BITS1: quantize to [0,1] then pack
            bool* unpacked = (bool*)boat_malloc(sizeof(bool) * total_elements, BOAT_DEVICE_CPU);
            if (!unpacked) {
                boat_tensor_unref(quantized);
                boat_free(scales);
                boat_free(zero_points);
                return NULL;
            }
            for (size_t c = 0; c < n_channels; c++) {
                float inv_scale = 1.0f / scales[c];
                int32_t zp = zero_points[c];
                for (size_t o = 0; o < outer_elements; o++) {
                    for (size_t i = 0; i < inner_elements; i++) {
                        size_t idx = (o * n_channels + c) * inner_elements + i;
                        int32_t q = (int32_t)roundf(src[idx] * inv_scale) + zp;
                        if (q < 0) q = 0;
                        if (q > 1) q = 1;
                        unpacked[idx] = (bool)q;
                    }
                }
            }
            boat_pack_bits1(unpacked, (uint8_t*)dst, total_elements);
            boat_free(unpacked);
        } else if (config->quant_dtype == BOAT_DTYPE_INT8) {
            int8_t* dst_i8 = (int8_t*)dst;
            for (size_t c = 0; c < n_channels; c++) {
                float inv_scale = 1.0f / scales[c];
                int32_t zp = zero_points[c];
                for (size_t o = 0; o < outer_elements; o++) {
                    for (size_t i = 0; i < inner_elements; i++) {
                        size_t idx = (o * n_channels + c) * inner_elements + i;
                        int32_t q = (int32_t)roundf(src[idx] * inv_scale) + zp;
                        if (q < -128) q = -128;
                        if (q > 127) q = 127;
                        dst_i8[idx] = (int8_t)q;
                    }
                }
            }
        } else {
            uint8_t* dst_u8 = (uint8_t*)dst;
            for (size_t c = 0; c < n_channels; c++) {
                float inv_scale = 1.0f / scales[c];
                int32_t zp = zero_points[c];
                for (size_t o = 0; o < outer_elements; o++) {
                    for (size_t i = 0; i < inner_elements; i++) {
                        size_t idx = (o * n_channels + c) * inner_elements + i;
                        int32_t q = (int32_t)roundf(src[idx] * inv_scale) + zp;
                        if (q < 0) q = 0;
                        if (q > 255) q = 255;
                        dst_u8[idx] = (uint8_t)q;
                    }
                }
            }
        }
    }

    boat_free(scales);
    boat_free(zero_points);
    return quantized;
}

BOAT_API boat_tensor_t* boat_dequantize_tensor_per_channel(const boat_tensor_t* quantized_tensor) {
    if (!quantized_tensor) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[DequantizePerChannel] NULL argument\n");
        return NULL;
    }
    if (!boat_tensor_is_per_channel(quantized_tensor)) {
        // Fall back to regular dequantize if not per-channel
        return boat_dequantize_tensor(quantized_tensor);
    }

    boat_dtype_t dt = boat_tensor_dtype(quantized_tensor);
    size_t n_channels = boat_tensor_num_channels(quantized_tensor);
    const float* scales = boat_tensor_get_scales(quantized_tensor);
    const int32_t* zero_points = boat_tensor_get_zero_points(quantized_tensor);

    if (!scales || !zero_points || n_channels == 0) {
        return boat_dequantize_tensor(quantized_tensor);
    }

    // Determine channel dimension from the stored structure
    // We need to reconstruct the channel layout. The per-channel quantize
    // stores channels along the last dimension that has size == n_channels.
    // But actually, we need to know the channel_dim. For now, we infer it
    // by assuming the stored data matches the shape and we iterate channels
    // along channel_dim. We'll use a heuristic: find the first dimension
    // that matches n_channels.
    size_t ndim = boat_tensor_ndim(quantized_tensor);
    const int64_t* shape = boat_tensor_shape(quantized_tensor);
    size_t channel_dim = 0;
    while (channel_dim < ndim && (size_t)shape[channel_dim] != n_channels) {
        channel_dim++;
    }
    if (channel_dim >= ndim) {
        // Can't determine channel_dim, fall back to per-tensor
        return boat_dequantize_tensor(quantized_tensor);
    }

    size_t outer_elements = 1;
    for (size_t i = 0; i < channel_dim; i++) {
        outer_elements *= shape[i];
    }
    size_t inner_elements = 1;
    for (size_t i = channel_dim + 1; i < ndim; i++) {
        inner_elements *= shape[i];
    }
    size_t total_elements = boat_tensor_nelements(quantized_tensor);

    const int64_t* fp32_shape = boat_tensor_shape(quantized_tensor);
    size_t fp32_ndim = boat_tensor_ndim(quantized_tensor);
    boat_tensor_t* fp32 = boat_tensor_create(fp32_shape, fp32_ndim, BOAT_DTYPE_FLOAT32,
                                              BOAT_DEVICE_CPU);
    if (!fp32) return NULL;

    float* dst = (float*)boat_tensor_data(fp32);

    if (dt == BOAT_DTYPE_FLOAT4) {
        const uint8_t* src = (const uint8_t*)boat_tensor_const_data(quantized_tensor);
        boat_unpack_float4(src, dst, total_elements);
        return fp32;
    }

    if (dt == BOAT_DTYPE_BITS2) {
        uint8_t* unpacked = (uint8_t*)boat_malloc(sizeof(uint8_t) * total_elements, BOAT_DEVICE_CPU);
        if (!unpacked) {
            boat_tensor_unref(fp32);
            return NULL;
        }
        const uint8_t* src = (const uint8_t*)boat_tensor_const_data(quantized_tensor);
        boat_unpack_bits2(src, unpacked, total_elements);
        for (size_t c = 0; c < n_channels; c++) {
            float s = scales[c];
            int32_t zp = zero_points[c];
            for (size_t o = 0; o < outer_elements; o++) {
                for (size_t i = 0; i < inner_elements; i++) {
                    size_t idx = (o * n_channels + c) * inner_elements + i;
                    dst[idx] = ((int32_t)unpacked[idx] - zp) * s;
                }
            }
        }
        boat_free(unpacked);
        return fp32;
    }

    if (dt == BOAT_DTYPE_BITS1) {
        bool* unpacked = (bool*)boat_malloc(sizeof(bool) * total_elements, BOAT_DEVICE_CPU);
        if (!unpacked) {
            boat_tensor_unref(fp32);
            return NULL;
        }
        const uint8_t* src = (const uint8_t*)boat_tensor_const_data(quantized_tensor);
        boat_unpack_bits1(src, unpacked, total_elements);
        for (size_t c = 0; c < n_channels; c++) {
            float s = scales[c];
            int32_t zp = zero_points[c];
            for (size_t o = 0; o < outer_elements; o++) {
                for (size_t i = 0; i < inner_elements; i++) {
                    size_t idx = (o * n_channels + c) * inner_elements + i;
                    dst[idx] = ((int32_t)unpacked[idx] - zp) * s;
                }
            }
        }
        boat_free(unpacked);
        return fp32;
    }

    if (dt == BOAT_DTYPE_INT8) {
        const int8_t* src = (const int8_t*)boat_tensor_const_data(quantized_tensor);
        for (size_t c = 0; c < n_channels; c++) {
            float s = scales[c];
            int32_t zp = zero_points[c];
            for (size_t o = 0; o < outer_elements; o++) {
                for (size_t i = 0; i < inner_elements; i++) {
                    size_t idx = (o * n_channels + c) * inner_elements + i;
                    dst[idx] = ((int32_t)src[idx] - zp) * s;
                }
            }
        }
    } else {
        const uint8_t* src = (const uint8_t*)boat_tensor_const_data(quantized_tensor);
        for (size_t c = 0; c < n_channels; c++) {
            float s = scales[c];
            int32_t zp = zero_points[c];
            for (size_t o = 0; o < outer_elements; o++) {
                for (size_t i = 0; i < inner_elements; i++) {
                    size_t idx = (o * n_channels + c) * inner_elements + i;
                    dst[idx] = ((int32_t)src[idx] - zp) * s;
                }
            }
        }
    }

    return fp32;
}

// ---------------------------------------------------------------------------
// QAT: Fake quantization (quantize -> dequantize in-place, tensor stays FP32)
// ---------------------------------------------------------------------------

BOAT_API bool boat_fake_quantize(boat_tensor_t* tensor, const boat_quant_config_t* config) {
    if (!tensor || !config) return false;
    if (boat_tensor_dtype(tensor) != BOAT_DTYPE_FLOAT32) return false;

    // Use per-tensor quantize then dequantize
    boat_tensor_t* quantized = boat_quantize_tensor(tensor, config);
    if (!quantized) return false;

    boat_tensor_t* dequantized = boat_dequantize_tensor(quantized);
    if (!dequantized) {
        boat_tensor_unref(quantized);
        return false;
    }

    // Copy dequantized values back to original tensor
    size_t nbytes = boat_tensor_nbytes(tensor);
    memcpy(boat_tensor_data(tensor), boat_tensor_data(dequantized), nbytes);

    boat_tensor_unref(quantized);
    boat_tensor_unref(dequantized);
    return true;
}

BOAT_API bool boat_fake_quantize_per_channel(boat_tensor_t* tensor,
                                              const boat_quant_config_t* config,
                                              size_t channel_dim) {
    if (!tensor || !config) return false;
    if (boat_tensor_dtype(tensor) != BOAT_DTYPE_FLOAT32) return false;

    boat_tensor_t* quantized = boat_quantize_tensor_per_channel(tensor, config, channel_dim);
    if (!quantized) return false;

    boat_tensor_t* dequantized = boat_dequantize_tensor_per_channel(quantized);
    if (!dequantized) {
        boat_tensor_unref(quantized);
        return false;
    }

    size_t nbytes = boat_tensor_nbytes(tensor);
    memcpy(boat_tensor_data(tensor), boat_tensor_data(dequantized), nbytes);

    boat_tensor_unref(quantized);
    boat_tensor_unref(dequantized);
    return true;
}
