// quantize.h - Post-Training Quantization (PTQ) API
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_QUANTIZE_H
#define BOAT_QUANTIZE_H

#include "tensor.h"
#include "export.h"

// Forward declarations
typedef struct boat_model_t boat_model_t;

#ifdef __cplusplus
extern "C" {
#endif

// Quantization configuration
typedef struct {
    boat_dtype_t quant_dtype;   // Target dtype (BOAT_DTYPE_UINT8 or BOAT_DTYPE_INT8)
    bool symmetric;             // true -> zero_point forced to 128 (UINT8) or 0 (INT8)
} boat_quant_config_t;

// Default config: UINT8, asymmetric
BOAT_API boat_quant_config_t boat_quant_config_default(void);

// Compute scale and zero_point from observed min/max values.
// For UINT8 asymmetric: scale = (max - min) / 255, zp = round(-min / scale)
// For UINT8 symmetric:   scale = max(|max|,|min|) / 127, zp = 128
// For INT8 symmetric:    scale = max(|max|,|min|) / 127, zp = 0
// For INT8 asymmetric:   scale = (max - min) / 255, zp = round(-min / scale), clamped to [-128, 127]
BOAT_API void boat_compute_quant_params(float min_val, float max_val,
                                         boat_dtype_t quant_dtype, bool symmetric,
                                         float* out_scale, int32_t* out_zero_point);

// Quantize a FP32 tensor to UINT8 or INT8 with per-tensor affine quantization.
// Returns a new tensor with dtype=config->quant_dtype, scale and zero_point set.
// Caller owns the returned tensor.
BOAT_API boat_tensor_t* boat_quantize_tensor(const boat_tensor_t* fp32_tensor,
                                              const boat_quant_config_t* config);

// Dequantize a UINT8 or INT8 tensor back to FP32.
// The input tensor must have dtype=UINT8 or INT8 and scale != 0.
// Returns a new FP32 tensor. Caller owns the returned tensor.
BOAT_API boat_tensor_t* boat_dequantize_tensor(const boat_tensor_t* quantized_tensor);

// Model-level quantization: replace all trainable weight tensors (Dense, Conv2D)
// with quantized UINT8/INT8 equivalents. Bias tensors are left as FP32.
// Returns true on success, false on error (error message set via boat_set_errorf).
BOAT_API bool boat_model_quantize(boat_model_t* model, const boat_quant_config_t* config);

// Model-level dequantization: restore all quantized weight tensors back to FP32.
// Returns true on success.
BOAT_API bool boat_model_dequantize(boat_model_t* model);

// Calibration data structure for activation range collection
typedef struct boat_calibration_data_t {
    size_t num_layers;
    float* layer_min;   // array[num_layers], initialized to +inf
    float* layer_max;   // array[num_layers], initialized to -inf
} boat_calibration_data_t;

// Create calibration data for a model with num_layers layers.
// All ranges are initialized to [+inf, -inf].
BOAT_API boat_calibration_data_t* boat_calibration_create(size_t num_layers);

// Observe an activation tensor during calibration forward pass.
// Updates min/max for the given layer index.
BOAT_API void boat_calibration_observe(boat_calibration_data_t* calib,
                                        size_t layer_index,
                                        const boat_tensor_t* activation);

// Get the observed range for a layer. Returns false if no observations yet.
BOAT_API bool boat_calibration_get_range(const boat_calibration_data_t* calib,
                                          size_t layer_index,
                                          float* out_min, float* out_max);

// Free calibration data.
BOAT_API void boat_calibration_free(boat_calibration_data_t* calib);

#ifdef __cplusplus
}
#endif

#endif // BOAT_QUANTIZE_H
