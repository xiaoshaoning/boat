// nougat_model.h - Nougat-LaTeX weight container and loader
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef NOUGAT_MODEL_H
#define NOUGAT_MODEL_H

#include <boat/tensor.h>
#include <boat/layers/swin.h>
#include <boat/layers/transformer_decoder.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // ----- Decoder (mBART) -----
    boat_decoder_config_t decoder_config;
    int num_decoder_layers;

    // Embedding + final layers
    boat_tensor_t* embed_tokens_weight;        // [50000, 1024]
    boat_tensor_t* embed_positions_weight;     // [4096, 1024]
    boat_tensor_t* layernorm_embedding_weight; // [1024]
    boat_tensor_t* layernorm_embedding_bias;   // [1024]
    boat_tensor_t* final_layer_norm_weight;    // [1024]
    boat_tensor_t* final_layer_norm_bias;      // [1024]
    boat_tensor_t* lm_head_weight;             // [50000, 1024] (not tied)

    // Decoder layers (10)
    boat_decoder_layer_weights_t** decoder_layers;

    // ----- Encoder (Donut-Swin) -----
    boat_swin_config_t swin_config;
    boat_swin_weights_t* encoder;
} nougat_model_t;

// Load all weights from model.safetensors in model_dir.
// Returns NULL on failure. All weights are FP32 on CPU.
// Use nougat_model_to_device() to move to GPU.
nougat_model_t* nougat_model_create(const char* model_dir);

// Free all weight tensors and the model struct.
void nougat_model_free(nougat_model_t* model);

// Move all weight tensors to the specified device.
int nougat_model_to_device(nougat_model_t* model, boat_device_t device);

#ifdef __cplusplus
}
#endif

#endif // NOUGAT_MODEL_H
