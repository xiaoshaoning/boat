// weights.h — Qwen3-ASR weight struct and loader API
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef QWEN3ASR_WEIGHTS_H
#define QWEN3ASR_WEIGHTS_H

#include <stddef.h>
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Encoder per-layer weights
typedef struct {
    float *q_proj, *k_proj, *v_proj, *o_proj;     // [896,896] each
    float *q_bias, *k_bias, *v_bias, *o_bias;      // [896] each
    float *attn_ln_w, *attn_ln_b;                   // self_attn_layer_norm
    float *fc1_w, *fc1_b;                           // [3584,896] / [3584]
    float *fc2_w, *fc2_b;                           // [896,3584] / [896]
    float *final_ln_w, *final_ln_b;                 // final_layer_norm
} qwen3asr_encoder_layer_weights_t;

// Decoder per-layer weights
typedef struct {
    float *q_proj;      // [1024, 2048]
    float *k_proj;      // [1024, 1024]
    float *v_proj;      // [1024, 1024]
    float *o_proj;      // [1024, 2048]
    float *q_norm;      // [128]
    float *k_norm;      // [128]
    float *gate_proj;   // [1024, 3072]
    float *up_proj;     // [1024, 3072]
    float *down_proj;   // [3072, 1024]
    float *input_ln;    // [1024]
    float *post_attn_ln;// [1024]
} qwen3asr_decoder_layer_weights_t;

// Top-level weights struct
typedef struct {
    // Encoder
    float *conv1_w, *conv1_b;    // [480,1,3,3] / [480]
    float *conv2_w, *conv2_b;    // [480,480,3,3] / [480]
    float *conv3_w, *conv3_b;    // [480,480,3,3] / [480]
    float *conv_out_w;           // [896, 7680]
    qwen3asr_encoder_layer_weights_t encoder_layers[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *ln_post_w, *ln_post_b;   // [896]
    float *proj1_w, *proj1_b;       // [896,896] / [896]
    float *proj2_w, *proj2_b;       // [1024,896] / [1024]

    // Decoder
    float *embed_tokens;  // [151936, 1024]
    float *norm_w;        // [1024] — final norm
    float *lm_head;       // [151936, 1024]
    qwen3asr_decoder_layer_weights_t decoder_layers[QWEN3ASR_DECODER_NUM_LAYERS];
} qwen3asr_weights_t;

// Load all weights from a safetensors file
// model_dir: directory containing qwen3_asr_weights.safetensors
// Returns: allocated weights struct, or NULL on failure
qwen3asr_weights_t* qwen3asr_weights_load(const char *model_dir);

// Free all weights
void qwen3asr_weights_free(qwen3asr_weights_t *w);

#ifdef __cplusplus
}
#endif

#endif // QWEN3ASR_WEIGHTS_H
