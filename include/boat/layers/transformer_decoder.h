// transformer_decoder.h - Cross-attention transformer decoder layer
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_TRANSFORMER_DECODER_H
#define BOAT_TRANSFORMER_DECODER_H

#include <stdint.h>
#include <stdbool.h>
#include <boat/tensor.h>
#include <boat/export.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
typedef struct {
    int32_t d_model;                // 1024
    int32_t num_heads;              // 16
    int32_t d_ff;                   // 4096
    float   layer_norm_eps;         // 1e-5
    bool    pre_norm;               // true for mBART
    const char* activation;         // "gelu" or "relu"
} boat_decoder_config_t;

// ---------------------------------------------------------------------------
// Single decoder layer weights
// ---------------------------------------------------------------------------
typedef struct {
    // Self-attention QKV + output (no fused QKV — separate per mBART)
    boat_tensor_t* self_q_weight;   // [d_model, d_model]
    boat_tensor_t* self_q_bias;     // [d_model]
    boat_tensor_t* self_k_weight;   // [d_model, d_model]
    boat_tensor_t* self_k_bias;     // [d_model]
    boat_tensor_t* self_v_weight;   // [d_model, d_model]
    boat_tensor_t* self_v_bias;     // [d_model]
    boat_tensor_t* self_o_weight;   // [d_model, d_model]
    boat_tensor_t* self_o_bias;     // [d_model]
    boat_tensor_t* self_ln_weight;  // [d_model]
    boat_tensor_t* self_ln_bias;    // [d_model]

    // Cross-attention (encoder-decoder)
    boat_tensor_t* cross_q_weight;  // [d_model, d_model]
    boat_tensor_t* cross_q_bias;    // [d_model]
    boat_tensor_t* cross_k_weight;  // [d_model, d_model]
    boat_tensor_t* cross_k_bias;    // [d_model]
    boat_tensor_t* cross_v_weight;  // [d_model, d_model]
    boat_tensor_t* cross_v_bias;    // [d_model]
    boat_tensor_t* cross_o_weight;  // [d_model, d_model]
    boat_tensor_t* cross_o_bias;    // [d_model]
    boat_tensor_t* cross_ln_weight; // [d_model]
    boat_tensor_t* cross_ln_bias;   // [d_model]

    // FFN
    boat_tensor_t* fc1_weight;      // [d_ff, d_model]
    boat_tensor_t* fc1_bias;        // [d_ff]
    boat_tensor_t* fc2_weight;      // [d_model, d_ff]
    boat_tensor_t* fc2_bias;        // [d_model]
    boat_tensor_t* ffn_ln_weight;   // [d_model]
    boat_tensor_t* ffn_ln_bias;     // [d_model]
} boat_decoder_layer_weights_t;

// ---------------------------------------------------------------------------
// Per-layer KV cache
// ---------------------------------------------------------------------------
typedef struct {
    boat_tensor_t* self_k;          // [B, num_heads, max_T, head_dim] (grows)
    boat_tensor_t* self_v;          // [B, num_heads, max_T, head_dim]
    boat_tensor_t* cross_k;         // [B, num_heads, S, head_dim] (static, from encoder)
    boat_tensor_t* cross_v;         // [B, num_heads, S, head_dim]
    int32_t length;                 // current cached length (0 initially)
    int32_t max_length;             // allocated capacity
} boat_decoder_cache_t;

// ---------------------------------------------------------------------------
// Layer forward
// ---------------------------------------------------------------------------
// x: [B, T, d_model] float32 (T=1 at inference step, T>1 for training)
// encoder_output: [B, S, d_model] float32
// cache: KV cache (mutated in-place to append new K,V for this step)
// step: current decoding position (0-indexed, used to index into cache)
//
// Returns: [B, T, d_model] float32 — caller must boat_tensor_unref
BOAT_API boat_tensor_t* boat_decoder_layer_forward(
    const boat_decoder_config_t* config,
    const boat_decoder_layer_weights_t* weights,
    const boat_tensor_t* x,
    const boat_tensor_t* encoder_output,
    boat_decoder_cache_t* cache,
    int32_t step);

// ---------------------------------------------------------------------------
// Cache management helpers
// ---------------------------------------------------------------------------
BOAT_API boat_decoder_cache_t* boat_decoder_cache_create(
    int32_t batch_size,
    int32_t num_heads,
    int32_t head_dim,
    int32_t max_t,
    int32_t encoder_seq_len);

BOAT_API boat_decoder_cache_t* boat_decoder_cache_create_ex(
    int32_t batch_size,
    int32_t num_heads,
    int32_t head_dim,
    int32_t max_t,
    int32_t encoder_seq_len,
    boat_device_t device);

BOAT_API void boat_decoder_cache_free(boat_decoder_cache_t* cache);

BOAT_API void boat_decoder_cache_reset(boat_decoder_cache_t* cache);

#ifdef __cplusplus
}
#endif

#endif // BOAT_TRANSFORMER_DECODER_H
