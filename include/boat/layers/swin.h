// swin.h - Swin Transformer encoder layer
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_SWIN_H
#define BOAT_SWIN_H

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
    int32_t embed_dim;             // 128
    int32_t depths[4];             // [2,2,14,2] — blocks per stage
    int32_t num_heads[4];          // [4,8,16,32] — heads per stage
    int32_t window_size;           // 7
    int32_t patch_size;            // 4
    int32_t num_channels;          // 3 (RGB)
    float   mlp_ratio;             // 4.0
    bool    qkv_bias;              // true
    float   layer_norm_eps;        // 1e-5
} boat_swin_config_t;

// ---------------------------------------------------------------------------
// Single Swin block weights (separate Q/K/V, matching donut-swin layout)
// ---------------------------------------------------------------------------
typedef struct {
    boat_tensor_t* norm1_weight;        // [dim]
    boat_tensor_t* norm1_bias;          // [dim]
    boat_tensor_t* query_weight;        // [dim, dim]
    boat_tensor_t* query_bias;          // [dim]
    boat_tensor_t* key_weight;          // [dim, dim]
    boat_tensor_t* key_bias;            // [dim]
    boat_tensor_t* value_weight;        // [dim, dim]
    boat_tensor_t* value_bias;          // [dim]
    boat_tensor_t* proj_weight;         // [dim, dim]
    boat_tensor_t* proj_bias;           // [dim]
    boat_tensor_t* norm2_weight;        // [dim]
    boat_tensor_t* norm2_bias;          // [dim]
    boat_tensor_t* mlp_fc1_weight;      // [4*dim, dim]
    boat_tensor_t* mlp_fc1_bias;        // [4*dim]
    boat_tensor_t* mlp_fc2_weight;      // [dim, 4*dim]
    boat_tensor_t* mlp_fc2_bias;        // [dim]
    boat_tensor_t* rel_pos_bias_table;  // [(2*ws-1)^2, num_heads]
    boat_tensor_t* rel_pos_index;       // [ws*ws, ws*ws] int64
} boat_swin_block_weights_t;

// ---------------------------------------------------------------------------
// PatchMerging (downsample) weights
// ---------------------------------------------------------------------------
typedef struct {
    boat_tensor_t* norm_weight;         // [4*dim]
    boat_tensor_t* norm_bias;           // [4*dim]
    boat_tensor_t* reduction_weight;    // [4*dim, 2*dim]
    boat_tensor_t* reduction_bias;      // [2*dim]
} boat_swin_downsample_weights_t;

// ---------------------------------------------------------------------------
// PatchEmbed weights
// ---------------------------------------------------------------------------
typedef struct {
    boat_tensor_t* proj_weight;         // [embed_dim, C, ph, pw]
    boat_tensor_t* proj_bias;           // [embed_dim]
    boat_tensor_t* norm_weight;         // [embed_dim]
    boat_tensor_t* norm_bias;           // [embed_dim]
} boat_swin_patch_embed_weights_t;

// ---------------------------------------------------------------------------
// Complete Swin weights: 4 stages with blocks and optional downsamplers
// ---------------------------------------------------------------------------
typedef struct {
    boat_swin_patch_embed_weights_t patch_embed;

    // stages[i].blocks: array of depths[i] boat_swin_block_weights_t
    // stages[i].downsample: non-NULL for i=0,1,2; NULL for i=3 (last)
    struct {
        boat_swin_block_weights_t* blocks;
        boat_swin_downsample_weights_t* downsample;
    } stages[4];
} boat_swin_weights_t;

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
// input : [N, C, H, W] float32
// returns: [N, num_patches, final_dim] float32
//   num_patches = (H/ph) * (W/ph) / 8  (after 3 mergings)
//   final_dim   = embed_dim * 8         (after 3 mergings)
BOAT_API boat_tensor_t* boat_swin_forward(
    const boat_swin_config_t* config,
    const boat_swin_weights_t* weights,
    const boat_tensor_t* input);

#ifdef __cplusplus
}
#endif

#endif // BOAT_SWIN_H
