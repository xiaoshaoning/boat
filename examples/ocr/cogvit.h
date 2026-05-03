// cogvit.h - CogViT vision encoder for GLM-OCR
#ifndef BOAT_OCR_COGVIT_H
#define BOAT_OCR_COGVIT_H

#include <boat/tensor.h>
#include "../common/safetensors.h"

// CogViT configuration
#define COGVIT_HIDDEN_SIZE 1024
#define COGVIT_NUM_HEADS 16
#define COGVIT_HEAD_DIM 64
#define COGVIT_INTERMEDIATE_SIZE 4096
#define COGVIT_NUM_LAYERS 24
#define COGVIT_PATCH_SIZE 14
#define COGVIT_TEMPORAL_PATCH_SIZE 2
#define COGVIT_SPATIAL_MERGE_SIZE 2
#define COGVIT_IMAGE_SIZE 336
#define COGVIT_OUT_HIDDEN_SIZE 1536

typedef struct {
    // Patch embedding
    boat_tensor_t* patch_embed_weight;  // [1024, 6, 14, 14]
    boat_tensor_t* patch_embed_bias;    // [1024]

    // Pre-norm (post_layernorm for after all blocks)
    boat_tensor_t* post_layernorm_weight;  // [1024]

    // 24 transformer blocks
    struct {
        boat_tensor_t* norm1_weight;             // [1024]
        boat_tensor_t* attn_qkv_weight;          // [3072, 1024]
        boat_tensor_t* attn_qkv_bias;            // [3072]
        boat_tensor_t* attn_q_norm_weight;       // [64]
        boat_tensor_t* attn_k_norm_weight;       // [64]
        boat_tensor_t* attn_proj_weight;         // [1024, 1024]
        boat_tensor_t* attn_proj_bias;           // [1024]
        boat_tensor_t* norm2_weight;             // [1024]
        boat_tensor_t* mlp_gate_proj_weight;     // [4096, 1024]
        boat_tensor_t* mlp_gate_proj_bias;       // [4096]
        boat_tensor_t* mlp_up_proj_weight;       // [4096, 1024]
        boat_tensor_t* mlp_up_proj_bias;         // [4096]
        boat_tensor_t* mlp_down_proj_weight;     // [1024, 4096]
        boat_tensor_t* mlp_down_proj_bias;       // [1024]
    } blocks[COGVIT_NUM_LAYERS];

    // Downsample (2x2 conv merge: H/2, W/2, hidden 1024→1536)
    boat_tensor_t* downsample_weight;  // [1536, 1024, 2, 2]
    boat_tensor_t* downsample_bias;    // [1536]

    // Merger (MLP to refine visual tokens)
    boat_tensor_t* merger_gate_proj_weight;      // [4608, 1536]
    boat_tensor_t* merger_up_proj_weight;        // [4608, 1536]
    boat_tensor_t* merger_down_proj_weight;      // [1536, 4608]
    boat_tensor_t* merger_proj_weight;           // [1536, 1536]
    boat_tensor_t* merger_post_norm_weight;      // [1536]
    boat_tensor_t* merger_post_norm_bias;        // [1536]
} cogvit_model_t;

// Load all CogViT weights from safetensors
// Returns 1 on success, 0 on failure
int cogvit_load(cogvit_model_t* model, safetensors_t* st);

// Free all weights
void cogvit_free(cogvit_model_t* model);

// Run vision encoder forward pass
// Input: image_tensor [1, 3, H, W], normalized with mean/std
//        H and W should be multiples of 14 (patch_size)
// Output: visual_tokens [1, N, 1536] where N = (H/28) * (W/28)
//         (caller must free with boat_tensor_unref)
boat_tensor_t* cogvit_forward(cogvit_model_t* model, const boat_tensor_t* image);

#endif // BOAT_OCR_COGVIT_H
