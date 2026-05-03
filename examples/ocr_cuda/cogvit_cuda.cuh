// cogvit_cuda.cuh - CUDA-accelerated CogViT vision encoder
#pragma once

#include <boat/tensor.h>
#include "../common/safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

// Same config as CPU version
#define COGVIT_HIDDEN_SIZE 1024
#define COGVIT_NUM_HEADS 16
#define COGVIT_HEAD_DIM 64
#define COGVIT_INTERMEDIATE_SIZE 4096
#define COGVIT_NUM_LAYERS 24
#define COGVIT_PATCH_SIZE 14
#define COGVIT_TEMPORAL_PATCH_SIZE 2
#define COGVIT_SPATIAL_MERGE_SIZE 2
#define COGVIT_OUT_HIDDEN_SIZE 1536

// GPU-side CogViT model (all weight pointers on device)
typedef struct {
    // Patch embedding
    float* d_patch_embed_weight; // [1024, 6, 14, 14] on GPU
    float* d_patch_embed_bias;   // [1024] on GPU

    // Post-layernorm
    float* d_post_layernorm_weight; // [1024] on GPU

    // 24 transformer blocks
    struct {
        float* d_norm1_weight;              // [1024]
        float* d_attn_qkv_weight;           // [3072, 1024]
        float* d_attn_qkv_bias;             // [3072]
        float* d_attn_q_norm_weight;        // [64]
        float* d_attn_k_norm_weight;        // [64]
        float* d_attn_proj_weight;          // [1024, 1024]
        float* d_attn_proj_bias;            // [1024]
        float* d_norm2_weight;              // [1024]
        float* d_mlp_gate_proj_weight;      // [4096, 1024]
        float* d_mlp_gate_proj_bias;        // [4096]
        float* d_mlp_up_proj_weight;        // [4096, 1024]
        float* d_mlp_up_proj_bias;          // [4096]
        float* d_mlp_down_proj_weight;      // [1024, 4096]
        float* d_mlp_down_proj_bias;        // [1024]
    } blocks[COGVIT_NUM_LAYERS];

    // Downsample
    float* d_downsample_weight;  // [1536, 1024, 2, 2]
    float* d_downsample_bias;    // [1536]

    // Merger
    float* d_merger_gate_proj_weight;  // [4608, 1536]
    float* d_merger_up_proj_weight;    // [4608, 1536]
    float* d_merger_down_proj_weight;  // [1536, 4608]
    float* d_merger_proj_weight;       // [1536, 1536]
    float* d_merger_post_norm_weight;  // [1536]
    float* d_merger_post_norm_bias;    // [1536]
} cogvit_cuda_model_t;

// Load all CogViT weights to GPU
// Returns 1 on success, 0 on failure
int cogvit_cuda_load(cogvit_cuda_model_t* model, safetensors_t* st);

// Free GPU memory
void cogvit_cuda_free(cogvit_cuda_model_t* model);

// Run vision encoder forward pass entirely on GPU
// Input: image_tensor [1, 3, H, W] on CPU, normalized with mean/std
// Output: visual_tokens tensor [1, N, COGVIT_OUT_HIDDEN_SIZE] on CPU
//         (caller must free with boat_tensor_unref)
boat_tensor_t* cogvit_cuda_forward(cogvit_cuda_model_t* model, const boat_tensor_t* image);

#ifdef __cplusplus
}
#endif
