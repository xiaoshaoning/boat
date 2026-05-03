// glm_cuda.cuh - CUDA-accelerated GLM decoder with GQA and KV cache
#pragma once

#include <boat/tensor.h>
#include "../common/safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

// GLM config (same as CPU version)
#define GLM_HIDDEN_SIZE 1536
#define GLM_NUM_HEADS 16
#define GLM_NUM_KV_HEADS 8
#define GLM_HEAD_DIM 128
#define GLM_INTERMEDIATE_SIZE 4608
#define GLM_NUM_LAYERS 16
#define GLM_VOCAB_SIZE 59392
#define GLM_ROPE_THETA 10000.0f
#define GLM_MAX_SEQ_LEN 2048

// GPU-side GLM model
typedef struct {
    // Token embeddings on GPU
    float* d_embed_tokens_weight; // [59392, 1536]

    // 16 decoder layers
    struct {
        float* d_input_layernorm_weight;           // [1536]
        float* d_q_proj_weight;                    // [2048, 1536]
        float* d_k_proj_weight;                    // [1024, 1536]
        float* d_v_proj_weight;                    // [1024, 1536]
        float* d_o_proj_weight;                    // [1536, 2048]
        float* d_post_self_attn_layernorm_weight;  // [1536]
        float* d_post_attention_layernorm_weight;  // [1536] (pre-MLP)
        float* d_gate_up_proj_weight;              // [9216, 1536]
        float* d_down_proj_weight;                 // [1536, 4608]
        float* d_post_mlp_layernorm_weight;        // [1536]

        // KV cache on GPU
        float* d_k_cache; // [GLM_MAX_SEQ_LEN, GLM_NUM_KV_HEADS * GLM_HEAD_DIM]
        float* d_v_cache; // [GLM_MAX_SEQ_LEN, GLM_NUM_KV_HEADS * GLM_HEAD_DIM]
        int seq_len;      // current sequence length in cache (on host)
    } layers[GLM_NUM_LAYERS];

    // Final norm and LM head
    float* d_norm_weight;   // [1536]
    float* d_lm_head_weight; // [59392, 1536]

    // GPU buffer for M-RoPE positions
    int* d_pos_t;
    int* d_pos_h;
    int* d_pos_w;
    int pos_buf_size; // allocated size of position buffers
} glm_cuda_model_t;

// Load all GLM weights to GPU
int glm_cuda_load(glm_cuda_model_t* model, safetensors_t* st);

// Free GPU memory
void glm_cuda_free(glm_cuda_model_t* model);

// Reset KV caches
void glm_cuda_kv_cache_reset(glm_cuda_model_t* model);

// Allocate position buffers for a given sequence length
void glm_cuda_ensure_pos_bufs(glm_cuda_model_t* model, int max_seq_len);

// Run decoder forward pass on GPU
// input_hidden: [seq_len, GLM_HIDDEN_SIZE] on GPU (pre-computed embeddings)
// Returns logits [1, GLM_VOCAB_SIZE] on CPU (caller must free with boat_tensor_unref)
boat_tensor_t* glm_cuda_forward(glm_cuda_model_t* model,
                                 const float* d_input_hidden,
                                 int seq_len,
                                 int prefill_pos_end,
                                 int gen_count,
                                 const int* h_pos_t,
                                 const int* h_pos_h,
                                 const int* h_pos_w);

// Single decode step on GPU
// d_embed: [1, GLM_HIDDEN_SIZE] on GPU
// Returns logits [1, GLM_VOCAB_SIZE] on CPU
boat_tensor_t* glm_cuda_decode_step(glm_cuda_model_t* model,
                                      const float* d_embed,
                                      int abs_pos);

// Compute M-RoPE positions for prefill (host-side)
void glm_compute_rope_positions(int* pos_t, int* pos_h, int* pos_w,
                                 int total_prefill, int vis_start,
                                 int num_vis_tokens, int vis_grid_h, int vis_grid_w);

#ifdef __cplusplus
}
#endif
