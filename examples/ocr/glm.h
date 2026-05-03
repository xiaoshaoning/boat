// glm.h - GLM decoder with GQA and KV cache for GLM-OCR
#ifndef BOAT_OCR_GLM_H
#define BOAT_OCR_GLM_H

#include <boat/tensor.h>
#include "../common/safetensors.h"

#define GLM_HIDDEN_SIZE 1536
#define GLM_NUM_HEADS 16
#define GLM_NUM_KV_HEADS 8
#define GLM_HEAD_DIM 128
#define GLM_INTERMEDIATE_SIZE 4608
#define GLM_NUM_LAYERS 16
#define GLM_VOCAB_SIZE 59392
#define GLM_ROPE_THETA 10000.0f
#define GLM_MAX_SEQ_LEN 2048  // max generation length (must accommodate visual tokens + text)

// KV cache for a single layer (flat 2D: [max_seq_len, num_kv_heads * head_dim])
typedef struct {
    boat_tensor_t* k_cache;  // [max_seq_len, num_kv_heads * head_dim]
    boat_tensor_t* v_cache;  // [max_seq_len, num_kv_heads * head_dim]
    int seq_len;
} glm_kv_cache_t;

// GLM decoder layer weights
typedef struct {
    boat_tensor_t* input_layernorm_weight;          // [1536]
    boat_tensor_t* q_proj_weight;                   // [2048, 1536]
    boat_tensor_t* k_proj_weight;                   // [1024, 1536]
    boat_tensor_t* v_proj_weight;                   // [1024, 1536]
    boat_tensor_t* o_proj_weight;                   // [1536, 2048]
    boat_tensor_t* post_attention_layernorm_weight; // [1536]
    boat_tensor_t* post_self_attn_layernorm_weight; // [1536]
    boat_tensor_t* gate_up_proj_weight;             // [9216, 1536]
    boat_tensor_t* down_proj_weight;                // [1536, 4608]
    boat_tensor_t* post_mlp_layernorm_weight;       // [1536]
} glm_layer_weights_t;

// GLM model with all weights and KV caches
typedef struct {
    boat_tensor_t* embed_tokens_weight;  // [59392, 1536]
    glm_layer_weights_t layers[GLM_NUM_LAYERS];
    boat_tensor_t* norm_weight;          // [1536], final RMSNorm

    // Output head
    boat_tensor_t* lm_head_weight;       // [59392, 1536]
    boat_tensor_t* embed_tokens_out_weight; // [59392, 1536] (from layer 16)

    // KV caches (one per layer)
    glm_kv_cache_t kv_caches[GLM_NUM_LAYERS];
} glm_model_t;

// Load all GLM weights from safetensors
int glm_load(glm_model_t* model, safetensors_t* st);

// Free all weights and KV caches
void glm_free(glm_model_t* model);

// Reset KV caches for new sequence
void glm_kv_cache_reset(glm_model_t* model);

// Run one forward step (prefill or decode)
// input_ids: [1, seq_len] token IDs
// use_kv_cache: if 1, uses/extends KV cache; if 0, full prefill
// Returns logits tensor [1, 1, vocab_size] for the NEXT token, or [1, seq_len, vocab_size] for prefill
// Caller must unref the returned tensor
boat_tensor_t* glm_forward(glm_model_t* model, const boat_tensor_t* input_ids, int use_kv_cache);

#endif // BOAT_OCR_GLM_H
