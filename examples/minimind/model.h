// model.h - MiniMind model state and forward pass
#pragma once
#include "config.h"
#include <stddef.h>

typedef struct {
    // ===== Weights (FP32, loaded from model.bin) =====
    float* embed_tokens; // [6400, 768]
    float* lm_head;      // points to embed_tokens (weight tied)

    float* q_proj[MINIMIND_NUM_LAYERS]; // each [768, 768]  (transposed from PT [768,768])
    float* k_proj[MINIMIND_NUM_LAYERS]; // each [384, 768]  (transposed from PT [384,768])
    float* v_proj[MINIMIND_NUM_LAYERS]; // each [384, 768]  (transposed from PT [384,768])
    float* o_proj[MINIMIND_NUM_LAYERS]; // each [768, 768]  (transposed from PT [768,768])

    float* q_norm_weight[MINIMIND_NUM_LAYERS]; // each [96]
    float* k_norm_weight[MINIMIND_NUM_LAYERS]; // each [96]

    float* input_layernorm_weight[MINIMIND_NUM_LAYERS];          // each [768]
    float* post_attention_layernorm_weight[MINIMIND_NUM_LAYERS]; // each [768]

    float* gate_proj[MINIMIND_NUM_LAYERS]; // each [2432, 768] (transposed from PT [2432,768])
    float* down_proj[MINIMIND_NUM_LAYERS]; // each [768, 2432]  (transposed from PT [768,2432])
    float* up_proj[MINIMIND_NUM_LAYERS];   // each [2432, 768] (transposed from PT [2432,768])

    float* final_norm_weight; // [768]

    // ===== Precomputed RoPE tables =====
    float* cos_table; // [max_seq_len, head_dim] = [2048, 96]
    float* sin_table; // [max_seq_len, head_dim] = [2048, 96]

    // ===== KV Cache (pre-allocated) =====
    // K cache: 8 layers, each [max_seq_len, num_kv_heads * head_dim] = [2048, 384]
    // V cache: same dimensions
    float* k_cache[MINIMIND_NUM_LAYERS];
    float* v_cache[MINIMIND_NUM_LAYERS];
    int kv_len; // current cache length (shared across layers)

    // ===== Working buffers (pre-allocated, reused) =====
    float* hidden;   // [max_seq_len, hidden_size]
    float* hidden2;  // [max_seq_len, hidden_size] for double-buffering
    float* q_buf;    // [max_seq_len, num_heads * head_dim] = [2048, 768]
    float* k_buf;    // [max_seq_len, num_kv_heads * head_dim] = [2048, 384]
    float* v_buf;    // [2048, 384]
    float* attn_out; // [max_seq_len, hidden_size]
    float* ffn_gate; // [max_seq_len, intermediate_size]
    float* ffn_up;   // [max_seq_len, intermediate_size]

    // ===== Config =====
    minimind_config_t config;
    int max_seq_len;

    // ===== Memory tracking =====
    void* block_data; // base pointer for all allocated blocks (for cleanup)
    size_t block_size;
} minimind_model_t;

// Initialize model: load weights from model_dir, precompute RoPE, allocate caches.
// Returns 0 on success, -1 on error.
int minimind_model_init(minimind_model_t* m, const char* model_dir);

// Free all memory.
void minimind_model_free(minimind_model_t* m);

// Reset KV cache for a new sequence.
void minimind_model_reset_kv_cache(minimind_model_t* m);

// --- Forward pass ---

// Prefill: process prompt_len tokens, populate KV cache, return logits for last position.
// logits_out must be pre-allocated as float[vocab_size].
void minimind_prefill(minimind_model_t* m, const int* tokens, int n_tokens, float* logits_out);

// Decode: process a single new token using KV cache, return logits.
// logits_out must be pre-allocated as float[vocab_size].
void minimind_decode(minimind_model_t* m, int token, float* logits_out);
