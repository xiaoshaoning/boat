// model.h - NanoChat GPU model (forward + decode)
#pragma once
#include "config.h"
#include "weights.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // All weight pointers point to GPU FP32 data
    float* d_embed_tokens;  // [vocab_size, hidden_size]
    float* d_lm_head;       // [vocab_size, hidden_size]

    // Per-layer weights (all on GPU)
    float* d_q_proj[NANOCHAT_NUM_LAYERS];  // each [hidden_size, hidden_size]
    float* d_k_proj[NANOCHAT_NUM_LAYERS];
    float* d_v_proj[NANOCHAT_NUM_LAYERS];
    float* d_o_proj[NANOCHAT_NUM_LAYERS];
    float* d_fc1[NANOCHAT_NUM_LAYERS];     // each [intermediate_size, hidden_size]
    float* d_fc2[NANOCHAT_NUM_LAYERS];     // each [hidden_size, intermediate_size]

    // KV cache (on GPU)
    float* d_k_cache[NANOCHAT_NUM_LAYERS]; // each [max_seq_len, n_heads * head_dim]
    float* d_v_cache[NANOCHAT_NUM_LAYERS]; // each [max_seq_len, n_heads * head_dim]
    int kv_len[NANOCHAT_NUM_LAYERS];       // current cache length per layer (on host)

    // RoPE position buffer (on GPU)
    int* d_pos;
    int pos_buf_size;

    // Pre-allocated decode buffers (reused per-token, avoids cudaMalloc/free)
    float* d_decode_tmp;     // [hidden_size + q_size + kv_size + kv_size + ff_dim + hidden_size]
    float* d_decode_hidden;  // [hidden_size]

    // Model config (copied from config for convenience)
    int n_layers;
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
} nanochat_cuda_model_t;

// Load CPU weights to GPU and initialize model
int nanochat_cuda_model_init(nanochat_cuda_model_t* model,
                              const nanochat_weights_t* weights);

// Free all GPU memory
void nanochat_cuda_model_free(nanochat_cuda_model_t* model);

// Reset KV caches
void nanochat_cuda_model_reset_kv_cache(nanochat_cuda_model_t* model);

// Prefill: run full forward pass on embedded tokens
// d_hidden: [seq_len, hidden_size] on GPU (embedded tokens, will be overwritten)
// Returns logits on GPU (caller must free with cudaFree)
float* nanochat_cuda_model_forward(nanochat_cuda_model_t* model,
                                    float* d_hidden, int seq_len);

// Decode: single token step
// d_embed: [1, hidden_size] on GPU
// Returns logits on GPU (caller must free with cudaFree)
float* nanochat_cuda_model_decode(nanochat_cuda_model_t* model,
                                   float* d_embed, int abs_pos);

#ifdef __cplusplus
}
#endif
