// model.h - NanoChat GPU model (forward + decode + training)
// Weights and KV cache stored as BF16 (model was trained in BF16)
#pragma once
#include "config.h"
#include "weights.h"
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nanochat_cuda_model_s {
    // All weight pointers point to GPU BF16 data
    __nv_bfloat16* d_embed_tokens; // [vocab_size, hidden_size]
    __nv_bfloat16* d_lm_head;      // [vocab_size, hidden_size]

    // Per-layer weights (all on GPU, BF16)
    __nv_bfloat16* d_q_proj[NANOCHAT_NUM_LAYERS]; // each [hidden_size, hidden_size]
    __nv_bfloat16* d_k_proj[NANOCHAT_NUM_LAYERS];
    __nv_bfloat16* d_v_proj[NANOCHAT_NUM_LAYERS];
    __nv_bfloat16* d_o_proj[NANOCHAT_NUM_LAYERS];
    __nv_bfloat16* d_fc1[NANOCHAT_NUM_LAYERS]; // each [intermediate_size, hidden_size]
    __nv_bfloat16* d_fc2[NANOCHAT_NUM_LAYERS]; // each [hidden_size, intermediate_size]

    // KV cache (on GPU, BF16)
    __nv_bfloat16* d_k_cache[NANOCHAT_NUM_LAYERS]; // each [max_seq_len, n_heads * head_dim]
    __nv_bfloat16* d_v_cache[NANOCHAT_NUM_LAYERS];
    int kv_len[NANOCHAT_NUM_LAYERS]; // current cache length per layer (on host)

    // RoPE position buffer (on GPU)
    int* d_pos;
    int pos_buf_size;

    // Pre-allocated decode buffers (BF16, reused per-token)
    __nv_bfloat16* d_decode_tmp;
    __nv_bfloat16* d_decode_hidden;

    // -----------------------------------------------------------------------
    // Training buffers — allocated by nanochat_cuda_model_alloc_train()
    // All gradient/state buffers are flat FP32 allocations, accessed via helper
    // macros computed from model config.
    // -----------------------------------------------------------------------

    // Gradient buffers (FP32): flat buffer indexed by training_offset()
    // Layout: [embed_grad] [lm_head_grad] [layer0_grads] ... [layer{L-1}_grads]
    float* d_grad_buf;

    // Adam optimizer state (FP32): same layout as d_grad_buf
    float* d_m_buf;
    float* d_v_buf;

    // Activation storage (BF16): flat buffer for per-layer saved activations
    // Layout: for each layer: h_in, q_norm, k_norm, v, attn_out, ff_act
    __nv_bfloat16* d_act_buf;
    int act_capacity; // allocated T (seq_len) for activation buffer

    // FP32 workspace: logits[T,V] + P workspace[H,T,T] + grad accumulation
    float* d_f32_buf;
    int f32_capacity; // allocated T (seq_len) for workspace

    // Training temp BF16 buffer (reused per operation, sized for max(T*H, T*FF))
    __nv_bfloat16* d_train_tmp_bf16;

    // Model config
    int n_layers;
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
} nanochat_cuda_model_t;

// -----------------------------------------------------------------------
// Helper macros for gradient/state buffer offsets (assumes model is populated)
// These compute byte offsets into d_grad_buf / d_m_buf / d_v_buf
// -----------------------------------------------------------------------
#define TRAIN_OFFSET_EMBED(m) ((size_t)0)
#define TRAIN_OFFSET_LMHEAD(m) ((size_t)(m)->vocab_size * (m)->hidden_size)
#define TRAIN_OFFSET_LAYER(m, l)                                                                   \
    (TRAIN_OFFSET_LMHEAD(m) + (size_t)(m)->vocab_size * (m)->hidden_size +                         \
     (size_t)(l) * ((size_t)4 * (m)->hidden_size * (m)->hidden_size +                              \
                    (size_t)2 * (m)->intermediate_size * (m)->hidden_size))
#define TRAIN_OFFSET_Q(m, l) (TRAIN_OFFSET_LAYER(m, l))
#define TRAIN_OFFSET_K(m, l) (TRAIN_OFFSET_Q(m, l) + (size_t)(m)->hidden_size * (m)->hidden_size)
#define TRAIN_OFFSET_V(m, l) (TRAIN_OFFSET_K(m, l) + (size_t)(m)->hidden_size * (m)->hidden_size)
#define TRAIN_OFFSET_O(m, l) (TRAIN_OFFSET_V(m, l) + (size_t)(m)->hidden_size * (m)->hidden_size)
#define TRAIN_OFFSET_FC1(m, l) (TRAIN_OFFSET_O(m, l) + (size_t)(m)->hidden_size * (m)->hidden_size)
#define TRAIN_OFFSET_FC2(m, l)                                                                     \
    (TRAIN_OFFSET_FC1(m, l) + (size_t)(m)->intermediate_size * (m)->hidden_size)
#define TRAIN_GRAD_TOTAL(m)                                                                        \
    (TRAIN_OFFSET_FC2(m, (m)->n_layers - 1) + (size_t)(m)->intermediate_size * (m)->hidden_size)

#define TRAIN_GRAD_PTR(m, off) ((m)->d_grad_buf + (off))
#define TRAIN_M_PTR(m, off) ((m)->d_m_buf + (off))
#define TRAIN_V_PTR(m, off) ((m)->d_v_buf + (off))

// -----------------------------------------------------------------------
// Helper macros for activation buffer offsets (BF16, per-layer, T-dependent)
// Each macro takes model, layer index, and seq_len T
// -----------------------------------------------------------------------
#define ACT_OFFSET_HIN(m, l, T)                                                                    \
    ((size_t)(l) * (6ULL * (size_t)(T) * (m)->hidden_size + (size_t)(T) * (m)->intermediate_size))
#define ACT_OFFSET_Q(m, l, T) (ACT_OFFSET_HIN(m, l, T) + (size_t)(T) * (m)->hidden_size)
#define ACT_OFFSET_K(m, l, T) (ACT_OFFSET_Q(m, l, T) + (size_t)(T) * (m)->hidden_size)
#define ACT_OFFSET_V(m, l, T) (ACT_OFFSET_K(m, l, T) + (size_t)(T) * (m)->hidden_size)
#define ACT_OFFSET_ATTN_OUT(m, l, T) (ACT_OFFSET_V(m, l, T) + (size_t)(T) * (m)->hidden_size)
#define ACT_OFFSET_FF(m, l, T) (ACT_OFFSET_ATTN_OUT(m, l, T) + (size_t)(T) * (m)->hidden_size)
#define ACT_LAYER_STRIDE(m, T)                                                                     \
    (6ULL * (size_t)(T) * (m)->hidden_size + (size_t)(T) * (m)->intermediate_size)
#define ACT_BUF_TOTAL(m, T) (ACT_LAYER_STRIDE(m, T) * (m)->n_layers)

// -----------------------------------------------------------------------
// API functions
// -----------------------------------------------------------------------

// Load CPU weights to GPU and initialize model
int nanochat_cuda_model_init(nanochat_cuda_model_t* model, const nanochat_weights_t* weights);

// Free all GPU memory (including training buffers if allocated)
void nanochat_cuda_model_free(nanochat_cuda_model_t* model);

// Allocate training buffers (gradients, Adam state, activations, workspace)
// call after nanochat_cuda_model_init(). max_T is the maximum sequence length.
// Returns non-zero on success.
int nanochat_cuda_model_alloc_train(nanochat_cuda_model_t* model, int max_T);

// Free training buffers only (keeps weights and inference state intact)
void nanochat_cuda_model_free_train(nanochat_cuda_model_t* model);

// Reset KV caches
void nanochat_cuda_model_reset_kv_cache(nanochat_cuda_model_t* model);

// Prefill: run full forward pass on embedded tokens
// d_hidden: [seq_len, hidden_size] on GPU (BF16 embedded tokens, will be overwritten)
// Returns logits on GPU (FP32, caller must free with cudaFree)
float* nanochat_cuda_model_forward(nanochat_cuda_model_t* model, __nv_bfloat16* d_hidden,
                                   int seq_len);

// Decode: single token step
// d_embed: [1, hidden_size] on GPU (BF16)
// Returns logits on GPU (FP32, caller must free with cudaFree)
float* nanochat_cuda_model_decode(nanochat_cuda_model_t* model, __nv_bfloat16* d_embed,
                                  int abs_pos);

#ifdef __cplusplus
}
#endif
