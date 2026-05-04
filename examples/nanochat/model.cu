// model.cu - NanoChat GPU model: weight upload, prefill forward, decode step
// All weights and internal compute in BF16; lm_head outputs FP32 logits
#include "model.h"
#include "weights.h"
#include "nanochat_kernels.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                        \
    }                                                                   \
} while(0)

extern "C" cublasHandle_t boat_cuda_get_cublas_handle(void);

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t stat = call;                                         \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                \
        fprintf(stderr, "[cuBLAS] %s:%d: error %d\n",                  \
                __FILE__, __LINE__, (int)stat);                         \
        exit(1);                                                        \
    }                                                                   \
} while(0)

// ---------------------------------------------------------------------------
// Upload a CPU float array to GPU as BF16 (convert on host)
// ---------------------------------------------------------------------------
static __nv_bfloat16* upload_to_gpu_bf16(const float* src, size_t n) {
    __nv_bfloat16* h_tmp = (__nv_bfloat16*)malloc(n * sizeof(__nv_bfloat16));
    if (!h_tmp) { fprintf(stderr, "malloc (%zu) failed\n", n * sizeof(__nv_bfloat16)); exit(1); }
    for (size_t i = 0; i < n; i++) h_tmp[i] = __float2bfloat16(src[i]);
    __nv_bfloat16* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_ptr, h_tmp, n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    free(h_tmp);
    return d_ptr;
}

// ---------------------------------------------------------------------------
// Initialize model: upload all weights to GPU as BF16
// ---------------------------------------------------------------------------
int nanochat_cuda_model_init(nanochat_cuda_model_t* model,
                              const nanochat_weights_t* weights) {
    memset(model, 0, sizeof(*model));

    model->n_layers = weights->n_layers;
    model->vocab_size = weights->vocab_size;
    model->hidden_size = weights->hidden_size;
    model->intermediate_size = weights->intermediate_size;
    model->num_heads = NANOCHAT_NUM_HEADS;
    model->num_kv_heads = NANOCHAT_NUM_KV_HEADS;
    model->head_dim = NANOCHAT_HEAD_DIM;
    model->max_seq_len = NANOCHAT_MAX_SEQ_LEN;

    size_t num_hidden = (size_t)weights->vocab_size * weights->hidden_size;
    size_t num_hidden_sq = (size_t)weights->hidden_size * weights->hidden_size;
    size_t num_ff_hidden = (size_t)weights->intermediate_size * weights->hidden_size;

    model->d_embed_tokens = upload_to_gpu_bf16(weights->embed_tokens, num_hidden);
    model->d_lm_head = upload_to_gpu_bf16(weights->lm_head, num_hidden);

    size_t layer_cache_bytes = (size_t)model->max_seq_len * model->num_kv_heads * model->head_dim * sizeof(__nv_bfloat16);

    for (int l = 0; l < model->n_layers; l++) {
        model->d_q_proj[l] = upload_to_gpu_bf16(weights->q_proj[l], num_hidden_sq);
        model->d_k_proj[l] = upload_to_gpu_bf16(weights->k_proj[l], num_hidden_sq);
        model->d_v_proj[l] = upload_to_gpu_bf16(weights->v_proj[l], num_hidden_sq);
        model->d_o_proj[l] = upload_to_gpu_bf16(weights->o_proj[l], num_hidden_sq);
        model->d_fc1[l] = upload_to_gpu_bf16(weights->fc1[l], num_ff_hidden);
        model->d_fc2[l] = upload_to_gpu_bf16(weights->fc2[l], num_ff_hidden);

        CUDA_CHECK(cudaMalloc(&model->d_k_cache[l], layer_cache_bytes));
        CUDA_CHECK(cudaMalloc(&model->d_v_cache[l], layer_cache_bytes));
        CUDA_CHECK(cudaMemset(model->d_k_cache[l], 0, layer_cache_bytes));
        CUDA_CHECK(cudaMemset(model->d_v_cache[l], 0, layer_cache_bytes));
        model->kv_len[l] = 0;
    }

    model->pos_buf_size = model->max_seq_len;
    CUDA_CHECK(cudaMalloc(&model->d_pos, model->max_seq_len * sizeof(int)));

    // Pre-allocate decode temp buffers (BF16)
    size_t decode_per_row = (size_t)(model->hidden_size +               // normed
        model->num_heads * model->head_dim +     // Q
        model->num_heads * model->head_dim +     // K
        model->num_heads * model->head_dim +     // V
        model->intermediate_size +               // FF
        model->hidden_size);                     // MLP out
    CUDA_CHECK(cudaMalloc(&model->d_decode_tmp, decode_per_row * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&model->d_decode_hidden, (size_t)model->hidden_size * sizeof(__nv_bfloat16)));

    fprintf(stderr, "[NanoChat-CUDA] Model loaded (BF16): %d layers, %d hidden, %d heads\n",
            model->n_layers, model->hidden_size, model->num_heads);
    return 1;
}

// ---------------------------------------------------------------------------
// Free all GPU memory
// ---------------------------------------------------------------------------
void nanochat_cuda_model_free(nanochat_cuda_model_t* model) {
    if (!model) return;
    #define FREE_GPU(p) do { if (p) { cudaFree(p); (p) = NULL; } } while(0)
    FREE_GPU(model->d_embed_tokens);
    FREE_GPU(model->d_lm_head);
    for (int l = 0; l < model->n_layers && l < NANOCHAT_NUM_LAYERS; l++) {
        FREE_GPU(model->d_q_proj[l]);
        FREE_GPU(model->d_k_proj[l]);
        FREE_GPU(model->d_v_proj[l]);
        FREE_GPU(model->d_o_proj[l]);
        FREE_GPU(model->d_fc1[l]);
        FREE_GPU(model->d_fc2[l]);
        FREE_GPU(model->d_k_cache[l]);
        FREE_GPU(model->d_v_cache[l]);
        model->kv_len[l] = 0;
    }
    FREE_GPU(model->d_pos);
    FREE_GPU(model->d_decode_tmp);
    FREE_GPU(model->d_decode_hidden);
    // Free training buffers if allocated
    FREE_GPU(model->d_grad_buf);
    FREE_GPU(model->d_m_buf);
    FREE_GPU(model->d_v_buf);
    FREE_GPU(model->d_act_buf);
    FREE_GPU(model->d_f32_buf);
    FREE_GPU(model->d_train_tmp_bf16);
    #undef FREE_GPU
    memset(model, 0, sizeof(*model));
}

// ---------------------------------------------------------------------------
// Reset KV caches
// ---------------------------------------------------------------------------
void nanochat_cuda_model_reset_kv_cache(nanochat_cuda_model_t* model) {
    size_t cache_bytes = (size_t)model->max_seq_len * model->num_kv_heads * model->head_dim * sizeof(__nv_bfloat16);
    for (int l = 0; l < model->n_layers; l++) {
        model->kv_len[l] = 0;
        CUDA_CHECK(cudaMemset(model->d_k_cache[l], 0, cache_bytes));
        CUDA_CHECK(cudaMemset(model->d_v_cache[l], 0, cache_bytes));
    }
}

// ============================================================================
// Layer prefill (forward for a full sequence through one layer) — BF16
// ============================================================================
static void nanochat_layer_prefill_cuda(cublasHandle_t handle,
    __nv_bfloat16* d_hidden, int seq_len,
    int hidden_size, int num_heads, int head_dim, int ff_dim,
    const __nv_bfloat16* d_q_w, const __nv_bfloat16* d_k_w, const __nv_bfloat16* d_v_w, const __nv_bfloat16* d_o_w,
    const __nv_bfloat16* d_fc1_w, const __nv_bfloat16* d_fc2_w,
    __nv_bfloat16* d_k_cache, __nv_bfloat16* d_v_cache, int* kv_len,
    const int* d_pos, __nv_bfloat16* d_tmp, cudaStream_t stream) {

    int q_size = num_heads * head_dim;
    int kv_size = num_heads * head_dim;

    // Buffer layout (all offsets from d_tmp):
    __nv_bfloat16* d_normed  = d_tmp;                                              // [S, H]
    __nv_bfloat16* d_q       = d_tmp + (size_t)seq_len * hidden_size;              // [S, Q]
    __nv_bfloat16* d_k       = d_q + (size_t)seq_len * q_size;                    // [S, KV]
    __nv_bfloat16* d_v       = d_k + (size_t)seq_len * kv_size;                   // [S, KV]
    __nv_bfloat16* d_ff      = d_v + (size_t)seq_len * kv_size;                   // [S, FF]
    __nv_bfloat16* d_mlp_out = d_ff + (size_t)seq_len * ff_dim;                   // [S, H]

    // ---- Attention phase ----

    // 1. Pre-attention RMSNorm (no weight)
    rmsnorm_nw_bf16_cuda(d_hidden, d_normed, seq_len, hidden_size, NANOCHAT_RMS_EPS, stream);

    // 2. QKV projections (BF16 matmul)
    matmul_bt_bf16_cuda(handle, d_normed, d_q_w, d_q, seq_len, hidden_size, q_size);
    matmul_bt_bf16_cuda(handle, d_normed, d_k_w, d_k, seq_len, hidden_size, kv_size);
    matmul_bt_bf16_cuda(handle, d_normed, d_v_w, d_v, seq_len, hidden_size, kv_size);

    // 3. RoPE (BF16)
    apply_rope_bf16_cuda(d_q, d_k, seq_len, num_heads, num_heads,
                    head_dim, NANOCHAT_ROPE_THETA, d_pos, stream);

    // 4. QK norm after RoPE (BF16 per-head RMSNorm)
    rmsnorm_nw_bf16_cuda(d_q, d_q, seq_len * num_heads, head_dim, NANOCHAT_RMS_EPS, stream);
    rmsnorm_nw_bf16_cuda(d_k, d_k, seq_len * num_heads, head_dim, NANOCHAT_RMS_EPS, stream);

    // 5. Store K,V in KV cache
    CUDA_CHECK(cudaMemcpyAsync(d_k_cache, d_k, (size_t)seq_len * kv_size * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v_cache, d_v, (size_t)seq_len * kv_size * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToDevice, stream));
    *kv_len = seq_len;

    // 6. Fused MHA prefill attention (BF16 I/O, FP32 internal)
    float scale = 1.0f / sqrtf((float)head_dim);
    fused_prefill_attention_bf16_cuda(d_q, d_k, d_v, d_q,
                                 seq_len, num_heads, head_dim, scale, stream);

    // 7. Output projection (BF16)
    matmul_bt_bf16_cuda(handle, d_q, d_o_w, d_normed, seq_len, q_size, hidden_size);

    // 8. Residual: d_hidden += d_normed
    residual_add_bf16_cuda(d_hidden, d_normed, seq_len * hidden_size, stream);

    // ---- MLP phase ----

    // 9. Pre-MLP RMSNorm
    rmsnorm_nw_bf16_cuda(d_hidden, d_normed, seq_len, hidden_size, NANOCHAT_RMS_EPS, stream);

    // 10. FC1 + ReLU² + FC2
    matmul_bt_bf16_cuda(handle, d_normed, d_fc1_w, d_ff, seq_len, hidden_size, ff_dim);
    relu2_bf16_cuda(d_ff, seq_len * ff_dim, stream);
    matmul_bt_bf16_cuda(handle, d_ff, d_fc2_w, d_mlp_out, seq_len, ff_dim, hidden_size);

    // 11. Residual: d_hidden += d_mlp_out
    residual_add_bf16_cuda(d_hidden, d_mlp_out, seq_len * hidden_size, stream);
}

// ============================================================================
// Layer decode (single token) — BF16
// ============================================================================
static void nanochat_layer_decode_cuda(cublasHandle_t handle,
    __nv_bfloat16* d_hidden,
    int hidden_size, int num_heads, int head_dim, int ff_dim,
    const __nv_bfloat16* d_q_w, const __nv_bfloat16* d_k_w, const __nv_bfloat16* d_v_w, const __nv_bfloat16* d_o_w,
    const __nv_bfloat16* d_fc1_w, const __nv_bfloat16* d_fc2_w,
    __nv_bfloat16* d_k_cache, __nv_bfloat16* d_v_cache, int* h_kv_len,
    const int* d_pos, __nv_bfloat16* d_tmp, cudaStream_t stream) {

    int q_size = num_heads * head_dim;
    int kv_size = num_heads * head_dim;
    int kv_len = *h_kv_len;

    // Buffer layout (same as prefill but S=1)
    __nv_bfloat16* d_normed  = d_tmp;
    __nv_bfloat16* d_q       = d_tmp + hidden_size;
    __nv_bfloat16* d_k       = d_q + q_size;
    __nv_bfloat16* d_v       = d_k + kv_size;
    __nv_bfloat16* d_ff      = d_v + kv_size;
    __nv_bfloat16* d_mlp_out = d_ff + ff_dim;

    // ---- Attention phase ----

    // 1. Pre-attention RMSNorm
    rmsnorm_nw_bf16_cuda(d_hidden, d_normed, 1, hidden_size, NANOCHAT_RMS_EPS, stream);

    // 2. QKV projections
    matmul_bt_bf16_cuda(handle, d_normed, d_q_w, d_q, 1, hidden_size, q_size);
    matmul_bt_bf16_cuda(handle, d_normed, d_k_w, d_k, 1, hidden_size, kv_size);
    matmul_bt_bf16_cuda(handle, d_normed, d_v_w, d_v, 1, hidden_size, kv_size);

    // 3. RoPE
    apply_rope_bf16_cuda(d_q, d_k, 1, num_heads, num_heads,
                    head_dim, NANOCHAT_ROPE_THETA, d_pos, stream);

    // QK norm after RoPE
    rmsnorm_nw_bf16_cuda(d_q, d_q, num_heads, head_dim, NANOCHAT_RMS_EPS, stream);
    rmsnorm_nw_bf16_cuda(d_k, d_k, num_heads, head_dim, NANOCHAT_RMS_EPS, stream);

    // 4. Append to KV cache
    CUDA_CHECK(cudaMemcpyAsync(d_k_cache + (size_t)kv_len * kv_size, d_k,
                                (size_t)kv_size * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v_cache + (size_t)kv_len * kv_size, d_v,
                                (size_t)kv_size * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToDevice, stream));
    (*h_kv_len)++;
    int new_kv_len = kv_len + 1;

    // 5. Fused MHA decode attention (BF16 I/O, FP32 internal)
    fused_decode_attention_bf16_cuda(d_q, d_k_cache, d_v_cache, d_q,
                                new_kv_len, num_heads, head_dim, stream);

    // 6. Output projection
    matmul_bt_bf16_cuda(handle, d_q, d_o_w, d_normed, 1, q_size, hidden_size);

    // 7. Residual
    residual_add_bf16_cuda(d_hidden, d_normed, hidden_size, stream);

    // ---- MLP phase ----

    // 8. Pre-MLP RMSNorm
    rmsnorm_nw_bf16_cuda(d_hidden, d_normed, 1, hidden_size, NANOCHAT_RMS_EPS, stream);

    // 9. FC1 + ReLU² + FC2
    matmul_bt_bf16_cuda(handle, d_normed, d_fc1_w, d_ff, 1, hidden_size, ff_dim);
    relu2_bf16_cuda(d_ff, ff_dim, stream);
    matmul_bt_bf16_cuda(handle, d_ff, d_fc2_w, d_mlp_out, 1, ff_dim, hidden_size);

    // 10. Residual
    residual_add_bf16_cuda(d_hidden, d_mlp_out, hidden_size, stream);
}

// ============================================================================
// Prefill forward: run all layers, return logits on GPU
// d_hidden: [seq_len, hidden_size] BF16 embedded tokens on GPU (overwritten)
// Returns FP32 logits on GPU (caller must free with cudaFree)
// ============================================================================
float* nanochat_cuda_model_forward(nanochat_cuda_model_t* model,
                                    __nv_bfloat16* d_hidden, int seq_len) {
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    cudaStream_t stream = 0;

    int hidden_size = model->hidden_size;
    int num_heads = model->num_heads;
    int head_dim = model->head_dim;
    int ff_dim = model->intermediate_size;

    // Set RoPE positions (0, 1, 2, ..., seq_len-1)
    int* h_pos = (int*)malloc((size_t)seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++) h_pos[i] = i;
    CUDA_CHECK(cudaMemcpy(model->d_pos, h_pos, (size_t)seq_len * sizeof(int),
                           cudaMemcpyHostToDevice));
    free(h_pos);

    // Allocate temporary buffer (BF16)
    size_t per_row = (size_t)(hidden_size + num_heads * head_dim +
                               num_heads * head_dim + num_heads * head_dim +
                               ff_dim + hidden_size);
    size_t tmp_size = (size_t)seq_len * per_row * sizeof(__nv_bfloat16);
    __nv_bfloat16* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size));

    // Pre-layers RMSNorm (BF16)
    rmsnorm_nw_bf16_cuda(d_hidden, d_hidden, seq_len, hidden_size, NANOCHAT_RMS_EPS, stream);

    // Run layers
    for (int l = 0; l < model->n_layers; l++) {
        nanochat_layer_prefill_cuda(handle,
            d_hidden, seq_len,
            hidden_size, num_heads, head_dim, ff_dim,
            model->d_q_proj[l], model->d_k_proj[l],
            model->d_v_proj[l], model->d_o_proj[l],
            model->d_fc1[l], model->d_fc2[l],
            model->d_k_cache[l], model->d_v_cache[l], &model->kv_len[l],
            model->d_pos, d_tmp, stream);
    }

    // Final RMSNorm
    rmsnorm_nw_bf16_cuda(d_hidden, d_hidden, seq_len, hidden_size, NANOCHAT_RMS_EPS, stream);

    // LM head: logits = hidden[last_pos] @ lm_head^T → FP32 output
    int last_pos = seq_len - 1;
    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)model->vocab_size * sizeof(float)));
    const __nv_bfloat16* d_last = d_hidden + (size_t)last_pos * hidden_size;
    matmul_bt_bf16_out_f32_cuda(handle, d_last, model->d_lm_head,
                   d_logits, 1, hidden_size, model->vocab_size);

    // Logit softcap (FP32)
    softcap_cuda(d_logits, model->vocab_size, NANOCHAT_SOFTCAP, stream);

    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return d_logits;
}

// ============================================================================
// Decode step: single token, return logits on GPU
// d_embed: [1, hidden_size] BF16 on GPU
// Returns FP32 logits on GPU (caller must free with cudaFree)
// ============================================================================
float* nanochat_cuda_model_decode(nanochat_cuda_model_t* model,
                                   __nv_bfloat16* d_embed, int abs_pos) {
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    cudaStream_t stream = 0;

    int hidden_size = model->hidden_size;
    int num_heads = model->num_heads;
    int head_dim = model->head_dim;
    int ff_dim = model->intermediate_size;

    // Set RoPE position for this single token
    CUDA_CHECK(cudaMemcpy(model->d_pos, &abs_pos, sizeof(int),
                           cudaMemcpyHostToDevice));

    // Use pre-allocated decode buffers (BF16)
    __nv_bfloat16* d_tmp = model->d_decode_tmp;

    // Copy embedding to persistent hidden buffer
    __nv_bfloat16* d_hidden = model->d_decode_hidden;
    CUDA_CHECK(cudaMemcpy(d_hidden, d_embed, (size_t)hidden_size * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice));

    // Pre-layers RMSNorm
    rmsnorm_nw_bf16_cuda(d_hidden, d_hidden, 1, hidden_size, NANOCHAT_RMS_EPS, stream);

    // Run layers
    for (int l = 0; l < model->n_layers; l++) {
        nanochat_layer_decode_cuda(handle, d_hidden,
            hidden_size, num_heads, head_dim, ff_dim,
            model->d_q_proj[l], model->d_k_proj[l],
            model->d_v_proj[l], model->d_o_proj[l],
            model->d_fc1[l], model->d_fc2[l],
            model->d_k_cache[l], model->d_v_cache[l], &model->kv_len[l],
            model->d_pos, d_tmp, stream);
    }

    // Final RMSNorm
    rmsnorm_nw_bf16_cuda(d_hidden, d_hidden, 1, hidden_size, NANOCHAT_RMS_EPS, stream);

    // LM head → FP32 logits
    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)model->vocab_size * sizeof(float)));
    matmul_bt_bf16_out_f32_cuda(handle, d_hidden, model->d_lm_head,
                   d_logits, 1, hidden_size, model->vocab_size);

    // Logit softcap
    softcap_cuda(d_logits, model->vocab_size, NANOCHAT_SOFTCAP, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    return d_logits;
}
