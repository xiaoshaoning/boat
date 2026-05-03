// model.cu - NanoChat GPU model: weight upload, prefill forward, decode step
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
// Upload a CPU float array to GPU
// ---------------------------------------------------------------------------
static float* upload_to_gpu(const float* src, size_t n) {
    float* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ptr, src, n * sizeof(float), cudaMemcpyHostToDevice));
    return d_ptr;
}

// ---------------------------------------------------------------------------
// Initialize model: upload all weights to GPU
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

    // Embed tokens
    model->d_embed_tokens = upload_to_gpu(weights->embed_tokens, num_hidden);

    // LM head
    model->d_lm_head = upload_to_gpu(weights->lm_head, num_hidden);

    // Per-layer weights
    size_t layer_cache_bytes = (size_t)model->max_seq_len * model->num_kv_heads * model->head_dim * sizeof(float);

    for (int l = 0; l < model->n_layers; l++) {
        model->d_q_proj[l] = upload_to_gpu(weights->q_proj[l], num_hidden_sq);
        model->d_k_proj[l] = upload_to_gpu(weights->k_proj[l], num_hidden_sq);
        model->d_v_proj[l] = upload_to_gpu(weights->v_proj[l], num_hidden_sq);
        model->d_o_proj[l] = upload_to_gpu(weights->o_proj[l], num_hidden_sq);
        model->d_fc1[l] = upload_to_gpu(weights->fc1[l], num_ff_hidden);
        model->d_fc2[l] = upload_to_gpu(weights->fc2[l], num_ff_hidden);

        // KV cache
        CUDA_CHECK(cudaMalloc(&model->d_k_cache[l], layer_cache_bytes));
        CUDA_CHECK(cudaMalloc(&model->d_v_cache[l], layer_cache_bytes));
        CUDA_CHECK(cudaMemset(model->d_k_cache[l], 0, layer_cache_bytes));
        CUDA_CHECK(cudaMemset(model->d_v_cache[l], 0, layer_cache_bytes));
        model->kv_len[l] = 0;
    }

    // RoPE position buffer
    model->pos_buf_size = model->max_seq_len;
    CUDA_CHECK(cudaMalloc(&model->d_pos, model->max_seq_len * sizeof(int)));

    fprintf(stderr, "[NanoChat-CUDA] Model loaded: %d layers, %d hidden, %d heads\n",
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
    #undef FREE_GPU
    memset(model, 0, sizeof(*model));
}

// ---------------------------------------------------------------------------
// Reset KV caches
// ---------------------------------------------------------------------------
void nanochat_cuda_model_reset_kv_cache(nanochat_cuda_model_t* model) {
    size_t cache_bytes = (size_t)model->max_seq_len * model->num_kv_heads * model->head_dim * sizeof(float);
    for (int l = 0; l < model->n_layers; l++) {
        model->kv_len[l] = 0;
        CUDA_CHECK(cudaMemset(model->d_k_cache[l], 0, cache_bytes));
        CUDA_CHECK(cudaMemset(model->d_v_cache[l], 0, cache_bytes));
    }
}

// ============================================================================
// Layer prefill (forward for a full sequence through one layer)
// ============================================================================
static void nanochat_layer_prefill_cuda(cublasHandle_t handle,
    float* d_hidden, int seq_len,
    int hidden_size, int num_heads, int head_dim, int ff_dim,
    const float* d_q_w, const float* d_k_w, const float* d_v_w, const float* d_o_w,
    const float* d_fc1_w, const float* d_fc2_w,
    float* d_k_cache, float* d_v_cache, int* kv_len,
    const int* d_pos, float* d_tmp, cudaStream_t stream) {

    int q_size = num_heads * head_dim;
    int kv_size = num_heads * head_dim; // MHA

    // Buffer layout (all offsets from d_tmp):
    float* d_normed  = d_tmp;                                              // [S, H]
    float* d_q       = d_tmp + (size_t)seq_len * hidden_size;              // [S, Q]
    float* d_k       = d_q + (size_t)seq_len * q_size;                    // [S, KV]
    float* d_v       = d_k + (size_t)seq_len * kv_size;                   // [S, KV]
    float* d_ff      = d_v + (size_t)seq_len * kv_size;                   // [S, FF]
    float* d_mlp_out = d_ff + (size_t)seq_len * ff_dim;                   // [S, H]

    // ---- Attention phase ----

    // 1. Pre-attention RMSNorm (no weight)
    rmsnorm_nw_cuda(d_hidden, d_normed, seq_len, hidden_size, NANOCHAT_RMS_EPS, stream);

    // 2. QKV projections
    matmul_bt_cuda(handle, d_normed, d_q_w, d_q, seq_len, hidden_size, q_size);
    matmul_bt_cuda(handle, d_normed, d_k_w, d_k, seq_len, hidden_size, kv_size);
    matmul_bt_cuda(handle, d_normed, d_v_w, d_v, seq_len, hidden_size, kv_size);

    // 3. RoPE (1D standard)
    apply_rope_cuda(d_q, d_k, seq_len, num_heads, num_heads,
                    head_dim, NANOCHAT_ROPE_THETA, d_pos, stream);

    // 4. Store K,V in KV cache
    CUDA_CHECK(cudaMemcpyAsync(d_k_cache, d_k, (size_t)seq_len * kv_size * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v_cache, d_v, (size_t)seq_len * kv_size * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
    *kv_len = seq_len;

    // 5. MHA prefill attention (output to d_q, reusing d_q space)
    float scale = 1.0f / sqrtf((float)head_dim);
    prefill_attention_cuda(handle, d_q, d_k, d_v, d_q,
                           seq_len, num_heads, head_dim, scale, stream);

    // 6. Output projection (result to d_normed, reusing d_normed space)
    matmul_bt_cuda(handle, d_q, d_o_w, d_normed, seq_len, q_size, hidden_size);

    // 7. Residual: d_hidden += d_normed
    {
        int total = seq_len * hidden_size;
        residual_add_cuda(d_hidden, d_normed, total, stream);
    }

    // ---- MLP phase ----

    // 8. Pre-MLP RMSNorm
    rmsnorm_nw_cuda(d_hidden, d_normed, seq_len, hidden_size, NANOCHAT_RMS_EPS, stream);

    // 9. FC1 (ReLU² MLP)
    matmul_bt_cuda(handle, d_normed, d_fc1_w, d_ff, seq_len, hidden_size, ff_dim);

    // 10. ReLU² activation
    relu2_cuda(d_ff, seq_len * ff_dim, stream);

    // 11. FC2
    matmul_bt_cuda(handle, d_ff, d_fc2_w, d_mlp_out, seq_len, ff_dim, hidden_size);

    // 12. Residual: d_hidden += d_mlp_out
    {
        int total = seq_len * hidden_size;
        residual_add_cuda(d_hidden, d_mlp_out, total, stream);
    }
}

// ============================================================================
// Layer decode (single token)
// ============================================================================
static void nanochat_layer_decode_cuda(cublasHandle_t handle,
    float* d_hidden,
    int hidden_size, int num_heads, int head_dim, int ff_dim,
    const float* d_q_w, const float* d_k_w, const float* d_v_w, const float* d_o_w,
    const float* d_fc1_w, const float* d_fc2_w,
    float* d_k_cache, float* d_v_cache, int* h_kv_len,
    const int* d_pos, float* d_tmp, cudaStream_t stream) {

    int q_size = num_heads * head_dim;
    int kv_size = num_heads * head_dim;
    int kv_len = *h_kv_len;

    // Buffer layout (same as prefill but S=1)
    float* d_normed  = d_tmp;
    float* d_q       = d_tmp + hidden_size;
    float* d_k       = d_q + q_size;
    float* d_v       = d_k + kv_size;
    float* d_ff      = d_v + kv_size;
    float* d_mlp_out = d_ff + ff_dim;

    // ---- Attention phase ----

    // 1. Pre-attention RMSNorm
    rmsnorm_nw_cuda(d_hidden, d_normed, 1, hidden_size, NANOCHAT_RMS_EPS, stream);

    // 2. QKV projections
    matmul_bt_cuda(handle, d_normed, d_q_w, d_q, 1, hidden_size, q_size);
    matmul_bt_cuda(handle, d_normed, d_k_w, d_k, 1, hidden_size, kv_size);
    matmul_bt_cuda(handle, d_normed, d_v_w, d_v, 1, hidden_size, kv_size);

    // 3. RoPE
    apply_rope_cuda(d_q, d_k, 1, num_heads, num_heads,
                    head_dim, NANOCHAT_ROPE_THETA, d_pos, stream);

    // 4. Append to KV cache
    CUDA_CHECK(cudaMemcpyAsync(d_k_cache + (size_t)kv_len * kv_size, d_k,
                                (size_t)kv_size * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v_cache + (size_t)kv_len * kv_size, d_v,
                                (size_t)kv_size * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
    (*h_kv_len)++;
    int new_kv_len = kv_len + 1;

    // 5. Decode attention (fused MHA)
    float* d_ctx;
    CUDA_CHECK(cudaMalloc(&d_ctx, (size_t)q_size * sizeof(float)));
    decode_attention_cuda(handle, d_q, d_k_cache, d_v_cache, d_ctx,
                          new_kv_len, num_heads, head_dim, stream);

    // 6. Output projection
    matmul_bt_cuda(handle, d_ctx, d_o_w, d_normed, 1, q_size, hidden_size);
    CUDA_CHECK(cudaFree(d_ctx));

    // 7. Residual
    residual_add_cuda(d_hidden, d_normed, hidden_size, stream);

    // ---- MLP phase ----

    // 8. Pre-MLP RMSNorm
    rmsnorm_nw_cuda(d_hidden, d_normed, 1, hidden_size, NANOCHAT_RMS_EPS, stream);

    // 9. FC1 + ReLU² + FC2
    matmul_bt_cuda(handle, d_normed, d_fc1_w, d_ff, 1, hidden_size, ff_dim);
    relu2_cuda(d_ff, ff_dim, stream);
    matmul_bt_cuda(handle, d_ff, d_fc2_w, d_mlp_out, 1, ff_dim, hidden_size);

    // 10. Residual
    residual_add_cuda(d_hidden, d_mlp_out, hidden_size, stream);
}

// ============================================================================
// Prefill forward: run all layers, return logits on GPU
// ============================================================================
float* nanochat_cuda_model_forward(nanochat_cuda_model_t* model,
                                    float* d_hidden, int seq_len) {
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

    // Allocate temporary buffer (one large alloc for all layers)
    // Layout: S*H + S*Q + S*KV + S*KV + S*FF + S*H
    size_t per_row = (size_t)(hidden_size + num_heads * head_dim +
                               num_heads * head_dim + num_heads * head_dim +
                               ff_dim + hidden_size);
    size_t tmp_size = (size_t)seq_len * per_row * sizeof(float);
    float* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size));

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

        // NaN check after each layer
        {
            float check[2];
            CUDA_CHECK(cudaMemcpy(check, d_hidden + (size_t)(seq_len - 1) * hidden_size,
                                   sizeof(check), cudaMemcpyDeviceToHost));
            if (isnan(check[0]) || isinf(check[0])) {
                fprintf(stderr, "[NanoChat-CUDA] LAYER %d: NaN detected!\n", l + 1);
                break;
            }
        }
    }

    // Final RMSNorm
    rmsnorm_nw_cuda(d_hidden, d_hidden, seq_len, hidden_size, NANOCHAT_RMS_EPS, stream);

    // LM head: logits = hidden[last_pos] @ lm_head^T
    int last_pos = seq_len - 1;
    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)model->vocab_size * sizeof(float)));
    const float* d_last = d_hidden + (size_t)last_pos * hidden_size;
    matmul_bt_cuda(handle, d_last, model->d_lm_head,
                   d_logits, 1, hidden_size, model->vocab_size);

    // Logit softcap
    softcap_cuda(d_logits, model->vocab_size, NANOCHAT_SOFTCAP, stream);

    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return d_logits;
}

// ============================================================================
// Decode step: single token, return logits on GPU
// ============================================================================
float* nanochat_cuda_model_decode(nanochat_cuda_model_t* model,
                                   float* d_embed, int abs_pos) {
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    cudaStream_t stream = 0;

    int hidden_size = model->hidden_size;
    int num_heads = model->num_heads;
    int head_dim = model->head_dim;
    int ff_dim = model->intermediate_size;

    // Set RoPE position for this single token
    CUDA_CHECK(cudaMemcpy(model->d_pos, &abs_pos, sizeof(int),
                           cudaMemcpyHostToDevice));

    // Allocate temp buffer (S=1, small)
    size_t per_row = (size_t)(hidden_size + num_heads * head_dim +
                               num_heads * head_dim + num_heads * head_dim +
                               ff_dim + hidden_size);
    size_t tmp_size = per_row * sizeof(float);
    float* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size));

    // Copy embedding to persistent hidden buffer
    float* d_hidden;
    CUDA_CHECK(cudaMalloc(&d_hidden, (size_t)hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_hidden, d_embed, (size_t)hidden_size * sizeof(float),
                           cudaMemcpyDeviceToDevice));

    // Run layers
    for (int l = 0; l < model->n_layers; l++) {
        float chk[2];
        CUDA_CHECK(cudaMemcpy(chk, d_hidden, sizeof(chk), cudaMemcpyDeviceToHost));
        if (isnan(chk[0]) || isnan(chk[1])) fprintf(stderr, "[DBG] L%d HIDDEN NaN before!\n", l);
        nanochat_layer_decode_cuda(handle, d_hidden,
            hidden_size, num_heads, head_dim, ff_dim,
            model->d_q_proj[l], model->d_k_proj[l],
            model->d_v_proj[l], model->d_o_proj[l],
            model->d_fc1[l], model->d_fc2[l],
            model->d_k_cache[l], model->d_v_cache[l], &model->kv_len[l],
            model->d_pos, d_tmp, stream);
    }

    // Final RMSNorm
    rmsnorm_nw_cuda(d_hidden, d_hidden, 1, hidden_size, NANOCHAT_RMS_EPS, stream);

    // LM head
    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)model->vocab_size * sizeof(float)));
    matmul_bt_cuda(handle, d_hidden, model->d_lm_head,
                   d_logits, 1, hidden_size, model->vocab_size);

    // Logit softcap
    softcap_cuda(d_logits, model->vocab_size, NANOCHAT_SOFTCAP, stream);

    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return d_logits;
}
