// decoder.c — Qwen3-ASR text decoder implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include "decoder.h"
#include "config.h"
#include "weights.h"

#include <boat/ops.h>
#include <boat/tensor.h>
#include <boat/layers/norm.h>
#include <boat/sampling.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// RMSNorm (stateless, raw pointer version for decoder)
// x: [N, D], weight: [D], out: [N, D]
// ---------------------------------------------------------------------------
static void rmsnorm_2d(float *out, const float *x, int N, int D,
                        const float *weight, float eps) {
    for (int i = 0; i < N; i++) {
        const float *row = x + (size_t)i * D;
        float *rout = out + (size_t)i * D;
        double ss = 0.0;
        for (int j = 0; j < D; j++) ss += (double)row[j] * row[j];
        double rms = 1.0 / sqrt(ss / D + eps);
        for (int j = 0; j < D; j++)
            rout[j] = (float)(row[j] * rms) * weight[j];
    }
}

// In-place RMSNorm
static void rmsnorm_inplace(float *x, int N, int D, const float *weight, float eps) {
    float *tmp = (float*)malloc((size_t)N * D * sizeof(float));
    rmsnorm_2d(tmp, x, N, D, weight, eps);
    memcpy(x, tmp, (size_t)N * D * sizeof(float));
    free(tmp);
}

// ---------------------------------------------------------------------------
// RoPE — Qwen3 uses first-half/second-half split (rotate_half), not standard
// even-odd pairing. Pairs (i, i+half) with the same cos/sin frequency.
// x: [T, num_heads, head_dim] float32 (in-place)
// pos_offset: starting position offset
// ---------------------------------------------------------------------------
static void rope_1d(float *x, int T, int num_heads, int head_dim,
                     int pos_offset, float theta) {
    int half = head_dim / 2;
    for (int p = 0; p < T; p++) {
        float pos = (float)(pos_offset + p);
        for (int h = 0; h < num_heads; h++) {
            size_t base = (size_t)((size_t)p * num_heads + h) * head_dim;
            float *xh = x + base;
            for (int i = 0; i < half; i++) {
                float freq = powf(theta, -2.0f * i / (float)head_dim);
                float cos_v = cosf(pos * freq);
                float sin_v = sinf(pos * freq);
                float a = xh[i];
                float b = xh[i + half];
                xh[i]          = a * cos_v - b * sin_v;
                xh[i + half]   = b * cos_v + a * sin_v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Matmul helper (returns new float* buffer)
// ---------------------------------------------------------------------------
static void matmul_f32_scratch(float *result, const float *A, const float *B,
                                int M, int K, int N) {
    size_t n = (size_t)M * N;
    size_t idx = 0;
    while (idx < n) { result[idx] = 0.0f; idx++; }

    int k = 0;
    while (k < K) {
        int i = 0;
        while (i < M) {
            float a_ik = A[(size_t)i * K + k];
            float *row = result + (size_t)i * N;
            const float *b_row = B + (size_t)k * N;
            int j = 0;
            while (j < N) {
                row[j] += a_ik * b_row[j];
                j++;
            }
            i++;
        }
        k++;
    }
}

static float* matmul_f32(const float *A, const float *B, int M, int K, int N) {
    size_t total = (size_t)M * N;
    float *result = (float*)malloc(total * sizeof(float));
    if (!result) return NULL;
    matmul_f32_scratch(result, A, B, M, K, N);
    return result;
}

// ---------------------------------------------------------------------------
// GQA attention (full attention with causal mask)
// Q: [T, num_heads, head_dim], K: [T, num_kv_heads, head_dim], V: same
// Returns: O [T, num_heads * head_dim]
// ---------------------------------------------------------------------------
static float* gqa_attention(const float *Q, const float *K, const float *V,
                             int T, int num_heads, int num_kv_heads, int head_dim,
                             float scale) {
    float *score = (float*)calloc((size_t)num_heads * T * T, sizeof(float));
    if (!score) return NULL;

    int G = num_heads / num_kv_heads;

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / G;
        for (int ti = 0; ti < T; ti++) {
            for (int tj = 0; tj <= ti; tj++) {
                double sum = 0.0;
                for (int d = 0; d < head_dim; d++) {
                    float qv = Q[((size_t)ti * num_heads + h) * head_dim + d];
                    float kv = K[((size_t)tj * num_kv_heads + kv_h) * head_dim + d];
                    sum += (double)qv * kv;
                }
                score[((size_t)h * T + ti) * T + tj] = (float)(sum * scale);
            }
        }
    }

    // Softmax per (head, query_position)
    float *weights = (float*)malloc((size_t)num_heads * T * T * sizeof(float));
    if (!weights) { free(score); return NULL; }
    for (int h = 0; h < num_heads; h++) {
        for (int ti = 0; ti < T; ti++) {
            float max_val = -INFINITY;
            for (int tj = 0; tj <= ti; tj++) {
                float v = score[((size_t)h * T + ti) * T + tj];
                if (v > max_val) max_val = v;
            }
            double sum = 0.0;
            for (int tj = 0; tj <= ti; tj++) {
                float v = score[((size_t)h * T + ti) * T + tj];
                float e = expf(v - max_val);
                weights[((size_t)h * T + ti) * T + tj] = e;
                sum += e;
            }
            double inv_sum = 1.0 / sum;
            for (int tj = 0; tj <= ti; tj++) {
                weights[((size_t)h * T + ti) * T + tj] *= (float)inv_sum;
            }
        }
    }
    free(score);

    // Weighted sum: O_h[ti,d] = sum_{tj} W_h[ti,tj] * V_kv[tj,d]
    float *output = (float*)calloc((size_t)T * num_heads * head_dim, sizeof(float));
    if (!output) { free(weights); return NULL; }

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / G;
        float *wh = weights + (size_t)h * T * T;
        for (int ti = 0; ti < T; ti++) {
            for (int d = 0; d < head_dim; d++) {
                double sum = 0.0;
                for (int tj = 0; tj <= ti; tj++) {
                    sum += wh[(size_t)ti * T + tj] *
                           V[((size_t)tj * num_kv_heads + kv_h) * head_dim + d];
                }
                output[((size_t)ti * num_heads + h) * head_dim + d] = (float)sum;
            }
        }
    }
    free(weights);
    return output;
}

// ---------------------------------------------------------------------------
// KV-cached decode attention
// q: [1, num_heads, head_dim], K_cache: [cache_len, num_kv_heads * head_dim]
// Returns: [1, num_heads * head_dim]
// ---------------------------------------------------------------------------
static float* decode_attention(const float *q, const float *K_cache, const float *V_cache,
                                int cache_len, int num_heads, int num_kv_heads,
                                int head_dim, float scale) {
    int G = num_heads / num_kv_heads;

    float *scores_all = (float*)malloc((size_t)num_heads * cache_len * sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / G;
        for (int j = 0; j < cache_len; j++) {
            double sum = 0.0;
            for (int d = 0; d < head_dim; d++) {
                float qv = q[(size_t)h * head_dim + d];
                float kv = K_cache[((size_t)j * num_kv_heads + kv_h) * head_dim + d];
                sum += (double)qv * kv;
            }
            scores_all[(size_t)h * cache_len + j] = (float)(sum * scale);
        }
    }

    // Softmax
    double *softmax_vals = (double*)calloc((size_t)num_heads * cache_len, sizeof(double));
    for (int h = 0; h < num_heads; h++) {
        float max_val = -INFINITY;
        for (int j = 0; j < cache_len; j++) {
            float v = scores_all[(size_t)h * cache_len + j];
            if (v > max_val) max_val = v;
        }
        double sum = 0.0;
        for (int j = 0; j < cache_len; j++) {
            double e = exp(scores_all[(size_t)h * cache_len + j] - max_val);
            softmax_vals[(size_t)h * cache_len + j] = e;
            sum += e;
        }
        double inv = 1.0 / sum;
        for (int j = 0; j < cache_len; j++)
            softmax_vals[(size_t)h * cache_len + j] *= inv;
    }
    free(scores_all);

    // Weighted sum of V
    float *output = (float*)calloc((size_t)num_heads * head_dim, sizeof(float));
    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / G;
        for (int d = 0; d < head_dim; d++) {
            double sum = 0.0;
            for (int j = 0; j < cache_len; j++) {
                sum += softmax_vals[(size_t)h * cache_len + j]
                     * V_cache[((size_t)j * num_kv_heads + kv_h) * head_dim + d];
            }
            output[(size_t)h * head_dim + d] = (float)sum;
        }
    }
    free(softmax_vals);
    return output;
}

// ---------------------------------------------------------------------------
// Decoder struct
// ---------------------------------------------------------------------------
struct qwen3asr_decoder_t {
    const qwen3asr_weights_t *w;

    // KV cache: [layer][max_seq_len, kv_dim]
    float *k_cache[QWEN3ASR_DECODER_NUM_LAYERS];
    float *v_cache[QWEN3ASR_DECODER_NUM_LAYERS];
    int kv_len;
    int max_seq_len;

    // Pre-computed RoPE cos/sin tables
    float *cos_table;
    float *sin_table;
};

qwen3asr_decoder_t* qwen3asr_decoder_create(const qwen3asr_weights_t *w) {
    qwen3asr_decoder_t *dec = (qwen3asr_decoder_t*)calloc(1, sizeof(qwen3asr_decoder_t));
    if (!dec) return NULL;

    dec->w = w;
    dec->max_seq_len = QWEN3ASR_MAX_SEQ_LEN;
    dec->kv_len = 0;

    int kv_dim = QWEN3ASR_DECODER_NUM_KV_HEADS * QWEN3ASR_DECODER_HEAD_DIM;
    int max_T = dec->max_seq_len;

    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++) {
        dec->k_cache[l] = (float*)calloc((size_t)max_T * kv_dim, sizeof(float));
        dec->v_cache[l] = (float*)calloc((size_t)max_T * kv_dim, sizeof(float));
        if (!dec->k_cache[l] || !dec->v_cache[l]) {
            qwen3asr_decoder_free(dec);
            return NULL;
        }
    }

    return dec;
}

void qwen3asr_decoder_free(qwen3asr_decoder_t *dec) {
    if (!dec) return;
    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++) {
        free(dec->k_cache[l]);
        free(dec->v_cache[l]);
    }
    free(dec);
}

void qwen3asr_decoder_reset_kv(qwen3asr_decoder_t *dec) {
    size_t cache_size = (size_t)dec->max_seq_len *
                        QWEN3ASR_DECODER_NUM_KV_HEADS * QWEN3ASR_DECODER_HEAD_DIM * sizeof(float);
    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++) {
        memset(dec->k_cache[l], 0, cache_size);
        memset(dec->v_cache[l], 0, cache_size);
    }
    dec->kv_len = 0;
}

// ---------------------------------------------------------------------------
// Single decoder layer forward (shared between prefix and decode)
// hidden: [T, 1024] — read-write (output overwrites input when T != 1)
// For T > 1: full prefix mode, fills KV cache from offset 0
// For T == 1: decode mode, appends to KV cache at position cache_offset
// cache_offset: where to store K/V in cache (0 for prefix, kv_len for decode)
// returns 0 on success
// ---------------------------------------------------------------------------
static int decoder_layer_forward(qwen3asr_decoder_t *dec, float *hidden, int T,
                                  const qwen3asr_decoder_layer_weights_t *lw,
                                  int layer_idx, int cache_offset) {
    int H = QWEN3ASR_DECODER_HIDDEN_SIZE;
    int NH = QWEN3ASR_DECODER_NUM_HEADS;
    int NKV = QWEN3ASR_DECODER_NUM_KV_HEADS;
    int HD = QWEN3ASR_DECODER_HEAD_DIM;
    int q_dim = NH * HD;
    int kv_dim = NKV * HD;

    // 1. RMSNorm
    float *normed = (float*)malloc((size_t)T * H * sizeof(float));
    rmsnorm_2d(normed, hidden, T, H, lw->input_ln, QWEN3ASR_RMS_EPS);

    // 2. QKV projections
    float *Q = matmul_f32(normed, lw->q_proj, T, H, q_dim);
    float *K = matmul_f32(normed, lw->k_proj, T, H, kv_dim);
    float *V = matmul_f32(normed, lw->v_proj, T, H, kv_dim);
    free(normed);
    if (!Q || !K || !V) { free(Q); free(K); free(V); return -1; }

    // 3. Per-head Q/K RMSNorm
    {
        float *q_reshaped = (float*)malloc((size_t)T * NH * HD * sizeof(float));
        memcpy(q_reshaped, Q, (size_t)T * NH * HD * sizeof(float));
        rmsnorm_2d(q_reshaped, q_reshaped, T * NH, HD, lw->q_norm, QWEN3ASR_RMS_EPS);
        memcpy(Q, q_reshaped, (size_t)T * NH * HD * sizeof(float));
        free(q_reshaped);

        float *k_reshaped = (float*)malloc((size_t)T * NKV * HD * sizeof(float));
        memcpy(k_reshaped, K, (size_t)T * NKV * HD * sizeof(float));
        rmsnorm_2d(k_reshaped, k_reshaped, T * NKV, HD, lw->k_norm, QWEN3ASR_RMS_EPS);
        memcpy(K, k_reshaped, (size_t)T * NKV * HD * sizeof(float));
        free(k_reshaped);
    }

    // 4. RoPE
    rope_1d(Q, T, NH, HD, cache_offset, QWEN3ASR_ROPE_THETA);
    rope_1d(K, T, NKV, HD, cache_offset, QWEN3ASR_ROPE_THETA);

    // 5. KV cache: store
    for (int t = 0; t < T; t++) {
        memcpy(dec->k_cache[layer_idx] + ((size_t)(cache_offset + t) * NKV * HD),
               K + (size_t)t * NKV * HD, (size_t)NKV * HD * sizeof(float));
        memcpy(dec->v_cache[layer_idx] + ((size_t)(cache_offset + t) * NKV * HD),
               V + (size_t)t * NKV * HD, (size_t)NKV * HD * sizeof(float));
    }

    // 6. Attention
    float *attn_out;
    if (T > 1) {
        attn_out = gqa_attention(Q, K, V, T, NH, NKV, HD, 1.0f / sqrtf((float)HD));
    } else {
        int total_cache = cache_offset + 1;
        attn_out = decode_attention(Q, dec->k_cache[layer_idx], dec->v_cache[layer_idx],
                                     total_cache, NH, NKV, HD, 1.0f / sqrtf((float)HD));
    }
    free(Q);
    free(K);
    free(V);
    if (!attn_out) return -1;

    // 7. O projection
    float *attn_proj = matmul_f32(attn_out, lw->o_proj, T, q_dim, H);
    free(attn_out);
    if (!attn_proj) return -1;

    // 8. Residual
    for (int i = 0; i < T * H; i++) hidden[i] += attn_proj[i];
    free(attn_proj);

    // 9. Post-attention RMSNorm
    float *post_norm = (float*)malloc((size_t)T * H * sizeof(float));
    rmsnorm_2d(post_norm, hidden, T, H, lw->post_attn_ln, QWEN3ASR_RMS_EPS);

    // 10. SiLU-gated MLP
    float *gate = matmul_f32(post_norm, lw->gate_proj, T, H, QWEN3ASR_DECODER_INTERMEDIATE);
    float *up = matmul_f32(post_norm, lw->up_proj, T, H, QWEN3ASR_DECODER_INTERMEDIATE);
    free(post_norm);
    if (!gate || !up) { free(gate); free(up); return -1; }

    // SiLU: gate *= sigmoid(gate)
    {
        int n = T * QWEN3ASR_DECODER_INTERMEDIATE;
        int i = 0;
        while (i < n) {
            gate[i] = gate[i] / (1.0f + expf(-gate[i]));
            i++;
        }
    }

    // Element-wise multiply: gate * up
    {
        int n = T * QWEN3ASR_DECODER_INTERMEDIATE;
        int i = 0;
        while (i < n) {
            gate[i] *= up[i];
            i++;
        }
    }
    free(up);

    // Down projection
    float *mlp_out = matmul_f32(gate, lw->down_proj, T, QWEN3ASR_DECODER_INTERMEDIATE, H);
    free(gate);
    if (!mlp_out) return -1;

    // 11. Residual
    {
        int n = T * H;
        int i = 0;
        while (i < n) {
            hidden[i] += mlp_out[i];
            i++;
        }
    }
    free(mlp_out);

    return 0;
}

// ---------------------------------------------------------------------------
// Prefix forward pass
// ---------------------------------------------------------------------------
boat_tensor_t* qwen3asr_decoder_forward(qwen3asr_decoder_t *dec,
                                          const boat_tensor_t *merged) {
    const float *input_data = (const float*)boat_tensor_data(merged);
    const int64_t *shape = boat_tensor_shape(merged);
    int T = (int)shape[0];
    int H = QWEN3ASR_DECODER_HIDDEN_SIZE;

    float *hidden = (float*)malloc((size_t)T * H * sizeof(float));
    memcpy(hidden, input_data, (size_t)T * H * sizeof(float));

    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++) {
        if (decoder_layer_forward(dec, hidden, T, &dec->w->decoder_layers[l], l, 0) != 0) {
            free(hidden);
            return NULL;
        }
    }

    // Final RMSNorm
    float *final_norm = (float*)malloc((size_t)T * H * sizeof(float));
    rmsnorm_2d(final_norm, hidden, T, H, dec->w->norm_w, QWEN3ASR_RMS_EPS);
    free(hidden);

    // LM head: [T, 1024] @ [1024, 151936] -> [T, 151936]
    int V = QWEN3ASR_VOCAB_SIZE;
    float *logits = matmul_f32(final_norm, dec->w->lm_head, T, H, V);
    free(final_norm);

    if (!logits) return NULL;

    const int64_t logit_shape[] = {1, V};
    boat_tensor_t *result = boat_tensor_from_data(logit_shape, 2, BOAT_DTYPE_FLOAT32, logits + (size_t)(T-1) * V);
    free(logits);
    return result;
}

// ---------------------------------------------------------------------------
// Single-token decode step
// ---------------------------------------------------------------------------
boat_tensor_t* qwen3asr_decoder_step(qwen3asr_decoder_t *dec,
                                      const boat_tensor_t *token_embed, int pos) {
    const float *embed_data = (const float*)boat_tensor_data(token_embed);
    int H = QWEN3ASR_DECODER_HIDDEN_SIZE;

    float *hidden = (float*)malloc((size_t)H * sizeof(float));
    memcpy(hidden, embed_data, (size_t)H * sizeof(float));

    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++) {
        if (decoder_layer_forward(dec, hidden, 1, &dec->w->decoder_layers[l], l, pos) != 0) {
            free(hidden);
            return NULL;
        }
    }

    float *final_norm = (float*)malloc((size_t)H * sizeof(float));
    rmsnorm_2d(final_norm, hidden, 1, H, dec->w->norm_w, QWEN3ASR_RMS_EPS);
    free(hidden);

    int V = QWEN3ASR_VOCAB_SIZE;
    float *logits = matmul_f32(final_norm, dec->w->lm_head, 1, H, V);
    free(final_norm);

    if (!logits) return NULL;

    const int64_t logit_shape[] = {1, V};
    boat_tensor_t *result = boat_tensor_from_data(logit_shape, 2, BOAT_DTYPE_FLOAT32, logits);
    free(logits);
    return result;
}

// ---------------------------------------------------------------------------
// Build prompt: embed text tokens + merge audio features at placeholder
// ---------------------------------------------------------------------------
boat_tensor_t* qwen3asr_build_prompt(const boat_tensor_t *embed_weight,
                                       const boat_tensor_t *audio_features) {
    int prompt_tokens[] = QWEN3ASR_PROMPT_TOKENS;
    int prompt_len = QWEN3ASR_PROMPT_LEN;
    int audio_T = (int)boat_tensor_shape(audio_features)[0];
    int H = QWEN3ASR_DECODER_HIDDEN_SIZE;

    const float *embed_data = (const float*)boat_tensor_data(embed_weight);
    float *embeddings = (float*)calloc((size_t)prompt_len * H, sizeof(float));

    for (int i = 0; i < prompt_len; i++) {
        if (i == QWEN3ASR_PROMPT_PLACEHOLDER_POS) {
            continue;
        }
        int token = prompt_tokens[i];
        if (token >= 0 && token < QWEN3ASR_VOCAB_SIZE) {
            memcpy(embeddings + (size_t)i * H,
                   embed_data + (size_t)token * H,
                   (size_t)H * sizeof(float));
        }
    }

    int before_len = QWEN3ASR_PROMPT_PLACEHOLDER_POS;
    int after_len = prompt_len - QWEN3ASR_PROMPT_PLACEHOLDER_POS - 1;
    int total_len = before_len + audio_T + after_len;

    float *merged = (float*)malloc((size_t)total_len * H * sizeof(float));

    memcpy(merged, embeddings, (size_t)before_len * H * sizeof(float));

    const float *audio_data = (const float*)boat_tensor_data(audio_features);
    memcpy(merged + (size_t)before_len * H, audio_data, (size_t)audio_T * H * sizeof(float));

    memcpy(merged + (size_t)(before_len + audio_T) * H,
           embeddings + (size_t)(before_len + 1) * H,
           (size_t)after_len * H * sizeof(float));

    free(embeddings);

    const int64_t merged_shape[] = {total_len, H};
    boat_tensor_t *result = boat_tensor_from_data(merged_shape, 2, BOAT_DTYPE_FLOAT32, merged);
    free(merged);
    return result;
}

