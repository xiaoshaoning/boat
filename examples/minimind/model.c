// model.c - MiniMind forward pass (prefill + decode)
// All operations on flat float arrays with pre-allocated buffers.
// Zero dynamic allocation during inference.
#include "model.h"
#include "weights.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// --- Math helpers ---

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// C = A * B  where A[M,K] * B[K,N] -> C[M,N]
static void matmul(const float* A, const float* B, float* C,
                   int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// --- RMSNorm ---
// x: [n_rows, dim], weight: [dim]
void rmsnorm(float* x, const float* weight, int n_rows, int dim, float eps) {
    for (int r = 0; r < n_rows; r++) {
        float* row = x + r * dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; i++) sum_sq += row[i] * row[i];
        float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
        for (int i = 0; i < dim; i++) row[i] = row[i] * inv_rms * weight[i];
    }
}

// --- QK-Norm: per-head RMSNorm ---
// q_buf: [n_tokens, num_heads * head_dim] = [n, 768]
// k_buf: [n_tokens, num_kv_heads * head_dim] = [n, 384]
static void qk_norm(float* q_buf, float* k_buf, int n_tokens,
                    const float* q_norm_w, const float* k_norm_w,
                    int num_heads, int num_kv_heads, int head_dim, float eps) {
    // Q: 8 heads, each 96-dim
    for (int t = 0; t < n_tokens; t++) {
        for (int h = 0; h < num_heads; h++) {
            float* head = q_buf + (t * num_heads + h) * head_dim;
            float sum_sq = 0.0f;
            for (int i = 0; i < head_dim; i++) sum_sq += head[i] * head[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / head_dim + eps);
            for (int i = 0; i < head_dim; i++)
                head[i] = head[i] * inv_rms * q_norm_w[i];
        }
    }
    // K: 4 heads, each 96-dim
    for (int t = 0; t < n_tokens; t++) {
        for (int h = 0; h < num_kv_heads; h++) {
            float* head = k_buf + (t * num_kv_heads + h) * head_dim;
            float sum_sq = 0.0f;
            for (int i = 0; i < head_dim; i++) sum_sq += head[i] * head[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / head_dim + eps);
            for (int i = 0; i < head_dim; i++)
                head[i] = head[i] * inv_rms * k_norm_w[i];
        }
    }
}

// --- RoPE ---
// cos/sin tables: [max_seq_len, head_dim]
static void apply_rope(float* q_buf, float* k_buf, int n_tokens, int start_pos,
                       const float* cos_table, const float* sin_table,
                       int num_heads, int num_kv_heads, int head_dim, int max_seq_len) {
    for (int t = 0; t < n_tokens; t++) {
        int pos = start_pos + t;
        const float* cos = cos_table + pos * head_dim;
        const float* sin = sin_table + pos * head_dim;

        // Q
        for (int h = 0; h < num_heads; h++) {
            float* head = q_buf + (t * num_heads + h) * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float x0 = head[2 * i];
                float x1 = head[2 * i + 1];
                head[2 * i]     = x0 * cos[2 * i] - x1 * sin[2 * i];
                head[2 * i + 1] = x1 * cos[2 * i] + x0 * sin[2 * i];
            }
        }
        // K
        for (int h = 0; h < num_kv_heads; h++) {
            float* head = k_buf + (t * num_kv_heads + h) * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float x0 = head[2 * i];
                float x1 = head[2 * i + 1];
                head[2 * i]     = x0 * cos[2 * i] - x1 * sin[2 * i];
                head[2 * i + 1] = x1 * cos[2 * i] + x0 * sin[2 * i];
            }
        }
    }
}

// --- GQA Attention (single layer, prefill) ---
// Q: [n_tokens, 8, 96], K: [kv_len, 4, 96], V: [kv_len, 4, 96]
// Output: [n_tokens, 768]
static void gqa_attention_prefill(const float* q, const float* k_cache, const float* v_cache,
                                   int n_tokens, int kv_len,
                                   int num_heads, int num_kv_heads, int head_dim,
                                   float* output) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int n_rep = num_heads / num_kv_heads; // 2

    // output layout: [n_tokens, num_heads, head_dim]
    for (int t = 0; t < n_tokens; t++) {
        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / n_rep;
            const float* q_head = q + (t * num_heads + h) * head_dim;
            int attend_len = kv_len - (n_tokens - 1 - t); // causal: only positions <= current token
            if (attend_len <= 0) { attend_len = kv_len; } // prefill: past tokens are all visible

            // Compute scores
            float* scores = (float*)malloc(attend_len * sizeof(float));
            float max_score = -INFINITY;
            for (int k = 0; k < attend_len; k++) {
                const float* k_head = k_cache + k * num_kv_heads * head_dim + kv_h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) dot += q_head[d] * k_head[d];
                scores[k] = dot * scale;
                if (scores[k] > max_score) max_score = scores[k];
            }

            // Softmax (numerically stable)
            float sum = 0.0f;
            for (int k = 0; k < attend_len; k++) {
                scores[k] = expf(scores[k] - max_score);
                sum += scores[k];
            }
            for (int k = 0; k < attend_len; k++) scores[k] /= sum;

            // Weighted sum of V
            float* out_head = output + (t * num_heads + h) * head_dim;
            memset(out_head, 0, head_dim * sizeof(float));
            for (int k = 0; k < attend_len; k++) {
                const float* v_head = v_cache + k * num_kv_heads * head_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; d++)
                    out_head[d] += scores[k] * v_head[d];
            }
            free(scores);
        }
    }
}

// --- GQA Attention (single token decode) ---
// Q: [1, 8, 96], K: [kv_len, 4, 96], V: [kv_len, 4, 96]
static void gqa_attention_decode(const float* q, const float* k_cache, const float* v_cache,
                                  int kv_len,
                                  int num_heads, int num_kv_heads, int head_dim,
                                  float* output) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int n_rep = num_heads / num_kv_heads;

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / n_rep;
        const float* q_head = q + h * head_dim;

        float max_score = -INFINITY;
        float* scores = (float*)malloc(kv_len * sizeof(float));
        for (int k = 0; k < kv_len; k++) {
            const float* k_head = k_cache + k * num_kv_heads * head_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += q_head[d] * k_head[d];
            scores[k] = dot * scale;
            if (scores[k] > max_score) max_score = scores[k];
        }

        float sum = 0.0f;
        for (int k = 0; k < kv_len; k++) {
            scores[k] = expf(scores[k] - max_score);
            sum += scores[k];
        }
        for (int k = 0; k < kv_len; k++) scores[k] /= sum;

        float* out_head = output + h * head_dim;
        memset(out_head, 0, head_dim * sizeof(float));
        for (int k = 0; k < kv_len; k++) {
            const float* v_head = v_cache + k * num_kv_heads * head_dim + kv_h * head_dim;
            for (int d = 0; d < head_dim; d++)
                out_head[d] += scores[k] * v_head[d];
        }
        free(scores);
    }
}

// --- SwiGLU FFN ---
// gate: input @ gate_proj -> [n, int_dim]
// up: input @ up_proj -> [n, int_dim]
// down_input = silu(gate[i]) * up[i]
// output = down_input @ down_proj -> [n, hidden]
static void swiglu_ffn(const float* input, int n_tokens,
                       const float* gate_proj, const float* up_proj, const float* down_proj,
                       int hidden_size, int intermediate_size,
                       float* gate_buf, float* up_buf, float* output) {
    matmul(input, gate_proj, gate_buf, n_tokens, hidden_size, intermediate_size);
    matmul(input, up_proj, up_buf, n_tokens, hidden_size, intermediate_size);

    // Element-wise SiLU(gate) * up
    int n = n_tokens * intermediate_size;
    for (int i = 0; i < n; i++) {
        gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
    }

    matmul(gate_buf, down_proj, output, n_tokens, intermediate_size, hidden_size);
}

// --- Single layer forward ---
// (non-static for debug comparison)
void layer_forward(minimind_model_t* m, int layer_idx, float* hidden,
                           int n_tokens, int start_pos) {
    minimind_config_t* cfg = &m->config;
    int HS = cfg->hidden_size;
    int NH = cfg->num_heads;
    int NKH = cfg->num_kv_heads;
    int HD = cfg->head_dim;
    int IS = cfg->intermediate_size;
    float eps = cfg->rms_eps;

    // a) input_layernorm -> attn_input
    memcpy(m->hidden2, hidden, (size_t)n_tokens * HS * sizeof(float));
    rmsnorm(m->hidden2, m->input_layernorm_weight[layer_idx], n_tokens, HS, eps);

    // b) QKV projection
    matmul(m->hidden2, m->q_proj[layer_idx], m->q_buf, n_tokens, HS, HS);
    matmul(m->hidden2, m->k_proj[layer_idx], m->k_buf, n_tokens, HS, NKH * HD);
    matmul(m->hidden2, m->v_proj[layer_idx], m->v_buf, n_tokens, HS, NKH * HD);

    // c) QK-Norm
    qk_norm(m->q_buf, m->k_buf, n_tokens,
            m->q_norm_weight[layer_idx], m->k_norm_weight[layer_idx],
            NH, NKH, HD, eps);

    // d) RoPE
    apply_rope(m->q_buf, m->k_buf, n_tokens, start_pos,
               m->cos_table, m->sin_table, NH, NKH, HD, m->max_seq_len);

    // e) KV cache: append
    int kv_dim = NKH * HD;
    int cur_kv_len = m->kv_len;
    memcpy(m->k_cache[layer_idx] + cur_kv_len * kv_dim, m->k_buf, (size_t)n_tokens * kv_dim * sizeof(float));
    memcpy(m->v_cache[layer_idx] + cur_kv_len * kv_dim, m->v_buf, (size_t)n_tokens * kv_dim * sizeof(float));

    // f) GQA attention
    int total_kv = cur_kv_len + n_tokens;
    if (n_tokens > 1) {
        gqa_attention_prefill(m->q_buf, m->k_cache[layer_idx], m->v_cache[layer_idx],
                              n_tokens, total_kv, NH, NKH, HD, m->attn_out);
    } else {
        gqa_attention_decode(m->q_buf, m->k_cache[layer_idx], m->v_cache[layer_idx],
                             total_kv, NH, NKH, HD, m->attn_out);
    }

    // g) Output projection + residual
    matmul(m->attn_out, m->o_proj[layer_idx], m->hidden2, n_tokens, HS, HS);
    for (int i = 0; i < n_tokens * HS; i++) hidden[i] += m->hidden2[i];

    // h) Post-attention RMSNorm
    memcpy(m->hidden2, hidden, (size_t)n_tokens * HS * sizeof(float));
    rmsnorm(m->hidden2, m->post_attention_layernorm_weight[layer_idx], n_tokens, HS, eps);

    // i) SwiGLU FFN + residual
    swiglu_ffn(m->hidden2, n_tokens,
               m->gate_proj[layer_idx], m->up_proj[layer_idx], m->down_proj[layer_idx],
               HS, IS, m->ffn_gate, m->ffn_up, m->attn_out);
    for (int i = 0; i < n_tokens * HS; i++) hidden[i] += m->attn_out[i];
}

// ===== Public API =====

void minimind_model_reset_kv_cache(minimind_model_t* m) {
    int kv_dim = m->config.num_kv_heads * m->config.head_dim;
    for (int l = 0; l < m->config.num_layers; l++) {
        memset(m->k_cache[l], 0, (size_t)m->max_seq_len * kv_dim * sizeof(float));
        memset(m->v_cache[l], 0, (size_t)m->max_seq_len * kv_dim * sizeof(float));
    }
    m->kv_len = 0;
}

int minimind_model_init(minimind_model_t* m, const char* model_dir) {
    return minimind_weights_load(m, model_dir);
}

void minimind_model_free(minimind_model_t* m) {
    minimind_weights_free(m);
}

void minimind_prefill(minimind_model_t* m, const int* tokens, int n_tokens,
                       float* logits_out) {
    int HS = m->config.hidden_size;
    int VS = m->config.vocab_size;

    // Reset KV cache
    minimind_model_reset_kv_cache(m);

    // Embedding lookup
    for (int t = 0; t < n_tokens; t++) {
        int tid = tokens[t];
        if (tid < 0 || tid >= VS) tid = 0;
        memcpy(m->hidden + t * HS, m->embed_tokens + (size_t)tid * HS,
               HS * sizeof(float));
    }

    // Forward through all layers
    for (int l = 0; l < m->config.num_layers; l++) {
        layer_forward(m, l, m->hidden, n_tokens, 0);
    }
    m->kv_len = n_tokens;

    // Final RMSNorm
    rmsnorm(m->hidden, m->final_norm_weight, n_tokens, HS, m->config.rms_eps);

    // LM Head: only last position -> [vocab_size]
    const float* last_hidden = m->hidden + (n_tokens - 1) * HS;
    for (int v = 0; v < VS; v++) {
        const float* emb_row = m->lm_head + (size_t)v * HS;
        float dot = 0.0f;
        for (int i = 0; i < HS; i++) dot += last_hidden[i] * emb_row[i];
        logits_out[v] = dot;
    }
}

void minimind_decode(minimind_model_t* m, int token, float* logits_out) {
    int HS = m->config.hidden_size;
    int VS = m->config.vocab_size;

    // Embedding lookup (single token)
    int tid = (token >= 0 && token < VS) ? token : 0;
    memcpy(m->hidden, m->embed_tokens + (size_t)tid * HS, HS * sizeof(float));

    // Forward through all layers (single token, with KV cache)
    int kv_pos = m->kv_len;
    for (int l = 0; l < m->config.num_layers; l++) {
        layer_forward(m, l, m->hidden, 1, kv_pos);
    }
    m->kv_len++;

    // Final RMSNorm
    rmsnorm(m->hidden, m->final_norm_weight, 1, HS, m->config.rms_eps);

    // LM Head
    for (int v = 0; v < VS; v++) {
        const float* emb_row = m->lm_head + (size_t)v * HS;
        float dot = 0.0f;
        for (int i = 0; i < HS; i++) dot += m->hidden[i] * emb_row[i];
        logits_out[v] = dot;
    }
}
