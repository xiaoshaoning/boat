// glm.c - GLM decoder with GQA, RoPE, and KV cache
#include "glm.h"
#include "ocr_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ========== RoPE ==========
void apply_rope_glm(float* q, float* k, int seq_len, int num_heads, int num_kv_heads,
                     int head_dim, float theta) {
    int n_q = seq_len * num_heads;
    int n_k = seq_len * num_kv_heads;

    for (int pos = 0; pos < seq_len; pos++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < head_dim; i += 2) {
                float freq = powf(theta, -(float)i / (float)head_dim);
                float cos_v = cosf(pos * freq);
                float sin_v = sinf(pos * freq);
                int idx = h * head_dim + i;
                float x0 = q[pos * num_heads * head_dim + idx];
                float x1 = q[pos * num_heads * head_dim + idx + 1];
                q[pos * num_heads * head_dim + idx]     = x0 * cos_v - x1 * sin_v;
                q[pos * num_heads * head_dim + idx + 1] = x1 * cos_v + x0 * sin_v;
            }
        }
        if (k) {
            for (int h = 0; h < num_kv_heads; h++) {
                for (int i = 0; i < head_dim; i += 2) {
                    float freq = powf(theta, -(float)i / (float)head_dim);
                    float cos_v = cosf(pos * freq);
                    float sin_v = sinf(pos * freq);
                    int idx = h * head_dim + i;
                    float x0 = k[pos * num_kv_heads * head_dim + idx];
                    float x1 = k[pos * num_kv_heads * head_dim + idx + 1];
                    k[pos * num_kv_heads * head_dim + idx]     = x0 * cos_v - x1 * sin_v;
                    k[pos * num_kv_heads * head_dim + idx + 1] = x1 * cos_v + x0 * sin_v;
                }
            }
        }
    }
}

// ========== GQA Attention with KV Cache ==========
static void gqa_attention(float* out, const float* x, int seq_len, int hidden_size,
                           const float* q_w, const float* k_w, const float* v_w, const float* o_w,
                           int num_heads, int num_kv_heads, int head_dim,
                           glm_kv_cache_t* kv_cache, int use_kv_cache, float rope_theta) {
    int q_size = num_heads * head_dim;       // 2048
    int kv_size = num_kv_heads * head_dim;    // 1024
    int groups = num_heads / num_kv_heads;    // 2

    // QKV projections
    float* q = (float*)malloc(seq_len * q_size * sizeof(float));
    float* k = (float*)malloc(seq_len * kv_size * sizeof(float));
    float* v = (float*)malloc(seq_len * kv_size * sizeof(float));

    matmul_bt(q, x, q_w, seq_len, hidden_size, q_size);
    matmul_bt(k, x, k_w, seq_len, hidden_size, kv_size);
    matmul_bt(v, x, v_w, seq_len, hidden_size, kv_size);

    // Apply RoPE
    apply_rope_glm(q, k, seq_len, num_heads, num_kv_heads, head_dim, rope_theta);

    // KV Cache management
    int total_seq_len;
    if (use_kv_cache) {
        total_seq_len = kv_cache->seq_len + seq_len;
        // Append K to cache
        float* k_cache_data = (float*)boat_tensor_data(kv_cache->k_cache);
        float* v_cache_data = (float*)boat_tensor_data(kv_cache->v_cache);
        for (int s = 0; s < seq_len; s++) {
            memcpy(k_cache_data + (kv_cache->seq_len + s) * kv_size, k + s * kv_size, kv_size * sizeof(float));
            memcpy(v_cache_data + (kv_cache->seq_len + s) * kv_size, v + s * kv_size, kv_size * sizeof(float));
        }
        kv_cache->seq_len = total_seq_len;
    } else {
        total_seq_len = seq_len;
    }

    // Use the appropriate K, V sources
    const float* k_ptr = k;
    const float* v_ptr = v;
    if (use_kv_cache) {
        k_ptr = (const float*)boat_tensor_const_data(kv_cache->k_cache);
        v_ptr = (const float*)boat_tensor_const_data(kv_cache->v_cache);
    }

    // GQA attention scores (per-head)
    // scores layout: [seq_len, num_heads, total_seq_len]
    float* scores = (float*)malloc(seq_len * num_heads * total_seq_len * sizeof(float));

    for (int i = 0; i < seq_len; i++) {
        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / groups;
            for (int j = 0; j < total_seq_len; j++) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    sum += q[i * q_size + h * head_dim + d] * k_ptr[j * kv_size + kv_h * head_dim + d];
                }
                // Apply causal mask for prefill
                if (use_kv_cache || i >= j) {
                    scores[(i * num_heads + h) * total_seq_len + j] = sum / sqrtf((float)head_dim);
                } else {
                    scores[(i * num_heads + h) * total_seq_len + j] = -INFINITY;
                }
            }
        }
    }

    // Softmax per head
    for (int i = 0; i < seq_len; i++) {
        for (int h = 0; h < num_heads; h++) {
            int base = (i * num_heads + h) * total_seq_len;
            float max_val = scores[base];
            for (int j = 1; j < total_seq_len; j++)
                if (scores[base + j] > max_val)
                    max_val = scores[base + j];
            float sum = 0.0f;
            for (int j = 0; j < total_seq_len; j++) {
                scores[base + j] = expf(scores[base + j] - max_val);
                sum += scores[base + j];
            }
            for (int j = 0; j < total_seq_len; j++)
                scores[base + j] /= sum;
        }
    }

    // Weighted sum of values
    float* context = (float*)calloc(seq_len * q_size, sizeof(float));
    for (int i = 0; i < seq_len; i++) {
        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / groups;
            for (int j = 0; j < total_seq_len; j++) {
                float attn = scores[(i * num_heads + h) * total_seq_len + j];
                for (int d = 0; d < head_dim; d++) {
                    context[i * q_size + h * head_dim + d] += attn * v_ptr[j * kv_size + kv_h * head_dim + d];
                }
            }
        }
    }

    // Output projection: context [seq_len, q_size] @ o_w [hidden_size, q_size]^T → [seq_len, hidden_size]
    matmul_bt(out, context, o_w, seq_len, q_size, hidden_size);

    free(q);
    free(k);
    free(v);
    free(scores);
    free(context);
}

// ========== GLM Decoder Layer ==========
static void decoder_layer_forward(float* hidden, int seq_len,
                                   const glm_layer_weights_t* w,
                                   glm_kv_cache_t* kv_cache, int use_kv_cache,
                                   float rope_theta) {
    float* residual = (float*)malloc(seq_len * GLM_HIDDEN_SIZE * sizeof(float));
    memcpy(residual, hidden, seq_len * GLM_HIDDEN_SIZE * sizeof(float));

    // 1. Pre-attention RMSNorm
    const float* in_ln = (const float*)boat_tensor_const_data(w->input_layernorm_weight);
    for (int i = 0; i < seq_len; i++)
        apply_rmsnorm(hidden + i * GLM_HIDDEN_SIZE, residual + i * GLM_HIDDEN_SIZE, in_ln, GLM_HIDDEN_SIZE, 1e-5f);

    // 2. GQA Attention
    float* attn_out = (float*)malloc(seq_len * GLM_HIDDEN_SIZE * sizeof(float));
    const float* q_w = (const float*)boat_tensor_const_data(w->q_proj_weight);
    const float* k_w = (const float*)boat_tensor_const_data(w->k_proj_weight);
    const float* v_w = (const float*)boat_tensor_const_data(w->v_proj_weight);
    const float* o_w = (const float*)boat_tensor_const_data(w->o_proj_weight);
    gqa_attention(attn_out, hidden, seq_len, GLM_HIDDEN_SIZE,
                  q_w, k_w, v_w, o_w,
                  GLM_NUM_HEADS, GLM_NUM_KV_HEADS, GLM_HEAD_DIM,
                  kv_cache, use_kv_cache, rope_theta);

    // 3. Post-self-attention RMSNorm
    const float* psa_ln = (const float*)boat_tensor_const_data(w->post_self_attn_layernorm_weight);
    for (int i = 0; i < seq_len; i++)
        apply_rmsnorm(attn_out + i * GLM_HIDDEN_SIZE, attn_out + i * GLM_HIDDEN_SIZE, psa_ln, GLM_HIDDEN_SIZE, 1e-5f);

    // 4. Residual add (attention path)
    for (int i = 0; i < seq_len * GLM_HIDDEN_SIZE; i++)
        hidden[i] = residual[i] + attn_out[i];
    free(attn_out);

    memcpy(residual, hidden, seq_len * GLM_HIDDEN_SIZE * sizeof(float));

    // 5. Pre-MLP RMSNorm (post_attention_layernorm)
    const float* pa_ln = (const float*)boat_tensor_const_data(w->post_attention_layernorm_weight);
    for (int i = 0; i < seq_len; i++)
        apply_rmsnorm(hidden + i * GLM_HIDDEN_SIZE, residual + i * GLM_HIDDEN_SIZE, pa_ln, GLM_HIDDEN_SIZE, 1e-5f);

    // 6. SiLU FFN with fused gate_up
    const float* gate_up_w = (const float*)boat_tensor_const_data(w->gate_up_proj_weight);
    const float* down_w = (const float*)boat_tensor_const_data(w->down_proj_weight);

    int ff_dim = GLM_INTERMEDIATE_SIZE;  // 4608
    float* gate_up = (float*)malloc(seq_len * 2 * ff_dim * sizeof(float));

    // gate_up is fused: [4608 * 2, 1536] → output [seq_len, 9216]
    // First half is gate, second half is up
    matmul_bt(gate_up, hidden, gate_up_w, seq_len, GLM_HIDDEN_SIZE, 2 * ff_dim);

    for (int i = 0; i < seq_len * ff_dim; i++)
        gate_up[i] = silu(gate_up[i]) * gate_up[i + seq_len * ff_dim];

    // Down projection
    float* mlp_out = (float*)malloc(seq_len * GLM_HIDDEN_SIZE * sizeof(float));
    matmul_bt(mlp_out, gate_up, down_w, seq_len, ff_dim, GLM_HIDDEN_SIZE);
    free(gate_up);

    // 7. Post-MLP RMSNorm
    const float* pm_ln = (const float*)boat_tensor_const_data(w->post_mlp_layernorm_weight);
    for (int i = 0; i < seq_len; i++)
        apply_rmsnorm(mlp_out + i * GLM_HIDDEN_SIZE, mlp_out + i * GLM_HIDDEN_SIZE, pm_ln, GLM_HIDDEN_SIZE, 1e-5f);

    // 8. Residual add (MLP path)
    for (int i = 0; i < seq_len * GLM_HIDDEN_SIZE; i++)
        hidden[i] = residual[i] + mlp_out[i];

    free(residual);
    free(mlp_out);
}

// ========== GLM Forward ==========
boat_tensor_t* glm_forward(glm_model_t* model, const boat_tensor_t* input_ids, int use_kv_cache) {
    const int* ids = (const int*)boat_tensor_const_data(input_ids);
    int seq_len = (int)boat_tensor_nelements(input_ids);
    if (boat_tensor_ndim(input_ids) == 2) {
        const int64_t* shape = boat_tensor_shape(input_ids);
        seq_len = (int)shape[1];
    }

    // 1. Token embedding lookup
    const float* embed = (const float*)boat_tensor_const_data(model->embed_tokens_weight);
    float* hidden = (float*)malloc(seq_len * GLM_HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < seq_len; i++) {
        int id = ids[i];
        if (id < 0 || id >= GLM_VOCAB_SIZE) id = 59246;  // UNK
        memcpy(hidden + i * GLM_HIDDEN_SIZE, embed + id * GLM_HIDDEN_SIZE, GLM_HIDDEN_SIZE * sizeof(float));
    }

    // 2. Decoder layers
    for (int l = 0; l < GLM_NUM_LAYERS; l++) {
        decoder_layer_forward(hidden, seq_len, &model->layers[l],
                              &model->kv_caches[l], use_kv_cache, GLM_ROPE_THETA);
    }

    // 3. Final RMSNorm
    const float* norm_w = (const float*)boat_tensor_const_data(model->norm_weight);
    for (int i = 0; i < seq_len; i++)
        apply_rmsnorm(hidden + i * GLM_HIDDEN_SIZE, hidden + i * GLM_HIDDEN_SIZE, norm_w, GLM_HIDDEN_SIZE, 1e-5f);

    // 4. LM head projection
    const float* lm_w = model->lm_head_weight ?
                        (const float*)boat_tensor_const_data(model->lm_head_weight) :
                        (const float*)boat_tensor_const_data(model->embed_tokens_out_weight);

    int64_t out_shape[] = { 1, seq_len, GLM_VOCAB_SIZE };
    boat_tensor_t* logits = boat_tensor_create(out_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!logits) { free(hidden); return NULL; }

    float* logits_data = (float*)boat_tensor_data(logits);

    // Compute logits for the last position only (for decoding efficiency)
    // During prefill, we need all positions for training, but for inference we only need the last
    if (use_kv_cache && seq_len == 1) {
        // Decode step: only compute logits for the single new token
        int last_pos = 0;
        for (int j = 0; j < GLM_VOCAB_SIZE; j++) {
            float sum = 0.0f;
            for (int k = 0; k < GLM_HIDDEN_SIZE; k++)
                sum += hidden[last_pos * GLM_HIDDEN_SIZE + k] * lm_w[j * GLM_HIDDEN_SIZE + k];
            logits_data[j] = sum;
        }
    } else {
        // Prefill: compute logits for all positions
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < GLM_VOCAB_SIZE; j++) {
                float sum = 0.0f;
                for (int k = 0; k < GLM_HIDDEN_SIZE; k++)
                    sum += hidden[i * GLM_HIDDEN_SIZE + k] * lm_w[j * GLM_HIDDEN_SIZE + k];
                logits_data[i * GLM_VOCAB_SIZE + j] = sum;
            }
        }
    }

    free(hidden);
    return logits;
}

// ========== KV Cache ==========
void glm_kv_cache_reset(glm_model_t* model) {
    for (int l = 0; l < GLM_NUM_LAYERS; l++) {
        glm_kv_cache_t* cache = &model->kv_caches[l];
        cache->seq_len = 0;
        if (cache->k_cache) { memset(boat_tensor_data(cache->k_cache), 0, boat_tensor_nbytes(cache->k_cache)); }
        if (cache->v_cache) { memset(boat_tensor_data(cache->v_cache), 0, boat_tensor_nbytes(cache->v_cache)); }
    }
}

static int create_kv_cache(glm_kv_cache_t* cache) {
    cache->seq_len = 0;
    int64_t k_shape[] = { GLM_MAX_SEQ_LEN, GLM_NUM_KV_HEADS * GLM_HEAD_DIM };
    cache->k_cache = boat_tensor_create(k_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    cache->v_cache = boat_tensor_create(k_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!cache->k_cache || !cache->v_cache) return 0;
    return 1;
}

// ========== Model management ==========

// Helper: load a tensor from safetensors, logging errors
static boat_tensor_t* load_weight(safetensors_t* st, const char* name, int do_transpose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        fprintf(stderr, "[WARN] Weight not found: %s\n", name);
        return NULL;
    }
    boat_tensor_t* t = safetensors_load_tensor(st, idx, do_transpose);
    if (!t) fprintf(stderr, "[ERROR] Failed to load: %s\n", name);
    return t;
}

int glm_load(glm_model_t* model, safetensors_t* st) {
    memset(model, 0, sizeof(*model));

    // Token embeddings
    model->embed_tokens_weight = load_weight(st, "model.language_model.embed_tokens.weight", 0);
    model->norm_weight = load_weight(st, "model.language_model.norm.weight", 0);
    model->lm_head_weight = load_weight(st, "lm_head.weight", 0);
    model->embed_tokens_out_weight = load_weight(st, "model.language_model.layers.16.embed_tokens.weight", 0);

    if (!model->norm_weight) { fprintf(stderr, "[ERROR] Missing norm.weight\n"); return 0; }

    // Per-layer weights
    for (int l = 0; l < GLM_NUM_LAYERS; l++) {
        char name[256];
        glm_layer_weights_t* layer = &model->layers[l];

        snprintf(name, sizeof(name), "model.language_model.layers.%d.input_layernorm.weight", l);
        layer->input_layernorm_weight = load_weight(st, name, 0);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_proj.weight", l);
        layer->q_proj_weight = load_weight(st, name, 0);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_proj.weight", l);
        layer->k_proj_weight = load_weight(st, name, 0);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.v_proj.weight", l);
        layer->v_proj_weight = load_weight(st, name, 0);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.o_proj.weight", l);
        layer->o_proj_weight = load_weight(st, name, 0);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.post_self_attn_layernorm.weight", l);
        layer->post_self_attn_layernorm_weight = load_weight(st, name, 0);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.post_attention_layernorm.weight", l);
        layer->post_attention_layernorm_weight = load_weight(st, name, 0);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.gate_up_proj.weight", l);
        layer->gate_up_proj_weight = load_weight(st, name, 0);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.down_proj.weight", l);
        layer->down_proj_weight = load_weight(st, name, 0);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.post_mlp_layernorm.weight", l);
        layer->post_mlp_layernorm_weight = load_weight(st, name, 0);

        // Verify essential weights
        if (!layer->q_proj_weight || !layer->k_proj_weight || !layer->v_proj_weight || !layer->o_proj_weight) {
            fprintf(stderr, "[ERROR] Missing attention weights for layer %d\n", l);
            return 0;
        }
        if (!layer->gate_up_proj_weight || !layer->down_proj_weight) {
            fprintf(stderr, "[ERROR] Missing MLP weights for layer %d\n", l);
            return 0;
        }

        // Create KV caches
        if (!create_kv_cache(&model->kv_caches[l])) {
            fprintf(stderr, "[ERROR] Failed to create KV cache for layer %d\n", l);
            return 0;
        }
    }

    fprintf(stderr, "[INFO] GLM loaded: %d layers, hidden=%d, heads=%d/%d\n",
            GLM_NUM_LAYERS, GLM_HIDDEN_SIZE, GLM_NUM_HEADS, GLM_NUM_KV_HEADS);
    return 1;
}

static void free_tensor(boat_tensor_t* t) { if (t) boat_tensor_unref(t); }

void glm_free(glm_model_t* model) {
    if (!model) return;
    free_tensor(model->embed_tokens_weight);
    free_tensor(model->norm_weight);
    free_tensor(model->lm_head_weight);
    free_tensor(model->embed_tokens_out_weight);
    for (int l = 0; l < GLM_NUM_LAYERS; l++) {
        glm_layer_weights_t* layer = &model->layers[l];
        free_tensor(layer->input_layernorm_weight);
        free_tensor(layer->q_proj_weight);
        free_tensor(layer->k_proj_weight);
        free_tensor(layer->v_proj_weight);
        free_tensor(layer->o_proj_weight);
        free_tensor(layer->post_self_attn_layernorm_weight);
        free_tensor(layer->post_attention_layernorm_weight);
        free_tensor(layer->gate_up_proj_weight);
        free_tensor(layer->down_proj_weight);
        free_tensor(layer->post_mlp_layernorm_weight);
        free_tensor(model->kv_caches[l].k_cache);
        free_tensor(model->kv_caches[l].v_cache);
    }
    memset(model, 0, sizeof(*model));
}
