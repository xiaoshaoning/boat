// translator.c - English-to-French translation using MarianMT with Boat
// Pure C inference engine. Loads safetensors weights, runs encoder-decoder.
//
// Build:
//   cd build && cmake .. -DBOAT_WITH_EXAMPLES=ON && make
// Usage:
//   ./examples/translator/translator <model_dir> "Hello, how are you?"
//
// Architecture: Helsinki-NLP/opus-mt-en-fr (MarianMT)
//   6-layer encoder + 6-layer decoder, d_model=512, 8 heads, d_ff=2048

#include <boat.h>
#include <boat/tensor.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <boat/layers/attention.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "safetensors.h"
#include "spm.h"

// ============================================================
// Model Configuration
// ============================================================
#define D_MODEL       512
#define N_HEADS       8
#define HEAD_DIM      64   // D_MODEL / N_HEADS
#define N_LAYERS      6
#define D_FF          2048
#define MAX_POS       512
#define VOCAB_SIZE    59514
#define MAX_OUT_TOKENS 256
#define EMBED_SCALE   22.627416998f  // sqrtf(512.0f)

// ============================================================
// MarianMT Model (loaded weights container)
// ============================================================
typedef struct {
    // Shared embedding weight [VOCAB_SIZE, D_MODEL] — transposed to [D_MODEL, VOCAB_SIZE]
    boat_tensor_t* shared_weight;
    // final_logits_bias [1, VOCAB_SIZE]
    boat_tensor_t* final_bias;

    // Per-layer encoder weights
    boat_tensor_t* enc_q_w[N_LAYERS], *enc_q_b[N_LAYERS];
    boat_tensor_t* enc_k_w[N_LAYERS], *enc_k_b[N_LAYERS];
    boat_tensor_t* enc_v_w[N_LAYERS], *enc_v_b[N_LAYERS];
    boat_tensor_t* enc_o_w[N_LAYERS], *enc_o_b[N_LAYERS];
    boat_tensor_t* enc_attn_ln_w[N_LAYERS], *enc_attn_ln_b[N_LAYERS];
    boat_tensor_t* enc_fc1_w[N_LAYERS], *enc_fc1_b[N_LAYERS];
    boat_tensor_t* enc_fc2_w[N_LAYERS], *enc_fc2_b[N_LAYERS];
    boat_tensor_t* enc_ffn_ln_w[N_LAYERS], *enc_ffn_ln_b[N_LAYERS];

    // Per-layer decoder weights
    boat_tensor_t* dec_self_q_w[N_LAYERS], *dec_self_q_b[N_LAYERS];
    boat_tensor_t* dec_self_k_w[N_LAYERS], *dec_self_k_b[N_LAYERS];
    boat_tensor_t* dec_self_v_w[N_LAYERS], *dec_self_v_b[N_LAYERS];
    boat_tensor_t* dec_self_o_w[N_LAYERS], *dec_self_o_b[N_LAYERS];
    boat_tensor_t* dec_self_ln_w[N_LAYERS], *dec_self_ln_b[N_LAYERS];

    boat_tensor_t* dec_cross_q_w[N_LAYERS], *dec_cross_q_b[N_LAYERS];
    boat_tensor_t* dec_cross_k_w[N_LAYERS], *dec_cross_k_b[N_LAYERS];
    boat_tensor_t* dec_cross_v_w[N_LAYERS], *dec_cross_v_b[N_LAYERS];
    boat_tensor_t* dec_cross_o_w[N_LAYERS], *dec_cross_o_b[N_LAYERS];
    boat_tensor_t* dec_cross_ln_w[N_LAYERS], *dec_cross_ln_b[N_LAYERS];

    boat_tensor_t* dec_fc1_w[N_LAYERS], *dec_fc1_b[N_LAYERS];
    boat_tensor_t* dec_fc2_w[N_LAYERS], *dec_fc2_b[N_LAYERS];
    boat_tensor_t* dec_ffn_ln_w[N_LAYERS], *dec_ffn_ln_b[N_LAYERS];

    // Pre-computed sinusoidal position encoding [MAX_POS, D_MODEL]
    boat_tensor_t* sin_pos;
} marian_t;

// ============================================================
// Forward declarations of internal helpers
// ============================================================
static boat_tensor_t* layer_norm(const boat_tensor_t* x,
                                  const boat_tensor_t* w, const boat_tensor_t* b, float eps);
static boat_tensor_t* linear(const boat_tensor_t* x,
                              const boat_tensor_t* w, const boat_tensor_t* b);
static boat_tensor_t* embedding_lookup(const boat_tensor_t* weight,
                                        const int* ids, int batch, int seq_len);
static boat_tensor_t* make_sin_pos(int max_len, int d_model);

// ============================================================
// Sinusoidal Position Encoding
// ============================================================
static boat_tensor_t* make_sin_pos(int max_len, int d_model) {
    int64_t shape[] = { max_len, d_model };
    boat_tensor_t* pe = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!pe) return NULL;
    float* d = (float*)boat_tensor_data(pe);
    // HuggingFace MarianSinusoidalPositionalEmbedding layout:
    //   First dim//2 entries = sin, second dim//2 entries = cos (not interleaved)
    //   position_enc[pos][j] = pos / 10000^(2 * (j//2) / dim)
    //   out[pos, 0:sentinel]   = sin(position_enc[pos, 0::2])
    //   out[pos, sentinel:]    = cos(position_enc[pos, 1::2])
    int sentinel = d_model / 2;
    for (int pos = 0; pos < max_len; pos++) {
        for (int j = 0; j < d_model; j++) {
            float angle = (float)pos / powf(10000.0f, (2.0f * (float)(j / 2)) / (float)d_model);
            // position_enc[pos][j] = angle
            if (j % 2 == 0) {
                // Even j -> first half (sin)
                d[pos * d_model + j / 2] = sinf(angle);
            } else {
                // Odd j -> second half (cos)
                d[pos * d_model + sentinel + j / 2] = cosf(angle);
            }
        }
    }
    return pe;
}

// ============================================================
// Layer Normalization (Post-LN)
// x: [batch, seq_len, d_model]
// w, b: [d_model]
// ============================================================
static boat_tensor_t* layer_norm(const boat_tensor_t* x,
                                  const boat_tensor_t* w, const boat_tensor_t* b, float eps) {
    const int64_t* shape = boat_tensor_shape(x);
    int64_t ndim = boat_tensor_ndim(x);
    int64_t d = shape[ndim - 1];
    int64_t outer = 1;
    for (int64_t i = 0; i < ndim - 1; i++) outer *= shape[i];

    boat_tensor_t* y = boat_tensor_create(shape, ndim, BOAT_DTYPE_FLOAT32, boat_tensor_device(x));
    if (!y) return NULL;

    const float* xd = (const float*)boat_tensor_const_data(x);
    const float* wd = (const float*)boat_tensor_const_data(w);
    const float* bd = (const float*)boat_tensor_const_data(b);
    float* yd = (float*)boat_tensor_data(y);
    int64_t n = d;

    for (int64_t i = 0; i < outer; i++) {
        float sum = 0.0f, sum2 = 0.0f;
        for (int64_t j = 0; j < n; j++) { sum += xd[i * n + j]; }
        float mean = sum / (float)n;
        for (int64_t j = 0; j < n; j++) {
            float diff = xd[i * n + j] - mean;
            sum2 += diff * diff;
        }
        float var = sum2 / (float)n;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int64_t j = 0; j < n; j++) {
            yd[i * n + j] = (xd[i * n + j] - mean) * inv_std * wd[j] + bd[j];
        }
    }
    return y;
}

// ============================================================
// Linear projection: y = x @ w + b
// x: [batch, seq_len, in_features]
// w: [in_features, out_features] (boat layout)
// b: [out_features] (optional, can be NULL)
// ============================================================
static boat_tensor_t* linear(const boat_tensor_t* x,
                              const boat_tensor_t* w, const boat_tensor_t* b) {
    // boat_matmul requires matching batch dims. Flatten 3D+ to 2D if needed.
    const int64_t* xshape = boat_tensor_shape(x);
    size_t xndim = boat_tensor_ndim(x);
    size_t wndim = boat_tensor_ndim(w);

    boat_tensor_t* y = NULL;

    if (xndim > 2 && wndim == 2) {
        // Flatten: [batch, seq, in_features] -> [batch*seq, in_features]
        int64_t flat_shape[] = { xshape[0] * xshape[1], xshape[2] };
        boat_tensor_t* x_flat = boat_tensor_reshape(x, flat_shape, 2);
        if (!x_flat) return NULL;
        boat_tensor_t* y_flat = boat_matmul(x_flat, w);
        boat_tensor_unref(x_flat);
        if (!y_flat) return NULL;
        // Restore: [batch*seq, out_features] -> [batch, seq, out_features]
        int64_t out_shape[] = { xshape[0], xshape[1], boat_tensor_shape(y_flat)[1] };
        y = boat_tensor_reshape(y_flat, out_shape, 3);
        boat_tensor_unref(y_flat);
    } else {
        y = boat_matmul(x, w);
    }
    if (!y) return NULL;

    if (b) {
        const int64_t* yshape = boat_tensor_shape(y);
        int64_t ydim = boat_tensor_ndim(y);
        int64_t batch = yshape[0];
        int64_t out_dim = yshape[ydim - 1];
        const float* bd = (const float*)boat_tensor_const_data(b);
        float* yd = (float*)boat_tensor_data(y);

        if (ydim == 3) {
            int64_t seq = yshape[1];
            for (int64_t i = 0; i < batch; i++)
                for (int64_t j = 0; j < seq; j++)
                    for (int64_t k = 0; k < out_dim; k++)
                        yd[(i * seq + j) * out_dim + k] += bd[k];
        } else if (ydim == 2) {
            for (int64_t i = 0; i < batch; i++)
                for (int64_t k = 0; k < out_dim; k++)
                    yd[i * out_dim + k] += bd[k];
        }
    }
    return y;
}

// ============================================================
// Embedding lookup with scaling
// weight: [d_model, vocab_size] (transposed from [vocab_size, d_model])
// ============================================================
static boat_tensor_t* embedding_lookup(const boat_tensor_t* weight,
                                        const int* ids, int batch, int seq_len) {
    const int64_t* wshape = boat_tensor_shape(weight);
    int64_t d_model = wshape[0];
    int64_t vocab_size = wshape[1];
    int64_t shape[] = { batch, seq_len, d_model };
    boat_tensor_t* e = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!e) return NULL;
    float* ed = (float*)boat_tensor_data(e);
    const float* wd = (const float*)boat_tensor_const_data(weight);
    // weight is [d_model, vocab_size] (transposed from [vocab_size, d_model])
    // Each column is one embedding, accessed as wd[k * vocab_size + tok]
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < seq_len; j++) {
            int tok = ids[i * seq_len + j];
            if (tok < 0) tok = 0;
            if (tok >= (int)vocab_size) tok = 0;
            float* dst = ed + ((int64_t)i * seq_len + j) * d_model;
            for (int64_t k = 0; k < d_model; k++)
                dst[k] = wd[k * vocab_size + tok] * EMBED_SCALE;
        }
    }
    return e;
}

// ============================================================
// Weight loading from safetensors
// ============================================================
static int load_weight(marian_t* m, safetensors_t* st,
                        const char* name, boat_tensor_t** w, int do_transpose) {
    (void)m;
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        fprintf(stderr, "[ERROR] Missing weight: %s\n", name);
        return 0;
    }
    *w = safetensors_load_tensor(st, idx, do_transpose);
    if (!*w) {
        fprintf(stderr, "[ERROR] Failed to load: %s\n", name);
        return 0;
    }
    return 1;
}

static int load_bias(marian_t* m, safetensors_t* st,
                      const char* name, boat_tensor_t** b) {
    return load_weight(m, st, name, b, 0);
}

static int load_linear(marian_t* m, safetensors_t* st,
                        const char* w_name, const char* b_name,
                        boat_tensor_t** w, boat_tensor_t** b) {
    return load_weight(m, st, w_name, w, 1) && load_bias(m, st, b_name, b);
}

static int load_attn(marian_t* m, safetensors_t* st,
                      const char* prefix, int layer, const char* attn_type,
                      boat_tensor_t** qw, boat_tensor_t** qb,
                      boat_tensor_t** kw, boat_tensor_t** kb,
                      boat_tensor_t** vw, boat_tensor_t** vb,
                      boat_tensor_t** ow, boat_tensor_t** ob) {
    char name[256];
    snprintf(name, sizeof(name), "%s.%d.%s.q_proj.weight", prefix, layer, attn_type);
    if (!load_weight(m, st, name, qw, 1)) return 0;
    snprintf(name, sizeof(name), "%s.%d.%s.q_proj.bias", prefix, layer, attn_type);
    load_bias(m, st, name, qb);

    snprintf(name, sizeof(name), "%s.%d.%s.k_proj.weight", prefix, layer, attn_type);
    load_weight(m, st, name, kw, 1);
    snprintf(name, sizeof(name), "%s.%d.%s.k_proj.bias", prefix, layer, attn_type);
    load_bias(m, st, name, kb);

    snprintf(name, sizeof(name), "%s.%d.%s.v_proj.weight", prefix, layer, attn_type);
    load_weight(m, st, name, vw, 1);
    snprintf(name, sizeof(name), "%s.%d.%s.v_proj.bias", prefix, layer, attn_type);
    load_bias(m, st, name, vb);

    snprintf(name, sizeof(name), "%s.%d.%s.out_proj.weight", prefix, layer, attn_type);
    load_weight(m, st, name, ow, 1);
    snprintf(name, sizeof(name), "%s.%d.%s.out_proj.bias", prefix, layer, attn_type);
    load_bias(m, st, name, ob);
    return 1;
}

static int load_model(marian_t* m, safetensors_t* st) {
    memset(m, 0, sizeof(*m));
    char name[256];

    // Shared embedding
    if (!load_weight(m, st, "model.shared.weight", &m->shared_weight, 1)) return 0;
    if (!load_bias(m, st, "final_logits_bias", &m->final_bias)) return 0;

    // Sinusoidal position encoding
    m->sin_pos = make_sin_pos(MAX_POS, D_MODEL);
    if (!m->sin_pos) return 0;

    // Encoder layers
    for (int i = 0; i < N_LAYERS; i++) {
        // Self-attention
        if (!load_attn(m, st, "model.encoder.layers", i, "self_attn",
                       &m->enc_q_w[i], &m->enc_q_b[i],
                       &m->enc_k_w[i], &m->enc_k_b[i],
                       &m->enc_v_w[i], &m->enc_v_b[i],
                       &m->enc_o_w[i], &m->enc_o_b[i])) return 0;

        // Attention LayerNorm
        snprintf(name, sizeof(name), "model.encoder.layers.%d.self_attn_layer_norm.weight", i);
        if (!load_weight(m, st, name, &m->enc_attn_ln_w[i], 0)) return 0;
        snprintf(name, sizeof(name), "model.encoder.layers.%d.self_attn_layer_norm.bias", i);
        if (!load_bias(m, st, name, &m->enc_attn_ln_b[i])) return 0;

        // FFN
        snprintf(name, sizeof(name), "model.encoder.layers.%d.fc1.weight", i);
        if (!load_weight(m, st, name, &m->enc_fc1_w[i], 1)) return 0;
        snprintf(name, sizeof(name), "model.encoder.layers.%d.fc1.bias", i);
        if (!load_bias(m, st, name, &m->enc_fc1_b[i])) return 0;
        snprintf(name, sizeof(name), "model.encoder.layers.%d.fc2.weight", i);
        if (!load_weight(m, st, name, &m->enc_fc2_w[i], 1)) return 0;
        snprintf(name, sizeof(name), "model.encoder.layers.%d.fc2.bias", i);
        if (!load_bias(m, st, name, &m->enc_fc2_b[i])) return 0;

        // FFN LayerNorm
        snprintf(name, sizeof(name), "model.encoder.layers.%d.final_layer_norm.weight", i);
        if (!load_weight(m, st, name, &m->enc_ffn_ln_w[i], 0)) return 0;
        snprintf(name, sizeof(name), "model.encoder.layers.%d.final_layer_norm.bias", i);
        if (!load_bias(m, st, name, &m->enc_ffn_ln_b[i])) return 0;
    }

    // Decoder layers
    for (int i = 0; i < N_LAYERS; i++) {
        // Self-attention
        if (!load_attn(m, st, "model.decoder.layers", i, "self_attn",
                       &m->dec_self_q_w[i], &m->dec_self_q_b[i],
                       &m->dec_self_k_w[i], &m->dec_self_k_b[i],
                       &m->dec_self_v_w[i], &m->dec_self_v_b[i],
                       &m->dec_self_o_w[i], &m->dec_self_o_b[i])) return 0;

        // Self-attention LayerNorm
        snprintf(name, sizeof(name), "model.decoder.layers.%d.self_attn_layer_norm.weight", i);
        if (!load_weight(m, st, name, &m->dec_self_ln_w[i], 0)) return 0;
        snprintf(name, sizeof(name), "model.decoder.layers.%d.self_attn_layer_norm.bias", i);
        if (!load_bias(m, st, name, &m->dec_self_ln_b[i])) return 0;

        // Cross-attention
        if (!load_attn(m, st, "model.decoder.layers", i, "encoder_attn",
                       &m->dec_cross_q_w[i], &m->dec_cross_q_b[i],
                       &m->dec_cross_k_w[i], &m->dec_cross_k_b[i],
                       &m->dec_cross_v_w[i], &m->dec_cross_v_b[i],
                       &m->dec_cross_o_w[i], &m->dec_cross_o_b[i])) return 0;

        // Cross-attention LayerNorm
        snprintf(name, sizeof(name), "model.decoder.layers.%d.encoder_attn_layer_norm.weight", i);
        if (!load_weight(m, st, name, &m->dec_cross_ln_w[i], 0)) return 0;
        snprintf(name, sizeof(name), "model.decoder.layers.%d.encoder_attn_layer_norm.bias", i);
        if (!load_bias(m, st, name, &m->dec_cross_ln_b[i])) return 0;

        // FFN
        snprintf(name, sizeof(name), "model.decoder.layers.%d.fc1.weight", i);
        if (!load_weight(m, st, name, &m->dec_fc1_w[i], 1)) return 0;
        snprintf(name, sizeof(name), "model.decoder.layers.%d.fc1.bias", i);
        if (!load_bias(m, st, name, &m->dec_fc1_b[i])) return 0;
        snprintf(name, sizeof(name), "model.decoder.layers.%d.fc2.weight", i);
        if (!load_weight(m, st, name, &m->dec_fc2_w[i], 1)) return 0;
        snprintf(name, sizeof(name), "model.decoder.layers.%d.fc2.bias", i);
        if (!load_bias(m, st, name, &m->dec_fc2_b[i])) return 0;

        // FFN LayerNorm
        snprintf(name, sizeof(name), "model.decoder.layers.%d.final_layer_norm.weight", i);
        if (!load_weight(m, st, name, &m->dec_ffn_ln_w[i], 0)) return 0;
        snprintf(name, sizeof(name), "model.decoder.layers.%d.final_layer_norm.bias", i);
        if (!load_bias(m, st, name, &m->dec_ffn_ln_b[i])) return 0;
    }

    return 1;
}

// ============================================================
// Free model
// ============================================================
static void free_model(marian_t* m) {
    if (!m) return;
    #define TF(x) if (x) { boat_tensor_unref(x); x = NULL; }
    TF(m->shared_weight); TF(m->final_bias); TF(m->sin_pos);
    for (int i = 0; i < N_LAYERS; i++) {
        TF(m->enc_q_w[i]); TF(m->enc_q_b[i]); TF(m->enc_k_w[i]); TF(m->enc_k_b[i]);
        TF(m->enc_v_w[i]); TF(m->enc_v_b[i]); TF(m->enc_o_w[i]); TF(m->enc_o_b[i]);
        TF(m->enc_attn_ln_w[i]); TF(m->enc_attn_ln_b[i]);
        TF(m->enc_fc1_w[i]); TF(m->enc_fc1_b[i]); TF(m->enc_fc2_w[i]); TF(m->enc_fc2_b[i]);
        TF(m->enc_ffn_ln_w[i]); TF(m->enc_ffn_ln_b[i]);

        TF(m->dec_self_q_w[i]); TF(m->dec_self_q_b[i]);
        TF(m->dec_self_k_w[i]); TF(m->dec_self_k_b[i]);
        TF(m->dec_self_v_w[i]); TF(m->dec_self_v_b[i]);
        TF(m->dec_self_o_w[i]); TF(m->dec_self_o_b[i]);
        TF(m->dec_self_ln_w[i]); TF(m->dec_self_ln_b[i]);

        TF(m->dec_cross_q_w[i]); TF(m->dec_cross_q_b[i]);
        TF(m->dec_cross_k_w[i]); TF(m->dec_cross_k_b[i]);
        TF(m->dec_cross_v_w[i]); TF(m->dec_cross_v_b[i]);
        TF(m->dec_cross_o_w[i]); TF(m->dec_cross_o_b[i]);
        TF(m->dec_cross_ln_w[i]); TF(m->dec_cross_ln_b[i]);

        TF(m->dec_fc1_w[i]); TF(m->dec_fc1_b[i]);
        TF(m->dec_fc2_w[i]); TF(m->dec_fc2_b[i]);
        TF(m->dec_ffn_ln_w[i]); TF(m->dec_ffn_ln_b[i]);
    }
    #undef TF
}

// ============================================================
// Encoder forward pass
// input_ids: [batch, seq_len] token IDs
// Returns encoder_hidden: [batch, seq_len, d_model]
// ============================================================
static boat_tensor_t* encoder_forward(marian_t* m, const int* input_ids,
                                       int batch, int seq_len) {
    // Embedding + position encoding
    boat_tensor_t* x = embedding_lookup(m->shared_weight, input_ids, batch, seq_len);
    if (!x) return NULL;

    // Add position encoding (slice to seq_len)
    {
        const int64_t* pshape = boat_tensor_shape(m->sin_pos);
        float* xd = (float*)boat_tensor_data(x);
        const float* pd = (const float*)boat_tensor_const_data(m->sin_pos);
        for (int i = 0; i < batch; i++)
            for (int j = 0; j < seq_len && j < pshape[0]; j++)
                for (int k = 0; k < D_MODEL; k++)
                    xd[((int64_t)i * seq_len + j) * D_MODEL + k] += pd[j * D_MODEL + k];
    }

    for (int layer = 0; layer < N_LAYERS; layer++) {
        // Self-attention with residual + post-LayerNorm
        boat_tensor_t* attn_out = NULL;
        {
            // Q, K, V projections
            boat_tensor_t* Q = linear(x, m->enc_q_w[layer], m->enc_q_b[layer]);
            boat_tensor_t* K = linear(x, m->enc_k_w[layer], m->enc_k_b[layer]);
            boat_tensor_t* V = linear(x, m->enc_v_w[layer], m->enc_v_b[layer]);
            if (!Q || !K || !V) {
                boat_tensor_unref(Q); boat_tensor_unref(K); boat_tensor_unref(V);
                boat_tensor_unref(x); return NULL;
            }

            // Reshape to multi-head format
            int64_t mh_shape[] = { batch, seq_len, N_HEADS, HEAD_DIM };
            boat_tensor_t* Q_reshaped = boat_tensor_reshape(Q, mh_shape, 4);
            boat_tensor_t* K_reshaped = boat_tensor_reshape(K, mh_shape, 4);
            boat_tensor_t* V_reshaped = boat_tensor_reshape(V, mh_shape, 4);
            boat_tensor_unref(Q); boat_tensor_unref(K); boat_tensor_unref(V);
            if (!Q_reshaped || !K_reshaped || !V_reshaped) {
                boat_tensor_unref(Q_reshaped); boat_tensor_unref(K_reshaped); boat_tensor_unref(V_reshaped);
                boat_tensor_unref(x); return NULL;
            }

            // Permute [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]
            size_t perm[] = { 0, 2, 1, 3 };
            boat_tensor_t* Q_mh = boat_tensor_transpose(Q_reshaped, perm, 4);
            boat_tensor_t* K_mh = boat_tensor_transpose(K_reshaped, perm, 4);
            boat_tensor_t* V_mh = boat_tensor_transpose(V_reshaped, perm, 4);
            boat_tensor_unref(Q_reshaped); boat_tensor_unref(K_reshaped); boat_tensor_unref(V_reshaped);
            if (!Q_mh || !K_mh || !V_mh) {
                boat_tensor_unref(Q_mh); boat_tensor_unref(K_mh); boat_tensor_unref(V_mh);
                boat_tensor_unref(x); return NULL;
            }

            // Scaled dot-product attention (no masking for encoder)
            float scale = 1.0f / sqrtf((float)HEAD_DIM);
            boat_tensor_t* attn = boat_scaled_dot_product_attention(
                Q_mh, K_mh, V_mh, scale, NULL, 0, 0.0f);
            boat_tensor_unref(Q_mh); boat_tensor_unref(K_mh); boat_tensor_unref(V_mh);
            if (!attn) { boat_tensor_unref(x); return NULL; }

            // Permute back: [batch, n_heads, seq, head_dim] -> [batch, seq, n_heads, head_dim]
            size_t perm_back[] = { 0, 2, 1, 3 };
            boat_tensor_t* attn_reshaped = boat_tensor_transpose(attn, perm_back, 4);
            boat_tensor_unref(attn);
            if (!attn_reshaped) { boat_tensor_unref(x); return NULL; }

            // Reshape to [batch, seq, d_model]
            int64_t final_shape[] = { batch, seq_len, D_MODEL };
            boat_tensor_t* attn_flat = boat_tensor_reshape(attn_reshaped, final_shape, 3);
            boat_tensor_unref(attn_reshaped);
            if (!attn_flat) { boat_tensor_unref(x); return NULL; }

            // Output projection
            attn_out = linear(attn_flat, m->enc_o_w[layer], m->enc_o_b[layer]);
            boat_tensor_unref(attn_flat);
        }
        if (!attn_out) { boat_tensor_unref(x); return NULL; }

        // Residual + LayerNorm
        boat_add_(x, attn_out);
        boat_tensor_unref(attn_out);
        boat_tensor_t* ln1 = layer_norm(x, m->enc_attn_ln_w[layer], m->enc_attn_ln_b[layer], 1e-5f);
        boat_tensor_unref(x);
        if (!ln1) return NULL;
        x = ln1;

        // FFN: fc1 -> SiLU -> fc2
        boat_tensor_t* fc1_out = linear(x, m->enc_fc1_w[layer], m->enc_fc1_b[layer]);
        if (!fc1_out) { boat_tensor_unref(x); return NULL; }
        boat_tensor_t* act = boat_silu(fc1_out);
        boat_tensor_unref(fc1_out);
        if (!act) { boat_tensor_unref(x); return NULL; }
        boat_tensor_t* ffn_out = linear(act, m->enc_fc2_w[layer], m->enc_fc2_b[layer]);
        boat_tensor_unref(act);
        if (!ffn_out) { boat_tensor_unref(x); return NULL; }

        // Residual + LayerNorm
        boat_add_(x, ffn_out);
        boat_tensor_unref(ffn_out);
        boat_tensor_t* ln2 = layer_norm(x, m->enc_ffn_ln_w[layer], m->enc_ffn_ln_b[layer], 1e-5f);
        boat_tensor_unref(x);
        if (!ln2) return NULL;
        x = ln2;
    }

    return x;
}

// ============================================================
// Decoder single step forward
// decoder_state: [batch, seq_len, d_model] (embedded + positioned)
// encoder_out: [batch, enc_seq_len, d_model]
// Returns logits: [batch, seq_len, vocab_size]
// ============================================================
static boat_tensor_t* decoder_step(marian_t* m, const boat_tensor_t* decoder_state,
                                    const boat_tensor_t* encoder_out,
                                    int batch, int seq_len) {
    boat_tensor_t* x = boat_tensor_create_like(decoder_state);
    if (!x) return NULL;
    memcpy(boat_tensor_data(x), boat_tensor_const_data(decoder_state),
           boat_tensor_nbytes(decoder_state));

    for (int layer = 0; layer < N_LAYERS; layer++) {
        // Self-attention (causal)
        boat_tensor_t* Q = linear(x, m->dec_self_q_w[layer], m->dec_self_q_b[layer]);
        boat_tensor_t* K = linear(x, m->dec_self_k_w[layer], m->dec_self_k_b[layer]);
        boat_tensor_t* V = linear(x, m->dec_self_v_w[layer], m->dec_self_v_b[layer]);
        if (!Q || !K || !V) {
            boat_tensor_unref(Q); boat_tensor_unref(K); boat_tensor_unref(V);
            boat_tensor_unref(x); return NULL;
        }

        int64_t mh_shape[] = { batch, seq_len, N_HEADS, HEAD_DIM };
        boat_tensor_t* Q_r = boat_tensor_reshape(Q, mh_shape, 4);
        boat_tensor_t* K_r = boat_tensor_reshape(K, mh_shape, 4);
        boat_tensor_t* V_r = boat_tensor_reshape(V, mh_shape, 4);
        boat_tensor_unref(Q); boat_tensor_unref(K); boat_tensor_unref(V);
        if (!Q_r || !K_r || !V_r) {
            boat_tensor_unref(Q_r); boat_tensor_unref(K_r); boat_tensor_unref(V_r);
            boat_tensor_unref(x); return NULL;
        }

        size_t perm[] = { 0, 2, 1, 3 };
        boat_tensor_t* Q_mh = boat_tensor_transpose(Q_r, perm, 4);
        boat_tensor_t* K_mh = boat_tensor_transpose(K_r, perm, 4);
        boat_tensor_t* V_mh = boat_tensor_transpose(V_r, perm, 4);
        boat_tensor_unref(Q_r); boat_tensor_unref(K_r); boat_tensor_unref(V_r);
        if (!Q_mh || !K_mh || !V_mh) {
            boat_tensor_unref(Q_mh); boat_tensor_unref(K_mh); boat_tensor_unref(V_mh);
            boat_tensor_unref(x); return NULL;
        }

        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        boat_tensor_t* attn = boat_scaled_dot_product_attention(
            Q_mh, K_mh, V_mh, scale, NULL, 1, 0.0f);
        boat_tensor_unref(Q_mh); boat_tensor_unref(K_mh); boat_tensor_unref(V_mh);
        if (!attn) { boat_tensor_unref(x); return NULL; }

        size_t perm_b[] = { 0, 2, 1, 3 };
        boat_tensor_t* attn_r = boat_tensor_transpose(attn, perm_b, 4);
        boat_tensor_unref(attn);
        int64_t flat[] = { batch, seq_len, D_MODEL };
        boat_tensor_t* attn_f = boat_tensor_reshape(attn_r, flat, 3);
        boat_tensor_unref(attn_r);
        boat_tensor_t* self_out = linear(attn_f, m->dec_self_o_w[layer], m->dec_self_o_b[layer]);
        boat_tensor_unref(attn_f);
        if (!self_out) { boat_tensor_unref(x); return NULL; }

        boat_add_(x, self_out);
        boat_tensor_unref(self_out);
        boat_tensor_t* ln1 = layer_norm(x, m->dec_self_ln_w[layer], m->dec_self_ln_b[layer], 1e-5f);
        boat_tensor_unref(x);
        if (!ln1) return NULL;
        x = ln1;

        // Cross-attention (Q from decoder, KV from encoder)
        Q = linear(x, m->dec_cross_q_w[layer], m->dec_cross_q_b[layer]);
        K = linear(encoder_out, m->dec_cross_k_w[layer], m->dec_cross_k_b[layer]);
        V = linear(encoder_out, m->dec_cross_v_w[layer], m->dec_cross_v_b[layer]);
        if (!Q || !K || !V) {
            boat_tensor_unref(Q); boat_tensor_unref(K); boat_tensor_unref(V);
            boat_tensor_unref(x); return NULL;
        }

        int64_t enc_seq_len_val = boat_tensor_shape(encoder_out)[1];
        int64_t cmh_q[] = { batch, seq_len, N_HEADS, HEAD_DIM };
        int64_t cmh_kv[] = { batch, enc_seq_len_val, N_HEADS, HEAD_DIM };
        Q_r = boat_tensor_reshape(Q, cmh_q, 4);
        K_r = boat_tensor_reshape(K, cmh_kv, 4);
        V_r = boat_tensor_reshape(V, cmh_kv, 4);
        boat_tensor_unref(Q); boat_tensor_unref(K); boat_tensor_unref(V);
        Q_mh = boat_tensor_transpose(Q_r, perm, 4);
        K_mh = boat_tensor_transpose(K_r, perm, 4);
        V_mh = boat_tensor_transpose(V_r, perm, 4);
        boat_tensor_unref(Q_r); boat_tensor_unref(K_r); boat_tensor_unref(V_r);

        attn = boat_scaled_dot_product_attention(Q_mh, K_mh, V_mh, scale, NULL, 0, 0.0f);
        boat_tensor_unref(Q_mh); boat_tensor_unref(K_mh); boat_tensor_unref(V_mh);
        if (!attn) { boat_tensor_unref(x); return NULL; }

        attn_r = boat_tensor_transpose(attn, perm_b, 4);
        boat_tensor_unref(attn);
        attn_f = boat_tensor_reshape(attn_r, flat, 3);
        boat_tensor_unref(attn_r);
        boat_tensor_t* cross_out = linear(attn_f, m->dec_cross_o_w[layer], m->dec_cross_o_b[layer]);
        boat_tensor_unref(attn_f);
        if (!cross_out) { boat_tensor_unref(x); return NULL; }

        boat_add_(x, cross_out);
        boat_tensor_unref(cross_out);
        boat_tensor_t* ln2 = layer_norm(x, m->dec_cross_ln_w[layer], m->dec_cross_ln_b[layer], 1e-5f);
        boat_tensor_unref(x);
        if (!ln2) return NULL;
        x = ln2;

        // FFN
        boat_tensor_t* f1 = linear(x, m->dec_fc1_w[layer], m->dec_fc1_b[layer]);
        if (!f1) { boat_tensor_unref(x); return NULL; }
        boat_tensor_t* act = boat_silu(f1);
        boat_tensor_unref(f1);
        boat_tensor_t* f2 = linear(act, m->dec_fc2_w[layer], m->dec_fc2_b[layer]);
        boat_tensor_unref(act);
        if (!f2) { boat_tensor_unref(x); return NULL; }

        boat_add_(x, f2);
        boat_tensor_unref(f2);
        boat_tensor_t* ln3 = layer_norm(x, m->dec_ffn_ln_w[layer], m->dec_ffn_ln_b[layer], 1e-5f);
        boat_tensor_unref(x);
        if (!ln3) return NULL;
        x = ln3;
    }

    boat_tensor_t* logits = linear(x, m->shared_weight, m->final_bias);
    boat_tensor_unref(x);
    if (!logits) return NULL;

    return logits;
}

// ============================================================
// Greedy decoding
// encoder_hidden: [1, enc_len, d_model]
// Returns malloc'd array of output token IDs, *out_len set.
// ============================================================
static int* greedy_decode(marian_t* m, const boat_tensor_t* encoder_hidden,
                           int* out_len) {
    int batch = 1;
    int max_tokens = MAX_OUT_TOKENS;
    int eos_id = 0;  // </s>
    int decoder_start_id = 59513;  // pad_token_id

    int* output = (int*)malloc(max_tokens * sizeof(int));
    if (!output) { *out_len = 0; return NULL; }
    int n = 0;

    // Start with decoder_start_token (pad_token_id in Marian)
    output[n++] = decoder_start_id;

    for (int step = 0; step < max_tokens - 1; step++) {
        // Build decoder input: embed + position
        boat_tensor_t* dec_embed = embedding_lookup(m->shared_weight, output, 1, n);
        if (!dec_embed) break;

        // Add position encoding
        {
            float* dd = (float*)boat_tensor_data(dec_embed);
            const float* pd = (const float*)boat_tensor_const_data(m->sin_pos);
            for (int j = 0; j < n && j < MAX_POS; j++)
                for (int k = 0; k < D_MODEL; k++)
                    dd[j * D_MODEL + k] += pd[j * D_MODEL + k];
        }

        // Decoder step
        boat_tensor_t* logits = decoder_step(m, dec_embed, encoder_hidden, 1, n);
        boat_tensor_unref(dec_embed);
        if (!logits) {
            break;
        }

        // Greedy: get last token's argmax
        const float* ld = (const float*)boat_tensor_const_data(logits);
        int64_t offset = (int64_t)(n - 1) * VOCAB_SIZE;
        int next_token = 0;
        float max_val = ld[offset];
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (ld[offset + i] > max_val) {
                max_val = ld[offset + i];
                next_token = i;
            }
        }
        boat_tensor_unref(logits);

        output[n++] = next_token;
        if (next_token == eos_id) break;
    }

    *out_len = n;
    return output;
}

// ============================================================
// Translate a single text string and print result
// ============================================================
static int translate_and_print(marian_t* model, spm_tokenizer_t* tok,
                                const char* text) {
    int in_len;
    int* in_ids = spm_encode(tok, text, strlen(text), &in_len);
    if (!in_ids || in_len == 0) {
        fprintf(stderr, "[ERROR] Tokenization failed\n");
        free(in_ids);
        return 0;
    }

    // Add </s> token (Marian marks end of input)
    {
        int* ids_extended = (int*)realloc(in_ids, (in_len + 1) * sizeof(int));
        if (ids_extended) {
            ids_extended[in_len++] = 0; // </s> = token 0
            in_ids = ids_extended;
        }
    }

    // Encoder
    boat_tensor_t* enc_out = encoder_forward(model, in_ids, 1, in_len);
    free(in_ids);
    if (!enc_out) {
        fprintf(stderr, "[ERROR] Encoder forward failed\n");
        return 0;
    }

    int out_len;
    int* out_ids = greedy_decode(model, enc_out, &out_len);
    boat_tensor_unref(enc_out);
    if (!out_ids) {
        fprintf(stderr, "[ERROR] Decoding failed\n");
        return 0;
    }

    // Skip decoder_start_token (pad) if present
    int skip = (out_len > 0 && out_ids[0] == 59513) ? 1 : 0;

    // Decode to text
    char* result = spm_decode(tok, out_ids + skip, out_len - skip);
    free(out_ids);
    if (!result) return 0;

    printf("%s\n", result);
    free(result);
    return 1;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    int verbose = 0;
    int arg_offset = 1;

    // Scan for --info flag before positional args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--info") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (arg_offset == 1) {
            arg_offset = i;
            break;
        }
    }
    // If no positional arg found yet, use the first non-flag as model_dir
    if (arg_offset == 1) {
        for (int i = 1; i < argc; i++) {
            if (argv[i][0] != '-') { arg_offset = i; break; }
        }
    }

    if (argc < 2 || arg_offset >= argc) {
        fprintf(stderr, "Usage: translator [--info] <model_dir> [\"English text\"]\n");
        fprintf(stderr, "  or:  translator [--info] <model_dir> (interactive mode)\n");
        return 1;
    }

    const char* model_dir = argv[arg_offset];
    char model_path[1024];
    char vocab_path[1024];
    snprintf(model_path, sizeof(model_path), "%s/model.safetensors", model_dir);
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);

    // Initialize boat
    boat_init();

    // Load tokenizer
    if (verbose) fprintf(stderr, "[INFO] Loading tokenizer...\n");
    spm_tokenizer_t tok;
    if (!spm_init(&tok, vocab_path)) {
        fprintf(stderr, "[ERROR] Failed to load tokenizer\n");
        boat_cleanup();
        return 1;
    }

    // Load model weights
    if (verbose) fprintf(stderr, "[INFO] Loading model (this may take a moment)...\n");
    safetensors_t st;
    if (!safetensors_open(&st, model_path)) {
        spm_free(&tok);
        boat_cleanup();
        return 1;
    }

    marian_t model;
    if (!load_model(&model, &st)) {
        safetensors_close(&st);
        spm_free(&tok);
        free_model(&model);
        boat_cleanup();
        return 1;
    }
    safetensors_close(&st);

    if (verbose) fprintf(stderr, "[INFO] Model loaded successfully\n");

    int text_arg = arg_offset + 1;
    if (argc > text_arg && argv[text_arg][0] != '-') {
        // Single translation
        translate_and_print(&model, &tok, argv[text_arg]);
    } else {
        // Interactive mode
        if (verbose) fprintf(stderr, "[INFO] Interactive mode. Type text to translate, or 'quit' to exit.\n");
        char line[4096];
        while (1) {
            fprintf(stderr, "> ");
            if (!fgets(line, sizeof(line), stdin)) break;
            size_t llen = strlen(line);
            while (llen > 0 && (line[llen-1] == '\n' || line[llen-1] == '\r')) line[--llen] = '\0';
            if (llen == 0) continue;
            if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) break;
            translate_and_print(&model, &tok, line);
        }
    }

    // Cleanup
    free_model(&model);
    spm_free(&tok);
    boat_cleanup();
    return 0;
}
