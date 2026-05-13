// nougat_decoder.c - Nougat-LaTeX decoder stack and autoregressive generation
#include "nougat_decoder.h"
#include <boat/ops.h>
#include <boat/layers/norm.h>
#include <boat/layers/embedding.h>
#include <boat/sampling.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Internal: token embedding lookup
// ---------------------------------------------------------------------------
// Returns [1, 1, d_model] float32 tensor for a single token ID
static boat_tensor_t* embed_token(const boat_tensor_t* weight, int32_t token_id, boat_device_t device) {
    const float* w = (const float*)boat_tensor_const_data(weight);
    int64_t d_model = boat_tensor_shape(weight)[1];
    int64_t shape[] = { 1, 1, d_model };
    boat_tensor_t* out = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, device);
    if (!out) return NULL;

    float* data = (float*)boat_tensor_data(out);
    if (device == BOAT_DEVICE_CPU) {
        memcpy(data, w + (size_t)token_id * (size_t)d_model, (size_t)d_model * sizeof(float));
    }
#ifdef BOAT_WITH_CUDA
    else if (device == BOAT_DEVICE_CUDA) {
        // Copy from CPU weight to GPU output using CUDA
        float* src_ptr = (float*)boat_tensor_const_data(weight) + (size_t)token_id * (size_t)d_model;
        boat_cuda_copy_to_device(data, src_ptr, (size_t)d_model * sizeof(float));
    }
#endif
    return out;
}

// ---------------------------------------------------------------------------
// Internal: positional embedding lookup
// ---------------------------------------------------------------------------
static boat_tensor_t* embed_position(const boat_tensor_t* weight, int position, boat_device_t device) {
    const float* w = (const float*)boat_tensor_const_data(weight);
    int64_t d_model = boat_tensor_shape(weight)[1];
    int64_t shape[] = { 1, 1, d_model };
    boat_tensor_t* out = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, device);
    if (!out) return NULL;

    float* data = (float*)boat_tensor_data(out);
    if (device == BOAT_DEVICE_CPU) {
        memcpy(data, w + (size_t)position * (size_t)d_model, (size_t)d_model * sizeof(float));
    }
#ifdef BOAT_WITH_CUDA
    else if (device == BOAT_DEVICE_CUDA) {
        float* src_ptr = (float*)boat_tensor_const_data(weight) + (size_t)position * (size_t)d_model;
        boat_cuda_copy_to_device(data, src_ptr, (size_t)d_model * sizeof(float));
    }
#endif
    return out;
}

// ---------------------------------------------------------------------------
// LayerNorm + affine helper (re-implemented here to use weight/bias from model)
// ---------------------------------------------------------------------------
static boat_tensor_t* ln_affine(const boat_tensor_t* x,
                                 const boat_tensor_t* gamma,
                                 const boat_tensor_t* beta,
                                 float eps, boat_device_t device)
{
    (void)device;
    const int64_t* shape = boat_tensor_shape(x);
    int64_t ndim = boat_tensor_ndim(x);
    int64_t D = shape[ndim - 1];
    int64_t ns[] = { D };
    boat_tensor_t* y = boat_layer_norm(x, ns, 1, eps);
    if (!y) return NULL;
    if (gamma) {
        boat_tensor_t* t = boat_mul(y, gamma);
        boat_tensor_unref(y);
        if (!t) return NULL;
        y = t;
    }
    if (beta) {
        boat_tensor_t* t = boat_add(y, beta);
        boat_tensor_unref(y);
        if (!t) return NULL;
        y = t;
    }
    return y;
}

// ---------------------------------------------------------------------------
// Linear: x @ W^T + b (weight is [in, out] from transpose during loading)
// x: [..., in], weight: [in, out], bias: [out]
// Returns: [..., out]
// ---------------------------------------------------------------------------
static boat_tensor_t* linear_fwd(const boat_tensor_t* x,
                                  const boat_tensor_t* weight,
                                  const boat_tensor_t* bias)
{
    size_t ndim = boat_tensor_ndim(x);
    int64_t in_features = boat_tensor_shape(x)[ndim - 1];
    boat_tensor_t* y;

    if (ndim == 2) {
        y = boat_matmul(x, weight);
    } else {
        int64_t outer = 1;
        for (size_t i = 0; i < ndim - 1; i++) outer *= boat_tensor_shape(x)[i];
        int64_t flat_shape[] = { outer, in_features };
        boat_tensor_t* x_flat = boat_tensor_reshape(x, flat_shape, 2);
        if (!x_flat) return NULL;
        y = boat_matmul(x_flat, weight);
        boat_tensor_unref(x_flat);
        if (!y) return NULL;
        int64_t out_features = boat_tensor_shape(y)[1];
        int64_t* orig = (int64_t*)malloc(ndim * sizeof(int64_t));
        memcpy(orig, boat_tensor_shape(x), (ndim - 1) * sizeof(int64_t));
        orig[ndim - 1] = out_features;
        boat_tensor_t* yr = boat_tensor_reshape(y, orig, ndim);
        free(orig);
        boat_tensor_unref(y);
        if (!yr) return NULL;
        y = yr;
    }
    if (!y) return NULL;

    if (bias) {
        boat_tensor_t* t = boat_add(y, bias);
        boat_tensor_unref(y);
        if (!t) return NULL;
        y = t;
    }
    return y;
}

// ---------------------------------------------------------------------------
// Decoder forward: one step through all 10 layers
// input: [1, 1, d_model] float32 (embedded + positioned + LN'd)
// encoder_output: [1, S, 1024] float32
// Returns: [1, 1, d_model] float32 before lm_head
// ---------------------------------------------------------------------------
static boat_tensor_t* decoder_stack_forward(
    const nougat_model_t* model,
    const boat_tensor_t* input,
    const boat_tensor_t* encoder_output,
    boat_decoder_cache_t** caches,
    int step,
    boat_device_t device)
{
    boat_tensor_t* x = (boat_tensor_t*)input;
    boat_tensor_ref(x);

    const boat_decoder_config_t* cfg = &model->decoder_config;

    for (int l = 0; l < model->num_decoder_layers; l++) {
        boat_tensor_t* next = boat_decoder_layer_forward(
            cfg, model->decoder_layers[l], x, encoder_output, caches[l], step);
        boat_tensor_unref(x);
        if (!next) return NULL;
        x = next;
    }

    // Final layer norm
    int64_t D = cfg->d_model;
    int64_t ns[] = { D };
    boat_tensor_t* normed = boat_layer_norm(x, ns, 1, cfg->layer_norm_eps);
    boat_tensor_unref(x);
    if (!normed) return NULL;

    // Apply gamma/beta
    if (model->final_layer_norm_weight) {
        boat_tensor_t* t = boat_mul(normed, model->final_layer_norm_weight);
        boat_tensor_unref(normed);
        if (!t) return NULL;
        normed = t;
    }
    if (model->final_layer_norm_bias) {
        boat_tensor_t* t = boat_add(normed, model->final_layer_norm_bias);
        boat_tensor_unref(normed);
        if (!t) return NULL;
        normed = t;
    }

    return normed;
}

// ---------------------------------------------------------------------------
// Autoregressive generation
// ---------------------------------------------------------------------------
int nougat_decoder_generate(
    const nougat_model_t* model,
    const boat_tensor_t* encoder_output,
    const boat_bpe_tokenizer_t* tokenizer,
    int max_steps,
    boat_device_t device,
    int32_t** out_ids,
    int* out_len)
{
    if (!model || !encoder_output || !out_ids || !out_len) return -1;

    const boat_decoder_config_t* cfg = &model->decoder_config;
    int32_t bos_id = boat_bpe_tokenizer_bos_id(tokenizer);
    int32_t eos_id = boat_bpe_tokenizer_eos_id(tokenizer);
    int64_t D = cfg->d_model;
    float eps = cfg->layer_norm_eps;
    float scale = sqrtf((float)D);

    // Allocate output buffer
    int capacity = 256;
    int32_t* ids = (int32_t*)malloc((size_t)capacity * sizeof(int32_t));
    if (!ids) return -1;
    int n_ids = 0;

    // Create KV caches (on the compute device)
    int32_t batch_size = 1;
    int32_t H = (int32_t)cfg->num_heads;
    int32_t head_dim = (int32_t)(D / H);
    int32_t max_T = (int32_t)(max_steps + 1);
    int32_t enc_seq_len = (int32_t)boat_tensor_shape(encoder_output)[1];

    boat_decoder_cache_t** caches = (boat_decoder_cache_t**)calloc(
        (size_t)model->num_decoder_layers, sizeof(boat_decoder_cache_t*));
    if (!caches) { free(ids); return -1; }

    for (int l = 0; l < model->num_decoder_layers; l++) {
        caches[l] = boat_decoder_cache_create_ex(
            batch_size, H, head_dim, max_T, enc_seq_len, device);
        if (!caches[l]) {
            for (int k = 0; k < l; k++) boat_decoder_cache_free(caches[k]);
            free(caches); free(ids); return -1;
        }
    }

    // ---- Autoregressive loop ----
    int32_t current_id = bos_id;

    for (int step = 0; step < max_steps; step++) {
        // Extend buffer if needed
        if (n_ids >= capacity) {
            capacity *= 2;
            int32_t* tmp = (int32_t*)realloc(ids, (size_t)capacity * sizeof(int32_t));
            if (!tmp) { free(ids); ids = NULL; break; }
            ids = tmp;
        }
        ids[n_ids++] = current_id;
        if (current_id == eos_id) break;

        // Embed current token
        boat_tensor_t* tok_emb = embed_token(model->embed_tokens_weight, current_id, device);
        if (!tok_emb) { free(ids); ids = NULL; break; }

        // Scale embedding
        boat_tensor_t* scaled = boat_mul_scalar(tok_emb, (double)scale);
        boat_tensor_unref(tok_emb);
        if (!scaled) { free(ids); ids = NULL; break; }

        // Add positional embedding (position = step, for single-token decoding)
        boat_tensor_t* pos_emb = embed_position(model->embed_positions_weight, step, device);
        if (!pos_emb) { boat_tensor_unref(scaled); free(ids); ids = NULL; break; }

        boat_tensor_t* emb = boat_add(scaled, pos_emb);
        boat_tensor_unref(scaled);
        boat_tensor_unref(pos_emb);
        if (!emb) { free(ids); ids = NULL; break; }

        // layernorm_embedding (mBART-specific)
        boat_tensor_t* h = ln_affine(emb, model->layernorm_embedding_weight,
                                       model->layernorm_embedding_bias, eps, device);
        boat_tensor_unref(emb);
        if (!h) { free(ids); ids = NULL; break; }

        // Decoder stack
        boat_tensor_t* dec_out = decoder_stack_forward(model, h, encoder_output, caches, step, device);
        boat_tensor_unref(h);
        if (!dec_out) { free(ids); ids = NULL; break; }

        // LM head: [1, 1, D] @ [D, vocab_size] -> [1, 1, vocab_size]
        boat_tensor_t* logits = linear_fwd(dec_out, model->lm_head_weight, NULL);
        boat_tensor_unref(dec_out);
        if (!logits) { free(ids); ids = NULL; break; }

        // Get last token logits (only token in sequence)
        const int64_t* lshape = boat_tensor_shape(logits);
        int vocab_size = (int)lshape[2];

        // Copy logits to CPU for sampling
        float* logits_data = NULL;
        float* local_logits = NULL;
        if (device == BOAT_DEVICE_CPU) {
            logits_data = (float*)boat_tensor_data(logits);
        }
#ifdef BOAT_WITH_CUDA
        else {
            local_logits = (float*)malloc((size_t)vocab_size * sizeof(float));
            if (local_logits) {
                boat_cuda_copy_from_device(local_logits, boat_tensor_data(logits),
                                           (size_t)vocab_size * sizeof(float));
                logits_data = local_logits;
            }
        }
#endif

        if (!logits_data) { boat_tensor_unref(logits); free(ids); ids = NULL; break; }

        // Greedy sampling (temperature = 0)
        current_id = boat_sample_token(logits_data, vocab_size, 0, 0.0f);

        if (local_logits) free(local_logits);
        boat_tensor_unref(logits);

        if (current_id < 0) { free(ids); ids = NULL; break; }
    }

    // Cleanup caches
    for (int l = 0; l < model->num_decoder_layers; l++) {
        if (caches[l]) boat_decoder_cache_free(caches[l]);
    }
    free(caches);

    if (!ids) {
        *out_ids = NULL;
        *out_len = 0;
        return -1;
    }

    *out_ids = ids;
    *out_len = n_ids;
    return 0;
}
