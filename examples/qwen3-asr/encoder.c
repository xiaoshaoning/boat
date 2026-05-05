// encoder.c — Qwen3-ASR audio encoder implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include "encoder.h"
#include "config.h"
#include "weights.h"

#include <boat/ops.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <boat/layers/norm.h>
#include <boat/layers/attention.h>

#include <boat/sgemm.h>
#ifdef BOAT_USE_OPENBLAS
#include <cblas.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Manual Conv2D (3x3 kernel, stride 2, padding 1, 1 group)
// input:  [C_in, H, W]  float32
// weight: [C_out, C_in, 3, 3] float32
// bias:   [C_out] float32, or NULL
// output: [C_out, (H+1)/2, (W+1)/2] float32 (stride 2)
// ---------------------------------------------------------------------------
static float* conv2d_stride2(const float *input, int C_in, int H, int W,
                              const float *weight, const float *bias,
                              int C_out) {
    int H_out = (H + 1) / 2;
    int W_out = (W + 1) / 2;
    float *out = (float*)calloc((size_t)C_out * H_out * W_out, sizeof(float));

    for (int co = 0; co < C_out; co++) {
        for (int ho = 0; ho < H_out; ho++) {
            for (int wo = 0; wo < W_out; wo++) {
                float sum = bias ? bias[co] : 0.0f;
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kh = 0; kh < 3; kh++) {
                        int hi = ho * 2 + kh - 1;
                        if (hi < 0 || hi >= H) continue;
                        for (int kw = 0; kw < 3; kw++) {
                            int wi = wo * 2 + kw - 1;
                            if (wi < 0 || wi >= W) continue;
                            // weight layout: [C_out, C_in, 3, 3]
                            float w = weight[((size_t)co * C_in + ci) * 9 + kh * 3 + kw];
                            sum += w * input[((size_t)ci * H + hi) * W + wi];
                        }
                    }
                }
                out[((size_t)co * H_out + ho) * W_out + wo] = sum;
            }
        }
    }
    return out;
}

// In-place GELU
static void gelu_inplace(float *data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float x = data[i];
        float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x3);
        data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// ---------------------------------------------------------------------------
// Layernorm (encoder uses LayerNorm, not RMSNorm)
// x: [N, D], weight: [D], bias: [D], out: [N, D]
// ---------------------------------------------------------------------------
static void layernorm_2d(float *out, const float *x, int N, int D,
                          const float *weight, const float *bias, float eps) {
    for (int i = 0; i < N; i++) {
        const float *row = x + (size_t)i * D;
        float *row_out = out + (size_t)i * D;
        double mean = 0.0, var = 0.0;
        for (int j = 0; j < D; j++) { mean += row[j]; }
        mean /= D;
        for (int j = 0; j < D; j++) { double d = row[j] - mean; var += d * d; }
        var /= D;
        double inv_std = 1.0 / sqrt(var + eps);
        for (int j = 0; j < D; j++) {
            row_out[j] = (float)((row[j] - mean) * inv_std) * weight[j] + (bias ? bias[j] : 0.0f);
        }
    }
}

// Matmul: C[M,N] = A[M,K] @ B[K,N] (row-major), returns raw float* buffer
static float* matmul_f32_ptr(const float *A, const float *B, int M, int K, int N) {
    float *result = (float*)malloc((size_t)M * N * sizeof(float));
    if (!result) return NULL;
    boat_sgemm(M, N, K, A, B, result);
    return result;
}

// Matmul returning boat_tensor_t (for compatibility)
static boat_tensor_t* matmul_2d(const float *A, const float *B, int M, int K, int N) {
    const int64_t a_shape[] = {M, K};
    const int64_t b_shape[] = {K, N};
    boat_tensor_t *a_t = boat_tensor_from_data(a_shape, 2, BOAT_DTYPE_FLOAT32, A);
    boat_tensor_t *b_t = boat_tensor_from_data(b_shape, 2, BOAT_DTYPE_FLOAT32, B);
    boat_tensor_t *c_t = boat_matmul(a_t, b_t);
    boat_tensor_unref(a_t);
    boat_tensor_unref(b_t);
    return c_t;  // [M, N]
}

// ---------------------------------------------------------------------------
// Encoder struct
// ---------------------------------------------------------------------------
struct qwen3asr_encoder_t {
    const qwen3asr_weights_t *w;
};

qwen3asr_encoder_t* qwen3asr_encoder_create(const qwen3asr_weights_t *w) {
    qwen3asr_encoder_t *enc = (qwen3asr_encoder_t*)calloc(1, sizeof(qwen3asr_encoder_t));
    if (!enc) return NULL;
    enc->w = w;
    return enc;
}

void qwen3asr_encoder_free(qwen3asr_encoder_t *enc) {
    if (!enc) return;
    // We don't own the weights — caller manages them
    free(enc);
}

// ---------------------------------------------------------------------------
// Encoder forward
// ---------------------------------------------------------------------------
boat_tensor_t* qwen3asr_encoder_forward(qwen3asr_encoder_t *enc, const boat_tensor_t *mel) {
    const qwen3asr_weights_t *w = enc->w;
    if (boat_tensor_ndim(mel) != 2) return NULL;

    const int64_t *mel_shape = boat_tensor_shape(mel);
    int T = (int)mel_shape[1];  // number of mel frames
    const float *mel_data = (const float*)boat_tensor_data(mel);

    if (mel_shape[0] != 128) return NULL;

    // ---- Step 1: Conv frontend with chunking ----
    int chunk_size = QWEN3ASR_CONV_CHUNK_SIZE;  // 100
    float *all_chunks[1024];  // max chunks
    int chunk_T[1024];
    int num_chunks = 0;

    for (int start = 0; start < T; start += chunk_size) {
        int input_len = (T - start < chunk_size) ? (T - start) : chunk_size;

        // Extract chunk [128, input_len]
        float *chunk = (float*)malloc((size_t)128 * input_len * sizeof(float));
        for (int i = 0; i < 128; i++)
            memcpy(chunk + (size_t)i * input_len,
                   mel_data + (size_t)i * T + start,
                   (size_t)input_len * sizeof(float));

        // Pad if needed: extend by repeating boundary
        int actual_len = input_len;
        if (input_len < chunk_size) {
            // Pad with mean of first column
            float pad_val = 0.0f;
            for (int i = 0; i < 128; i++) pad_val += mel_data[(size_t)i * T];
            pad_val /= 128.0f;
            chunk = (float*)realloc(chunk, (size_t)128 * chunk_size * sizeof(float));
            for (int i = 0; i < 128; i++)
                for (int j = input_len; j < chunk_size; j++)
                    chunk[(size_t)i * chunk_size + j] = pad_val;
            actual_len = chunk_size;
        }

        // Conv1: [128, T] -> [480, T1, F1]
        float *c1 = conv2d_stride2(chunk, 1, 128, actual_len,
                                     w->conv1_w, w->conv1_b, 480);
        gelu_inplace(c1, 480UL * ((128+1)/2) * ((actual_len+1)/2));
        free(chunk);

        // Conv2:
        int H1 = (128+1)/2, W1 = (actual_len+1)/2;
        float *c2 = conv2d_stride2(c1, 480, H1, W1, w->conv2_w, w->conv2_b, 480);
        gelu_inplace(c2, 480UL * ((H1+1)/2) * ((W1+1)/2));
        free(c1);

        // Conv3:
        int H2 = (H1+1)/2, W2 = (W1+1)/2;
        float *c3 = conv2d_stride2(c2, 480, H2, W2, w->conv3_w, w->conv3_b, 480);
        gelu_inplace(c3, 480UL * ((H2+1)/2) * ((W2+1)/2));
        free(c2);

        // Permute + reshape: [480, H3, W3] -> [W3, 480*H3] -> linear -> [W3, 896]
        int H3 = (H2+1)/2, W3 = (W2+1)/2;
        int feat_dim = 480 * H3;  // should be 7680 for standard case

        // Transpose: [480, H3, W3] -> [W3, 480*H3]
        float *reshaped = (float*)malloc((size_t)W3 * feat_dim * sizeof(float));
        for (int t = 0; t < W3; t++)
            for (int f = 0; f < 480; f++)
                for (int h = 0; h < H3; h++)
                    reshaped[(size_t)t * feat_dim + (size_t)f * H3 + h] =
                        c3[((size_t)f * H3 + h) * W3 + t];
        free(c3);

        // Linear projection: [W3, feat_dim] @ [896, feat_dim]^T  -> [W3, 896]
        // Our weight is stored as [896, feat_dim] (boat convention after transpose)
        boat_tensor_t *reshaped_t = boat_tensor_from_data(
            (int64_t[]){W3, feat_dim}, 2, BOAT_DTYPE_FLOAT32, reshaped);
        boat_tensor_t *conv_out_t = boat_tensor_from_data(
            (int64_t[]){feat_dim, 896}, 2, BOAT_DTYPE_FLOAT32, w->conv_out_w);
        boat_tensor_t *proj_t = boat_matmul(reshaped_t, conv_out_t);  // [W3, 896]
        boat_tensor_unref(reshaped_t);
        boat_tensor_unref(conv_out_t);
        free(reshaped);

        if (!proj_t) { /* TODO: cleanup previous chunks */ return NULL; }
        float *proj = (float*)malloc((size_t)W3 * 896 * sizeof(float));
        memcpy(proj, boat_tensor_data(proj_t), (size_t)W3 * 896 * sizeof(float));
        boat_tensor_unref(proj_t);

        // Trim to valid frames (matching Python: trim based on conv strides)
        int feat_len = (input_len - 1) / 2 + 1;
        int temp = (feat_len - 1) / 2 + 1;
        int valid = (temp - 1) / 2 + 1;

        // Add sinusoidal PE
        boat_tensor_t *pe = boat_sinusoidal_embedding((size_t)valid, QWEN3ASR_ENCODER_D_MODEL, 10000.0f);
        if (!pe) { free(proj); /* TODO: cleanup */ return NULL; }
        const float *pe_data = (const float*)boat_tensor_data(pe);

        float *chunk_out = (float*)malloc((size_t)valid * 896 * sizeof(float));
        for (int i = 0; i < valid; i++) {
            for (int j = 0; j < 896; j++) {
                chunk_out[(size_t)i * 896 + j] = proj[(size_t)i * 896 + j] + pe_data[(size_t)i * 896 + j];
            }
        }
        boat_tensor_unref(pe);
        free(proj);

        all_chunks[num_chunks] = chunk_out;
        chunk_T[num_chunks] = valid;
        num_chunks++;
    }

    // Concatenate all chunks along time axis
    int total_T = 0;
    for (int i = 0; i < num_chunks; i++) total_T += chunk_T[i];

    float *encoder_input = (float*)malloc((size_t)total_T * 896 * sizeof(float));
    size_t offset = 0;
    for (int i = 0; i < num_chunks; i++) {
        memcpy(encoder_input + offset, all_chunks[i], (size_t)chunk_T[i] * 896 * sizeof(float));
        free(all_chunks[i]);
        offset += (size_t)chunk_T[i] * 896;
    }

    // ---- Step 2: Transformer encoder layers ----
    float *layer_in = encoder_input;
    float *layer_out = (float*)malloc((size_t)total_T * 896 * sizeof(float));

    for (int l = 0; l < QWEN3ASR_ENCODER_NUM_LAYERS; l++) {
        const qwen3asr_encoder_layer_weights_t *lw = &w->encoder_layers[l];
        int NH = QWEN3ASR_ENCODER_NUM_HEADS;
        int HD = QWEN3ASR_ENCODER_HEAD_DIM;

        // --- Pre-LN Self-Attention (manual MHA) ---
        float *normed = (float*)malloc((size_t)total_T * 896 * sizeof(float));
        layernorm_2d(normed, layer_in, total_T, 896, lw->attn_ln_w, lw->attn_ln_b, 1e-5f);

        // QKV projections
        float *Q = matmul_f32_ptr(normed, lw->q_proj, total_T, 896, 896);
        float *K = matmul_f32_ptr(normed, lw->k_proj, total_T, 896, 896);
        float *V = matmul_f32_ptr(normed, lw->v_proj, total_T, 896, 896);
        free(normed);
        if (!Q || !K || !V) { free(Q); free(K); free(V); free(layer_in); free(layer_out); return NULL; }

        // Add biases
        for (int i = 0; i < total_T * 896; i++) {
            Q[i] += lw->q_bias[i % 896];
            K[i] += lw->k_bias[i % 896];
            V[i] += lw->v_bias[i % 896];
        }

        // Manual MHA: scores = Q @ K^T / sqrt(HD), softmax, weighted sum
        // Q, K, V are [T, 896] = [T, NH*HD]. Working directly with raw pointers.
        int T = total_T;
        float *score = (float*)malloc((size_t)NH * T * T * sizeof(float));
        float scale = 1.0f / sqrtf((float)HD);

#ifdef BOAT_USE_OPENBLAS
        // Q @ K^T per head via BLAS
        for (int h = 0; h < NH; h++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, T, HD, scale,
                        Q + (size_t)h * HD, NH * HD,
                        K + (size_t)h * HD, NH * HD,
                        0.0f,
                        score + (size_t)h * (size_t)T * T, T);
        }
#else
        for (int h = 0; h < NH; h++) {
            for (int ti = 0; ti < T; ti++) {
                for (int tj = 0; tj < T; tj++) {
                    double sum = 0.0;
                    for (int d = 0; d < HD; d++) {
                        sum += (double)Q[((size_t)ti * NH + h) * HD + d] *
                               K[((size_t)tj * NH + h) * HD + d];
                    }
                    score[((size_t)h * T + ti) * T + tj] = (float)(sum * scale);
                }
            }
        }
#endif
        free(Q);
        free(K);

        // Softmax per (head, query_position)
        float *attn_w = (float*)malloc((size_t)NH * T * T * sizeof(float));
        for (int h = 0; h < NH; h++) {
            for (int ti = 0; ti < T; ti++) {
                float max_val = -INFINITY;
                for (int tj = 0; tj < T; tj++) {
                    float v = score[((size_t)h * T + ti) * T + tj];
                    if (v > max_val) max_val = v;
                }
                double sum = 0.0;
                for (int tj = 0; tj < T; tj++) {
                    float e = expf(score[((size_t)h * T + ti) * T + tj] - max_val);
                    attn_w[((size_t)h * T + ti) * T + tj] = e;
                    sum += e;
                }
                double inv = 1.0 / sum;
                for (int tj = 0; tj < T; tj++)
                    attn_w[((size_t)h * T + ti) * T + tj] *= (float)inv;
            }
        }
        free(score);

        // Weighted sum of V
        float *attn_out = (float*)malloc((size_t)T * 896 * sizeof(float));
#ifdef BOAT_USE_OPENBLAS
        for (int h = 0; h < NH; h++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        T, HD, T, 1.0f,
                        attn_w + (size_t)h * (size_t)T * T, T,
                        V + (size_t)h * HD, NH * HD,
                        0.0f,
                        attn_out + (size_t)h * HD, NH * HD);
        }
#else
        for (int h = 0; h < NH; h++) {
            for (int ti = 0; ti < T; ti++) {
                for (int d = 0; d < HD; d++) {
                    double sum = 0.0;
                    for (int tj = 0; tj < T; tj++) {
                        sum += attn_w[((size_t)h * T + ti) * T + tj] *
                               V[((size_t)tj * NH + h) * HD + d];
                    }
                    attn_out[((size_t)ti * NH + h) * HD + d] = (float)sum;
                }
            }
        }
#endif
        free(attn_w);
        free(V);

        // O projection [T, 896] @ [896, 896] -> [T, 896]
        float *o_proj = matmul_f32_ptr(attn_out, lw->o_proj, T, 896, 896);
        free(attn_out);
        if (!o_proj) { free(layer_in); free(layer_out); return NULL; }
        for (int i = 0; i < T * 896; i++) o_proj[i] += lw->o_bias[i % 896];

        // Residual: layer_in + o_proj
        float *residual1 = (float*)malloc((size_t)T * 896 * sizeof(float));
        for (int i = 0; i < T * 896; i++)
            residual1[i] = layer_in[i] + o_proj[i];
        free(o_proj);

        // --- Pre-LN FFN ---
        float *ffn_norm = (float*)malloc((size_t)total_T * 896 * sizeof(float));
        layernorm_2d(ffn_norm, residual1, total_T, 896, lw->final_ln_w, lw->final_ln_b, 1e-5f);

        // fc1: [T,896] @ [3584,896]^T  — our fc1_w is [3584, 896] = [out, in] for boat's matmul convention?
        // Actually, our weights are LOAD_WEIGHT_2D which transposes [out,in] -> [in,out]
        // So fc1_w is [in, out] = [896, 3584]
        // boat_matmul(X, W) = X @ W where X [M,K], W [K,N] => [M,N]
        // We want [T, 896] @ [896, 3584] = [T, 3584]
        boat_tensor_t *fc1_out = matmul_2d(ffn_norm, lw->fc1_w, total_T, 896, 3584);
        if (fc1_out) {
            float *fc1_d = (float*)boat_tensor_data(fc1_out);
            for (int i = 0; i < total_T * 3584; i++)
                fc1_d[i] += lw->fc1_b[i % 3584];
            // GELU
            boat_tensor_t *gelu_out = boat_gelu(fc1_out);
            boat_tensor_unref(fc1_out);
            if (gelu_out) {
                // fc2: [T,3584] @ [3584,896] -> [T,896]
                // fc2_w stored as [in, out] = [3584, 896]
                boat_tensor_t *fc2_out = matmul_2d((const float*)boat_tensor_data(gelu_out),
                                                     lw->fc2_w, total_T, 3584, 896);
                boat_tensor_unref(gelu_out);
                if (fc2_out) {
                    float *fc2_d = (float*)boat_tensor_data(fc2_out);
                    for (int i = 0; i < total_T * 896; i++)
                        fc2_d[i] += lw->fc2_b[i % 896];

                    // Residual: residual1 + fc2_out -> layer_out
                    for (int i = 0; i < total_T * 896; i++)
                        layer_out[i] = residual1[i] + fc2_d[i];

                    boat_tensor_unref(fc2_out);
                }
            }
        }
        free(ffn_norm);
        free(residual1);

        // Swap buffers for next layer
        if (l < QWEN3ASR_ENCODER_NUM_LAYERS - 1) {
            float *tmp = layer_in;
            layer_in = layer_out;
            layer_out = tmp;
        }
    }
    free(layer_in);  // original encoder_input or swapped buffer

    // ---- Step 3: Post-projection ----
    // ln_post: layernorm over [T, 896]
    float *post_norm = (float*)malloc((size_t)total_T * 896 * sizeof(float));
    layernorm_2d(post_norm, layer_out, total_T, 896, w->ln_post_w, w->ln_post_b, 1e-5f);
    free(layer_out);

    // proj1: [T,896] @ [896,896] -> [T,896] + GELU
    float *p1_d = matmul_f32_ptr(post_norm, w->proj1_w, total_T, 896, 896);
    free(post_norm);
    if (!p1_d) return NULL;
    for (int i = 0; i < total_T * 896; i++) p1_d[i] += w->proj1_b[i % 896];
    // Wrap for GELU
    boat_tensor_t *p1 = boat_tensor_from_data((int64_t[]){total_T, 896}, 2, BOAT_DTYPE_FLOAT32, p1_d);
    free(p1_d);
    boat_tensor_t *g1 = boat_gelu(p1);
    boat_tensor_unref(p1);
    if (!g1) return NULL;

    // proj2: [T,896] @ [896,1024] -> [T,1024]
    float *g1_d = (float*)boat_tensor_data(g1);
    float *p2_d = matmul_f32_ptr(g1_d, w->proj2_w, total_T, 896, 1024);
    boat_tensor_unref(g1);
    if (!p2_d) return NULL;
    for (int i = 0; i < total_T * 1024; i++) p2_d[i] += w->proj2_b[i % 1024];

    // Build final [T, 1024] tensor
    const int64_t final_shape[] = {total_T, 1024};
    boat_tensor_t *result = boat_tensor_from_data(final_shape, 2, BOAT_DTYPE_FLOAT32, p2_d);
    free(p2_d);
    return result;  // caller frees
}
