// transformer_decoder.c - Cross-attention decoder layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers/transformer_decoder.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <boat/layers/norm.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef BOAT_WITH_CUDA
#include <boat/cuda_runtime.h>
#endif

// =========================================================================
// Internal helpers
// =========================================================================

static float* data_f32(const boat_tensor_t* t) {
    return (float*)boat_tensor_data(t);
}

// LayerNorm + affine on last dim
static boat_tensor_t* layer_norm_affine(const boat_tensor_t* x, const boat_tensor_t* gamma,
                                        const boat_tensor_t* beta, float eps) {
    const int64_t* shape = boat_tensor_shape(x);
    int64_t ndim = boat_tensor_ndim(x);
    int64_t D = shape[ndim - 1];
    int64_t ns[] = {D};
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

// Linear: y = x @ W + b
// x: [..., in_features], W: [in_features, out_features], b: [out_features]
static boat_tensor_t* linear(const boat_tensor_t* x, const boat_tensor_t* weight,
                             const boat_tensor_t* bias) {
    size_t ndim = boat_tensor_ndim(x);
    int64_t in_features = boat_tensor_shape(x)[ndim - 1];
    boat_tensor_t* y;

    if (ndim == 2) {
        y = boat_matmul(x, weight);
    } else {
        // Flatten leading dims for matmul (needs matching batch dims)
        int64_t outer = 1;
        for (size_t i = 0; i < ndim - 1; i++)
            outer *= boat_tensor_shape(x)[i];
        int64_t flat_shape[] = {outer, in_features};
        boat_tensor_t* x_flat = boat_tensor_reshape(x, flat_shape, 2);
        if (!x_flat) return NULL;

        y = boat_matmul(x_flat, weight);
        boat_tensor_unref(x_flat);
        if (!y) return NULL;

        // Restore leading dims
        int64_t out_features = boat_tensor_shape(y)[1];
        int64_t* orig = (int64_t*)malloc((size_t)ndim * sizeof(int64_t));
        memcpy(orig, boat_tensor_shape(x), (size_t)(ndim - 1) * sizeof(int64_t));
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

// Fused attention for decoder self-attention with KV cache support.
// x: [B, T, D]
// cache_k, cache_v: [B, num_heads, max_T, head_dim]
// step: current decoding step
// Returns: [B, T, D]
static boat_tensor_t* self_attention(const boat_tensor_t* x, const boat_decoder_config_t* config,
                                     const boat_decoder_layer_weights_t* w,
                                     boat_decoder_cache_t* cache, int32_t step, bool causal) {
    int B = (int)boat_tensor_shape(x)[0];
    int T = (int)boat_tensor_shape(x)[1];
    int D = config->d_model;
    int H = config->num_heads;
    int head_dim = D / H;

    // Q, K, V projections
    boat_tensor_t* Q = linear(x, w->self_q_weight, w->self_q_bias);
    if (!Q) return NULL;
    boat_tensor_t* K = linear(x, w->self_k_weight, w->self_k_bias);
    if (!K) {
        boat_tensor_unref(Q);
        return NULL;
    }
    boat_tensor_t* V = linear(x, w->self_v_weight, w->self_v_bias);
    if (!V) {
        boat_tensor_unref(Q);
        boat_tensor_unref(K);
        return NULL;
    }

    // Reshape to [B, T, H, head_dim] -> [B, H, T, head_dim]
    int64_t mh_shape[] = {B, T, H, head_dim};
    boat_tensor_t* Q_mh = boat_tensor_reshape(Q, mh_shape, 4);
    boat_tensor_unref(Q);
    if (!Q_mh) {
        boat_tensor_unref(K);
        boat_tensor_unref(V);
        return NULL;
    }
    Q_mh = boat_transpose(Q_mh, 1, 2);
    if (!Q_mh) {
        boat_tensor_unref(K);
        boat_tensor_unref(V);
        return NULL;
    }

    boat_tensor_t* K_mh = boat_tensor_reshape(K, mh_shape, 4);
    boat_tensor_unref(K);
    if (!K_mh) {
        boat_tensor_unref(Q_mh);
        boat_tensor_unref(V);
        return NULL;
    }
    K_mh = boat_transpose(K_mh, 1, 2);
    if (!K_mh) {
        boat_tensor_unref(Q_mh);
        boat_tensor_unref(V);
        return NULL;
    }

    boat_tensor_t* V_mh = boat_tensor_reshape(V, mh_shape, 4);
    boat_tensor_unref(V);
    if (!V_mh) {
        boat_tensor_unref(Q_mh);
        boat_tensor_unref(K_mh);
        return NULL;
    }
    V_mh = boat_transpose(V_mh, 1, 2);
    if (!V_mh) {
        boat_tensor_unref(Q_mh);
        boat_tensor_unref(K_mh);
        return NULL;
    }

    // KV cache: append new K, V
    if (cache && step >= 0) {
        int cache_max_T = cache->max_length;
#ifdef BOAT_WITH_CUDA
        if (boat_tensor_device(x) == BOAT_DEVICE_CUDA) {
            float* kd = data_f32(K_mh);
            float* vd = data_f32(V_mh);
            float* ckd = data_f32(cache->self_k);
            float* cvd = data_f32(cache->self_v);
            if (kd && vd && ckd && cvd) {
                boat_cuda_kv_cache_append_f32(kd, ckd, B, H, T, head_dim, cache_max_T, step);
                boat_cuda_kv_cache_append_f32(vd, cvd, B, H, T, head_dim, cache_max_T, step);
            }
        } else
#endif
        {
            float* k_data = data_f32(K_mh);
            float* v_data = data_f32(V_mh);
            float* ck_data = data_f32(cache->self_k);
            float* cv_data = data_f32(cache->self_v);
            if (k_data && v_data && ck_data && cv_data) {
                for (int b = 0; b < B; b++)
                    for (int h = 0; h < H; h++)
                        for (int t = 0; t < T; t++)
                            for (int d = 0; d < head_dim; d++) {
                                int src = ((b * H + h) * T + t) * head_dim + d;
                                int dst = ((b * H + h) * cache_max_T + step + t) * head_dim + d;
                                ck_data[dst] = k_data[src];
                                cv_data[dst] = v_data[src];
                            }
            }
        }
        cache->length = step + T;

        // Use full cache for attention
        int L = cache->length;
        int64_t k_full_shape[] = {B, H, L, head_dim};
        boat_tensor_t* K_full =
            boat_tensor_create(k_full_shape, 4, BOAT_DTYPE_FLOAT32, boat_tensor_device(x));
        boat_tensor_t* V_full =
            boat_tensor_create(k_full_shape, 4, BOAT_DTYPE_FLOAT32, boat_tensor_device(x));
        if (K_full && V_full) {
#ifdef BOAT_WITH_CUDA
            if (boat_tensor_device(x) == BOAT_DEVICE_CUDA) {
                float* ckd = data_f32(cache->self_k);
                float* cvd = data_f32(cache->self_v);
                float* kfd = data_f32(K_full);
                float* vfd = data_f32(V_full);
                if (ckd && cvd && kfd && vfd) {
                    boat_cuda_kv_cache_extract_f32(ckd, kfd, B, H, L, head_dim, cache_max_T);
                    boat_cuda_kv_cache_extract_f32(cvd, vfd, B, H, L, head_dim, cache_max_T);
                }
            } else
#endif
            {
                float* ck_data = data_f32(cache->self_k);
                float* cv_data = data_f32(cache->self_v);
                float* kf = data_f32(K_full);
                float* vf = data_f32(V_full);
                if (kf && vf && ck_data && cv_data) {
                    for (int i = 0; i < B * H * L * head_dim; i++) {
                        int bi = i / (H * L * head_dim);
                        int hi = (i / (L * head_dim)) % H;
                        int ti = (i / head_dim) % L;
                        int di = i % head_dim;
                        int src_off = ((bi * H + hi) * cache_max_T + ti) * head_dim + di;
                        kf[i] = ck_data[src_off];
                        vf[i] = cv_data[src_off];
                    }
                }
            }
            boat_tensor_unref(K_mh);
            K_mh = K_full;
            boat_tensor_unref(V_mh);
            V_mh = V_full;
        } else {
            if (K_full) boat_tensor_unref(K_full);
            if (V_full) boat_tensor_unref(V_full);
        }
    }

    // Scaled dot-product attention
    // Q: [B, H, T, head_dim], K: [B, H, L, head_dim], V: [B, H, L, head_dim]
    // scores = Q @ K^T / sqrt(head_dim)
    int L = (int)boat_tensor_shape(K_mh)[2];

    // Flatten to 2D batch matmul
    float scale = 1.0f / sqrtf((float)head_dim);

    // scores = Q @ K^T  -> [B*H, T, L]
    boat_tensor_t* K_T = boat_transpose(K_mh, 2, 3);
    if (!K_T) {
        boat_tensor_unref(Q_mh);
        boat_tensor_unref(K_mh);
        boat_tensor_unref(V_mh);
        return NULL;
    }
    boat_tensor_unref(K_mh);
    K_mh = NULL;

    // Flatten Q: [B, H, T, head_dim] -> [B*H, T, head_dim]
    // Flatten K_T: [B, H, head_dim, L] -> [B*H, head_dim, L]
    int64_t q2d[] = {B * H, T, head_dim};
    int64_t kt2d[] = {B * H, head_dim, L};
    int64_t s2d[] = {B * H, T, L};

    boat_tensor_t* Q_2d = boat_tensor_reshape(Q_mh, q2d, 3);
    boat_tensor_unref(Q_mh);
    Q_mh = NULL;
    boat_tensor_t* KT_2d = boat_tensor_reshape(K_T, kt2d, 3);
    boat_tensor_unref(K_T);
    K_T = NULL;

    boat_tensor_t* scores = boat_tensor_create(s2d, 3, BOAT_DTYPE_FLOAT32, boat_tensor_device(x));
    if (!scores) {
        boat_tensor_unref(Q_2d);
        boat_tensor_unref(KT_2d);
        boat_tensor_unref(V_mh);
        return NULL;
    }

    float* qd = data_f32(Q_2d);
    float* ktd = data_f32(KT_2d);
    float* sd = data_f32(scores);

    if (qd && ktd && sd) {
#ifdef BOAT_WITH_CUDA
        if (boat_tensor_device(x) == BOAT_DEVICE_CUDA) {
            boat_cuda_batched_matmul_scale_f32(qd, ktd, sd, B * H, T, L, head_dim, scale);
        } else
#endif
        {
            for (int i = 0; i < B * H; i++) {
                for (int ti = 0; ti < T; ti++) {
                    for (int li = 0; li < L; li++) {
                        float sum = 0.0f;
                        for (int hd = 0; hd < head_dim; hd++) {
                            sum += qd[i * T * head_dim + ti * head_dim + hd] *
                                   ktd[i * head_dim * L + hd * L + li];
                        }
                        sd[i * T * L + ti * L + li] = sum * scale;
                    }
                }
            }
        }
    }

    boat_tensor_unref(Q_2d);
    boat_tensor_unref(KT_2d);

    // Causal mask
    if (causal && T > 1) {
#ifdef BOAT_WITH_CUDA
        if (boat_tensor_device(x) == BOAT_DEVICE_CUDA) {
            boat_cuda_add_causal_mask_f32(sd, B * H, T, L, step >= 0 ? step : 0);
        } else
#endif
        {
            for (int i = 0; i < B * H; i++) {
                for (int ti = 0; ti < T; ti++) {
                    int global_ti = step >= 0 ? step + ti : ti;
                    for (int li = 0; li < L; li++) {
                        if (li > global_ti) {
                            sd[i * T * L + ti * L + li] = -INFINITY;
                        }
                    }
                }
            }
        }
    }

    // Softmax over L
    // [B*H, T, L] -> reshape to [B*H*T, L] for boat_softmax
    int64_t sm_shape[] = {B * H * T, L};
    boat_tensor_t* sm_in = boat_tensor_reshape(scores, sm_shape, 2);
    boat_tensor_unref(scores);
    if (!sm_in) {
        boat_tensor_unref(V_mh);
        return NULL;
    }

    boat_tensor_t* sm_out = boat_softmax(sm_in, 1);
    boat_tensor_unref(sm_in);
    if (!sm_out) {
        boat_tensor_unref(V_mh);
        return NULL;
    }

    // Reshape back
    int64_t sm_orig[] = {B * H, T, L};
    sm_out = boat_tensor_reshape(sm_out, sm_orig, 3);
    if (!sm_out) {
        boat_tensor_unref(V_mh);
        return NULL;
    }

    // attn @ V: [B*H, T, L] @ V: [B*H, L, head_dim] -> [B*H, T, head_dim]
    int64_t v2d[] = {B * H, L, head_dim};
    int64_t out2d_shape[] = {B * H, T, head_dim};
    boat_tensor_t* V_2d = boat_tensor_reshape(V_mh, v2d, 3);
    boat_tensor_unref(V_mh);
    V_mh = NULL;

    boat_tensor_t* out_2d =
        boat_tensor_create(out2d_shape, 3, BOAT_DTYPE_FLOAT32, boat_tensor_device(x));
    if (!out_2d) {
        boat_tensor_unref(sm_out);
        boat_tensor_unref(V_2d);
        return NULL;
    }

    float* attn_d = data_f32(sm_out);
    float* vd = data_f32(V_2d);
    float* out_d = data_f32(out_2d);

    if (attn_d && vd && out_d) {
#ifdef BOAT_WITH_CUDA
        if (boat_tensor_device(x) == BOAT_DEVICE_CUDA) {
            boat_cuda_batched_matmul_scale_f32(attn_d, vd, out_d, B * H, T, head_dim, L, 1.0f);
        } else
#endif
        {
            for (int i = 0; i < B * H; i++) {
                for (int ti = 0; ti < T; ti++) {
                    for (int hd = 0; hd < head_dim; hd++) {
                        float sum = 0.0f;
                        for (int li = 0; li < L; li++) {
                            sum += attn_d[i * T * L + ti * L + li] *
                                   vd[i * L * head_dim + li * head_dim + hd];
                        }
                        out_d[i * T * head_dim + ti * head_dim + hd] = sum;
                    }
                }
            }
        }
    }

    boat_tensor_unref(sm_out);
    boat_tensor_unref(V_2d);

    // Reshape back to [B, H, T, head_dim] -> [B, T, H, head_dim] -> [B, T, D]
    int64_t out_mh[] = {B, H, T, head_dim};
    boat_tensor_t* out_mh_t = boat_tensor_reshape(out_2d, out_mh, 4);
    boat_tensor_unref(out_2d);
    if (!out_mh_t) return NULL;

    out_mh_t = boat_transpose(out_mh_t, 1, 2);
    if (!out_mh_t) return NULL;

    int64_t out_final[] = {B, T, D};
    boat_tensor_t* out = boat_tensor_reshape(out_mh_t, out_final, 3);
    boat_tensor_unref(out_mh_t);
    if (!out) return NULL;

    // Output projection
    out = linear(out, w->self_o_weight, w->self_o_bias);
    return out;
}

// Cross-attention: Q from decoder, K,V from encoder output.
// x: [B, T, D], enc_out: [B, S, D]
static boat_tensor_t* cross_attention(const boat_tensor_t* x, const boat_tensor_t* enc_out,
                                      const boat_decoder_config_t* config,
                                      const boat_decoder_layer_weights_t* w,
                                      boat_decoder_cache_t* cache) {
    int B = (int)boat_tensor_shape(x)[0];
    int T = (int)boat_tensor_shape(x)[1];
    int D = config->d_model;
    int H = config->num_heads;
    int head_dim = D / H;

    // Q from decoder
    boat_tensor_t* Q = linear(x, w->cross_q_weight, w->cross_q_bias);
    if (!Q) return NULL;

    // K,V from encoder output (compute once, cache)
    // For cross-attention, K,V are static (from encoder)
    // Cache them if not already cached
    if (cache && !cache->cross_k && enc_out) {
        int S = (int)boat_tensor_shape(enc_out)[1];
        (void)S;
        cache->cross_k = linear(enc_out, w->cross_k_weight, w->cross_k_bias);
        cache->cross_v = linear(enc_out, w->cross_v_weight, w->cross_v_bias);
        if (!cache->cross_k || !cache->cross_v) {
            boat_tensor_unref(Q);
            return NULL;
        }
    }

    boat_tensor_t* K = cache && cache->cross_k ? cache->cross_k : NULL;
    boat_tensor_t* V = cache && cache->cross_v ? cache->cross_v : NULL;
    if (!K || !V) {
        // No cache: compute directly
        K = linear(enc_out, w->cross_k_weight, w->cross_k_bias);
        V = linear(enc_out, w->cross_v_weight, w->cross_v_bias);
        if (!K || !V) {
            boat_tensor_unref(Q);
            if (K) boat_tensor_unref(K);
            return NULL;
        }
    }

    int S = (int)boat_tensor_shape(K)[1];

    // Reshape for multi-head attention
    int64_t q_mh_shape[] = {B, T, H, head_dim};
    int64_t k_mh_shape[] = {B, S, H, head_dim};

    boat_tensor_t* Q_mh = boat_tensor_reshape(Q, q_mh_shape, 4);
    boat_tensor_unref(Q);
    if (!Q_mh) return NULL;
    Q_mh = boat_transpose(Q_mh, 1, 2);

    boat_tensor_t* K_mh = boat_tensor_reshape(K, k_mh_shape, 4);
    if (!cache) {
        boat_tensor_unref(K);
    }
    if (!K_mh) {
        boat_tensor_unref(Q_mh);
        return NULL;
    }
    K_mh = boat_transpose(K_mh, 1, 2);

    boat_tensor_t* V_mh = boat_tensor_reshape(V, k_mh_shape, 4);
    if (!cache) {
        boat_tensor_unref(V);
    }
    if (!V_mh) {
        boat_tensor_unref(Q_mh);
        boat_tensor_unref(K_mh);
        return NULL;
    }
    V_mh = boat_transpose(V_mh, 1, 2);

    float scale = 1.0f / sqrtf((float)head_dim);

    // Flatten batch: [B*H, T, head_dim] @ [B*H, head_dim, S] -> [B*H, T, S]
    int64_t q2d[] = {B * H, T, head_dim};
    int64_t k2d[] = {B * H, head_dim, S};

    // Note: if cache was used, we DON'T own K,V (they're in the cache struct)
    // So we only do ref/unref for the cache-owned case
    boat_tensor_t* Q_2d = boat_tensor_reshape(Q_mh, q2d, 3);
    boat_tensor_unref(Q_mh);

    boat_tensor_t* K_T = boat_transpose(K_mh, 2, 3);
    boat_tensor_unref(K_mh);
    boat_tensor_t* KT_2d = boat_tensor_reshape(K_T, k2d, 3);
    boat_tensor_unref(K_T);

    int64_t s2d[] = {B * H, T, S};
    boat_tensor_t* scores = boat_tensor_create(s2d, 3, BOAT_DTYPE_FLOAT32, boat_tensor_device(x));
    if (!scores) {
        boat_tensor_unref(Q_2d);
        boat_tensor_unref(KT_2d);
        boat_tensor_unref(V_mh);
        return NULL;
    }

    float* qd = data_f32(Q_2d);
    float* ktd = data_f32(KT_2d);
    float* sd = data_f32(scores);

    if (qd && ktd && sd) {
#ifdef BOAT_WITH_CUDA
        if (boat_tensor_device(x) == BOAT_DEVICE_CUDA) {
            boat_cuda_batched_matmul_scale_f32(qd, ktd, sd, B * H, T, S, head_dim, scale);
        } else
#endif
        {
            for (int i = 0; i < B * H; i++) {
                for (int ti = 0; ti < T; ti++) {
                    for (int si = 0; si < S; si++) {
                        float sum = 0.0f;
                        for (int hd = 0; hd < head_dim; hd++) {
                            sum += qd[i * T * head_dim + ti * head_dim + hd] *
                                   ktd[i * head_dim * S + hd * S + si];
                        }
                        sd[i * T * S + ti * S + si] = sum * scale;
                    }
                }
            }
        }
    }

    boat_tensor_unref(Q_2d);
    boat_tensor_unref(KT_2d);

    // Softmax over S
    int64_t sm_shape[] = {B * H * T, S};
    boat_tensor_t* sm_in = boat_tensor_reshape(scores, sm_shape, 2);
    boat_tensor_unref(scores);
    if (!sm_in) {
        boat_tensor_unref(V_mh);
        return NULL;
    }

    boat_tensor_t* sm_out = boat_softmax(sm_in, 1);
    boat_tensor_unref(sm_in);
    if (!sm_out) {
        boat_tensor_unref(V_mh);
        return NULL;
    }

    int64_t sm_orig[] = {B * H, T, S};
    sm_out = boat_tensor_reshape(sm_out, sm_orig, 3);
    if (!sm_out) {
        boat_tensor_unref(V_mh);
        return NULL;
    }

    // [B*H, T, S] @ [B*H, S, head_dim] -> [B*H, T, head_dim]
    int64_t v2d[] = {B * H, S, head_dim};
    boat_tensor_t* V_2d = boat_tensor_reshape(V_mh, v2d, 3);
    boat_tensor_unref(V_mh);

    int64_t out2d_shape[] = {B * H, T, head_dim};
    boat_tensor_t* out_2d =
        boat_tensor_create(out2d_shape, 3, BOAT_DTYPE_FLOAT32, boat_tensor_device(x));
    if (!out_2d) {
        boat_tensor_unref(sm_out);
        boat_tensor_unref(V_2d);
        return NULL;
    }

    float* attn_d = data_f32(sm_out);
    float* vd = data_f32(V_2d);
    float* out_d = data_f32(out_2d);

    if (attn_d && vd && out_d) {
#ifdef BOAT_WITH_CUDA
        if (boat_tensor_device(x) == BOAT_DEVICE_CUDA) {
            boat_cuda_batched_matmul_scale_f32(attn_d, vd, out_d, B * H, T, head_dim, S, 1.0f);
        } else
#endif
        {
            for (int i = 0; i < B * H; i++) {
                for (int ti = 0; ti < T; ti++) {
                    for (int hd = 0; hd < head_dim; hd++) {
                        float sum = 0.0f;
                        for (int si = 0; si < S; si++) {
                            sum += attn_d[i * T * S + ti * S + si] *
                                   vd[i * S * head_dim + si * head_dim + hd];
                        }
                        out_d[i * T * head_dim + ti * head_dim + hd] = sum;
                    }
                }
            }
        }
    }

    boat_tensor_unref(sm_out);
    boat_tensor_unref(V_2d);

    // Reshape back to [B, T, D]
    int64_t out_mh[] = {B, H, T, head_dim};
    boat_tensor_t* out_mh_t = boat_tensor_reshape(out_2d, out_mh, 4);
    boat_tensor_unref(out_2d);
    out_mh_t = boat_transpose(out_mh_t, 1, 2);

    int64_t out_final[] = {B, T, D};
    boat_tensor_t* out = boat_tensor_reshape(out_mh_t, out_final, 3);
    boat_tensor_unref(out_mh_t);
    if (!out) return NULL;

    out = linear(out, w->cross_o_weight, w->cross_o_bias);
    return out;
}

// =========================================================================
// Public API
// =========================================================================

BOAT_API boat_tensor_t* boat_decoder_layer_forward(const boat_decoder_config_t* config,
                                                   const boat_decoder_layer_weights_t* weights,
                                                   const boat_tensor_t* x,
                                                   const boat_tensor_t* encoder_output,
                                                   boat_decoder_cache_t* cache, int32_t step) {
    if (!config || !weights || !x) return NULL;

    float eps = config->layer_norm_eps;

    // Pre-norm architecture (mBART-style)
    if (config->pre_norm) {
        // 1. Self-attention + residual
        boat_tensor_t* residual = (boat_tensor_t*)x;
        boat_tensor_ref(residual);

        boat_tensor_t* normed =
            layer_norm_affine(x, weights->self_ln_weight, weights->self_ln_bias, eps);
        if (!normed) {
            boat_tensor_unref(residual);
            return NULL;
        }

        boat_tensor_t* attn_out = self_attention(normed, config, weights, cache, step, true);
        boat_tensor_unref(normed);
        if (!attn_out) {
            boat_tensor_unref(residual);
            return NULL;
        }

        boat_tensor_t* h = boat_add(residual, attn_out);
        boat_tensor_unref(residual);
        boat_tensor_unref(attn_out);
        if (!h) return NULL;

        // 2. Cross-attention + residual
        if (encoder_output) {
            residual = h;
            boat_tensor_ref(residual);

            normed = layer_norm_affine(h, weights->cross_ln_weight, weights->cross_ln_bias, eps);
            boat_tensor_unref(h);
            if (!normed) {
                boat_tensor_unref(residual);
                return NULL;
            }

            attn_out = cross_attention(normed, encoder_output, config, weights, cache);
            boat_tensor_unref(normed);
            if (!attn_out) {
                boat_tensor_unref(residual);
                return NULL;
            }

            h = boat_add(residual, attn_out);
            boat_tensor_unref(residual);
            boat_tensor_unref(attn_out);
            if (!h) return NULL;
        }

        // 3. FFN + residual
        residual = h;
        boat_tensor_ref(residual);

        normed = layer_norm_affine(h, weights->ffn_ln_weight, weights->ffn_ln_bias, eps);
        boat_tensor_unref(h);
        if (!normed) {
            boat_tensor_unref(residual);
            return NULL;
        }

        // FFN: FC1 -> activation -> FC2
        boat_tensor_t* ffn = linear(normed, weights->fc1_weight, weights->fc1_bias);
        boat_tensor_unref(normed);
        if (!ffn) {
            boat_tensor_unref(residual);
            return NULL;
        }

        if (config->activation && strcmp(config->activation, "gelu") == 0) {
            boat_tensor_t* act = boat_gelu(ffn);
            boat_tensor_unref(ffn);
            if (!act) {
                boat_tensor_unref(residual);
                return NULL;
            }
            ffn = act;
        } else {
            // Default: ReLU
            boat_tensor_t* act = boat_relu(ffn);
            boat_tensor_unref(ffn);
            if (!act) {
                boat_tensor_unref(residual);
                return NULL;
            }
            ffn = act;
        }

        boat_tensor_t* ffn_out = linear(ffn, weights->fc2_weight, weights->fc2_bias);
        boat_tensor_unref(ffn);
        if (!ffn_out) {
            boat_tensor_unref(residual);
            return NULL;
        }

        boat_tensor_t* out = boat_add(residual, ffn_out);
        boat_tensor_unref(residual);
        boat_tensor_unref(ffn_out);
        return out;
    }

    // Post-norm (not used by mBART, but supported)
    // TODO: implement post-norm path
    return NULL;
}

// =========================================================================
// KV Cache management
// =========================================================================

BOAT_API boat_decoder_cache_t* boat_decoder_cache_create(int32_t batch_size, int32_t num_heads,
                                                         int32_t head_dim, int32_t max_t,
                                                         int32_t encoder_seq_len) {
    return boat_decoder_cache_create_ex(batch_size, num_heads, head_dim, max_t, encoder_seq_len,
                                        BOAT_DEVICE_CPU);
}

BOAT_API boat_decoder_cache_t* boat_decoder_cache_create_ex(int32_t batch_size, int32_t num_heads,
                                                            int32_t head_dim, int32_t max_t,
                                                            int32_t encoder_seq_len,
                                                            boat_device_t device) {
    (void)encoder_seq_len;
    boat_decoder_cache_t* cache =
        (boat_decoder_cache_t*)boat_malloc(sizeof(boat_decoder_cache_t), BOAT_DEVICE_CPU);
    if (!cache) return NULL;
    memset(cache, 0, sizeof(*cache));

    cache->max_length = max_t;
    cache->length = 0;

    int64_t kv_shape[] = {batch_size, num_heads, max_t, head_dim};
    cache->self_k = boat_tensor_create(kv_shape, 4, BOAT_DTYPE_FLOAT32, device);
    cache->self_v = boat_tensor_create(kv_shape, 4, BOAT_DTYPE_FLOAT32, device);
    // Cross K,V are set lazily when encoder output is available
    cache->cross_k = NULL;
    cache->cross_v = NULL;

    if (!cache->self_k || !cache->self_v) {
        boat_decoder_cache_free(cache);
        return NULL;
    }

    return cache;
}

BOAT_API void boat_decoder_cache_free(boat_decoder_cache_t* cache) {
    if (!cache) return;
    if (cache->self_k) boat_tensor_unref(cache->self_k);
    if (cache->self_v) boat_tensor_unref(cache->self_v);
    if (cache->cross_k) boat_tensor_unref(cache->cross_k);
    if (cache->cross_v) boat_tensor_unref(cache->cross_v);
    boat_free(cache);
}

BOAT_API void boat_decoder_cache_reset(boat_decoder_cache_t* cache) {
    if (!cache) return;
    cache->length = 0;
}
