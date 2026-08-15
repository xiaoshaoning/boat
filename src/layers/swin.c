// swin.c - Swin Transformer encoder implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers/swin.h>
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

static float* tensor_data_f32(const boat_tensor_t* t) {
    return (float*)boat_tensor_data(t);
}

// Apply LayerNorm + affine to last dim.
// input: [..., D], returns [..., D]
static boat_tensor_t* apply_layernorm(const boat_tensor_t* x, const boat_tensor_t* gamma,
                                      const boat_tensor_t* beta, float eps) {
    int64_t ndim = (int64_t)boat_tensor_ndim(x);
    const int64_t* shape = boat_tensor_shape(x);
    int64_t D = shape[ndim - 1];
    int64_t norm_shape[] = {D};

    boat_tensor_t* y = boat_layer_norm(x, norm_shape, 1, eps);
    if (!y) return NULL;

    // Flatten to 2D [outer, D] for simpler affine transform
    int64_t outer = (int64_t)(boat_tensor_nelements(y) / (size_t)D);
    boat_device_t dev = boat_tensor_device(y);
    (void)dev;

    if (gamma || beta) {
        int64_t flat_shape[] = {outer, D};
        boat_tensor_t* y_flat = boat_tensor_reshape(y, flat_shape, 2);
        if (!y_flat) {
            boat_tensor_unref(y);
            return NULL;
        }

        boat_tensor_t* out = boat_tensor_create(flat_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!out) {
            boat_tensor_unref(y_flat);
            boat_tensor_unref(y);
            return NULL;
        }

        // Move to CPU for affine, handle broadcast gamma[D] and beta[D]
        float* yp = (float*)boat_tensor_data(y_flat);
        float* op = (float*)boat_tensor_data(out);
        float* gp = gamma ? (float*)boat_tensor_data(gamma) : NULL;
        float* bp = beta ? (float*)boat_tensor_data(beta) : NULL;

#ifdef BOAT_WITH_CUDA
        float *y_cpu = NULL, *g_cpu = NULL, *b_cpu = NULL;
        if (dev == BOAT_DEVICE_CUDA) {
            y_cpu = (float*)malloc((size_t)outer * (size_t)D * sizeof(float));
            boat_cuda_memcpy_d2h(y_cpu, yp, (size_t)outer * (size_t)D * sizeof(float));
            yp = y_cpu;
            if (gp) {
                g_cpu = (float*)malloc((size_t)D * sizeof(float));
                boat_cuda_memcpy_d2h(g_cpu, gp, (size_t)D * sizeof(float));
                gp = g_cpu;
            }
            if (bp) {
                b_cpu = (float*)malloc((size_t)D * sizeof(float));
                boat_cuda_memcpy_d2h(b_cpu, bp, (size_t)D * sizeof(float));
                bp = b_cpu;
            }
        }
#endif

        for (int64_t i = 0; i < outer; i++) {
            for (int64_t j = 0; j < D; j++) {
                float val = yp[i * D + j];
                if (gp) val *= gp[j];
                if (bp) val += bp[j];
                op[i * D + j] = val;
            }
        }

#ifdef BOAT_WITH_CUDA
        if (dev == BOAT_DEVICE_CUDA) {
            boat_tensor_t* d_out =
                boat_tensor_create(flat_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CUDA);
            if (d_out) {
                boat_cuda_memcpy_h2d(boat_tensor_data(d_out), op,
                                     (size_t)outer * (size_t)D * sizeof(float));
            }
            boat_tensor_unref(out);
            out = d_out;
            free(y_cpu);
            free(g_cpu);
            free(b_cpu);
        }
#endif

        boat_tensor_unref(y_flat);
        boat_tensor_unref(y);

        if (!out) return NULL;
        // Restore original shape
        int64_t* orig_shape = (int64_t*)malloc((size_t)ndim * sizeof(int64_t));
        memcpy(orig_shape, shape, (size_t)ndim * sizeof(int64_t));
        boat_tensor_t* y_out = boat_tensor_reshape(out, orig_shape, (size_t)ndim);
        free(orig_shape);
        boat_tensor_unref(out);
        return y_out;
    }

    return y;
}

// Window partition: [B, H, W, C] -> [B*nh*nw, ws*ws, C]
// where nh=H/ws, nw=W/ws. H,W must be divisible by ws.
static boat_tensor_t* window_partition(const boat_tensor_t* x, int ws) {
    const int64_t* s = boat_tensor_shape(x);
    int B = (int)s[0], H = (int)s[1], W = (int)s[2], C = (int)s[3];
    int nh = H / ws, nw = W / ws;
    int num_windows = nh * nw;
    int64_t out_shape[] = {B * num_windows, ws * ws, C};
    boat_tensor_t* out =
        boat_tensor_create(out_shape, 3, BOAT_DTYPE_FLOAT32, boat_tensor_device(x));
    if (!out) return NULL;

    float* src = tensor_data_f32(x);
    float* dst = tensor_data_f32(out);
    if (!src || !dst) {
        boat_tensor_unref(out);
        return NULL;
    }

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(x) == BOAT_DEVICE_CUDA) {
        boat_cuda_swin_window_partition_f32(src, dst, B, H, W, C, ws);
    } else
#endif
    {
        for (int b = 0; b < B; b++) {
            for (int hi = 0; hi < nh; hi++) {
                for (int wi = 0; wi < nw; wi++) {
                    int win_idx = (b * nh + hi) * nw + wi;
                    for (int i = 0; i < ws; i++) {
                        for (int j = 0; j < ws; j++) {
                            int h_idx = hi * ws + i;
                            int w_idx = wi * ws + j;
                            for (int c = 0; c < C; c++) {
                                int src_idx = ((b * H + h_idx) * W + w_idx) * C + c;
                                int dst_idx = ((win_idx * ws + i) * ws + j) * C + c;
                                dst[dst_idx] = src[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

// Window reverse: [B*nh*nw, ws*ws, C] -> [B, H, W, C]
static boat_tensor_t* window_reverse(const boat_tensor_t* windows, int H, int W, int C, int ws) {
    const int64_t* s = boat_tensor_shape(windows);
    int total_win = (int)s[0];
    int nh = H / ws, nw = W / ws;
    int B = total_win / (nh * nw);
    int64_t out_shape[] = {B, H, W, C};
    boat_tensor_t* out =
        boat_tensor_create(out_shape, 4, BOAT_DTYPE_FLOAT32, boat_tensor_device(windows));
    if (!out) return NULL;

    float* src = tensor_data_f32(windows);
    float* dst = tensor_data_f32(out);
    if (!src || !dst) {
        boat_tensor_unref(out);
        return NULL;
    }

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(windows) == BOAT_DEVICE_CUDA) {
        boat_cuda_swin_window_reverse_f32(src, dst, B, H, W, C, ws);
    } else
#endif
    {
        for (int b = 0; b < B; b++) {
            for (int hi = 0; hi < nh; hi++) {
                for (int wi = 0; wi < nw; wi++) {
                    int win_idx = (b * nh + hi) * nw + wi;
                    for (int i = 0; i < ws; i++) {
                        for (int j = 0; j < ws; j++) {
                            for (int c = 0; c < C; c++) {
                                int src_idx = ((win_idx * ws + i) * ws + j) * C + c;
                                int dst_idx = ((b * H + hi * ws + i) * W + wi * ws + j) * C + c;
                                dst[dst_idx] = src[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

// Cyclic shift: roll by (-shift, -shift) on H,W dims
// Input: [B, H, W, C]  Output: [B, H, W, C]
// shift > 0 for forward (SW-MSA), shift = 0 = identity
static boat_tensor_t* cyclic_shift(const boat_tensor_t* x, int shift, bool reverse) {
    if (shift == 0) {
        boat_tensor_ref((boat_tensor_t*)x);
        return (boat_tensor_t*)x;
    }
    int sh = reverse ? shift : -shift;

    const int64_t* s = boat_tensor_shape(x);
    int B = (int)s[0], H = (int)s[1], W = (int)s[2], C = (int)s[3];
    int64_t out_shape[] = {B, H, W, C};
    boat_tensor_t* out =
        boat_tensor_create(out_shape, 4, BOAT_DTYPE_FLOAT32, boat_tensor_device(x));
    if (!out) return NULL;

    float* src = tensor_data_f32(x);
    float* dst = tensor_data_f32(out);
    if (!src || !dst) {
        boat_tensor_unref(out);
        return NULL;
    }

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(x) == BOAT_DEVICE_CUDA) {
        boat_cuda_swin_cyclic_shift_f32(src, dst, B, H, W, C, shift, reverse ? 1 : 0);
    } else
#endif
    {
        int sh_h = (sh % H + H) % H;
        int sh_w = (sh % W + W) % W;
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                int src_h = (h + sh_h) % H;
                for (int w = 0; w < W; w++) {
                    int src_w = (w + sh_w) % W;
                    for (int c = 0; c < C; c++) {
                        int src_idx = ((b * H + src_h) * W + src_w) * C + c;
                        int dst_idx = ((b * H + h) * W + w) * C + c;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
    return out;
}

// Window attention with relative position bias.
// x: [num_windows, N, dim] where N = ws*ws
// Returns: [num_windows*N, dim] (flat for downstream projection)
static boat_tensor_t* window_attention(const boat_tensor_t* x, const boat_swin_block_weights_t* w,
                                       int dim, int num_heads, int ws, bool use_cuda) {
    (void)use_cuda;
    int N = ws * ws;
    int head_dim = dim / num_heads;
    const int64_t* s = boat_tensor_shape(x);
    int num_windows = (int)s[0];
    int B_flat = num_windows * N;
    boat_device_t dev = boat_tensor_device(x);
    int num_heads_total = num_windows * num_heads;

    // === QKV projections (2D matmul) ===
    int64_t flat_shape[] = {B_flat, dim};
    boat_tensor_t* x_flat = boat_tensor_reshape(x, flat_shape, 2);
    if (!x_flat) return NULL;

    boat_tensor_t* Q = boat_matmul(x_flat, w->query_weight);
    if (!Q) {
        boat_tensor_unref(x_flat);
        return NULL;
    }
    if (w->query_bias) {
        boat_tensor_t* t = boat_add(Q, w->query_bias);
        boat_tensor_unref(Q);
        Q = t;
    }

    boat_tensor_t* K = boat_matmul(x_flat, w->key_weight);
    if (!K) {
        boat_tensor_unref(x_flat);
        boat_tensor_unref(Q);
        return NULL;
    }
    if (w->key_bias) {
        boat_tensor_t* t = boat_add(K, w->key_bias);
        boat_tensor_unref(K);
        K = t;
    }

    boat_tensor_t* V = boat_matmul(x_flat, w->value_weight);
    if (!V) {
        boat_tensor_unref(x_flat);
        boat_tensor_unref(Q);
        boat_tensor_unref(K);
        return NULL;
    }
    if (w->value_bias) {
        boat_tensor_t* t = boat_add(V, w->value_bias);
        boat_tensor_unref(V);
        V = t;
    }

    boat_tensor_unref(x_flat);

    // === Flat data pointers (Q,K,V are 2D [B_flat, dim]) ===
    float* Qd = tensor_data_f32(Q);
    float* Kd = tensor_data_f32(K);
    float* Vd = tensor_data_f32(V);

    // === Allocate scores: [num_heads_total, N, N] ===
    int64_t scores_shape[] = {num_heads_total, N, N};
    boat_tensor_t* scores_base = boat_tensor_create(scores_shape, 3, BOAT_DTYPE_FLOAT32, dev);
    if (!scores_base) {
        boat_tensor_unref(Q);
        boat_tensor_unref(K);
        boat_tensor_unref(V);
        return NULL;
    }
    float* Sd = tensor_data_f32(scores_base);

    // === scores[w*h + h, i, j] = sum_k Q[w*N+i][h*hd+k] * K[w*N+j][h*hd+k] * scale ===
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int wi = 0; wi < num_windows; wi++) {
        for (int hi = 0; hi < num_heads; hi++) {
            int batch_idx = wi * num_heads + hi;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < head_dim; k++) {
                        float qv = Qd[(wi * N + i) * dim + hi * head_dim + k];
                        float kv = Kd[(wi * N + j) * dim + hi * head_dim + k];
                        sum += qv * kv;
                    }
                    Sd[batch_idx * N * N + i * N + j] = sum * scale;
                }
            }
        }
    }
    boat_tensor_unref(Q);
    boat_tensor_unref(K);

    // === Add relative position bias ===
    if (w->rel_pos_bias_table && w->rel_pos_index) {
        float* bias_tbl = tensor_data_f32(w->rel_pos_bias_table);
        int64_t* rpi = (int64_t*)boat_tensor_data(w->rel_pos_index);
        if (bias_tbl && rpi) {
            for (int wi = 0; wi < num_windows; wi++) {
                for (int hi = 0; hi < num_heads; hi++) {
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            int sidx = ((wi * num_heads + hi) * N + i) * N + j;
                            int bias_idx = (int)rpi[i * N + j];
                            Sd[sidx] += bias_tbl[bias_idx * num_heads + hi];
                        }
                    }
                }
            }
        }
    }

    // === Softmax over last dim N (in-place) ===
    for (int wh = 0; wh < num_heads_total; wh++) {
        for (int i = 0; i < N; i++) {
            int row_base = (wh * N + i) * N;
            float max_val = Sd[row_base];
            for (int j = 1; j < N; j++) {
                if (Sd[row_base + j] > max_val) max_val = Sd[row_base + j];
            }
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                float ev = expf(Sd[row_base + j] - max_val);
                Sd[row_base + j] = ev;
                sum += ev;
            }
            float inv_sum = 1.0f / sum;
            for (int j = 0; j < N; j++) {
                Sd[row_base + j] *= inv_sum;
            }
        }
    }

    // === attn @ V: output[wi*N + i][h*hd + k] = sum_j attn[wh][i][j] * V[wi*N + j][h*hd + k] ===
    int64_t out_flat_shape[] = {B_flat, dim};
    boat_tensor_t* attn_out = boat_tensor_create(out_flat_shape, 2, BOAT_DTYPE_FLOAT32, dev);
    if (!attn_out) {
        boat_tensor_unref(scores_base);
        boat_tensor_unref(V);
        return NULL;
    }
    float* Od = tensor_data_f32(attn_out);

    for (int wi = 0; wi < num_windows; wi++) {
        for (int i = 0; i < N; i++) {
            for (int hi = 0; hi < num_heads; hi++) {
                for (int k = 0; k < head_dim; k++) {
                    int wh = wi * num_heads + hi;
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++) {
                        sum +=
                            Sd[(wh * N + i) * N + j] * Vd[(wi * N + j) * dim + hi * head_dim + k];
                    }
                    Od[(wi * N + i) * dim + hi * head_dim + k] = sum;
                }
            }
        }
    }
    boat_tensor_unref(scores_base);
    boat_tensor_unref(V);

    // === Output projection ===
    boat_tensor_t* projected = boat_matmul(attn_out, w->proj_weight);
    boat_tensor_unref(attn_out);
    if (!projected) return NULL;
    if (w->proj_bias) {
        boat_tensor_t* t = boat_add(projected, w->proj_bias);
        boat_tensor_unref(projected);
        projected = t;
    }
    return projected;
}

// MLP: FC1 -> GELU -> FC2
static boat_tensor_t* swin_mlp(const boat_tensor_t* x, const boat_swin_block_weights_t* w) {
    // Flatten to 2D [batch, dim] for matmul, keep original shape to restore
    size_t ndim = boat_tensor_ndim(x);
    const int64_t* shape = boat_tensor_shape(x);
    int64_t D = shape[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++)
        outer *= shape[i];
    int64_t flat_shape[] = {outer, D};
    boat_tensor_t* x_flat = boat_tensor_reshape(x, flat_shape, 2);
    if (!x_flat) return NULL;

    boat_tensor_t* h = boat_matmul(x_flat, w->mlp_fc1_weight);
    boat_tensor_unref(x_flat);
    if (!h) return NULL;
    if (w->mlp_fc1_bias) {
        boat_tensor_t* t = boat_add(h, w->mlp_fc1_bias);
        boat_tensor_unref(h);
        h = t;
        if (!h) return NULL;
    }
    boat_tensor_t* a = boat_gelu(h);
    boat_tensor_unref(h);
    if (!a) return NULL;

    boat_tensor_t* out = boat_matmul(a, w->mlp_fc2_weight);
    boat_tensor_unref(a);
    if (!out) return NULL;
    if (w->mlp_fc2_bias) {
        boat_tensor_t* t = boat_add(out, w->mlp_fc2_bias);
        boat_tensor_unref(out);
        out = t;
    }

    // Restore original shape (except last dim)
    int64_t* orig_shape = (int64_t*)malloc((size_t)ndim * sizeof(int64_t));
    memcpy(orig_shape, shape, (size_t)ndim * sizeof(int64_t));
    boat_tensor_t* y = boat_tensor_reshape(out, orig_shape, (size_t)ndim);
    free(orig_shape);
    boat_tensor_unref(out);
    return y;
}

// PatchMerging
static boat_tensor_t* patch_merging(const boat_tensor_t* x, const boat_swin_downsample_weights_t* w,
                                    float eps) {
    const int64_t* s = boat_tensor_shape(x);
    int B = (int)s[0], H = (int)s[1], W = (int)s[2], C = (int)s[3];
    int H2 = H / 2, W2 = W / 2;

    // x is [B, H, W, C]. Extract 2x2 patches -> [B, H2, W2, 4*C]
    int64_t merged_shape[] = {B, H2, W2, 4 * C};
    boat_tensor_t* merged =
        boat_tensor_create(merged_shape, 4, BOAT_DTYPE_FLOAT32, boat_tensor_device(x));
    if (!merged) return NULL;

    float* src = tensor_data_f32(x);
    float* dst = tensor_data_f32(merged);
    if (!src || !dst) {
        boat_tensor_unref(merged);
        return NULL;
    }

    for (int b = 0; b < B; b++) {
        for (int hi = 0; hi < H2; hi++) {
            for (int wi = 0; wi < W2; wi++) {
                int out_off = ((b * H2 + hi) * W2 + wi) * 4 * C;
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        int in_off = ((b * H + (hi * 2 + i)) * W + (wi * 2 + j)) * C;
                        int base = ((i * 2 + j)) * C;
                        for (int c = 0; c < C; c++) {
                            dst[out_off + base + c] = src[in_off + c];
                        }
                    }
                }
            }
        }
    }

    // LayerNorm (output keeps 4D shape [B, H2, W2, 4*C])
    boat_tensor_t* normed = apply_layernorm(merged, w->norm_weight, w->norm_bias, eps);
    boat_tensor_unref(merged);
    if (!normed) return NULL;

    // Flatten to 2D [B*H2*W2, 4*C] for matmul with 2D weight
    int64_t normed_dim = boat_tensor_shape(normed)[boat_tensor_ndim(normed) - 1];
    int64_t normed_outer = (int64_t)(boat_tensor_nelements(normed) / (size_t)normed_dim);
    int64_t normed_flat_shape[] = {normed_outer, normed_dim};
    boat_tensor_t* normed_flat = boat_tensor_reshape(normed, normed_flat_shape, 2);
    boat_tensor_unref(normed);
    if (!normed_flat) return NULL;

    // Linear: 4*C -> 2*C
    boat_tensor_t* reduced = boat_matmul(normed_flat, w->reduction_weight);
    boat_tensor_unref(normed_flat);
    if (!reduced) return NULL;
    if (w->reduction_bias) {
        boat_tensor_t* t = boat_add(reduced, w->reduction_bias);
        boat_tensor_unref(reduced);
        reduced = t;
    }

    // Reshape back to 4D: [B, H2, W2, 2*C]
    int64_t final_shape[] = {B, H2, W2, 2 * C};
    boat_tensor_t* reshaped = boat_tensor_reshape(reduced, final_shape, 4);
    boat_tensor_unref(reduced);
    return reshaped;
}

// Create SW-MSA attention mask for shifted windows
static boat_tensor_t* create_attn_mask(int H, int W, int ws, int shift_size) {
    // TODO: implement proper attention mask for SW-MSA
    // Without mask, shifted window attention attends across cyclic boundaries,
    // which degrades quality slightly but is acceptable initially.
    (void)H;
    (void)W;
    (void)ws;
    (void)shift_size;
    return NULL;
}

// =========================================================================
// Swin block forward
// =========================================================================
static boat_tensor_t* swin_block_forward(const boat_tensor_t* x, const boat_swin_block_weights_t* w,
                                         int dim, int num_heads, int ws, int shift_size,
                                         boat_tensor_t* attn_mask, float eps) {
    // Save residual
    boat_tensor_t* residual = (boat_tensor_t*)x;
    boat_tensor_ref(residual);

    // LayerNorm before attention
    boat_tensor_t* normed = apply_layernorm(x, w->norm1_weight, w->norm1_bias, eps);
    if (!normed) {
        boat_tensor_unref(residual);
        return NULL;
    }

    // Cyclic shift for SW-MSA
    boat_tensor_t* shifted = NULL;
    if (shift_size > 0) {
        shifted = cyclic_shift(normed, shift_size, false);
        boat_tensor_unref(normed);
        if (!shifted) {
            boat_tensor_unref(residual);
            return NULL;
        }
    } else {
        shifted = normed;
    }

    // Window partition
    const int64_t* ns = boat_tensor_shape(shifted);
    int H = (int)ns[1], W = (int)ns[2];
    boat_tensor_t* windows = window_partition(shifted, ws);
    boat_tensor_unref(shifted);
    if (!windows) {
        boat_tensor_unref(residual);
        return NULL;
    }

    // Window attention (returns 2D [num_windows*N, dim])
    boat_tensor_t* attn_out =
        window_attention(windows, w, dim, num_heads, ws, boat_tensor_device(x) == BOAT_DEVICE_CUDA);
    boat_tensor_unref(windows);
    if (!attn_out) {
        boat_tensor_unref(residual);
        return NULL;
    }

    // Reshape to 3D [num_windows, N, dim] for window_reverse
    int N_win = ws * ws;
    int64_t wa_3d_shape[] = {boat_tensor_shape(attn_out)[0] / N_win, N_win, dim};
    boat_tensor_t* attn_3d = boat_tensor_reshape(attn_out, wa_3d_shape, 3);
    boat_tensor_unref(attn_out);
    if (!attn_3d) {
        boat_tensor_unref(residual);
        return NULL;
    }

    // Window reverse
    boat_tensor_t* restored = window_reverse(attn_3d, H, W, dim, ws);
    boat_tensor_unref(attn_3d);
    if (!restored) {
        boat_tensor_unref(residual);
        return NULL;
    }

    // Reverse cyclic shift (for SW-MSA)
    if (shift_size > 0) {
        boat_tensor_t* unshifted = cyclic_shift(restored, shift_size, true);
        boat_tensor_unref(restored);
        if (!unshifted) {
            boat_tensor_unref(residual);
            return NULL;
        }
        restored = unshifted;
    }

    // Residual
    boat_tensor_t* after_attn = boat_add(residual, restored);
    boat_tensor_unref(residual);
    boat_tensor_unref(restored);
    if (!after_attn) return NULL;

    // MLP
    residual = after_attn;
    boat_tensor_ref(residual);

    normed = apply_layernorm(after_attn, w->norm2_weight, w->norm2_bias, eps);
    boat_tensor_unref(after_attn);
    if (!normed) {
        boat_tensor_unref(residual);
        return NULL;
    }

    boat_tensor_t* mlp_out = swin_mlp(normed, w);
    boat_tensor_unref(normed);
    if (!mlp_out) {
        boat_tensor_unref(residual);
        return NULL;
    }

    boat_tensor_t* out = boat_add(residual, mlp_out);
    boat_tensor_unref(residual);
    boat_tensor_unref(mlp_out);
    if (!out) return NULL;
    return out;

    (void)attn_mask;
}

// =========================================================================
// Public API
// =========================================================================

BOAT_API boat_tensor_t* boat_swin_forward(const boat_swin_config_t* config,
                                          const boat_swin_weights_t* weights,
                                          const boat_tensor_t* input) {
    if (!config || !weights || !input) return NULL;
    if (boat_tensor_ndim(input) != 4) return NULL;
    if (boat_tensor_dtype(input) != BOAT_DTYPE_FLOAT32) return NULL;

    boat_device_t dev = boat_tensor_device(input);
    float eps = config->layer_norm_eps;
    int ws = config->window_size;
    int ps = config->patch_size;
    const int64_t* in_shape = boat_tensor_shape(input);
    int N = (int)in_shape[0], C = (int)in_shape[1];
    int H = (int)in_shape[2], W = (int)in_shape[3];

    // ===================================================================
    // Patch Embed: Conv2d + LayerNorm
    // ===================================================================
    const boat_swin_patch_embed_weights_t* pe = &weights->patch_embed;

    // For CPU: implement conv2d directly (no im2col needed for 4x4 small kernel)
    int Hp = H / ps, Wp = W / ps;
    int embed_dim = config->embed_dim;
    int64_t embed_shape[] = {N, Hp, Wp, embed_dim};
    boat_tensor_t* patch_embed = boat_tensor_create(embed_shape, 4, BOAT_DTYPE_FLOAT32, dev);
    if (!patch_embed) return NULL;

    float* input_data = tensor_data_f32(input);
    float* pe_data = tensor_data_f32(patch_embed);
    float* pe_w = tensor_data_f32(pe->proj_weight);
    float* pe_b = tensor_data_f32(pe->proj_bias);

    if (!input_data || !pe_data || !pe_w) {
        boat_tensor_unref(patch_embed);
        return NULL;
    }

#ifdef BOAT_WITH_CUDA
    if (dev == BOAT_DEVICE_CUDA) {
        boat_cuda_swin_patch_embed_f32(input_data, pe_w, pe_b, pe_data, N, C, H, W, embed_dim, ps);
        boat_cuda_synchronize();
    } else
#endif
    {
        int ps2 = ps * ps;
        (void)ps2;
        for (int n = 0; n < N; n++) {
            for (int hi = 0; hi < Hp; hi++) {
                for (int wi = 0; wi < Wp; wi++) {
                    for (int oc = 0; oc < embed_dim; oc++) {
                        float sum = pe_b ? pe_b[oc] : 0.0f;
                        for (int ic = 0; ic < C; ic++) {
                            for (int i = 0; i < ps; i++) {
                                for (int j = 0; j < ps; j++) {
                                    int in_idx = ((n * C + ic) * H + hi * ps + i) * W + wi * ps + j;
                                    int w_idx = ((oc * C + ic) * ps + i) * ps + j;
                                    sum += input_data[in_idx] * pe_w[w_idx];
                                }
                            }
                        }
                        int out_idx = ((n * Hp + hi) * Wp + wi) * embed_dim + oc;
                        pe_data[out_idx] = sum;
                    }
                }
            }
        }
    }

    // LayerNorm over last dim
    boat_tensor_t* x = apply_layernorm(patch_embed, pe->norm_weight, pe->norm_bias, eps);
    boat_tensor_unref(patch_embed);
    if (!x) return NULL;

    // ===================================================================
    // 4 Stages
    // ===================================================================
    int dim = embed_dim;
    int h = Hp, w = Wp;

    for (int stage = 0; stage < 4; stage++) {
        int num_blocks = config->depths[stage];
        int num_heads = config->num_heads[stage];
        int block_dim = dim;

        boat_swin_block_weights_t* blocks = weights->stages[stage].blocks;

        for (int bi = 0; bi < num_blocks; bi++) {
            int shift_size = (bi % 2 == 1) ? (ws / 2) : 0;

            boat_tensor_t* attn_mask = NULL;
            if (shift_size > 0) {
                attn_mask = create_attn_mask(h, w, ws, shift_size);
            }

            boat_tensor_t* next = swin_block_forward(x, &blocks[bi], block_dim, num_heads, ws,
                                                     shift_size, attn_mask, eps);
            boat_tensor_unref(x);

            if (attn_mask) boat_tensor_unref(attn_mask);
            if (!next) return NULL;
            x = next;
        }

        // Downsample (except last stage)
        if (stage < 3 && weights->stages[stage].downsample) {
            boat_swin_downsample_weights_t* ds = weights->stages[stage].downsample;
            boat_tensor_t* merged = patch_merging(x, ds, eps);
            boat_tensor_unref(x);
            if (!merged) return NULL;
            x = merged;
            h /= 2;
            w /= 2;
            dim *= 2;
        }
    }

    // ===================================================================
    // Final reshape: [N, h, w, dim] -> [N, h*w, dim]
    // ===================================================================
    int64_t final_shape[] = {N, h * w, dim};
    boat_tensor_t* result = boat_tensor_reshape(x, final_shape, 3);
    boat_tensor_unref(x);
    return result;
}
