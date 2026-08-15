// lstm.c - LSTM layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/simd.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <boat.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// LSTM layer structure
struct boat_lstm_layer_t {
    size_t input_size;
    size_t hidden_size;
    size_t num_layers;
    bool bidirectional;
    float dropout;

    // Parameters (weights and biases)
    boat_tensor_t* weight_ih; // Input-hidden weights [input_size, 4*hidden]
    boat_tensor_t* weight_hh; // Hidden-hidden weights [hidden_size, 4*hidden]
    boat_tensor_t* bias_ih;   // Input-hidden biases [4*hidden]
    boat_tensor_t* bias_hh;   // Hidden-hidden biases [4*hidden]

    // Gradient accumulators for training
    boat_tensor_t* grad_weight_ih;
    boat_tensor_t* grad_weight_hh;
    boat_tensor_t* grad_bias_ih;
    boat_tensor_t* grad_bias_hh;

    // Internal state
    boat_tensor_t* hidden_state;
    boat_tensor_t* cell_state;

    // Cached states for backward pass
    boat_tensor_t* cache_input; // Input tensor [batch, seq_len, input_size]
    size_t cache_seq_len;
    size_t cache_batch;
    float** cache_gates; // Raw gate pre-activations per timestep
    float** cache_h;     // Hidden state per timestep [batch, hidden]
    float** cache_c;     // Cell state per timestep [batch, hidden]
};

// LSTM gate layout: 4 * hidden_size, ordered as [input, forget, cell, output].
// Weights use the boat [in, out] convention (W_ih: [input_size, 4h], W_hh: [hidden_size, 4h]).
// Input is [batch, seq_len, input_size]; output is [batch, seq_len, hidden_size].

static float lstm_sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// c[m, n] = a[m, k] @ b[k, n]
static void lstm_matmul_f32(const float* a, const float* b, float* c, size_t m, size_t k,
                            size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// c[m, n] = a[m, k] @ b[n, k]^T  (b stored as [n, k])
static void lstm_matmul_bt(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
}

// c[m, n] = a[k, m]^T @ b[k, n]  (a stored as [k, m])
static void lstm_matmul_at(const float* a, const float* b, float* c, size_t k, size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += a[p * m + i] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// c[m, n] += bias[n] broadcast over rows
static void lstm_add_bias_f32(float* c, const float* bias, size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            c[i * n + j] += bias[j];
        }
    }
}

static void lstm_clear_cache(boat_lstm_layer_t* layer) {
    if (layer->cache_input) {
        boat_tensor_unref(layer->cache_input);
        layer->cache_input = NULL;
    }
    if (layer->cache_gates) {
        for (size_t t = 0; t < layer->cache_seq_len; t++) {
            if (layer->cache_gates[t]) boat_free(layer->cache_gates[t]);
        }
        boat_free(layer->cache_gates);
        layer->cache_gates = NULL;
    }
    if (layer->cache_h) {
        for (size_t t = 0; t < layer->cache_seq_len; t++) {
            if (layer->cache_h[t]) boat_free(layer->cache_h[t]);
        }
        boat_free(layer->cache_h);
        layer->cache_h = NULL;
    }
    if (layer->cache_c) {
        for (size_t t = 0; t < layer->cache_seq_len; t++) {
            if (layer->cache_c[t]) boat_free(layer->cache_c[t]);
        }
        boat_free(layer->cache_c);
        layer->cache_c = NULL;
    }
    layer->cache_seq_len = 0;
    layer->cache_batch = 0;
}

static void lstm_update_tensor(boat_tensor_t* w, boat_tensor_t* gw, float learning_rate) {
    if (!w || !gw) return;
    float* wd = (float*)boat_tensor_data(w);
    float* gd = (float*)boat_tensor_data(gw);
    size_t n = boat_tensor_nelements(w);
    for (size_t i = 0; i < n; i++) {
        wd[i] -= learning_rate * gd[i];
    }
    memset(gd, 0, n * sizeof(float));
}

// Create LSTM layer
BOAT_API boat_lstm_layer_t* BOAT_CALL boat_lstm_layer_create(size_t input_size, size_t hidden_size,
                                                             size_t num_layers, bool bidirectional,
                                                             float dropout) {
    boat_lstm_layer_t* layer =
        (boat_lstm_layer_t*)boat_malloc(sizeof(boat_lstm_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }

    layer->input_size = input_size;
    layer->hidden_size = hidden_size;
    layer->num_layers = num_layers;
    layer->bidirectional = bidirectional;
    layer->dropout = dropout;

    // Initialize parameters to NULL (will be set by model loading)
    layer->weight_ih = NULL;
    layer->weight_hh = NULL;
    layer->bias_ih = NULL;
    layer->bias_hh = NULL;
    layer->hidden_state = NULL;
    layer->cell_state = NULL;

    // Gradient accumulators (created lazily on first backward pass)
    layer->grad_weight_ih = NULL;
    layer->grad_weight_hh = NULL;
    layer->grad_bias_ih = NULL;
    layer->grad_bias_hh = NULL;

    // Cached activations for the backward pass
    layer->cache_input = NULL;
    layer->cache_seq_len = 0;
    layer->cache_batch = 0;
    layer->cache_gates = NULL;
    layer->cache_h = NULL;
    layer->cache_c = NULL;

    return layer;
}

// Free LSTM layer
BOAT_API void BOAT_CALL boat_lstm_layer_free(boat_lstm_layer_t* layer) {
    if (!layer) return;

    if (layer->weight_ih) boat_tensor_unref(layer->weight_ih);
    if (layer->weight_hh) boat_tensor_unref(layer->weight_hh);
    if (layer->bias_ih) boat_tensor_unref(layer->bias_ih);
    if (layer->bias_hh) boat_tensor_unref(layer->bias_hh);
    if (layer->hidden_state) boat_tensor_unref(layer->hidden_state);
    if (layer->cell_state) boat_tensor_unref(layer->cell_state);
    if (layer->grad_weight_ih) boat_tensor_unref(layer->grad_weight_ih);
    if (layer->grad_weight_hh) boat_tensor_unref(layer->grad_weight_hh);
    if (layer->grad_bias_ih) boat_tensor_unref(layer->grad_bias_ih);
    if (layer->grad_bias_hh) boat_tensor_unref(layer->grad_bias_hh);

    lstm_clear_cache(layer);

    boat_free(layer);
}

// Forward pass
BOAT_API boat_tensor_t* BOAT_CALL boat_lstm_layer_forward(boat_lstm_layer_t* layer,
                                                          const boat_tensor_t* input) {
    if (!layer || !input) {
        return NULL;
    }
    if (!layer->weight_ih || !layer->weight_hh || !layer->bias_ih || !layer->bias_hh) {
        return NULL;
    }
    if (boat_tensor_dtype(input) != BOAT_DTYPE_FLOAT32 ||
        boat_tensor_dtype(layer->weight_ih) != BOAT_DTYPE_FLOAT32 ||
        boat_tensor_dtype(layer->weight_hh) != BOAT_DTYPE_FLOAT32 ||
        boat_tensor_dtype(layer->bias_ih) != BOAT_DTYPE_FLOAT32 ||
        boat_tensor_dtype(layer->bias_hh) != BOAT_DTYPE_FLOAT32) {
        return NULL;
    }

    size_t ndim = boat_tensor_ndim(input);
    const int64_t* shape = boat_tensor_shape(input);
    if (ndim != 3) {
        return NULL;
    }
    size_t batch = (size_t)shape[0];
    size_t seq_len = (size_t)shape[1];
    size_t input_size = (size_t)shape[2];
    size_t hidden = layer->hidden_size;
    size_t gate_dim = 4 * hidden;

    if (input_size != layer->input_size) {
        return NULL;
    }

    // Clear any previous cache and clone the input for the backward pass
    lstm_clear_cache(layer);
    layer->cache_input = boat_tensor_clone(input);
    layer->cache_seq_len = seq_len;
    layer->cache_batch = batch;

    // Allocate per-timestep cache arrays
    layer->cache_gates = (float**)boat_malloc(seq_len * sizeof(float*), BOAT_DEVICE_CPU);
    layer->cache_h = (float**)boat_malloc(seq_len * sizeof(float*), BOAT_DEVICE_CPU);
    layer->cache_c = (float**)boat_malloc(seq_len * sizeof(float*), BOAT_DEVICE_CPU);
    if (!layer->cache_gates || !layer->cache_h || !layer->cache_c) {
        // Free whatever was allocated so lstm_clear_cache has nothing to walk
        if (layer->cache_gates) boat_free(layer->cache_gates);
        if (layer->cache_h) boat_free(layer->cache_h);
        if (layer->cache_c) boat_free(layer->cache_c);
        layer->cache_gates = NULL;
        layer->cache_h = NULL;
        layer->cache_c = NULL;
        lstm_clear_cache(layer);
        return NULL;
    }
    for (size_t t = 0; t < seq_len; t++) {
        layer->cache_gates[t] = NULL;
        layer->cache_h[t] = NULL;
        layer->cache_c[t] = NULL;
    }

    // Output tensor [batch, seq_len, hidden]
    const int64_t out_shape[] = {(int64_t)batch, (int64_t)seq_len, (int64_t)hidden};
    boat_tensor_t* output = boat_tensor_create(out_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!output) {
        lstm_clear_cache(layer);
        return NULL;
    }

    const float* x = (const float*)boat_tensor_const_data(input);
    const float* w_ih = (const float*)boat_tensor_const_data(layer->weight_ih);
    const float* w_hh = (const float*)boat_tensor_const_data(layer->weight_hh);
    const float* b_ih = (const float*)boat_tensor_const_data(layer->bias_ih);
    const float* b_hh = (const float*)boat_tensor_const_data(layer->bias_hh);
    float* out_data = (float*)boat_tensor_data(output);

    // Temporary buffers
    float* gates = (float*)boat_malloc(batch * gate_dim * sizeof(float), BOAT_DEVICE_CPU);
    float* tmp = (float*)boat_malloc(batch * gate_dim * sizeof(float), BOAT_DEVICE_CPU);
    float* h_prev = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* h_curr = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* c_prev = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* c_curr = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    if (!gates || !tmp || !h_prev || !h_curr || !c_prev || !c_curr) {
        boat_free(gates);
        boat_free(tmp);
        boat_free(h_prev);
        boat_free(h_curr);
        boat_free(c_prev);
        boat_free(c_curr);
        boat_tensor_free(output);
        lstm_clear_cache(layer);
        return NULL;
    }

    memset(h_prev, 0, batch * hidden * sizeof(float));
    memset(c_prev, 0, batch * hidden * sizeof(float));

    for (size_t t = 0; t < seq_len; t++) {
        const float* x_t = x + t * batch * input_size;

        // gates = x_t @ W_ih + b_ih + (h_prev @ W_hh + b_hh)
        lstm_matmul_f32(x_t, w_ih, gates, batch, input_size, gate_dim);
        lstm_add_bias_f32(gates, b_ih, batch, gate_dim);
        lstm_matmul_f32(h_prev, w_hh, tmp, batch, hidden, gate_dim);
        for (size_t i = 0; i < batch * gate_dim; i++) {
            gates[i] += tmp[i] + b_hh[i % gate_dim];
        }

        // Gate activations: i = sigmoid(g[0:h]), f = sigmoid(g[h:2h]),
        // cell = tanh(g[2h:3h]), o = sigmoid(g[3h:4h]).
        // c_t = f .* c_{t-1} + i .* cell; h_t = o .* tanh(c_t)
        for (size_t b = 0; b < batch; b++) {
            float* row = gates + b * gate_dim;
            float* hb = h_curr + b * hidden;
            float* cb = c_curr + b * hidden;
            const float* cpb = c_prev + b * hidden;
            size_t j = 0;
#if BOAT_HAVE_AVX2
            const size_t h2 = hidden * 2;
            const size_t h3 = hidden * 3;
            for (; j + 8 <= hidden; j += 8) {
                __m256 gi = _mm256_loadu_ps(row + j);
                __m256 gf = _mm256_loadu_ps(row + hidden + j);
                __m256 gc = _mm256_loadu_ps(row + h2 + j);
                __m256 go = _mm256_loadu_ps(row + h3 + j);
                __m256 ig = boat_simd_sigmoid256(gi);
                __m256 fg = boat_simd_sigmoid256(gf);
                __m256 cg = boat_simd_tanh256(gc);
                __m256 og = boat_simd_sigmoid256(go);
                __m256 cp = _mm256_loadu_ps(cpb + j);
                __m256 cn = _mm256_add_ps(_mm256_mul_ps(fg, cp), _mm256_mul_ps(ig, cg));
                _mm256_storeu_ps(cb + j, cn);
                _mm256_storeu_ps(hb + j, _mm256_mul_ps(og, boat_simd_tanh256(cn)));
            }
#endif
            for (; j < hidden; j++) {
                float ig = lstm_sigmoid_f32(row[j]);
                float fg = lstm_sigmoid_f32(row[hidden + j]);
                float cg = tanhf(row[2 * hidden + j]);
                float og = lstm_sigmoid_f32(row[3 * hidden + j]);

                float c_new = fg * cpb[j] + ig * cg;
                cb[j] = c_new;
                hb[j] = og * tanhf(c_new);
            }
        }

        // Cache raw gates, hidden and cell states for the backward pass
        layer->cache_gates[t] =
            (float*)boat_malloc(batch * gate_dim * sizeof(float), BOAT_DEVICE_CPU);
        layer->cache_h[t] = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
        layer->cache_c[t] = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
        if (!layer->cache_gates[t] || !layer->cache_h[t] || !layer->cache_c[t]) {
            if (layer->cache_gates[t]) boat_free(layer->cache_gates[t]);
            if (layer->cache_h[t]) boat_free(layer->cache_h[t]);
            if (layer->cache_c[t]) boat_free(layer->cache_c[t]);
            layer->cache_gates[t] = NULL;
            layer->cache_h[t] = NULL;
            layer->cache_c[t] = NULL;
            boat_free(gates);
            boat_free(tmp);
            boat_free(h_prev);
            boat_free(h_curr);
            boat_free(c_prev);
            boat_free(c_curr);
            boat_tensor_free(output);
            lstm_clear_cache(layer);
            return NULL;
        }
        memcpy(layer->cache_gates[t], gates, batch * gate_dim * sizeof(float));
        memcpy(layer->cache_h[t], h_curr, batch * hidden * sizeof(float));
        memcpy(layer->cache_c[t], c_curr, batch * hidden * sizeof(float));

        // h_t goes to output[t]
        memcpy(out_data + t * batch * hidden, h_curr, batch * hidden * sizeof(float));

        // Advance to the next timestep
        memcpy(h_prev, h_curr, batch * hidden * sizeof(float));
        memcpy(c_prev, c_curr, batch * hidden * sizeof(float));
    }

    boat_free(gates);
    boat_free(tmp);
    boat_free(h_prev);
    boat_free(h_curr);
    boat_free(c_prev);
    boat_free(c_curr);

    return output;
}

// Backward pass
BOAT_API boat_tensor_t* BOAT_CALL boat_lstm_layer_backward(boat_lstm_layer_t* layer,
                                                           const boat_tensor_t* grad_output) {
    if (!layer || !grad_output) {
        return NULL;
    }
    if (!layer->cache_input || !layer->cache_gates || !layer->cache_h || !layer->cache_c) {
        return NULL;
    }
    if (!layer->weight_ih || !layer->weight_hh || !layer->bias_ih || !layer->bias_hh) {
        return NULL;
    }

    size_t batch = layer->cache_batch;
    size_t seq_len = layer->cache_seq_len;
    size_t input_size = layer->input_size;
    size_t hidden = layer->hidden_size;
    size_t gate_dim = 4 * hidden;

    // Validate grad_output shape [batch, seq_len, hidden]
    const int64_t* go_shape = boat_tensor_shape(grad_output);
    if (boat_tensor_ndim(grad_output) != 3 || (size_t)go_shape[0] != batch ||
        (size_t)go_shape[1] != seq_len || (size_t)go_shape[2] != hidden) {
        return NULL;
    }

    // Gradient w.r.t. input [batch, seq_len, input_size]
    const int64_t in_shape[] = {(int64_t)batch, (int64_t)seq_len, (int64_t)input_size};
    boat_tensor_t* grad_input =
        boat_tensor_create(in_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!grad_input) {
        return NULL;
    }

    const float* x = (const float*)boat_tensor_const_data(layer->cache_input);
    const float* dy = (const float*)boat_tensor_const_data(grad_output);
    const float* w_ih = (const float*)boat_tensor_const_data(layer->weight_ih);
    const float* w_hh = (const float*)boat_tensor_const_data(layer->weight_hh);
    float* dx = (float*)boat_tensor_data(grad_input);

    // Lazy-create gradient accumulators
    if (!layer->grad_weight_ih) {
        layer->grad_weight_ih = boat_tensor_create_like(layer->weight_ih);
        if (layer->grad_weight_ih) {
            memset(boat_tensor_data(layer->grad_weight_ih), 0,
                   boat_tensor_nbytes(layer->grad_weight_ih));
        }
    }
    if (!layer->grad_weight_hh) {
        layer->grad_weight_hh = boat_tensor_create_like(layer->weight_hh);
        if (layer->grad_weight_hh) {
            memset(boat_tensor_data(layer->grad_weight_hh), 0,
                   boat_tensor_nbytes(layer->grad_weight_hh));
        }
    }
    if (!layer->grad_bias_ih) {
        layer->grad_bias_ih = boat_tensor_create_like(layer->bias_ih);
        if (layer->grad_bias_ih) {
            memset(boat_tensor_data(layer->grad_bias_ih), 0,
                   boat_tensor_nbytes(layer->grad_bias_ih));
        }
    }
    if (!layer->grad_bias_hh) {
        layer->grad_bias_hh = boat_tensor_create_like(layer->bias_hh);
        if (layer->grad_bias_hh) {
            memset(boat_tensor_data(layer->grad_bias_hh), 0,
                   boat_tensor_nbytes(layer->grad_bias_hh));
        }
    }
    if (!layer->grad_weight_ih || !layer->grad_weight_hh || !layer->grad_bias_ih ||
        !layer->grad_bias_hh) {
        boat_tensor_free(grad_input);
        return NULL;
    }

    float* gw_ih = (float*)boat_tensor_data(layer->grad_weight_ih);
    float* gw_hh = (float*)boat_tensor_data(layer->grad_weight_hh);
    float* gb_ih = (float*)boat_tensor_data(layer->grad_bias_ih);
    float* gb_hh = (float*)boat_tensor_data(layer->grad_bias_hh);

    // Temporary buffers
    float* d_gates = (float*)boat_malloc(batch * gate_dim * sizeof(float), BOAT_DEVICE_CPU);
    float* d_x_t = (float*)boat_malloc(batch * input_size * sizeof(float), BOAT_DEVICE_CPU);
    float* d_h = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* d_c = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* d_h_next = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* d_c_next = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    // Scratch for grad_W = x^T @ d_gates: needs max(input_size, hidden) * gate_dim
    float* acc = (float*)boat_malloc(
        ((input_size > hidden ? input_size : hidden) * gate_dim) * sizeof(float), BOAT_DEVICE_CPU);
    float* acc_bias = (float*)boat_malloc(gate_dim * sizeof(float), BOAT_DEVICE_CPU);
    if (!d_gates || !d_x_t || !d_h || !d_c || !d_h_next || !d_c_next || !acc || !acc_bias) {
        boat_free(d_gates);
        boat_free(d_x_t);
        boat_free(d_h);
        boat_free(d_c);
        boat_free(d_h_next);
        boat_free(d_c_next);
        boat_free(acc);
        boat_free(acc_bias);
        boat_tensor_free(grad_input);
        return NULL;
    }

    memset(d_h_next, 0, batch * hidden * sizeof(float));
    memset(d_c_next, 0, batch * hidden * sizeof(float));

    for (size_t t = seq_len; t-- > 0;) {
        const float* dy_t = dy + t * batch * hidden;
        const float* gates_t = layer->cache_gates[t];
        const float* c_t = layer->cache_c[t];
        const float* c_prev_t = (t == 0) ? NULL : layer->cache_c[t - 1];
        const float* h_prev_t = (t == 0) ? NULL : layer->cache_h[t - 1];

        // d_h = dy_t + d_h_next
        for (size_t i = 0; i < batch * hidden; i++) {
            d_h[i] = dy_t[i] + d_h_next[i];
        }

        // Backprop through the gates using cached pre-activations and states
        for (size_t b = 0; b < batch; b++) {
            const float* row = gates_t + b * gate_dim;
            const float* cb = c_t + b * hidden;
            const float* cpb = (c_prev_t != NULL) ? c_prev_t + b * hidden : NULL;
            float* dg = d_gates + b * gate_dim;
            float* dhb = d_h + b * hidden;
            float* dcb = d_c + b * hidden;

            size_t j = 0;
#if BOAT_HAVE_AVX2
            const size_t h2 = hidden * 2;
            const size_t h3 = hidden * 3;
            const __m256 one = _mm256_set1_ps(1.0f);
            const __m256 zero = _mm256_setzero_ps();
            const float* dcn = d_c_next + b * hidden;
            for (; j + 8 <= hidden; j += 8) {
                __m256 vig = boat_simd_sigmoid256(_mm256_loadu_ps(row + j));
                __m256 vfg = boat_simd_sigmoid256(_mm256_loadu_ps(row + hidden + j));
                __m256 vcg = boat_simd_tanh256(_mm256_loadu_ps(row + h2 + j));
                __m256 vog = boat_simd_sigmoid256(_mm256_loadu_ps(row + h3 + j));
                __m256 vtanh_c = boat_simd_tanh256(_mm256_loadu_ps(cb + j));

                __m256 vdhb = _mm256_loadu_ps(dhb + j);
                __m256 vd_o = _mm256_mul_ps(vdhb, vtanh_c);
                __m256 vdc = _mm256_mul_ps(
                    vdhb, _mm256_mul_ps(vog, _mm256_fnmadd_ps(vtanh_c, vtanh_c, one)));
                vdc = _mm256_add_ps(vdc, _mm256_loadu_ps(dcn + j));
                __m256 vcpb = cpb ? _mm256_loadu_ps(cpb + j) : zero;
                __m256 vdf = _mm256_mul_ps(vdc, vcpb);
                __m256 vdi = _mm256_mul_ps(vdc, vcg);
                __m256 vdcell = _mm256_mul_ps(vdc, vig);

                _mm256_storeu_ps(dg + j, _mm256_mul_ps(vdi, _mm256_mul_ps(vig, _mm256_sub_ps(one, vig))));
                _mm256_storeu_ps(dg + hidden + j,
                                 _mm256_mul_ps(vdf, _mm256_mul_ps(vfg, _mm256_sub_ps(one, vfg))));
                _mm256_storeu_ps(dg + h2 + j, _mm256_mul_ps(vdcell, _mm256_fnmadd_ps(vcg, vcg, one)));
                _mm256_storeu_ps(dg + h3 + j,
                                 _mm256_mul_ps(vd_o, _mm256_mul_ps(vog, _mm256_sub_ps(one, vog))));

                _mm256_storeu_ps(dcb + j, vdc);
            }
#endif
            for (; j < hidden; j++) {
                float ig = lstm_sigmoid_f32(row[j]);
                float fg = lstm_sigmoid_f32(row[hidden + j]);
                float cg = tanhf(row[2 * hidden + j]);
                float og = lstm_sigmoid_f32(row[3 * hidden + j]);
                float tanh_c = tanhf(cb[j]);

                float d_o = dhb[j] * tanh_c;
                float d_c_full = dhb[j] * og * (1.0f - tanh_c * tanh_c) + d_c_next[b * hidden + j];
                float df = d_c_full * (cpb ? cpb[j] : 0.0f);
                float di = d_c_full * cg;
                float dcell = d_c_full * ig;

                dg[j] = di * ig * (1.0f - ig);
                dg[hidden + j] = df * fg * (1.0f - fg);
                dg[2 * hidden + j] = dcell * (1.0f - cg * cg);
                dg[3 * hidden + j] = d_o * og * (1.0f - og);

                dcb[j] = d_c_full;
            }
        }

        // d_x_t = d_gates @ W_ih^T
        lstm_matmul_bt(d_gates, w_ih, d_x_t, batch, gate_dim, input_size);
        memcpy(dx + t * batch * input_size, d_x_t, batch * input_size * sizeof(float));

        // grad_W_ih += x_t^T @ d_gates ; grad_W_hh += h_prev^T @ d_gates
        lstm_matmul_at(x + t * batch * input_size, d_gates, acc, batch, input_size, gate_dim);
        for (size_t i = 0; i < input_size * gate_dim; i++) {
            gw_ih[i] += acc[i];
        }
        if (h_prev_t != NULL) {
            lstm_matmul_at(h_prev_t, d_gates, acc, batch, hidden, gate_dim);
            for (size_t i = 0; i < hidden * gate_dim; i++) {
                gw_hh[i] += acc[i];
            }
        }

        // grad_bias += sum over the batch dimension of d_gates
        memset(acc_bias, 0, gate_dim * sizeof(float));
        for (size_t b = 0; b < batch; b++) {
            const float* dg = d_gates + b * gate_dim;
            for (size_t j = 0; j < gate_dim; j++) {
                acc_bias[j] += dg[j];
            }
        }
        for (size_t j = 0; j < gate_dim; j++) {
            gb_ih[j] += acc_bias[j];
            gb_hh[j] += acc_bias[j];
        }

        // d_h_next = d_gates @ W_hh^T ; d_c_next = d_c .* f
        lstm_matmul_bt(d_gates, w_hh, d_h_next, batch, gate_dim, hidden);
        for (size_t b = 0; b < batch; b++) {
            const float* row = gates_t + b * gate_dim;
            float* dcn = d_c_next + b * hidden;
            for (size_t j = 0; j < hidden; j++) {
                float fg = lstm_sigmoid_f32(row[hidden + j]);
                dcn[j] = d_c[b * hidden + j] * fg;
            }
        }
    }

    boat_free(d_gates);
    boat_free(d_x_t);
    boat_free(d_h);
    boat_free(d_c);
    boat_free(d_h_next);
    boat_free(d_c_next);
    boat_free(acc);
    boat_free(acc_bias);

    return grad_input;
}

// Update parameters
BOAT_API void BOAT_CALL boat_lstm_layer_update(boat_lstm_layer_t* layer, float learning_rate) {
    if (!layer) return;
    lstm_update_tensor(layer->weight_ih, layer->grad_weight_ih, learning_rate);
    lstm_update_tensor(layer->weight_hh, layer->grad_weight_hh, learning_rate);
    lstm_update_tensor(layer->bias_ih, layer->grad_bias_ih, learning_rate);
    lstm_update_tensor(layer->bias_hh, layer->grad_bias_hh, learning_rate);
}

// Parameter setters for model loading
BOAT_API void BOAT_CALL boat_lstm_layer_set_weight_ih(boat_lstm_layer_t* layer,
                                                      boat_tensor_t* weight) {
    if (layer->weight_ih) boat_tensor_unref(layer->weight_ih);
    layer->weight_ih = weight;
    if (weight) boat_tensor_ref(weight);
}

BOAT_API void BOAT_CALL boat_lstm_layer_set_weight_hh(boat_lstm_layer_t* layer,
                                                      boat_tensor_t* weight) {
    if (layer->weight_hh) boat_tensor_unref(layer->weight_hh);
    layer->weight_hh = weight;
    if (weight) boat_tensor_ref(weight);
}

BOAT_API void BOAT_CALL boat_lstm_layer_set_bias_ih(boat_lstm_layer_t* layer, boat_tensor_t* bias) {
    if (layer->bias_ih) boat_tensor_unref(layer->bias_ih);
    layer->bias_ih = bias;
    if (bias) boat_tensor_ref(bias);
}

BOAT_API void BOAT_CALL boat_lstm_layer_set_bias_hh(boat_lstm_layer_t* layer, boat_tensor_t* bias) {
    if (layer->bias_hh) boat_tensor_unref(layer->bias_hh);
    layer->bias_hh = bias;
    if (bias) boat_tensor_ref(bias);
}

// Gradient tensor getters
BOAT_API boat_tensor_t* BOAT_CALL
boat_lstm_layer_get_grad_weight_ih(const boat_lstm_layer_t* layer) {
    if (!layer) return NULL;
    return layer->grad_weight_ih;
}

BOAT_API boat_tensor_t* BOAT_CALL
boat_lstm_layer_get_grad_weight_hh(const boat_lstm_layer_t* layer) {
    if (!layer) return NULL;
    return layer->grad_weight_hh;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_lstm_layer_get_grad_bias_ih(const boat_lstm_layer_t* layer) {
    if (!layer) return NULL;
    return layer->grad_bias_ih;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_lstm_layer_get_grad_bias_hh(const boat_lstm_layer_t* layer) {
    if (!layer) return NULL;
    return layer->grad_bias_hh;
}
