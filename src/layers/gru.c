// gru.c - GRU layer implementation
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

// GRU layer structure
struct boat_gru_layer_t {
    size_t input_size;
    size_t hidden_size;
    size_t num_layers;
    bool bidirectional;
    float dropout;

    // Parameters (weights and biases)
    boat_tensor_t* weight_ih; // Input-hidden weights [input_size, 3*hidden]
    boat_tensor_t* weight_hh; // Hidden-hidden weights [hidden_size, 3*hidden]
    boat_tensor_t* bias_ih;   // Input-hidden biases [3*hidden]
    boat_tensor_t* bias_hh;   // Hidden-hidden biases [3*hidden]

    // Gradient accumulators for training
    boat_tensor_t* grad_weight_ih;
    boat_tensor_t* grad_weight_hh;
    boat_tensor_t* grad_bias_ih;
    boat_tensor_t* grad_bias_hh;

    // Internal state
    boat_tensor_t* hidden_state;

    // Cached states for backward pass
    boat_tensor_t* cache_input; // Input tensor [batch, seq_len, input_size]
    size_t cache_seq_len;
    size_t cache_batch;
    float** cache_gates; // Combined raw pre-activations per timestep [batch, 3*hidden]
    float** cache_a_hh;  // Hidden contribution per timestep [batch, 3*hidden]
    float** cache_h;     // Hidden state per timestep [batch, hidden]
};

// GRU gate layout: 3 * hidden_size, ordered as [reset, update, new].
// Weights use the boat [in, out] convention (W_ih: [input_size, 3h], W_hh: [hidden_size, 3h]).
// Input is [batch, seq_len, input_size]; output is [batch, seq_len, hidden_size].
// Recurrence (PyTorch convention):
//   r = sigmoid(x @ W_ir + b_ir + h_prev @ W_hr + b_hr)
//   z = sigmoid(x @ W_iz + b_iz + h_prev @ W_hz + b_hz)
//   n = tanh(x @ W_in + b_in + r .* (h_prev @ W_hn + b_hn))
//   h = (1 - z) .* n + z .* h_prev

static float gru_sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// c[m, n] = a[m, k] @ b[k, n]
static void gru_matmul_f32(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
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
static void gru_matmul_bt(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
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
static void gru_matmul_at(const float* a, const float* b, float* c, size_t k, size_t m, size_t n) {
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
static void gru_add_bias_f32(float* c, const float* bias, size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            c[i * n + j] += bias[j];
        }
    }
}

static void gru_clear_cache(boat_gru_layer_t* layer) {
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
    if (layer->cache_a_hh) {
        for (size_t t = 0; t < layer->cache_seq_len; t++) {
            if (layer->cache_a_hh[t]) boat_free(layer->cache_a_hh[t]);
        }
        boat_free(layer->cache_a_hh);
        layer->cache_a_hh = NULL;
    }
    if (layer->cache_h) {
        for (size_t t = 0; t < layer->cache_seq_len; t++) {
            if (layer->cache_h[t]) boat_free(layer->cache_h[t]);
        }
        boat_free(layer->cache_h);
        layer->cache_h = NULL;
    }
    layer->cache_seq_len = 0;
    layer->cache_batch = 0;
}

static void gru_update_tensor(boat_tensor_t* w, boat_tensor_t* gw, float learning_rate) {
    if (!w || !gw) return;
    float* wd = (float*)boat_tensor_data(w);
    float* gd = (float*)boat_tensor_data(gw);
    size_t n = boat_tensor_nelements(w);
    for (size_t i = 0; i < n; i++) {
        wd[i] -= learning_rate * gd[i];
    }
    memset(gd, 0, n * sizeof(float));
}

// Create GRU layer
BOAT_API boat_gru_layer_t* BOAT_CALL boat_gru_layer_create(size_t input_size, size_t hidden_size,
                                                           size_t num_layers, bool bidirectional,
                                                           float dropout) {
    boat_gru_layer_t* layer =
        (boat_gru_layer_t*)boat_malloc(sizeof(boat_gru_layer_t), BOAT_DEVICE_CPU);
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
    layer->cache_a_hh = NULL;
    layer->cache_h = NULL;

    return layer;
}

// Free GRU layer
BOAT_API void BOAT_CALL boat_gru_layer_free(boat_gru_layer_t* layer) {
    if (!layer) return;

    if (layer->weight_ih) boat_tensor_unref(layer->weight_ih);
    if (layer->weight_hh) boat_tensor_unref(layer->weight_hh);
    if (layer->bias_ih) boat_tensor_unref(layer->bias_ih);
    if (layer->bias_hh) boat_tensor_unref(layer->bias_hh);
    if (layer->hidden_state) boat_tensor_unref(layer->hidden_state);
    if (layer->grad_weight_ih) boat_tensor_unref(layer->grad_weight_ih);
    if (layer->grad_weight_hh) boat_tensor_unref(layer->grad_weight_hh);
    if (layer->grad_bias_ih) boat_tensor_unref(layer->grad_bias_ih);
    if (layer->grad_bias_hh) boat_tensor_unref(layer->grad_bias_hh);

    gru_clear_cache(layer);

    boat_free(layer);
}

// Forward pass
BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_forward(boat_gru_layer_t* layer,
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
    size_t gate_dim = 3 * hidden;

    if (input_size != layer->input_size) {
        return NULL;
    }

    // Clear any previous cache and clone the input for the backward pass
    gru_clear_cache(layer);
    layer->cache_input = boat_tensor_clone(input);
    layer->cache_seq_len = seq_len;
    layer->cache_batch = batch;

    // Allocate per-timestep cache arrays
    layer->cache_gates = (float**)boat_malloc(seq_len * sizeof(float*), BOAT_DEVICE_CPU);
    layer->cache_a_hh = (float**)boat_malloc(seq_len * sizeof(float*), BOAT_DEVICE_CPU);
    layer->cache_h = (float**)boat_malloc(seq_len * sizeof(float*), BOAT_DEVICE_CPU);
    if (!layer->cache_gates || !layer->cache_a_hh || !layer->cache_h) {
        // Free whatever was allocated so gru_clear_cache has nothing to walk
        if (layer->cache_gates) boat_free(layer->cache_gates);
        if (layer->cache_a_hh) boat_free(layer->cache_a_hh);
        if (layer->cache_h) boat_free(layer->cache_h);
        layer->cache_gates = NULL;
        layer->cache_a_hh = NULL;
        layer->cache_h = NULL;
        gru_clear_cache(layer);
        return NULL;
    }
    for (size_t t = 0; t < seq_len; t++) {
        layer->cache_gates[t] = NULL;
        layer->cache_a_hh[t] = NULL;
        layer->cache_h[t] = NULL;
    }

    // Output tensor [batch, seq_len, hidden]
    const int64_t out_shape[] = {(int64_t)batch, (int64_t)seq_len, (int64_t)hidden};
    boat_tensor_t* output = boat_tensor_create(out_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!output) {
        gru_clear_cache(layer);
        return NULL;
    }

    const float* x = (const float*)boat_tensor_const_data(input);
    const float* w_ih = (const float*)boat_tensor_const_data(layer->weight_ih);
    const float* w_hh = (const float*)boat_tensor_const_data(layer->weight_hh);
    const float* b_ih = (const float*)boat_tensor_const_data(layer->bias_ih);
    const float* b_hh = (const float*)boat_tensor_const_data(layer->bias_hh);
    float* out_data = (float*)boat_tensor_data(output);

    // Temporary buffers
    float* a_ih = (float*)boat_malloc(batch * gate_dim * sizeof(float), BOAT_DEVICE_CPU);
    float* a_hh = (float*)boat_malloc(batch * gate_dim * sizeof(float), BOAT_DEVICE_CPU);
    float* combined = (float*)boat_malloc(batch * gate_dim * sizeof(float), BOAT_DEVICE_CPU);
    float* h_prev = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* h_curr = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* rbuf = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    if (!a_ih || !a_hh || !combined || !h_prev || !h_curr || !rbuf) {
        boat_free(a_ih);
        boat_free(a_hh);
        boat_free(combined);
        boat_free(h_prev);
        boat_free(h_curr);
        boat_free(rbuf);
        boat_tensor_free(output);
        gru_clear_cache(layer);
        return NULL;
    }

    memset(h_prev, 0, batch * hidden * sizeof(float));

    for (size_t t = 0; t < seq_len; t++) {
        const float* x_t = x + t * batch * input_size;

        // a_ih = x_t @ W_ih + b_ih ; a_hh = h_prev @ W_hh + b_hh
        gru_matmul_f32(x_t, w_ih, a_ih, batch, input_size, gate_dim);
        gru_add_bias_f32(a_ih, b_ih, batch, gate_dim);
        gru_matmul_f32(h_prev, w_hh, a_hh, batch, hidden, gate_dim);
        gru_add_bias_f32(a_hh, b_hh, batch, gate_dim);

        // Combined pre-activations for the reset and update gates
        for (size_t i = 0; i < batch * gate_dim; i++) {
            combined[i] = a_ih[i] + a_hh[i];
        }

        // Gate activations: r = sigmoid(g[0:h]), z = sigmoid(g[h:2h]).
        // The candidate applies the reset to h_prev BEFORE the recurrent
        // matmul (MATLAB/PyTorch convention):
        //   n = tanh(a_ih[2h:3h] + (r .* h_prev) @ W_hh[2h:3h] + b_hh[2h:3h])
        //   h = (1 - z) .* h_prev + z .* n
        // r/z gates (vectorized sigmoid over the combined pre-activations).
        for (size_t b = 0; b < batch; b++) {
            const float* row = combined + b * gate_dim;
            float* rb = rbuf + b * hidden;
            size_t j = 0;
#if BOAT_HAVE_AVX2
            for (; j + 8 <= hidden; j += 8) {
                __m256 r = boat_simd_sigmoid256(_mm256_loadu_ps(row + j));
                _mm256_storeu_ps(rb + j, r);
            }
#endif
            for (; j < hidden; j++)
                rb[j] = gru_sigmoid_f32(row[j]);
        }
        // Candidate + hidden-state update (scalar; needs the masked matmul).
        for (size_t b = 0; b < batch; b++) {
            const float* row = combined + b * gate_dim;
            const float* aih = a_ih + b * gate_dim;
            const float* hpb = h_prev + b * hidden;
            const float* rb = rbuf + b * hidden;
            const float* bhh = b_hh + 2 * hidden;
            const float* whh_n = w_hh + 2 * hidden; // w_hh[k][2h+j]
            float* hb = h_curr + b * hidden;
            for (size_t j = 0; j < hidden; j++) {
                float z = gru_sigmoid_f32(row[hidden + j]);
                float acc = bhh[j];
                for (size_t k = 0; k < hidden; k++)
                    acc += whh_n[k * gate_dim + j] * (rb[k] * hpb[k]);
                float n = tanhf(aih[2 * hidden + j] + acc);
                hb[j] = (1.0f - z) * hpb[j] + z * n;
            }
        }

        // Cache combined gates, hidden contribution and hidden state for backward
        layer->cache_gates[t] =
            (float*)boat_malloc(batch * gate_dim * sizeof(float), BOAT_DEVICE_CPU);
        layer->cache_a_hh[t] =
            (float*)boat_malloc(batch * gate_dim * sizeof(float), BOAT_DEVICE_CPU);
        layer->cache_h[t] = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
        if (!layer->cache_gates[t] || !layer->cache_a_hh[t] || !layer->cache_h[t]) {
            if (layer->cache_gates[t]) boat_free(layer->cache_gates[t]);
            if (layer->cache_a_hh[t]) boat_free(layer->cache_a_hh[t]);
            if (layer->cache_h[t]) boat_free(layer->cache_h[t]);
            layer->cache_gates[t] = NULL;
            layer->cache_a_hh[t] = NULL;
            layer->cache_h[t] = NULL;
            boat_free(a_ih);
            boat_free(a_hh);
            boat_free(combined);
            boat_free(h_prev);
            boat_free(h_curr);
            boat_free(rbuf);
            boat_tensor_free(output);
            gru_clear_cache(layer);
            return NULL;
        }
        memcpy(layer->cache_gates[t], combined, batch * gate_dim * sizeof(float));
        memcpy(layer->cache_a_hh[t], a_hh, batch * gate_dim * sizeof(float));
        memcpy(layer->cache_h[t], h_curr, batch * hidden * sizeof(float));

        // h_t goes to output[t]
        memcpy(out_data + t * batch * hidden, h_curr, batch * hidden * sizeof(float));

        // Advance to the next timestep
        memcpy(h_prev, h_curr, batch * hidden * sizeof(float));
    }

    boat_free(a_ih);
    boat_free(a_hh);
    boat_free(combined);
    boat_free(h_prev);
    boat_free(h_curr);
    boat_free(rbuf);

    return output;
}

// Backward pass
BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_backward(boat_gru_layer_t* layer,
                                                          const boat_tensor_t* grad_output) {
    if (!layer || !grad_output) {
        return NULL;
    }
    if (!layer->cache_input || !layer->cache_gates || !layer->cache_a_hh || !layer->cache_h) {
        return NULL;
    }
    if (!layer->weight_ih || !layer->weight_hh || !layer->bias_ih || !layer->bias_hh) {
        return NULL;
    }

    size_t batch = layer->cache_batch;
    size_t seq_len = layer->cache_seq_len;
    size_t input_size = layer->input_size;
    size_t hidden = layer->hidden_size;
    size_t gate_dim = 3 * hidden;

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
    const float* b_ih = (const float*)boat_tensor_const_data(layer->bias_ih);
    const float* b_hh = (const float*)boat_tensor_const_data(layer->bias_hh);
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
    float* d_a_ih = (float*)boat_malloc(batch * gate_dim * sizeof(float), BOAT_DEVICE_CPU);
    float* d_a_hh = (float*)boat_malloc(batch * gate_dim * sizeof(float), BOAT_DEVICE_CPU);
    float* d_x_t = (float*)boat_malloc(batch * input_size * sizeof(float), BOAT_DEVICE_CPU);
    float* d_h = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* d_h_rec = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* d_h_next = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* d_a_n = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* hm = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* da_hh_rz = (float*)boat_malloc(batch * gate_dim * sizeof(float), BOAT_DEVICE_CPU);
    float* rv2 = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    float* zv2 = (float*)boat_malloc(batch * hidden * sizeof(float), BOAT_DEVICE_CPU);
    // Scratch for grad_W = x^T @ d_a: needs max(input_size, hidden) * gate_dim
    float* acc = (float*)boat_malloc(
        ((input_size > hidden ? input_size : hidden) * gate_dim) * sizeof(float), BOAT_DEVICE_CPU);
    float* acc_bias = (float*)boat_malloc(gate_dim * sizeof(float), BOAT_DEVICE_CPU);
    if (!d_a_ih || !d_a_hh || !d_x_t || !d_h || !d_h_rec || !d_h_next || !acc || !acc_bias ||
        !d_a_n || !hm || !da_hh_rz || !rv2 || !zv2) {
        boat_free(d_a_ih);
        boat_free(d_a_hh);
        boat_free(d_x_t);
        boat_free(d_h);
        boat_free(d_h_rec);
        boat_free(d_h_next);
        boat_free(d_a_n);
        boat_free(hm);
        boat_free(da_hh_rz);
        boat_free(rv2);
        boat_free(zv2);
        boat_free(acc);
        boat_free(acc_bias);
        boat_tensor_free(grad_input);
        return NULL;
    }

    memset(d_h_next, 0, batch * hidden * sizeof(float));

    for (size_t t = seq_len; t-- > 0;) {
        const float* dy_t = dy + t * batch * hidden;
        const float* gates_t = layer->cache_gates[t];
        const float* h_prev_t = (t == 0) ? NULL : layer->cache_h[t - 1];

        // d_h = dy_t + d_h_next
        for (size_t i = 0; i < batch * hidden; i++) {
            d_h[i] = dy_t[i] + d_h_next[i];
        }

        /* Backprop through the reset/update/new gates. The candidate applies
           the reset to h_prev BEFORE the recurrent matmul (see forward):
           a_n = a_ih[n] + (r .* h_prev) @ W_hh[n] + b_hh[n]. r/z are
           recomputed from the cached combined pre-activations. */
        memset(da_hh_rz, 0, batch * gate_dim * sizeof(float));
        for (size_t b = 0; b < batch; b++) {
            const float* row = gates_t + b * gate_dim;
            const float* hpb = (h_prev_t != NULL) ? h_prev_t + b * hidden : NULL;
            const float* xt = x + t * batch * input_size + b * input_size;
            float* da_ih = d_a_ih + b * gate_dim;
            float* dhb = d_h + b * hidden;
            float* dhr = d_h_rec + b * hidden;
            float* hmb = hm + b * hidden;

            /* first pass A: r/z gates for every unit of this sample (the
               candidate sum needs r_p for ALL p, so gates are precomputed) */
            for (size_t j = 0; j < hidden; j++) {
                float r = gru_sigmoid_f32(row[j]);
                float z = gru_sigmoid_f32(row[hidden + j]);
                rv2[b * hidden + j] = r;
                zv2[b * hidden + j] = z;
                hmb[j] = r * (hpb ? hpb[j] : 0.0f);
            }
            /* first pass B: candidate pre-activation (reset applied to h_prev
               per unit BEFORE the recurrent matmul) + per-unit deltas */
            for (size_t j = 0; j < hidden; j++) {
                float hp = hpb ? hpb[j] : 0.0f;
                float z = zv2[b * hidden + j];

                float a_n = b_ih[2 * hidden + j] + b_hh[2 * hidden + j];
                for (size_t p = 0; p < input_size; p++)
                    a_n += xt[p] * w_ih[p * gate_dim + 2 * hidden + j];
                for (size_t p = 0; p < hidden; p++)
                    a_n += w_hh[p * gate_dim + 2 * hidden + j] *
                           (rv2[b * hidden + p] * (hpb ? hpb[p] : 0.0f));
                float n = tanhf(a_n);

                float d_n_raw = (dhb[j] * z) * (1.0f - n * n);
                float d_z_raw = (dhb[j] * (n - hp)) * z * (1.0f - z);

                dhr[j] = dhb[j] * (1.0f - z);
                da_ih[j] = 0.0f; /* d_a_r filled in the second pass */
                da_ih[hidden + j] = d_z_raw;
                da_ih[2 * hidden + j] = d_n_raw;
                d_a_n[b * hidden + j] = d_n_raw;
            }
            /* second pass: d_r from the masked recurrent term; d_h from the
               r/z gates into every destination unit p */
            for (size_t j = 0; j < hidden; j++) {
                float d_r = 0.0f;
                /* d_r_j = h_j * sum_p W_hh[j][n+p] * d_a_n_p (row j of the
                   n-block dotted with the candidate deltas) */
                for (size_t p = 0; p < hidden; p++)
                    d_r += w_hh[j * gate_dim + 2 * hidden + p] * d_a_n[b * hidden + p];
                d_r *= (hpb ? hpb[j] : 0.0f);
                float r = rv2[b * hidden + j];
                float d_r_raw = d_r * r * (1.0f - r);
                da_ih[j] = d_r_raw;
                for (size_t p = 0; p < hidden; p++) {
                    dhr[p] += w_hh[p * gate_dim + j] * d_r_raw +
                              w_hh[p * gate_dim + hidden + j] * da_ih[hidden + j];
                }
            }
            /* d_h from the n gate: r_p * (W_hh[n]^T d_a_n)_p */
            for (size_t p = 0; p < hidden; p++) {
                float s = 0.0f;
                for (size_t j = 0; j < hidden; j++)
                    s += w_hh[p * gate_dim + 2 * hidden + j] * d_a_n[b * hidden + j];
                dhr[p] += rv2[b * hidden + p] * s;
            }

            /* the generic W_hh / bias accumulation uses the r/z blocks only;
               the n block is handled via the masked state below */
            memcpy(da_hh_rz + b * gate_dim, da_ih, hidden * sizeof(float));
            memcpy(da_hh_rz + b * gate_dim + hidden, da_ih + hidden, hidden * sizeof(float));
        }

        /* d_x_t = d_a_ih @ W_ih^T ; d_h_next = da_hh_rz @ W_hh^T + d_h_rec */
        gru_matmul_bt(d_a_ih, w_ih, d_x_t, batch, gate_dim, input_size);
        memcpy(dx + t * batch * input_size, d_x_t, batch * input_size * sizeof(float));
        gru_matmul_bt(da_hh_rz, w_hh, d_h_next, batch, gate_dim, hidden);
        for (size_t i = 0; i < batch * hidden; i++) {
            d_h_next[i] += d_h_rec[i];
        }

        /* grad_W_ih += x_t^T @ d_a_ih ; grad_W_hh: r/z blocks via h_prev,
           n block via the masked state (r .* h_prev) */
        gru_matmul_at(x + t * batch * input_size, d_a_ih, acc, batch, input_size, gate_dim);
        for (size_t i = 0; i < input_size * gate_dim; i++) {
            gw_ih[i] += acc[i];
        }
        if (h_prev_t != NULL) {
            gru_matmul_at(h_prev_t, da_hh_rz, acc, batch, hidden, gate_dim);
            for (size_t i = 0; i < hidden * gate_dim; i++) {
                gw_hh[i] += acc[i];
            }
            for (size_t p = 0; p < hidden; p++) {
                for (size_t j = 0; j < hidden; j++) {
                    float s = 0.0f;
                    for (size_t b = 0; b < batch; b++)
                        s += hm[b * hidden + p] * d_a_n[b * hidden + j];
                    gw_hh[p * gate_dim + 2 * hidden + j] += s;
                }
            }
        }

        /* Both biases receive sum_b d_a_ih (b_ih in a_ih/combined; b_hh in
           combined and in the candidate's explicit b_hh[n] term). */
        memset(acc_bias, 0, gate_dim * sizeof(float));
        for (size_t b = 0; b < batch; b++) {
            const float* da = d_a_ih + b * gate_dim;
            for (size_t j = 0; j < gate_dim; j++) {
                acc_bias[j] += da[j];
            }
        }
        for (size_t j = 0; j < gate_dim; j++) {
            gb_ih[j] += acc_bias[j];
            gb_hh[j] += acc_bias[j];
        }
    }

    boat_free(d_a_ih);
    boat_free(d_a_hh);
    boat_free(d_x_t);
    boat_free(d_h);
    boat_free(d_h_rec);
    boat_free(d_h_next);
    boat_free(d_a_n);
    boat_free(hm);
    boat_free(da_hh_rz);
    boat_free(rv2);
    boat_free(zv2);
    boat_free(acc);
    boat_free(acc_bias);

    return grad_input;
}

// Update parameters
BOAT_API void BOAT_CALL boat_gru_layer_update(boat_gru_layer_t* layer, float learning_rate) {
    if (!layer) return;
    gru_update_tensor(layer->weight_ih, layer->grad_weight_ih, learning_rate);
    gru_update_tensor(layer->weight_hh, layer->grad_weight_hh, learning_rate);
    gru_update_tensor(layer->bias_ih, layer->grad_bias_ih, learning_rate);
    gru_update_tensor(layer->bias_hh, layer->grad_bias_hh, learning_rate);
}

// Parameter setters for model loading
BOAT_API void BOAT_CALL boat_gru_layer_set_weight_ih(boat_gru_layer_t* layer,
                                                     boat_tensor_t* weight) {
    if (layer->weight_ih) boat_tensor_unref(layer->weight_ih);
    layer->weight_ih = weight;
    if (weight) boat_tensor_ref(weight);
}

BOAT_API void BOAT_CALL boat_gru_layer_set_weight_hh(boat_gru_layer_t* layer,
                                                     boat_tensor_t* weight) {
    if (layer->weight_hh) boat_tensor_unref(layer->weight_hh);
    layer->weight_hh = weight;
    if (weight) boat_tensor_ref(weight);
}

BOAT_API void BOAT_CALL boat_gru_layer_set_bias_ih(boat_gru_layer_t* layer, boat_tensor_t* bias) {
    if (layer->bias_ih) boat_tensor_unref(layer->bias_ih);
    layer->bias_ih = bias;
    if (bias) boat_tensor_ref(bias);
}

BOAT_API void BOAT_CALL boat_gru_layer_set_bias_hh(boat_gru_layer_t* layer, boat_tensor_t* bias) {
    if (layer->bias_hh) boat_tensor_unref(layer->bias_hh);
    layer->bias_hh = bias;
    if (bias) boat_tensor_ref(bias);
}

// Parameter getters
BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_get_weight_ih(const boat_gru_layer_t* layer) {
    if (!layer) return NULL;
    return layer->weight_ih;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_get_weight_hh(const boat_gru_layer_t* layer) {
    if (!layer) return NULL;
    return layer->weight_hh;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_get_bias_ih(const boat_gru_layer_t* layer) {
    if (!layer) return NULL;
    return layer->bias_ih;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_get_bias_hh(const boat_gru_layer_t* layer) {
    if (!layer) return NULL;
    return layer->bias_hh;
}

// Gradient tensor getters
BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_get_grad_weight_ih(const boat_gru_layer_t* layer) {
    if (!layer) return NULL;
    return layer->grad_weight_ih;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_get_grad_weight_hh(const boat_gru_layer_t* layer) {
    if (!layer) return NULL;
    return layer->grad_weight_hh;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_get_grad_bias_ih(const boat_gru_layer_t* layer) {
    if (!layer) return NULL;
    return layer->grad_bias_ih;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_get_grad_bias_hh(const boat_gru_layer_t* layer) {
    if (!layer) return NULL;
    return layer->grad_bias_hh;
}
