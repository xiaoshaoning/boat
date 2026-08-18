// test_lstm_gru.c - Forward, backward (numerical gradient) and update tests for LSTM/GRU
// Copyright (c) 2026 Shaoning, Xiao
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_failures = 0;

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            printf("  FAIL: %s\n", msg);                                                           \
            g_failures++;                                                                          \
        } else {                                                                                   \
            printf("  OK: %s\n", msg);                                                             \
        }                                                                                          \
    } while (0)

#define ALLOC_F32(n) ((float*)malloc((n) * sizeof(float)))

static float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ---------- reference forward implementations ----------

// LSTM reference. Weights use [in, out] convention;
// gate order is [input, forget, cell, output] over 4*hidden.
static void ref_lstm_forward(const float* x, const float* w_ih, const float* w_hh,
                             const float* b_ih, const float* b_hh, size_t batch, size_t T,
                             size_t in, size_t h, float* out) {
    size_t gdim = 4 * h;
    float* h_prev = (float*)calloc(batch * h, sizeof(float));
    float* c_prev = (float*)calloc(batch * h, sizeof(float));
    float* h_curr = ALLOC_F32(batch * h);
    float* c_curr = ALLOC_F32(batch * h);
    if (!h_prev || !c_prev || !h_curr || !c_curr) {
        free(h_prev);
        free(c_prev);
        free(h_curr);
        free(c_curr);
        return;
    }

    for (size_t t = 0; t < T; t++) {
        for (size_t b = 0; b < batch; b++) {
            const float* xt = x + (t * batch + b) * in;
            const float* hp = h_prev + b * h;
            const float* cp = c_prev + b * h;
            float* hb = h_curr + b * h;
            float* cb = c_curr + b * h;
            for (size_t j = 0; j < h; j++) {
                float ig = b_ih[j] + b_hh[j];
                float fg = b_ih[h + j] + b_hh[h + j];
                float cg = b_ih[2 * h + j] + b_hh[2 * h + j];
                float og = b_ih[3 * h + j] + b_hh[3 * h + j];
                for (size_t p = 0; p < in; p++) {
                    ig += xt[p] * w_ih[p * gdim + j];
                    fg += xt[p] * w_ih[p * gdim + h + j];
                    cg += xt[p] * w_ih[p * gdim + 2 * h + j];
                    og += xt[p] * w_ih[p * gdim + 3 * h + j];
                }
                for (size_t p = 0; p < h; p++) {
                    ig += hp[p] * w_hh[p * gdim + j];
                    fg += hp[p] * w_hh[p * gdim + h + j];
                    cg += hp[p] * w_hh[p * gdim + 2 * h + j];
                    og += hp[p] * w_hh[p * gdim + 3 * h + j];
                }
                float c_new = sigmoidf(fg) * cp[j] + sigmoidf(ig) * tanhf(cg);
                cb[j] = c_new;
                hb[j] = sigmoidf(og) * tanhf(c_new);
            }
        }
        memcpy(out + t * batch * h, h_curr, batch * h * sizeof(float));
        memcpy(h_prev, h_curr, batch * h * sizeof(float));
        memcpy(c_prev, c_curr, batch * h * sizeof(float));
    }

    free(h_prev);
    free(c_prev);
    free(h_curr);
    free(c_curr);
}

// GRU reference. Gate order is [reset, update, new] over 3*hidden. The
// candidate applies the reset to h_prev BEFORE the recurrent matmul, and
// the update is h = (1 - z)*h_prev + z*n (MATLAB/PyTorch convention).
static void ref_gru_forward(const float* x, const float* w_ih, const float* w_hh, const float* b_ih,
                            const float* b_hh, size_t batch, size_t T, size_t in, size_t h,
                            float* out) {
    size_t gdim = 3 * h;
    float* h_prev = (float*)calloc(batch * h, sizeof(float));
    float* h_curr = ALLOC_F32(batch * h);
    if (!h_prev || !h_curr) {
        free(h_prev);
        free(h_curr);
        return;
    }

    for (size_t t = 0; t < T; t++) {
        for (size_t b = 0; b < batch; b++) {
            const float* xt = x + (t * batch + b) * in;
            const float* hp = h_prev + b * h;
            float* hb = h_curr + b * h;
            float* rv = ALLOC_F32(h);
            float* zv = ALLOC_F32(h);
            for (size_t j = 0; j < h; j++) {
                float a_r = b_ih[j] + b_hh[j];
                float a_z = b_ih[h + j] + b_hh[h + j];
                for (size_t p = 0; p < in; p++) {
                    a_r += xt[p] * w_ih[p * gdim + j];
                    a_z += xt[p] * w_ih[p * gdim + h + j];
                }
                for (size_t p = 0; p < h; p++) {
                    a_r += hp[p] * w_hh[p * gdim + j];
                    a_z += hp[p] * w_hh[p * gdim + h + j];
                }
                rv[j] = sigmoidf(a_r);
                zv[j] = sigmoidf(a_z);
            }
            for (size_t j = 0; j < h; j++) {
                float a_n = b_ih[2 * h + j];
                for (size_t p = 0; p < in; p++)
                    a_n += xt[p] * w_ih[p * gdim + 2 * h + j];
                for (size_t p = 0; p < h; p++)
                    a_n += (rv[p] * hp[p]) * w_hh[p * gdim + 2 * h + j];
                float n = tanhf(a_n + b_hh[2 * h + j]);
                hb[j] = (1.0f - zv[j]) * hp[j] + zv[j] * n;
            }
            free(rv);
            free(zv);
        }
        memcpy(out + t * batch * h, h_curr, batch * h * sizeof(float));
        memcpy(h_prev, h_curr, batch * h * sizeof(float));
    }

    free(h_prev);
    free(h_curr);
}

// ---------- helpers ----------

static boat_tensor_t* make_tensor(const int64_t* shape, size_t ndim, float* data, size_t n) {
    boat_tensor_t* t = boat_tensor_create(shape, ndim, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!t) return NULL;
    memcpy(boat_tensor_data(t), data, n * sizeof(float));
    return t;
}

static void fill_random(float* data, size_t n, float scale, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

static float run_lstm_loss(boat_lstm_layer_t* layer, boat_tensor_t* input) {
    boat_tensor_t* out = boat_lstm_layer_forward(layer, input);
    if (!out) return NAN;
    const float* d = (const float*)boat_tensor_const_data(out);
    size_t n = boat_tensor_nelements(out);
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++)
        sum += d[i];
    boat_tensor_free(out);
    return sum;
}

static float run_gru_loss(boat_gru_layer_t* layer, boat_tensor_t* input) {
    boat_tensor_t* out = boat_gru_layer_forward(layer, input);
    if (!out) return NAN;
    const float* d = (const float*)boat_tensor_const_data(out);
    size_t n = boat_tensor_nelements(out);
    /* accumulate in double: float32 summing of 30+ outputs leaves the
       finite-difference numerator (~2e-6) inside the rounding error */
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
        sum += (double) d[i];
    boat_tensor_free(out);
    return (float) sum;
}

// Central finite difference of a scalar loss w.r.t. one element of `param`.
static float numerical_grad(float (*loss_fn)(void* ctx), void* ctx, boat_tensor_t* param,
                            size_t idx, float eps) {
    float* data = (float*)boat_tensor_data(param);
    float orig = data[idx];
    data[idx] = orig + eps;
    float plus = loss_fn(ctx);
    data[idx] = orig - eps;
    float minus = loss_fn(ctx);
    data[idx] = orig;
    return (plus - minus) / (2.0f * eps);
}

static int compare_grads(const char* name, const float* analytic, const float* numeric, size_t n,
                         float atol, float rtol) {
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(analytic[i] - numeric[i]);
        if (diff <= atol) continue;
        float scale = fabsf(analytic[i]) + fabsf(numeric[i]);
        if (scale > 0.0f && diff / scale <= rtol) continue;
        bad++;
        if (bad <= 5) {
            printf("    %s[%zu] analytic=%.6f numeric=%.6f diff=%.6f\n", name, i, analytic[i],
                   numeric[i], diff);
        }
    }
    return bad;
}

static int check_tensor_close(const char* name, const float* a, const float* b, size_t n,
                              float atol, float rtol) {
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff <= atol) continue;
        float scale = fabsf(a[i]) + fabsf(b[i]);
        if (scale > 0.0f && diff / scale <= rtol) continue;
        bad++;
        if (bad <= 3) {
            printf("    %s[%zu] got=%.6f expected=%.6f diff=%.6f\n", name, i, a[i], b[i], diff);
        }
    }
    return bad;
}

// ---------- LSTM tests ----------

static int test_lstm_forward_reference(void) {
    printf("LSTM forward matches reference\n");
    const size_t batch = 2, T = 3, in = 4, h = 5;
    const size_t gdim = 4 * h;
    int bad = 0;

    float* w_ih = ALLOC_F32(in * gdim);
    float* w_hh = ALLOC_F32(h * gdim);
    float* b_ih = ALLOC_F32(gdim);
    float* b_hh = ALLOC_F32(gdim);
    float* x = ALLOC_F32(batch * T * in);
    float* ref = ALLOC_F32(batch * T * h);
    if (!w_ih || !w_hh || !b_ih || !b_hh || !x || !ref) {
        printf("  FAIL: allocation failed\n");
        g_failures++;
        goto done;
    }
    fill_random(w_ih, in * gdim, 0.2f, 11);
    fill_random(w_hh, h * gdim, 0.2f, 12);
    fill_random(b_ih, gdim, 0.1f, 13);
    fill_random(b_hh, gdim, 0.1f, 14);
    fill_random(x, batch * T * in, 0.3f, 15);

    boat_lstm_layer_t* layer = boat_lstm_layer_create(in, h, 1, false, 0.0f);
    int64_t shp_ih[] = {(int64_t)in, (int64_t)gdim};
    int64_t shp_hh[] = {(int64_t)h, (int64_t)gdim};
    int64_t shp_b[] = {(int64_t)gdim};
    boat_tensor_t* t_wih = make_tensor(shp_ih, 2, w_ih, in * gdim);
    boat_tensor_t* t_whh = make_tensor(shp_hh, 2, w_hh, h * gdim);
    boat_tensor_t* t_bih = make_tensor(shp_b, 1, b_ih, gdim);
    boat_tensor_t* t_bhh = make_tensor(shp_b, 1, b_hh, gdim);
    boat_lstm_layer_set_weight_ih(layer, t_wih);
    boat_lstm_layer_set_weight_hh(layer, t_whh);
    boat_lstm_layer_set_bias_ih(layer, t_bih);
    boat_lstm_layer_set_bias_hh(layer, t_bhh);
    boat_tensor_unref(t_wih);
    boat_tensor_unref(t_whh);
    boat_tensor_unref(t_bih);
    boat_tensor_unref(t_bhh);

    int64_t shp_x[] = {(int64_t)batch, (int64_t)T, (int64_t)in};
    boat_tensor_t* input = make_tensor(shp_x, 3, x, batch * T * in);

    boat_tensor_t* out = boat_lstm_layer_forward(layer, input);
    CHECK(out != NULL, "LSTM forward returns output");
    if (out) {
        const int64_t* oshape = boat_tensor_shape(out);
        if (boat_tensor_ndim(out) == 3 && oshape[0] == (int64_t)batch && oshape[1] == (int64_t)T &&
            oshape[2] == (int64_t)h) {
            printf("  OK: output shape [%lld,%lld,%lld]\n", (long long)oshape[0],
                   (long long)oshape[1], (long long)oshape[2]);
        } else {
            printf("  FAIL: unexpected output shape\n");
            g_failures++;
        }
        ref_lstm_forward(x, w_ih, w_hh, b_ih, b_hh, batch, T, in, h, ref);
        bad = check_tensor_close("lstm_out", (const float*)boat_tensor_const_data(out), ref,
                                 batch * T * h, 1e-4f, 1e-3f);
        if (bad == 0)
            printf("  OK: forward values match reference\n");
        else {
            printf("  FAIL: %d output mismatches vs reference\n", bad);
            g_failures++;
        }
        boat_tensor_free(out);
    }

    boat_tensor_free(input);
    boat_lstm_layer_free(layer);
done:
    free(w_ih);
    free(w_hh);
    free(b_ih);
    free(b_hh);
    free(x);
    free(ref);
    return bad;
}

typedef struct {
    boat_lstm_layer_t* layer;
    boat_tensor_t* input;
} lstm_ctx_t;

static float lstm_loss_cb(void* ctx) {
    lstm_ctx_t* c = (lstm_ctx_t*)ctx;
    return run_lstm_loss(c->layer, c->input);
}

static int test_lstm_backward_gradients(void) {
    printf("LSTM backward numerical gradient check\n");
    const size_t batch = 2, T = 3, in = 4, h = 5;
    const size_t gdim = 4 * h;
    int bad = 0;

    float* w_ih = ALLOC_F32(in * gdim);
    float* w_hh = ALLOC_F32(h * gdim);
    float* b_ih = ALLOC_F32(gdim);
    float* b_hh = ALLOC_F32(gdim);
    float* x = ALLOC_F32(batch * T * in);
    float* num_wih = ALLOC_F32(in * gdim);
    float* num_whh = ALLOC_F32(h * gdim);
    float* num_bih = ALLOC_F32(gdim);
    float* num_bhh = ALLOC_F32(gdim);
    float* num_x = ALLOC_F32(batch * T * in);
    if (!w_ih || !w_hh || !b_ih || !b_hh || !x || !num_wih || !num_whh || !num_bih || !num_bhh ||
        !num_x) {
        printf("  FAIL: allocation failed\n");
        g_failures++;
        goto done;
    }
    fill_random(w_ih, in * gdim, 0.2f, 21);
    fill_random(w_hh, h * gdim, 0.2f, 22);
    fill_random(b_ih, gdim, 0.1f, 23);
    fill_random(b_hh, gdim, 0.1f, 24);
    fill_random(x, batch * T * in, 0.3f, 25);

    boat_lstm_layer_t* layer = boat_lstm_layer_create(in, h, 1, false, 0.0f);
    int64_t shp_ih[] = {(int64_t)in, (int64_t)gdim};
    int64_t shp_hh[] = {(int64_t)h, (int64_t)gdim};
    int64_t shp_b[] = {(int64_t)gdim};
    boat_tensor_t* t_wih = make_tensor(shp_ih, 2, w_ih, in * gdim);
    boat_tensor_t* t_whh = make_tensor(shp_hh, 2, w_hh, h * gdim);
    boat_tensor_t* t_bih = make_tensor(shp_b, 1, b_ih, gdim);
    boat_tensor_t* t_bhh = make_tensor(shp_b, 1, b_hh, gdim);
    boat_lstm_layer_set_weight_ih(layer, t_wih);
    boat_lstm_layer_set_weight_hh(layer, t_whh);
    boat_lstm_layer_set_bias_ih(layer, t_bih);
    boat_lstm_layer_set_bias_hh(layer, t_bhh);
    boat_tensor_unref(t_wih);
    boat_tensor_unref(t_whh);
    boat_tensor_unref(t_bih);
    boat_tensor_unref(t_bhh);

    int64_t shp_x[] = {(int64_t)batch, (int64_t)T, (int64_t)in};
    boat_tensor_t* input = make_tensor(shp_x, 3, x, batch * T * in);

    lstm_ctx_t ctx = {layer, input};
    const float eps = 1e-3f, atol = 1e-3f, rtol = 1e-2f;

    for (size_t i = 0; i < in * gdim; i++)
        num_wih[i] = numerical_grad(lstm_loss_cb, &ctx, t_wih, i, eps);
    for (size_t i = 0; i < h * gdim; i++)
        num_whh[i] = numerical_grad(lstm_loss_cb, &ctx, t_whh, i, eps);
    for (size_t i = 0; i < gdim; i++)
        num_bih[i] = numerical_grad(lstm_loss_cb, &ctx, t_bih, i, eps);
    for (size_t i = 0; i < gdim; i++)
        num_bhh[i] = numerical_grad(lstm_loss_cb, &ctx, t_bhh, i, eps);
    for (size_t i = 0; i < batch * T * in; i++)
        num_x[i] = numerical_grad(lstm_loss_cb, &ctx, input, i, eps);

    // Analytical gradients (single forward + backward with ones)
    boat_tensor_t* out = boat_lstm_layer_forward(layer, input);
    int64_t shp_go[] = {(int64_t)batch, (int64_t)T, (int64_t)h};
    boat_tensor_t* grad_out = boat_tensor_create(shp_go, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* god = (float*)boat_tensor_data(grad_out);
    for (size_t i = 0; i < batch * T * h; i++)
        god[i] = 1.0f;
    boat_tensor_free(out);

    boat_tensor_t* grad_in = boat_lstm_layer_backward(layer, grad_out);
    CHECK(grad_in != NULL, "LSTM backward returns grad_input");
    if (!grad_in) {
        g_failures++;
        boat_tensor_free(grad_out);
        goto done;
    }
    CHECK(boat_tensor_nelements(grad_in) == batch * T * in, "grad_input shape matches input");

    const float* gwih =
        (const float*)boat_tensor_const_data(boat_lstm_layer_get_grad_weight_ih(layer));
    const float* gwhh =
        (const float*)boat_tensor_const_data(boat_lstm_layer_get_grad_weight_hh(layer));
    const float* gbih =
        (const float*)boat_tensor_const_data(boat_lstm_layer_get_grad_bias_ih(layer));
    const float* gbhh =
        (const float*)boat_tensor_const_data(boat_lstm_layer_get_grad_bias_hh(layer));
    const float* gx = (const float*)boat_tensor_const_data(grad_in);

    bad = compare_grads("w_ih", gwih, num_wih, in * gdim, atol, rtol);
    printf("  w_ih gradient mismatches: %d\n", bad);
    g_failures += bad;
    bad = compare_grads("w_hh", gwhh, num_whh, h * gdim, atol, rtol);
    printf("  w_hh gradient mismatches: %d\n", bad);
    g_failures += bad;
    bad = compare_grads("b_ih", gbih, num_bih, gdim, atol, rtol);
    printf("  b_ih gradient mismatches: %d\n", bad);
    g_failures += bad;
    bad = compare_grads("b_hh", gbhh, num_bhh, gdim, atol, rtol);
    printf("  b_hh gradient mismatches: %d\n", bad);
    g_failures += bad;
    bad = compare_grads("x", gx, num_x, batch * T * in, atol, rtol);
    printf("  input gradient mismatches: %d\n", bad);
    g_failures += bad;

    boat_tensor_free(grad_out);
    boat_tensor_free(grad_in);
    boat_tensor_free(input);
    boat_lstm_layer_free(layer);
done:
    free(w_ih);
    free(w_hh);
    free(b_ih);
    free(b_hh);
    free(x);
    free(num_wih);
    free(num_whh);
    free(num_bih);
    free(num_bhh);
    free(num_x);
    return bad;
}

static int test_lstm_update(void) {
    printf("LSTM update applies SGD and zeroes gradients\n");
    const size_t in = 3, h = 4;
    const size_t gdim = 4 * h;
    float* w_ih = ALLOC_F32(in * gdim);
    float* w_hh = ALLOC_F32(h * gdim);
    float* b_ih = ALLOC_F32(gdim);
    float* b_hh = ALLOC_F32(gdim);
    float* x = ALLOC_F32(1 * 2 * in);
    float* w_ih_copy = ALLOC_F32(in * gdim);
    float* b_ih_copy = ALLOC_F32(gdim);
    float* gwih_snap = ALLOC_F32(in * gdim);
    float* gbih_snap = ALLOC_F32(gdim);
    int bad = 0;
    if (!w_ih || !w_hh || !b_ih || !b_hh || !x || !w_ih_copy || !b_ih_copy || !gwih_snap ||
        !gbih_snap) {
        printf("  FAIL: allocation failed\n");
        g_failures++;
        goto done;
    }
    fill_random(w_ih, in * gdim, 0.2f, 31);
    fill_random(w_hh, h * gdim, 0.2f, 32);
    fill_random(b_ih, gdim, 0.1f, 33);
    fill_random(b_hh, gdim, 0.1f, 34);
    memcpy(w_ih_copy, w_ih, in * gdim * sizeof(float));
    memcpy(b_ih_copy, b_ih, gdim * sizeof(float));

    boat_lstm_layer_t* layer = boat_lstm_layer_create(in, h, 1, false, 0.0f);
    int64_t shp_ih[] = {(int64_t)in, (int64_t)gdim};
    int64_t shp_hh[] = {(int64_t)h, (int64_t)gdim};
    int64_t shp_b[] = {(int64_t)gdim};
    boat_tensor_t* t_wih = make_tensor(shp_ih, 2, w_ih, in * gdim);
    boat_tensor_t* t_whh = make_tensor(shp_hh, 2, w_hh, h * gdim);
    boat_tensor_t* t_bih = make_tensor(shp_b, 1, b_ih, gdim);
    boat_tensor_t* t_bhh = make_tensor(shp_b, 1, b_hh, gdim);
    boat_lstm_layer_set_weight_ih(layer, t_wih);
    boat_lstm_layer_set_weight_hh(layer, t_whh);
    boat_lstm_layer_set_bias_ih(layer, t_bih);
    boat_lstm_layer_set_bias_hh(layer, t_bhh);
    // Keep local references so we can read the updated weights after update()

    int64_t shp_x[] = {1, 2, (int64_t)in};
    fill_random(x, 1 * 2 * in, 0.3f, 35);
    boat_tensor_t* input = make_tensor(shp_x, 3, x, 1 * 2 * in);

    boat_tensor_t* out = NULL;
    boat_tensor_t* grad_out = NULL;
    boat_tensor_t* grad_in = NULL;
    out = boat_lstm_layer_forward(layer, input);
    int64_t shp_go[] = {1, 2, (int64_t)h};
    grad_out = boat_tensor_create(shp_go, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* god = (float*)boat_tensor_data(grad_out);
    for (size_t i = 0; i < 1 * 2 * h; i++)
        god[i] = 0.5f;
    grad_in = boat_lstm_layer_backward(layer, grad_out);
    CHECK(grad_in != NULL, "backward before update");
    if (!grad_in) {
        g_failures++;
        goto cleanup;
    }
    float lr = 0.1f;
    float* gwih = (float*)boat_tensor_data(boat_lstm_layer_get_grad_weight_ih(layer));
    float* gbih = (float*)boat_tensor_data(boat_lstm_layer_get_grad_bias_ih(layer));
    memcpy(gwih_snap, gwih, in * gdim * sizeof(float));
    memcpy(gbih_snap, gbih, gdim * sizeof(float));

    boat_lstm_layer_update(layer, lr);

    const float* wih_after = (const float*)boat_tensor_const_data(t_wih);
    const float* bih_after = (const float*)boat_tensor_const_data(t_bih);
    for (size_t i = 0; i < in * gdim; i++) {
        if (fabsf(wih_after[i] - (w_ih_copy[i] - lr * gwih_snap[i])) > 1e-4f) bad++;
    }
    for (size_t i = 0; i < gdim; i++) {
        if (fabsf(bih_after[i] - (b_ih_copy[i] - lr * gbih_snap[i])) > 1e-4f) bad++;
    }
    CHECK(bad == 0, "weights updated by -lr * grad");

    int nonzero = 0;
    for (size_t i = 0; i < in * gdim; i++) {
        if (gwih[i] != 0.0f) nonzero++;
    }
    for (size_t i = 0; i < gdim; i++) {
        if (gbih[i] != 0.0f) nonzero++;
    }
    CHECK(nonzero == 0, "gradients zeroed after update");

cleanup:
    boat_tensor_free(grad_out);
    if (grad_in) boat_tensor_free(grad_in);
    if (out) boat_tensor_free(out);
    boat_tensor_free(input);
    boat_tensor_free(t_wih);
    boat_tensor_free(t_whh);
    boat_tensor_free(t_bih);
    boat_tensor_free(t_bhh);
    boat_lstm_layer_free(layer);
done:
    free(w_ih);
    free(w_hh);
    free(b_ih);
    free(b_hh);
    free(x);
    free(w_ih_copy);
    free(b_ih_copy);
    free(gwih_snap);
    free(gbih_snap);
    return bad;
}

// ---------- GRU tests ----------

static int test_gru_forward_reference(void) {
    printf("GRU forward matches reference\n");
    const size_t batch = 2, T = 3, in = 4, h = 5;
    const size_t gdim = 3 * h;
    int bad = 0;

    float* w_ih = ALLOC_F32(in * gdim);
    float* w_hh = ALLOC_F32(h * gdim);
    float* b_ih = ALLOC_F32(gdim);
    float* b_hh = ALLOC_F32(gdim);
    float* x = ALLOC_F32(batch * T * in);
    float* ref = ALLOC_F32(batch * T * h);
    if (!w_ih || !w_hh || !b_ih || !b_hh || !x || !ref) {
        printf("  FAIL: allocation failed\n");
        g_failures++;
        goto done;
    }
    fill_random(w_ih, in * gdim, 0.2f, 41);
    fill_random(w_hh, h * gdim, 0.2f, 42);
    fill_random(b_ih, gdim, 0.1f, 43);
    fill_random(b_hh, gdim, 0.1f, 44);
    fill_random(x, batch * T * in, 0.3f, 45);

    boat_gru_layer_t* layer = boat_gru_layer_create(in, h, 1, false, 0.0f);
    int64_t shp_ih[] = {(int64_t)in, (int64_t)gdim};
    int64_t shp_hh[] = {(int64_t)h, (int64_t)gdim};
    int64_t shp_b[] = {(int64_t)gdim};
    boat_tensor_t* t_wih = make_tensor(shp_ih, 2, w_ih, in * gdim);
    boat_tensor_t* t_whh = make_tensor(shp_hh, 2, w_hh, h * gdim);
    boat_tensor_t* t_bih = make_tensor(shp_b, 1, b_ih, gdim);
    boat_tensor_t* t_bhh = make_tensor(shp_b, 1, b_hh, gdim);
    boat_gru_layer_set_weight_ih(layer, t_wih);
    boat_gru_layer_set_weight_hh(layer, t_whh);
    boat_gru_layer_set_bias_ih(layer, t_bih);
    boat_gru_layer_set_bias_hh(layer, t_bhh);
    boat_tensor_unref(t_wih);
    boat_tensor_unref(t_whh);
    boat_tensor_unref(t_bih);
    boat_tensor_unref(t_bhh);

    int64_t shp_x[] = {(int64_t)batch, (int64_t)T, (int64_t)in};
    boat_tensor_t* input = make_tensor(shp_x, 3, x, batch * T * in);

    boat_tensor_t* out = boat_gru_layer_forward(layer, input);
    CHECK(out != NULL, "GRU forward returns output");
    if (out) {
        const int64_t* oshape = boat_tensor_shape(out);
        if (boat_tensor_ndim(out) == 3 && oshape[0] == (int64_t)batch && oshape[1] == (int64_t)T &&
            oshape[2] == (int64_t)h) {
            printf("  OK: output shape [%lld,%lld,%lld]\n", (long long)oshape[0],
                   (long long)oshape[1], (long long)oshape[2]);
        } else {
            printf("  FAIL: unexpected output shape\n");
            g_failures++;
        }
        ref_gru_forward(x, w_ih, w_hh, b_ih, b_hh, batch, T, in, h, ref);
        bad = check_tensor_close("gru_out", (const float*)boat_tensor_const_data(out), ref,
                                 batch * T * h, 1e-4f, 1e-3f);
        if (bad == 0)
            printf("  OK: forward values match reference\n");
        else {
            printf("  FAIL: %d output mismatches vs reference\n", bad);
            g_failures++;
        }
        boat_tensor_free(out);
    }

    boat_tensor_free(input);
    boat_gru_layer_free(layer);
done:
    free(w_ih);
    free(w_hh);
    free(b_ih);
    free(b_hh);
    free(x);
    free(ref);
    return bad;
}

typedef struct {
    boat_gru_layer_t* layer;
    boat_tensor_t* input;
} gru_ctx_t;

static float gru_loss_cb(void* ctx) {
    gru_ctx_t* c = (gru_ctx_t*)ctx;
    return run_gru_loss(c->layer, c->input);
}

static int test_gru_backward_gradients(void) {
    printf("GRU backward numerical gradient check\n");
    const size_t batch = 2, T = 3, in = 4, h = 5;
    const size_t gdim = 3 * h;
    int bad = 0;

    float* w_ih = ALLOC_F32(in * gdim);
    float* w_hh = ALLOC_F32(h * gdim);
    float* b_ih = ALLOC_F32(gdim);
    float* b_hh = ALLOC_F32(gdim);
    float* x = ALLOC_F32(batch * T * in);
    float* num_wih = ALLOC_F32(in * gdim);
    float* num_whh = ALLOC_F32(h * gdim);
    float* num_bih = ALLOC_F32(gdim);
    float* num_bhh = ALLOC_F32(gdim);
    float* num_x = ALLOC_F32(batch * T * in);
    if (!w_ih || !w_hh || !b_ih || !b_hh || !x || !num_wih || !num_whh || !num_bih || !num_bhh ||
        !num_x) {
        printf("  FAIL: allocation failed\n");
        g_failures++;
        goto done;
    }
    fill_random(w_ih, in * gdim, 0.2f, 51);
    fill_random(w_hh, h * gdim, 0.2f, 52);
    fill_random(b_ih, gdim, 0.1f, 53);
    fill_random(b_hh, gdim, 0.1f, 54);
    fill_random(x, batch * T * in, 0.3f, 55);

    boat_gru_layer_t* layer = boat_gru_layer_create(in, h, 1, false, 0.0f);
    int64_t shp_ih[] = {(int64_t)in, (int64_t)gdim};
    int64_t shp_hh[] = {(int64_t)h, (int64_t)gdim};
    int64_t shp_b[] = {(int64_t)gdim};
    boat_tensor_t* t_wih = make_tensor(shp_ih, 2, w_ih, in * gdim);
    boat_tensor_t* t_whh = make_tensor(shp_hh, 2, w_hh, h * gdim);
    boat_tensor_t* t_bih = make_tensor(shp_b, 1, b_ih, gdim);
    boat_tensor_t* t_bhh = make_tensor(shp_b, 1, b_hh, gdim);
    boat_gru_layer_set_weight_ih(layer, t_wih);
    boat_gru_layer_set_weight_hh(layer, t_whh);
    boat_gru_layer_set_bias_ih(layer, t_bih);
    boat_gru_layer_set_bias_hh(layer, t_bhh);
    boat_tensor_unref(t_wih);
    boat_tensor_unref(t_whh);
    boat_tensor_unref(t_bih);
    boat_tensor_unref(t_bhh);

    int64_t shp_x[] = {(int64_t)batch, (int64_t)T, (int64_t)in};
    boat_tensor_t* input = make_tensor(shp_x, 3, x, batch * T * in);

    gru_ctx_t ctx = {layer, input};
    const float eps = 1e-3f, atol = 2e-3f, rtol = 2e-2f;

    for (size_t i = 0; i < in * gdim; i++)
        num_wih[i] = numerical_grad(gru_loss_cb, &ctx, t_wih, i, eps);
    for (size_t i = 0; i < h * gdim; i++)
        num_whh[i] = numerical_grad(gru_loss_cb, &ctx, t_whh, i, eps);
    for (size_t i = 0; i < gdim; i++)
        num_bih[i] = numerical_grad(gru_loss_cb, &ctx, t_bih, i, eps);
    for (size_t i = 0; i < gdim; i++)
        num_bhh[i] = numerical_grad(gru_loss_cb, &ctx, t_bhh, i, eps);
    for (size_t i = 0; i < batch * T * in; i++)
        num_x[i] = numerical_grad(gru_loss_cb, &ctx, input, i, eps);

    boat_tensor_t* out = boat_gru_layer_forward(layer, input);
    int64_t shp_go[] = {(int64_t)batch, (int64_t)T, (int64_t)h};
    boat_tensor_t* grad_out = boat_tensor_create(shp_go, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* god = (float*)boat_tensor_data(grad_out);
    for (size_t i = 0; i < batch * T * h; i++)
        god[i] = 1.0f;
    boat_tensor_free(out);

    boat_tensor_t* grad_in = boat_gru_layer_backward(layer, grad_out);
    CHECK(grad_in != NULL, "GRU backward returns grad_input");
    if (!grad_in) {
        g_failures++;
        boat_tensor_free(grad_out);
        goto done;
    }
    CHECK(boat_tensor_nelements(grad_in) == batch * T * in, "grad_input shape matches input");

    const float* gwih =
        (const float*)boat_tensor_const_data(boat_gru_layer_get_grad_weight_ih(layer));
    const float* gwhh =
        (const float*)boat_tensor_const_data(boat_gru_layer_get_grad_weight_hh(layer));
    const float* gbih =
        (const float*)boat_tensor_const_data(boat_gru_layer_get_grad_bias_ih(layer));
    const float* gbhh =
        (const float*)boat_tensor_const_data(boat_gru_layer_get_grad_bias_hh(layer));
    const float* gx = (const float*)boat_tensor_const_data(grad_in);

    bad = compare_grads("w_ih", gwih, num_wih, in * gdim, atol, rtol);
    printf("  w_ih gradient mismatches: %d\n", bad);
    g_failures += bad;
    bad = compare_grads("w_hh", gwhh, num_whh, h * gdim, atol, rtol);
    printf("  w_hh gradient mismatches: %d\n", bad);
    g_failures += bad;
    bad = compare_grads("b_ih", gbih, num_bih, gdim, atol, rtol);
    printf("  b_ih gradient mismatches: %d\n", bad);
    g_failures += bad;
    bad = compare_grads("b_hh", gbhh, num_bhh, gdim, atol, rtol);
    printf("  b_hh gradient mismatches: %d\n", bad);
    g_failures += bad;
    bad = compare_grads("x", gx, num_x, batch * T * in, atol, rtol);
    printf("  input gradient mismatches: %d\n", bad);
    g_failures += bad;

    boat_tensor_free(grad_out);
    boat_tensor_free(grad_in);
    boat_tensor_free(input);
    boat_gru_layer_free(layer);
done:
    free(w_ih);
    free(w_hh);
    free(b_ih);
    free(b_hh);
    free(x);
    free(num_wih);
    free(num_whh);
    free(num_bih);
    free(num_bhh);
    free(num_x);
    return bad;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== LSTM/GRU layer tests ===\n");
    test_lstm_forward_reference();
    test_lstm_backward_gradients();
    test_lstm_update();
    test_gru_forward_reference();
    test_gru_backward_gradients();
    printf("\n%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures > 0 ? 1 : 0;
}
