// test_simd.c - SIMD kernels (transpose2d, reductions) and their wiring into
// boat_transpose / boat_sum|mean|max|min / boat_conv_layer_forward.
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/simd.h>
#include <boat/tensor.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned g_seed = 0x9E3779B9u;
static float rnd(void) {
    g_seed = g_seed * 1103515245u + 12345u;
    return ((float)((int)(g_seed >> 8) % 2001) - 1000.0f) / 100.0f;
}

static int g_fail = 0;
#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);                                 \
            g_fail++;                                                                              \
        }                                                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// SIMD kernels vs scalar
// ---------------------------------------------------------------------------

static void test_transpose2d_kernel(void) {
    const size_t rows_list[] = {1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33};
    const size_t cols_list[] = {1, 3, 4, 5, 7, 8, 9, 16, 17, 33, 64};
    for (size_t ri = 0; ri < sizeof(rows_list) / sizeof(rows_list[0]); ri++) {
        for (size_t ci = 0; ci < sizeof(cols_list) / sizeof(cols_list[0]); ci++) {
            size_t rows = rows_list[ri], cols = cols_list[ci];
            float* a = (float*)malloc(rows * cols * sizeof(float));
            float* b = (float*)malloc(rows * cols * sizeof(float));
            for (size_t i = 0; i < rows * cols; i++)
                a[i] = rnd();
            boat_simd_transpose2d_f32(a, b, rows, cols);
            int ok = 1;
            for (size_t i = 0; i < rows && ok; i++) {
                for (size_t j = 0; j < cols && ok; j++) {
                    if (fabsf(b[j * rows + i] - a[i * cols + j]) > 1e-4f) ok = 0;
                }
            }
            char msg[64];
            snprintf(msg, sizeof(msg), "transpose2d kernel %zux%zu", rows, cols);
            CHECK(ok, msg);
            free(a);
            free(b);
        }
    }
}

static void test_activation_kernels(void) {
    // Compare the SIMD transcendental kernels against scalar libm for
    // various lengths (exercises tails) and a range covering sigmoid/tanh
    // saturation.
    const size_t lens[] = {1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 100, 1000};
    for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        size_t n = lens[li];
        float* a = (float*)malloc(n * sizeof(float));
        float* got = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) a[i] = rnd();

        // A few extreme values to exercise saturation.
        a[0] = 6.0f;
        a[n - 1] = -6.0f;

        boat_simd_sigmoid_f32(a, got, n);
        for (size_t i = 0; i < n; i++) {
            float ref = 1.0f / (1.0f + expf(-a[i]));
            if (fabsf(got[i] - ref) > 2e-5f) {
                char msg[64];
                snprintf(msg, sizeof(msg), "sigmoid n=%zu i=%zu", n, i);
                CHECK(0, msg);
            }
        }
        boat_simd_tanh_f32(a, got, n);
        for (size_t i = 0; i < n; i++) {
            if (fabsf(got[i] - tanhf(a[i])) > 2e-5f) {
                char msg[64];
                snprintf(msg, sizeof(msg), "tanh n=%zu i=%zu", n, i);
                CHECK(0, msg);
            }
        }
        boat_simd_silu_f32(a, got, n);
        for (size_t i = 0; i < n; i++) {
            float ref = a[i] / (1.0f + expf(-a[i]));
            if (fabsf(got[i] - ref) > 2e-5f) {
                char msg[64];
                snprintf(msg, sizeof(msg), "silu n=%zu i=%zu", n, i);
                CHECK(0, msg);
            }
        }
        boat_simd_gelu_f32(a, got, n);
        for (size_t i = 0; i < n; i++) {
            float x = a[i];
            float ref = 0.5f * x * (1.0f + tanhf(0.7978845608028654f *
                                                 (x + 0.044715f * x * x * x)));
            if (fabsf(got[i] - ref) > 2e-5f) {
                char msg[64];
                snprintf(msg, sizeof(msg), "gelu n=%zu i=%zu", n, i);
                CHECK(0, msg);
            }
        }
        free(a);
        free(got);
    }

    // Softmax over the last dim: verify row sums to 1 and ordering preserved.
    {
        size_t rows = 5, cols = 64;
        float* a = (float*)malloc(rows * cols * sizeof(float));
        float* got = (float*)malloc(rows * cols * sizeof(float));
        for (size_t i = 0; i < rows * cols; i++) a[i] = rnd();
        boat_simd_softmax_f32(a, got, rows, cols);
        int ok = 1;
        for (size_t r = 0; r < rows && ok; r++) {
            float sum = 0.0f;
            for (size_t c = 0; c < cols; c++) sum += got[r * cols + c];
            if (fabsf(sum - 1.0f) > 1e-4f) ok = 0;
        }
        CHECK(ok, "softmax rows sum to 1");
        // Monotonicity: softmax preserves argmax.
        int order_ok = 1;
        for (size_t r = 0; r < rows && order_ok; r++) {
            size_t am = 0, gm = 0;
            for (size_t c = 1; c < cols; c++) {
                if (a[r * cols + c] > a[r * cols + am]) am = c;
                if (got[r * cols + c] > got[r * cols + gm]) gm = c;
            }
            if (am != gm) order_ok = 0;
        }
        CHECK(order_ok, "softmax preserves argmax");
        free(a);
        free(got);
    }
    printf("activation kernels checked vs libm\n");
}

static void test_backward_kernels(void) {
    // Compare the fused activation-derivative / softmax / loss-backward SIMD
    // kernels against scalar references over various lengths.
    const size_t lens[] = {1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 100, 1000};
    for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        size_t n = lens[li];
        float* x = (float*)malloc(n * sizeof(float));
        float* y = (float*)malloc(n * sizeof(float));
        float* dy = (float*)malloc(n * sizeof(float));
        float* got = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            x[i] = rnd();
            y[i] = rnd();  // forward outputs (e.g. sigmoid/tanh values)
            dy[i] = rnd();
        }
        x[0] = 6.0f;
        x[n - 1] = -6.0f;

        boat_simd_sigmoid_backward_f32(dy, y, got, n);
        for (size_t i = 0; i < n; i++) {
            float ref = dy[i] * (y[i] * (1.0f - y[i]));
            if (fabsf(got[i] - ref) > 2e-5f * (1.0f + fabsf(ref))) {
                char msg[64];
                snprintf(msg, sizeof(msg), "sigmoid_bw n=%zu i=%zu", n, i);
                CHECK(0, msg);
            }
        }
        boat_simd_tanh_backward_f32(dy, y, got, n);
        for (size_t i = 0; i < n; i++) {
            float ref = dy[i] * (1.0f - y[i] * y[i]);
            if (fabsf(got[i] - ref) > 2e-5f * (1.0f + fabsf(ref))) {
                char msg[64];
                snprintf(msg, sizeof(msg), "tanh_bw n=%zu i=%zu", n, i);
                CHECK(0, msg);
            }
        }
        boat_simd_relu_backward_f32(dy, x, got, n);
        for (size_t i = 0; i < n; i++) {
            float ref = x[i] > 0.0f ? dy[i] : 0.0f;
            if (fabsf(got[i] - ref) > 1e-6f) {
                char msg[64];
                snprintf(msg, sizeof(msg), "relu_bw n=%zu i=%zu", n, i);
                CHECK(0, msg);
            }
        }
        boat_simd_gelu_backward_f32(dy, x, got, n);
        for (size_t i = 0; i < n; i++) {
            float xv = x[i];
            float a = 0.7978845608028654f * (xv + 0.044715f * xv * xv * xv);
            float t = tanhf(a);
            float d = 0.5f * (1.0f + t) +
                      0.5f * xv * (1.0f - t * t) * 0.7978845608028654f *
                          (1.0f + 3.0f * 0.044715f * xv * xv);
            if (fabsf(got[i] - dy[i] * d) > 2e-5f) {
                char msg[64];
                snprintf(msg, sizeof(msg), "gelu_bw n=%zu i=%zu", n, i);
                CHECK(0, msg);
            }
        }
        free(x);
        free(y);
        free(dy);
        free(got);
    }

    // Row-wise softmax / log-softmax backward.
    {
        size_t rows = 5, cols = 64;
        float* y = (float*)malloc(rows * cols * sizeof(float));
        float* dy = (float*)malloc(rows * cols * sizeof(float));
        float* got = (float*)malloc(rows * cols * sizeof(float));
        for (size_t i = 0; i < rows * cols; i++) {
            y[i] = rnd() * 0.1f + 0.01f;  // softmax outputs stay positive
            dy[i] = rnd();
        }
        boat_simd_softmax_backward_f32(dy, y, got, rows, cols);
        int ok = 1;
        for (size_t r = 0; r < rows && ok; r++) {
            float sum = 0.0f;
            for (size_t c = 0; c < cols; c++) sum += dy[r * cols + c] * y[r * cols + c];
            for (size_t c = 0; c < cols; c++) {
                float ref = y[r * cols + c] * (dy[r * cols + c] - sum);
                if (fabsf(got[r * cols + c] - ref) > 1e-4f) {
                    char msg[64];
                    snprintf(msg, sizeof(msg), "softmax_bw r=%zu c=%zu", r, c);
                    CHECK(0, msg);
                    ok = 0;
                    break;
                }
            }
        }
        CHECK(ok, "softmax backward vs scalar");
        boat_simd_log_softmax_backward_f32(dy, y, got, rows, cols);
        ok = 1;
        for (size_t r = 0; r < rows && ok; r++) {
            float sum = 0.0f;
            for (size_t c = 0; c < cols; c++) sum += dy[r * cols + c];
            for (size_t c = 0; c < cols; c++) {
                float ref = dy[r * cols + c] - expf(y[r * cols + c]) * sum;
                if (fabsf(got[r * cols + c] - ref) > 1e-4f) {
                    char msg[64];
                    snprintf(msg, sizeof(msg), "logsoftmax_bw r=%zu c=%zu", r, c);
                    CHECK(0, msg);
                    ok = 0;
                    break;
                }
            }
        }
        CHECK(ok, "log-softmax backward vs scalar");
        free(y);
        free(dy);
        free(got);
    }

    // Fused softmax-CE backward: (softmax - onehot) * inv_batch.
    {
        size_t rows = 7, cols = 16;
        float* logits = (float*)malloc(rows * cols * sizeof(float));
        int64_t* labels = (int64_t*)malloc(rows * sizeof(int64_t));
        float* got = (float*)malloc(rows * cols * sizeof(float));
        for (size_t i = 0; i < rows * cols; i++) logits[i] = rnd();
        for (size_t r = 0; r < rows; r++) labels[r] = (int64_t)(r * 2 % cols);
        float inv_batch = 1.0f / (float)rows;
        boat_simd_softmax_ce_backward_f32(logits, labels, got, rows, cols, inv_batch);
        int ok = 1;
        for (size_t r = 0; r < rows && ok; r++) {
            // Reference softmax over the row.
            float mx = logits[r * cols];
            for (size_t c = 1; c < cols; c++) {
                if (logits[r * cols + c] > mx) mx = logits[r * cols + c];
            }
            float sum = 0.0f;
            for (size_t c = 0; c < cols; c++) {
                sum += expf(logits[r * cols + c] - mx);
            }
            for (size_t c = 0; c < cols; c++) {
                float p = expf(logits[r * cols + c] - mx) / sum;
                float ref = (p - (c == (size_t)labels[r] ? 1.0f : 0.0f)) * inv_batch;
                if (fabsf(got[r * cols + c] - ref) > 1e-4f) {
                    char msg[64];
                    snprintf(msg, sizeof(msg), "softmax_ce_bw r=%zu c=%zu", r, c);
                    CHECK(0, msg);
                    ok = 0;
                    break;
                }
            }
        }
        CHECK(ok, "softmax-CE backward vs scalar");
        free(logits);
        free(labels);
        free(got);
    }

    // Plain CE backward: -inv_n * t / clip(p).
    {
        size_t n = 1000;
        float* pred = (float*)malloc(n * sizeof(float));
        float* target = (float*)malloc(n * sizeof(float));
        float* got = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            pred[i] = fabsf(rnd()) * 0.5f + 1e-9f;  // keep away from the clamp boundary
            target[i] = fabsf(rnd()) * 0.1f;
        }
        float inv_n = 1.0f / (float)n;
        boat_simd_ce_backward_f32(pred, target, got, n, inv_n, 1e-7f);
        int ok = 1;
        for (size_t i = 0; i < n && ok; i++) {
            float p = pred[i] < 1e-7f ? 1e-7f : (pred[i] > 1.0f - 1e-7f ? 1.0f - 1e-7f : pred[i]);
            float ref = -inv_n * target[i] / p;
            if (fabsf(got[i] - ref) > 1e-5f * (1.0f + fabsf(ref))) {
                char msg[64];
                snprintf(msg, sizeof(msg), "ce_bw n=%zu i=%zu", n, i);
                CHECK(0, msg);
                ok = 0;
            }
        }
        CHECK(ok, "CE backward vs scalar");
        free(pred);
        free(target);
        free(got);
    }
    printf("backward kernels checked vs scalar\n");
}

static void test_elementwise_and_norm_kernels(void) {
    // Elementwise binary / scalar kernels vs scalar references.
    const size_t lens[] = {1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 100, 1000};
    for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        size_t n = lens[li];
        float* a = (float*)malloc(n * sizeof(float));
        float* b = (float*)malloc(n * sizeof(float));
        float* got = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            a[i] = rnd();
            b[i] = fabsf(rnd()) * 0.5f + 0.1f;  // avoid div-by-zero
        }
        float s = 0.75f;
        boat_simd_add_f32(a, b, got, n);
        for (size_t i = 0; i < n; i++) CHECK(fabsf(got[i] - (a[i] + b[i])) <= 1e-6f, "add");
        boat_simd_sub_f32(a, b, got, n);
        for (size_t i = 0; i < n; i++) CHECK(fabsf(got[i] - (a[i] - b[i])) <= 1e-6f, "sub");
        boat_simd_mul_f32(a, b, got, n);
        for (size_t i = 0; i < n; i++) CHECK(fabsf(got[i] - (a[i] * b[i])) <= 1e-6f, "mul");
        boat_simd_div_f32(a, b, got, n);
        for (size_t i = 0; i < n; i++)
            CHECK(fabsf(got[i] - (a[i] / b[i])) <= 1e-6f, "div");
        boat_simd_add_scalar_f32(a, s, got, n);
        for (size_t i = 0; i < n; i++) CHECK(fabsf(got[i] - (a[i] + s)) <= 1e-6f, "add_scalar");
        boat_simd_sub_scalar_f32(a, s, got, n);
        for (size_t i = 0; i < n; i++) CHECK(fabsf(got[i] - (a[i] - s)) <= 1e-6f, "sub_scalar");
        boat_simd_mul_scalar_f32(a, s, got, n);
        for (size_t i = 0; i < n; i++) CHECK(fabsf(got[i] - (a[i] * s)) <= 1e-6f, "mul_scalar");
        boat_simd_div_scalar_f32(a, s, got, n);
        for (size_t i = 0; i < n; i++) CHECK(fabsf(got[i] - (a[i] / s)) <= 1e-6f, "div_scalar");
        boat_simd_abs_f32(a, got, n);
        for (size_t i = 0; i < n; i++) CHECK(fabsf(got[i] - fabsf(a[i])) <= 1e-6f, "abs");
        free(a);
        free(b);
        free(got);
    }

    // Row-wise mean/var and rms.
    {
        size_t rows = 5, cols = 64;
        float* a = (float*)malloc(rows * cols * sizeof(float));
        float* mean = (float*)malloc(rows * sizeof(float));
        float* var = (float*)malloc(rows * sizeof(float));
        float* rms = (float*)malloc(rows * sizeof(float));
        for (size_t i = 0; i < rows * cols; i++) a[i] = rnd();
        boat_simd_mean_var_f32(a, mean, var, rows, cols);
        int ok = 1;
        for (size_t o = 0; o < rows && ok; o++) {
            float sm = 0.0f, sq = 0.0f;
            for (size_t c = 0; c < cols; c++) {
                sm += a[o * cols + c];
                sq += a[o * cols + c] * a[o * cols + c];
            }
            float m = sm / (float)cols;
            float v = sq / (float)cols - m * m;
            if (fabsf(mean[o] - m) > 1e-4f || fabsf(var[o] - v) > 1e-3f) ok = 0;
        }
        CHECK(ok, "mean_var vs scalar");
        boat_simd_rms_f32(a, rms, rows, cols);
        ok = 1;
        for (size_t o = 0; o < rows && ok; o++) {
            float sq = 0.0f;
            for (size_t c = 0; c < cols; c++) sq += a[o * cols + c] * a[o * cols + c];
            if (fabsf(rms[o] - sqrtf(sq / (float)cols)) > 1e-3f) ok = 0;
        }
        CHECK(ok, "rms vs scalar");
        free(a);
        free(mean);
        free(var);
        free(rms);
    }

    // norm_affine with/without weight/bias.
    {
        size_t rows = 4, cols = 32;
        float* x = (float*)malloc(rows * cols * sizeof(float));
        float* w = (float*)malloc(cols * sizeof(float));
        float* b = (float*)malloc(cols * sizeof(float));
        float* out = (float*)malloc(rows * cols * sizeof(float));
        float* mean = (float*)malloc(rows * sizeof(float));
        float* inv_std = (float*)malloc(rows * sizeof(float));
        for (size_t i = 0; i < rows * cols; i++) x[i] = rnd();
        for (size_t i = 0; i < cols; i++) {
            w[i] = rnd() * 0.5f;
            b[i] = rnd() * 0.5f;
        }
        for (size_t o = 0; o < rows; o++) {
            mean[o] = rnd() * 0.1f;
            inv_std[o] = fabsf(rnd()) * 0.5f + 0.5f;
        }
        boat_simd_norm_affine_f32(x, w, b, out, rows, cols, mean, inv_std);
        int ok = 1;
        for (size_t o = 0; o < rows && ok; o++) {
            for (size_t c = 0; c < cols; c++) {
                float ref = (x[o * cols + c] - mean[o]) * inv_std[o] * w[c] + b[c];
                if (fabsf(out[o * cols + c] - ref) > 1e-4f * (1.0f + fabsf(ref))) {
                    ok = 0;
                    break;
                }
            }
        }
        CHECK(ok, "norm_affine w/b");
        boat_simd_norm_affine_f32(x, NULL, NULL, out, rows, cols, NULL, inv_std);
        ok = 1;
        for (size_t o = 0; o < rows && ok; o++) {
            for (size_t c = 0; c < cols; c++) {
                float ref = x[o * cols + c] * inv_std[o];
                if (fabsf(out[o * cols + c] - ref) > 1e-5f) {
                    ok = 0;
                    break;
                }
            }
        }
        CHECK(ok, "norm_affine identity");
        free(x);
        free(w);
        free(b);
        free(out);
        free(mean);
        free(inv_std);
    }

    // LayerNorm backward vs the scalar reference from norm.c.
    {
        size_t rows = 4, cols = 64;
        float eps = 1e-5f;
        float* x = (float*)malloc(rows * cols * sizeof(float));
        float* dy = (float*)malloc(rows * cols * sizeof(float));
        float* gamma = (float*)malloc(cols * sizeof(float));
        float* dx = (float*)malloc(rows * cols * sizeof(float));
        float* dx_ref = (float*)malloc(rows * cols * sizeof(float));
        float* gw = (float*)calloc(cols, sizeof(float));
        float* gw_ref = (float*)calloc(cols, sizeof(float));
        float* gb = (float*)calloc(cols, sizeof(float));
        float* gb_ref = (float*)calloc(cols, sizeof(float));
        for (size_t i = 0; i < rows * cols; i++) {
            x[i] = rnd();
            dy[i] = rnd();
        }
        for (size_t i = 0; i < cols; i++) gamma[i] = rnd() * 0.5f;
        boat_simd_layernorm_backward_f32(x, dy, gamma, dx, gw, gb, rows, cols, eps);
        // Scalar reference (mirrors boat_layernorm_backward).
        for (size_t o = 0; o < rows; o++) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (size_t c = 0; c < cols; c++) {
                sum += x[o * cols + c];
                sum_sq += x[o * cols + c] * x[o * cols + c];
            }
            float m = sum / (float)cols;
            float var = sum_sq / (float)cols - m * m;
            float inv_std = 1.0f / sqrtf(var + eps);
            for (size_t c = 0; c < cols; c++) {
                float x_hat = (x[o * cols + c] - m) * inv_std;
                gw_ref[c] += dy[o * cols + c] * x_hat;
                gb_ref[c] += dy[o * cols + c];
            }
            float s1 = 0.0f, s2 = 0.0f;
            for (size_t c = 0; c < cols; c++) {
                float x_hat = (x[o * cols + c] - m) * inv_std;
                float dy_g = dy[o * cols + c] * gamma[c];
                s1 += dy_g;
                s2 += dy_g * x_hat;
            }
            for (size_t c = 0; c < cols; c++) {
                float x_hat = (x[o * cols + c] - m) * inv_std;
                float dy_g = dy[o * cols + c] * gamma[c];
                dx_ref[o * cols + c] =
                    (dy_g - (s1 + s2 * x_hat) / (float)cols) * inv_std;
            }
        }
        int ok = 1;
        for (size_t i = 0; i < rows * cols && ok; i++) {
            if (fabsf(dx[i] - dx_ref[i]) > 1e-3f * (1.0f + fabsf(dx_ref[i]))) ok = 0;
        }
        for (size_t i = 0; i < cols && ok; i++) {
            if (fabsf(gw[i] - gw_ref[i]) > 1e-3f * (1.0f + fabsf(gw_ref[i]))) ok = 0;
            if (fabsf(gb[i] - gb_ref[i]) > 1e-3f * (1.0f + fabsf(gb_ref[i]))) ok = 0;
        }
        CHECK(ok, "layernorm backward vs scalar");
        free(x);
        free(dy);
        free(gamma);
        free(dx);
        free(dx_ref);
        free(gw);
        free(gw_ref);
        free(gb);
        free(gb_ref);
    }

    // RMSNorm backward vs the scalar reference from norm.c.
    {
        size_t rows = 4, cols = 64;
        float eps = 1e-5f;
        float* x = (float*)malloc(rows * cols * sizeof(float));
        float* dy = (float*)malloc(rows * cols * sizeof(float));
        float* gamma = (float*)malloc(cols * sizeof(float));
        float* dx = (float*)malloc(rows * cols * sizeof(float));
        float* dx_ref = (float*)malloc(rows * cols * sizeof(float));
        float* gw = (float*)calloc(cols, sizeof(float));
        float* gw_ref = (float*)calloc(cols, sizeof(float));
        for (size_t i = 0; i < rows * cols; i++) {
            x[i] = rnd();
            dy[i] = rnd();
        }
        for (size_t i = 0; i < cols; i++) gamma[i] = rnd() * 0.5f;
        boat_simd_rmsnorm_backward_f32(x, dy, gamma, dx, gw, rows, cols, eps);
        for (size_t o = 0; o < rows; o++) {
            float sum_sq = 0.0f;
            for (size_t c = 0; c < cols; c++) sum_sq += x[o * cols + c] * x[o * cols + c];
            float inv_rms = 1.0f / (sqrtf(sum_sq / (float)cols) + eps);
            for (size_t c = 0; c < cols; c++) {
                gw_ref[c] += dy[o * cols + c] * (x[o * cols + c] * inv_rms);
            }
            float s = 0.0f;
            for (size_t c = 0; c < cols; c++) {
                s += dy[o * cols + c] * gamma[c] * x[o * cols + c];
            }
            float scale = s / (float)cols * inv_rms * inv_rms * inv_rms;
            for (size_t c = 0; c < cols; c++) {
                dx_ref[o * cols + c] =
                    dy[o * cols + c] * gamma[c] * inv_rms - x[o * cols + c] * scale;
            }
        }
        int ok = 1;
        for (size_t i = 0; i < rows * cols && ok; i++) {
            if (fabsf(dx[i] - dx_ref[i]) > 1e-3f * (1.0f + fabsf(dx_ref[i]))) ok = 0;
        }
        for (size_t i = 0; i < cols && ok; i++) {
            if (fabsf(gw[i] - gw_ref[i]) > 1e-3f * (1.0f + fabsf(gw_ref[i]))) ok = 0;
        }
        CHECK(ok, "rmsnorm backward vs scalar");
        free(x);
        free(dy);
        free(gamma);
        free(dx);
        free(dx_ref);
        free(gw);
        free(gw_ref);
    }
    printf("elementwise + norm kernels checked vs scalar\n");
}

static void test_reduce_kernels(void) {
    const size_t lens[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 1000};
    for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        size_t n = lens[li];
        float* a = (float*)malloc(n * sizeof(float));
        float sum = 0.0f, mx = 1e30f, mn = -1e30f;
        for (size_t i = 0; i < n; i++) {
            a[i] = rnd();
            sum += a[i];
            if (a[i] > mn) mn = a[i];
            if (a[i] < mx) mx = a[i];
        }
        char msg[64];
        snprintf(msg, sizeof(msg), "sum reduce n=%zu", n);
        CHECK(fabsf(sum - boat_simd_sum_reduce_f32(a, n)) <= 1e-3f * (float)(n + 1), msg);
        snprintf(msg, sizeof(msg), "max reduce n=%zu", n);
        CHECK(fabsf(mn - boat_simd_max_reduce_f32(a, n)) < 1e-4f, msg);
        snprintf(msg, sizeof(msg), "min reduce n=%zu", n);
        CHECK(fabsf(mx - boat_simd_min_reduce_f32(a, n)) < 1e-4f, msg);
        free(a);
    }
}

// ---------------------------------------------------------------------------
// boat_transpose (fast trailing-2-dims path + general path)
// ---------------------------------------------------------------------------

static void scalar_transpose(const float* in, float* out, const int64_t* shape, size_t ndim,
                             int dim0, int dim1) {
    size_t stride[8], oshape[8], ostride[8];
    stride[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--)
        stride[i] = stride[i + 1] * (size_t)shape[i + 1];
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++)
        total *= (size_t)shape[i];
    for (size_t i = 0; i < ndim; i++)
        oshape[i] = shape[i];
    oshape[dim0] = shape[dim1];
    oshape[dim1] = shape[dim0];
    ostride[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--)
        ostride[i] = ostride[i + 1] * oshape[i + 1];
    for (size_t idx = 0; idx < total; idx++) {
        size_t coords[8], t = idx;
        for (int i = (int)ndim - 1; i >= 0; i--) {
            coords[i] = t % (size_t)shape[i];
            t /= (size_t)shape[i];
        }
        size_t tmp = coords[dim0];
        coords[dim0] = coords[dim1];
        coords[dim1] = tmp;
        size_t oidx = 0;
        for (size_t i = 0; i < ndim; i++)
            oidx += coords[i] * ostride[i];
        out[oidx] = in[idx];
    }
}

static void test_transpose_op(void) {
    // Trailing-2-dims (SIMD fast path).
    {
        int64_t sh[] = {2, 3, 5, 7};
        float a[2 * 3 * 5 * 7], ref[2 * 3 * 5 * 7];
        for (size_t i = 0; i < sizeof(a) / sizeof(a[0]); i++)
            a[i] = rnd();
        boat_tensor_t* t = boat_tensor_from_data(sh, 4, BOAT_DTYPE_FLOAT32, a);
        boat_tensor_t* o = boat_transpose(t, 2, 3);
        CHECK(o != NULL, "transpose 4D trailing dims returns tensor");
        const float* od = (const float*)boat_tensor_const_data(o);
        scalar_transpose(a, ref, sh, 4, 2, 3);
        int ok = 1;
        for (size_t i = 0; i < sizeof(a) / sizeof(a[0]) && ok; i++) {
            if (fabsf(od[i] - ref[i]) > 1e-4f) ok = 0;
        }
        CHECK(ok, "transpose 4D trailing dims matches scalar");
        boat_tensor_unref(o);
        boat_tensor_unref(t);
    }
    // 2D (fast path).
    {
        int64_t sh[] = {7, 5};
        float a[35], ref[35];
        for (size_t i = 0; i < 35; i++)
            a[i] = rnd();
        boat_tensor_t* t = boat_tensor_from_data(sh, 2, BOAT_DTYPE_FLOAT32, a);
        boat_tensor_t* o = boat_transpose(t, 0, 1);
        scalar_transpose(a, ref, sh, 2, 0, 1);
        const float* od = (const float*)boat_tensor_const_data(o);
        int ok = 1;
        for (size_t i = 0; i < 35 && ok; i++) {
            if (fabsf(od[i] - ref[i]) > 1e-4f) ok = 0;
        }
        CHECK(ok, "transpose 2D matches scalar");
        boat_tensor_unref(o);
        boat_tensor_unref(t);
    }
    // General path: swap dims 0 and 2 of a 4D tensor.
    {
        int64_t sh[] = {3, 4, 5, 2};
        float a[3 * 4 * 5 * 2], ref[3 * 4 * 5 * 2];
        for (size_t i = 0; i < sizeof(a) / sizeof(a[0]); i++)
            a[i] = rnd();
        boat_tensor_t* t = boat_tensor_from_data(sh, 4, BOAT_DTYPE_FLOAT32, a);
        boat_tensor_t* o = boat_transpose(t, 0, 2);
        scalar_transpose(a, ref, sh, 4, 0, 2);
        const float* od = (const float*)boat_tensor_const_data(o);
        int ok = 1;
        for (size_t i = 0; i < sizeof(a) / sizeof(a[0]) && ok; i++) {
            if (fabsf(od[i] - ref[i]) > 1e-4f) ok = 0;
        }
        CHECK(ok, "transpose general dims matches scalar");
        boat_tensor_unref(o);
        boat_tensor_unref(t);
    }
}

// ---------------------------------------------------------------------------
// Reductions (boat_sum/mean/max/min) vs a scalar reference
// ---------------------------------------------------------------------------

typedef enum { R_SUM, R_MEAN, R_MAX, R_MIN } rkind_t;

static void scalar_reduce(const float* in, float* out, const int64_t* shape, size_t ndim,
                          const int64_t* dims, size_t n_dims, int keepdim, rkind_t kind) {
    bool red[8];
    for (size_t i = 0; i < ndim; i++)
        red[i] = false;
    if (n_dims == 0) {
        for (size_t i = 0; i < ndim; i++)
            red[i] = true;
    } else {
        for (size_t i = 0; i < n_dims; i++)
            red[dims[i]] = true;
    }
    size_t stride[8];
    stride[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--)
        stride[i] = stride[i + 1] * (size_t)shape[i + 1];

    // Reduced dims (ascending), same order as the implementation.
    size_t rd_size[8], rd_stride[8], nrd = 0;
    size_t red_total = 1;
    for (size_t d = 0; d < ndim; d++) {
        if (red[d]) {
            rd_size[nrd] = (size_t)shape[d];
            rd_stride[nrd] = stride[d];
            red_total *= (size_t)shape[d];
            nrd++;
        }
    }

    // Output shape + input-dim mapping.
    int64_t oshape[8];
    size_t out_to_in[8];
    size_t ondim = 0;
    for (size_t d = 0; d < ndim; d++) {
        if (!red[d]) {
            out_to_in[ondim] = d;
            oshape[ondim++] = shape[d];
        } else if (keepdim) {
            out_to_in[ondim] = d;
            oshape[ondim++] = 1;
        }
    }
    size_t ostride[8];
    size_t onelems = 1;
    if (ondim > 0) {
        ostride[ondim - 1] = 1;
        for (int i = (int)ondim - 2; i >= 0; i--)
            ostride[i] = ostride[i + 1] * (size_t)oshape[i + 1];
        for (size_t i = 0; i < ondim; i++)
            onelems *= (size_t)oshape[i];
    }

    for (size_t oi = 0; oi < onelems; oi++) {
        size_t rem = oi;
        size_t base = 0;
        for (size_t d = 0; d < ondim; d++) {
            size_t c = rem / ostride[d];
            rem %= ostride[d];
            base += c * stride[out_to_in[d]];
        }
        // Walk reduced dims ascending; the last reduced dim is innermost.
        size_t outer = nrd ? nrd - 1 : 0;
        size_t outer_total = 1;
        for (size_t d = 0; d < outer; d++)
            outer_total *= rd_size[d];
        size_t inner_n = nrd ? rd_size[nrd - 1] : 0;
        size_t inner_stride = nrd ? rd_stride[nrd - 1] : 0;
        double acc = 0.0;
        bool first = true;
        for (size_t ro = 0; ro < outer_total; ro++) {
            size_t rr = ro;
            size_t off = 0;
            for (size_t d = 0; d < outer; d++) {
                size_t c = rr % rd_size[d];
                rr /= rd_size[d];
                off += c * rd_stride[d];
            }
            for (size_t j = 0; j < inner_n; j++) {
                float v = in[base + off + j * inner_stride];
                if (kind == R_SUM || kind == R_MEAN) {
                    acc += v;
                } else if (kind == R_MAX) {
                    if (first || v > acc) acc = v;
                    first = false;
                } else {
                    if (first || v < acc) acc = v;
                    first = false;
                }
            }
        }
        if (kind == R_MEAN && red_total > 0) acc /= (double)red_total;
        out[oi] = (float)acc;
    }
}

static void test_reduce_op(void) {
    int64_t sh[] = {2, 3, 4};
    float a[24];
    for (size_t i = 0; i < 24; i++)
        a[i] = rnd();
    boat_tensor_t* t = boat_tensor_from_data(sh, 3, BOAT_DTYPE_FLOAT32, a);

    struct {
        const int64_t* dims;
        size_t ndims;
        int keepdim;
    } cases[] = {
        {NULL, 0, 0},              // all dims
        {NULL, 0, 1},              // all dims keepdim
        {(int64_t[]){2}, 1, 0},    // last dim (SIMD path)
        {(int64_t[]){1}, 1, 0},    // middle dim (strided)
        {(int64_t[]){0}, 1, 0},    // first dim (strided)
        {(int64_t[]){1, 2}, 2, 0}, // two dims, contiguous inner
        {(int64_t[]){0, 2}, 2, 0}, // two dims, strided inner
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        rkind_t kinds[] = {R_SUM, R_MEAN, R_MAX, R_MIN};
        for (size_t ki = 0; ki < sizeof(kinds) / sizeof(kinds[0]); ki++) {
            boat_tensor_t* o = NULL;
            const int64_t* dims = cases[ci].dims;
            size_t nd = cases[ci].ndims;
            int kd = cases[ci].keepdim;
            switch (kinds[ki]) {
            case R_SUM: o = boat_sum(t, dims, nd, kd); break;
            case R_MEAN: o = boat_mean(t, dims, nd, kd); break;
            case R_MAX: o = boat_max(t, dims, nd, kd); break;
            case R_MIN: o = boat_min(t, dims, nd, kd); break;
            }
            CHECK(o != NULL, "reduce returns tensor");
            // Compute reference output shape via boat_tensor_shape(o).
            const int64_t* osh = boat_tensor_shape(o);
            size_t ondim = boat_tensor_ndim(o);
            size_t onelems = 1;
            for (size_t i = 0; i < ondim; i++)
                onelems *= (size_t)osh[i];
            float* ref = (float*)malloc(onelems * sizeof(float));
            scalar_reduce(a, ref, sh, 3, dims, nd, kd, kinds[ki]);
            const float* od = (const float*)boat_tensor_const_data(o);
            int ok = 1;
            for (size_t i = 0; i < onelems && ok; i++) {
                float d = od[i] - ref[i];
                if (d < 0) d = -d;
                if (d > 1e-3f) ok = 0;
            }
            char msg[96];
            snprintf(msg, sizeof(msg), "reduce case %zu kind %zu", ci, ki);
            CHECK(ok, msg);
            free(ref);
            boat_tensor_unref(o);
        }
    }
    boat_tensor_unref(t);
}

// ---------------------------------------------------------------------------
// Conv2d stride-1 forward vs an independent scalar reference
// ---------------------------------------------------------------------------

static void scalar_conv2d(const float* in, const float* w, const float* bias, float* out, int batch,
                          int in_ch, int out_ch, int h, int wi, int kh, int kw, int pad, int stride,
                          int groups) {
    int ho = (h + 2 * pad - kh) / stride + 1;
    int wo = (wi + 2 * pad - kw) / stride + 1;
    int och_per_g = out_ch / groups, ich_per_g = in_ch / groups;
    for (int i = 0; i < batch * out_ch * ho * wo; i++)
        out[i] = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int g = 0; g < groups; g++) {
            for (int oc = g * och_per_g; oc < (g + 1) * och_per_g; oc++) {
                for (int ic = g * ich_per_g; ic < (g + 1) * ich_per_g; ic++) {
                    int icl = ic - g * ich_per_g;
                    for (int oh = 0; oh < ho; oh++) {
                        for (int ow = 0; ow < wo; ow++) {
                            float s = 0.0f;
                            for (int i = 0; i < kh; i++) {
                                int ih = oh * stride - pad + i;
                                if (ih < 0 || ih >= h) continue;
                                for (int j = 0; j < kw; j++) {
                                    int iw = ow * stride - pad + j;
                                    if (iw < 0 || iw >= wi) continue;
                                    s += in[((b * in_ch + ic) * h + ih) * wi + iw] *
                                         w[((oc * ich_per_g + icl) * kh + i) * kw + j];
                                }
                            }
                            out[((b * out_ch + oc) * ho + oh) * wo + ow] += s;
                        }
                    }
                }
            }
        }
    }
    if (bias) {
        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < out_ch; oc++) {
                for (int i = 0; i < ho * wo; i++) {
                    out[((b * out_ch + oc) * ho * wo) + i] += bias[oc];
                }
            }
        }
    }
}

static void test_conv_case(const char* label, int batch, int in_ch, int out_ch, int h, int wi,
                           int kh, int kw, int pad, int stride, int groups) {
    int ho = (h + 2 * pad - kh) / stride + 1;
    int wo = (wi + 2 * pad - kw) / stride + 1;
    int ich_per_g = in_ch / groups;
    size_t nin = (size_t)batch * in_ch * h * wi;
    size_t nw = (size_t)out_ch * ich_per_g * kh * kw;
    float* in = (float*)malloc(nin * sizeof(float));
    float* w = (float*)malloc(nw * sizeof(float));
    float* bias = (float*)malloc((size_t)out_ch * sizeof(float));
    for (size_t i = 0; i < nin; i++) in[i] = rnd();
    for (size_t i = 0; i < nw; i++) w[i] = rnd();
    for (int i = 0; i < out_ch; i++) bias[i] = rnd();

    int64_t ish[] = {batch, in_ch, h, wi};
    int64_t wsh[] = {out_ch, ich_per_g, kh, kw};
    int64_t bsh[] = {out_ch};
    boat_tensor_t* it = boat_tensor_from_data(ish, 4, BOAT_DTYPE_FLOAT32, in);
    boat_tensor_t* wt = boat_tensor_from_data(wsh, 4, BOAT_DTYPE_FLOAT32, w);
    boat_tensor_t* bt = boat_tensor_from_data(bsh, 1, BOAT_DTYPE_FLOAT32, bias);

    boat_conv_layer_t* conv = boat_conv_layer_create((size_t)in_ch, (size_t)out_ch, (size_t)kh,
                                                     (size_t)stride, (size_t)pad,
                                                     (size_t)groups);
    CHECK(conv != NULL, "conv layer create");
    boat_conv_layer_set_weight(conv, wt);
    boat_conv_layer_set_bias(conv, bt);
    boat_tensor_t* o = boat_conv_layer_forward(conv, it);
    CHECK(o != NULL, "conv forward");

    size_t nout = (size_t)batch * out_ch * ho * wo;
    float* ref = (float*)malloc(nout * sizeof(float));
    scalar_conv2d(in, w, bias, ref, batch, in_ch, out_ch, h, wi, kh, kw, pad, stride, groups);
    const float* od = (const float*)boat_tensor_const_data(o);
    int ok = 1;
    for (size_t i = 0; i < nout; i++) {
        if (fabsf(od[i] - ref[i]) > 1e-3f) ok = 0;
    }
    char msg[96];
    snprintf(msg, sizeof(msg), "conv2d %s matches scalar reference", label);
    CHECK(ok, msg);

    free(ref);
    boat_tensor_unref(o);
    boat_tensor_unref(it);
    boat_tensor_unref(wt);
    boat_tensor_unref(bt);
    boat_conv_layer_free(conv);
    free(in);
    free(w);
    free(bias);
}

static void test_conv_stride1(void) {
    // Small case: SIMD interior path (below the im2col size gate).
    test_conv_case("stride-1 small", 1, 2, 2, 8, 8, 3, 3, 1, 1, 1);
    // Larger cases: im2col + SGEMM path (out_ch*ckk >= 1024, npos >= 64).
    test_conv_case("stride-1 im2col", 2, 8, 16, 16, 16, 3, 3, 1, 1, 1);
    test_conv_case("stride-2 im2col", 1, 8, 16, 16, 16, 3, 3, 1, 2, 1);
    test_conv_case("im2col groups", 1, 8, 32, 16, 16, 3, 3, 0, 1, 2);
    test_conv_case("im2col 5x5 pad2", 1, 4, 12, 20, 20, 5, 5, 2, 1, 1);
}

int main(void) {
    test_transpose2d_kernel();
    test_activation_kernels();
    test_backward_kernels();
    test_elementwise_and_norm_kernels();
    test_reduce_kernels();
    test_transpose_op();
    test_reduce_op();
    test_conv_stride1();
    if (g_fail) {
        printf("%d test(s) FAILED\n", g_fail);
        return 1;
    }
    printf("All SIMD tests passed.\n");
    return 0;
}
