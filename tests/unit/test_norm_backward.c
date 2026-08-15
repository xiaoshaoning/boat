// test_norm_backward.c - Forward and backward (numerical gradient) checks for LayerNorm/RMSNorm
// Copyright (c) 2026 Shaoning, Xiao
// Licensed under the Apache License, Version 2.0

#include <boat/layers/norm.h>
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

static float scalar_loss(const float* v, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++)
        s += v[i];
    return s;
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

// ---------- LayerNorm ----------

static int test_layernorm_forward_and_gradients(void) {
    printf("LayerNorm forward and backward numerical gradient check\n");
    const size_t batch = 2, seq = 3, hidden = 8;
    const size_t total = batch * seq * hidden;
    const float eps = 1e-5f;
    int bad = 0;

    float* x = ALLOC_F32(total);
    float* gamma = ALLOC_F32(hidden);
    float* beta = ALLOC_F32(hidden);
    float* ref = ALLOC_F32(total);
    float* num_x = ALLOC_F32(total);
    float* num_g = ALLOC_F32(hidden);
    float* num_b = ALLOC_F32(hidden);
    if (!x || !gamma || !beta || !ref || !num_x || !num_g || !num_b) {
        printf("  FAIL: allocation failed\n");
        g_failures++;
        goto done;
    }
    srand(101);
    for (size_t i = 0; i < total; i++)
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (size_t i = 0; i < hidden; i++)
        gamma[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (size_t i = 0; i < hidden; i++)
        beta[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    boat_layernorm_config_t cfg = {hidden, eps, true, true};
    boat_layernorm_t* ln = boat_layernorm_create(&cfg);
    int64_t shp1[] = {(int64_t)hidden};
    int64_t shp3[] = {(int64_t)batch, (int64_t)seq, (int64_t)hidden};
    boat_tensor_t* t_x = boat_tensor_from_data(shp3, 3, BOAT_DTYPE_FLOAT32, x);
    boat_tensor_t* t_g = boat_tensor_from_data(shp1, 1, BOAT_DTYPE_FLOAT32, gamma);
    boat_tensor_t* t_b = boat_tensor_from_data(shp1, 1, BOAT_DTYPE_FLOAT32, beta);
    boat_layernorm_set_weight(ln, t_g);
    boat_layernorm_set_bias(ln, t_b);
    // Keep local references so numerical gradients can perturb tensor data directly

    // Reference forward
    for (size_t o = 0; o < batch * seq; o++) {
        float mean = 0.0f, var = 0.0f;
        for (size_t j = 0; j < hidden; j++)
            mean += x[o * hidden + j];
        mean /= (float)hidden;
        for (size_t j = 0; j < hidden; j++) {
            float d = x[o * hidden + j] - mean;
            var += d * d;
        }
        var /= (float)hidden;
        float inv = 1.0f / sqrtf(var + eps);
        for (size_t j = 0; j < hidden; j++) {
            float xh = (x[o * hidden + j] - mean) * inv;
            ref[o * hidden + j] = xh * gamma[j] + beta[j];
        }
    }

    boat_tensor_t* out = boat_layernorm_forward(ln, t_x);
    CHECK(out != NULL, "layernorm forward returns output");
    if (!out) {
        g_failures++;
        goto done;
    }
    bad = 0;
    for (size_t i = 0; i < total; i++) {
        if (fabsf(((const float*)boat_tensor_const_data(out))[i] - ref[i]) > 1e-4f) bad++;
    }
    printf("  forward mismatches vs reference: %d\n", bad);
    g_failures += bad;
    boat_tensor_free(out);

    // Numerical gradients (loss = sum of outputs)
    const float deps = 1e-3f, atol = 1e-3f, rtol = 1e-2f;
    float* xd = (float*)boat_tensor_data(t_x);
    float* gd = (float*)boat_tensor_data(t_g);
    float* bd = (float*)boat_tensor_data(t_b);
    for (size_t i = 0; i < total; i++) {
        float orig = xd[i];
        xd[i] = orig + deps;
        boat_tensor_t* o1 = boat_layernorm_forward(ln, t_x);
        float l1 = o1 ? scalar_loss((const float*)boat_tensor_const_data(o1), total) : 0.0f;
        if (o1) boat_tensor_free(o1);
        xd[i] = orig - deps;
        boat_tensor_t* o2 = boat_layernorm_forward(ln, t_x);
        float l2 = o2 ? scalar_loss((const float*)boat_tensor_const_data(o2), total) : 0.0f;
        if (o2) boat_tensor_free(o2);
        xd[i] = orig;
        num_x[i] = (l1 - l2) / (2.0f * deps);
    }
    for (size_t i = 0; i < hidden; i++) {
        float orig = gd[i];
        gd[i] = orig + deps;
        boat_tensor_t* o1 = boat_layernorm_forward(ln, t_x);
        float l1 = o1 ? scalar_loss((const float*)boat_tensor_const_data(o1), total) : 0.0f;
        if (o1) boat_tensor_free(o1);
        gd[i] = orig - deps;
        boat_tensor_t* o2 = boat_layernorm_forward(ln, t_x);
        float l2 = o2 ? scalar_loss((const float*)boat_tensor_const_data(o2), total) : 0.0f;
        if (o2) boat_tensor_free(o2);
        gd[i] = orig;
        num_g[i] = (l1 - l2) / (2.0f * deps);
    }
    for (size_t i = 0; i < hidden; i++) {
        float orig = bd[i];
        bd[i] = orig + deps;
        boat_tensor_t* o1 = boat_layernorm_forward(ln, t_x);
        float l1 = o1 ? scalar_loss((const float*)boat_tensor_const_data(o1), total) : 0.0f;
        if (o1) boat_tensor_free(o1);
        bd[i] = orig - deps;
        boat_tensor_t* o2 = boat_layernorm_forward(ln, t_x);
        float l2 = o2 ? scalar_loss((const float*)boat_tensor_const_data(o2), total) : 0.0f;
        if (o2) boat_tensor_free(o2);
        bd[i] = orig;
        num_b[i] = (l1 - l2) / (2.0f * deps);
    }

    // Analytical gradients: forward then backward with ones
    boat_tensor_t* o = boat_layernorm_forward(ln, t_x);
    boat_tensor_t* grad_out = boat_tensor_create_like(o);
    memset(boat_tensor_data(grad_out), 0, boat_tensor_nbytes(grad_out));
    float* god = (float*)boat_tensor_data(grad_out);
    for (size_t i = 0; i < total; i++)
        god[i] = 1.0f;
    boat_tensor_free(o);

    boat_tensor_t* grad_in = boat_layernorm_backward(ln, grad_out);
    CHECK(grad_in != NULL, "layernorm backward returns grad_input");
    if (!grad_in) {
        g_failures++;
        boat_tensor_free(grad_out);
        goto done;
    }
    bad = compare_grads("grad_input", (const float*)boat_tensor_const_data(grad_in), num_x, total,
                        atol, rtol);
    printf("  grad_input mismatches: %d\n", bad);
    g_failures += bad;

    boat_tensor_t* gw = boat_layernorm_get_grad_weight(ln);
    boat_tensor_t* gb = boat_layernorm_get_grad_bias(ln);
    CHECK(gw != NULL && gb != NULL, "layernorm exposes grad_weight/grad_bias");
    if (gw && gb) {
        bad = compare_grads("grad_weight", (const float*)boat_tensor_const_data(gw), num_g, hidden,
                            atol, rtol);
        printf("  grad_weight mismatches: %d\n", bad);
        g_failures += bad;
        bad = compare_grads("grad_bias", (const float*)boat_tensor_const_data(gb), num_b, hidden,
                            atol, rtol);
        printf("  grad_bias mismatches: %d\n", bad);
        g_failures += bad;
    }

    boat_tensor_free(grad_out);
    boat_tensor_free(grad_in);
    boat_tensor_free(t_x);
    boat_tensor_free(t_g);
    boat_tensor_free(t_b);
    boat_layernorm_free(ln);
done:
    free(x);
    free(gamma);
    free(beta);
    free(ref);
    free(num_x);
    free(num_g);
    free(num_b);
    return bad;
}

// ---------- RMSNorm ----------

static int test_rmsnorm_forward_and_gradients(void) {
    printf("RMSNorm forward and backward numerical gradient check\n");
    const size_t batch = 2, seq = 3, hidden = 8;
    const size_t total = batch * seq * hidden;
    const float eps = 1e-5f;
    int bad = 0;

    float* x = ALLOC_F32(total);
    float* gamma = ALLOC_F32(hidden);
    float* ref = ALLOC_F32(total);
    float* num_x = ALLOC_F32(total);
    float* num_g = ALLOC_F32(hidden);
    if (!x || !gamma || !ref || !num_x || !num_g) {
        printf("  FAIL: allocation failed\n");
        g_failures++;
        goto done;
    }
    srand(202);
    for (size_t i = 0; i < total; i++)
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (size_t i = 0; i < hidden; i++)
        gamma[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    boat_rmsnorm_config_t cfg = {hidden, eps, true};
    boat_rmsnorm_t* rn = boat_rmsnorm_create(&cfg);
    int64_t shp1[] = {(int64_t)hidden};
    int64_t shp3[] = {(int64_t)batch, (int64_t)seq, (int64_t)hidden};
    boat_tensor_t* t_x = boat_tensor_from_data(shp3, 3, BOAT_DTYPE_FLOAT32, x);
    boat_tensor_t* t_g = boat_tensor_from_data(shp1, 1, BOAT_DTYPE_FLOAT32, gamma);
    boat_rmsnorm_set_weight(rn, t_g);
    // Keep local reference so numerical gradients can perturb tensor data directly

    // Reference forward
    for (size_t o = 0; o < batch * seq; o++) {
        float ss = 0.0f;
        for (size_t j = 0; j < hidden; j++)
            ss += x[o * hidden + j] * x[o * hidden + j];
        float rms = sqrtf(ss / (float)hidden);
        float inv = 1.0f / (rms + eps);
        for (size_t j = 0; j < hidden; j++) {
            ref[o * hidden + j] = x[o * hidden + j] * inv * gamma[j];
        }
    }

    boat_tensor_t* out = boat_rmsnorm_forward(rn, t_x);
    CHECK(out != NULL, "rmsnorm forward returns output");
    if (!out) {
        g_failures++;
        goto done;
    }
    bad = 0;
    for (size_t i = 0; i < total; i++) {
        if (fabsf(((const float*)boat_tensor_const_data(out))[i] - ref[i]) > 1e-4f) bad++;
    }
    printf("  forward mismatches vs reference: %d\n", bad);
    g_failures += bad;
    boat_tensor_free(out);

    const float deps = 1e-3f, atol = 1e-3f, rtol = 1e-2f;
    float* xd = (float*)boat_tensor_data(t_x);
    float* gd = (float*)boat_tensor_data(t_g);
    for (size_t i = 0; i < total; i++) {
        float orig = xd[i];
        xd[i] = orig + deps;
        boat_tensor_t* o1 = boat_rmsnorm_forward(rn, t_x);
        float l1 = o1 ? scalar_loss((const float*)boat_tensor_const_data(o1), total) : 0.0f;
        if (o1) boat_tensor_free(o1);
        xd[i] = orig - deps;
        boat_tensor_t* o2 = boat_rmsnorm_forward(rn, t_x);
        float l2 = o2 ? scalar_loss((const float*)boat_tensor_const_data(o2), total) : 0.0f;
        if (o2) boat_tensor_free(o2);
        xd[i] = orig;
        num_x[i] = (l1 - l2) / (2.0f * deps);
    }
    for (size_t i = 0; i < hidden; i++) {
        float orig = gd[i];
        gd[i] = orig + deps;
        boat_tensor_t* o1 = boat_rmsnorm_forward(rn, t_x);
        float l1 = o1 ? scalar_loss((const float*)boat_tensor_const_data(o1), total) : 0.0f;
        if (o1) boat_tensor_free(o1);
        gd[i] = orig - deps;
        boat_tensor_t* o2 = boat_rmsnorm_forward(rn, t_x);
        float l2 = o2 ? scalar_loss((const float*)boat_tensor_const_data(o2), total) : 0.0f;
        if (o2) boat_tensor_free(o2);
        gd[i] = orig;
        num_g[i] = (l1 - l2) / (2.0f * deps);
    }

    boat_tensor_t* o = boat_rmsnorm_forward(rn, t_x);
    boat_tensor_t* grad_out = boat_tensor_create_like(o);
    memset(boat_tensor_data(grad_out), 0, boat_tensor_nbytes(grad_out));
    float* god = (float*)boat_tensor_data(grad_out);
    for (size_t i = 0; i < total; i++)
        god[i] = 1.0f;
    boat_tensor_free(o);

    boat_tensor_t* grad_in = boat_rmsnorm_backward(rn, grad_out);
    CHECK(grad_in != NULL, "rmsnorm backward returns grad_input");
    if (!grad_in) {
        g_failures++;
        boat_tensor_free(grad_out);
        goto done;
    }
    bad = compare_grads("grad_input", (const float*)boat_tensor_const_data(grad_in), num_x, total,
                        atol, rtol);
    printf("  grad_input mismatches: %d\n", bad);
    g_failures += bad;

    boat_tensor_t* gw = boat_rmsnorm_get_grad_weight(rn);
    CHECK(gw != NULL, "rmsnorm exposes grad_weight");
    if (gw) {
        bad = compare_grads("grad_weight", (const float*)boat_tensor_const_data(gw), num_g, hidden,
                            atol, rtol);
        printf("  grad_weight mismatches: %d\n", bad);
        g_failures += bad;
    }

    boat_tensor_free(grad_out);
    boat_tensor_free(grad_in);
    boat_tensor_free(t_x);
    boat_tensor_free(t_g);
    boat_rmsnorm_free(rn);
done:
    free(x);
    free(gamma);
    free(ref);
    free(num_x);
    free(num_g);
    return bad;
}

// ---------- standalone functions ----------

static int test_standalone_norms(void) {
    printf("Standalone boat_layer_norm / boat_rms_norm\n");
    const size_t batch = 2, seq = 2, hidden = 6;
    const size_t total = batch * seq * hidden;
    const float eps = 1e-5f;
    int bad = 0;

    float* x = ALLOC_F32(total);
    float* ref = ALLOC_F32(total);
    if (!x || !ref) {
        printf("  FAIL: allocation failed\n");
        g_failures++;
        goto done;
    }
    srand(303);
    for (size_t i = 0; i < total; i++)
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    int64_t shp[] = {(int64_t)batch, (int64_t)seq, (int64_t)hidden};
    int64_t ns[] = {(int64_t)hidden};
    boat_tensor_t* t_x = boat_tensor_from_data(shp, 3, BOAT_DTYPE_FLOAT32, x);

    // LayerNorm standalone
    boat_tensor_t* ln_out = boat_layer_norm(t_x, ns, 1, eps);
    CHECK(ln_out != NULL, "boat_layer_norm returns output");
    if (ln_out) {
        for (size_t o = 0; o < batch * seq; o++) {
            float mean = 0.0f, var = 0.0f;
            for (size_t j = 0; j < hidden; j++)
                mean += x[o * hidden + j];
            mean /= (float)hidden;
            for (size_t j = 0; j < hidden; j++) {
                float d = x[o * hidden + j] - mean;
                var += d * d;
            }
            var /= (float)hidden;
            float inv = 1.0f / sqrtf(var + eps);
            for (size_t j = 0; j < hidden; j++) {
                ref[o * hidden + j] = (x[o * hidden + j] - mean) * inv;
            }
        }
        bad = 0;
        for (size_t i = 0; i < total; i++) {
            if (fabsf(((const float*)boat_tensor_const_data(ln_out))[i] - ref[i]) > 1e-4f) bad++;
        }
        printf("  layer_norm mismatches: %d\n", bad);
        g_failures += bad;
        boat_tensor_free(ln_out);
    }

    // RMSNorm standalone
    boat_tensor_t* rn_out = boat_rms_norm(t_x, ns, 1, eps);
    CHECK(rn_out != NULL, "boat_rms_norm returns output");
    if (rn_out) {
        for (size_t o = 0; o < batch * seq; o++) {
            float ss = 0.0f;
            for (size_t j = 0; j < hidden; j++)
                ss += x[o * hidden + j] * x[o * hidden + j];
            float inv = 1.0f / (sqrtf(ss / (float)hidden) + eps);
            for (size_t j = 0; j < hidden; j++) {
                ref[o * hidden + j] = x[o * hidden + j] * inv;
            }
        }
        bad = 0;
        for (size_t i = 0; i < total; i++) {
            if (fabsf(((const float*)boat_tensor_const_data(rn_out))[i] - ref[i]) > 1e-4f) bad++;
        }
        printf("  rms_norm mismatches: %d\n", bad);
        g_failures += bad;
        boat_tensor_free(rn_out);
    }

    boat_tensor_free(t_x);
done:
    free(x);
    free(ref);
    return bad;
}

int main(void) {
    printf("=== Normalization backward tests ===\n");
    test_layernorm_forward_and_gradients();
    test_rmsnorm_forward_and_gradients();
    test_standalone_norms();
    printf("\n%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures > 0 ? 1 : 0;
}
