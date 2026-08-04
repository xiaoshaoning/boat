// test_activation.c - Known-value checks for sigmoid / tanh / SELU
// Copyright (c) 2026 Shaoning, Xiao
// Licensed under the Apache License, Version 2.0

#include <boat/ops.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        g_failures++; \
    } else { \
        printf("  OK: %s\n", msg); \
    } \
} while (0)

static int check_vec(const char* name, const float* got, const float* want, size_t n, float tol) {
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (fabsf(got[i] - want[i]) > tol) {
            bad++;
            if (bad <= 3) {
                printf("    %s[%zu] got=%.7f want=%.7f\n", name, i, got[i], want[i]);
            }
        }
    }
    return bad;
}

int main(void) {
    printf("=== Activation function tests ===\n");

    const float inputs[] = { -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f };
    const size_t n = sizeof(inputs) / sizeof(inputs[0]);
    const int64_t shape[] = { (int64_t)n };
    float* want = (float*)malloc(n * sizeof(float));

    // sigmoid (FP32)
    boat_tensor_t* t = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, inputs);
    boat_tensor_t* out = boat_sigmoid(t);
    CHECK(out != NULL, "boat_sigmoid returns output");
    if (out) {
        for (size_t i = 0; i < n; i++) want[i] = 1.0f / (1.0f + expf(-inputs[i]));
        int bad = check_vec("sigmoid", (const float*)boat_tensor_const_data(out), want, n, 1e-6f);
        CHECK(bad == 0, "sigmoid matches 1/(1+exp(-x))");
        boat_tensor_free(out);
    }

    // tanh (FP32)
    out = boat_tanh(t);
    CHECK(out != NULL, "boat_tanh returns output");
    if (out) {
        for (size_t i = 0; i < n; i++) want[i] = tanhf(inputs[i]);
        int bad = check_vec("tanh", (const float*)boat_tensor_const_data(out), want, n, 1e-6f);
        CHECK(bad == 0, "tanh matches tanhf");
        boat_tensor_free(out);
    }

    // SELU (FP32)
    out = boat_selu(t);
    CHECK(out != NULL, "boat_selu returns output");
    if (out) {
        const float scale = 1.0507009873554804934193349852946f;
        const float alpha = 1.6732632423543772848170429916717f;
        for (size_t i = 0; i < n; i++) {
            float x = inputs[i];
            want[i] = scale * (x > 0.0f ? x : alpha * (expf(x) - 1.0f));
        }
        int bad = check_vec("selu", (const float*)boat_tensor_const_data(out), want, n, 1e-6f);
        CHECK(bad == 0, "selu matches scale*(x>0 ? x : alpha*(exp(x)-1))");
        boat_tensor_free(out);
    }
    boat_tensor_free(t);

    // FP64 path
    double inputs64[] = { -2.0, -0.5, 0.0, 0.5, 2.0 };
    const int64_t shape64[] = { 5 };
    boat_tensor_t* t64 = boat_tensor_from_data(shape64, 1, BOAT_DTYPE_FLOAT64, inputs64);
    out = boat_sigmoid(t64);
    CHECK(out != NULL, "boat_sigmoid FP64 returns output");
    if (out) {
        const double* d = (const double*)boat_tensor_const_data(out);
        int bad = 0;
        for (size_t i = 0; i < 5; i++) {
            double w = 1.0 / (1.0 + exp(-inputs64[i]));
            if (fabs(d[i] - w) > 1e-12) bad++;
        }
        CHECK(bad == 0, "sigmoid FP64 values");
        boat_tensor_free(out);
    }
    out = boat_tanh(t64);
    CHECK(out != NULL, "boat_tanh FP64 returns output");
    if (out) {
        const double* d = (const double*)boat_tensor_const_data(out);
        int bad = 0;
        for (size_t i = 0; i < 5; i++) {
            double w = tanh(inputs64[i]);
            if (fabs(d[i] - w) > 1e-12) bad++;
        }
        CHECK(bad == 0, "tanh FP64 values");
        boat_tensor_free(out);
    }
    out = boat_selu(t64);
    CHECK(out != NULL, "boat_selu FP64 returns output");
    if (out) {
        const double scale = 1.0507009873554804934193349852946;
        const double alpha = 1.6732632423543772848170429916717;
        const double* d = (const double*)boat_tensor_const_data(out);
        int bad = 0;
        for (size_t i = 0; i < 5; i++) {
            double x = inputs64[i];
            double w = scale * (x > 0.0 ? x : alpha * (exp(x) - 1.0));
            if (fabs(d[i] - w) > 1e-12) bad++;
        }
        CHECK(bad == 0, "selu FP64 values");
        boat_tensor_free(out);
    }
    boat_tensor_free(t64);

    free(want);
    printf("\n%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures > 0 ? 1 : 0;
}
