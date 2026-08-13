// test_softmax_ce.c - Softmax cross-entropy loss tests
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/loss.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

static float fabsdiff(float a, float b) { float d = a - b; return d < 0 ? -d : d; }

static void softmax_ref(const float* l, size_t n, float* out) {
    float m = l[0];
    for (size_t i = 1; i < n; i++) if (l[i] > m) m = l[i];
    float s = 0;
    for (size_t i = 0; i < n; i++) { out[i] = expf(l[i] - m); s += out[i]; }
    for (size_t i = 0; i < n; i++) out[i] /= s;
}

static void test_forward(void) {
    printf("Testing softmax CE forward...\n");
    int64_t lshape[] = {2, 3};
    float ldata[] = {1, 2, 3, 1, 0, 0};
    int64_t tshape[] = {2};
    int64_t tdata[] = {2, 0};

    boat_tensor_t* logits = boat_tensor_from_data(lshape, 2, BOAT_DTYPE_FLOAT32, ldata);
    boat_tensor_t* target = boat_tensor_from_data(tshape, 1, BOAT_DTYPE_INT64, tdata);
    boat_loss_t* loss = boat_softmax_cross_entropy_loss_create();
    assert(loss != NULL);

    float l = boat_loss_compute(loss, logits, target);

    // Reference: mean(-log(softmax(logits)[label]))
    float sm[3];
    softmax_ref(ldata, 3, sm);
    float l0 = -logf(sm[2]);
    softmax_ref(ldata + 3, 3, sm);
    float l1 = -logf(sm[0]);
    float want = (l0 + l1) / 2.0f;

    assert(fabsdiff(l, want) < 1e-5);

    boat_loss_free(loss);
    boat_tensor_free(logits);
    boat_tensor_free(target);
    printf("  OK (loss=%f)\n", l);
}

static void test_backward_gradient_check(void) {
    printf("Testing softmax CE backward (numerical)...\n");
    int64_t lshape[] = {2, 3};
    float ldata[] = {0.2f, 1.5f, -0.7f, 2.0f, 0.1f, -1.0f};
    int64_t tshape[] = {2};
    int64_t tdata[] = {1, 0};

    boat_tensor_t* logits = boat_tensor_from_data(lshape, 2, BOAT_DTYPE_FLOAT32, ldata);
    boat_tensor_t* target = boat_tensor_from_data(tshape, 1, BOAT_DTYPE_INT64, tdata);
    boat_loss_t* loss = boat_softmax_cross_entropy_loss_create();

    boat_tensor_t* grad = boat_loss_backward(loss, logits, target);
    assert(grad != NULL);
    float* gd = (float*)boat_tensor_data(grad);

    // Numerical gradient check per element
    float* l = (float*)boat_tensor_data(logits);
    float eps = 1e-4f;
    int n = 6;
    for (int i = 0; i < n; i++) {
        float orig = l[i];
        l[i] = orig + eps;
        float lp = boat_loss_compute(loss, logits, target);
        l[i] = orig - eps;
        float lm = boat_loss_compute(loss, logits, target);
        l[i] = orig;
        float num = (lp - lm) / (2.0f * eps);
        assert(fabsdiff(gd[i], num) < 1e-3);
    }

    boat_tensor_free(grad);
    boat_loss_free(loss);
    boat_tensor_free(logits);
    boat_tensor_free(target);
    printf("  OK\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Softmax Cross-Entropy Tests ===\n\n");
    test_forward();
    test_backward_gradient_check();
    printf("\n=== All softmax CE tests passed ===\n");
    return 0;
}
