// test_matmul_softmax.c - matmul batch broadcasting + softmax/log_softmax axis
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/autodiff.h>
#include <boat/ops.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

static float fd(float a, float b) { float d = a - b; return d < 0 ? -d : d; }

static void test_matmul_broadcast_forward(void) {
    printf("Testing matmul batch broadcast (forward)...\n");
    int64_t ash[] = {2, 3, 4}, bsh[] = {4, 5};
    float ad[24], bd[20];
    for (int i = 0; i < 24; i++) ad[i] = (float)(i + 1);
    for (int i = 0; i < 20; i++) bd[i] = (float)(i % 5 + 1);

    boat_tensor_t* a = boat_tensor_from_data(ash, 3, BOAT_DTYPE_FLOAT32, ad);
    boat_tensor_t* b = boat_tensor_from_data(bsh, 2, BOAT_DTYPE_FLOAT32, bd);
    boat_tensor_t* c = boat_matmul(a, b);
    assert(c && boat_tensor_ndim(c) == 3);
    assert(boat_tensor_shape(c)[0] == 2 && boat_tensor_shape(c)[1] == 3 && boat_tensor_shape(c)[2] == 5);

    float* cd = (float*)boat_tensor_data(c);
    for (int bb = 0; bb < 2; bb++)
        for (int t = 0; t < 3; t++)
            for (int j = 0; j < 5; j++) {
                float s = 0;
                for (int l = 0; l < 4; l++) s += ad[(bb * 3 + t) * 4 + l] * bd[l * 5 + j];
                assert(fd(cd[(bb * 3 + t) * 5 + j], s) < 1e-4);
            }
    boat_tensor_free(a); boat_tensor_free(b); boat_tensor_free(c);
    printf("  OK\n");
}

// scalar loss = sum(a @ b) computed with the tensor-level matmul.
static float matmul_loss(const float* a_data, const float* b_data) {
    int64_t ash[] = {2, 2, 3}, bsh[] = {3, 2};
    boat_tensor_t* a = boat_tensor_from_data(ash, 3, BOAT_DTYPE_FLOAT32, a_data);
    boat_tensor_t* b = boat_tensor_from_data(bsh, 2, BOAT_DTYPE_FLOAT32, b_data);
    boat_tensor_t* c = boat_matmul(a, b);
    float sum = 0;
    float* cd = (float*)boat_tensor_data(c);
    for (size_t i = 0; i < boat_tensor_nelements(c); i++) sum += cd[i];
    boat_tensor_free(a); boat_tensor_free(b); boat_tensor_free(c);
    return sum;
}

static void test_matmul_broadcast_backward(void) {
    printf("Testing matmul batch broadcast (backward, numerical)...\n");
    int64_t ash[] = {2, 2, 3}, bsh[] = {3, 2};
    float av[] = {1,2,3, 4,5,6,  7,8,9, 10,11,12};
    float bv[] = {0.5f,1.0f, 1.5f,2.0f, 2.5f,3.0f};

    boat_variable_t* a = boat_variable_create_with_shape(ash, 3, BOAT_DTYPE_FLOAT32, true);
    boat_variable_t* b = boat_variable_create_with_shape(bsh, 2, BOAT_DTYPE_FLOAT32, true);
    float* ad = (float*)boat_tensor_data(boat_variable_data(a));
    float* bd = (float*)boat_tensor_data(boat_variable_data(b));
    for (int i = 0; i < 12; i++) ad[i] = av[i];
    for (int i = 0; i < 6; i++) bd[i] = bv[i];

    boat_variable_t* c = boat_var_matmul(a, b);
    // loss = sum(c); backward with ones grad.
    int64_t gsh[] = {2, 2, 2};
    boat_tensor_t* gout = boat_tensor_create(gsh, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* gd = (float*)boat_tensor_data(gout);
    for (int i = 0; i < 8; i++) gd[i] = 1.0f;
    boat_variable_backward(c, gout);

    float* ga = (float*)boat_tensor_data(boat_variable_grad(a));
    float* gb = (float*)boat_tensor_data(boat_variable_grad(b));

    float eps = 1e-3f;
    // numeric grad of sum(a@b) w.r.t. each element of a
    float a_copy[12], b_copy[6];
    for (int i = 0; i < 12; i++) a_copy[i] = av[i];
    for (int i = 0; i < 6; i++) b_copy[i] = bv[i];
    for (int i = 0; i < 12; i++) {
        float o = a_copy[i];
        a_copy[i] = o + eps; float lp = matmul_loss(a_copy, b_copy);
        a_copy[i] = o - eps; float lm = matmul_loss(a_copy, b_copy);
        a_copy[i] = o;
        assert(fd(ga[i], (lp - lm) / (2 * eps)) < 5e-2);
    }
    for (int i = 0; i < 6; i++) {
        float o = b_copy[i];
        b_copy[i] = o + eps; float lp = matmul_loss(a_copy, b_copy);
        b_copy[i] = o - eps; float lm = matmul_loss(a_copy, b_copy);
        b_copy[i] = o;
        assert(fd(gb[i], (lp - lm) / (2 * eps)) < 5e-2);
    }

    boat_tensor_free(gout);
    boat_variable_free(c);
    boat_variable_free(a);
    boat_variable_free(b);
    printf("  OK\n");
}

// scalar loss = sum(softmax(x, axis)) with fixed weights w.
static float sm_loss(const float* x, int n, int axis, const float* w) {
    int64_t sh[] = {2, 3};
    boat_tensor_t* t = boat_tensor_from_data(sh, 2, BOAT_DTYPE_FLOAT32, x);
    boat_tensor_t* s = boat_softmax(t, axis);
    float sum = 0;
    float* sd = (float*)boat_tensor_data(s);
    for (int i = 0; i < n; i++) sum += sd[i] * w[i];
    boat_tensor_free(t); boat_tensor_free(s);
    return sum;
}

static void test_softmax_axis(void) {
    printf("Testing softmax/log_softmax axis...\n");
    int64_t sh[] = {2, 3};
    float vals[] = {1, 2, 3, 1, 0, 1};

    // Forward: softmax over axis 0 (columns).
    boat_tensor_t* t = boat_tensor_from_data(sh, 2, BOAT_DTYPE_FLOAT32, vals);
    boat_tensor_t* s = boat_softmax(t, 0);
    float* sd = (float*)boat_tensor_data(s);
    assert(fd(sd[0], 0.5f) < 1e-4 && fd(sd[3], 0.5f) < 1e-4);
    assert(fd(sd[1], 0.880797f) < 1e-3 && fd(sd[4], 0.119203f) < 1e-3);
    boat_tensor_free(s); boat_tensor_free(t);

    // Forward: log_softmax over axis 1 (rows).
    t = boat_tensor_from_data(sh, 2, BOAT_DTYPE_FLOAT32, vals);
    boat_tensor_t* ls = boat_log_softmax(t, 1);
    float* lsd = (float*)boat_tensor_data(ls);
    assert(fd(lsd[0], -2.407606f) < 1e-3);
    assert(fd(lsd[1], -1.407606f) < 1e-3);
    assert(fd(lsd[2], -0.407606f) < 1e-3);
    boat_tensor_free(ls); boat_tensor_free(t);

    // Backward: numerical gradient of sum(softmax(x, axis=0) * w).
    boat_variable_t* x = boat_variable_create_with_shape(sh, 2, BOAT_DTYPE_FLOAT32, true);
    float* xd = (float*)boat_tensor_data(boat_variable_data(x));
    for (int i = 0; i < 6; i++) xd[i] = vals[i];
    boat_variable_t* sv = boat_var_softmax(x, 0);
    // grad_output = weights w (so grad = d(sum(softmax*w))/dx)
    float w[] = {1, -2, 0.5, 3, 0.5, -1};
    int64_t wsh[] = {2, 3};
    boat_tensor_t* wt = boat_tensor_from_data(wsh, 2, BOAT_DTYPE_FLOAT32, w);
    boat_variable_backward(sv, wt);
    float* g = (float*)boat_tensor_data(boat_variable_grad(x));

    float x_copy[6];
    for (int i = 0; i < 6; i++) x_copy[i] = vals[i];
    float eps = 1e-3f;
    for (int i = 0; i < 6; i++) {
        float o = x_copy[i];
        x_copy[i] = o + eps; float lp = sm_loss(x_copy, 6, 0, w);
        x_copy[i] = o - eps; float lm = sm_loss(x_copy, 6, 0, w);
        x_copy[i] = o;
        assert(fd(g[i], (lp - lm) / (2 * eps)) < 5e-2);
    }

    boat_tensor_free(wt);
    boat_variable_free(sv);
    boat_variable_free(x);
    printf("  OK\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Matmul Broadcast + Softmax Axis Tests ===\n\n");
    test_matmul_broadcast_forward();
    test_matmul_broadcast_backward();
    test_softmax_axis();
    printf("\n=== All matmul/softmax tests passed ===\n");
    return 0;
}
