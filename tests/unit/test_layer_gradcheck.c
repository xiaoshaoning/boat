// test_layer_gradcheck.c - Numerical gradient checks for layer backward passes
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0
//
// Covers dense (grad_input/weight/bias) and max-pool (grad_input) backward,
// which previously had no gradient-check coverage.

#include <boat/layers.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

static int close_enough(float a, float b) {
    float d = fabsf(a - b);
    float scale = fabsf(a) + fabsf(b);
    return d < 5e-2f * scale + 1e-3f;
}

// Sum of all output elements after a forward pass.
static float dense_sum(boat_dense_layer_t* layer, boat_tensor_t* input) {
    boat_tensor_t* out = boat_dense_layer_forward(layer, input);
    assert(out != NULL);
    float sum = 0.0f;
    float* od = (float*)boat_tensor_data(out);
    for (size_t i = 0; i < boat_tensor_nelements(out); i++) sum += od[i];
    boat_tensor_free(out);
    return sum;
}

static float pool_sum(boat_pool_layer_t* layer, boat_tensor_t* input) {
    boat_tensor_t* out = boat_pool_layer_forward(layer, input);
    assert(out != NULL);
    float sum = 0.0f;
    float* od = (float*)boat_tensor_data(out);
    for (size_t i = 0; i < boat_tensor_nelements(out); i++) sum += od[i];
    boat_tensor_free(out);
    return sum;
}

static void fill_ones(boat_tensor_t* t) {
    float* d = (float*)boat_tensor_data(t);
    for (size_t i = 0; i < boat_tensor_nelements(t); i++) d[i] = 1.0f;
}

static void test_dense_gradcheck(void) {
    printf("Testing dense backward (numerical)...\n");
    size_t in = 4, out_features = 3, batch = 2;
    boat_dense_layer_t* layer = boat_dense_layer_create(in, out_features, true);
    assert(layer != NULL);

    int64_t ish[] = {(int64_t)batch, (int64_t)in};
    boat_tensor_t* input = boat_tensor_create(ish, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* id = (float*)boat_tensor_data(input);
    float iv[] = {0.5f, -1.0f, 2.0f, 0.0f,  1.5f, 0.25f, -0.5f, 3.0f};
    for (int i = 0; i < 8; i++) id[i] = iv[i];

    // Forward then backward with grad_output = ones.
    boat_tensor_t* out = boat_dense_layer_forward(layer, input);
    assert(out != NULL);
    int64_t osh[] = {(int64_t)batch, (int64_t)out_features};
    boat_tensor_t* grad_out = boat_tensor_create(osh, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_ones(grad_out);
    boat_tensor_t* grad_in = boat_dense_layer_backward(layer, grad_out);
    assert(grad_in != NULL);
    float* gi = (float*)boat_tensor_data(grad_in);
    float* gw = (float*)boat_tensor_data(boat_dense_layer_get_grad_weight(layer));
    float* gb = (float*)boat_tensor_data(boat_dense_layer_get_grad_bias(layer));
    boat_tensor_free(out);

    // Numerical gradient w.r.t. input.
    float eps = 1e-3f;
    for (int i = 0; i < 8; i++) {
        float o = id[i];
        id[i] = o + eps; float lp = dense_sum(layer, input);
        id[i] = o - eps; float lm = dense_sum(layer, input);
        id[i] = o;
        assert(close_enough(gi[i], (lp - lm) / (2 * eps)));
    }

    // Numerical gradient w.r.t. weight.
    boat_tensor_t* wt = boat_dense_layer_get_weight(layer);
    float* wd = (float*)boat_tensor_data(wt);
    size_t wn = boat_tensor_nelements(wt);
    for (size_t i = 0; i < wn; i++) {
        float o = wd[i];
        wd[i] = o + eps; float lp = dense_sum(layer, input);
        wd[i] = o - eps; float lm = dense_sum(layer, input);
        wd[i] = o;
        assert(close_enough(gw[i], (lp - lm) / (2 * eps)));
    }

    // Numerical gradient w.r.t. bias.
    boat_tensor_t* bt = boat_dense_layer_get_bias(layer);
    float* bd = (float*)boat_tensor_data(bt);
    size_t bn = boat_tensor_nelements(bt);
    for (size_t i = 0; i < bn; i++) {
        float o = bd[i];
        bd[i] = o + eps; float lp = dense_sum(layer, input);
        bd[i] = o - eps; float lm = dense_sum(layer, input);
        bd[i] = o;
        assert(close_enough(gb[i], (lp - lm) / (2 * eps)));
    }

    boat_tensor_free(grad_out);
    boat_tensor_free(grad_in);
    boat_tensor_free(input);
    boat_dense_layer_free(layer);
    printf("  OK\n");
}

static void test_maxpool_gradcheck(void) {
    printf("Testing max-pool backward (numerical)...\n");
    boat_pool_layer_t* layer = boat_pool_layer_create(2, 2, 0);
    assert(layer != NULL);

    int64_t ish[] = {1, 1, 4, 4};
    boat_tensor_t* input = boat_tensor_create(ish, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* id = (float*)boat_tensor_data(input);
    // 4x4 with distinct values so argmax is unambiguous.
    float iv[] = {1,2,3,4,  5,6,7,8,  9,10,11,12,  13,14,15,16};
    for (int i = 0; i < 16; i++) id[i] = iv[i];

    boat_tensor_t* out = boat_pool_layer_forward(layer, input);
    assert(out != NULL);
    int64_t osh[] = {1, 1, 2, 2};
    boat_tensor_t* grad_out = boat_tensor_create(osh, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_ones(grad_out);
    boat_tensor_t* grad_in = boat_pool_layer_backward(layer, grad_out);
    assert(grad_in != NULL);
    float* gi = (float*)boat_tensor_data(grad_in);
    boat_tensor_free(out);

    // Max pool forward value check: output should be [6, 8, 14, 16].
    boat_tensor_t* fwd_out = boat_pool_layer_forward(layer, input);
    float* od = (float*)boat_tensor_data(fwd_out);
    assert(od[0] == 6 && od[1] == 8 && od[2] == 14 && od[3] == 16);
    boat_tensor_free(fwd_out);

    float eps = 1e-3f;
    for (int i = 0; i < 16; i++) {
        float o = id[i];
        id[i] = o + eps; float lp = pool_sum(layer, input);
        id[i] = o - eps; float lm = pool_sum(layer, input);
        id[i] = o;
        assert(close_enough(gi[i], (lp - lm) / (2 * eps)));
    }

    boat_tensor_free(grad_out);
    boat_tensor_free(grad_in);
    boat_tensor_free(input);
    boat_pool_layer_free(layer);
    printf("  OK\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Layer Backward Gradient-Check Tests ===\n\n");
    test_dense_gradcheck();
    test_maxpool_gradcheck();
    printf("\n=== All layer gradient checks passed ===\n");
    return 0;
}
