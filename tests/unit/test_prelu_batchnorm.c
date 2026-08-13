// test_prelu_batchnorm.c - Numerical gradient checks for PReLU and BatchNorm2d
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

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

static float prelu_sum(boat_prelu_layer_t* layer, boat_tensor_t* input) {
    boat_tensor_t* out = boat_prelu_layer_forward(layer, input);
    assert(out != NULL);
    float s = 0; float* od = (float*)boat_tensor_data(out);
    for (size_t i = 0; i < boat_tensor_nelements(out); i++) s += od[i];
    boat_tensor_free(out);
    return s;
}

static float bn_sum(boat_batchnorm2d_layer_t* layer, boat_tensor_t* input) {
    boat_tensor_t* out = boat_batchnorm2d_layer_forward(layer, input);
    assert(out != NULL);
    float s = 0; float* od = (float*)boat_tensor_data(out);
    for (size_t i = 0; i < boat_tensor_nelements(out); i++) s += od[i];
    boat_tensor_free(out);
    return s;
}

static void test_prelu_gradcheck(void) {
    printf("Testing PReLU backward (numerical)...\n");
    boat_prelu_layer_t* layer = boat_prelu_layer_create(2);
    assert(layer != NULL);

    int64_t ish[] = {1, 2, 2, 2};
    boat_tensor_t* input = boat_tensor_create(ish, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* id = (float*)boat_tensor_data(input);
    float iv[] = {1.0f, -2.0f, 3.0f, -4.0f,  -1.0f, 2.0f, -3.0f, 4.0f};
    for (int i = 0; i < 8; i++) id[i] = iv[i];

    // Set per-channel slopes [0.25, 0.5].
    int64_t ssh[] = {2, 1, 1};
    float sv[] = {0.25f, 0.5f};
    boat_tensor_t* slope = boat_tensor_from_data(ssh, 3, BOAT_DTYPE_FLOAT32, sv);
    boat_prelu_layer_set_slope(layer, slope);
    boat_tensor_free(slope);

    // Forward then backward with ones.
    boat_tensor_t* out = boat_prelu_layer_forward(layer, input);
    assert(out != NULL);
    int64_t osh[] = {1, 2, 2, 2};
    boat_tensor_t* grad_out = boat_tensor_create(osh, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* gd = (float*)boat_tensor_data(grad_out);
    for (int i = 0; i < 8; i++) gd[i] = 1.0f;
    boat_tensor_t* grad_in = boat_prelu_layer_backward(layer, grad_out);
    assert(grad_in != NULL);
    float* gi = (float*)boat_tensor_data(grad_in);
    float* gslope = (float*)boat_tensor_data(boat_prelu_layer_get_grad_slope(layer));
    boat_tensor_free(out);

    // Numerical gradient w.r.t. input.
    float eps = 1e-3f;
    for (int i = 0; i < 8; i++) {
        float o = id[i];
        id[i] = o + eps; float lp = prelu_sum(layer, input);
        id[i] = o - eps; float lm = prelu_sum(layer, input);
        id[i] = o;
        assert(close_enough(gi[i], (lp - lm) / (2 * eps)));
    }

    // Numerical gradient w.r.t. slope.
    boat_tensor_t* sl = boat_prelu_layer_get_slope(layer);
    float* sd = (float*)boat_tensor_data(sl);
    for (int c = 0; c < 2; c++) {
        float o = sd[c];
        sd[c] = o + eps; float lp = prelu_sum(layer, input);
        sd[c] = o - eps; float lm = prelu_sum(layer, input);
        sd[c] = o;
        assert(close_enough(gslope[c], (lp - lm) / (2 * eps)));
    }

    boat_tensor_free(grad_out);
    boat_tensor_free(grad_in);
    boat_tensor_free(input);
    boat_prelu_layer_free(layer);
    printf("  OK\n");
}

static void test_batchnorm_gradcheck(void) {
    printf("Testing BatchNorm2d backward (numerical, training mode)...\n");
    size_t C = 2;
    boat_batchnorm2d_layer_t* layer = boat_batchnorm2d_layer_create(C, 1e-5f, 0.1f, true);
    assert(layer != NULL);
    boat_batchnorm2d_layer_set_training(layer, true);

    int64_t ish[] = {2, (int64_t)C, 2, 2};  // [N, C, H, W] = [2, 2, 2, 2]
    boat_tensor_t* input = boat_tensor_create(ish, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* id = (float*)boat_tensor_data(input);
    float iv[] = {1, 2, 3, 4,  5, 6, 7, 8,   2, 4, 6, 8,  10, 12, 14, 16};
    for (int i = 0; i < 16; i++) id[i] = iv[i];

    // Set weight (gamma) and bias (beta) to non-trivial values.
    int64_t wsh[] = {(int64_t)C};
    float wv[] = {1.5f, 0.5f};
    boat_tensor_t* wt = boat_tensor_from_data(wsh, 1, BOAT_DTYPE_FLOAT32, wv);
    float bv[] = {0.5f, -0.5f};
    boat_tensor_t* bt = boat_tensor_from_data(wsh, 1, BOAT_DTYPE_FLOAT32, bv);
    boat_batchnorm2d_layer_set_weight(layer, wt);
    boat_batchnorm2d_layer_set_bias(layer, bt);
    boat_tensor_free(wt);
    boat_tensor_free(bt);

    boat_tensor_t* out = boat_batchnorm2d_layer_forward(layer, input);
    assert(out != NULL);
    int64_t osh[] = {2, (int64_t)C, 2, 2};
    boat_tensor_t* grad_out = boat_tensor_create(osh, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* gd = (float*)boat_tensor_data(grad_out);
    for (int i = 0; i < 16; i++) gd[i] = 1.0f;
    boat_tensor_t* grad_in = boat_batchnorm2d_layer_backward(layer, grad_out);
    assert(grad_in != NULL);
    float* gi = (float*)boat_tensor_data(grad_in);
    float* gw = (float*)boat_tensor_data(boat_batchnorm2d_layer_get_grad_weight(layer));
    float* gb = (float*)boat_tensor_data(boat_batchnorm2d_layer_get_grad_bias(layer));
    boat_tensor_free(out);

    // Numerical gradient w.r.t. input.
    float eps = 1e-3f;
    for (int i = 0; i < 16; i++) {
        float o = id[i];
        id[i] = o + eps; float lp = bn_sum(layer, input);
        id[i] = o - eps; float lm = bn_sum(layer, input);
        id[i] = o;
        assert(close_enough(gi[i], (lp - lm) / (2 * eps)));
    }

    // Numerical gradient w.r.t. gamma and beta.
    float* wd = (float*)boat_tensor_data(boat_batchnorm2d_layer_get_weight(layer));
    float* bd = (float*)boat_tensor_data(boat_batchnorm2d_layer_get_bias(layer));
    for (int c = 0; c < 2; c++) {
        float o = wd[c];
        wd[c] = o + eps; float lp = bn_sum(layer, input);
        wd[c] = o - eps; float lm = bn_sum(layer, input);
        wd[c] = o;
        assert(close_enough(gw[c], (lp - lm) / (2 * eps)));

        o = bd[c];
        bd[c] = o + eps; lp = bn_sum(layer, input);
        bd[c] = o - eps; lm = bn_sum(layer, input);
        bd[c] = o;
        assert(close_enough(gb[c], (lp - lm) / (2 * eps)));
    }

    boat_tensor_free(grad_out);
    boat_tensor_free(grad_in);
    boat_tensor_free(input);
    boat_batchnorm2d_layer_free(layer);
    printf("  OK\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== PReLU + BatchNorm2d Gradient-Check Tests ===\n\n");
    test_prelu_gradcheck();
    test_batchnorm_gradcheck();
    printf("\n=== All PReLU/BatchNorm tests passed ===\n");
    return 0;
}
