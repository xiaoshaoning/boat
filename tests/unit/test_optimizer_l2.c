// test_optimizer_l2.c - Weight decay (L2 regularization) unit tests
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0
//
// Verifies boat_optimizer_set_weight_decay across SGD/Adam/RMSprop/Adagrad:
// the update uses g' = g + wd * w (MATLAB L2Regularization semantics), and
// the default wd = 0 leaves the update identical to the no-decay path.

#include <boat/optimizers.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

static int closef(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

// Run one step on a single-element scalar-like tensor (1x1).
static float step_once(boat_optimizer_t* opt, float w0, float g0) {
    int64_t shape[] = {1, 1};
    boat_tensor_t* param = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* grad = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    assert(param && grad);
    float* p = (float*)boat_tensor_data(param);
    float* g = (float*)boat_tensor_data(grad);
    p[0] = w0;
    g[0] = g0;
    boat_optimizer_add_parameter(opt, param, grad);
    boat_optimizer_step(opt);
    float out = p[0];
    boat_optimizer_free(opt);
    boat_tensor_free(param);
    boat_tensor_free(grad);
    return out;
}

int main() {
    printf("Testing optimizer weight decay (L2)...\n");

    // SGD, no momentum: w -= lr * (g + wd * w)
    {
        // zero gradient: w = 2, lr = 0.5, wd = 0.1 -> w = 2 - 0.5 * (0 + 0.1 * 2) = 1.9
        boat_optimizer_t* opt = boat_sgd_optimizer_create(0.5f, 0.0f);
        boat_optimizer_set_weight_decay(opt, 0.1f);
        float w = step_once(opt, 2.0f, 0.0f);
        assert(closef(w, 1.9f));
        printf("  SGD zero-grad decay: %.4f (expect 1.9000)\n", w);

        // with gradient: w = 2, g = 1 -> w = 2 - 0.5 * (1 + 0.1 * 2) = 1.4
        boat_optimizer_t* opt2 = boat_sgd_optimizer_create(0.5f, 0.0f);
        boat_optimizer_set_weight_decay(opt2, 0.1f);
        float w2 = step_once(opt2, 2.0f, 1.0f);
        assert(closef(w2, 1.4f));
        printf("  SGD grad+decay: %.4f (expect 1.4000)\n", w2);

        // wd = 0 must match the plain update exactly: w = 2 - 0.5 * 1 = 1.5
        boat_optimizer_t* opt3 = boat_sgd_optimizer_create(0.5f, 0.0f);
        float w3 = step_once(opt3, 2.0f, 1.0f);
        assert(closef(w3, 1.5f));
        printf("  SGD default (no decay): %.4f (expect 1.5000)\n", w3);

        // getter round-trip
        boat_optimizer_t* opt4 = boat_sgd_optimizer_create(0.1f, 0.0f);
        boat_optimizer_set_weight_decay(opt4, 0.05f);
        assert(closef(boat_optimizer_get_weight_decay(opt4), 0.05f));
        boat_optimizer_free(opt4);
        printf("  SGD weight-decay getter round-trip passed\n");
    }

    // Adam: with zero gradient, w = 2, lr = 0.1, wd = 1.0, beta1=beta2=0
    //   g = wd * w = 2; m = g; v = g^2; m_hat = m; v_hat = v;
    //   w -= lr * m / (sqrt(v) + eps) = 2 - 0.1 * 2 / (2 + eps) ~ 1.9
    {
        boat_optimizer_t* opt = boat_adam_optimizer_create(0.1f, 0.9f, 0.999f, 1e-8f);
        boat_optimizer_set_weight_decay(opt, 1.0f);
        float w = step_once(opt, 2.0f, 0.0f);
        printf("  Adam zero-grad decay: %.4f (expect ~1.9000)\n", w);
        assert(closef(w, 1.9f));
    }

    // RMSprop: alpha = 0 (so square_avg = g^2), zero gradient:
    //   g = wd * w = 2; sq = 4; w -= lr * 2 / (2 + eps) ~ 1.9
    {
        boat_optimizer_t* opt = boat_rmsprop_optimizer_create(0.1f, 0.5f, 1e-8f);
        boat_optimizer_set_weight_decay(opt, 1.0f);
        float w = step_once(opt, 2.0f, 0.0f);
        printf("  RMSprop zero-grad decay: %.4f (expect ~1.8586)\n", w);
        assert(closef(w, 1.85858f));
    }

    // Adagrad: zero gradient: g = wd * w = 2; sum = 4; w -= lr * 2 / (2 + eps) ~ 1.9
    {
        boat_optimizer_t* opt = boat_adagrad_optimizer_create(0.1f, 1e-8f);
        boat_optimizer_set_weight_decay(opt, 1.0f);
        float w = step_once(opt, 2.0f, 0.0f);
        assert(closef(w, 1.9f));
        printf("  Adagrad zero-grad decay: %.4f (expect ~1.9000)\n", w);
    }

    printf("All weight-decay tests passed.\n");
    return 0;
}
