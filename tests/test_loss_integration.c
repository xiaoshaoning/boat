// test_loss_integration.c - Loss function forward, backward and training integration tests
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <boat/loss.h>
#include <boat/layers.h>
#include <boat/optimizers.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Test infrastructure
static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { printf("  %s ... ", name); tests_total++; fflush(stdout); } while(0)
#define PASS() do { printf("PASS\n"); fflush(stdout); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); fflush(stdout); return 1; } while(0)

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

static void fill_tensor_linear(boat_tensor_t* t, float base) {
    float* d = (float*)boat_tensor_data(t);
    size_t n = boat_tensor_nelements(t);
    for (size_t i = 0; i < n; i++) {
        d[i] = base + (float)i;
    }
}

static void fill_tensor_rand(boat_tensor_t* t, unsigned int seed) {
    srand(seed);
    float* d = (float*)boat_tensor_data(t);
    size_t n = boat_tensor_nelements(t);
    for (size_t i = 0; i < n; i++) {
        d[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

static void fill_tensor_one_hot(boat_tensor_t* t, const int* labels, int num_labels) {
    float* d = (float*)boat_tensor_data(t);
    size_t n = boat_tensor_nelements(t);
    memset(d, 0, n * sizeof(float));
    int batch_size = (int)(n / num_labels);
    for (int i = 0; i < batch_size; i++) {
        int label = labels[i];
        if (label >= 0 && label < num_labels) {
            d[i * num_labels + label] = 1.0f;
        }
    }
}

static float compute_numerical_loss_gradient(
    boat_loss_t* loss, boat_tensor_t* pred, const boat_tensor_t* target,
    size_t idx, float epsilon)
{
    float* data = (float*)boat_tensor_data(pred);
    float original = data[idx];

    data[idx] = original + epsilon;
    float loss_plus = boat_loss_compute(loss, pred, target);

    data[idx] = original - epsilon;
    float loss_minus = boat_loss_compute(loss, pred, target);

    data[idx] = original;
    return (loss_plus - loss_minus) / (2.0f * epsilon);
}

static float rand_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

// ---------------------------------------------------------------------------
// MSE Tests
// ---------------------------------------------------------------------------

static int test_mse_forward(void) {
    TEST("MSE forward compute");

    int64_t shape1[] = {3};
    int64_t shape2[] = {2, 2};

    // Test 1: pred=[2,4,6], target=[1,3,5] => MSE = (1+1+1)/3 = 1.0
    boat_tensor_t* p1 = boat_tensor_create(shape1, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* t1 = boat_tensor_create(shape1, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor_linear(p1, 2.0f);  // [2, 3, 4] wait, linear starts at base so [2,3,4]
    // Actually fill_tensor_linear fills with base + i, so base=2 => [2,3,4]
    // Let's set manually
    {
        float* pd = (float*)boat_tensor_data(p1);
        pd[0] = 2.0f; pd[1] = 4.0f; pd[2] = 6.0f;
        float* td = (float*)boat_tensor_data(t1);
        td[0] = 1.0f; td[1] = 3.0f; td[2] = 5.0f;
    }

    boat_loss_t* mse = boat_mse_loss_create();
    float loss_val = boat_loss_compute(mse, p1, t1);
    if (fabsf(loss_val - 1.0f) > 1e-5f) {
        boat_tensor_unref(p1); boat_tensor_unref(t1); boat_loss_free(mse);
        FAIL("MSE [2,4,6]/[1,3,5] not 1.0");
    }

    // Test 2: pred=[1,2,3,4], target=[0,0,0,0] => MSE = (1+4+9+16)/4 = 7.5
    int64_t shape4[] = {4};
    boat_tensor_t* p2 = boat_tensor_create(shape4, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* t2 = boat_tensor_create(shape4, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    {
        float* pd = (float*)boat_tensor_data(p2);
        pd[0] = 1.0f; pd[1] = 2.0f; pd[2] = 3.0f; pd[3] = 4.0f;
        float* td = (float*)boat_tensor_data(t2);
        td[0] = 0.0f; td[1] = 0.0f; td[2] = 0.0f; td[3] = 0.0f;
    }
    loss_val = boat_loss_compute(mse, p2, t2);
    if (fabsf(loss_val - 7.5f) > 1e-5f) {
        boat_tensor_unref(p1); boat_tensor_unref(t1);
        boat_tensor_unref(p2); boat_tensor_unref(t2); boat_loss_free(mse);
        FAIL("MSE [1,2,3,4]/zeros not 7.5");
    }
    boat_tensor_unref(p2); boat_tensor_unref(t2);

    // Test 3: 2D, pred=[[1,2],[3,4]], target=[[1,2],[3,4]] => MSE = 0
    boat_tensor_t* p3 = boat_tensor_create(shape2, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* t3 = boat_tensor_create(shape2, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor_linear(p3, 1.0f);
    fill_tensor_linear(t3, 1.0f);
    loss_val = boat_loss_compute(mse, p3, t3);
    if (fabsf(loss_val) > 1e-6f) {
        boat_tensor_unref(p1); boat_tensor_unref(t1);
        boat_tensor_unref(p3); boat_tensor_unref(t3); boat_loss_free(mse);
        FAIL("MSE identical tensors not 0");
    }

    boat_tensor_unref(p1); boat_tensor_unref(t1);
    boat_tensor_unref(p3); boat_tensor_unref(t3); boat_loss_free(mse);
    PASS();
    return 0;
}

static int test_mse_backward_gradient(void) {
    TEST("MSE backward gradient check");

    int64_t shape[] = {2, 3};
    boat_tensor_t* pred = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* target = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor_rand(pred, 42);
    fill_tensor_rand(target, 123);

    boat_loss_t* mse = boat_mse_loss_create();

    // Compute analytical gradient
    boat_tensor_t* grad = boat_loss_backward(mse, pred, target);
    if (!grad) {
        boat_tensor_unref(pred); boat_tensor_unref(target);
        boat_loss_free(mse);
        FAIL("MSE backward returned NULL");
    }

    // Numerical gradient check on each element
    float* grad_data = (float*)boat_tensor_data(grad);
    size_t n = boat_tensor_nelements(pred);
    float eps = 1e-4f;
    int mismatches = 0;

    for (size_t i = 0; i < n; i++) {
        float numerical = compute_numerical_loss_gradient(mse, pred, target, i, eps);
        float analytical = grad_data[i];
        float abs_err = fabsf(analytical - numerical);
        float rel_err = (fabsf(analytical) + fabsf(numerical)) > 1e-10f
            ? fabsf(analytical - numerical) / (fabsf(analytical) + fabsf(numerical))
            : 0.0f;
        if (abs_err > 1e-3f && rel_err > 1e-3f) {
            if (mismatches < 3) {
                printf("\n        [%zu] analytical=%.6f numerical=%.6f", i, analytical, numerical);
            }
            mismatches++;
        }
    }

    boat_tensor_unref(grad);
    boat_tensor_unref(pred); boat_tensor_unref(target);
    boat_loss_free(mse);

    if (mismatches > 0) {
        FAIL("MSE gradient mismatch");
    }
    PASS();
    return 0;
}

static int test_mse_training(void) {
    TEST("MSE training loop");

    // Create model: Dense(1->8) -> ReLU -> Dense(8->1)
    boat_dense_layer_t* fc1 = boat_dense_layer_create(1, 8, 1);
    boat_relu_layer_t* relu = boat_relu_layer_create();
    boat_dense_layer_t* fc2 = boat_dense_layer_create(8, 1, 1);

    // Training data: y = 2x + 1, 32 samples
    int batch_size = 32;
    int64_t x_shape[] = {batch_size, 1};
    int64_t y_shape[] = {batch_size, 1};
    boat_tensor_t* x = boat_tensor_create(x_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* y = boat_tensor_create(y_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* xd = (float*)boat_tensor_data(x);
    float* yd = (float*)boat_tensor_data(y);
    for (int i = 0; i < batch_size; i++) {
        xd[i] = (float)i / (float)batch_size;
        yd[i] = 2.0f * xd[i] + 1.0f;
    }

    boat_loss_t* mse = boat_mse_loss_create();

    int n_epochs = 50;
    float lr = 0.02f;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        // Forward
        boat_tensor_t* a1 = boat_dense_layer_forward(fc1, x);
        boat_tensor_t* a2 = boat_relu_layer_forward(relu, a1);
        boat_tensor_t* out = boat_dense_layer_forward(fc2, a2);

        float loss = boat_loss_compute(mse, out, y);

        if (epoch == 0) initial_loss = loss;
        if (epoch == n_epochs - 1) final_loss = loss;

        // Backward
        boat_tensor_t* grad = boat_loss_backward(mse, out, y);
        if (grad) {
            boat_tensor_t* g = grad;
            boat_tensor_t* t;
            t = boat_dense_layer_backward(fc2, g);
            if (t) { boat_tensor_unref(g); g = t; } else { boat_tensor_unref(g); break; }
            t = boat_relu_layer_backward(relu, g);
            if (t) { boat_tensor_unref(g); g = t; } else { boat_tensor_unref(g); break; }
            t = boat_dense_layer_backward(fc1, g);
            if (t) { boat_tensor_unref(g); g = t; } else { boat_tensor_unref(g); break; }
            boat_tensor_unref(g);
        }

        // Update
        boat_dense_layer_update(fc1, lr);
        boat_dense_layer_update(fc2, lr);

        boat_tensor_unref(out);
        boat_tensor_unref(a2);
        boat_tensor_unref(a1);
    }

    boat_loss_free(mse);
    boat_tensor_unref(x); boat_tensor_unref(y);
    boat_dense_layer_free(fc2);
    boat_relu_layer_free(relu);
    boat_dense_layer_free(fc1);

    if (final_loss >= initial_loss) {
        FAIL("MSE training did not reduce loss");
    }
    if (final_loss > 0.05f) {
        FAIL("MSE final loss too high");
    }
    PASS();
    return 0;
}

// ---------------------------------------------------------------------------
// Cross-Entropy Tests
// ---------------------------------------------------------------------------

static int test_ce_forward(void) {
    TEST("CrossEntropy forward compute");

    int64_t shape[] = {1, 2};
    boat_tensor_t* pred = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* target = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* pd = (float*)boat_tensor_data(pred);
    float* td = (float*)boat_tensor_data(target);
    boat_loss_t* ce = boat_cross_entropy_loss_create();

    // Test 1: pred=[0.5, 0.5], target=[1, 0], N=2
    // CE = -(1*log(0.5) + 0*log(0.5))/2 = -log(0.5)/2 = 0.3466
    pd[0] = 0.5f; pd[1] = 0.5f;
    td[0] = 1.0f; td[1] = 0.0f;
    float loss = boat_loss_compute(ce, pred, target);
    float expected = -logf(0.5f) / 2.0f;
    if (fabsf(loss - expected) > 1e-5f) {
        boat_tensor_unref(pred); boat_tensor_unref(target); boat_loss_free(ce);
        FAIL("CE [0.5,0.5]/[1,0] wrong");
    }

    // Test 2: pred=[0.9, 0.1], target=[1, 0], N=2
    // CE = -(1*log(0.9) + 0*log(0.1))/2 = -log(0.9)/2 = 0.05268
    pd[0] = 0.9f; pd[1] = 0.1f;
    td[0] = 1.0f; td[1] = 0.0f;
    loss = boat_loss_compute(ce, pred, target);
    expected = -logf(0.9f) / 2.0f;
    if (fabsf(loss - expected) > 1e-5f) {
        boat_tensor_unref(pred); boat_tensor_unref(target); boat_loss_free(ce);
        FAIL("CE [0.9,0.1]/[1,0] wrong");
    }

    // Test 3: perfect match, pred=[1,0], target=[1,0]
    // CE = -(1*log(1) + 0*log(0))/2 ... but pred clipped to [eps, 1-eps]
    // clipped: [1-1e-7, 1e-7], so loss = -(1*log(0.9999999) + 0*log(1e-7))/2
    // = -log(0.9999999)/2 ≈ 5e-8
    pd[0] = 1.0f; pd[1] = 0.0f;
    td[0] = 1.0f; td[1] = 0.0f;
    loss = boat_loss_compute(ce, pred, target);
    if (loss < 0 || loss > 1e-5f) {
        boat_tensor_unref(pred); boat_tensor_unref(target); boat_loss_free(ce);
        FAIL("CE perfect match not near 0");
    }

    boat_tensor_unref(pred); boat_tensor_unref(target); boat_loss_free(ce);
    PASS();
    return 0;
}

static int test_ce_backward_gradient(void) {
    TEST("CrossEntropy backward gradient check");

    int64_t shape[] = {2, 3};
    boat_tensor_t* pred = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* target = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    // Fill predictions with reasonable probabilities
    srand(99);
    float* pd = (float*)boat_tensor_data(pred);
    float* td = (float*)boat_tensor_data(target);
    size_t n = boat_tensor_nelements(pred);
    for (size_t i = 0; i < n; i++) {
        pd[i] = rand_float() * 0.8f + 0.1f;  // [0.1, 0.9]
        td[i] = rand_float() < 0.3f ? 1.0f : 0.0f;  // sparse one-ish
    }

    boat_loss_t* ce = boat_cross_entropy_loss_create();

    // Compute analytical gradient
    boat_tensor_t* grad = boat_loss_backward(ce, pred, target);
    if (!grad) {
        boat_tensor_unref(pred); boat_tensor_unref(target); boat_loss_free(ce);
        FAIL("CE backward returned NULL");
    }

    // Numerical gradient check
    float* grad_data = (float*)boat_tensor_data(grad);
    float eps = 1e-4f;
    int mismatches = 0;

    for (size_t i = 0; i < n; i++) {
        // Skip elements where pred is in clipping range -- numerical vs analytical
        // will disagree because of the non-differentiable clip
        if (pd[i] < 1.1e-7f || pd[i] > 1.0f - 1.1e-7f) continue;

        float numerical = compute_numerical_loss_gradient(ce, pred, target, i, eps);
        float analytical = grad_data[i];
        float abs_err = fabsf(analytical - numerical);
        float denom = fabsf(analytical) + fabsf(numerical);
        float rel_err = denom > 1e-10f ? abs_err / denom : 0.0f;
        if (abs_err > 1e-3f && rel_err > 1e-2f) {
            if (mismatches < 3) {
                printf("\n        [%zu] analytical=%.6f numerical=%.6f", i, analytical, numerical);
            }
            mismatches++;
        }
    }

    boat_tensor_unref(grad);
    boat_tensor_unref(pred); boat_tensor_unref(target);
    boat_loss_free(ce);

    if (mismatches > 0) {
        FAIL("CE gradient mismatch");
    }
    PASS();
    return 0;
}

static int test_ce_training(void) {
    TEST("CrossEntropy training loop");

    // Model: Dense(4->8) -> ReLU -> Dense(8->2) -> Softmax
    boat_dense_layer_t* fc1 = boat_dense_layer_create(4, 8, 1);
    boat_relu_layer_t* relu = boat_relu_layer_create();
    boat_dense_layer_t* fc2 = boat_dense_layer_create(8, 2, 1);
    boat_softmax_layer_t* sm = boat_softmax_layer_create(1);  // softmax over classes (axis=1)

    // Synthetic 2-class data: class 0 near [0,0,0,0], class 1 near [1,1,1,1]
    int batch_size = 16;
    int n_classes = 2;
    int64_t x_shape[] = {batch_size, 4};
    int64_t y_shape[] = {batch_size, n_classes};
    int64_t label_shape[] = {batch_size};
    boat_tensor_t* x = boat_tensor_create(x_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* y = boat_tensor_create(y_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    int* labels = (int*)malloc(batch_size * sizeof(int));

    float* xd = (float*)boat_tensor_data(x);
    srand(777);
    for (int i = 0; i < batch_size; i++) {
        int label = i % 2;
        labels[i] = label;
        for (int j = 0; j < 4; j++) {
            xd[i * 4 + j] = (label == 0) ? rand_float() * 0.3f : 0.7f + rand_float() * 0.3f;
        }
    }
    fill_tensor_one_hot(y, labels, n_classes);

    boat_loss_t* ce = boat_cross_entropy_loss_create();
    float lr = 0.02f;
    int n_epochs = 100;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        // Forward: fc1 -> relu -> fc2 -> softmax
        boat_tensor_t* a1 = boat_dense_layer_forward(fc1, x);
        boat_tensor_t* a2 = boat_relu_layer_forward(relu, a1);
        boat_tensor_t* a3 = boat_dense_layer_forward(fc2, a2);
        boat_tensor_t* out = boat_softmax_layer_forward(sm, a3);

        float loss = boat_loss_compute(ce, out, y);
        if (epoch == 0) initial_loss = loss;
        if (epoch == n_epochs - 1) final_loss = loss;

        // Loss backward (w.r.t. softmax output)
        boat_tensor_t* grad = boat_loss_backward(ce, out, y);
        if (grad) {
            // Backward chain: softmax -> fc2 -> relu -> fc1
            boat_tensor_t* g = grad;
            boat_tensor_t* t;
            t = boat_softmax_layer_backward(sm, g);
            if (t) { boat_tensor_unref(g); g = t; } else { boat_tensor_unref(g); break; }
            t = boat_dense_layer_backward(fc2, g);
            if (t) { boat_tensor_unref(g); g = t; } else { boat_tensor_unref(g); break; }
            t = boat_relu_layer_backward(relu, g);
            if (t) { boat_tensor_unref(g); g = t; } else { boat_tensor_unref(g); break; }
            t = boat_dense_layer_backward(fc1, g);
            if (t) { boat_tensor_unref(g); g = t; } else { boat_tensor_unref(g); break; }
            boat_tensor_unref(g);
        }

        // Update
        boat_dense_layer_update(fc1, lr);
        boat_dense_layer_update(fc2, lr);

        boat_tensor_unref(out);
        boat_tensor_unref(a3);
        boat_tensor_unref(a2);
        boat_tensor_unref(a1);
    }

    free(labels);
    boat_loss_free(ce);
    boat_tensor_unref(x); boat_tensor_unref(y);
    boat_softmax_layer_free(sm);
    boat_dense_layer_free(fc2);
    boat_relu_layer_free(relu);
    boat_dense_layer_free(fc1);

    if (final_loss >= initial_loss) {
        FAIL("CE training did not reduce loss");
    }
    if (final_loss > 0.3f) {
        FAIL("CE final loss too high");
    }
    PASS();
    return 0;
}

// ---------------------------------------------------------------------------
// Huber Tests
// ---------------------------------------------------------------------------

static int test_huber_forward(void) {
    TEST("Huber forward compute");

    int64_t shape[] = {2};
    boat_tensor_t* pred = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* target = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* pd = (float*)boat_tensor_data(pred);
    float* td = (float*)boat_tensor_data(target);
    boat_loss_t* huber = boat_huber_loss_create(1.0f);

    // Test 1: pred=[1,2], target=[0,0], delta=1, N=2
    // |1|<=1 => 0.5*1^2=0.5; |2|>1 => 1*(2-0.5)=1.5; total=2.0; mean=1.0
    pd[0] = 1.0f; pd[1] = 2.0f;
    td[0] = 0.0f; td[1] = 0.0f;
    float loss = boat_loss_compute(huber, pred, target);
    if (fabsf(loss - 1.0f) > 1e-5f) {
        boat_tensor_unref(pred); boat_tensor_unref(target); boat_loss_free(huber);
        FAIL("Huber [1,2]/[0,0] not 1.0");
    }

    // Test 2: pred=[0.5,-0.5], target=[0,0], delta=1, N=2
    // |0.5|<=1 => 0.5*0.25=0.125; |-0.5|<=1 => 0.125; total=0.25; mean=0.125
    pd[0] = 0.5f; pd[1] = -0.5f;
    td[0] = 0.0f; td[1] = 0.0f;
    loss = boat_loss_compute(huber, pred, target);
    if (fabsf(loss - 0.125f) > 1e-5f) {
        boat_tensor_unref(pred); boat_tensor_unref(target); boat_loss_free(huber);
        FAIL("Huber [0.5,-0.5]/zeros not 0.125");
    }

    // Test 3: pred=[-3], target=[0], delta=1, N=1
    // |-3|>1 => 1*(3-0.5)=2.5; mean=2.5
    int64_t shape1[] = {1};
    boat_tensor_t* p3 = boat_tensor_create(shape1, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* t3 = boat_tensor_create(shape1, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    *(float*)boat_tensor_data(p3) = -3.0f;
    *(float*)boat_tensor_data(t3) = 0.0f;
    loss = boat_loss_compute(huber, p3, t3);
    if (fabsf(loss - 2.5f) > 1e-5f) {
        boat_tensor_unref(pred); boat_tensor_unref(target);
        boat_tensor_unref(p3); boat_tensor_unref(t3); boat_loss_free(huber);
        FAIL("Huber [-3]/[0] not 2.5");
    }

    // Test 4: delta=2, pred=[-5], target=[0]
    // |-5|>2 => 2*(5-0.5*2)=2*4=8; mean=8
    *(float*)boat_tensor_data(p3) = -5.0f;
    boat_loss_t* huber2 = boat_huber_loss_create(2.0f);
    loss = boat_loss_compute(huber2, p3, t3);
    if (fabsf(loss - 8.0f) > 1e-5f) {
        boat_tensor_unref(pred); boat_tensor_unref(target);
        boat_tensor_unref(p3); boat_tensor_unref(t3);
        boat_loss_free(huber); boat_loss_free(huber2);
        FAIL("Huber delta=2 [-5]/[0] not 8");
    }

    boat_tensor_unref(pred); boat_tensor_unref(target);
    boat_tensor_unref(p3); boat_tensor_unref(t3);
    boat_loss_free(huber); boat_loss_free(huber2);
    PASS();
    return 0;
}

static int test_huber_backward_gradient(void) {
    TEST("Huber backward gradient check");

    int64_t shape[] = {2, 4};
    boat_tensor_t* pred = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* target = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    // Mix of small and large differences to test both regimes
    float* pd = (float*)boat_tensor_data(pred);
    float* td = (float*)boat_tensor_data(target);
    size_t n = boat_tensor_nelements(pred);
    srand(42);
    for (size_t i = 0; i < n; i++) {
        pd[i] = rand_float() * 4.0f - 2.0f;  // [-2, 2]
        td[i] = 0.0f;
    }

    boat_loss_t* huber = boat_huber_loss_create(1.0f);

    boat_tensor_t* grad = boat_loss_backward(huber, pred, target);
    if (!grad) {
        boat_tensor_unref(pred); boat_tensor_unref(target); boat_loss_free(huber);
        FAIL("Huber backward returned NULL");
    }

    float* grad_data = (float*)boat_tensor_data(grad);
    float eps = 1e-4f;
    int mismatches = 0;

    for (size_t i = 0; i < n; i++) {
        float numerical = compute_numerical_loss_gradient(huber, pred, target, i, eps);
        float analytical = grad_data[i];
        float abs_err = fabsf(analytical - numerical);
        float denom = fabsf(analytical) + fabsf(numerical);
        float rel_err = denom > 1e-10f ? abs_err / denom : 0.0f;
        if (abs_err > 1e-3f && rel_err > 1e-3f) {
            if (mismatches < 3) {
                printf("\n        [%zu] analytical=%.6f numerical=%.6f", i, analytical, numerical);
            }
            mismatches++;
        }
    }

    boat_tensor_unref(grad);
    boat_tensor_unref(pred); boat_tensor_unref(target);
    boat_loss_free(huber);

    if (mismatches > 0) {
        FAIL("Huber gradient mismatch");
    }
    PASS();
    return 0;
}

static int test_huber_training(void) {
    TEST("Huber training loop");

    // Model: Dense(1->8) -> ReLU -> Dense(8->1), same as MSE
    boat_dense_layer_t* fc1 = boat_dense_layer_create(1, 8, 1);
    boat_relu_layer_t* relu = boat_relu_layer_create();
    boat_dense_layer_t* fc2 = boat_dense_layer_create(8, 1, 1);

    // Training data: y = 2x + 1, with one injected outlier (y=100)
    int batch_size = 16;
    int64_t x_shape[] = {batch_size, 1};
    int64_t y_shape[] = {batch_size, 1};
    boat_tensor_t* x = boat_tensor_create(x_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* y = boat_tensor_create(y_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* xd = (float*)boat_tensor_data(x);
    float* yd = (float*)boat_tensor_data(y);
    for (int i = 0; i < batch_size; i++) {
        xd[i] = (float)i / (float)batch_size;
        yd[i] = 2.0f * xd[i] + 1.0f;
    }
    // Inject outlier at sample 0
    yd[0] = 10.0f;

    boat_loss_t* huber = boat_huber_loss_create(1.0f);

    int n_epochs = 100;
    float lr = 0.02f;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        // Forward
        boat_tensor_t* a1 = boat_dense_layer_forward(fc1, x);
        boat_tensor_t* a2 = boat_relu_layer_forward(relu, a1);
        boat_tensor_t* out = boat_dense_layer_forward(fc2, a2);

        float loss = boat_loss_compute(huber, out, y);
        if (epoch == 0) initial_loss = loss;
        if (epoch == n_epochs - 1) final_loss = loss;

        // Backward
        boat_tensor_t* grad = boat_loss_backward(huber, out, y);
        if (grad) {
            boat_tensor_t* g = grad;
            boat_tensor_t* t;
            t = boat_dense_layer_backward(fc2, g);
            if (t) { boat_tensor_unref(g); g = t; } else { boat_tensor_unref(g); break; }
            t = boat_relu_layer_backward(relu, g);
            if (t) { boat_tensor_unref(g); g = t; } else { boat_tensor_unref(g); break; }
            t = boat_dense_layer_backward(fc1, g);
            if (t) { boat_tensor_unref(g); g = t; } else { boat_tensor_unref(g); break; }
            boat_tensor_unref(g);
        }

        boat_dense_layer_update(fc1, lr);
        boat_dense_layer_update(fc2, lr);

        boat_tensor_unref(out);
        boat_tensor_unref(a2);
        boat_tensor_unref(a1);
    }

    boat_loss_free(huber);
    boat_tensor_unref(x); boat_tensor_unref(y);
    boat_dense_layer_free(fc2);
    boat_relu_layer_free(relu);
    boat_dense_layer_free(fc1);

    if (final_loss >= initial_loss) {
        FAIL("Huber training did not reduce loss");
    }
    if (final_loss > 5.0f) {
        FAIL("Huber final loss too high");
    }
    PASS();
    return 0;
}

// ---------------------------------------------------------------------------
// Edge case tests
// ---------------------------------------------------------------------------

static int test_loss_edge_cases(void) {
    printf("  NULL loss handle ... ");
    int64_t shape[] = {2};
    boat_tensor_t* pred = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* target = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor_linear(pred, 1.0f);
    fill_tensor_linear(target, 0.0f);
    float val = boat_loss_compute(NULL, pred, target);
    if (val != 0.0f) { printf("FAIL: compute(NULL) != 0\n"); boat_tensor_unref(pred); boat_tensor_unref(target); return 1; }
    boat_tensor_t* g = boat_loss_backward(NULL, pred, target);
    if (g != NULL) { printf("FAIL: backward(NULL) != NULL\n"); boat_tensor_unref(pred); boat_tensor_unref(target); return 1; }
    printf("PASS\n"); tests_passed++; tests_total++;

    printf("  NULL predictions ... ");
    boat_loss_t* mse = boat_mse_loss_create();
    val = boat_loss_compute(mse, NULL, target);
    if (val != 0.0f) { printf("FAIL\n"); boat_loss_free(mse); boat_tensor_unref(pred); boat_tensor_unref(target); return 1; }
    g = boat_loss_backward(mse, NULL, target);
    if (g != NULL) { printf("FAIL\n"); boat_loss_free(mse); boat_tensor_unref(pred); boat_tensor_unref(target); return 1; }
    printf("PASS\n"); tests_passed++; tests_total++;

    printf("  NULL targets ... ");
    val = boat_loss_compute(mse, pred, NULL);
    if (val != 0.0f) { printf("FAIL\n"); boat_loss_free(mse); boat_tensor_unref(pred); boat_tensor_unref(target); return 1; }
    g = boat_loss_backward(mse, pred, NULL);
    if (g != NULL) { printf("FAIL\n"); boat_loss_free(mse); boat_tensor_unref(pred); boat_tensor_unref(target); return 1; }
    printf("PASS\n"); tests_passed++; tests_total++;

    printf("  shape mismatch ... ");
    int64_t bad_shape[] = {2, 1};
    boat_tensor_t* bad = boat_tensor_create(bad_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor_linear(bad, 0.0f);
    val = boat_loss_compute(mse, pred, bad);
    if (val != 0.0f) { printf("FAIL\n"); boat_loss_free(mse); boat_tensor_unref(pred); boat_tensor_unref(target); boat_tensor_unref(bad); return 1; }
    g = boat_loss_backward(mse, pred, bad);
    if (g != NULL) { printf("FAIL\n"); boat_loss_free(mse); boat_tensor_unref(pred); boat_tensor_unref(target); boat_tensor_unref(bad); return 1; }
    printf("PASS\n"); tests_passed++; tests_total++;

    printf("  dtype mismatch ... ");
    boat_tensor_t* int_t = boat_tensor_create(shape, 1, BOAT_DTYPE_INT64, BOAT_DEVICE_CPU);
    fill_tensor_linear(int_t, 0.0f);
    val = boat_loss_compute(mse, pred, int_t);
    if (val != 0.0f) { printf("FAIL\n"); boat_loss_free(mse); boat_tensor_unref(pred); boat_tensor_unref(target); boat_tensor_unref(bad); boat_tensor_unref(int_t); return 1; }
    g = boat_loss_backward(mse, pred, int_t);
    if (g != NULL) { printf("FAIL\n"); boat_loss_free(mse); boat_tensor_unref(pred); boat_tensor_unref(target); boat_tensor_unref(bad); boat_tensor_unref(int_t); return 1; }
    printf("PASS\n"); tests_passed++; tests_total++;

    printf("  Huber delta<=0 returns NULL ... ");
    boat_loss_t* bad_huber = boat_huber_loss_create(0.0f);
    if (bad_huber != NULL) { printf("FAIL\n"); boat_loss_free(mse); boat_tensor_unref(pred); boat_tensor_unref(target); boat_tensor_unref(bad); boat_tensor_unref(int_t); return 1; }
    bad_huber = boat_huber_loss_create(-1.0f);
    if (bad_huber != NULL) { printf("FAIL\n"); boat_loss_free(mse); boat_tensor_unref(pred); boat_tensor_unref(target); boat_tensor_unref(bad); boat_tensor_unref(int_t); return 1; }
    printf("PASS\n"); tests_passed++; tests_total++;

    printf("  NULL both tensors (all three losses) ... ");
    // MSE
    val = boat_loss_compute(mse, NULL, NULL);
    if (val != 0.0f) { printf("FAIL\n"); boat_loss_free(mse); boat_tensor_unref(pred); boat_tensor_unref(target); boat_tensor_unref(bad); boat_tensor_unref(int_t); return 1; }
    g = boat_loss_backward(mse, NULL, NULL);
    if (g != NULL) { printf("FAIL\n"); boat_loss_free(mse); boat_tensor_unref(pred); boat_tensor_unref(target); boat_tensor_unref(bad); boat_tensor_unref(int_t); return 1; }
    boat_loss_free(mse);
    // CE
    boat_loss_t* ce = boat_cross_entropy_loss_create();
    val = boat_loss_compute(ce, NULL, NULL);
    if (val != 0.0f) { printf("FAIL\n"); boat_loss_free(ce); boat_tensor_unref(pred); boat_tensor_unref(target); boat_tensor_unref(bad); boat_tensor_unref(int_t); return 1; }
    g = boat_loss_backward(ce, NULL, NULL);
    if (g != NULL) { printf("FAIL\n"); boat_loss_free(ce); boat_tensor_unref(pred); boat_tensor_unref(target); boat_tensor_unref(bad); boat_tensor_unref(int_t); return 1; }
    boat_loss_free(ce);
    // Huber
    boat_loss_t* hub = boat_huber_loss_create(1.0f);
    val = boat_loss_compute(hub, NULL, NULL);
    if (val != 0.0f) { printf("FAIL\n"); boat_loss_free(hub); boat_tensor_unref(pred); boat_tensor_unref(target); boat_tensor_unref(bad); boat_tensor_unref(int_t); return 1; }
    g = boat_loss_backward(hub, NULL, NULL);
    if (g != NULL) { printf("FAIL\n"); boat_loss_free(hub); boat_tensor_unref(pred); boat_tensor_unref(target); boat_tensor_unref(bad); boat_tensor_unref(int_t); return 1; }
    boat_loss_free(hub);

    printf("PASS\n"); tests_passed++; tests_total++;

    boat_tensor_unref(pred); boat_tensor_unref(target);
    boat_tensor_unref(bad); boat_tensor_unref(int_t);
    return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(void) {
    printf("Loss Function Integration Tests\n");
    printf("===============================\n\n");

    int fail = 0;

    printf("--- MSE ---\n");
    fail |= test_mse_forward();
    fail |= test_mse_backward_gradient();
    fail |= test_mse_training();

    printf("\n--- Cross-Entropy ---\n");
    fail |= test_ce_forward();
    fail |= test_ce_backward_gradient();
    fail |= test_ce_training();

    printf("\n--- Huber ---\n");
    fail |= test_huber_forward();
    fail |= test_huber_backward_gradient();
    fail |= test_huber_training();

    printf("\n--- Edge Cases ---\n");
    fail |= test_loss_edge_cases();

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return fail ? 1 : 0;
}
