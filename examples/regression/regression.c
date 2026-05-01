// regression.c - Regression and time series prediction with Boat framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <boat/tensor.h>
#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/optimizers.h>
#include <boat/loss.h>
#include <boat/memory.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

// =========================================================================
// Utility helpers
// =========================================================================

static float frand(void) {
    return (float)rand() / (float)RAND_MAX;
}

// Generate N samples of y = a * sin(b * x + c) + noise * N(0,1)
// Returns two float arrays (caller must free)
static void generate_regression_data(float** xs_out, float** ys_out, size_t n) {
    *xs_out = (float*)malloc(n * sizeof(float));
    *ys_out = (float*)malloc(n * sizeof(float));
    if (!*xs_out || !*ys_out) return;

    for (size_t i = 0; i < n; i++) {
        float x = 2.0f * 3.14159265f * (float)i / (float)(n - 1);
        float noise = 0.15f * (frand() * 2.0f - 1.0f);
        float y = sinf(x) + noise;
        (*xs_out)[i] = x;
        (*ys_out)[i] = y;
    }
}

// Generate sine wave sequence for time series
static void generate_timeseries_data(float** seq_out, size_t n) {
    *seq_out = (float*)malloc(n * sizeof(float));
    if (!*seq_out) return;

    for (size_t i = 0; i < n; i++) {
        float t = 0.1f * (float)i;
        (*seq_out)[i] = sinf(t);
    }
}

// Build sliding windows from a 1D sequence into X (windows) and y (targets)
// Window i = seq[i..i+window_size-1], target = seq[i+window_size]
static void build_sliding_windows(const float* seq, size_t seq_len,
                                   size_t window_size,
                                   float** X_out, float** y_out, size_t* num_windows) {
    *num_windows = seq_len - window_size;
    *X_out = (float*)malloc(*num_windows * window_size * sizeof(float));
    *y_out = (float*)malloc(*num_windows * sizeof(float));
    if (!*X_out || !*y_out) return;

    for (size_t i = 0; i < *num_windows; i++) {
        for (size_t j = 0; j < window_size; j++) {
            (*X_out)[i * window_size + j] = seq[i + j];
        }
        (*y_out)[i] = seq[i + window_size];
    }
}

// =========================================================================
// Regression demo: MLP fits y = sin(x) + noise
// =========================================================================

typedef struct {
    boat_dense_layer_t* fc1;
    boat_dense_layer_t* fc2;
    boat_dense_layer_t* fc3;
    boat_relu_layer_t* relu1;
    boat_relu_layer_t* relu2;
} reg_model_t;

static reg_model_t* create_reg_model(void) {
    reg_model_t* m = (reg_model_t*)malloc(sizeof(reg_model_t));
    if (!m) return NULL;

    m->fc1 = boat_dense_layer_create(1, 32, true);
    m->relu1 = boat_relu_layer_create();
    m->fc2 = boat_dense_layer_create(32, 32, true);
    m->relu2 = boat_relu_layer_create();
    m->fc3 = boat_dense_layer_create(32, 1, true);

    if (!m->fc1 || !m->relu1 || !m->fc2 || !m->relu2 || !m->fc3) {
        free(m);
        return NULL;
    }
    return m;
}

static void free_reg_model(reg_model_t* m) {
    if (!m) return;
    if (m->fc1) boat_dense_layer_free(m->fc1);
    if (m->relu1) boat_relu_layer_free(m->relu1);
    if (m->fc2) boat_dense_layer_free(m->fc2);
    if (m->relu2) boat_relu_layer_free(m->relu2);
    if (m->fc3) boat_dense_layer_free(m->fc3);
    free(m);
}

static void register_reg_params(boat_optimizer_t* opt, reg_model_t* m) {
    boat_optimizer_add_parameter(opt, boat_dense_layer_get_weight(m->fc1), boat_dense_layer_get_grad_weight(m->fc1));
    boat_optimizer_add_parameter(opt, boat_dense_layer_get_bias(m->fc1), boat_dense_layer_get_grad_bias(m->fc1));
    boat_optimizer_add_parameter(opt, boat_dense_layer_get_weight(m->fc2), boat_dense_layer_get_grad_weight(m->fc2));
    boat_optimizer_add_parameter(opt, boat_dense_layer_get_bias(m->fc2), boat_dense_layer_get_grad_bias(m->fc2));
    boat_optimizer_add_parameter(opt, boat_dense_layer_get_weight(m->fc3), boat_dense_layer_get_grad_weight(m->fc3));
    boat_optimizer_add_parameter(opt, boat_dense_layer_get_bias(m->fc3), boat_dense_layer_get_grad_bias(m->fc3));
}

static boat_tensor_t* forward_reg(reg_model_t* m, boat_tensor_t* input) {
    boat_tensor_t* x = input;
    boat_tensor_t* tmp;

    x = boat_dense_layer_forward(m->fc1, x);
    if (!x) return NULL;

    tmp = boat_relu_layer_forward(m->relu1, x);
    if (!tmp) { boat_tensor_unref(x); return NULL; }
    boat_tensor_unref(x); x = tmp;

    tmp = boat_dense_layer_forward(m->fc2, x);
    if (!tmp) { boat_tensor_unref(x); return NULL; }
    boat_tensor_unref(x); x = tmp;

    tmp = boat_relu_layer_forward(m->relu2, x);
    if (!tmp) { boat_tensor_unref(x); return NULL; }
    boat_tensor_unref(x); x = tmp;

    tmp = boat_dense_layer_forward(m->fc3, x);
    if (!tmp) { boat_tensor_unref(x); return NULL; }
    boat_tensor_unref(x); x = tmp;

    return x;
}

static void backward_reg(reg_model_t* m, boat_tensor_t* grad) {
    boat_tensor_t* out;

    out = boat_dense_layer_backward(m->fc3, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_relu_layer_backward(m->relu2, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_dense_layer_backward(m->fc2, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_relu_layer_backward(m->relu1, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_dense_layer_backward(m->fc1, grad);
    boat_tensor_unref(grad);
    if (out) boat_tensor_unref(out);
}

static int run_regression_demo(void) {
    printf("\n=== Regression Demo: y = sin(x) + noise ===\n\n");

    srand(42);
    size_t N = 2000;
    float* xs, *ys;
    generate_regression_data(&xs, &ys, N);

    // Standardize targets: y_norm = (y - mean) / std
    // and normalize inputs to [0, 1]
    double y_sum = 0.0, y_sum_sq = 0.0;
    for (size_t i = 0; i < N; i++) {
        y_sum += ys[i];
        y_sum_sq += ys[i] * ys[i];
        xs[i] /= (2.0f * 3.14159265f);  // normalize x to [0, 1]
    }
    float y_mean = (float)(y_sum / N);
    float y_std = (float)sqrt(y_sum_sq / N - y_mean * y_mean);
    for (size_t i = 0; i < N; i++) {
        ys[i] = (ys[i] - y_mean) / y_std;
    }

    // Train/test split 80/20
    size_t n_train = (size_t)(N * 0.8f);
    size_t n_test = N - n_train;

    // Create tensors
    int64_t train_shape[] = {(int64_t)n_train, 1};
    int64_t test_shape[] = {(int64_t)n_test, 1};

    boat_tensor_t* train_x = boat_tensor_create(train_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* train_y = boat_tensor_create(train_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* test_x = boat_tensor_create(test_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* test_y = boat_tensor_create(test_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!train_x || !train_y || !test_x || !test_y) {
        fprintf(stderr, "Failed to create tensors\n");
        free(xs); free(ys);
        return 1;
    }

    memcpy(boat_tensor_data(train_x), xs, n_train * sizeof(float));
    memcpy(boat_tensor_data(train_y), ys, n_train * sizeof(float));
    memcpy(boat_tensor_data(test_x), xs + n_train, n_test * sizeof(float));
    memcpy(boat_tensor_data(test_y), ys + n_train, n_test * sizeof(float));
    free(xs); free(ys);

    reg_model_t* model = create_reg_model();
    if (!model) { fprintf(stderr, "Failed to create model\n"); return 1; }

    boat_optimizer_t* opt = boat_adam_optimizer_create(0.001f, 0.9f, 0.999f, 1e-8f);
    if (!opt) { fprintf(stderr, "Failed to create optimizer\n"); return 1; }
    register_reg_params(opt, model);

    boat_loss_t* loss_fn = boat_mse_loss_create();
    if (!loss_fn) { fprintf(stderr, "Failed to create loss\n"); return 1; }

    // Training
    size_t batch_size = 32;
    size_t n_batches = n_train / batch_size;
    int epochs = 40;

    printf("Training MLP (1->32->32->1) with MSE loss, Adam lr=%.3f\n", 0.001f);
    printf("Samples: %zu train, %zu test | Batch: %zu | Epochs: %d\n\n",
           n_train, n_test, batch_size, epochs);

    int64_t batch_shape[] = {(int64_t)batch_size, 1};
    boat_tensor_t* batch_x = boat_tensor_create(batch_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* batch_y = boat_tensor_create(batch_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!batch_x || !batch_y) { fprintf(stderr, "Failed batch tensors\n"); return 1; }

    const float* train_x_data = (const float*)boat_tensor_const_data(train_x);
    const float* train_y_data = (const float*)boat_tensor_const_data(train_y);

    for (int ep = 0; ep < epochs; ep++) {
        for (size_t b = 0; b < n_batches; b++) {
            size_t off = b * batch_size;

            memcpy(boat_tensor_data(batch_x), train_x_data + off, batch_size * sizeof(float));
            memcpy(boat_tensor_data(batch_y), train_y_data + off, batch_size * sizeof(float));

            boat_tensor_t* pred = forward_reg(model, batch_x);
            if (!pred) continue;

            float loss = boat_loss_compute(loss_fn, pred, batch_y);
            (void)loss;

            boat_tensor_t* grad = boat_loss_backward(loss_fn, pred, batch_y);
            if (grad) {
                backward_reg(model, grad);
                boat_tensor_unref(grad);
            }
            boat_tensor_unref(pred);

            boat_optimizer_step(opt);
            boat_optimizer_zero_grad(opt);
        }

        if ((ep + 1) % 20 == 0) {
            // Evaluate on test set
            boat_tensor_t* test_pred = forward_reg(model, test_x);
            float test_loss_val = 0.0f;
            if (test_pred) {
                test_loss_val = boat_loss_compute(loss_fn, test_pred, test_y);
                boat_tensor_unref(test_pred);
            }
            printf("  Epoch %3d/%-3d  test MSE = %.6f\n", ep + 1, epochs, test_loss_val);
        }
    }

    // Print predictions
    printf("\nPredictions (denormalized):\n");
    boat_tensor_t* all_pred = forward_reg(model, train_x);
    if (all_pred) {
        const float* pred_data = (const float*)boat_tensor_const_data(all_pred);
        const float* x_data = (const float*)boat_tensor_const_data(train_x);
        printf("%6s  %10s  %10s  %10s\n", "x", "pred", "actual", "error");
        for (size_t i = 0; i < n_train; i += n_train / 10) {
            float p = pred_data[i] * y_std + y_mean;
            float a = train_y_data[i] * y_std + y_mean;
            float x_denorm = x_data[i] * 6.2831853f;  // denormalize for display
            printf("%6.2f  %10.4f  %10.4f  %10.4f\n",
                   x_denorm, p, a, p - a);
        }
        boat_tensor_unref(all_pred);
    }

    boat_tensor_unref(batch_x);
    boat_tensor_unref(batch_y);
    boat_loss_free(loss_fn);
    boat_optimizer_free(opt);
    free_reg_model(model);
    boat_tensor_unref(train_x);
    boat_tensor_unref(train_y);
    boat_tensor_unref(test_x);
    boat_tensor_unref(test_y);

    printf("  Regression demo completed!\n");
    return 0;
}

// =========================================================================
// Time series demo: sliding window MLP predicts sine wave
// =========================================================================

typedef struct {
    boat_dense_layer_t* fc1;
    boat_dense_layer_t* fc2;
    boat_dense_layer_t* fc3;
    boat_relu_layer_t* relu1;
    boat_relu_layer_t* relu2;
} ts_model_t;

static ts_model_t* create_ts_model(size_t window_size) {
    ts_model_t* m = (ts_model_t*)malloc(sizeof(ts_model_t));
    if (!m) return NULL;

    m->fc1 = boat_dense_layer_create(window_size, 32, true);
    m->relu1 = boat_relu_layer_create();
    m->fc2 = boat_dense_layer_create(32, 32, true);
    m->relu2 = boat_relu_layer_create();
    m->fc3 = boat_dense_layer_create(32, 1, true);

    if (!m->fc1 || !m->relu1 || !m->fc2 || !m->relu2 || !m->fc3) {
        free(m);
        return NULL;
    }
    return m;
}

static void free_ts_model(ts_model_t* m) {
    if (!m) return;
    if (m->fc1) boat_dense_layer_free(m->fc1);
    if (m->relu1) boat_relu_layer_free(m->relu1);
    if (m->fc2) boat_dense_layer_free(m->fc2);
    if (m->relu2) boat_relu_layer_free(m->relu2);
    if (m->fc3) boat_dense_layer_free(m->fc3);
    free(m);
}

static void register_ts_params(boat_optimizer_t* opt, ts_model_t* m) {
    boat_optimizer_add_parameter(opt, boat_dense_layer_get_weight(m->fc1), boat_dense_layer_get_grad_weight(m->fc1));
    boat_optimizer_add_parameter(opt, boat_dense_layer_get_bias(m->fc1), boat_dense_layer_get_grad_bias(m->fc1));
    boat_optimizer_add_parameter(opt, boat_dense_layer_get_weight(m->fc2), boat_dense_layer_get_grad_weight(m->fc2));
    boat_optimizer_add_parameter(opt, boat_dense_layer_get_bias(m->fc2), boat_dense_layer_get_grad_bias(m->fc2));
    boat_optimizer_add_parameter(opt, boat_dense_layer_get_weight(m->fc3), boat_dense_layer_get_grad_weight(m->fc3));
    boat_optimizer_add_parameter(opt, boat_dense_layer_get_bias(m->fc3), boat_dense_layer_get_grad_bias(m->fc3));
}

static boat_tensor_t* forward_ts(ts_model_t* m, boat_tensor_t* input) {
    boat_tensor_t* x = input;
    boat_tensor_t* tmp;

    x = boat_dense_layer_forward(m->fc1, x);
    if (!x) return NULL;

    tmp = boat_relu_layer_forward(m->relu1, x);
    if (!tmp) { boat_tensor_unref(x); return NULL; }
    boat_tensor_unref(x); x = tmp;

    tmp = boat_dense_layer_forward(m->fc2, x);
    if (!tmp) { boat_tensor_unref(x); return NULL; }
    boat_tensor_unref(x); x = tmp;

    tmp = boat_relu_layer_forward(m->relu2, x);
    if (!tmp) { boat_tensor_unref(x); return NULL; }
    boat_tensor_unref(x); x = tmp;

    tmp = boat_dense_layer_forward(m->fc3, x);
    if (!tmp) { boat_tensor_unref(x); return NULL; }
    boat_tensor_unref(x); x = tmp;

    return x;
}

static void backward_ts(ts_model_t* m, boat_tensor_t* grad) {
    boat_tensor_t* out;

    out = boat_dense_layer_backward(m->fc3, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_relu_layer_backward(m->relu2, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_dense_layer_backward(m->fc2, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_relu_layer_backward(m->relu1, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_dense_layer_backward(m->fc1, grad);
    boat_tensor_unref(grad);
    if (out) boat_tensor_unref(out);
}

static int run_timeseries_demo(void) {
    printf("\n=== Time Series Demo: Sine Wave Prediction ===\n\n");

    srand(123);
    size_t seq_len = 1000;
    float* seq;
    generate_timeseries_data(&seq, seq_len);

    size_t window_size = 8;
    float* X, *y;
    size_t n_windows;
    build_sliding_windows(seq, seq_len, window_size, &X, &y, &n_windows);
    free(seq);

    // Standardize
    double y_sum = 0.0, y_sum_sq = 0.0;
    for (size_t i = 0; i < n_windows; i++) {
        y_sum += y[i];
        y_sum_sq += y[i] * y[i];
    }
    float y_mean = (float)(y_sum / n_windows);
    float y_std = (float)sqrt(y_sum_sq / n_windows - y_mean * y_mean);

    // Also standardize X per-sample (each window has its own mean)
    for (size_t i = 0; i < n_windows; i++) {
        y[i] = (y[i] - y_mean) / y_std;
    }

    size_t n_train = (size_t)(n_windows * 0.8);
    size_t n_test = n_windows - n_train;

    int64_t train_shape[] = {(int64_t)n_train, (int64_t)window_size};
    int64_t test_shape[] = {(int64_t)n_test, (int64_t)window_size};
    int64_t train_y_shape[] = {(int64_t)n_train, 1};
    int64_t test_y_shape[] = {(int64_t)n_test, 1};

    boat_tensor_t* train_x = boat_tensor_create(train_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* test_x = boat_tensor_create(test_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* train_y_t = boat_tensor_create(train_y_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* test_y_t = boat_tensor_create(test_y_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!train_x || !test_x || !train_y_t || !test_y_t) {
        fprintf(stderr, "Failed to create tensors\n");
        free(X); free(y);
        return 1;
    }

    memcpy(boat_tensor_data(train_x), X, n_train * window_size * sizeof(float));
    memcpy(boat_tensor_data(test_x), X + n_train * window_size, n_test * window_size * sizeof(float));
    memcpy(boat_tensor_data(train_y_t), y, n_train * sizeof(float));
    memcpy(boat_tensor_data(test_y_t), y + n_train, n_test * sizeof(float));
    free(X); free(y);

    ts_model_t* model = create_ts_model(window_size);
    if (!model) { fprintf(stderr, "Failed model\n"); return 1; }

    boat_optimizer_t* opt = boat_adam_optimizer_create(0.01f, 0.9f, 0.999f, 1e-8f);
    if (!opt) { fprintf(stderr, "Failed optimizer\n"); return 1; }
    register_ts_params(opt, model);

    boat_loss_t* loss_fn = boat_mse_loss_create();
    if (!loss_fn) { fprintf(stderr, "Failed loss\n"); return 1; }

    size_t batch_size = 32;
    size_t n_batches = n_train / batch_size;
    int epochs = 80;

    printf("Training sliding-window MLP (8->32->32->1) on sine wave\n");
    printf("Windows: %zu train, %zu test | Batch: %zu | Epochs: %d\n\n",
           n_train, n_test, batch_size, epochs);

    int64_t batch_shape[] = {(int64_t)batch_size, (int64_t)window_size};
    int64_t batch_y_shape[] = {(int64_t)batch_size, 1};
    boat_tensor_t* batch_x = boat_tensor_create(batch_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* batch_yt = boat_tensor_create(batch_y_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!batch_x || !batch_yt) { fprintf(stderr, "Failed batch\n"); return 1; }

    const float* train_x_data = (const float*)boat_tensor_const_data(train_x);
    const float* train_y_data = (const float*)boat_tensor_const_data(train_y_t);

    for (int ep = 0; ep < epochs; ep++) {
        for (size_t b = 0; b < n_batches; b++) {
            size_t off = b * batch_size;

            memcpy(boat_tensor_data(batch_x), train_x_data + off * window_size,
                   batch_size * window_size * sizeof(float));
            memcpy(boat_tensor_data(batch_yt), train_y_data + off,
                   batch_size * sizeof(float));

            boat_tensor_t* pred = forward_ts(model, batch_x);
            if (!pred) continue;

            float loss = boat_loss_compute(loss_fn, pred, batch_yt);
            (void)loss;

            boat_tensor_t* grad = boat_loss_backward(loss_fn, pred, batch_yt);
            if (grad) {
                backward_ts(model, grad);
                boat_tensor_unref(grad);
            }
            boat_tensor_unref(pred);

            boat_optimizer_step(opt);
            boat_optimizer_zero_grad(opt);
        }

        if ((ep + 1) % 20 == 0) {
            boat_tensor_t* tpred = forward_ts(model, test_x);
            float tloss = 0.0f;
            if (tpred) {
                tloss = boat_loss_compute(loss_fn, tpred, test_y_t);
                boat_tensor_unref(tpred);
            }
            printf("  Epoch %3d/%-3d  test MSE = %.6f\n", ep + 1, epochs, tloss);
        }
    }

    // Single-step predictions on test set
    printf("\nSingle-step predictions on test set:\n");
    {
        const float* raw_x = (const float*)boat_tensor_const_data(test_x);
        const float* raw_y = (const float*)boat_tensor_const_data(test_y_t);
        boat_tensor_t* single_pred = forward_ts(model, test_x);
        if (single_pred) {
            const float* sp = (const float*)boat_tensor_const_data(single_pred);
            printf("%6s  %10s  %10s  %10s\n", "window", "pred", "actual", "error");
            for (size_t i = 0; i < n_test && i < 10; i++) {
                float p = sp[i] * y_std + y_mean;
                float a = raw_y[i] * y_std + y_mean;
                printf("  %4zu  %10.4f  %10.4f  %+.4f\n", i, p, a, p - a);
            }
            boat_tensor_unref(single_pred);
        }
    }

    // Multi-step iterative prediction (accumulates error)
    printf("\nMulti-step iterative prediction (error accumulates):\n");
    {
        // Grab the last window from training set as seed
        float seed[8];
        memcpy(seed, train_x_data + (n_train - 1) * window_size, window_size * sizeof(float));

        printf("  Seed window: ");
        for (size_t i = 0; i < window_size; i++) printf("%.4f ", seed[i]);
        printf("\n");

        printf("\n  Step  |  Predicted  |  Actual  |  Error\n");
        printf("  ------+------------+----------+--------\n");

        // Get the corresponding actual values for comparison
        const float* test_y_raw = (const float*)boat_tensor_const_data(test_y_t);

        float window[8];
        memcpy(window, seed, sizeof(seed));

        for (size_t step = 0; step < 20; step++) {
            // Predict next value
            int64_t single_shape[] = {1, 8};
            boat_tensor_t* input_t = boat_tensor_create(single_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
            if (!input_t) break;
            memcpy(boat_tensor_data(input_t), window, window_size * sizeof(float));

            boat_tensor_t* pred_t = forward_ts(model, input_t);
            float pred_val = 0.0f;
            if (pred_t) {
                pred_val = ((const float*)boat_tensor_const_data(pred_t))[0];
                boat_tensor_unref(pred_t);
            }
            boat_tensor_unref(input_t);

            // Denormalize
            float pred_denorm = pred_val * y_std + y_mean;
            float actual = (step < n_test) ? test_y_raw[step] * y_std + y_mean : 0.0f;

            printf("  %4zu  |  %10.4f  |  %8.4f  |  %+.4f\n",
                   step + 1, pred_denorm, actual, pred_denorm - actual);

            // Shift window: remove first, append prediction
            for (size_t i = 0; i < window_size - 1; i++) {
                window[i] = window[i + 1];
            }
            window[window_size - 1] = pred_val;
        }
    }

    boat_tensor_unref(batch_x);
    boat_tensor_unref(batch_yt);
    boat_loss_free(loss_fn);
    boat_optimizer_free(opt);
    free_ts_model(model);
    boat_tensor_unref(train_x);
    boat_tensor_unref(test_x);
    boat_tensor_unref(train_y_t);
    boat_tensor_unref(test_y_t);

    printf("  Time series demo completed!\n");
    return 0;
}

// =========================================================================
// Main
// =========================================================================

int main(int argc, char* argv[]) {
    printf("=== Boat Framework: Regression & Time Series Demo ===\n");
    setvbuf(stdout, NULL, _IONBF, 0);

    int use_timeseries = 0;
    if (argc > 1) {
        if (strcmp(argv[1], "--timeseries") == 0 || strcmp(argv[1], "-t") == 0) {
            use_timeseries = 1;
        }
    }
    const char* env = getenv("BOAT_DEMO");
    if (env && strcmp(env, "timeseries") == 0) {
        use_timeseries = 1;
    }

    if (use_timeseries) {
        return run_timeseries_demo();
    } else {
        return run_regression_demo();
    }
}
