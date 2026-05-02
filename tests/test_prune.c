// test_prune.c - Weight pruning tests
#include <boat.h>
#include <boat/tensor.h>
#include <boat/layers.h>
#include <boat/model.h>
#include <boat/prune.h>
#include <boat/memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { printf("  %s ... ", name); tests_total++; } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)

// Helper: wrap layer
static boat_layer_t* wrap(void* data, boat_layer_type_t type) {
    boat_layer_t* w = malloc(sizeof(boat_layer_t));
    if (w) { w->data = data; w->type = type; w->ops = NULL; }
    return w;
}

// Helper: fill tensor with deterministic pattern
static void fill_tensor(boat_tensor_t* t, float base) {
    float* d = (float*)boat_tensor_data(t);
    size_t n = boat_tensor_nelements(t);
    for (size_t i = 0; i < n; i++) d[i] = base + (float)(i % 7) * 0.1f;
}

// --- Test 1: compute pruning threshold ---
static int test_compute_threshold(void) {
    TEST("Compute prune threshold");
    int64_t shape[] = {5};
    boat_tensor_t* w = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(w);
    d[0] = 1.0f; d[1] = 2.0f; d[2] = 3.0f; d[3] = 4.0f; d[4] = 5.0f;

    float thr = boat_compute_prune_threshold(w, 0.5f);
    // Sorted |vals| = [1,2,3,4,5], idx = 0.5*4 = 2, value = 3
    if (fabsf(thr - 3.0f) > 0.01f) FAIL("expected ~3.0");

    thr = boat_compute_prune_threshold(w, 0.0f);
    if (thr != -FLT_MAX) FAIL("expected -inf for sparsity=0");

    thr = boat_compute_prune_threshold(w, 1.0f);
    if (thr != FLT_MAX) FAIL("expected +inf for sparsity=1");

    boat_tensor_unref(w);
    PASS(); return 0;
}

// --- Test 2: create magnitude mask ---
static int test_magnitude_mask(void) {
    TEST("Create magnitude mask");
    int64_t shape[] = {3, 4};
    boat_tensor_t* w = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor(w, 0.0f);

    float thr = 0.3f;
    boat_tensor_t* mask = boat_create_magnitude_mask(w, thr);
    if (!mask) FAIL("mask is NULL");
    if (boat_tensor_dtype(mask) != BOAT_DTYPE_FLOAT32) FAIL("mask dtype not FP32");

    const float* m = (const float*)boat_tensor_const_data(mask);
    const float* wd = (const float*)boat_tensor_const_data(w);
    size_t n = boat_tensor_nelements(w);
    for (size_t i = 0; i < n; i++) {
        int expected = (fabsf(wd[i]) <= thr) ? 0 : 1;
        if ((int)m[i] != expected) FAIL("mask value mismatch");
    }

    boat_tensor_unref(w);
    boat_tensor_unref(mask);
    PASS(); return 0;
}

// --- Test 3: apply mask ---
static int test_apply_mask(void) {
    TEST("Apply mask");
    int64_t shape[] = {2, 3};
    boat_tensor_t* w = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor(w, 1.0f);

    boat_tensor_t* mask = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* m = (float*)boat_tensor_data(mask);
    size_t n = boat_tensor_nelements(mask);
    for (size_t i = 0; i < n; i++) m[i] = (i % 2 == 0) ? 1.0f : 0.0f;

    if (!boat_apply_mask(w, mask)) FAIL("apply_mask failed");

    const float* wd = (const float*)boat_tensor_const_data(w);
    for (size_t i = 0; i < n; i++) {
        float expected = (i % 2 == 0) ? (1.0f + (float)(i % 7) * 0.1f) : 0.0f;
        if (fabsf(wd[i] - expected) > 0.001f) FAIL("mask application mismatch");
    }

    boat_tensor_unref(w);
    boat_tensor_unref(mask);
    PASS(); return 0;
}

// --- Test 4: magnitude pruning on Dense layer ---
static int test_magnitude_pruning_dense(void) {
    TEST("Magnitude pruning Dense");
    boat_dense_layer_t* dl = boat_dense_layer_create(4, 8, false);
    boat_tensor_t* w = boat_dense_layer_get_weight(dl);
    // Fill with varied values
    float* d = (float*)boat_tensor_data(w);
    size_t n = boat_tensor_nelements(w);
    for (size_t i = 0; i < n; i++) d[i] = (float)(i % 10) - 5.0f;

    boat_model_t* model = boat_model_create();
    boat_model_add_layer(model, wrap(dl, BOAT_LAYER_TYPE_DENSE));

    boat_prune_context_t* ctx = boat_prune_context_create(model);
    if (!ctx) FAIL("context is NULL");

    boat_prune_config_t cfg = boat_prune_config_default();
    cfg.sparsity = 0.5f;
    if (!boat_prune_model(ctx, &cfg)) FAIL("prune_model failed");

    float sparsity = boat_compute_sparsity(w);
    if (sparsity < 0.45f) FAIL("sparsity too low");

    boat_prune_context_free(ctx);
    boat_model_free(model);
    PASS(); return 0;
}

// --- Test 5: magnitude pruning on Conv2D layer ---
static int test_magnitude_pruning_conv(void) {
    TEST("Magnitude pruning Conv2D");
    boat_conv_layer_t* cl = boat_conv_layer_create(3, 6, 3, 1, 1, 1);
    boat_tensor_t* w = boat_conv_layer_get_weight(cl);
    float* d = (float*)boat_tensor_data(w);
    size_t n = boat_tensor_nelements(w);
    for (size_t i = 0; i < n; i++) d[i] = (float)(i % 10) - 5.0f;

    boat_model_t* model = boat_model_create();
    boat_model_add_layer(model, wrap(cl, BOAT_LAYER_TYPE_CONV2D));

    boat_prune_context_t* ctx = boat_prune_context_create(model);
    if (!ctx) FAIL("context is NULL");

    boat_prune_config_t cfg = boat_prune_config_default();
    cfg.sparsity = 0.5f;
    if (!boat_prune_model(ctx, &cfg)) FAIL("prune_model failed");

    float sparsity = boat_compute_sparsity(w);
    if (sparsity < 0.45f) FAIL("sparsity too low");

    boat_prune_context_free(ctx);
    boat_model_free(model);
    PASS(); return 0;
}

// --- Test 6: structured pruning Dense (along output_features) ---
static int test_structured_pruning_dense(void) {
    TEST("Structured pruning Dense");
    boat_dense_layer_t* dl = boat_dense_layer_create(4, 8, false);
    boat_tensor_t* w = boat_dense_layer_get_weight(dl);
    float* d = (float*)boat_tensor_data(w);
    // Make some output neurons have large weights, some small
    // shape: [4, 8], dim=1 means 8 output neurons
    for (size_t oc = 0; oc < 8; oc++) {
        float val = (oc < 4) ? 10.0f : 0.1f; // first 4 large, last 4 small
        for (size_t ic = 0; ic < 4; ic++) {
            d[ic * 8 + oc] = val;
        }
    }

    boat_model_t* model = boat_model_create();
    boat_model_add_layer(model, wrap(dl, BOAT_LAYER_TYPE_DENSE));

    boat_prune_context_t* ctx = boat_prune_context_create(model);
    if (!ctx) FAIL("context is NULL");

    boat_prune_config_t cfg = boat_prune_config_default();
    cfg.structured = true;
    cfg.prune_dim = 1;
    cfg.min_keep_ratio = 0.1f;
    cfg.sparsity = 0.5f;
    if (!boat_prune_model(ctx, &cfg)) FAIL("prune_model failed");

    float sparsity = boat_compute_structured_sparsity(w, 1);
    // Should have pruned ~4 of 8 neurons
    if (sparsity < 0.25f) FAIL("structured sparsity too low");

    boat_prune_context_free(ctx);
    boat_model_free(model);
    PASS(); return 0;
}

// --- Test 7: structured pruning Conv2D (along out_channels) ---
static int test_structured_pruning_conv(void) {
    TEST("Structured pruning Conv2D");
    boat_conv_layer_t* cl = boat_conv_layer_create(3, 6, 3, 1, 1, 1);
    boat_tensor_t* w = boat_conv_layer_get_weight(cl);
    float* d = (float*)boat_tensor_data(w);
    // shape: [6, 3, 3, 3] — dim=0 means 6 output filters
    size_t filter_size = 3 * 3 * 3;
    for (size_t oc = 0; oc < 6; oc++) {
        float val = (oc < 3) ? 10.0f : 0.1f; // first 3 large, last 3 small
        for (size_t k = 0; k < filter_size; k++) {
            d[oc * filter_size + k] = val;
        }
    }

    boat_model_t* model = boat_model_create();
    boat_model_add_layer(model, wrap(cl, BOAT_LAYER_TYPE_CONV2D));

    boat_prune_context_t* ctx = boat_prune_context_create(model);
    if (!ctx) FAIL("context is NULL");

    boat_prune_config_t cfg = boat_prune_config_default();
    cfg.structured = true;
    cfg.prune_dim = 0;
    cfg.min_keep_ratio = 0.1f;
    cfg.sparsity = 0.5f;
    if (!boat_prune_model(ctx, &cfg)) FAIL("prune_model failed");

    float sparsity = boat_compute_structured_sparsity(w, 0);
    // Should have pruned ~3 of 6 filters
    if (sparsity < 0.2f) FAIL("structured sparsity too low");

    boat_prune_context_free(ctx);
    boat_model_free(model);
    PASS(); return 0;
}

// --- Test 8: min_keep_ratio in structured pruning ---
static int test_structured_min_keep(void) {
    TEST("Structured min_keep_ratio");
    boat_dense_layer_t* dl = boat_dense_layer_create(2, 10, false);
    boat_tensor_t* w = boat_dense_layer_get_weight(dl);
    float* d = (float*)boat_tensor_data(w);
    // All weights tiny (everything should be pruned, but min_keep prevents it)
    for (size_t i = 0; i < boat_tensor_nelements(w); i++) d[i] = 0.01f;

    boat_model_t* model = boat_model_create();
    boat_model_add_layer(model, wrap(dl, BOAT_LAYER_TYPE_DENSE));

    boat_prune_context_t* ctx = boat_prune_context_create(model);
    if (!ctx) FAIL("context is NULL");

    boat_prune_config_t cfg = boat_prune_config_default();
    cfg.structured = true;
    cfg.prune_dim = 1;
    cfg.min_keep_ratio = 0.3f; // keep at least 30%
    cfg.sparsity = 0.9f;
    if (!boat_prune_model(ctx, &cfg)) FAIL("prune_model failed");

    float sparsity = boat_compute_structured_sparsity(w, 1);
    // Should keep at least 30% = 3 neurons, so sparsity <= 0.7
    if (sparsity > 0.75f) FAIL("pruned too much despite min_keep");

    boat_prune_context_free(ctx);
    boat_model_free(model);
    PASS(); return 0;
}

// --- Test 9: mask re-application (simulating optimizer step) ---
static int test_mask_reapply(void) {
    TEST("Mask re-application");
    int64_t shape[] = {2, 4};
    boat_tensor_t* w = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor(w, 1.0f);

    // Create mask: half zeros
    boat_tensor_t* mask = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* m = (float*)boat_tensor_data(mask);
    size_t n = boat_tensor_nelements(mask);
    for (size_t i = 0; i < n; i++) m[i] = (i < n/2) ? 0.0f : 1.0f;

    // Apply mask
    if (!boat_apply_mask(w, mask)) FAIL("apply_mask failed");

    // Verify zeros
    const float* wd = (const float*)boat_tensor_const_data(w);
    for (size_t i = 0; i < n; i++) {
        if (i < n/2 && wd[i] != 0.0f) FAIL("expected zero after mask");
    }

    // Simulate optimizer step: modify all weights
    float* wd_mut = (float*)boat_tensor_data(w);
    for (size_t i = 0; i < n; i++) wd_mut[i] += 5.0f;

    // Re-apply mask
    for (size_t i = 0; i < n; i++) wd_mut[i] *= m[i];

    // Verify pruned weights are zero again
    for (size_t i = 0; i < n; i++) {
        if (i < n/2 && wd_mut[i] != 0.0f) FAIL("expected zero after re-apply");
        if (i >= n/2 && wd_mut[i] == 0.0f) FAIL("unpruned weight should not be zero");
    }

    boat_tensor_unref(w);
    boat_tensor_unref(mask);
    PASS(); return 0;
}

// --- Test 10: iterative pruning with config ---
static int test_iterative_pruning(void) {
    TEST("Iterative pruning config");
    boat_dense_layer_t* dl = boat_dense_layer_create(4, 8, false);
    boat_tensor_t* w = boat_dense_layer_get_weight(dl);
    float* d = (float*)boat_tensor_data(w);
    size_t n = boat_tensor_nelements(w);
    for (size_t i = 0; i < n; i++) d[i] = (float)(i % 10) - 5.0f;

    boat_model_t* model = boat_model_create();
    boat_model_add_layer(model, wrap(dl, BOAT_LAYER_TYPE_DENSE));

    boat_prune_context_t* ctx = boat_prune_context_create(model);
    if (!ctx) FAIL("context is NULL");

    // Prune once with 30% sparsity
    boat_prune_config_t cfg = boat_prune_config_default();
    cfg.sparsity = 0.3f;
    cfg.iterative_steps = 1;
    if (!boat_prune_model(ctx, &cfg)) FAIL("prune_model failed");

    // Verify some pruning happened
    float sparsity = boat_compute_sparsity(w);
    if (sparsity < 0.25f) FAIL("sparsity too low after first prune");

    // Re-apply masks (simulating finetune step)
    if (!boat_prune_apply_masks(ctx)) FAIL("apply_masks failed");

    boat_prune_context_free(ctx);
    boat_model_free(model);
    PASS(); return 0;
}

// --- Test 11: prune context ---
static int test_prune_context(void) {
    TEST("Prune context");
    boat_dense_layer_t* dl = boat_dense_layer_create(4, 8, true);
    boat_conv_layer_t* cl = boat_conv_layer_create(3, 6, 3, 1, 1, 1);
    boat_relu_layer_t* rl = boat_relu_layer_create();

    boat_model_t* model = boat_model_create();
    boat_model_add_layer(model, wrap(dl, BOAT_LAYER_TYPE_DENSE));
    boat_model_add_layer(model, wrap(cl, BOAT_LAYER_TYPE_CONV2D));
    boat_model_add_layer(model, wrap(rl, BOAT_LAYER_TYPE_RELU));

    boat_prune_context_t* ctx = boat_prune_context_create(model);
    if (!ctx) FAIL("context is NULL");
    // Should have 2 entries (Dense + Conv2D, but not ReLU)

    boat_prune_config_t cfg = boat_prune_config_default();
    cfg.sparsity = 0.3f;
    if (!boat_prune_model(ctx, &cfg)) FAIL("prune_model failed");

    // Verify masks exist for Dense and Conv, but not ReLU
    boat_tensor_t* m0 = boat_prune_get_mask(ctx, 0);
    boat_tensor_t* m1 = boat_prune_get_mask(ctx, 1);
    boat_tensor_t* m2 = boat_prune_get_mask(ctx, 2);
    if (!m0) FAIL("no mask for layer 0");
    if (!m1) FAIL("no mask for layer 1");
    if (m2 != NULL) FAIL("unexpected mask for layer 2");

    boat_prune_context_free(ctx);
    boat_model_free(model);
    PASS(); return 0;
}

// --- Test 12: compute sparsity ---
static int test_compute_sparsity(void) {
    TEST("Compute sparsity");
    int64_t shape[] = {4, 4};
    boat_tensor_t* t = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(t);
    size_t n = boat_tensor_nelements(t);
    // Set first half to zero
    for (size_t i = 0; i < n; i++) d[i] = (i < n/2) ? 0.0f : 1.0f;

    float s = boat_compute_sparsity(t);
    if (fabsf(s - 0.5f) > 0.01f) FAIL("expected sparsity=0.5");

    // All zeros
    for (size_t i = 0; i < n; i++) d[i] = 0.0f;
    s = boat_compute_sparsity(t);
    if (fabsf(s - 1.0f) > 0.01f) FAIL("expected sparsity=1.0");

    // No zeros
    for (size_t i = 0; i < n; i++) d[i] = 1.0f;
    s = boat_compute_sparsity(t);
    if (fabsf(s - 0.0f) > 0.01f) FAIL("expected sparsity=0.0");

    boat_tensor_unref(t);
    PASS(); return 0;
}

// --- Test 13: compute structured sparsity ---
static int test_compute_structured_sparsity(void) {
    TEST("Compute structured sparsity");
    int64_t shape[] = {4, 2, 3}; // 4 channels along dim=0
    boat_tensor_t* t = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(t);
    size_t n = boat_tensor_nelements(t);
    // First 3 channels zero, last 1 non-zero
    size_t channel_size = n / 4;
    for (size_t i = 0; i < n; i++) d[i] = (i < 3 * channel_size) ? 0.0f : 1.0f;

    float s = boat_compute_structured_sparsity(t, 0);
    if (fabsf(s - 0.75f) > 0.01f) FAIL("expected structured sparsity=0.75");

    boat_tensor_unref(t);
    PASS(); return 0;
}

// --- Test 14: prune + QAT fake quantize ---
static int test_prune_fake_quantize(void) {
    TEST("Prune + QAT fake quantize");
    boat_dense_layer_t* dl = boat_dense_layer_create(4, 8, false);
    boat_tensor_t* w = boat_dense_layer_get_weight(dl);
    float* d = (float*)boat_tensor_data(w);
    size_t n = boat_tensor_nelements(w);
    for (size_t i = 0; i < n; i++) d[i] = (float)(i % 10) - 5.0f;

    boat_model_t* model = boat_model_create();
    boat_model_add_layer(model, wrap(dl, BOAT_LAYER_TYPE_DENSE));

    boat_prune_context_t* ctx = boat_prune_context_create(model);
    if (!ctx) FAIL("context is NULL");

    // Prune
    boat_prune_config_t pcfg = boat_prune_config_default();
    pcfg.sparsity = 0.3f;
    if (!boat_prune_model(ctx, &pcfg)) FAIL("prune_model failed");

    float sparsity_before = boat_compute_sparsity(w);

    // Apply QAT fake quantize
    boat_quant_config_t qcfg = boat_quant_config_default();
    if (!boat_prune_fake_quantize_model(ctx, &qcfg)) FAIL("fake_quantize_model failed");

    // Weight should still be FP32
    if (boat_tensor_dtype(w) != BOAT_DTYPE_FLOAT32) FAIL("weight dtype changed from FP32");

    // Sparsity should be preserved
    float sparsity_after = boat_compute_sparsity(w);
    if (fabsf(sparsity_after - sparsity_before) > 0.1f) FAIL("sparsity changed after QAT");

    boat_prune_context_free(ctx);
    boat_model_free(model);
    PASS(); return 0;
}

// --- Test 15: prune then forward pass ---
static int test_prune_forward(void) {
    TEST("Prune + forward pass");
    boat_dense_layer_t* dl = boat_dense_layer_create(4, 8, false);
    boat_tensor_t* w = boat_dense_layer_get_weight(dl);
    float* d = (float*)boat_tensor_data(w);
    size_t n = boat_tensor_nelements(w);
    for (size_t i = 0; i < n; i++) d[i] = (float)(i % 10) - 5.0f;

    boat_model_t* model = boat_model_create();
    boat_model_add_layer(model, wrap(dl, BOAT_LAYER_TYPE_DENSE));

    boat_prune_context_t* ctx = boat_prune_context_create(model);
    if (!ctx) FAIL("context is NULL");

    boat_prune_config_t cfg = boat_prune_config_default();
    cfg.sparsity = 0.5f;
    if (!boat_prune_model(ctx, &cfg)) FAIL("prune_model failed");

    // Run forward pass
    int64_t in_shape[] = {2, 4};
    boat_tensor_t* input = boat_tensor_create(in_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor(input, 0.5f);

    boat_tensor_t* output = boat_model_forward(model, input);
    if (!output) FAIL("forward returned NULL");

    const int64_t* out_shape = boat_tensor_shape(output);
    if (out_shape[0] != 2 || out_shape[1] != 8) FAIL("output shape mismatch");

    boat_tensor_unref(input);
    boat_tensor_unref(output);
    boat_prune_context_free(ctx);
    boat_model_free(model);
    PASS(); return 0;
}

// --- Test 16: remove mask ---
static int test_remove_mask(void) {
    TEST("Remove mask");
    boat_dense_layer_t* dl = boat_dense_layer_create(4, 8, false);
    boat_model_t* model = boat_model_create();
    boat_model_add_layer(model, wrap(dl, BOAT_LAYER_TYPE_DENSE));

    boat_prune_context_t* ctx = boat_prune_context_create(model);
    boat_prune_config_t cfg = boat_prune_config_default();
    cfg.sparsity = 0.3f;
    if (!boat_prune_model(ctx, &cfg)) FAIL("prune_model failed");
    if (!boat_prune_get_mask(ctx, 0)) FAIL("mask should exist");

    if (!boat_prune_remove_mask(ctx, 0)) FAIL("remove_mask failed");
    if (boat_prune_get_mask(ctx, 0) != NULL) FAIL("mask should be NULL after remove");

    boat_prune_context_free(ctx);
    boat_model_free(model);
    PASS(); return 0;
}

// --- Test 17: remove all masks ---
static int test_remove_all_masks(void) {
    TEST("Remove all masks");
    boat_dense_layer_t* dl = boat_dense_layer_create(4, 8, false);
    boat_conv_layer_t* cl = boat_conv_layer_create(3, 6, 3, 1, 1, 1);
    boat_model_t* model = boat_model_create();
    boat_model_add_layer(model, wrap(dl, BOAT_LAYER_TYPE_DENSE));
    boat_model_add_layer(model, wrap(cl, BOAT_LAYER_TYPE_CONV2D));

    boat_prune_context_t* ctx = boat_prune_context_create(model);
    boat_prune_config_t cfg = boat_prune_config_default();
    cfg.sparsity = 0.3f;
    if (!boat_prune_model(ctx, &cfg)) FAIL("prune_model failed");

    boat_prune_remove_all_masks(ctx);
    if (boat_prune_get_mask(ctx, 0) != NULL) FAIL("mask 0 should be NULL");
    if (boat_prune_get_mask(ctx, 1) != NULL) FAIL("mask 1 should be NULL");

    boat_prune_context_free(ctx);
    boat_model_free(model);
    PASS(); return 0;
}

// --- Test 18: magnitude pruning with custom threshold ---
static int test_magnitude_threshold(void) {
    TEST("Magnitude pruning custom threshold");
    int64_t shape[] = {4, 4};
    boat_tensor_t* w = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(w);
    size_t n = boat_tensor_nelements(w);
    for (size_t i = 0; i < n; i++) d[i] = (float)(i % 16) - 8.0f; // values -8..7

    boat_tensor_t* mask = boat_create_magnitude_mask(w, 3.0f);
    if (!mask) FAIL("mask is NULL");

    const float* m = (const float*)boat_tensor_const_data(mask);
    const float* wd = (const float*)boat_tensor_const_data(w);
    for (size_t i = 0; i < n; i++) {
        int expected = (fabsf(wd[i]) <= 3.0f) ? 0 : 1;
        if ((int)m[i] != expected) {
            char buf[128];
            snprintf(buf, sizeof(buf), "mask[%zu]=%d expected %d (val=%.1f)", i, (int)m[i], expected, wd[i]);
            FAIL(buf);
        }
    }

    boat_tensor_unref(w);
    boat_tensor_unref(mask);
    PASS(); return 0;
}

static int run_all_tests(void) {
    int failed = 0;
    failed += test_compute_threshold();
    failed += test_magnitude_mask();
    failed += test_apply_mask();
    failed += test_magnitude_pruning_dense();
    failed += test_magnitude_pruning_conv();
    failed += test_structured_pruning_dense();
    failed += test_structured_pruning_conv();
    failed += test_structured_min_keep();
    failed += test_mask_reapply();
    failed += test_iterative_pruning();
    failed += test_prune_context();
    failed += test_compute_sparsity();
    failed += test_compute_structured_sparsity();
    failed += test_prune_fake_quantize();
    failed += test_prune_forward();
    failed += test_remove_mask();
    failed += test_remove_all_masks();
    failed += test_magnitude_threshold();
    return failed;
}

int main(void) {
    boat_init();
    printf("Pruning tests:\n");
    int failed = run_all_tests();
    printf("  %d/%d passed, %d failed\n", tests_passed, tests_total, failed);
    boat_cleanup();
    return failed;
}
