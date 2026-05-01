// test_serialization_integration.c - End-to-end model serialization tests
#include <boat/model.h>
#include <boat/layers.h>
#include <boat/optimizers.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { printf("  %s ... ", name); tests_total++; } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)

// Helper: create a boat_layer_t wrapper
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

// Helper: compare two float tensors byte-exact
static int tensors_equal(const boat_tensor_t* a, const boat_tensor_t* b) {
    if (!a || !b) return 0;
    if (boat_tensor_ndim(a) != boat_tensor_ndim(b)) return 0;
    if (boat_tensor_nbytes(a) != boat_tensor_nbytes(b)) return 0;
    const int64_t* sa = boat_tensor_shape(a);
    const int64_t* sb = boat_tensor_shape(b);
    for (size_t i = 0; i < boat_tensor_ndim(a); i++)
        if (sa[i] != sb[i]) return 0;
    return memcmp(boat_tensor_const_data(a), boat_tensor_const_data(b), boat_tensor_nbytes(a)) == 0;
}

// Forward pass through multi-layer model (manual, like MNIST)
static boat_tensor_t* manual_forward(
    boat_conv_layer_t* conv1, boat_relu_layer_t* r1, boat_pool_layer_t* p1,
    boat_conv_layer_t* conv2, boat_relu_layer_t* r2, boat_pool_layer_t* p2,
    boat_flatten_layer_t* flat,
    boat_dense_layer_t* fc1, boat_relu_layer_t* r3, boat_dense_layer_t* fc2,
    boat_softmax_layer_t* sm,
    const boat_tensor_t* input)
{
    boat_tensor_t* x;
    x = boat_conv_layer_forward(conv1, input);
    boat_tensor_t* t = boat_relu_layer_forward(r1, x);     boat_tensor_unref(x); x = t;
    t = boat_pool_layer_forward(p1, x);                     boat_tensor_unref(x); x = t;
    t = boat_conv_layer_forward(conv2, x);                  boat_tensor_unref(x); x = t;
    t = boat_relu_layer_forward(r2, x);                     boat_tensor_unref(x); x = t;
    t = boat_pool_layer_forward(p2, x);                     boat_tensor_unref(x); x = t;
    t = boat_flatten_layer_forward(flat, x);                boat_tensor_unref(x); x = t;
    t = boat_dense_layer_forward(fc1, x);                   boat_tensor_unref(x); x = t;
    t = boat_relu_layer_forward(r3, x);                     boat_tensor_unref(x); x = t;
    t = boat_dense_layer_forward(fc2, x);                   boat_tensor_unref(x); x = t;
    t = boat_softmax_layer_forward(sm, x);                  boat_tensor_unref(x); x = t;
    return x;
}

// --- Test 1: Dense layer round-trip ---
static int test_dense_roundtrip(void) {
    TEST("Dense round-trip (with bias)");
    boat_model_t* m = boat_model_create();
    boat_dense_layer_t* d = boat_dense_layer_create(4, 3, true);
    fill_tensor(boat_dense_layer_get_weight(d), 1.0f);
    fill_tensor(boat_dense_layer_get_bias(d), 10.0f);
    boat_model_add_layer(m, wrap(d, BOAT_LAYER_TYPE_DENSE));
    if (!boat_model_save(m, "test_dense.boat")) FAIL("save failed");

    // Save original data
    size_t w_bytes = boat_tensor_nbytes(boat_dense_layer_get_weight(d));
    size_t b_bytes = boat_tensor_nbytes(boat_dense_layer_get_bias(d));
    float* w_orig = malloc(w_bytes); memcpy(w_orig, boat_tensor_const_data(boat_dense_layer_get_weight(d)), w_bytes);
    float* b_orig = malloc(b_bytes); memcpy(b_orig, boat_tensor_const_data(boat_dense_layer_get_bias(d)), b_bytes);
    boat_model_free(m);

    boat_model_t* loaded = boat_model_load("test_dense.boat");
    if (!loaded) FAIL("load returned NULL");
    if (boat_model_layer_count(loaded) != 1) FAIL("wrong layer count");
    boat_layer_t* l0 = boat_model_get_layer(loaded, 0);
    if (!l0) FAIL("layer is NULL");
    if (l0->type != BOAT_LAYER_TYPE_DENSE) FAIL("wrong type");
    boat_dense_layer_t* ld = (boat_dense_layer_t*)l0->data;
    if (memcmp(w_orig, boat_tensor_const_data(boat_dense_layer_get_weight(ld)), w_bytes) != 0)
        FAIL("weight mismatch");
    if (memcmp(b_orig, boat_tensor_const_data(boat_dense_layer_get_bias(ld)), b_bytes) != 0)
        FAIL("bias mismatch");
    if (!l0->ops) FAIL("ops not set");
    boat_model_free(loaded); free(w_orig); free(b_orig);
    PASS(); return 0;
}

// --- Test 2: Dense without bias ---
static int test_dense_no_bias(void) {
    TEST("Dense round-trip (no bias)");
    boat_model_t* m = boat_model_create();
    boat_dense_layer_t* d = boat_dense_layer_create(4, 3, false);
    fill_tensor(boat_dense_layer_get_weight(d), 2.0f);
    boat_model_add_layer(m, wrap(d, BOAT_LAYER_TYPE_DENSE));
    if (!boat_model_save(m, "test_dense_nb.boat")) FAIL("save failed");

    size_t w_bytes = boat_tensor_nbytes(boat_dense_layer_get_weight(d));
    float* w_orig = malloc(w_bytes);
    memcpy(w_orig, boat_tensor_const_data(boat_dense_layer_get_weight(d)), w_bytes);
    boat_model_free(m);

    boat_model_t* loaded = boat_model_load("test_dense_nb.boat");
    if (!loaded) FAIL("load returned NULL");
    boat_layer_t* l0 = boat_model_get_layer(loaded, 0);
    boat_dense_layer_t* ld = (boat_dense_layer_t*)l0->data;
    if (boat_dense_layer_get_bias(ld) != NULL) FAIL("bias should be NULL");
    if (memcmp(w_orig, boat_tensor_const_data(boat_dense_layer_get_weight(ld)), w_bytes) != 0)
        FAIL("weight mismatch");
    boat_model_free(loaded); free(w_orig);
    PASS(); return 0;
}

// --- Test 3: Conv2D round-trip ---
static int test_conv_roundtrip(void) {
    TEST("Conv2D round-trip");
    boat_model_t* m = boat_model_create();
    boat_conv_layer_t* c = boat_conv_layer_create(1, 4, 3, 1, 1);
    fill_tensor(boat_conv_layer_get_weight(c), 3.0f);
    fill_tensor(boat_conv_layer_get_bias(c), 30.0f);
    boat_model_add_layer(m, wrap(c, BOAT_LAYER_TYPE_CONV2D));
    if (!boat_model_save(m, "test_conv.boat")) FAIL("save failed");

    size_t w_bytes = boat_tensor_nbytes(boat_conv_layer_get_weight(c));
    size_t b_bytes = boat_tensor_nbytes(boat_conv_layer_get_bias(c));
    float* w_orig = malloc(w_bytes); memcpy(w_orig, boat_tensor_const_data(boat_conv_layer_get_weight(c)), w_bytes);
    float* b_orig = malloc(b_bytes); memcpy(b_orig, boat_tensor_const_data(boat_conv_layer_get_bias(c)), b_bytes);
    boat_model_free(m);

    boat_model_t* loaded = boat_model_load("test_conv.boat");
    if (!loaded) FAIL("load returned NULL");
    boat_layer_t* l0 = boat_model_get_layer(loaded, 0);
    if (l0->type != BOAT_LAYER_TYPE_CONV2D) FAIL("wrong type");
    boat_conv_layer_t* lc = (boat_conv_layer_t*)l0->data;
    if (memcmp(w_orig, boat_tensor_const_data(boat_conv_layer_get_weight(lc)), w_bytes) != 0)
        FAIL("weight mismatch");
    if (memcmp(b_orig, boat_tensor_const_data(boat_conv_layer_get_bias(lc)), b_bytes) != 0)
        FAIL("bias mismatch");
    if (boat_conv_layer_get_stride(lc) != 1 || boat_conv_layer_get_padding(lc) != 1)
        FAIL("hyperparams mismatch");
    if (!l0->ops) FAIL("ops not set");
    boat_model_free(loaded); free(w_orig); free(b_orig);
    PASS(); return 0;
}

// --- Test 4: MaxPool2D + Softmax round-trip ---
static int test_paramless_roundtrip(void) {
    TEST("MaxPool2D + Softmax round-trip");
    boat_model_t* m = boat_model_create();
    boat_pool_layer_t* p = boat_pool_layer_create(2, 2, 0);
    boat_softmax_layer_t* s = boat_softmax_layer_create(-1);
    boat_model_add_layer(m, wrap(p, BOAT_LAYER_TYPE_MAXPOOL2D));
    boat_model_add_layer(m, wrap(s, BOAT_LAYER_TYPE_SOFTMAX));
    if (!boat_model_save(m, "test_paramless.boat")) FAIL("save failed");
    boat_model_free(m);

    boat_model_t* loaded = boat_model_load("test_paramless.boat");
    if (!loaded) FAIL("load returned NULL");
    if (boat_model_layer_count(loaded) != 2) FAIL("wrong layer count");

    boat_layer_t* lp = boat_model_get_layer(loaded, 0);
    if (lp->type != BOAT_LAYER_TYPE_MAXPOOL2D) FAIL("layer 0 wrong type");
    boat_pool_layer_t* pl = (boat_pool_layer_t*)lp->data;
    if (boat_pool_layer_get_pool_size(pl) != 2 || boat_pool_layer_get_stride(pl) != 2 ||
        boat_pool_layer_get_padding(pl) != 0) FAIL("pool HP mismatch");
    if (!lp->ops) FAIL("pool ops not set");

    boat_layer_t* ls = boat_model_get_layer(loaded, 1);
    if (ls->type != BOAT_LAYER_TYPE_SOFTMAX) FAIL("layer 1 wrong type");
    boat_softmax_layer_t* sl = (boat_softmax_layer_t*)ls->data;
    if (boat_softmax_layer_get_axis(sl) != -1) FAIL("softmax axis mismatch");
    if (!ls->ops) FAIL("softmax ops not set");

    boat_model_free(loaded);
    PASS(); return 0;
}

// --- Test 5: Multi-layer model (MNIST architecture) ---
static int test_mnist_model_roundtrip(void) {
    TEST("MNIST architecture round-trip");
    boat_model_t* m = boat_model_create();
    boat_conv_layer_t* c1 = boat_conv_layer_create(1, 32, 3, 1, 1);
    boat_relu_layer_t* r1 = boat_relu_layer_create();
    boat_pool_layer_t* p1 = boat_pool_layer_create(2, 2, 0);
    boat_conv_layer_t* c2 = boat_conv_layer_create(32, 64, 3, 1, 1);
    boat_relu_layer_t* r2 = boat_relu_layer_create();
    boat_pool_layer_t* p2 = boat_pool_layer_create(2, 2, 0);
    boat_flatten_layer_t* f = boat_flatten_layer_create();
    boat_dense_layer_t* d1 = boat_dense_layer_create(3136, 128, true);
    boat_relu_layer_t* r3 = boat_relu_layer_create();
    boat_dense_layer_t* d2 = boat_dense_layer_create(128, 10, true);
    boat_softmax_layer_t* sm = boat_softmax_layer_create(-1);

    fill_tensor(boat_conv_layer_get_weight(c1), 1.0f);
    fill_tensor(boat_conv_layer_get_bias(c1), 0.1f);
    fill_tensor(boat_conv_layer_get_weight(c2), 2.0f);
    fill_tensor(boat_conv_layer_get_bias(c2), 0.2f);
    fill_tensor(boat_dense_layer_get_weight(d1), 3.0f);
    fill_tensor(boat_dense_layer_get_bias(d1), 0.3f);
    fill_tensor(boat_dense_layer_get_weight(d2), 4.0f);
    fill_tensor(boat_dense_layer_get_bias(d2), 0.4f);

    boat_model_add_layer(m, wrap(c1, BOAT_LAYER_TYPE_CONV2D));
    boat_model_add_layer(m, wrap(r1, BOAT_LAYER_TYPE_RELU));
    boat_model_add_layer(m, wrap(p1, BOAT_LAYER_TYPE_MAXPOOL2D));
    boat_model_add_layer(m, wrap(c2, BOAT_LAYER_TYPE_CONV2D));
    boat_model_add_layer(m, wrap(r2, BOAT_LAYER_TYPE_RELU));
    boat_model_add_layer(m, wrap(p2, BOAT_LAYER_TYPE_MAXPOOL2D));
    boat_model_add_layer(m, wrap(f, BOAT_LAYER_TYPE_FLATTEN));
    boat_model_add_layer(m, wrap(d1, BOAT_LAYER_TYPE_DENSE));
    boat_model_add_layer(m, wrap(r3, BOAT_LAYER_TYPE_RELU));
    boat_model_add_layer(m, wrap(d2, BOAT_LAYER_TYPE_DENSE));
    boat_model_add_layer(m, wrap(sm, BOAT_LAYER_TYPE_SOFTMAX));

    // Run forward before save to compare later
    int64_t in_shape[] = {1, 1, 28, 28};
    boat_tensor_t* input = boat_tensor_create(in_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor(input, 0.5f);
    boat_tensor_t* orig_out = manual_forward(c1, r1, p1, c2, r2, p2, f, d1, r3, d2, sm, input);

    if (!boat_model_save(m, "test_mnist.boat")) FAIL("save failed");
    boat_model_free(m);

    boat_model_t* loaded = boat_model_load("test_mnist.boat");
    if (!loaded) FAIL("load returned NULL");
    if (boat_model_layer_count(loaded) != 11) FAIL("wrong layer count");

    boat_layer_t* types[11];
    boat_layer_type_t expected[] = {
        BOAT_LAYER_TYPE_CONV2D, BOAT_LAYER_TYPE_RELU, BOAT_LAYER_TYPE_MAXPOOL2D,
        BOAT_LAYER_TYPE_CONV2D, BOAT_LAYER_TYPE_RELU, BOAT_LAYER_TYPE_MAXPOOL2D,
        BOAT_LAYER_TYPE_FLATTEN, BOAT_LAYER_TYPE_DENSE, BOAT_LAYER_TYPE_RELU,
        BOAT_LAYER_TYPE_DENSE, BOAT_LAYER_TYPE_SOFTMAX
    };
    for (int i = 0; i < 11; i++) {
        types[i] = boat_model_get_layer(loaded, i);
        if (!types[i]) FAIL("layer NULL");
        if (types[i]->type != expected[i]) FAIL("type mismatch");
        if (!types[i]->ops) FAIL("ops not set");
    }

    // Compare forward output
    boat_tensor_t* loaded_out = boat_model_forward(loaded, input);
    if (!loaded_out) FAIL("forward pass failed on loaded model");
    if (!tensors_equal(orig_out, loaded_out)) FAIL("forward output mismatch");
    boat_tensor_unref(orig_out);
    boat_tensor_unref(loaded_out);
    boat_tensor_unref(input);
    boat_model_free(loaded);
    PASS(); return 0;
}

// --- Test 6: Train -> save -> load -> inference ---
static int test_train_save_load_infer(void) {
    TEST("Train -> save -> load -> inference");
    // Small model: Dense(4->8) -> ReLU -> Dense(8->2) -> Softmax
    boat_dense_layer_t* fc1 = boat_dense_layer_create(4, 8, true);
    boat_relu_layer_t* relu = boat_relu_layer_create();
    boat_dense_layer_t* fc2 = boat_dense_layer_create(8, 2, true);
    boat_softmax_layer_t* sm = boat_softmax_layer_create(-1);

    // Synthetic training data: 32 samples, 4 features, 2 classes
    int batch_size = 32, n_features = 4, n_classes = 2, n_epochs = 5;
    int64_t x_shape[] = {batch_size, n_features};
    int64_t y_shape[] = {batch_size};
    boat_tensor_t* x = boat_tensor_create(x_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* y = boat_tensor_create(y_shape, 1, BOAT_DTYPE_INT64, BOAT_DEVICE_CPU);
    int* y_data = (int*)boat_tensor_data(y);
    float* x_data = (float*)boat_tensor_data(x);
    for (int i = 0; i < batch_size; i++) {
        y_data[i] = i % 2;
        x_data[i * 4 + 0] = (float)(i % 2);
        x_data[i * 4 + 1] = (float)((i + 1) % 2);
        x_data[i * 4 + 2] = (float)(i % 3) * 0.5f;
        x_data[i * 4 + 3] = (float)(i % 4) * 0.25f;
    }

    // Create optimizer
    boat_layer_t* train_wrappers[4];
    train_wrappers[0] = wrap(fc1, BOAT_LAYER_TYPE_DENSE);
    train_wrappers[1] = wrap(relu, BOAT_LAYER_TYPE_RELU);
    train_wrappers[2] = wrap(fc2, BOAT_LAYER_TYPE_DENSE);
    train_wrappers[3] = wrap(sm, BOAT_LAYER_TYPE_SOFTMAX);
    // Set ops for training
    static const boat_layer_ops_t dense_ops = {
        .forward = NULL, .backward = NULL, .update = NULL, .free = NULL
    };
    static const boat_layer_ops_t relu_ops = {
        .forward = NULL, .backward = NULL, .update = NULL, .free = NULL
    };
    static const boat_layer_ops_t sm_ops = {
        .forward = NULL, .backward = NULL, .update = NULL, .free = NULL
    };
    // Use NULL ops -- we train manually, ops aren't needed for raw layer API
    // The ops are only needed for save/load

    // Manual training loop
    float lr = 0.01f;
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        // Forward
        boat_tensor_t* a1 = boat_dense_layer_forward(fc1, x);
        boat_tensor_t* a2 = boat_relu_layer_forward(relu, a1); boat_tensor_unref(a1);
        boat_tensor_t* a3 = boat_dense_layer_forward(fc2, a2); boat_tensor_unref(a2);
        boat_tensor_t* out = boat_softmax_layer_forward(sm, a3); boat_tensor_unref(a3);

        // Gradient: out - one_hot(y) / batch_size
        float* out_data = (float*)boat_tensor_data(out);
        int64_t out_shape[] = {batch_size, n_classes};
        boat_tensor_t* grad = boat_tensor_create(out_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        float* grad_data = (float*)boat_tensor_data(grad);
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < n_classes; j++) {
                grad_data[i * 2 + j] = out_data[i * 2 + j];
            }
            grad_data[i * 2 + y_data[i]] -= 1.0f;
        }
        // Scale by 1/batch_size
        for (int i = 0; i < batch_size * n_classes; i++) grad_data[i] /= (float)batch_size;

        // Backward
        boat_tensor_t* g = grad;
        boat_tensor_t* t;
        t = boat_softmax_layer_backward(sm, g);                if (t) { boat_tensor_unref(g); g = t; } else break;
        t = boat_dense_layer_backward(fc2, g);                 if (t) { boat_tensor_unref(g); g = t; } else break;
        t = boat_relu_layer_backward(relu, g);                 if (t) { boat_tensor_unref(g); g = t; } else break;
        t = boat_dense_layer_backward(fc1, g);                 if (t) { boat_tensor_unref(g); g = t; } else break;
        boat_tensor_unref(g);

        // Update
        boat_dense_layer_update(fc1, lr);
        boat_dense_layer_update(fc2, lr);
        boat_tensor_unref(out);
    }

    // Run inference on trained model
    boat_tensor_t* orig_infer_out;
    {
        boat_tensor_t* a1 = boat_dense_layer_forward(fc1, x);
        boat_tensor_t* a2 = boat_relu_layer_forward(relu, a1); boat_tensor_unref(a1);
        boat_tensor_t* a3 = boat_dense_layer_forward(fc2, a2); boat_tensor_unref(a2);
        orig_infer_out = boat_softmax_layer_forward(sm, a3); boat_tensor_unref(a3);
    }

    // Save trained model
    boat_model_t* save_m = boat_model_create();
    // Re-use train_wrappers (ops already set, but they have NULL forward -- fine for save)
    for (int i = 0; i < 4; i++)
        boat_model_add_layer(save_m, train_wrappers[i]);
    if (!boat_model_save(save_m, "test_trained.boat")) FAIL("save failed");
    boat_model_free(save_m);

    // Load and run inference on loaded model
    boat_model_t* loaded = boat_model_load("test_trained.boat");
    if (!loaded) FAIL("load returned NULL");
    boat_tensor_t* loaded_out = boat_model_forward(loaded, x);
    if (!loaded_out) FAIL("forward on loaded model returned NULL");

    // Compare outputs
    if (!tensors_equal(orig_infer_out, loaded_out)) FAIL("inference output mismatch");
    boat_tensor_unref(orig_infer_out);
    boat_tensor_unref(loaded_out);

    // Continue training on loaded model (reset optimizer)
    boat_optimizer_t* opt = boat_adam_optimizer_create(lr, 0.9f, 0.999f, 1e-8f);
    if (!opt) FAIL("optimizer create failed");

    // Register parameters from loaded model's trainable layers
    size_t n_layers = boat_model_layer_count(loaded);
    for (size_t i = 0; i < n_layers; i++) {
        boat_layer_t* layer = boat_model_get_layer(loaded, i);
        if (layer->type == BOAT_LAYER_TYPE_DENSE) {
            boat_dense_layer_t* dl = (boat_dense_layer_t*)layer->data;
            boat_optimizer_add_parameter(opt, boat_dense_layer_get_weight(dl), boat_dense_layer_get_grad_weight(dl));
            boat_tensor_t* bias = boat_dense_layer_get_bias(dl);
            boat_tensor_t* grad_bias = boat_dense_layer_get_grad_bias(dl);
            if (bias && grad_bias)
                boat_optimizer_add_parameter(opt, bias, grad_bias);
        }
    }

    for (int epoch = 0; epoch < 2; epoch++) {
        boat_tensor_t* out = boat_model_forward(loaded, x);
        if (!out) FAIL("forward failed during continued training");

        // Compute gradient: (out - one_hot(y)) / batch_size
        float* od = (float*)boat_tensor_data(out);
        int64_t g_shape[] = {batch_size, n_classes};
        boat_tensor_t* g = boat_tensor_create(g_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        float* gd = (float*)boat_tensor_data(g);
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < n_classes; j++)
                gd[i * 2 + j] = od[i * 2 + j];
            gd[i * 2 + y_data[i]] -= 1.0f;
            gd[i * 2] /= (float)batch_size;
            gd[i * 2 + 1] /= (float)batch_size;
        }

        // Manual backward pass in reverse layer order (model_backward is a stub)
        boat_tensor_t* grad = g;
        for (int i = (int)n_layers - 1; i >= 0; i--) {
            boat_layer_t* layer = boat_model_get_layer(loaded, (size_t)i);
            if (!layer->ops || !layer->ops->backward) {
                boat_tensor_unref(grad); grad = NULL;
                break;
            }
            boat_tensor_t* next_grad = layer->ops->backward(layer, grad);
            if (next_grad) { boat_tensor_unref(grad); grad = next_grad; }
            else break;
        }
        if (grad) boat_tensor_unref(grad);

        boat_optimizer_step(opt);
        boat_optimizer_zero_grad(opt);
        boat_tensor_unref(out);
    }
    boat_optimizer_free(opt);

    boat_tensor_unref(x); boat_tensor_unref(y);
    boat_model_free(loaded);
    PASS(); return 0;
}

// --- Test 7: Edge cases ---
static int test_edge_cases(void) {
    int ok = 1;

    // Edge cases: use manual printf/PASS instead of TEST macro since we track inline
    printf("  save(NULL, file) returns false ... ");
    if (boat_model_save(NULL, "test.boat") != false) { printf("FAIL\n"); ok = 0; } else { printf("PASS\n"); } tests_passed++; tests_total++;

    printf("  save(model, NULL) returns false ... ");
    boat_model_t* m = boat_model_create();
    if (boat_model_save(m, NULL) != false) { printf("FAIL\n"); ok = 0; } else { printf("PASS\n"); } tests_passed++; tests_total++;
    boat_model_free(m);

    printf("  save(empty_model) returns false ... ");
    m = boat_model_create();
    if (boat_model_save(m, "test_empty.boat") != false) { printf("FAIL\n"); ok = 0; } else { printf("PASS\n"); } tests_passed++; tests_total++;
    boat_model_free(m);

    printf("  load(NULL) returns NULL ... ");
    if (boat_model_load(NULL) != NULL) { printf("FAIL\n"); ok = 0; } else { printf("PASS\n"); } tests_passed++; tests_total++;

    printf("  load(nonexistent) returns NULL ... ");
    if (boat_model_load("nonexistent_file.boat") != NULL) { printf("FAIL\n"); ok = 0; } else { printf("PASS\n"); } tests_passed++; tests_total++;

    printf("  load(corrupted magic) returns NULL ... ");
    FILE* f = fopen("bad_magic.boat", "wb");
    uint32_t bad_magic = 0xDEADBEEF, ver = 1, cnt = 1;
    fwrite(&bad_magic, 4, 1, f);
    fwrite(&ver, 4, 1, f);
    fwrite(&cnt, 4, 1, f);
    fclose(f);
    if (boat_model_load("bad_magic.boat") != NULL) { printf("FAIL\n"); ok = 0; } else { printf("PASS\n"); } tests_passed++; tests_total++;

    return ok ? 0 : 1;
}

int main(void) {
    printf("Serialization Integration Tests\n");
    printf("===============================\n\n");

    int fail = 0;
    fail |= test_dense_roundtrip();
    fail |= test_dense_no_bias();
    fail |= test_conv_roundtrip();
    fail |= test_paramless_roundtrip();
    fail |= test_mnist_model_roundtrip();
    fail |= test_train_save_load_infer();
    fail |= test_edge_cases();

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    remove("test_dense.boat");
    remove("test_dense_nb.boat");
    remove("test_conv.boat");
    remove("test_paramless.boat");
    remove("test_mnist.boat");
    remove("test_trained.boat");
    remove("test_empty.boat");
    remove("bad_magic.boat");
    return fail ? 1 : 0;
}
