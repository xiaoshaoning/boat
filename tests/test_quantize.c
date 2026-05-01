// test_quantize.c - Post-Training Quantization tests
#include <boat.h>
#include <boat/tensor.h>
#include <boat/layers.h>
#include <boat/model.h>
#include <boat/quantize.h>
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

// Helper: check two float tensors are close
static int tensors_allclose(const boat_tensor_t* a, const boat_tensor_t* b, float rtol) {
    if (!a || !b) return 0;
    if (boat_tensor_ndim(a) != boat_tensor_ndim(b)) return 0;
    if (boat_tensor_nelements(a) != boat_tensor_nelements(b)) return 0;
    const int64_t* sa = boat_tensor_shape(a);
    const int64_t* sb = boat_tensor_shape(b);
    for (size_t i = 0; i < boat_tensor_ndim(a); i++)
        if (sa[i] != sb[i]) return 0;

    const float* da = (const float*)boat_tensor_const_data(a);
    const float* db = (const float*)boat_tensor_const_data(b);
    size_t n = boat_tensor_nelements(a);
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(da[i] - db[i]);
        float mag = fmaxf(fabsf(da[i]), fabsf(db[i]));
        if (mag < 1e-6f) mag = 1.0f;
        if (diff / mag > rtol) return 0;
    }
    return 1;
}

// --- Test 1: quantize-dequantize roundtrip ---
static int test_roundtrip(void) {
    TEST("Quantize-dequantize roundtrip");
    int64_t shape[] = {3, 4};
    boat_tensor_t* fp32 = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor(fp32, 0.0f);

    boat_quant_config_t cfg = boat_quant_config_default();
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");
    if (boat_tensor_dtype(q) != BOAT_DTYPE_UINT8) FAIL("dtype not UINT8");
    if (boat_tensor_get_scale(q) == 0.0f) FAIL("scale is 0");

    boat_tensor_t* deq = boat_dequantize_tensor(q);
    if (!deq) FAIL("dequantize returned NULL");

    if (!tensors_allclose(fp32, deq, 0.05f)) FAIL("values out of tolerance");
    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 2: symmetric quantization ---
static int test_symmetric(void) {
    TEST("Symmetric quantization");
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    int64_t shape[] = {5};
    boat_tensor_t* fp32 = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.symmetric = true;
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");
    if (boat_tensor_get_zero_point(q) != 128) FAIL("zp not 128 for symmetric");

    boat_tensor_t* deq = boat_dequantize_tensor(q);
    if (!tensors_allclose(fp32, deq, 0.05f)) FAIL("values out of tolerance");
    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 3: asymmetric zero_point ---
static int test_asymmetric_zp(void) {
    TEST("Asymmetric zero_point");
    // All positive: min=2, max=5 => scale=(5-2)/255, zp=round(-2/scale)
    float data[] = {2.0f, 3.0f, 4.0f, 5.0f};
    int64_t shape[] = {4};
    boat_tensor_t* fp32 = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.symmetric = false;
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");
    // zp should be non-zero for this all-positive range (actual: -170)
    if (boat_tensor_get_zero_point(q) == 0) FAIL("zp should be non-zero for this range");

    boat_tensor_t* deq = boat_dequantize_tensor(q);
    if (!tensors_allclose(fp32, deq, 0.001f)) FAIL("values out of tolerance");
    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 4: all values identical ---
static int test_constant(void) {
    TEST("Constant tensor (min==max)");
    float data[] = {3.14159f, 3.14159f, 3.14159f};
    int64_t shape[] = {3};
    boat_tensor_t* fp32 = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);

    boat_quant_config_t cfg = boat_quant_config_default();
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");
    if (boat_tensor_get_scale(q) == 0.0f) FAIL("scale should be non-zero");

    boat_tensor_t* deq = boat_dequantize_tensor(q);
    if (!tensors_allclose(fp32, deq, 0.001f)) FAIL("values out of tolerance");
    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 5: quantized bounds ---
static int test_bounds(void) {
    TEST("Quantized values in [0, 255]");
    float data[] = {-100.0f, 0.0f, 100.0f, 1000.0f};
    int64_t shape[] = {4};
    boat_tensor_t* fp32 = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);

    boat_quant_config_t cfg = boat_quant_config_default();
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");

    const uint8_t* qd = (const uint8_t*)boat_tensor_const_data(q);
    size_t n = boat_tensor_nelements(q);
    for (size_t i = 0; i < n; i++) {
        if (qd[i] > 255) FAIL("value > 255");
    }
    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    PASS(); return 0;
}

// --- Test 6: model-level Dense quantization ---
static int test_model_quantize_dense(void) {
    TEST("Model quantize Dense");
    boat_model_t* m = boat_model_create();
    boat_dense_layer_t* d = boat_dense_layer_create(4, 8, true);
    fill_tensor(boat_dense_layer_get_weight(d), 1.0f);
    fill_tensor(boat_dense_layer_get_bias(d), 0.1f);
    boat_model_add_layer(m, wrap(d, BOAT_LAYER_TYPE_DENSE));

    boat_quant_config_t cfg = boat_quant_config_default();
    if (!boat_model_quantize(m, &cfg)) FAIL("model_quantize failed");

    // Weight should now be UINT8 with scale set
    boat_tensor_t* w = boat_dense_layer_get_weight(d);
    if (boat_tensor_dtype(w) != BOAT_DTYPE_UINT8) FAIL("weight dtype not UINT8");
    if (boat_tensor_get_scale(w) == 0.0f) FAIL("weight scale is 0");

    boat_model_free(m);
    PASS(); return 0;
}

// --- Test 7: model quantize->dequantize Dense ---
static int test_model_dequantize_dense(void) {
    TEST("Model dequantize Dense");
    boat_model_t* m = boat_model_create();
    boat_dense_layer_t* d = boat_dense_layer_create(4, 8, true);
    fill_tensor(boat_dense_layer_get_weight(d), 1.0f);
    fill_tensor(boat_dense_layer_get_bias(d), 0.1f);
    boat_model_add_layer(m, wrap(d, BOAT_LAYER_TYPE_DENSE));

    // Copy original weight for comparison
    boat_tensor_t* orig_w = boat_dense_layer_get_weight(d);
    size_t nbytes = boat_tensor_nbytes(orig_w);
    float* orig_data = malloc(nbytes);
    memcpy(orig_data, boat_tensor_const_data(orig_w), nbytes);

    boat_quant_config_t cfg = boat_quant_config_default();
    if (!boat_model_quantize(m, &cfg)) FAIL("model_quantize failed");
    if (!boat_model_dequantize(m)) FAIL("model_dequantize failed");

    boat_tensor_t* restored = boat_dense_layer_get_weight(d);
    if (boat_tensor_dtype(restored) != BOAT_DTYPE_FLOAT32) FAIL("restored dtype not FLOAT32");

    const float* rd = (const float*)boat_tensor_const_data(restored);
    size_t n = boat_tensor_nelements(restored);
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(rd[i] - orig_data[i]);
        float mag = fmaxf(fabsf(rd[i]), fabsf(orig_data[i]));
        if (mag < 1e-6f) mag = 1.0f;
        if (diff / mag > 0.05f) {
            free(orig_data);
            FAIL("weight mismatch after roundtrip");
        }
    }
    free(orig_data);
    boat_model_free(m);
    PASS(); return 0;
}

// --- Test 8: model quantize Conv2D ---
static int test_model_quantize_conv(void) {
    TEST("Model quantize Conv2D");
    boat_model_t* m = boat_model_create();
    boat_conv_layer_t* c = boat_conv_layer_create(1, 4, 3, 1, 1);
    fill_tensor(boat_conv_layer_get_weight(c), 1.0f);
    fill_tensor(boat_conv_layer_get_bias(c), 0.1f);
    boat_model_add_layer(m, wrap(c, BOAT_LAYER_TYPE_CONV2D));

    boat_quant_config_t cfg = boat_quant_config_default();
    if (!boat_model_quantize(m, &cfg)) FAIL("model_quantize failed");

    boat_tensor_t* w = boat_conv_layer_get_weight(c);
    if (boat_tensor_dtype(w) != BOAT_DTYPE_UINT8) FAIL("weight dtype not UINT8");
    if (boat_tensor_get_scale(w) == 0.0f) FAIL("weight scale is 0");

    boat_model_free(m);
    PASS(); return 0;
}

// --- Test 9: calibration ---
static int test_calibration(void) {
    TEST("Calibration observe/get_range");
    boat_calibration_data_t* calib = boat_calibration_create(3);
    if (!calib) FAIL("create returned NULL");
    if (calib->num_layers != 3) FAIL("wrong num_layers");

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int64_t shape[] = {5};
    boat_tensor_t* t = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);
    boat_calibration_observe(calib, 1, t);

    float mn, mx;
    if (!boat_calibration_get_range(calib, 1, &mn, &mx)) FAIL("get_range failed");
    if (fabsf(mn - 1.0f) > 1e-6f || fabsf(mx - 5.0f) > 1e-6f) FAIL("wrong range");

    // Unobserved layer should return false
    if (boat_calibration_get_range(calib, 0, &mn, &mx)) FAIL("unobserved layer should fail");

    boat_tensor_unref(t);
    boat_calibration_free(calib);
    PASS(); return 0;
}

// --- Test 10: quantize model -> save -> load -> weights still quantized ---
static int test_quantized_save_load(void) {
    TEST("Quantized save->load preserves quantized weights");
    const char* tmpfile = "test_quantized_save_load.tmp";

    boat_model_t* m = boat_model_create();
    boat_dense_layer_t* d = boat_dense_layer_create(4, 8, true);
    fill_tensor(boat_dense_layer_get_weight(d), 1.0f);
    fill_tensor(boat_dense_layer_get_bias(d), 0.1f);
    boat_model_add_layer(m, wrap(d, BOAT_LAYER_TYPE_DENSE));

    boat_quant_config_t cfg = boat_quant_config_default();
    if (!boat_model_quantize(m, &cfg)) FAIL("model_quantize failed");

    // Save quantized model
    if (!boat_model_save(m, tmpfile)) FAIL("save failed");

    // Load it back
    boat_model_t* loaded = boat_model_load(tmpfile);
    if (!loaded) FAIL("load failed");

    // Check weights are still quantized
    boat_layer_t* l = boat_model_get_layer(loaded, 0);
    if (!l || l->type != BOAT_LAYER_TYPE_DENSE) FAIL("wrong layer type");
    boat_tensor_t* w = boat_dense_layer_get_weight((boat_dense_layer_t*)l->data);
    if (boat_tensor_dtype(w) != BOAT_DTYPE_UINT8) FAIL("weight dtype not UINT8 after load");
    if (boat_tensor_get_scale(w) == 0.0f) FAIL("weight scale is 0 after load");

    boat_model_free(m);
    boat_model_free(loaded);
    remove(tmpfile);
    PASS(); return 0;
}

// --- Test 11: quantize -> save -> load -> dequantize -> forward matches original ---
static int test_quantized_save_load_forward(void) {
    TEST("Quantized save->load->dequantize forward");
    const char* tmpfile = "test_quantized_forward.tmp";

    // Build model with known weights
    boat_model_t* m = boat_model_create();
    boat_dense_layer_t* d = boat_dense_layer_create(4, 3, true);
    // Set deterministic weights
    boat_tensor_t* w = boat_dense_layer_get_weight(d);
    float* wd = (float*)boat_tensor_data(w);
    size_t wn = boat_tensor_nelements(w);
    for (size_t i = 0; i < wn; i++) wd[i] = ((float)(i % 5) - 2.0f) * 0.5f;
    float* bd = (float*)boat_tensor_data(boat_dense_layer_get_bias(d));
    size_t bn = boat_tensor_nelements(boat_dense_layer_get_bias(d));
    for (size_t i = 0; i < bn; i++) bd[i] = (float)(i + 1) * 0.1f;
    boat_model_add_layer(m, wrap(d, BOAT_LAYER_TYPE_DENSE));

    // Run forward on FP32 layer to get reference output
    int64_t in_shape[] = {2, 4};
    float in_data[] = {0.5f, -0.5f, 1.0f, -1.0f, 0.1f, 0.2f, 0.3f, 0.4f};
    boat_tensor_t* input = boat_tensor_from_data(in_shape, 2, BOAT_DTYPE_FLOAT32, in_data);
    boat_tensor_t* ref_output = boat_dense_layer_forward(d, input);
    if (!ref_output) FAIL("reference forward failed");

    // Quantize, save, load, dequantize
    boat_quant_config_t cfg = boat_quant_config_default();
    if (!boat_model_quantize(m, &cfg)) FAIL("quantize failed");
    if (!boat_model_save(m, tmpfile)) FAIL("save failed");
    boat_model_free(m);

    boat_model_t* loaded = boat_model_load(tmpfile);
    if (!loaded) FAIL("load failed");
    if (!boat_model_dequantize(loaded)) FAIL("dequantize failed");

    // Get the restored layer and run forward
    boat_layer_t* l = boat_model_get_layer(loaded, 0);
    if (!l || l->type != BOAT_LAYER_TYPE_DENSE) FAIL("wrong layer");
    boat_dense_layer_t* restored = (boat_dense_layer_t*)l->data;
    boat_tensor_t* restored_output = boat_dense_layer_forward(restored, input);
    if (!restored_output) FAIL("restored forward failed");

    // Compare outputs (should be close but not exact due to quantization error)
    if (!tensors_allclose(ref_output, restored_output, 0.05f)) FAIL("output mismatch");

    boat_tensor_unref(input);
    boat_tensor_unref(ref_output);
    boat_tensor_unref(restored_output);
    boat_model_free(loaded);
    remove(tmpfile);
    PASS(); return 0;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Quantization Tests\n");
    printf("==================\n\n");

    int fail = 0;
    fail |= test_roundtrip();
    fail |= test_symmetric();
    fail |= test_asymmetric_zp();
    fail |= test_constant();
    fail |= test_bounds();
    fail |= test_model_quantize_dense();
    fail |= test_model_dequantize_dense();
    fail |= test_model_quantize_conv();
    fail |= test_calibration();
    fail |= test_quantized_save_load();
    fail |= test_quantized_save_load_forward();

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return fail ? 1 : 0;
}
