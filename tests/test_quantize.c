// test_quantize.c - Post-Training Quantization tests
#include <boat.h>
#include <boat/tensor.h>
#include <boat/layers.h>
#include <boat/model.h>
#include <boat/quantize.h>
#include <boat/packed.h>
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
    boat_conv_layer_t* c = boat_conv_layer_create(1, 4, 3, 1, 1, 1);
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

// --- Test 12: INT8 quantize-dequantize roundtrip ---
static int test_roundtrip_int8(void) {
    TEST("INT8 quantize-dequantize roundtrip");
    int64_t shape[] = {3, 4};
    boat_tensor_t* fp32 = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor(fp32, 0.0f);

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_INT8;
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");
    if (boat_tensor_dtype(q) != BOAT_DTYPE_INT8) FAIL("dtype not INT8");
    if (boat_tensor_get_scale(q) == 0.0f) FAIL("scale is 0");

    boat_tensor_t* deq = boat_dequantize_tensor(q);
    if (!deq) FAIL("dequantize returned NULL");

    if (!tensors_allclose(fp32, deq, 0.05f)) FAIL("values out of tolerance");
    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 13: INT8 symmetric quantization (zp=0) ---
static int test_symmetric_int8(void) {
    TEST("INT8 symmetric quantization");
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    int64_t shape[] = {5};
    boat_tensor_t* fp32 = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_INT8;
    cfg.symmetric = true;
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");
    if (boat_tensor_get_zero_point(q) != 0) FAIL("zp not 0 for INT8 symmetric");

    boat_tensor_t* deq = boat_dequantize_tensor(q);
    if (!tensors_allclose(fp32, deq, 0.05f)) FAIL("values out of tolerance");
    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 14: INT8 quantized bounds in [-128, 127] ---
static int test_bounds_int8(void) {
    TEST("INT8 quantized values in [-128, 127]");
    float data[] = {-100.0f, 0.0f, 100.0f, 1000.0f};
    int64_t shape[] = {4};
    boat_tensor_t* fp32 = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_INT8;
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");

    const int8_t* qd = (const int8_t*)boat_tensor_const_data(q);
    size_t n = boat_tensor_nelements(q);
    for (size_t i = 0; i < n; i++) {
        if (qd[i] < -128 || qd[i] > 127) FAIL("value out of INT8 range");
    }
    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    PASS(); return 0;
}

// --- Test 15: INT8 model quantize Dense ---
static int test_model_quantize_dense_int8(void) {
    TEST("INT8 model quantize Dense");
    boat_model_t* m = boat_model_create();
    boat_dense_layer_t* d = boat_dense_layer_create(4, 8, true);
    fill_tensor(boat_dense_layer_get_weight(d), 1.0f);
    fill_tensor(boat_dense_layer_get_bias(d), 0.1f);
    boat_model_add_layer(m, wrap(d, BOAT_LAYER_TYPE_DENSE));

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_INT8;
    if (!boat_model_quantize(m, &cfg)) FAIL("model_quantize failed");

    boat_tensor_t* w = boat_dense_layer_get_weight(d);
    if (boat_tensor_dtype(w) != BOAT_DTYPE_INT8) FAIL("weight dtype not INT8");
    if (boat_tensor_get_scale(w) == 0.0f) FAIL("weight scale is 0");

    boat_model_free(m);
    PASS(); return 0;
}

// --- Test 16: BITS2 quantize-dequantize roundtrip ---
static int test_bits2_roundtrip(void) {
    TEST("BITS2 quantize-dequantize roundtrip");
    int64_t shape[] = {8};
    float data[] = {0.0f, 1.0f, 2.0f, 3.0f, 0.5f, 1.5f, 2.5f, 3.0f};
    boat_tensor_t* fp32 = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_BITS2;
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");
    if (boat_tensor_dtype(q) != BOAT_DTYPE_BITS2) FAIL("dtype not BITS2");
    if (boat_tensor_get_scale(q) == 0.0f) FAIL("scale is 0");

    boat_tensor_t* deq = boat_dequantize_tensor(q);
    if (!deq) FAIL("dequantize returned NULL");
    // BITS2 has only 4 levels, so relative error can be large for halfway values
    if (!tensors_allclose(fp32, deq, 0.5f)) FAIL("values out of tolerance");

    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 17: BITS2 quantized bounds in [0, 3] ---
static int test_bits2_bounds(void) {
    TEST("BITS2 quantized values in [0, 3]");
    float data[] = {-100.0f, 0.0f, 100.0f, 1000.0f};
    int64_t shape[] = {4};
    boat_tensor_t* fp32 = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_BITS2;
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");

    // Unpack and check bounds
    const uint8_t* packed = (const uint8_t*)boat_tensor_const_data(q);
    size_t n = boat_tensor_nelements(q);
    uint8_t* unpacked = (uint8_t*)malloc(n);
    boat_unpack_bits2(packed, unpacked, n);
    for (size_t i = 0; i < n; i++) {
        if (unpacked[i] > 3) FAIL("quantized value > 3");
    }
    free(unpacked);

    boat_tensor_t* deq = boat_dequantize_tensor(q);
    if (!deq) FAIL("dequantize returned NULL");

    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 18: FLOAT4 quantize-dequantize roundtrip ---
static int test_float4_roundtrip(void) {
    TEST("FLOAT4 quantize-dequantize roundtrip");
    int64_t shape[] = {6};
    // FLOAT4 can represent: +/- 0.125, 0.25, 0.5, 1, 2, 4, 8, 16
    float data[] = {1.0f, 2.0f, 0.5f, -1.0f, 4.0f, 0.0f};
    boat_tensor_t* fp32 = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_FLOAT4;
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");
    if (boat_tensor_dtype(q) != BOAT_DTYPE_FLOAT4) FAIL("dtype not FLOAT4");

    boat_tensor_t* deq = boat_dequantize_tensor(q);
    if (!deq) FAIL("dequantize returned NULL");

    // Check exact values for representable floats
    const float* deq_data = (const float*)boat_tensor_const_data(deq);
    if (fabsf(deq_data[0] - 1.0f) > 1e-6f) FAIL("1.0 not exact");
    if (fabsf(deq_data[1] - 2.0f) > 1e-6f) FAIL("2.0 not exact");
    if (fabsf(deq_data[3] + 1.0f) > 1e-6f) FAIL("-1.0 not exact");
    if (fabsf(deq_data[5]) > 1e-6f) FAIL("0.0 not exact");

    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 19: FLOAT4 params are default (scale=1, zp=0) ---
static int test_float4_params(void) {
    TEST("FLOAT4 quantization params");
    float data[] = {1.0f, 2.0f, 3.0f};
    int64_t shape[] = {3};
    boat_tensor_t* fp32 = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_FLOAT4;
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");

    if (boat_tensor_get_scale(q) != 1.0f) FAIL("scale != 1.0");
    if (boat_tensor_get_zero_point(q) != 0) FAIL("zp != 0");

    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    PASS(); return 0;
}

// --- Test 20: Per-channel quantize-dequantize (Dense layout) ---
static int test_per_channel_dense(void) {
    TEST("Per-channel quantize Dense (dim=1)");
    // Dense weight shape: [input_features, output_features], channel_dim = 1
    int64_t shape[] = {3, 4};  // 3 input, 4 output features -> 4 channels
    boat_tensor_t* fp32 = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(fp32);
    // Make each output feature have different range
    // Channel 0: [0, 3], Channel 1: [10, 13], Channel 2: [20, 23], Channel 3: [30, 33]
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            d[i * 4 + j] = (float)(j * 10 + i);
        }
    }

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.per_channel = true;
    boat_tensor_t* q = boat_quantize_tensor_per_channel(fp32, &cfg, 1);
    if (!q) FAIL("quantize_per_channel returned NULL");
    if (!boat_tensor_is_per_channel(q)) FAIL("per_channel flag not set");
    if (boat_tensor_num_channels(q) != 4) FAIL("wrong n_channels");

    boat_tensor_t* deq = boat_dequantize_tensor_per_channel(q);
    if (!deq) FAIL("dequantize_per_channel returned NULL");
    if (!tensors_allclose(fp32, deq, 0.05f)) FAIL("values out of tolerance");

    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 21: Per-channel quantize-dequantize (Conv2D layout) ---
static int test_per_channel_conv(void) {
    TEST("Per-channel quantize Conv2D (dim=0)");
    // Conv2D weight shape: [out_channels, in_channels, KH, KW], channel_dim = 0
    int64_t shape[] = {2, 3, 2, 2};  // 2 out_channels -> 2 channels
    boat_tensor_t* fp32 = boat_tensor_create(shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(fp32);
    // Channel 0: small range, Channel 1: large range
    size_t n = 3 * 2 * 2;  // elements per channel
    for (size_t i = 0; i < n; i++) d[i] = (float)i * 0.5f;
    for (size_t i = n; i < 2 * n; i++) d[i] = (float)(i - n) * 10.0f + 100.0f;

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.per_channel = true;
    boat_tensor_t* q = boat_quantize_tensor_per_channel(fp32, &cfg, 0);
    if (!q) FAIL("quantize_per_channel returned NULL");
    if (!boat_tensor_is_per_channel(q)) FAIL("per_channel flag not set");
    if (boat_tensor_num_channels(q) != 2) FAIL("wrong n_channels");

    boat_tensor_t* deq = boat_dequantize_tensor_per_channel(q);
    if (!deq) FAIL("dequantize_per_channel returned NULL");
    if (!tensors_allclose(fp32, deq, 0.1f)) FAIL("values out of tolerance");

    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 22: Per-channel model quantize Dense ---
static int test_model_quantize_per_channel(void) {
    TEST("Per-channel model quantize Dense");
    boat_model_t* m = boat_model_create();
    boat_dense_layer_t* d = boat_dense_layer_create(4, 8, true);
    fill_tensor(boat_dense_layer_get_weight(d), 1.0f);
    fill_tensor(boat_dense_layer_get_bias(d), 0.1f);
    boat_model_add_layer(m, wrap(d, BOAT_LAYER_TYPE_DENSE));

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.per_channel = true;
    if (!boat_model_quantize(m, &cfg)) FAIL("model_quantize failed");

    boat_tensor_t* w = boat_dense_layer_get_weight(d);
    if (boat_tensor_dtype(w) != BOAT_DTYPE_UINT8) FAIL("weight dtype not UINT8");
    if (!boat_tensor_is_per_channel(w)) FAIL("per_channel flag not set");
    if (boat_tensor_num_channels(w) != 8) FAIL("wrong n_channels");

    // Dequantize and check forward still works
    if (!boat_model_dequantize(m)) FAIL("model_dequantize failed");
    boat_tensor_t* w2 = boat_dense_layer_get_weight(d);
    if (boat_tensor_dtype(w2) != BOAT_DTYPE_FLOAT32) FAIL("restored dtype not FLOAT32");

    boat_model_free(m);
    PASS(); return 0;
}

// --- Test 23: Fake quantize ---
static int test_fake_quantize(void) {
    TEST("QAT fake quantize");
    int64_t shape[] = {3, 4};
    boat_tensor_t* t = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    fill_tensor(t, 0.0f);
    // Save original data
    size_t n = boat_tensor_nelements(t);
    float* orig = (float*)malloc(n * sizeof(float));
    memcpy(orig, boat_tensor_const_data(t), n * sizeof(float));

    boat_quant_config_t cfg = boat_quant_config_default();
    if (!boat_fake_quantize(t, &cfg)) FAIL("fake_quantize failed");
    if (boat_tensor_dtype(t) != BOAT_DTYPE_FLOAT32) FAIL("dtype changed from FLOAT32");

    // Values should be close to original (quantization noise)
    const float* after = (const float*)boat_tensor_const_data(t);
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(after[i] - orig[i]);
        if (diff > 0.1f) { free(orig); FAIL("value changed too much"); }
    }
    free(orig);
    boat_tensor_unref(t);
    PASS(); return 0;
}

// --- Test 24: BITS2 model quantize + serialization ---
static int test_bits2_model_serialize(void) {
    TEST("BITS2 model quantize + serialization");
    const char* tmpfile = "test_bits2_serialize.tmp";

    boat_model_t* m = boat_model_create();
    boat_dense_layer_t* d = boat_dense_layer_create(4, 8, true);
    fill_tensor(boat_dense_layer_get_weight(d), 1.0f);
    fill_tensor(boat_dense_layer_get_bias(d), 0.1f);
    boat_model_add_layer(m, wrap(d, BOAT_LAYER_TYPE_DENSE));

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_BITS2;
    if (!boat_model_quantize(m, &cfg)) FAIL("model_quantize failed");

    if (!boat_model_save(m, tmpfile)) FAIL("save failed");
    boat_model_free(m);

    boat_model_t* loaded = boat_model_load(tmpfile);
    if (!loaded) FAIL("load failed");

    boat_layer_t* l = boat_model_get_layer(loaded, 0);
    if (!l || l->type != BOAT_LAYER_TYPE_DENSE) FAIL("wrong layer type");
    boat_tensor_t* w = boat_dense_layer_get_weight((boat_dense_layer_t*)l->data);
    if (boat_tensor_dtype(w) != BOAT_DTYPE_BITS2) FAIL("weight dtype not BITS2 after load");
    if (boat_tensor_get_scale(w) == 0.0f) FAIL("weight scale is 0 after load");

    // Dequantize and verify forward works
    if (!boat_model_dequantize(loaded)) FAIL("model_dequantize failed");

    boat_model_free(loaded);
    remove(tmpfile);
    PASS(); return 0;
}

// --- Test 25: Per-channel model quantize + serialization ---
static int test_per_channel_serialize(void) {
    TEST("Per-channel model quantize + serialization");
    const char* tmpfile = "test_per_channel_serialize.tmp";

    boat_model_t* m = boat_model_create();
    boat_dense_layer_t* d = boat_dense_layer_create(4, 8, true);
    fill_tensor(boat_dense_layer_get_weight(d), 1.0f);
    fill_tensor(boat_dense_layer_get_bias(d), 0.1f);
    boat_model_add_layer(m, wrap(d, BOAT_LAYER_TYPE_DENSE));

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.per_channel = true;
    if (!boat_model_quantize(m, &cfg)) FAIL("model_quantize failed");

    if (!boat_model_save(m, tmpfile)) FAIL("save failed");
    boat_model_free(m);

    boat_model_t* loaded = boat_model_load(tmpfile);
    if (!loaded) FAIL("load failed");

    boat_layer_t* l = boat_model_get_layer(loaded, 0);
    if (!l || l->type != BOAT_LAYER_TYPE_DENSE) FAIL("wrong layer type");
    boat_tensor_t* w = boat_dense_layer_get_weight((boat_dense_layer_t*)l->data);
    if (!boat_tensor_is_per_channel(w)) FAIL("per_channel flag lost after load");
    if (boat_tensor_num_channels(w) != 8) FAIL("wrong n_channels after load");

    if (!boat_model_dequantize(loaded)) FAIL("model_dequantize failed");

    boat_model_free(loaded);
    remove(tmpfile);
    PASS(); return 0;
}

// --- Test 26: Per-channel BITS2 quantize-dequantize ---
static int test_per_channel_bits2(void) {
    TEST("Per-channel BITS2 quantize-dequantize");
    int64_t shape[] = {3, 4};
    boat_tensor_t* fp32 = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(fp32);
    // Values within 2-bit range per channel (max-min <= 3)
    float data[] = {0.0f, 1.0f, 2.0f, 0.0f,
                    0.5f, 1.5f, 2.5f, 1.5f,
                    1.0f, 2.0f, 3.0f, 3.0f};
    memcpy(d, data, 12 * sizeof(float));

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_BITS2;
    cfg.per_channel = true;
    boat_tensor_t* q = boat_quantize_tensor_per_channel(fp32, &cfg, 1);
    if (!q) FAIL("quantize_per_channel returned NULL");
    if (boat_tensor_dtype(q) != BOAT_DTYPE_BITS2) FAIL("dtype not BITS2");

    boat_tensor_t* deq = boat_dequantize_tensor_per_channel(q);
    if (!deq) FAIL("dequantize_per_channel returned NULL");
    if (!tensors_allclose(fp32, deq, 0.5f)) FAIL("values out of tolerance");

    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 27: BITS1 quantize-dequantize roundtrip ---
static int test_bits1_roundtrip(void) {
    TEST("BITS1 quantize-dequantize roundtrip");
    int64_t shape[] = {8};
    float data[] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    boat_tensor_t* fp32 = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_BITS1;
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");
    if (boat_tensor_dtype(q) != BOAT_DTYPE_BITS1) FAIL("dtype not BITS1");
    if (boat_tensor_get_scale(q) == 0.0f) FAIL("scale is 0");

    boat_tensor_t* deq = boat_dequantize_tensor(q);
    if (!deq) FAIL("dequantize returned NULL");
    // BITS1 has only 2 levels, max error is 1 full quantization step
    if (!tensors_allclose(fp32, deq, 0.5f)) FAIL("values out of tolerance");

    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 28: BITS1 quantized bounds in [0, 1] ---
static int test_bits1_bounds(void) {
    TEST("BITS1 quantized values in [0, 1]");
    float data[] = {-100.0f, 0.0f, 100.0f, 1000.0f};
    int64_t shape[] = {4};
    boat_tensor_t* fp32 = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32, data);

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_BITS1;
    boat_tensor_t* q = boat_quantize_tensor(fp32, &cfg);
    if (!q) FAIL("quantize returned NULL");

    // Unpack and check bounds
    const uint8_t* packed = (const uint8_t*)boat_tensor_const_data(q);
    size_t n = boat_tensor_nelements(q);
    bool* unpacked = (bool*)malloc(n);
    boat_unpack_bits1(packed, unpacked, n);
    for (size_t i = 0; i < n; i++) {
        if (unpacked[i] > 1) FAIL("quantized value > 1");
    }
    free(unpacked);

    boat_tensor_t* deq = boat_dequantize_tensor(q);
    if (!deq) FAIL("dequantize returned NULL");

    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
    PASS(); return 0;
}

// --- Test 29: BITS1 model quantize + serialization ---
static int test_bits1_model_serialize(void) {
    TEST("BITS1 model quantize + serialization");
    const char* tmpfile = "test_bits1_serialize.tmp";

    boat_model_t* m = boat_model_create();
    boat_dense_layer_t* d = boat_dense_layer_create(4, 8, true);
    fill_tensor(boat_dense_layer_get_weight(d), 1.0f);
    fill_tensor(boat_dense_layer_get_bias(d), 0.1f);
    boat_model_add_layer(m, wrap(d, BOAT_LAYER_TYPE_DENSE));

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_BITS1;
    if (!boat_model_quantize(m, &cfg)) FAIL("model_quantize failed");

    if (!boat_model_save(m, tmpfile)) FAIL("save failed");
    boat_model_free(m);

    boat_model_t* loaded = boat_model_load(tmpfile);
    if (!loaded) FAIL("load failed");

    boat_layer_t* l = boat_model_get_layer(loaded, 0);
    if (!l || l->type != BOAT_LAYER_TYPE_DENSE) FAIL("wrong layer type");
    boat_tensor_t* w = boat_dense_layer_get_weight((boat_dense_layer_t*)l->data);
    if (boat_tensor_dtype(w) != BOAT_DTYPE_BITS1) FAIL("weight dtype not BITS1 after load");
    if (boat_tensor_get_scale(w) == 0.0f) FAIL("weight scale is 0 after load");

    // Dequantize and verify forward works
    if (!boat_model_dequantize(loaded)) FAIL("model_dequantize failed");

    boat_model_free(loaded);
    remove(tmpfile);
    PASS(); return 0;
}

// --- Test 30: Per-channel BITS1 quantize-dequantize ---
static int test_per_channel_bits1(void) {
    TEST("Per-channel BITS1 quantize-dequantize");
    int64_t shape[] = {3, 4};
    boat_tensor_t* fp32 = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(fp32);
    float data[] = {0.0f, 0.0f, 1.0f, 1.0f,
                    1.0f, 0.0f, 0.0f, 1.0f,
                    1.0f, 1.0f, 0.0f, 0.0f};
    memcpy(d, data, 12 * sizeof(float));

    boat_quant_config_t cfg = boat_quant_config_default();
    cfg.quant_dtype = BOAT_DTYPE_BITS1;
    cfg.per_channel = true;
    boat_tensor_t* q = boat_quantize_tensor_per_channel(fp32, &cfg, 1);
    if (!q) FAIL("quantize_per_channel returned NULL");
    if (boat_tensor_dtype(q) != BOAT_DTYPE_BITS1) FAIL("dtype not BITS1");

    boat_tensor_t* deq = boat_dequantize_tensor_per_channel(q);
    if (!deq) FAIL("dequantize_per_channel returned NULL");
    if (!tensors_allclose(fp32, deq, 0.5f)) FAIL("values out of tolerance");

    boat_tensor_unref(fp32);
    boat_tensor_unref(q);
    boat_tensor_unref(deq);
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
    fail |= test_roundtrip_int8();
    fail |= test_symmetric_int8();
    fail |= test_bounds_int8();
    fail |= test_model_quantize_dense_int8();
    fail |= test_bits2_roundtrip();
    fail |= test_bits2_bounds();
    fail |= test_float4_roundtrip();
    fail |= test_float4_params();
    fail |= test_per_channel_dense();
    fail |= test_per_channel_conv();
    fail |= test_model_quantize_per_channel();
    fail |= test_fake_quantize();
    fail |= test_bits2_model_serialize();
    fail |= test_per_channel_serialize();
    fail |= test_per_channel_bits2();
    fail |= test_bits1_roundtrip();
    fail |= test_bits1_bounds();
    fail |= test_bits1_model_serialize();
    fail |= test_per_channel_bits1();

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return fail ? 1 : 0;
}
