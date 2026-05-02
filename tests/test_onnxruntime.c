// test_onnxruntime.c - ONNX Runtime inference backend tests
#include <boat.h>
#include <boat/format/onnxruntime.h>
#include <boat/layers.h>
#include <boat/model.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Include the protobuf builder for constructing test model bytes
#include "format/onnx_pb.h"

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { printf("  %s ... ", name); tests_total++; } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); return 1; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); return 1; } } while(0)

// Tolerance for float comparisons
#define TOL 1e-4f

// -----------------------------------------------------------------------
// Protobuf helper: write an INT attribute { name, type: 2 (INT), i: val }
// AttributeProto is field 5 inside NodeProto for this ONNX version
// -----------------------------------------------------------------------
static void write_int_attr(pb_builder_t* b, const char* name, int64_t val) {
    size_t pos = pb_begin_submessage(b, 5);
    pb_write_string(b, 1, name);
    pb_write_tag(b, 20, 0); pb_write_varint(b, 2);  // type = INT
    pb_write_tag(b, 3, 0); pb_write_varint(b, val);
    pb_patch_length(b, pos);
}

// -----------------------------------------------------------------------
// Helper: write an INTS attribute { name, type: 7 (INTS), ints: [vals...] }
// -----------------------------------------------------------------------
static void write_ints_attr(pb_builder_t* b, const char* name, const int64_t* vals, int count) {
    size_t pos = pb_begin_submessage(b, 5);
    pb_write_string(b, 1, name);
    pb_write_tag(b, 20, 0); pb_write_varint(b, 7);  // type = INTS
    for (int i = 0; i < count; i++) {
        pb_write_tag(b, 8, 0); pb_write_varint(b, vals[i]);  // ints (field 8)
    }
    pb_patch_length(b, pos);
}

// -----------------------------------------------------------------------
// Helper: build a minimal ONNX model (Gemm + Relu) in memory
// Model: input(4) -> Gemm(3, transB=1) -> Relu -> output(3)
// Weight: [3,4], Bias: [3]
// -----------------------------------------------------------------------
static uint8_t* build_gemm_relu_model(size_t* out_size) {
    pb_builder_t b;
    pb_builder_init(&b);

    // ir_version = 4 (field 1, int64)
    pb_write_tag(&b, 1, 0);
    pb_write_varint(&b, 4);

    // opset_import { domain: "", version: 9 } (field 8, message)
    size_t opset_pos = pb_begin_submessage(&b, 8);
    pb_write_string(&b, 1, "");
    pb_write_tag(&b, 2, 0);
    pb_write_varint(&b, 9);
    pb_patch_length(&b, opset_pos);

    // --- GraphProto (field 7) ---
    size_t graph_pos = pb_begin_submessage(&b, 7);
    pb_write_string(&b, 2, "test");

    // Graph inputs: "input" (field 11, ValueInfoProto) — shape [1, 4]
    size_t in_vip = pb_begin_submessage(&b, 11);
    pb_write_string(&b, 1, "input");
    size_t in_type = pb_begin_submessage(&b, 2);
    size_t in_tensor = pb_begin_submessage(&b, 1);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1);  // elem_type = FLOAT
    // TensorShapeProto: each dim is a Dimension submessage (field 1)
    { size_t in_shape = pb_begin_submessage(&b, 2);
      { size_t d = pb_begin_submessage(&b, 1); pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1); pb_patch_length(&b, d); }
      { size_t d = pb_begin_submessage(&b, 1); pb_write_tag(&b, 1, 0); pb_write_varint(&b, 4); pb_patch_length(&b, d); }
      pb_patch_length(&b, in_shape); }
    pb_patch_length(&b, in_tensor);
    pb_patch_length(&b, in_type);
    pb_patch_length(&b, in_vip);

    // Graph outputs: "output" (field 12, ValueInfoProto) — shape [1, 3]
    size_t out_vip = pb_begin_submessage(&b, 12);
    pb_write_string(&b, 1, "output");
    size_t out_type = pb_begin_submessage(&b, 2);
    size_t out_tensor = pb_begin_submessage(&b, 1);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1);  // elem_type = FLOAT
    { size_t out_shape = pb_begin_submessage(&b, 2);
      { size_t d = pb_begin_submessage(&b, 1); pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1); pb_patch_length(&b, d); }
      { size_t d = pb_begin_submessage(&b, 1); pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3); pb_patch_length(&b, d); }
      pb_patch_length(&b, out_shape); }
    pb_patch_length(&b, out_tensor);
    pb_patch_length(&b, out_type);
    pb_patch_length(&b, out_vip);

    // --- Weight initializer ---
    float weight_data[12] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f
    };
    size_t init_w_pos = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 4);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);  // data_type = FLOAT
    pb_write_string(&b, 8, "weight");
    pb_write_bytes(&b, 9, weight_data, sizeof(weight_data));
    pb_patch_length(&b, init_w_pos);

    // Bias initializer
    float bias_data[3] = { 0.01f, 0.02f, 0.03f };
    size_t init_b_pos = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "bias");
    pb_write_bytes(&b, 9, bias_data, sizeof(bias_data));
    pb_patch_length(&b, init_b_pos);

    // --- Node 1: Gemm ---
    size_t gemm_pos = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "input");             // input[0]
    pb_write_string(&b, 1, "weight");             // input[1]
    pb_write_string(&b, 1, "bias");               // input[2]
    pb_write_string(&b, 2, "gemm_out");           // output[0]
    pb_write_string(&b, 3, "gemm_node");          // name
    pb_write_string(&b, 4, "Gemm");               // op_type
    write_int_attr(&b, "transB", 1);
    pb_patch_length(&b, gemm_pos);

    // --- Node 2: Relu ---
    size_t relu_pos = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "gemm_out");
    pb_write_string(&b, 2, "output");
    pb_write_string(&b, 3, "relu_node");
    pb_write_string(&b, 4, "Relu");
    pb_patch_length(&b, relu_pos);

    pb_patch_length(&b, graph_pos);

    *out_size = b.size;
    return b.data;
}

// -----------------------------------------------------------------------
// Helper: build a simple Conv model in memory
// Input: [1,1,4,4] -> Conv(kernel=2x2, pads=0, strides=1) -> [1,1,3,3]
// -----------------------------------------------------------------------
static uint8_t* build_conv_model(size_t* out_size) {
    pb_builder_t b;
    pb_builder_init(&b);

    pb_write_tag(&b, 1, 0);
    pb_write_varint(&b, 4);

    size_t opset_pos = pb_begin_submessage(&b, 8);
    pb_write_string(&b, 1, "");
    pb_write_tag(&b, 2, 0);
    pb_write_varint(&b, 9);
    pb_patch_length(&b, opset_pos);

    size_t graph_pos = pb_begin_submessage(&b, 7);
    pb_write_string(&b, 2, "conv_test");

    // Input: "input" [1,1,4,4]
    size_t in_vip = pb_begin_submessage(&b, 11);
    pb_write_string(&b, 1, "input");
    size_t in_type = pb_begin_submessage(&b, 2);
    size_t in_tensor = pb_begin_submessage(&b, 1);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1);
    { size_t in_shape = pb_begin_submessage(&b, 2);
      { size_t d = pb_begin_submessage(&b, 1); pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1); pb_patch_length(&b, d); }
      { size_t d = pb_begin_submessage(&b, 1); pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1); pb_patch_length(&b, d); }
      { size_t d = pb_begin_submessage(&b, 1); pb_write_tag(&b, 1, 0); pb_write_varint(&b, 4); pb_patch_length(&b, d); }
      { size_t d = pb_begin_submessage(&b, 1); pb_write_tag(&b, 1, 0); pb_write_varint(&b, 4); pb_patch_length(&b, d); }
      pb_patch_length(&b, in_shape); }
    pb_patch_length(&b, in_tensor);
    pb_patch_length(&b, in_type);
    pb_patch_length(&b, in_vip);

    // Output: "output" [1,1,3,3]
    size_t out_vip = pb_begin_submessage(&b, 12);
    pb_write_string(&b, 1, "output");
    size_t out_type = pb_begin_submessage(&b, 2);
    size_t out_tensor = pb_begin_submessage(&b, 1);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1);
    { size_t out_shape = pb_begin_submessage(&b, 2);
      { size_t d = pb_begin_submessage(&b, 1); pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1); pb_patch_length(&b, d); }
      { size_t d = pb_begin_submessage(&b, 1); pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1); pb_patch_length(&b, d); }
      { size_t d = pb_begin_submessage(&b, 1); pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3); pb_patch_length(&b, d); }
      { size_t d = pb_begin_submessage(&b, 1); pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3); pb_patch_length(&b, d); }
      pb_patch_length(&b, out_shape); }
    pb_patch_length(&b, out_tensor);
    pb_patch_length(&b, out_type);
    pb_patch_length(&b, out_vip);

    // Weight: [1,1,2,2] — TensorProto dims are raw int64, not submessages
    float wdata[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    size_t iw = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "conv_w");
    pb_write_bytes(&b, 9, wdata, sizeof(wdata));
    pb_patch_length(&b, iw);

    // Conv node with attributes
    size_t conv_pos = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "input");
    pb_write_string(&b, 1, "conv_w");
    pb_write_string(&b, 2, "output");
    pb_write_string(&b, 3, "conv_node");
    pb_write_string(&b, 4, "Conv");
    { int64_t ks[] = {2, 2}; write_ints_attr(&b, "kernel_shape", ks, 2);
      int64_t pd[] = {0, 0, 0, 0}; write_ints_attr(&b, "pads", pd, 4);
      int64_t st[] = {1, 1}; write_ints_attr(&b, "strides", st, 2); }
    pb_patch_length(&b, conv_pos);

    pb_patch_length(&b, graph_pos);

    *out_size = b.size;
    return b.data;
}

// -----------------------------------------------------------------------
// Test 1: Session creation from buffer
// -----------------------------------------------------------------------
static int test_create_from_buffer(void) {
    TEST("Create session from buffer");

    size_t model_size;
    uint8_t* model_data = build_gemm_relu_model(&model_size);
    ASSERT(model_data != NULL, "Failed to build model");

    boat_onnxruntime_session_t* session = boat_onnxruntime_create_from_buffer(
        model_data, model_size, BOAT_ORT_CPU);
    free(model_data);

    ASSERT(session != NULL, "Session creation failed");
    ASSERT(boat_onnxruntime_input_count(session) == 1,
           "Expected 1 input");
    ASSERT(boat_onnxruntime_output_count(session) == 1,
           "Expected 1 output");

    const char* in_name = boat_onnxruntime_input_name(session, 0);
    ASSERT(in_name != NULL, "Input name is NULL");

    const char* out_name = boat_onnxruntime_output_name(session, 0);
    ASSERT(out_name != NULL, "Output name is NULL");

    boat_onnxruntime_free(session);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test 2: Run inference with Gemm + Relu
// -----------------------------------------------------------------------
static int test_run_inference(void) {
    TEST("Run inference Gemm+Relu");

    size_t model_size;
    uint8_t* model_data = build_gemm_relu_model(&model_size);

    boat_onnxruntime_session_t* session = boat_onnxruntime_create_from_buffer(
        model_data, model_size, BOAT_ORT_CPU);
    free(model_data);
    ASSERT(session != NULL, "Session creation failed");

    // Create input tensor: [4], values = [1.0, 2.0, 3.0, 4.0]
    // Expected: Gemm: y = x*W^T + B with transB=1
    // W^T = [0.1, 0.5, 0.9; 0.2, 0.6, 1.0; 0.3, 0.7, 1.1; 0.4, 0.8, 1.2]
    // x*W^T = [1*0.1+2*0.2+3*0.3+4*0.4, 1*0.5+2*0.6+3*0.7+4*0.8, 1*0.9+2*1.0+3*1.1+4*1.2]
    //       = [0.1+0.4+0.9+1.6, 0.5+1.2+2.1+3.2, 0.9+2.0+3.3+4.8]
    //       = [3.0, 7.0, 11.0]
    // + bias: [3.01, 7.02, 11.03]
    // Relu: [3.01, 7.02, 11.03] (all positive)
    float expected[3] = {3.01f, 7.02f, 11.03f};

    int64_t shape[] = {1, 4};
    boat_tensor_t* input = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    ASSERT(input != NULL, "Input tensor creation failed");
    float* data = (float*)boat_tensor_data(input);
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f; data[3] = 4.0f;

    boat_tensor_t* output = boat_onnxruntime_run(session, input);
    ASSERT(output != NULL, "Inference failed");

    ASSERT(boat_tensor_ndim(output) == 2, "Expected 2D output");
    const int64_t* out_shape = boat_tensor_shape(output);
    ASSERT(out_shape[0] == 1, "Expected output dim 0 = 1");
    ASSERT(out_shape[1] == 3, "Expected output dim 1 = 3");

    float* out_data = (float*)boat_tensor_data(output);
    for (int i = 0; i < 3; i++) {
        float diff = fabsf(out_data[i] - expected[i]);
        ASSERT(diff < TOL, "Output value mismatch");
    }

    boat_tensor_unref(input);
    boat_tensor_unref(output);
    boat_onnxruntime_free(session);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test 3: Run Conv model inference
// -----------------------------------------------------------------------
static int test_conv_inference(void) {
    TEST("Run Conv inference");

    size_t model_size;
    uint8_t* model_data = build_conv_model(&model_size);

    boat_onnxruntime_session_t* session = boat_onnxruntime_create_from_buffer(
        model_data, model_size, BOAT_ORT_CPU);
    free(model_data);
    ASSERT(session != NULL, "Session creation failed");

    // Input: [1,1,4,4] = all 1.0f
    // Conv: 2x2 ones kernel, stride=1, pad=0
    // Output: [1,1,3,3], each element = sum of 2x2 block = 4.0f
    int64_t shape[] = {1, 1, 4, 4};
    boat_tensor_t* input = boat_tensor_create(shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    ASSERT(input != NULL, "Input creation failed");
    float* data = (float*)boat_tensor_data(input);
    for (int i = 0; i < 16; i++) data[i] = 1.0f;

    boat_tensor_t* output = boat_onnxruntime_run(session, input);
    ASSERT(output != NULL, "Inference failed");

    const int64_t* out_shape = boat_tensor_shape(output);
    ASSERT(out_shape[0] == 1 && out_shape[1] == 1 &&
           out_shape[2] == 3 && out_shape[3] == 3,
           "Expected output [1,1,3,3]");

    float* out_data = (float*)boat_tensor_data(output);
    for (int i = 0; i < 9; i++) {
        ASSERT(fabsf(out_data[i] - 4.0f) < TOL,
               "Conv output should be 4.0");
    }

    boat_tensor_unref(input);
    boat_tensor_unref(output);
    boat_onnxruntime_free(session);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test 4: NULL/error handling
// -----------------------------------------------------------------------
static int test_error_handling(void) {
    TEST("Error handling");

    ASSERT(boat_onnxruntime_create(NULL, BOAT_ORT_CPU) == NULL,
           "NULL path should return NULL");
    ASSERT(boat_onnxruntime_create_from_buffer(NULL, 10, BOAT_ORT_CPU) == NULL,
           "NULL data should return NULL");
    ASSERT(boat_onnxruntime_create_from_buffer("x", 0, BOAT_ORT_CPU) == NULL,
           "Zero size should return NULL");

    ASSERT(boat_onnxruntime_input_count(NULL) == 0,
           "NULL session input count");
    ASSERT(boat_onnxruntime_output_count(NULL) == 0,
           "NULL session output count");
    ASSERT(boat_onnxruntime_input_name(NULL, 0) == NULL,
           "NULL session input name");
    ASSERT(boat_onnxruntime_output_name(NULL, 0) == NULL,
           "NULL session output name");

    boat_onnxruntime_free(NULL);  // Should not crash
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test 5: Run inference with boat model -> save -> ORT load -> compare
// Tests that ORT-run outputs match boat-run outputs for the same model
// -----------------------------------------------------------------------
static int test_boat_ort_consistency(void) {
    TEST("Boat-ORT consistency (Gemm+Relu)");

    // Build a boat model directly
    boat_model_t* model = boat_sequential_create();
    boat_dense_layer_t* dense = boat_dense_layer_create(4, 3, true);

    int64_t wshape[] = {4, 3};
    boat_tensor_t* wt = boat_tensor_from_data(wshape, 2, BOAT_DTYPE_FLOAT32,
        (float[]){0.1f, 0.2f, 0.3f,
                  0.4f, 0.5f, 0.6f,
                  0.7f, 0.8f, 0.9f,
                  1.0f, 1.1f, 1.2f});
    boat_dense_layer_set_weight(dense, wt);
    boat_tensor_unref(wt);

    int64_t bshape[] = {3};
    boat_tensor_t* bt = boat_tensor_from_data(bshape, 1, BOAT_DTYPE_FLOAT32,
        (float[]){0.01f, 0.02f, 0.03f});
    boat_dense_layer_set_bias(dense, bt);
    boat_tensor_unref(bt);

    // Wrap dense layer in a boat_layer_t (cannot cast — different struct layouts)
    boat_layer_t* d_wrapper = (boat_layer_t*)malloc(sizeof(boat_layer_t));
    d_wrapper->data = dense;
    d_wrapper->ops = NULL;
    d_wrapper->type = BOAT_LAYER_TYPE_DENSE;
    boat_sequential_add(model, d_wrapper);

    boat_relu_layer_t* relu = boat_relu_layer_create();
    boat_layer_t* r_wrapper = (boat_layer_t*)malloc(sizeof(boat_layer_t));
    r_wrapper->data = relu;
    r_wrapper->ops = NULL;
    r_wrapper->type = BOAT_LAYER_TYPE_RELU;
    boat_sequential_add(model, r_wrapper);

    // Run boat model inference
    int64_t shape[] = {1, 4};
    boat_tensor_t* boat_input = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(boat_input);
    d[0] = 1.0f; d[1] = 2.0f; d[2] = 3.0f; d[3] = 4.0f;

    boat_tensor_t* boat_output = boat_model_forward(model, boat_input);
    ASSERT(boat_output != NULL, "Boat forward failed");

    // Save boat model to ONNX buffer
    void* onnx_data = NULL;
    size_t onnx_size = 0;
    int save_ok = boat_onnx_save_to_memory(model, &onnx_data, &onnx_size);
    ASSERT(save_ok && onnx_data != NULL && onnx_size > 0,
           "boat_onnx_save_to_memory failed");

    // Load into ORT session
    boat_onnxruntime_session_t* ort_session = boat_onnxruntime_create_from_buffer(
        onnx_data, onnx_size, BOAT_ORT_CPU);
    ASSERT(ort_session != NULL, "ORT session from saved model failed");

    boat_tensor_unref(boat_input);

    // Create input for ORT
    int64_t ort_shape[] = {1, 4};
    boat_tensor_t* ort_input = boat_tensor_create(ort_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* od = (float*)boat_tensor_data(ort_input);
    od[0] = 1.0f; od[1] = 2.0f; od[2] = 3.0f; od[3] = 4.0f;

    boat_tensor_t* ort_output = boat_onnxruntime_run(ort_session, ort_input);
    ASSERT(ort_output != NULL, "ORT forward failed");

    // Compare outputs
    const int64_t* boat_os = boat_tensor_shape(boat_output);
    const int64_t* ort_os = boat_tensor_shape(ort_output);
    ASSERT(boat_os[0] == ort_os[0] && boat_os[1] == ort_os[1],
           "Output shape mismatch");

    float* boat_od = (float*)boat_tensor_data(boat_output);
    float* ort_od = (float*)boat_tensor_data(ort_output);
    size_t n = (size_t)(boat_os[0] * boat_os[1]);
    for (size_t i = 0; i < n; i++) {
        ASSERT(fabsf(boat_od[i] - ort_od[i]) < 1e-3f,
               "Boat-ORT output mismatch");
    }

    boat_tensor_unref(ort_input);
    boat_tensor_unref(ort_output);
    boat_tensor_unref(boat_output);
    free(onnx_data);
    boat_onnxruntime_free(ort_session);
    boat_model_free(model);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
#ifdef BOAT_WITH_ONNXRUNTIME
    printf("=== ONNX Runtime Backend Tests ===\n");

    if (test_create_from_buffer()) return 1;
    if (test_run_inference()) return 1;
    if (test_conv_inference()) return 1;
    if (test_error_handling()) return 1;
    if (test_boat_ort_consistency()) return 1;

    printf("  %d/%d tests passed\n", tests_passed, tests_total);
    printf("=== ALL TESTS PASSED ===\n");
    return tests_passed == tests_total ? 0 : 1;
#else
    printf("ONNX Runtime not enabled. Skipping tests.\n");
    return 0;
#endif
}
