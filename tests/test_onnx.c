// test_onnx.c - ONNX loader tests
#include <boat.h>
#include <boat/format/onnx.h>
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

// -----------------------------------------------------------------------
// Helper: build a minimal ONNX model in memory using the protobuf builder
//
// Model: input(4) -> Gemm(3, transB=1) -> Relu -> output(3)
// Weight: [3,4], Bias: [3]
// -----------------------------------------------------------------------
static uint8_t* build_test_onnx(size_t* out_size) {
    pb_builder_t b;
    pb_builder_init(&b);

    // ModelProto fields written directly

    // ir_version = 4  (field 1, int64)
    pb_write_tag(&b, 1, 0);
    pb_write_varint(&b, 4);

    // opset_import { domain: "", version: 9 }  (field 8, message)
    size_t opset_pos = pb_begin_submessage(&b, 8);
    pb_write_string(&b, 1, "");   // domain = ""
    pb_write_tag(&b, 2, 0);
    pb_write_varint(&b, 9);        // version = 9
    pb_patch_length(&b, opset_pos);

    // --- GraphProto (field 7) ---
    size_t graph_pos = pb_begin_submessage(&b, 7);

    // graph.name = "test" (field 2)
    pb_write_string(&b, 2, "test");

    // --- Weight initializer ---
    // TensorProto { dims: [3,4], data_type: 1 (FLOAT), name: "weight", raw_data: <12 floats> }
    // Weight values: row-major, shape [3, 4]
    // w[0] = [0.1, 0.2, 0.3, 0.4]
    // w[1] = [0.5, 0.6, 0.7, 0.8]
    // w[2] = [0.9, 1.0, 1.1, 1.2]
    float weight_data[12] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f
    };
    size_t init_w_pos = pb_begin_submessage(&b, 5);  // initializer[0]
    // dims: individual varints (non-packed, field 1)
    pb_write_tag(&b, 1, 0);
    pb_write_varint(&b, 3);
    pb_write_tag(&b, 1, 0);
    pb_write_varint(&b, 4);
    // data_type = 1 (field 5, int32)
    pb_write_tag(&b, 5, 0);
    pb_write_varint(&b, 1);
    // name = "weight" (field 7, string)
    pb_write_string(&b, 7, "weight");
    // raw_data (field 10, bytes)
    pb_write_bytes(&b, 10, weight_data, sizeof(weight_data));
    pb_patch_length(&b, init_w_pos);

    // --- Bias initializer ---
    // TensorProto { dims: [3], data_type: 1, name: "bias", raw_data: <3 floats> }
    // bias = [0.01, 0.02, 0.03]
    float bias_data[3] = { 0.01f, 0.02f, 0.03f };
    size_t init_b_pos = pb_begin_submessage(&b, 5);  // initializer[1]
    pb_write_tag(&b, 1, 0);
    pb_write_varint(&b, 3);
    pb_write_tag(&b, 5, 0);
    pb_write_varint(&b, 1);
    pb_write_string(&b, 7, "bias");
    pb_write_bytes(&b, 10, bias_data, sizeof(bias_data));
    pb_patch_length(&b, init_b_pos);

    // --- Gemm node ---
    // NodeProto { input: ["input","weight","bias"], output: ["gemm_out"], name: "gemm", op_type: "Gemm", attribute: {name:"transB",type:2,i:1} }
    size_t gemm_pos = pb_begin_submessage(&b, 1);  // node[0]
    pb_write_string(&b, 1, "input");
    pb_write_string(&b, 1, "weight");
    pb_write_string(&b, 1, "bias");
    pb_write_string(&b, 2, "gemm_out");
    pb_write_string(&b, 3, "gemm");
    pb_write_string(&b, 4, "Gemm");
    // Attribute: transB = 1
    // AttributeProto { name: "transB", type: 2 (INT), i: 1 }
    size_t attr_pos = pb_begin_submessage(&b, 5);
    pb_write_string(&b, 1, "transB");
    pb_write_tag(&b, 5, 0);
    pb_write_varint(&b, 2);  // type = INT
    pb_write_tag(&b, 3, 0);
    pb_write_varint(&b, 1);  // i = 1 (true)
    pb_patch_length(&b, attr_pos);
    pb_patch_length(&b, gemm_pos);

    // --- Relu node ---
    // NodeProto { input: ["gemm_out"], output: ["output"], name: "relu", op_type: "Relu" }
    size_t relu_pos = pb_begin_submessage(&b, 1);  // node[1]
    pb_write_string(&b, 1, "gemm_out");
    pb_write_string(&b, 2, "output");
    pb_write_string(&b, 3, "relu");
    pb_write_string(&b, 4, "Relu");
    pb_patch_length(&b, relu_pos);

    // input: ValueInfoProto { name: "input" }  (field 11, we just write the name)
    size_t inp_pos = pb_begin_submessage(&b, 11);
    pb_write_string(&b, 1, "input");
    pb_patch_length(&b, inp_pos);

    // output: ValueInfoProto { name: "output" }  (field 12)
    size_t out_pos = pb_begin_submessage(&b, 12);
    pb_write_string(&b, 1, "output");
    pb_patch_length(&b, out_pos);

    pb_patch_length(&b, graph_pos);

    *out_size = b.size;
    return b.data;  // caller must free
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

static int test_onnx_load_null(void) {
    TEST("onnx_load(NULL) returns NULL");
    if (boat_onnx_load(NULL) != NULL) FAIL("expected NULL");
    PASS(); return 0;
}

static int test_onnx_load_nonexistent(void) {
    TEST("onnx_load(nonexistent) returns NULL");
    if (boat_onnx_load("nonexistent.onnx") != NULL) FAIL("expected NULL");
    PASS(); return 0;
}

static int test_onnx_load_from_memory_null(void) {
    TEST("onnx_load_from_memory(NULL) returns NULL");
    if (boat_onnx_load_from_memory(NULL, 0) != NULL) FAIL("expected NULL");
    PASS(); return 0;
}

static int test_onnx_load_and_forward(void) {
    TEST("ONNX load + forward (Gemm+Relu)");

    size_t model_size;
    uint8_t* model_bytes = build_test_onnx(&model_size);
    if (!model_bytes) FAIL("build_test_onnx failed");

    boat_model_t* model = boat_onnx_load_from_memory(model_bytes, model_size);
    if (!model) { free(model_bytes); FAIL("onnx_load_from_memory failed"); }

    // Create input: [batch=2, features=4]
    int64_t in_shape[] = {2, 4};
    float in_data[] = { -0.5f, 0.5f, -1.0f, 1.0f,
                         1.0f, 0.0f, -1.0f, -0.5f };
    boat_tensor_t* input = boat_tensor_from_data(in_shape, 2, BOAT_DTYPE_FLOAT32, in_data);
    if (!input) { boat_model_free(model); free(model_bytes); FAIL("input creation failed"); }

    boat_tensor_t* output = boat_model_forward(model, input);
    if (!output) { FAIL("relu forward failed"); }
    if (!output) {
        boat_tensor_unref(input);
        boat_model_free(model);
        free(model_bytes);
        FAIL("forward pass returned NULL");
    }

    // Verify output
    const int64_t* out_shape = boat_tensor_shape(output);
    if (boat_tensor_ndim(output) != 2 || out_shape[0] != 2 || out_shape[1] != 3) {
        boat_tensor_unref(input);
        boat_tensor_unref(output);
        boat_model_free(model);
        free(model_bytes);
        FAIL("output shape mismatch");
    }

    const float* out_data = (const float*)boat_tensor_const_data(output);

    // Expected values (computed manually):
    // Gemm with transB=1: Y = A * B^T + C
    // A = input [2,4], B = weight [3,4], B^T = [4,3], C = bias [3]
    // For batch 0: A[0] = [-0.5, 0.5, -1.0, 1.0]
    //   y0 = -0.5*0.1 + 0.5*0.2 + -1.0*0.3 + 1.0*0.4 + 0.01 = -0.05+0.10-0.30+0.40+0.01 = 0.16
    //   y1 = -0.5*0.5 + 0.5*0.6 + -1.0*0.7 + 1.0*0.8 + 0.02 = -0.25+0.30-0.70+0.80+0.02 = 0.17
    //   y2 = -0.5*0.9 + 0.5*1.0 + -1.0*1.1 + 1.0*1.2 + 0.03 = -0.45+0.50-1.10+1.20+0.03 = 0.18
    // After ReLU: all positive, unchanged
    float expected[] = {
        0.16f, 0.17f, 0.18f,
        // batch 1: A[1] = [1.0, 0.0, -1.0, -0.5]
        //   y0 = 1.0*0.1 + 0.0*0.2 + -1.0*0.3 + -0.5*0.4 + 0.01 = 0.10+0-0.30-0.20+0.01 = -0.39
        //   y1 = 1.0*0.5 + 0.0*0.6 + -1.0*0.7 + -0.5*0.8 + 0.02 = 0.50+0-0.70-0.40+0.02 = -0.58
        //   y2 = 1.0*0.9 + 0.0*1.0 + -1.0*1.1 + -0.5*1.2 + 0.03 = 0.90+0-1.10-0.60+0.03 = -0.77
        // After ReLU: [0.0, 0.0, 0.0]
        0.0f, 0.0f, 0.0f
    };

    for (int i = 0; i < 6; i++) {
        float diff = fabsf(out_data[i] - expected[i]);
        if (diff > 1e-5f) {
            printf("FAIL: output[%d] = %f, expected %f\n", i, out_data[i], expected[i]);
            boat_tensor_unref(input);
            boat_tensor_unref(output);
            boat_model_free(model);
            free(model_bytes);
            return 1;
        }
    }

    boat_tensor_unref(input);
    boat_tensor_unref(output);
    boat_model_free(model);
    free(model_bytes);
    PASS(); return 0;
}

static int test_onnx_check(void) {
    TEST("onnx_check on valid model");
    size_t model_size;
    uint8_t* model_bytes = build_test_onnx(&model_size);
    if (!model_size) FAIL("build failed");

    // Check from memory by writing to temp file
    const char* tmpfile = "test_onnx_check.tmp";
    FILE* f = fopen(tmpfile, "wb");
    if (!f) { free(model_bytes); FAIL("fopen failed"); }
    fwrite(model_bytes, 1, model_size, f);
    fclose(f);

    if (!boat_onnx_check(tmpfile)) {
        remove(tmpfile);
        free(model_bytes);
        FAIL("onnx_check on valid model returned false");
    }

    // Check on nonexistent file
    if (boat_onnx_check("nonexistent.onnx")) {
        remove(tmpfile);
        free(model_bytes);
        FAIL("onnx_check on nonexistent file returned true");
    }

    remove(tmpfile);
    free(model_bytes);
    PASS(); return 0;
}

static int test_onnx_get_version(void) {
    TEST("onnx_get_version");
    size_t model_size;
    uint8_t* model_bytes = build_test_onnx(&model_size);
    if (!model_size) FAIL("build failed");

    const char* tmpfile = "test_onnx_version.tmp";
    FILE* f = fopen(tmpfile, "wb");
    if (!f) { free(model_bytes); FAIL("fopen failed"); }
    fwrite(model_bytes, 1, model_size, f);
    fclose(f);

    int major = 0, minor = 0, patch = 0;
    if (!boat_onnx_get_version(tmpfile, &major, &minor, &patch)) {
        remove(tmpfile);
        free(model_bytes);
        FAIL("get_version failed");
    }
    if (major != 4) {
        printf("FAIL: expected version 4, got %d\n", major);
        remove(tmpfile);
        free(model_bytes);
        return 1;
    }

    remove(tmpfile);
    free(model_bytes);
    PASS(); return 0;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("ONNX Loader Tests\n");
    printf("=================\n\n");

    int fail = 0;
    fail |= test_onnx_load_null();
    fail |= test_onnx_load_nonexistent();
    fail |= test_onnx_load_from_memory_null();
    fail |= test_onnx_load_and_forward();
    fail |= test_onnx_check();
    fail |= test_onnx_get_version();

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return fail ? 1 : 0;
}
