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
    pb_write_tag(&b, 2, 0);
    pb_write_varint(&b, 1);
    // name = "weight" (field 7, string)
    pb_write_string(&b, 8, "weight");
    // raw_data (field 10, bytes)
    pb_write_bytes(&b, 9,weight_data, sizeof(weight_data));
    pb_patch_length(&b, init_w_pos);

    // --- Bias initializer ---
    // TensorProto { dims: [3], data_type: 1, name: "bias", raw_data: <3 floats> }
    // bias = [0.01, 0.02, 0.03]
    float bias_data[3] = { 0.01f, 0.02f, 0.03f };
    size_t init_b_pos = pb_begin_submessage(&b, 5);  // initializer[1]
    pb_write_tag(&b, 1, 0);
    pb_write_varint(&b, 3);
    pb_write_tag(&b, 2, 0);
    pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "bias");
    pb_write_bytes(&b, 9,bias_data, sizeof(bias_data));
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
    pb_write_tag(&b, 20, 0);
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
// Helper: write ONNX attribute protos for test model building
// -----------------------------------------------------------------------

// Write an INT attribute { name, type: 2 (INT), i: val }
static void write_int_attr(pb_builder_t* b, const char* name, int64_t val) {
    size_t pos = pb_begin_submessage(b, 5);
    pb_write_string(b, 1, name);
    pb_write_tag(b, 20, 0); pb_write_varint(b, 2);
    pb_write_tag(b, 3, 0); pb_write_varint(b, val);
    pb_patch_length(b, pos);
}

// Write an INTS attribute { name, type: 7 (INTS), ints: [vals...] }
static void write_ints_attr(pb_builder_t* b, const char* name, const int64_t* vals, int count) {
    size_t pos = pb_begin_submessage(b, 5);
    pb_write_string(b, 1, name);
    pb_write_tag(b, 20, 0); pb_write_varint(b, 7);
    for (int i = 0; i < count; i++) {
        pb_write_tag(b, 7, 0); pb_write_varint(b, vals[i]);
    }
    pb_patch_length(b, pos);
}

// Write a FLOAT attribute { name, type: 1 (FLOAT), f: val }
static void write_float_attr(pb_builder_t* b, const char* name, float val) {
    size_t pos = pb_begin_submessage(b, 5);
    pb_write_string(b, 1, name);
    pb_write_tag(b, 20, 0); pb_write_varint(b, 1);
    pb_write_tag(b, 2, 5);
    pb_write_raw(b, &val, sizeof(float));
    pb_patch_length(b, pos);
}

// -----------------------------------------------------------------------
// Builder: Conv model (1 input channel, 2 output channels, 3x3 kernel)
// Input: [1,1,4,4], Output: [1,2,2,2]
// -----------------------------------------------------------------------
static uint8_t* build_test_conv_model(size_t* out_size) {
    pb_builder_t b;
    pb_builder_init(&b);

    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 4);

    size_t opset_pos = pb_begin_submessage(&b, 8);
    pb_write_string(&b, 1, ""); pb_write_tag(&b, 2, 0); pb_write_varint(&b, 9);
    pb_patch_length(&b, opset_pos);

    size_t graph_pos = pb_begin_submessage(&b, 7);
    pb_write_string(&b, 2, "conv_test");

    // Weight [2,1,3,3] = all 1.0
    float wd[18]; for (int i = 0; i < 18; i++) wd[i] = 1.0f;
    size_t iw = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "conv_weight");
    pb_write_bytes(&b, 9,wd, sizeof(wd));
    pb_patch_length(&b, iw);

    // Bias [2] = [1.0, 2.0]
    float bd[2] = {1.0f, 2.0f};
    size_t ib = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "conv_bias");
    pb_write_bytes(&b, 9,bd, sizeof(bd));
    pb_patch_length(&b, ib);

    // Conv node
    size_t cn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "input");
    pb_write_string(&b, 1, "conv_weight");
    pb_write_string(&b, 1, "conv_bias");
    pb_write_string(&b, 2, "conv_out");
    pb_write_string(&b, 3, "conv");
    pb_write_string(&b, 4, "Conv");
    { int64_t ks[] = {3, 3}; write_ints_attr(&b, "kernel_shape", ks, 2);
      int64_t st[] = {1, 1}; write_ints_attr(&b, "strides", st, 2);
      int64_t pd[] = {0, 0}; write_ints_attr(&b, "pads", pd, 2); }
    pb_patch_length(&b, cn);

    size_t inp = pb_begin_submessage(&b, 11);
    pb_write_string(&b, 1, "input"); pb_patch_length(&b, inp);
    size_t out = pb_begin_submessage(&b, 12);
    pb_write_string(&b, 1, "conv_out"); pb_patch_length(&b, out);

    pb_patch_length(&b, graph_pos);
    *out_size = b.size;
    return b.data;
}

// -----------------------------------------------------------------------
// Builder: Conv -> BatchNormalization -> Relu model
// Input: [1,1,4,4], Output: [1,2,2,2]
// -----------------------------------------------------------------------
static uint8_t* build_test_batchnorm_model(size_t* out_size) {
    pb_builder_t b;
    pb_builder_init(&b);

    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 4);

    size_t opset_pos = pb_begin_submessage(&b, 8);
    pb_write_string(&b, 1, ""); pb_write_tag(&b, 2, 0); pb_write_varint(&b, 9);
    pb_patch_length(&b, opset_pos);

    size_t graph_pos = pb_begin_submessage(&b, 7);
    pb_write_string(&b, 2, "bn_test");

    // Conv weight [2,1,3,3] = all 1.0
    float cw[18]; for (int i = 0; i < 18; i++) cw[i] = 1.0f;
    size_t iw = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "conv_weight");
    pb_write_bytes(&b, 9,cw, sizeof(cw));
    pb_patch_length(&b, iw);

    // Conv bias [2] = [1.0, 2.0]
    float cb[2] = {1.0f, 2.0f};
    size_t ib = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "conv_bias");
    pb_write_bytes(&b, 9,cb, sizeof(cb));
    pb_patch_length(&b, ib);

    // BN scale [2] = [1, 1]
    float bs[2] = {1.0f, 1.0f};
    size_t ibs = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "bn_scale");
    pb_write_bytes(&b, 9,bs, sizeof(bs));
    pb_patch_length(&b, ibs);

    // BN bias [2] = [0, 0]
    float bb[2] = {0.0f, 0.0f};
    size_t ibb = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "bn_bias");
    pb_write_bytes(&b, 9,bb, sizeof(bb));
    pb_patch_length(&b, ibb);

    // BN mean [2] = [0, 0]
    float bm[2] = {0.0f, 0.0f};
    size_t ibm = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "bn_mean");
    pb_write_bytes(&b, 9,bm, sizeof(bm));
    pb_patch_length(&b, ibm);

    // BN var [2] = [1, 1]
    float bv[2] = {1.0f, 1.0f};
    size_t ibv = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "bn_var");
    pb_write_bytes(&b, 9,bv, sizeof(bv));
    pb_patch_length(&b, ibv);

    // Conv node
    size_t cn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "input");
    pb_write_string(&b, 1, "conv_weight");
    pb_write_string(&b, 1, "conv_bias");
    pb_write_string(&b, 2, "conv_out");
    pb_write_string(&b, 3, "conv");
    pb_write_string(&b, 4, "Conv");
    { int64_t ks[] = {3, 3}; write_ints_attr(&b, "kernel_shape", ks, 2);
      int64_t st[] = {1, 1}; write_ints_attr(&b, "strides", st, 2);
      int64_t pd[] = {0, 0}; write_ints_attr(&b, "pads", pd, 2); }
    pb_patch_length(&b, cn);

    // BN node
    size_t bn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "conv_out");
    pb_write_string(&b, 1, "bn_scale");
    pb_write_string(&b, 1, "bn_bias");
    pb_write_string(&b, 1, "bn_mean");
    pb_write_string(&b, 1, "bn_var");
    pb_write_string(&b, 2, "bn_out");
    pb_write_string(&b, 3, "bn");
    pb_write_string(&b, 4, "BatchNormalization");
    write_float_attr(&b, "epsilon", 1e-5f);
    pb_patch_length(&b, bn);

    // Relu node
    size_t rn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "bn_out");
    pb_write_string(&b, 2, "output");
    pb_write_string(&b, 3, "relu");
    pb_write_string(&b, 4, "Relu");
    pb_patch_length(&b, rn);

    size_t inp = pb_begin_submessage(&b, 11);
    pb_write_string(&b, 1, "input"); pb_patch_length(&b, inp);
    size_t out = pb_begin_submessage(&b, 12);
    pb_write_string(&b, 1, "output"); pb_patch_length(&b, out);

    pb_patch_length(&b, graph_pos);
    *out_size = b.size;
    return b.data;
}

// -----------------------------------------------------------------------
// Builder: MaxPool model (2x2, stride=2)
// Input: [1,1,4,4], Output: [1,1,2,2]
// -----------------------------------------------------------------------
static uint8_t* build_test_maxpool_model(size_t* out_size) {
    pb_builder_t b;
    pb_builder_init(&b);

    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 4);

    size_t opset_pos = pb_begin_submessage(&b, 8);
    pb_write_string(&b, 1, ""); pb_write_tag(&b, 2, 0); pb_write_varint(&b, 9);
    pb_patch_length(&b, opset_pos);

    size_t graph_pos = pb_begin_submessage(&b, 7);
    pb_write_string(&b, 2, "pool_test");

    size_t nn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "input");
    pb_write_string(&b, 2, "output");
    pb_write_string(&b, 3, "pool");
    pb_write_string(&b, 4, "MaxPool");
    { int64_t ks[] = {2, 2}; write_ints_attr(&b, "kernel_shape", ks, 2);
      int64_t st[] = {2, 2}; write_ints_attr(&b, "strides", st, 2);
      int64_t pd[] = {0, 0}; write_ints_attr(&b, "pads", pd, 2); }
    pb_patch_length(&b, nn);

    size_t inp = pb_begin_submessage(&b, 11);
    pb_write_string(&b, 1, "input"); pb_patch_length(&b, inp);
    size_t out = pb_begin_submessage(&b, 12);
    pb_write_string(&b, 1, "output"); pb_patch_length(&b, out);

    pb_patch_length(&b, graph_pos);
    *out_size = b.size;
    return b.data;
}

// -----------------------------------------------------------------------
// Builder: Full CNN - Conv+BN+Relu+MaxPool+Flatten+Gemm+Softmax
// Input: [1,1,4,4], Output: [1,3]
// -----------------------------------------------------------------------
static uint8_t* build_test_cnn_model(size_t* out_size) {
    pb_builder_t b;
    pb_builder_init(&b);

    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 4);

    size_t opset_pos = pb_begin_submessage(&b, 8);
    pb_write_string(&b, 1, ""); pb_write_tag(&b, 2, 0); pb_write_varint(&b, 9);
    pb_patch_length(&b, opset_pos);

    size_t graph_pos = pb_begin_submessage(&b, 7);
    pb_write_string(&b, 2, "cnn_test");

    // Conv weight [2,1,3,3] = all 1.0
    float cw[18]; for (int i = 0; i < 18; i++) cw[i] = 1.0f;
    size_t iw = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 1);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "conv_weight");
    pb_write_bytes(&b, 9,cw, sizeof(cw));
    pb_patch_length(&b, iw);

    // Conv bias [2] = [0.0, 0.0]
    float cb[2] = {0.0f, 0.0f};
    size_t ib = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "conv_bias");
    pb_write_bytes(&b, 9,cb, sizeof(cb));
    pb_patch_length(&b, ib);

    // BN scale [2] = [1, 1]
    float bs[2] = {1.0f, 1.0f};
    size_t ibs = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "bn_scale");
    pb_write_bytes(&b, 9,bs, sizeof(bs));
    pb_patch_length(&b, ibs);

    // BN bias [2] = [0.0, 0.0]
    float bb[2] = {0.0f, 0.0f};
    size_t ibb = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "bn_bias");
    pb_write_bytes(&b, 9,bb, sizeof(bb));
    pb_patch_length(&b, ibb);

    // BN mean [2] = [0.0, 0.0]
    float bm[2] = {0.0f, 0.0f};
    size_t ibm = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "bn_mean");
    pb_write_bytes(&b, 9,bm, sizeof(bm));
    pb_patch_length(&b, ibm);

    // BN var [2] = [1.0, 1.0]
    float bv[2] = {1.0f, 1.0f};
    size_t ibv = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "bn_var");
    pb_write_bytes(&b, 9,bv, sizeof(bv));
    pb_patch_length(&b, ibv);

    // Gemm weight [3, 2] = all 1.0 (transB=1)
    float gw[6]; for (int i = 0; i < 6; i++) gw[i] = 1.0f;
    size_t igw = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 2);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "gemm_weight");
    pb_write_bytes(&b, 9,gw, sizeof(gw));
    pb_patch_length(&b, igw);

    // Gemm bias [3] = [0.0, 0.0, 0.0]
    float gb[3] = {0.0f, 0.0f, 0.0f};
    size_t igb = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "gemm_bias");
    pb_write_bytes(&b, 9,gb, sizeof(gb));
    pb_patch_length(&b, igb);

    // Conv node
    size_t cn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "input");
    pb_write_string(&b, 1, "conv_weight");
    pb_write_string(&b, 1, "conv_bias");
    pb_write_string(&b, 2, "conv_out");
    pb_write_string(&b, 3, "conv");
    pb_write_string(&b, 4, "Conv");
    { int64_t ks[] = {3, 3}; write_ints_attr(&b, "kernel_shape", ks, 2);
      int64_t st[] = {1, 1}; write_ints_attr(&b, "strides", st, 2);
      int64_t pd[] = {0, 0}; write_ints_attr(&b, "pads", pd, 2); }
    pb_patch_length(&b, cn);

    // BN node
    size_t bn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "conv_out");
    pb_write_string(&b, 1, "bn_scale");
    pb_write_string(&b, 1, "bn_bias");
    pb_write_string(&b, 1, "bn_mean");
    pb_write_string(&b, 1, "bn_var");
    pb_write_string(&b, 2, "bn_out");
    pb_write_string(&b, 3, "bn");
    pb_write_string(&b, 4, "BatchNormalization");
    write_float_attr(&b, "epsilon", 1e-5f);
    pb_patch_length(&b, bn);

    // Relu node
    size_t rn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "bn_out");
    pb_write_string(&b, 2, "relu_out");
    pb_write_string(&b, 3, "relu");
    pb_write_string(&b, 4, "Relu");
    pb_patch_length(&b, rn);

    // MaxPool node
    size_t pn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "relu_out");
    pb_write_string(&b, 2, "pool_out");
    pb_write_string(&b, 3, "pool");
    pb_write_string(&b, 4, "MaxPool");
    { int64_t ks[] = {2, 2}; write_ints_attr(&b, "kernel_shape", ks, 2);
      int64_t st[] = {2, 2}; write_ints_attr(&b, "strides", st, 2);
      int64_t pd[] = {0, 0}; write_ints_attr(&b, "pads", pd, 2); }
    pb_patch_length(&b, pn);

    // Flatten node
    size_t fn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "pool_out");
    pb_write_string(&b, 2, "flat_out");
    pb_write_string(&b, 3, "flatten");
    pb_write_string(&b, 4, "Flatten");
    pb_patch_length(&b, fn);

    // Gemm node
    size_t gn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "flat_out");
    pb_write_string(&b, 1, "gemm_weight");
    pb_write_string(&b, 1, "gemm_bias");
    pb_write_string(&b, 2, "gemm_out");
    pb_write_string(&b, 3, "gemm");
    pb_write_string(&b, 4, "Gemm");
    write_int_attr(&b, "transB", 1);
    pb_patch_length(&b, gn);

    // Softmax node
    size_t sn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "gemm_out");
    pb_write_string(&b, 2, "output");
    pb_write_string(&b, 3, "softmax");
    pb_write_string(&b, 4, "Softmax");
    pb_patch_length(&b, sn);

    size_t inp = pb_begin_submessage(&b, 11);
    pb_write_string(&b, 1, "input"); pb_patch_length(&b, inp);
    size_t out = pb_begin_submessage(&b, 12);
    pb_write_string(&b, 1, "output"); pb_patch_length(&b, out);

    pb_patch_length(&b, graph_pos);
    *out_size = b.size;
    return b.data;
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

static int test_onnx_load_conv(void) {
    TEST("ONNX load + Conv forward");

    size_t model_size;
    uint8_t* model_bytes = build_test_conv_model(&model_size);
    if (!model_bytes) FAIL("build_test_conv_model failed");

    boat_model_t* model = boat_onnx_load_from_memory(model_bytes, model_size);
    if (!model) { free(model_bytes); FAIL("onnx_load_from_memory failed"); }

    int64_t in_shape[] = {1, 1, 4, 4};
    float in_data[16];
    for (int i = 0; i < 16; i++) in_data[i] = 1.0f;
    boat_tensor_t* input = boat_tensor_from_data(in_shape, 4, BOAT_DTYPE_FLOAT32, in_data);
    if (!input) { boat_model_free(model); free(model_bytes); FAIL("input failed"); }

    boat_tensor_t* output = boat_model_forward(model, input);
    if (!output) { boat_tensor_unref(input); boat_model_free(model); free(model_bytes); FAIL("forward failed"); }

    // Expected shape: [1, 2, 2, 2]
    const int64_t* os = boat_tensor_shape(output);
    if (boat_tensor_ndim(output) != 4 || os[0] != 1 || os[1] != 2 || os[2] != 2 || os[3] != 2) {
        boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
        FAIL("output shape mismatch");
    }

    // Conv(3x3, S=1, P=0) with all-1 input and all-1 weight: each output = sum(3x3) = 9. Bias [1,2]
    // ch0 = 9+1 = 10, ch1 = 9+2 = 11
    const float* od = (const float*)boat_tensor_const_data(output);
    for (int i = 0; i < 4; i++) {
        if (fabsf(od[i] - 10.0f) > 1e-5f) {
            printf("FAIL: ch0[%d]=%f, expected 10.0\n", i, od[i]);
            boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
            return 1;
        }
    }
    for (int i = 4; i < 8; i++) {
        if (fabsf(od[i] - 11.0f) > 1e-5f) {
            printf("FAIL: ch1[%d]=%f, expected 11.0\n", i - 4, od[i]);
            boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
            return 1;
        }
    }

    boat_tensor_unref(input);
    boat_tensor_unref(output);
    boat_model_free(model);
    free(model_bytes);
    PASS(); return 0;
}

static int test_onnx_load_batchnorm(void) {
    TEST("ONNX load + Conv+BN+Relu forward");

    size_t model_size;
    uint8_t* model_bytes = build_test_batchnorm_model(&model_size);
    if (!model_bytes) FAIL("build_test_batchnorm_model failed");

    boat_model_t* model = boat_onnx_load_from_memory(model_bytes, model_size);
    if (!model) { free(model_bytes); FAIL("onnx_load_from_memory failed"); }

    int64_t in_shape[] = {1, 1, 4, 4};
    float in_data[16];
    for (int i = 0; i < 16; i++) in_data[i] = 1.0f;
    boat_tensor_t* input = boat_tensor_from_data(in_shape, 4, BOAT_DTYPE_FLOAT32, in_data);
    if (!input) { boat_model_free(model); free(model_bytes); FAIL("input failed"); }

    boat_tensor_t* output = boat_model_forward(model, input);
    if (!output) { boat_tensor_unref(input); boat_model_free(model); free(model_bytes); FAIL("forward failed"); }

    // Shape: [1, 2, 2, 2]
    const int64_t* os = boat_tensor_shape(output);
    if (boat_tensor_ndim(output) != 4 || os[0] != 1 || os[1] != 2 || os[2] != 2 || os[3] != 2) {
        boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
        FAIL("output shape mismatch");
    }

    // All values should be positive (ReLU after BN)
    const float* od = (const float*)boat_tensor_const_data(output);
    for (int i = 0; i < 8; i++) {
        if (od[i] < 0) {
            boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
            FAIL("expected all positive after BN+Relu");
        }
    }

    boat_tensor_unref(input);
    boat_tensor_unref(output);
    boat_model_free(model);
    free(model_bytes);
    PASS(); return 0;
}

static int test_onnx_load_maxpool(void) {
    TEST("ONNX load + MaxPool forward");

    size_t model_size;
    uint8_t* model_bytes = build_test_maxpool_model(&model_size);
    if (!model_bytes) FAIL("build_test_maxpool_model failed");

    boat_model_t* model = boat_onnx_load_from_memory(model_bytes, model_size);
    if (!model) { free(model_bytes); FAIL("onnx_load_from_memory failed"); }

    int64_t in_shape[] = {1, 1, 4, 4};
    float in_data[16];
    for (int i = 0; i < 16; i++) in_data[i] = (float)(i + 1); // 1..16
    boat_tensor_t* input = boat_tensor_from_data(in_shape, 4, BOAT_DTYPE_FLOAT32, in_data);
    if (!input) { boat_model_free(model); free(model_bytes); FAIL("input failed"); }

    boat_tensor_t* output = boat_model_forward(model, input);
    if (!output) { boat_tensor_unref(input); boat_model_free(model); free(model_bytes); FAIL("forward failed"); }

    // Shape: [1, 1, 2, 2]
    const int64_t* os = boat_tensor_shape(output);
    if (boat_tensor_ndim(output) != 4 || os[0] != 1 || os[1] != 1 || os[2] != 2 || os[3] != 2) {
        boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
        FAIL("output shape mismatch");
    }

    // MaxPool(2x2, S=2): each window selects max
    // windows: [1,2,5,6]=6, [3,4,7,8]=8, [9,10,13,14]=14, [11,12,15,16]=16
    float expected[4] = {6.0f, 8.0f, 14.0f, 16.0f};
    const float* od = (const float*)boat_tensor_const_data(output);
    for (int i = 0; i < 4; i++) {
        if (fabsf(od[i] - expected[i]) > 1e-5f) {
            printf("FAIL: output[%d]=%f, expected %f\n", i, od[i], expected[i]);
            boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
            return 1;
        }
    }

    boat_tensor_unref(input);
    boat_tensor_unref(output);
    boat_model_free(model);
    free(model_bytes);
    PASS(); return 0;
}

static int test_onnx_load_softmax(void) {
    TEST("ONNX load + Gemm+Softmax forward");

    pb_builder_t b;
    pb_builder_init(&b);

    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 4);

    size_t opset_pos = pb_begin_submessage(&b, 8);
    pb_write_string(&b, 1, ""); pb_write_tag(&b, 2, 0); pb_write_varint(&b, 9);
    pb_patch_length(&b, opset_pos);

    size_t graph_pos = pb_begin_submessage(&b, 7);
    pb_write_string(&b, 2, "softmax_test");

    // Gemm weight [3, 4], all 0.1
    float wd[12]; for (int i = 0; i < 12; i++) wd[i] = 0.1f;
    size_t iw = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 4);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "weight");
    pb_write_bytes(&b, 9,wd, sizeof(wd));
    pb_patch_length(&b, iw);

    // Gemm bias [3] = [0, 0, 0]
    float bd[3] = {0, 0, 0};
    size_t ib = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "bias");
    pb_write_bytes(&b, 9,bd, sizeof(bd));
    pb_patch_length(&b, ib);

    // Gemm node
    size_t gn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "input");
    pb_write_string(&b, 1, "weight");
    pb_write_string(&b, 1, "bias");
    pb_write_string(&b, 2, "gemm_out");
    pb_write_string(&b, 3, "gemm");
    pb_write_string(&b, 4, "Gemm");
    write_int_attr(&b, "transB", 1);
    pb_patch_length(&b, gn);

    // Softmax node
    size_t sn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "gemm_out");
    pb_write_string(&b, 2, "output");
    pb_write_string(&b, 3, "softmax");
    pb_write_string(&b, 4, "Softmax");
    pb_patch_length(&b, sn);

    size_t inp = pb_begin_submessage(&b, 11);
    pb_write_string(&b, 1, "input"); pb_patch_length(&b, inp);
    size_t out = pb_begin_submessage(&b, 12);
    pb_write_string(&b, 1, "output"); pb_patch_length(&b, out);

    pb_patch_length(&b, graph_pos);

    uint8_t* model_bytes = b.data;
    size_t model_size = b.size;

    boat_model_t* model = boat_onnx_load_from_memory(model_bytes, model_size);
    if (!model) { free(model_bytes); FAIL("onnx_load_from_memory failed"); }

    int64_t in_shape[] = {2, 4};
    float in_data[] = {0.5f, 1.0f, 1.5f, 2.0f, -0.5f, -1.0f, -1.5f, -2.0f};
    boat_tensor_t* input = boat_tensor_from_data(in_shape, 2, BOAT_DTYPE_FLOAT32, in_data);
    if (!input) { boat_model_free(model); free(model_bytes); FAIL("input failed"); }

    boat_tensor_t* output = boat_model_forward(model, input);
    if (!output) { boat_tensor_unref(input); boat_model_free(model); free(model_bytes); FAIL("forward failed"); }

    // Shape: [2, 3]
    const int64_t* os = boat_tensor_shape(output);
    if (boat_tensor_ndim(output) != 2 || os[0] != 2 || os[1] != 3) {
        boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
        FAIL("output shape mismatch");
    }

    // Softmax should sum to ~1.0 along axis=1
    const float* od = (const float*)boat_tensor_const_data(output);
    for (int batch = 0; batch < 2; batch++) {
        float sum = 0;
        for (int j = 0; j < 3; j++) sum += od[batch * 3 + j];
        if (fabsf(sum - 1.0f) > 1e-4f) {
            printf("FAIL: batch %d softmax sum = %f\n", batch, sum);
            boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
            return 1;
        }
    }

    boat_tensor_unref(input);
    boat_tensor_unref(output);
    boat_model_free(model);
    free(model_bytes);
    PASS(); return 0;
}

static int test_onnx_load_flatten(void) {
    TEST("ONNX load + Gemm+Flatten forward");

    pb_builder_t b;
    pb_builder_init(&b);

    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 4);

    size_t opset_pos = pb_begin_submessage(&b, 8);
    pb_write_string(&b, 1, ""); pb_write_tag(&b, 2, 0); pb_write_varint(&b, 9);
    pb_patch_length(&b, opset_pos);

    size_t graph_pos = pb_begin_submessage(&b, 7);
    pb_write_string(&b, 2, "flatten_test");

    // Gemm weight [3, 4], all 0.1
    float wd[12]; for (int i = 0; i < 12; i++) wd[i] = 0.1f;
    size_t iw = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 4);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "weight");
    pb_write_bytes(&b, 9,wd, sizeof(wd));
    pb_patch_length(&b, iw);

    // Bias [3] = [0.01, 0.02, 0.03]
    float bd[3] = {0.01f, 0.02f, 0.03f};
    size_t ib = pb_begin_submessage(&b, 5);
    pb_write_tag(&b, 1, 0); pb_write_varint(&b, 3);
    pb_write_tag(&b, 2, 0); pb_write_varint(&b, 1);
    pb_write_string(&b, 8, "bias");
    pb_write_bytes(&b, 9,bd, sizeof(bd));
    pb_patch_length(&b, ib);

    // Gemm node
    size_t gn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "input");
    pb_write_string(&b, 1, "weight");
    pb_write_string(&b, 1, "bias");
    pb_write_string(&b, 2, "gemm_out");
    pb_write_string(&b, 3, "gemm");
    pb_write_string(&b, 4, "Gemm");
    write_int_attr(&b, "transB", 1);
    pb_patch_length(&b, gn);

    // Flatten node
    size_t fn = pb_begin_submessage(&b, 1);
    pb_write_string(&b, 1, "gemm_out");
    pb_write_string(&b, 2, "output");
    pb_write_string(&b, 3, "flatten");
    pb_write_string(&b, 4, "Flatten");
    pb_patch_length(&b, fn);

    size_t inp = pb_begin_submessage(&b, 11);
    pb_write_string(&b, 1, "input"); pb_patch_length(&b, inp);
    size_t out = pb_begin_submessage(&b, 12);
    pb_write_string(&b, 1, "output"); pb_patch_length(&b, out);

    pb_patch_length(&b, graph_pos);

    uint8_t* model_bytes = b.data;
    size_t model_size = b.size;

    boat_model_t* model = boat_onnx_load_from_memory(model_bytes, model_size);
    if (!model) { free(model_bytes); FAIL("onnx_load_from_memory failed"); }

    int64_t in_shape[] = {2, 4};
    float in_data[] = {0.5f, 1.0f, 1.5f, 2.0f, -0.5f, -1.0f, -1.5f, -2.0f};
    boat_tensor_t* input = boat_tensor_from_data(in_shape, 2, BOAT_DTYPE_FLOAT32, in_data);
    if (!input) { boat_model_free(model); free(model_bytes); FAIL("input failed"); }

    boat_tensor_t* output = boat_model_forward(model, input);
    if (!output) { boat_tensor_unref(input); boat_model_free(model); free(model_bytes); FAIL("forward failed"); }

    // Flatten from axis=1 on 2D: should be unchanged [2, 3]
    const int64_t* os = boat_tensor_shape(output);
    if (boat_tensor_ndim(output) != 2 || os[0] != 2 || os[1] != 3) {
        boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
        FAIL("output shape mismatch");
    }

    // Gemm: weight=0.1, bias=[0.01,0.02,0.03]
    // batch 0: sum(input)=0.5+1+1.5+2=5, y=5*0.1+bias = [0.51,0.52,0.53]
    // batch 1: sum(input)=-0.5-1-1.5-2=-5, y=-5*0.1+bias = [-0.49,-0.48,-0.47]
    float expected[] = {0.51f, 0.52f, 0.53f, -0.49f, -0.48f, -0.47f};
    const float* od = (const float*)boat_tensor_const_data(output);
    for (int i = 0; i < 6; i++) {
        float diff = fabsf(od[i] - expected[i]);
        if (diff > 1e-5f) {
            printf("FAIL: output[%d]=%f, expected %f\n", i, od[i], expected[i]);
            boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
            return 1;
        }
    }

    boat_tensor_unref(input);
    boat_tensor_unref(output);
    boat_model_free(model);
    free(model_bytes);
    PASS(); return 0;
}

static int test_onnx_load_cnn(void) {
    TEST("ONNX load + full CNN forward");

    size_t model_size;
    uint8_t* model_bytes = build_test_cnn_model(&model_size);
    if (!model_bytes) FAIL("build_test_cnn_model failed");

    boat_model_t* model = boat_onnx_load_from_memory(model_bytes, model_size);
    if (!model) { free(model_bytes); FAIL("onnx_load_from_memory failed"); }

    int64_t in_shape[] = {1, 1, 4, 4};
    float in_data[16];
    for (int i = 0; i < 16; i++) in_data[i] = 1.0f;
    boat_tensor_t* input = boat_tensor_from_data(in_shape, 4, BOAT_DTYPE_FLOAT32, in_data);
    if (!input) { boat_model_free(model); free(model_bytes); FAIL("input failed"); }

    boat_tensor_t* output = boat_model_forward(model, input);
    if (!output) { boat_tensor_unref(input); boat_model_free(model); free(model_bytes); FAIL("forward failed"); }

    // Expected shape: [1, 3]
    const int64_t* os = boat_tensor_shape(output);
    if (boat_tensor_ndim(output) != 2 || os[0] != 1 || os[1] != 3) {
        boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
        FAIL("output shape mismatch");
    }

    // Softmax output should be valid probabilities
    const float* od = (const float*)boat_tensor_const_data(output);
    float sum = 0;
    for (int i = 0; i < 3; i++) {
        if (!isfinite(od[i]) || od[i] < 0 || od[i] > 1) {
            printf("FAIL: output[%d]=%f (invalid prob)\n", i, od[i]);
            boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
            return 1;
        }
        sum += od[i];
    }
    if (fabsf(sum - 1.0f) > 1e-4f) {
        printf("FAIL: softmax sum = %f\n", sum);
        boat_tensor_unref(input); boat_tensor_unref(output); boat_model_free(model); free(model_bytes);
        return 1;
    }

    boat_tensor_unref(input);
    boat_tensor_unref(output);
    boat_model_free(model);
    free(model_bytes);
    PASS(); return 0;
}

static int test_onnx_save_model(void) {
    TEST("ONNX save model -> load -> forward");

    // Build a boat model: Dense(4->3, bias=true) + Relu
    boat_model_t* model = boat_model_create();
    if (!model) FAIL("model_create failed");

    boat_dense_layer_t* dense = boat_dense_layer_create(4, 3, true);
    if (!dense) { boat_model_free(model); FAIL("dense_create failed"); }

    int64_t w_shape[] = {4, 3};
    float w_data[12];
    for (int i = 0; i < 12; i++) w_data[i] = 0.1f * (float)(i + 1);
    boat_tensor_t* w = boat_tensor_from_data(w_shape, 2, BOAT_DTYPE_FLOAT32, w_data);
    boat_dense_layer_set_weight(dense, w);
    boat_tensor_unref(w);

    int64_t b_shape[] = {3};
    float b_data[3] = {0.01f, 0.02f, 0.03f};
    boat_tensor_t* b = boat_tensor_from_data(b_shape, 1, BOAT_DTYPE_FLOAT32, b_data);
    boat_dense_layer_set_bias(dense, b);
    boat_tensor_unref(b);

    boat_layer_t* w1 = (boat_layer_t*)malloc(sizeof(boat_layer_t));
    if (!w1) { boat_dense_layer_free(dense); boat_model_free(model); FAIL("malloc failed"); }
    w1->data = dense; w1->type = BOAT_LAYER_TYPE_DENSE; w1->ops = NULL;
    boat_model_add_layer(model, w1);

    boat_relu_layer_t* relu = boat_relu_layer_create();
    boat_layer_t* w2 = (boat_layer_t*)malloc(sizeof(boat_layer_t));
    if (!w2) { boat_relu_layer_free(relu); boat_model_free(model); FAIL("malloc failed"); }
    w2->data = relu; w2->type = BOAT_LAYER_TYPE_RELU; w2->ops = NULL;
    boat_model_add_layer(model, w2);

    // Save to ONNX memory buffer
    void* onnx_data;
    size_t onnx_size;
    if (!boat_onnx_save_to_memory(model, &onnx_data, &onnx_size)) {
        boat_model_free(model);
        FAIL("save_to_memory failed");
    }

    // Load back from ONNX
    boat_model_t* loaded = boat_onnx_load_from_memory(onnx_data, onnx_size);
    if (!loaded) {
        free(onnx_data); boat_model_free(model);
        FAIL("load_from_memory failed");
    }

    // Forward with original model
    int64_t in_shape[] = {2, 4};
    float in_data[] = {0.5f, 1.0f, 1.5f, 2.0f, -0.5f, -1.0f, -1.5f, -2.0f};
    boat_tensor_t* input = boat_tensor_from_data(in_shape, 2, BOAT_DTYPE_FLOAT32, in_data);
    if (!input) { free(onnx_data); boat_model_free(model); boat_model_free(loaded); FAIL("input failed"); }

    boat_tensor_t* orig_out = boat_model_forward(model, input);
    if (!orig_out) {
        boat_tensor_unref(input); free(onnx_data); boat_model_free(model); boat_model_free(loaded);
        FAIL("original forward failed");
    }

    // Forward with loaded model
    boat_tensor_t* in2 = boat_tensor_from_data(in_shape, 2, BOAT_DTYPE_FLOAT32, in_data);
    boat_tensor_t* loaded_out = boat_model_forward(loaded, in2);
    if (!loaded_out) {
        boat_tensor_unref(orig_out); boat_tensor_unref(input); boat_tensor_unref(in2);
        free(onnx_data); boat_model_free(model); boat_model_free(loaded);
        FAIL("loaded forward failed");
    }

    // Compare
    const float* od_orig = (const float*)boat_tensor_const_data(orig_out);
    const float* od_loaded = (const float*)boat_tensor_const_data(loaded_out);
    bool match = true;
    for (int i = 0; i < 6; i++) {
        if (fabsf(od_orig[i] - od_loaded[i]) > 1e-5f) {
            printf("FAIL: output[%d] orig=%f loaded=%f\n", i, od_orig[i], od_loaded[i]);
            match = false; break;
        }
    }

    boat_tensor_unref(orig_out);
    boat_tensor_unref(loaded_out);
    boat_tensor_unref(input);
    boat_tensor_unref(in2);
    free(onnx_data);
    boat_model_free(model);
    boat_model_free(loaded);

    if (!match) return 1;
    PASS(); return 0;
}

static int test_onnx_save_roundtrip(void) {
    TEST("ONNX save roundtrip (pb -> load -> save -> load)");

    size_t model_size;
    uint8_t* model_bytes = build_test_onnx(&model_size);
    if (!model_bytes) FAIL("build_test_onnx failed");

    boat_model_t* model = boat_onnx_load_from_memory(model_bytes, model_size);
    if (!model) { free(model_bytes); FAIL("initial load failed"); }

    // Forward to get reference
    int64_t in_shape[] = {2, 4};
    float in_data[] = {-0.5f, 0.5f, -1.0f, 1.0f, 1.0f, 0.0f, -1.0f, -0.5f};
    boat_tensor_t* input = boat_tensor_from_data(in_shape, 2, BOAT_DTYPE_FLOAT32, in_data);
    if (!input) { free(model_bytes); boat_model_free(model); FAIL("input failed"); }

    boat_tensor_t* ref_out = boat_model_forward(model, input);
    if (!ref_out) {
        boat_tensor_unref(input); free(model_bytes); boat_model_free(model);
        FAIL("reference forward failed");
    }

    // Save and reload
    void* save_data;
    size_t save_size;
    if (!boat_onnx_save_to_memory(model, &save_data, &save_size)) {
        boat_tensor_unref(ref_out); boat_tensor_unref(input); free(model_bytes);
        boat_model_free(model); FAIL("save_to_memory failed");
    }

    boat_model_t* reloaded = boat_onnx_load_from_memory(save_data, save_size);
    if (!reloaded) {
        boat_tensor_unref(ref_out); boat_tensor_unref(input); free(save_data);
        free(model_bytes); boat_model_free(model);
        FAIL("reload failed");
    }

    boat_tensor_t* in2 = boat_tensor_from_data(in_shape, 2, BOAT_DTYPE_FLOAT32, in_data);
    boat_tensor_t* reloaded_out = boat_model_forward(reloaded, in2);
    if (!reloaded_out) {
        FAIL("reloaded forward failed");
    }

    const float* ref_data = (const float*)boat_tensor_const_data(ref_out);
    const float* reload_data = (const float*)boat_tensor_const_data(reloaded_out);
    bool match = true;
    for (int i = 0; i < 6; i++) {
        if (fabsf(ref_data[i] - reload_data[i]) > 1e-5f) {
            printf("FAIL: output[%d] ref=%f reload=%f\n", i, ref_data[i], reload_data[i]);
            match = false; break;
        }
    }

    boat_tensor_unref(ref_out);
    boat_tensor_unref(reloaded_out);
    boat_tensor_unref(input);
    boat_tensor_unref(in2);
    free(save_data);
    free(model_bytes);
    boat_model_free(model);
    boat_model_free(reloaded);

    if (!match) return 1;
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
    fail |= test_onnx_load_conv();
    fail |= test_onnx_load_batchnorm();
    fail |= test_onnx_load_maxpool();
    fail |= test_onnx_load_softmax();
    fail |= test_onnx_load_flatten();
    fail |= test_onnx_load_cnn();
    fail |= test_onnx_save_model();
    fail |= test_onnx_save_roundtrip();

    printf("\nResults: %d/%d passed\n", tests_passed, tests_total);
    return fail ? 1 : 0;
}
