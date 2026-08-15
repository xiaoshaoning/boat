// test_tensorflow.c - TensorFlow frozen-graph loader tests
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0
//
// Builds a small frozen GraphDef (.pb) in memory with a self-contained
// protobuf writer, loads it via boat_tensorflow_load_from_memory and verifies
// the forward pass against a manual computation. Also exercises
// boat_tensorflow_check / load_savedmodel / save-not-implemented.

#include <boat/format/tensorflow.h>
#include <boat/layers.h>
#include <boat/model.h>
#include <boat/tensor.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

static int g_fail = 0;
#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);                                 \
            g_fail++;                                                                              \
        }                                                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// Minimal protobuf wire-format writer (for building the test graph)
// ---------------------------------------------------------------------------

typedef struct {
    uint8_t* d;
    size_t n;
    size_t cap;
} bw_t;

static void bw_reserve(bw_t* b, size_t extra) {
    if (b->n + extra > b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 256;
        while (nc < b->n + extra)
            nc *= 2;
        b->d = (uint8_t*)realloc(b->d, nc);
        b->cap = nc;
    }
}

static void bw_varint(bw_t* b, uint64_t v) {
    while (v >= 0x80) {
        bw_reserve(b, 1);
        b->d[b->n++] = (uint8_t)(v | 0x80);
        v >>= 7;
    }
    bw_reserve(b, 1);
    b->d[b->n++] = (uint8_t)v;
}

static void bw_tag(bw_t* b, uint32_t field, uint32_t wire) {
    bw_varint(b, ((uint64_t)field << 3) | wire);
}

static void bw_bytes_field(bw_t* b, uint32_t field, const void* data, size_t len) {
    bw_tag(b, field, 2);
    bw_varint(b, len);
    bw_reserve(b, len);
    if (len) memcpy(b->d + b->n, data, len);
    b->n += len;
}

static void bw_str_field(bw_t* b, uint32_t field, const char* s) {
    bw_bytes_field(b, field, s, strlen(s));
}

static size_t bw_submsg_begin(bw_t* b, uint32_t field) {
    bw_tag(b, field, 2);
    size_t len_pos = b->n;
    bw_varint(b, 0); // placeholder length
    return len_pos;
}

static void bw_patch_len(bw_t* b, size_t len_pos) {
    // Length of the submessage contents after the placeholder byte.
    size_t len = b->n - (len_pos + 1);
    uint8_t tmp[10];
    size_t nv = 0;
    uint64_t v = len;
    while (v >= 0x80) {
        tmp[nv++] = (uint8_t)(v | 0x80);
        v >>= 7;
    }
    tmp[nv++] = (uint8_t)v;
    if (nv == 1) {
        b->d[len_pos] = tmp[0];
    } else {
        // Shift the contents right to fit a multi-byte length varint.
        memmove(b->d + len_pos + nv, b->d + len_pos + 1, len);
        for (size_t i = 0; i < nv; i++)
            b->d[len_pos + i] = tmp[i];
        b->n += (nv - 1);
    }
}

// ---------------------------------------------------------------------------
// TensorFlow proto builders
// ---------------------------------------------------------------------------

static void build_tensor_proto(bw_t* b, int ndims, const int64_t* dims, const float* data,
                               size_t nelems) {
    // AttrValue (field 2) { tensor (field 8) { TensorProto } }
    size_t av = bw_submsg_begin(b, 2);
    size_t tp = bw_submsg_begin(b, 8); // AttrValue.tensor
    bw_tag(b, 1, 0);
    bw_varint(b, 1); // dtype = DT_FLOAT
    // tensor_shape (field 2)
    size_t ts = bw_submsg_begin(b, 2);
    for (int i = 0; i < ndims; i++) {
        size_t dim = bw_submsg_begin(b, 2);
        bw_tag(b, 1, 0);
        bw_varint(b, (uint64_t)dims[i]); // dim.size
        bw_patch_len(b, dim);
    }
    bw_patch_len(b, ts);
    bw_bytes_field(b, 4, data, nelems * sizeof(float)); // tensor_content
    bw_patch_len(b, tp);
    bw_patch_len(b, av);
}

static void add_const_node(bw_t* b, const char* name, int ndims, const int64_t* dims,
                           const float* data, size_t nelems) {
    size_t nd = bw_submsg_begin(b, 1); // GraphDef.node
    bw_str_field(b, 1, name);
    bw_str_field(b, 2, "Const");
    // attr { key: "value", value { tensor { ... } } }
    size_t attr = bw_submsg_begin(b, 5);
    bw_str_field(b, 1, "value");
    build_tensor_proto(b, ndims, dims, data, nelems);
    bw_patch_len(b, attr);
    bw_patch_len(b, nd);
}

static void add_compute_node(bw_t* b, const char* name, const char* op, const char* in0,
                             const char* in1) {
    size_t nd = bw_submsg_begin(b, 1);
    bw_str_field(b, 1, name);
    bw_str_field(b, 2, op);
    if (in0) bw_str_field(b, 3, in0);
    if (in1) bw_str_field(b, 3, in1);
    bw_patch_len(b, nd);
}

static void build_test_graph(uint8_t** out, size_t* out_n) {
    bw_t b = {0};
    const float w1[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0}; // [4,3]
    const int64_t w1d[2] = {4, 3};
    const float b1[3] = {0.1f, 0.2f, 0.3f};
    const int64_t b1d[1] = {3};
    const float w2[6] = {1, 0, 0, 1, 1, 1}; // [3,2]
    const int64_t w2d[2] = {3, 2};
    const float b2[2] = {-0.1f, 0.05f};
    const int64_t b2d[1] = {2};

    add_compute_node(&b, "input", "Placeholder", NULL, NULL);
    add_const_node(&b, "w1", 2, w1d, w1, 12);
    add_compute_node(&b, "mm1", "MatMul", "input", "w1");
    add_const_node(&b, "b1", 1, b1d, b1, 3);
    add_compute_node(&b, "bias1", "BiasAdd", "mm1", "b1");
    add_compute_node(&b, "relu1", "Relu", "bias1", NULL);
    add_const_node(&b, "w2", 2, w2d, w2, 6);
    add_compute_node(&b, "mm2", "MatMul", "relu1", "w2");
    add_const_node(&b, "b2", 1, b2d, b2, 2);
    add_compute_node(&b, "bias2", "BiasAdd", "mm2", "b2");
    add_compute_node(&b, "relu2", "Relu", "bias2", NULL);
    add_compute_node(&b, "output", "Identity", "relu2", NULL);

    *out = b.d;
    *out_n = b.n;
}

// Wrap the GraphDef into a SavedModel: SavedModel { meta_graphs (field 2) {
//   MetaGraphDef { graph_def (field 2) { GraphDef } } } }.
static void wrap_savedmodel(const uint8_t* gd, size_t gdn, uint8_t** out, size_t* out_n) {
    bw_t b = {0};
    size_t mg = bw_submsg_begin(&b, 2); // SavedModel.meta_graphs
    bw_bytes_field(&b, 2, gd, gdn);     // MetaGraphDef.graph_def
    bw_patch_len(&b, mg);
    *out = b.d;
    *out_n = b.n;
}

// ---------------------------------------------------------------------------
// Forward reference
// ---------------------------------------------------------------------------

static void relu_vec(float* v, int n) {
    for (int i = 0; i < n; i++)
        v[i] = v[i] > 0 ? v[i] : 0.0f;
}

static void matmul_vec(const float* x, int in, const float* w, int out, float* y) {
    for (int o = 0; o < out; o++) {
        float s = 0;
        for (int i = 0; i < in; i++)
            s += x[i] * w[i * out + o];
        y[o] = s;
    }
}

// ---------------------------------------------------------------------------

static void test_load_and_forward(void) {
    uint8_t* gd;
    size_t gdn;
    build_test_graph(&gd, &gdn);

    boat_model_t* probe = boat_tensorflow_load_from_memory(gd, gdn);
    CHECK(probe != NULL, "load from memory");
    boat_model_free(probe);

    boat_model_t* model = boat_tensorflow_load_from_memory(gd, gdn);
    CHECK(model != NULL, "load from memory (model)");
    if (!model) {
        free(gd);
        return;
    }

    // Input [1,4].
    const float xv[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    int64_t ishape[] = {1, 4};
    boat_tensor_t* x = boat_tensor_from_data(ishape, 2, BOAT_DTYPE_FLOAT32, xv);
    boat_tensor_t* y = boat_model_forward(model, x);
    CHECK(y != NULL, "forward");
    if (y) {
        // Manual: h1 = relu(x @ w1 + b1); y = relu(h1 @ w2 + b2)
        const float w1[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0};
        const float b1[3] = {0.1f, 0.2f, 0.3f};
        const float w2[6] = {1, 0, 0, 1, 1, 1};
        const float b2[2] = {-0.1f, 0.05f};
        float h1[3], h2[2];
        matmul_vec(xv, 4, w1, 3, h1);
        for (int i = 0; i < 3; i++)
            h1[i] += b1[i];
        relu_vec(h1, 3);
        matmul_vec(h1, 3, w2, 2, h2);
        for (int i = 0; i < 2; i++)
            h2[i] += b2[i];
        relu_vec(h2, 2);

        const float* yd = (const float*)boat_tensor_const_data(y);
        CHECK(fabsf(yd[0] - h2[0]) < 1e-4f, "forward matches manual (out 0)");
        CHECK(fabsf(yd[1] - h2[1]) < 1e-4f, "forward matches manual (out 1)");

        // Manual compute of expected values:
        // x @ w1: x=[1,-2,3,-4]; w1 rows: [1,0,0],[0,1,0],[0,0,1],[1,0,0]
        //   o0 = 1*1 + -2*0 + 3*0 + -4*1 = -3
        //   o1 = 1*0 + -2*1 + 3*0 + -4*0 = -2
        //   o2 = 1*0 + -2*0 + 3*1 + -4*0 = 3
        // +b1: [-2.9, -1.8, 3.3]; relu: [0, 0, 3.3]
        // h1 @ w2: w2 rows: [1,0],[0,1],[1,1]
        //   o0 = 0*1 + 0*0 + 3.3*1 = 3.3
        //   o1 = 0*0 + 0*1 + 3.3*1 = 3.3
        // +b2: [3.2, 3.35]; relu: [3.2, 3.35]
        CHECK(fabsf(h2[0] - 3.2f) < 1e-4f && fabsf(h2[1] - 3.35f) < 1e-4f,
              "manual reference values correct");

        boat_tensor_unref(y);
    }
    boat_tensor_unref(x);
    boat_model_free(model);
    free(gd);
}

static void test_check_and_bad_inputs(void) {
    uint8_t* gd;
    size_t gdn;
    build_test_graph(&gd, &gdn);

    // Write to a temp file and check.
    FILE* f = fopen("tf_test_graph.pb", "wb");
    CHECK(f != NULL, "open temp pb");
    if (f) {
        fwrite(gd, 1, gdn, f);
        fclose(f);
        CHECK(boat_tensorflow_check("tf_test_graph.pb"), "check valid pb");
        CHECK(!boat_tensorflow_check("nonexistent_xyz.pb"), "check missing file");
        boat_model_t* m = boat_tensorflow_load("tf_test_graph.pb");
        CHECK(m != NULL, "load from file");
        boat_model_free(m);
    }

    // SavedModel dir.
#ifdef _WIN32
    _mkdir("tf_savedmodel_dir");
#else
    mkdir("tf_savedmodel_dir", 0755);
#endif
    uint8_t* sm;
    size_t smn;
    wrap_savedmodel(gd, gdn, &sm, &smn);
    FILE* d = fopen("tf_savedmodel_dir/saved_model.pb", "wb");
    CHECK(d != NULL, "open saved_model dir");
    if (d) {
        fwrite(sm, 1, smn, d);
        fclose(d);
        CHECK(boat_tensorflow_check("tf_savedmodel_dir"), "check SavedModel dir");
        boat_model_t* m = boat_tensorflow_load_savedmodel("tf_savedmodel_dir");
        CHECK(m != NULL, "load SavedModel");
        boat_model_free(m);
    }

    // Garbage -> NULL.
    CHECK(boat_tensorflow_load_from_memory("not a pb", 9) == NULL, "load garbage -> NULL");
    CHECK(boat_tensorflow_load_from_memory(NULL, 0) == NULL, "load NULL -> NULL");

    // Save is not implemented.
    void* vout = NULL;
    size_t outn = 0;
    CHECK(!boat_tensorflow_save_to_memory(NULL, &vout, &outn), "save_to_memory not implemented");

    free(sm);
    free(gd);
}

int main(void) {
    test_load_and_forward();
    test_check_and_bad_inputs();
    if (g_fail) {
        printf("%d test(s) FAILED\n", g_fail);
        return 1;
    }
    printf("All TensorFlow loader tests passed.\n");
    return 0;
}
