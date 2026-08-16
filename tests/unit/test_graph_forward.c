// test_graph_forward.c - boat_graph_forward + merge layers (concat/add)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <boat/graph.h>
#include <boat/layers.h>
#include <boat/model.h>
#include <boat/ops.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);                                 \
            g_fail++;                                                                              \
        }                                                                                          \
    } while (0)

static boat_tensor_t* f32_tensor(const int64_t* shape, size_t ndim, const float* vals) {
    return boat_tensor_from_data(shape, ndim, BOAT_DTYPE_FLOAT32, vals);
}

static int allclose(const boat_tensor_t* t, const float* expected, size_t n, float tol) {
    if (boat_tensor_nelements(t) != n) return 0;
    const float* d = (const float*)boat_tensor_data(t);
    for (size_t i = 0; i < n; i++)
        if (fabsf(d[i] - expected[i]) > tol) return 0;
    return 1;
}

// Wrap a layer struct into a boat_layer_t for graph nodes (data owned here).
// ops is filled by boat_layer_resolve_ops (invoked by the executor).
static boat_layer_t* wrap_layer(void* data, boat_layer_type_t type) {
    boat_layer_t* w = (boat_layer_t*)malloc(sizeof(boat_layer_t));
    if (!w) return NULL;
    w->data = data;
    w->type = type;
    w->ops = NULL;
    return w;
}

// ---------------------------------------------------------------------------
// 1. Two placeholders -> concat(dim 0) -> output: [3,2] + [2,2] = [5,2]
// ---------------------------------------------------------------------------
static void test_concat_two_inputs(void) {
    boat_graph_t* g = boat_graph_create();
    boat_node_t* p1 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
    boat_node_t* p2 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
    boat_concat_layer_t* cl = boat_concat_layer_create(0);
    boat_node_t* cn = boat_graph_add_node(g, wrap_layer(cl, BOAT_LAYER_TYPE_CONCAT),
                                          BOAT_NODE_TYPE_OPERATION, NULL);
    boat_node_t* out = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);
    boat_graph_add_edge(g, p1, cn, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, p2, cn, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, cn, out, BOAT_EDGE_DIRECTION_FORWARD);

    const float av[6] = {1, 2, 3, 4, 5, 6};
    const float bv[4] = {7, 8, 9, 10};
    const int64_t sa[2] = {3, 2}, sb[2] = {2, 2};
    boat_tensor_t* a = f32_tensor(sa, 2, av);
    boat_tensor_t* b = f32_tensor(sb, 2, bv);

    boat_graph_io_t inputs[2] = {{p1, a}, {p2, b}};
    boat_graph_io_t outputs[1] = {{out, NULL}};
    CHECK(boat_graph_forward(g, inputs, 2, outputs, 1) == 0, "concat forward returns 0");
    boat_tensor_t* o = outputs[0].tensor;
    CHECK(o != NULL, "concat output produced");
    const int64_t* os = o ? boat_tensor_shape(o) : NULL;
    CHECK(os && os[0] == 5 && os[1] == 2, "concat shape [5,2]");
    const float exp_[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    CHECK(o && allclose(o, exp_, 10, 1e-6f), "concat values");

    if (o) boat_tensor_unref(o);
    boat_tensor_unref(a);
    boat_tensor_unref(b);
    boat_concat_layer_free(cl);
    boat_graph_free(g);
}

// ---------------------------------------------------------------------------
// 2. concat(dim -1) (negative axis) + addition layer: [2,3]+[2,4]=[2,7] dim1
//    and element-wise sum of two [2,3] inputs.
// ---------------------------------------------------------------------------
static void test_concat_negative_axis_and_add(void) {
    // concat along the last axis (dim -1 == 1 for 2-D tensors)
    {
        boat_graph_t* g = boat_graph_create();
        boat_node_t* p1 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
        boat_node_t* p2 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
        boat_concat_layer_t* cl = boat_concat_layer_create(-1);
        boat_node_t* cn = boat_graph_add_node(g, wrap_layer(cl, BOAT_LAYER_TYPE_CONCAT),
                                              BOAT_NODE_TYPE_OPERATION, NULL);
        boat_node_t* out = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);
        boat_graph_add_edge(g, p1, cn, BOAT_EDGE_DIRECTION_FORWARD);
        boat_graph_add_edge(g, p2, cn, BOAT_EDGE_DIRECTION_FORWARD);
        boat_graph_add_edge(g, cn, out, BOAT_EDGE_DIRECTION_FORWARD);

        const float av[6] = {1, 2, 3, 4, 5, 6};
        const float bv[8] = {10, 11, 12, 13, 14, 15, 16, 17};
        const int64_t sa[2] = {2, 3}, sb[2] = {2, 4};
        boat_tensor_t* a = f32_tensor(sa, 2, av);
        boat_tensor_t* b = f32_tensor(sb, 2, bv);

        boat_graph_io_t inputs[2] = {{p1, a}, {p2, b}};
        boat_graph_io_t outputs[1] = {{out, NULL}};
        CHECK(boat_graph_forward(g, inputs, 2, outputs, 1) == 0, "concat dim -1 returns 0");
        boat_tensor_t* o = outputs[0].tensor;
        const int64_t* os = o ? boat_tensor_shape(o) : NULL;
        CHECK(os && os[0] == 2 && os[1] == 7, "concat dim -1 shape [2,7]");
        const float exp_[14] = {1, 2, 3, 10, 11, 12, 13, 4, 5, 6, 14, 15, 16, 17};
        CHECK(o && allclose(o, exp_, 14, 1e-6f), "concat dim -1 values");

        if (o) boat_tensor_unref(o);
        boat_tensor_unref(a);
        boat_tensor_unref(b);
        boat_concat_layer_free(cl);
        boat_graph_free(g);
    }

    // addition layer: two [2,3] inputs sum element-wise
    {
        boat_graph_t* g = boat_graph_create();
        boat_node_t* p1 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
        boat_node_t* p2 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
        boat_add_layer_t* al = boat_add_layer_create();
        boat_node_t* an = boat_graph_add_node(g, wrap_layer(al, BOAT_LAYER_TYPE_ADD),
                                              BOAT_NODE_TYPE_OPERATION, NULL);
        boat_node_t* out = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);
        boat_graph_add_edge(g, p1, an, BOAT_EDGE_DIRECTION_FORWARD);
        boat_graph_add_edge(g, p2, an, BOAT_EDGE_DIRECTION_FORWARD);
        boat_graph_add_edge(g, an, out, BOAT_EDGE_DIRECTION_FORWARD);

        const float av[6] = {1, 2, 3, 4, 5, 6};
        const float bv[6] = {10, 20, 30, 40, 50, 60};
        const int64_t sa[2] = {2, 3};
        boat_tensor_t* a = f32_tensor(sa, 2, av);
        boat_tensor_t* b = f32_tensor(sa, 2, bv);

        boat_graph_io_t inputs[2] = {{p1, a}, {p2, b}};
        boat_graph_io_t outputs[1] = {{out, NULL}};
        CHECK(boat_graph_forward(g, inputs, 2, outputs, 1) == 0, "add returns 0");
        boat_tensor_t* o = outputs[0].tensor;
        const float exp_[6] = {11, 22, 33, 44, 55, 66};
        CHECK(o && allclose(o, exp_, 6, 1e-6f), "add values");

        if (o) boat_tensor_unref(o);
        boat_tensor_unref(a);
        boat_tensor_unref(b);
        boat_add_layer_free(al);
        boat_graph_free(g);
    }
}

// ---------------------------------------------------------------------------
// 3. Mixed DAG with branches and both merge ops:
//      p -> dense1 -> relu1
//      p -> dense2
//      relu1 + dense2 -> add
//      p -> dense3
//      add || dense3 -> concat(dim 0) -> out
// ---------------------------------------------------------------------------
static void test_mixed_dag(void) {
    boat_graph_t* g = boat_graph_create();
    boat_node_t* p = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);

    // dense1: 2 -> 3 (weight layout is [in, out])
    boat_dense_layer_t* d1 = boat_dense_layer_create(2, 3, true);
    const float w1[6] = {1, 0, 1, 0, 1, 1};
    const float b1[3] = {0.5f, 0.5f, 0.5f};
    boat_dense_layer_set_weight(d1, f32_tensor((const int64_t[]){2, 3}, 2, w1));
    boat_dense_layer_set_bias(d1, f32_tensor((const int64_t[]){3}, 1, b1));
    boat_node_t* n1 = boat_graph_add_node(g, wrap_layer(d1, BOAT_LAYER_TYPE_DENSE),
                                          BOAT_NODE_TYPE_OPERATION, NULL);
    boat_relu_layer_t* rl = boat_relu_layer_create();
    boat_node_t* n_relu = boat_graph_add_node(g, wrap_layer(rl, BOAT_LAYER_TYPE_RELU),
                                              BOAT_NODE_TYPE_OPERATION, NULL);

    // dense2: 2 -> 3
    boat_dense_layer_t* d2 = boat_dense_layer_create(2, 3, true);
    const float w2[6] = {1, 1, 1, 0, 1, 0};
    const float b2[3] = {0, 0, 0};
    boat_dense_layer_set_weight(d2, f32_tensor((const int64_t[]){2, 3}, 2, w2));
    boat_dense_layer_set_bias(d2, f32_tensor((const int64_t[]){3}, 1, b2));
    boat_node_t* n2 = boat_graph_add_node(g, wrap_layer(d2, BOAT_LAYER_TYPE_DENSE),
                                          BOAT_NODE_TYPE_OPERATION, NULL);

    // add
    boat_add_layer_t* al = boat_add_layer_create();
    boat_node_t* n_add =
        boat_graph_add_node(g, wrap_layer(al, BOAT_LAYER_TYPE_ADD), BOAT_NODE_TYPE_OPERATION, NULL);

    // dense3: 2 -> 3
    boat_dense_layer_t* d3 = boat_dense_layer_create(2, 3, true);
    const float w3[6] = {0, 1, 0, 1, 0, 1};
    const float b3[3] = {1, 1, 1};
    boat_dense_layer_set_weight(d3, f32_tensor((const int64_t[]){2, 3}, 2, w3));
    boat_dense_layer_set_bias(d3, f32_tensor((const int64_t[]){3}, 1, b3));
    boat_node_t* n3 = boat_graph_add_node(g, wrap_layer(d3, BOAT_LAYER_TYPE_DENSE),
                                          BOAT_NODE_TYPE_OPERATION, NULL);

    // concat(dim -1) of add[1,3] and dense3[1,3] -> [1,6]
    boat_concat_layer_t* cl = boat_concat_layer_create(-1);
    boat_node_t* n_cat = boat_graph_add_node(g, wrap_layer(cl, BOAT_LAYER_TYPE_CONCAT),
                                             BOAT_NODE_TYPE_OPERATION, NULL);
    boat_node_t* out = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);

    boat_graph_add_edge(g, p, n1, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, n1, n_relu, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, p, n2, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, n_relu, n_add, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, n2, n_add, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, p, n3, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, n_add, n_cat, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, n3, n_cat, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, n_cat, out, BOAT_EDGE_DIRECTION_FORWARD);

    const float xv[2] = {2, -1};
    boat_tensor_t* x = f32_tensor((const int64_t[]){1, 2}, 2, xv); // [batch, features]
    boat_graph_io_t inputs[1] = {{p, x}};
    boat_graph_io_t outputs[1] = {{out, NULL}};
    CHECK(boat_graph_forward(g, inputs, 1, outputs, 1) == 0, "mixed DAG returns 0");
    boat_tensor_t* o = outputs[0].tensor;
    const int64_t* os = o ? boat_tensor_shape(o) : NULL;
    CHECK(os && os[0] == 1 && os[1] == 6, "mixed DAG output shape [1,6]");

    // manual reference: x = [2, -1]
    // d1: h1 = W1 x^T + b1 = [2*1+(-1)*0, 2*0+(-1)*1, 2*1+(-1)*1] + .5 = [2.5, -.5, 1.5]
    // relu1: [2.5, 0, 1.5]
    // d2: h2 = W2 x^T + b2 = [2*1+(-1)*0, 2*1+(-1)*1, 2*1+(-1)*0] = [2, 1, 2]
    // add: [4.5, 1, 3.5]
    // d3: h3 = W3 x^T + b3 = [0*2+1*(-1), 1*2+0, 0*2+1*(-1)] + 1 = [0, 3, 0]
    // concat: [4.5, 1, 3.5, 0, 3, 0]
    const float exp_[6] = {4.5f, 1.0f, 3.5f, 0.0f, 3.0f, 0.0f};
    CHECK(o && allclose(o, exp_, 6, 1e-5f), "mixed DAG values");

    if (o) boat_tensor_unref(o);
    boat_tensor_unref(x);
    boat_dense_layer_free(d1);
    boat_dense_layer_free(d2);
    boat_dense_layer_free(d3);
    boat_relu_layer_free(rl);
    boat_add_layer_free(al);
    boat_concat_layer_free(cl);
    boat_graph_free(g);
}

// ---------------------------------------------------------------------------
// 4. Error paths: unbound placeholder and cyclic graph are rejected.
// ---------------------------------------------------------------------------
static void test_errors(void) {
    // unbound placeholder
    {
        boat_graph_t* g = boat_graph_create();
        boat_node_t* p1 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
        boat_node_t* p2 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
        boat_concat_layer_t* cl = boat_concat_layer_create(0);
        boat_node_t* cn = boat_graph_add_node(g, wrap_layer(cl, BOAT_LAYER_TYPE_CONCAT),
                                              BOAT_NODE_TYPE_OPERATION, NULL);
        boat_node_t* out = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);
        boat_graph_add_edge(g, p1, cn, BOAT_EDGE_DIRECTION_FORWARD);
        boat_graph_add_edge(g, p2, cn, BOAT_EDGE_DIRECTION_FORWARD);
        boat_graph_add_edge(g, cn, out, BOAT_EDGE_DIRECTION_FORWARD);

        const float av[6] = {1, 2, 3, 4, 5, 6};
        boat_tensor_t* a = f32_tensor((const int64_t[]){3, 2}, 2, av);
        boat_graph_io_t inputs[1] = {{p1, a}}; // p2 never bound
        boat_graph_io_t outputs[1] = {{out, NULL}};
        CHECK(boat_graph_forward(g, inputs, 1, outputs, 1) != 0, "unbound placeholder rejected");
        boat_tensor_unref(a);
        boat_concat_layer_free(cl);
        boat_graph_free(g);
    }

    // cycle
    {
        boat_graph_t* g = boat_graph_create();
        boat_node_t* p = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
        boat_relu_layer_t* rl = boat_relu_layer_create();
        boat_node_t* op = boat_graph_add_node(g, wrap_layer(rl, BOAT_LAYER_TYPE_RELU),
                                              BOAT_NODE_TYPE_OPERATION, NULL);
        boat_graph_add_edge(g, p, op, BOAT_EDGE_DIRECTION_FORWARD);
        boat_graph_add_edge(g, op, p, BOAT_EDGE_DIRECTION_FORWARD); // back edge -> cycle

        const float xv[2] = {1, 2};
        boat_tensor_t* x = f32_tensor((const int64_t[]){2}, 1, xv);
        boat_graph_io_t inputs[1] = {{p, x}};
        boat_graph_io_t outputs[1] = {{op, NULL}};
        CHECK(boat_graph_forward(g, inputs, 1, outputs, 1) != 0, "cycle rejected");
        boat_tensor_unref(x);
        boat_relu_layer_free(rl);
        boat_graph_free(g);
    }
}

// ---------------------------------------------------------------------------
// 5. Custom forward_fn override (non-layer operation nodes).
// ---------------------------------------------------------------------------
static boat_tensor_t* square_op(const boat_graph_t* graph, const boat_node_t* node,
                                const boat_tensor_t* const* inputs, size_t n_inputs) {
    (void)graph;
    (void)node;
    if (n_inputs != 1 || !inputs[0]) return NULL;
    return boat_mul(inputs[0], inputs[0]);
}

static void test_forward_fn_override(void) {
    boat_graph_t* g = boat_graph_create();
    boat_graph_set_forward_fn(g, square_op);
    boat_node_t* p = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
    boat_node_t* op = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    boat_node_t* out = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);
    boat_graph_add_edge(g, p, op, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, op, out, BOAT_EDGE_DIRECTION_FORWARD);

    const float xv[3] = {2, 3, 4};
    boat_tensor_t* x = f32_tensor((const int64_t[]){3}, 1, xv);
    boat_graph_io_t inputs[1] = {{p, x}};
    boat_graph_io_t outputs[1] = {{out, NULL}};
    CHECK(boat_graph_forward(g, inputs, 1, outputs, 1) == 0, "custom fn returns 0");
    boat_tensor_t* o = outputs[0].tensor;
    const float exp_[3] = {4, 9, 16};
    CHECK(o && allclose(o, exp_, 3, 1e-6f), "custom fn squares values");
    if (o) boat_tensor_unref(o);
    boat_tensor_unref(x);
    boat_graph_free(g);
}

// ---------------------------------------------------------------------------
// 6. Constant node data resolves to a tensor (node data = boat_tensor_t*).
// ---------------------------------------------------------------------------
static void test_constant_node(void) {
    boat_graph_t* g = boat_graph_create();
    const float cv[2] = {7, 8};
    boat_tensor_t* c = f32_tensor((const int64_t[]){2}, 1, cv);
    boat_node_t* cn = boat_graph_add_node(g, c, BOAT_NODE_TYPE_CONSTANT, NULL);
    boat_relu_layer_t* rl = boat_relu_layer_create();
    boat_node_t* op = boat_graph_add_node(g, wrap_layer(rl, BOAT_LAYER_TYPE_RELU),
                                          BOAT_NODE_TYPE_OPERATION, NULL);
    boat_node_t* out = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);
    boat_graph_add_edge(g, cn, op, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, op, out, BOAT_EDGE_DIRECTION_FORWARD);

    boat_graph_io_t outputs[1] = {{out, NULL}};
    CHECK(boat_graph_forward(g, NULL, 0, outputs, 1) == 0, "constant graph returns 0");
    boat_tensor_t* o = outputs[0].tensor;
    const float exp_[2] = {7, 8};
    CHECK(o && allclose(o, exp_, 2, 1e-6f), "constant node passthrough");
    if (o) boat_tensor_unref(o);
    boat_relu_layer_free(rl);
    boat_tensor_unref(c);
    boat_graph_free(g);
}

int main(void) {
    test_concat_two_inputs();
    test_concat_negative_axis_and_add();
    test_mixed_dag();
    test_errors();
    test_forward_fn_override();
    test_constant_node();
    if (g_fail) {
        printf("%d graph-forward test(s) FAILED\n", g_fail);
        return 1;
    }
    printf("All graph-forward (concat/add) tests passed.\n");
    return 0;
}
