// test_graph_optimize.c - Dead-node elimination and edge cleanup
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/graph.h>
#include <boat/layers.h>
#include <boat/model.h>
#include <boat/tensor.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_fail = 0;
#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);            \
            g_fail++;                                                         \
        }                                                                     \
    } while (0)

static int node_in_graph(const boat_graph_t* g, const boat_node_t* n) {
    size_t cnt = boat_graph_node_count(g);
    for (size_t i = 0; i < cnt; i++) {
        if (boat_graph_get_node_at_index(g, i) == n) return 1;
    }
    return 0;
}

static void test_prune_dead_branch(void) {
    boat_graph_t* g = boat_graph_create();
    boat_node_t* input = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
    boat_node_t* a = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    boat_node_t* b = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    boat_node_t* output = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);
    // Dead branch: not connected to the output.
    boat_node_t* d = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    boat_node_t* e = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    CHECK(boat_graph_node_count(g) == 6, "graph has 6 nodes");

    // Live path: input -> a -> b -> output.
    boat_graph_add_edge(g, input, a, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, a, b, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, b, output, BOAT_EDGE_DIRECTION_FORWARD);
    // Dead branch: input -> d -> e (e goes nowhere).
    boat_graph_add_edge(g, input, d, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, d, e, BOAT_EDGE_DIRECTION_FORWARD);

    const boat_node_t* outputs[] = {output};
    CHECK(boat_graph_prune_unreachable(g, outputs, 1), "prune succeeds");

    CHECK(boat_graph_node_count(g) == 4, "dead branch removed (4 nodes left)");
    CHECK(node_in_graph(g, input) && node_in_graph(g, a) && node_in_graph(g, b) &&
              node_in_graph(g, output),
          "live path kept");
    CHECK(!node_in_graph(g, d) && !node_in_graph(g, e), "dead nodes removed");
    CHECK(boat_graph_edge_count(g) == 3, "3 forward edges remain");
    boat_graph_free(g);
}

static void test_prune_multi_output(void) {
    boat_graph_t* g = boat_graph_create();
    boat_node_t* input = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
    boat_node_t* a = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    boat_node_t* o1 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);
    boat_node_t* o2 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);
    // Dead branch.
    boat_node_t* dead = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    boat_graph_add_edge(g, input, a, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, a, o1, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, a, o2, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, input, dead, BOAT_EDGE_DIRECTION_FORWARD);

    const boat_node_t* outputs[] = {o1, o2};
    boat_graph_prune_unreachable(g, outputs, 2);
    CHECK(boat_graph_node_count(g) == 4, "multi-output prune keeps 4 nodes");
    CHECK(!node_in_graph(g, dead), "dead node removed (multi-output)");
    boat_graph_free(g);
}

static void test_optimize_dup_edges(void) {
    boat_graph_t* g = boat_graph_create();
    boat_node_t* input = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
    boat_node_t* a = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    boat_node_t* output = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);
    boat_graph_add_edge(g, input, a, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, a, output, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, a, output, BOAT_EDGE_DIRECTION_FORWARD);  // duplicate
    CHECK(boat_graph_edge_count(g) == 3, "3 edges before optimize");

    boat_graph_optimize(g, BOAT_OPTIMIZE_ALL);
    CHECK(boat_graph_node_count(g) == 3, "all nodes kept (all reachable)");
    CHECK(boat_graph_edge_count(g) == 2, "duplicate edge removed");
    boat_graph_free(g);
}

static void test_flag_dce(void) {
    // The flag-based boat_graph_optimize(_, BOAT_OPTIMIZE_DCE) must find OUTPUT
    // nodes itself and eliminate the dead branch.
    boat_graph_t* g = boat_graph_create();
    boat_node_t* input = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_PLACEHOLDER, NULL);
    boat_node_t* a = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    boat_node_t* output = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);
    boat_node_t* dead = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    boat_graph_add_edge(g, input, a, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, a, output, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, input, dead, BOAT_EDGE_DIRECTION_FORWARD);
    CHECK(boat_graph_node_count(g) == 4, "4 nodes before flag DCE");

    boat_graph_optimize(g, BOAT_OPTIMIZE_DCE);
    CHECK(boat_graph_node_count(g) == 3, "flag DCE removed the dead node");
    CHECK(!node_in_graph(g, dead), "dead node gone (flag DCE)");
    boat_graph_free(g);
}

static void test_invalid_args(void) {
    boat_graph_t* g = boat_graph_create();
    boat_node_t* n = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    CHECK(!boat_graph_prune_unreachable(NULL, (const boat_node_t* const*)&n, 1),
          "prune NULL graph -> false");
    CHECK(!boat_graph_prune_unreachable(g, NULL, 1), "prune NULL outputs -> false");
    CHECK(!boat_graph_prune_unreachable(g, (const boat_node_t* const*)&n, 0),
          "prune zero outputs -> false");
    boat_graph_free(g);
}

// --- Constant folding ------------------------------------------------------

// Evaluator for the test graph: an OPERATION node whose data is the string
// "add" sums the tensors of its constant inputs.
static boat_tensor_t* test_evaluator(const boat_graph_t* graph, const boat_node_t* op_node) {
    const char* op_name = (const char*)boat_node_data(op_node);
    if (!op_name || strcmp(op_name, "add") != 0) return NULL;
    float acc = 0.0f;
    int n = 0;
    size_t ne = boat_graph_edge_count(graph);
    for (size_t e = 0; e < ne; e++) {
        const boat_edge_t* edge = boat_graph_get_edge_at_index(graph, e);
        if (!edge || boat_edge_direction(edge) != BOAT_EDGE_DIRECTION_FORWARD) continue;
        if (boat_edge_target(edge) == op_node) {
            const boat_node_t* src = boat_edge_source(edge);
            if (boat_node_type(src) != BOAT_NODE_TYPE_CONSTANT) return NULL;
            const float* d = (const float*)boat_node_data(src);
            acc += d[0];
            n++;
        }
    }
    if (n == 0) return NULL;
    int64_t sh[] = {1};
    float v = acc;
    return boat_tensor_from_data(sh, 1, BOAT_DTYPE_FLOAT32, &v);
}

static void test_fold_constants(void) {
    boat_graph_t* g = boat_graph_create();
    float* c2 = (float*)malloc(sizeof(float));
    float* c3 = (float*)malloc(sizeof(float));
    *c2 = 2.0f;
    *c3 = 3.0f;
    boat_node_t* ca = boat_graph_add_node(g, c2, BOAT_NODE_TYPE_CONSTANT, free);
    boat_node_t* cb = boat_graph_add_node(g, c3, BOAT_NODE_TYPE_CONSTANT, free);
    boat_node_t* op = boat_graph_add_node(g, (void*)"add", BOAT_NODE_TYPE_OPERATION, NULL);
    boat_node_t* out = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OUTPUT, NULL);
    boat_graph_add_edge(g, ca, op, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, cb, op, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(g, op, out, BOAT_EDGE_DIRECTION_FORWARD);
    CHECK(boat_graph_node_count(g) == 4, "graph has 4 nodes before fold");

    boat_graph_set_evaluator(g, test_evaluator);
    boat_graph_fold_constants(g);

    CHECK(boat_graph_node_count(g) == 4, "op folded into a constant (ca, cb, const, out)");
    // The output node must still receive a value (from the new constant).
    size_t in_edges = 0;
    size_t ne = boat_graph_edge_count(g);
    for (size_t e = 0; e < ne; e++) {
        const boat_edge_t* edge = boat_graph_get_edge_at_index(g, e);
        if (edge && boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD &&
            boat_edge_target(edge) == out) {
            in_edges++;
        }
    }
    CHECK(in_edges == 1, "output still fed by one node after fold");
    // The node feeding the output must be the folded constant holding 5.0.
    int found5 = 0;
    for (size_t e = 0; e < ne; e++) {
        const boat_edge_t* edge = boat_graph_get_edge_at_index(g, e);
        if (edge && boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD &&
            boat_edge_target(edge) == out) {
            const boat_node_t* src = boat_edge_source(edge);
            if (boat_node_type(src) == BOAT_NODE_TYPE_CONSTANT) {
                const float* d =
                    (const float*)boat_tensor_data((boat_tensor_t*)boat_node_data(src));
                if (d && d[0] == 5.0f) found5 = 1;
            }
        }
    }
    CHECK(found5, "folded constant feeds the output and holds 5.0");
    boat_graph_free(g);
}

// --- Dense/Conv + ReLU fusion (model forward) -----------------------------

static boat_layer_t* wrap_dense(boat_dense_layer_t* d) {
    boat_layer_t* w = (boat_layer_t*)malloc(sizeof(boat_layer_t));
    w->data = d;
    w->type = BOAT_LAYER_TYPE_DENSE;
    w->ops = NULL;
    return w;
}

static boat_layer_t* wrap_relu(void) {
    boat_layer_t* w = (boat_layer_t*)malloc(sizeof(boat_layer_t));
    w->data = boat_relu_layer_create();
    w->type = BOAT_LAYER_TYPE_RELU;
    w->ops = NULL;
    return w;
}

static void test_dense_relu_fusion(void) {
    // Dense(4,3) -> ReLU -> Dense(3,2): the first ReLU must be fused into the
    // dense forward (in-place relu), giving the same result as relu(dense(x)).
    const float w1[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0};  // [4,3]
    const float b1[3] = {0.1f, 0.2f, 0.3f};
    const float w2[6] = {1, 0, 0, 1, 1, 1};                     // [3,2]
    const float b2[2] = {-0.1f, 0.05f};

    boat_dense_layer_t* d1 = boat_dense_layer_create(4, 3, true);
    boat_dense_layer_t* d2 = boat_dense_layer_create(3, 2, true);
    int64_t s1[] = {4, 3}, s2[] = {3, 2}, sb[] = {3}, sb2[] = {2};
    boat_tensor_t* w1t = boat_tensor_from_data(s1, 2, BOAT_DTYPE_FLOAT32, w1);
    boat_tensor_t* b1t = boat_tensor_from_data(sb, 1, BOAT_DTYPE_FLOAT32, b1);
    boat_tensor_t* w2t = boat_tensor_from_data(s2, 2, BOAT_DTYPE_FLOAT32, w2);
    boat_tensor_t* b2t = boat_tensor_from_data(sb2, 1, BOAT_DTYPE_FLOAT32, b2);
    boat_dense_layer_set_weight(d1, w1t);
    boat_dense_layer_set_bias(d1, b1t);
    boat_dense_layer_set_weight(d2, w2t);
    boat_dense_layer_set_bias(d2, b2t);
    boat_tensor_unref(w1t);
    boat_tensor_unref(b1t);
    boat_tensor_unref(w2t);
    boat_tensor_unref(b2t);

    boat_model_t* model = boat_model_create();
    boat_model_add_layer(model, wrap_dense(d1));
    boat_model_add_layer(model, wrap_relu());
    boat_model_add_layer(model, wrap_dense(d2));

    const float xv[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    int64_t ish[] = {1, 4};
    boat_tensor_t* x = boat_tensor_from_data(ish, 2, BOAT_DTYPE_FLOAT32, xv);
    boat_tensor_t* y = boat_model_forward(model, x);
    CHECK(y != NULL, "fused model forward");
    if (y) {
        // manual: h = relu(x @ w1 + b1); out = h @ w2 + b2
        float h[3], out[2];
        for (int o = 0; o < 3; o++) {
            float s = 0;
            for (int i = 0; i < 4; i++) s += xv[i] * w1[i * 3 + o];
            h[o] = (s + b1[o]) > 0 ? (s + b1[o]) : 0.0f;
        }
        for (int o = 0; o < 2; o++) {
            float s = 0;
            for (int i = 0; i < 3; i++) s += h[i] * w2[i * 2 + o];
            out[o] = s + b2[o];
        }
        const float* yd = (const float*)boat_tensor_const_data(y);
        CHECK(fabsf(yd[0] - out[0]) < 1e-4f && fabsf(yd[1] - out[1]) < 1e-4f,
              "fused Dense+ReLU forward matches manual");
        boat_tensor_unref(y);
    }
    boat_tensor_unref(x);
    boat_model_free(model);
}

int main(void) {
    test_prune_dead_branch();
    test_prune_multi_output();
    test_optimize_dup_edges();
    test_flag_dce();
    test_invalid_args();
    test_fold_constants();
    test_dense_relu_fusion();
    if (g_fail) {
        printf("%d test(s) FAILED\n", g_fail);
        return 1;
    }
    printf("All graph optimize tests passed.\n");
    return 0;
}
