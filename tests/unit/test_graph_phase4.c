// test_graph_phase4.c - Tests for graph robustness fixes (merge, to_dot,
// node-removal adjacency consistency) and sequential model lifecycle.
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/graph.h>
#include <boat/model.h>
#include <boat/layers.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

// Regression: boat_graph_merge must not overflow its node-index map.
static void test_graph_merge(void) {
    printf("Testing graph merge...\n");

    boat_graph_t* src = boat_graph_create();
    boat_graph_t* dest = boat_graph_create();
    assert(src != NULL);
    assert(dest != NULL);

    boat_node_t* a = boat_graph_add_node(src, NULL, BOAT_NODE_TYPE_VARIABLE, NULL);
    boat_node_t* b = boat_graph_add_node(src, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(boat_graph_add_edge(src, a, b, BOAT_EDGE_DIRECTION_FORWARD) != NULL);

    size_t src_nodes = boat_graph_node_count(src);
    size_t src_edges = boat_graph_edge_count(src);

    boat_graph_merge(dest, src);
    assert(boat_graph_node_count(dest) == src_nodes);
    assert(boat_graph_edge_count(dest) == src_edges);

    boat_graph_free(src);
    boat_graph_free(dest);
    printf("  OK graph merge\n");
}

// Regression: boat_graph_to_dot must not overflow its output buffer.
static void test_graph_to_dot(void) {
    printf("Testing graph to_dot...\n");

    boat_graph_t* g = boat_graph_create();
    assert(g != NULL);

    // Enough nodes/edges to exercise the buffer sizing math.
    boat_node_t* nodes[16];
    for (int i = 0; i < 16; i++) {
        nodes[i] = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
        assert(nodes[i] != NULL);
    }
    for (int i = 0; i < 15; i++) {
        assert(boat_graph_add_edge(g, nodes[i], nodes[i + 1], BOAT_EDGE_DIRECTION_FORWARD) != NULL);
    }

    char* dot = boat_graph_to_dot(g);
    assert(dot != NULL);
    assert(strncmp(dot, "digraph", 7) == 0);
    boat_free(dot);

    boat_graph_free(g);
    printf("  OK graph to_dot\n");
}

// Regression: removing a middle node must keep the adjacency lists aligned
// with the (shifted) nodes array.
static void test_node_removal_adjacency(void) {
    printf("Testing node removal adjacency consistency...\n");

    boat_graph_t* g = boat_graph_create();
    assert(g != NULL);

    boat_node_t* n0 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_VARIABLE, NULL);
    boat_node_t* n1 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    boat_node_t* n2 = boat_graph_add_node(g, NULL, BOAT_NODE_TYPE_OPERATION, NULL);
    assert(n0 != NULL && n1 != NULL && n2 != NULL);

    // n2 -> n0 (a persistent edge that must survive the removal of n1).
    assert(boat_graph_add_edge(g, n2, n0, BOAT_EDGE_DIRECTION_FORWARD) != NULL);
    assert(boat_graph_out_degree(g, n2) == 1);
    assert(boat_graph_in_degree(g, n0) == 1);

    // Remove the middle node n1 (index 1); n2 shifts left to index 1.
    assert(boat_graph_safe_remove_node(g, n1) == true);
    assert(boat_graph_node_count(g) == 2);

    // The n2 -> n0 edge must still be reachable through the adjacency lists.
    assert(boat_graph_out_degree(g, n2) == 1);
    assert(boat_graph_in_degree(g, n0) == 1);

    boat_graph_free(g);
    printf("  OK node removal adjacency\n");
}

// Regression: boat_sequential_create/Add/free must not leak the internal
// layer array (the private state destructor must free it).
static void test_sequential_lifecycle(void) {
    printf("Testing sequential model lifecycle...\n");

    boat_sequential_model_t* model = boat_sequential_create();
    assert(model != NULL);

    // Add enough layers to force growth of the internal array.
    for (int i = 0; i < 8; i++) {
        boat_dense_layer_t* dense = boat_dense_layer_create(4, 2, true);
        assert(dense != NULL);

        boat_layer_t* wrapper = (boat_layer_t*)malloc(sizeof(boat_layer_t));
        assert(wrapper != NULL);
        wrapper->data = dense;
        wrapper->ops = NULL; // auto-assigned by boat_model_add_layer
        wrapper->type = BOAT_LAYER_TYPE_DENSE;
        boat_sequential_add(model, wrapper);
    }

    // boat_model_free frees the wrappers AND the layer data (via the
    // auto-assigned ops) AND the internal layer array (via the private-state
    // destructor). No separate layer free is needed.
    boat_model_free((boat_model_t*)model);

    printf("  OK sequential lifecycle\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0); // Disable output buffering
    printf("\n=== Phase 4 Graph Robustness Test Suite ===\n\n");

    test_graph_merge();
    test_graph_to_dot();
    test_node_removal_adjacency();
    test_sequential_lifecycle();

    printf("\n=== All Phase 4 tests completed successfully ===\n");
    return 0;
}
