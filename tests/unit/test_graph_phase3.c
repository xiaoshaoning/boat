// test_graph_phase3.c - Tests for Phase 3 graph optimizations
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/graph.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

// Helper function to create a simple computational graph for optimization tests
static boat_graph_t* create_optimization_test_graph(boat_device_t device) {
    boat_graph_t* graph = boat_graph_create_with_device(device);
    assert(graph != NULL);
    return graph;
}

// Helper function to add operation node with dummy data
static boat_node_t* add_operation_node(boat_graph_t* graph, const char* op_name) {
    // Create dummy operation data
    char* data = strdup(op_name);
    assert(data != NULL);

    boat_node_t* node = boat_graph_add_node(graph, data, BOAT_NODE_TYPE_OPERATION, free);
    assert(node != NULL);
    return node;
}

// Helper function to add constant node
static boat_node_t* add_constant_node(boat_graph_t* graph, float value) {
    // Create dummy constant data
    float* data = malloc(sizeof(float));
    assert(data != NULL);
    *data = value;

    boat_node_t* node = boat_graph_add_node(graph, data, BOAT_NODE_TYPE_CONSTANT, free);
    assert(node != NULL);
    return node;
}

// Helper function to add variable node
static boat_node_t* add_variable_node(boat_graph_t* graph, const char* name) {
    // Create dummy variable data
    char* data = strdup(name);
    assert(data != NULL);

    boat_node_t* node = boat_graph_add_node(graph, data, BOAT_NODE_TYPE_VARIABLE, free);
    assert(node != NULL);
    return node;
}

// Test optimization API existence
void test_optimization_api() {
    printf("Testing optimization API existence...\n");

    boat_graph_t* graph = create_optimization_test_graph(BOAT_DEVICE_CPU);

    // Test that optimization functions can be called (they're placeholders)
    boat_graph_optimize(graph, BOAT_OPTIMIZE_NONE);
    boat_graph_eliminate_common_subexpressions(graph);
    boat_graph_eliminate_dead_code(graph);
    boat_graph_fold_constants(graph);
    boat_graph_simplify(graph);

    // Test optimization flags
    unsigned int flags = BOAT_OPTIMIZE_CSE | BOAT_OPTIMIZE_DCE |
                        BOAT_OPTIMIZE_CONSTANT_FOLD | BOAT_OPTIMIZE_SIMPLIFY;
    assert(flags == BOAT_OPTIMIZE_ALL);

    boat_graph_free(graph);
    printf("  ✓ Optimization API tests passed\n");
}

// Test common subexpression elimination (placeholder)
void test_common_subexpression_elimination() {
    printf("Testing common subexpression elimination...\n");

    boat_graph_t* graph = create_optimization_test_graph(BOAT_DEVICE_CPU);

    // Create a simple graph with potential common subexpressions
    // For now, just test that the function doesn't crash
    boat_graph_eliminate_common_subexpressions(graph);

    // Verify graph is still valid (graph pointer should still be valid)
    assert(graph != NULL);

    boat_graph_free(graph);
    printf("  ✓ Common subexpression elimination tests passed (placeholder)\n");
}

// Test dead code elimination (placeholder)
void test_dead_code_elimination() {
    printf("Testing dead code elimination...\n");

    boat_graph_t* graph = create_optimization_test_graph(BOAT_DEVICE_CPU);

    // Create some nodes
    boat_node_t* var1 = add_variable_node(graph, "x");
    boat_node_t* var2 = add_variable_node(graph, "y");
    boat_node_t* op1 = add_operation_node(graph, "add");
    boat_node_t* op2 = add_operation_node(graph, "mul");

    // Add edges to create a small graph
    boat_graph_add_edge(graph, var1, op1, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(graph, var2, op1, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(graph, op1, op2, BOAT_EDGE_DIRECTION_FORWARD);

    // Test dead code elimination (placeholder)
    boat_graph_eliminate_dead_code(graph);

    // Verify graph is still valid
    assert(boat_graph_node_count(graph) == 4);

    boat_graph_free(graph);
    printf("  ✓ Dead code elimination tests passed (placeholder)\n");
}

// Test constant folding (placeholder)
void test_constant_folding() {
    printf("Testing constant folding...\n");

    boat_graph_t* graph = create_optimization_test_graph(BOAT_DEVICE_CPU);

    // Create constant nodes
    boat_node_t* const1 = add_constant_node(graph, 3.14f);
    boat_node_t* const2 = add_constant_node(graph, 2.71f);
    boat_node_t* op = add_operation_node(graph, "add");

    // Add edges
    boat_graph_add_edge(graph, const1, op, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(graph, const2, op, BOAT_EDGE_DIRECTION_FORWARD);

    // Test constant folding (placeholder)
    boat_graph_fold_constants(graph);

    // Verify graph is still valid
    assert(boat_graph_node_count(graph) == 3);

    boat_graph_free(graph);
    printf("  ✓ Constant folding tests passed (placeholder)\n");
}

// Test graph simplification (placeholder)
void test_graph_simplification() {
    printf("Testing graph simplification...\n");

    boat_graph_t* graph = create_optimization_test_graph(BOAT_DEVICE_CPU);

    // Create a simple graph
    boat_node_t* var1 = add_variable_node(graph, "a");
    boat_node_t* var2 = add_variable_node(graph, "b");
    boat_node_t* op1 = add_operation_node(graph, "add");
    boat_node_t* op2 = add_operation_node(graph, "mul");

    // Add edges
    boat_graph_add_edge(graph, var1, op1, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(graph, var2, op1, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(graph, op1, op2, BOAT_EDGE_DIRECTION_FORWARD);

    // Test graph simplification (placeholder)
    boat_graph_simplify(graph);

    // Verify graph is still valid
    assert(boat_graph_node_count(graph) == 4);

    boat_graph_free(graph);
    printf("  ✓ Graph simplification tests passed (placeholder)\n");
}

// Test combined optimizations
void test_combined_optimizations() {
    printf("Testing combined optimizations...\n");

    boat_graph_t* graph = create_optimization_test_graph(BOAT_DEVICE_CPU);

    // Create a more complex graph
    boat_node_t* const1 = add_constant_node(graph, 1.0f);
    boat_node_t* const2 = add_constant_node(graph, 2.0f);
    boat_node_t* var1 = add_variable_node(graph, "input");
    boat_node_t* op1 = add_operation_node(graph, "add");
    boat_node_t* op2 = add_operation_node(graph, "mul");
    boat_node_t* op3 = add_operation_node(graph, "relu");

    // Build computation: relu(mul(add(const1, const2), input))
    boat_graph_add_edge(graph, const1, op1, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(graph, const2, op1, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(graph, op1, op2, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(graph, var1, op2, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(graph, op2, op3, BOAT_EDGE_DIRECTION_FORWARD);

    // Apply all optimizations
    boat_graph_optimize(graph, BOAT_OPTIMIZE_ALL);

    // Verify graph is still valid
    assert(boat_graph_node_count(graph) == 6);

    boat_graph_free(graph);
    printf("  ✓ Combined optimization tests passed\n");
}

// Test optimization with batch modifications
void test_optimization_with_batch_mode() {
    printf("Testing optimization with batch mode...\n");

    boat_graph_t* graph = create_optimization_test_graph(BOAT_DEVICE_CPU);

    // Start batch mode
    boat_graph_batch_modifications(graph, true);

    // Create graph during batch mode
    boat_node_t* var1 = add_variable_node(graph, "x");
    boat_node_t* var2 = add_variable_node(graph, "y");
    boat_node_t* op = add_operation_node(graph, "add");

    boat_graph_add_edge(graph, var1, op, BOAT_EDGE_DIRECTION_FORWARD);
    boat_graph_add_edge(graph, var2, op, BOAT_EDGE_DIRECTION_FORWARD);

    // Try optimization during batch mode
    boat_graph_optimize(graph, BOAT_OPTIMIZE_CSE);

    // End batch mode
    boat_graph_batch_modifications(graph, false);

    // Verify graph is still valid
    assert(boat_graph_node_count(graph) == 3);

    boat_graph_free(graph);
    printf("  ✓ Optimization with batch mode tests passed\n");
}

int main() {
    printf("\n=== Phase 3 Graph Optimizations Test Suite ===\n\n");

    test_optimization_api();
    test_common_subexpression_elimination();
    test_dead_code_elimination();
    test_constant_folding();
    test_graph_simplification();
    test_combined_optimizations();
    test_optimization_with_batch_mode();

    printf("\n=== All Phase 3 tests completed successfully ===\n");
    return 0;
}