// test_autodiff_graph.c - Autodiff graph visualization utilities
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/autodiff.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Autodiff Graph Utilities Test ===\n\n");

    int64_t sh[] = {2, 3};
    boat_variable_t* a = boat_variable_create_with_shape(sh, 2, BOAT_DTYPE_FLOAT32, true);
    boat_variable_t* b = boat_variable_create_with_shape(sh, 2, BOAT_DTYPE_FLOAT32, true);
    assert(a && b);

    boat_variable_t* c = boat_var_add(a, b);
    boat_variable_t* d = boat_var_relu(c);
    boat_variable_t* e = boat_var_mul(d, d);
    assert(c && d && e);

    // graph_to_dot returns a non-NULL DOT string.
    char* dot = boat_autodiff_graph_to_dot(e);
    assert(dot != NULL);
    assert(strstr(dot, "digraph") != NULL);
    assert(strstr(dot, "add") != NULL);
    assert(strstr(dot, "relu") != NULL);
    assert(strstr(dot, "mul") != NULL);
    // Variable nodes show their shape.
    assert(strstr(dot, "[2,3]") != NULL);
    boat_free(dot);

    // print_graph must not crash (prints to stdout).
    boat_autodiff_print_graph(e);

    // retain_grad is a stored hint; just verify it does not crash and a
    // variable without a graph is handled.
    boat_variable_retain_grad(c, true);
    boat_variable_retain_grad(NULL, true);

    // graph_to_dot on a variable with no graph returns NULL.
    assert(boat_autodiff_graph_to_dot(NULL) == NULL);

    // checkpointing toggle is a no-op setter (exercised for coverage).
    boat_autodiff_set_grad_checkpointing(true);
    boat_autodiff_set_grad_checkpointing(false);

    boat_variable_free(a);
    boat_variable_free(b);
    boat_variable_free(c);
    boat_variable_free(d);
    boat_variable_free(e);

    printf("\n=== Autodiff graph utility tests passed ===\n");
    return 0;
}
