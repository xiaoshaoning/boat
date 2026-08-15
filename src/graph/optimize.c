// optimize.c - Dynamic graph optimizations: dead-node elimination, edge
// cleanup, and constant folding.
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/graph.h>
#include <boat/memory.h>
#include <boat/tensor.h>
#include "graph_private.h"
#include <stdlib.h>

static void tensor_unref_free(void* p) {
    if (p) boat_tensor_unref((boat_tensor_t*)p);
}

BOAT_API void boat_graph_set_evaluator(boat_graph_t* graph, boat_graph_evaluator_t eval) {
    if (!graph) return;
    graph->evaluator = eval;
}

// A node is foldable when every forward input is a CONSTANT node.
static bool all_inputs_constant(const boat_graph_t* graph, const boat_node_t* node) {
    const size_t ne = boat_graph_edge_count(graph);
    for (size_t e = 0; e < ne; e++) {
        const boat_edge_t* edge = boat_graph_get_edge_at_index(graph, e);
        if (!edge || boat_edge_direction(edge) != BOAT_EDGE_DIRECTION_FORWARD) continue;
        if (boat_edge_target(edge) == node &&
            boat_node_type(boat_edge_source(edge)) != BOAT_NODE_TYPE_CONSTANT) {
            return false;
        }
    }
    return true;
}

// Fold OPERATION nodes whose inputs are all constants into a single CONSTANT
// node, using the graph's pluggable evaluator. Iterates to fold chains.
BOAT_API void boat_graph_fold_constants(const boat_graph_t* graph) {
    if (!graph) return;
    boat_graph_t* g = (boat_graph_t*)graph;
    if (!g->evaluator) return;

    int changed = 1;
    while (changed) {
        changed = 0;
        const size_t n = boat_graph_node_count(g);
        for (size_t i = 0; i < n; i++) {
            boat_node_t* node = boat_graph_get_node_at_index(g, i);
            if (!node || boat_node_type(node) != BOAT_NODE_TYPE_OPERATION) continue;
            if (!all_inputs_constant(g, node)) continue;

            boat_tensor_t* val = g->evaluator(g, node);
            if (!val) continue;

            // The constant node takes ownership of the folded tensor (its
            // free_fn unrefs it); the evaluator hands over its reference.
            boat_node_t* cn =
                boat_graph_add_node(g, val, BOAT_NODE_TYPE_CONSTANT, tensor_unref_free);
            if (!cn) {
                boat_tensor_unref(val);
                continue;
            }

            // Rewire the op's outgoing forward edges to the new constant.
            const size_t ne = boat_graph_edge_count(g);
            boat_node_t** consumers = (boat_node_t**)boat_malloc(
                ne * sizeof(boat_node_t*), boat_graph_device(g));
            if (!consumers) {
                boat_graph_safe_remove_node(g, cn);
                continue;
            }
            size_t ncons = 0;
            for (size_t e = 0; e < ne; e++) {
                const boat_edge_t* edge = boat_graph_get_edge_at_index(g, e);
                if (!edge || boat_edge_direction(edge) != BOAT_EDGE_DIRECTION_FORWARD) continue;
                if (boat_edge_source(edge) == node) consumers[ncons++] =
                    (boat_node_t*)boat_edge_target(edge);
            }
            for (size_t k = 0; k < ncons; k++) {
                boat_graph_add_edge(g, cn, consumers[k], BOAT_EDGE_DIRECTION_FORWARD);
            }
            boat_free(consumers);

            // Remove the folded operation node (frees its data and edges).
            boat_graph_safe_remove_node(g, node);
            changed = 1;
            break;  // node list changed; restart the scan
        }
    }

    if (!g->in_batch_mode) {
        boat_graph_validate(g);
    }
}

// Map a node pointer to its index in the graph's node array.
static int node_index(const boat_graph_t* graph, const boat_node_t* node) {
    const size_t n = boat_graph_node_count(graph);
    for (size_t i = 0; i < n; i++) {
        if (boat_graph_get_node_at_index(graph, i) == node) return (int)i;
    }
    return -1;
}

BOAT_API bool boat_graph_prune_unreachable(boat_graph_t* graph,
                                           const boat_node_t* const* outputs,
                                           size_t n_outputs) {
    if (!graph || !outputs || n_outputs == 0) return false;

    const size_t n = boat_graph_node_count(graph);
    if (n == 0) return true;

    bool* marked = (bool*)boat_calloc(n, boat_graph_device(graph));
    if (!marked) return false;

    // Seed: the outputs themselves are always kept.
    for (size_t i = 0; i < n_outputs; i++) {
        if (!outputs[i]) continue;
        int idx = node_index(graph, outputs[i]);
        if (idx >= 0) marked[idx] = true;
    }

    // Fixpoint: a node is needed when it has a forward edge into a needed node
    // (i.e. its output flows toward an output, possibly transitively).
    int changed = 1;
    while (changed) {
        changed = 0;
        const size_t ne = boat_graph_edge_count(graph);
        for (size_t i = 0; i < n; i++) {
            if (marked[i]) continue;
            const boat_node_t* node = boat_graph_get_node_at_index(graph, i);
            for (size_t e = 0; e < ne; e++) {
                const boat_edge_t* edge = boat_graph_get_edge_at_index(graph, e);
                if (!edge || boat_edge_direction(edge) != BOAT_EDGE_DIRECTION_FORWARD) continue;
                if (boat_edge_source(edge) != node) continue;
                int t = node_index(graph, boat_edge_target(edge));
                if (t >= 0 && marked[t]) {
                    marked[i] = true;
                    changed = 1;
                    break;
                }
            }
        }
    }

    // Remove unmarked nodes (and their edges). Collect first: removal shifts
    // the node array, so indices are stale during removal.
    boat_node_t** dead = (boat_node_t**)boat_malloc(n * sizeof(boat_node_t*),
                                                    boat_graph_device(graph));
    if (!dead) {
        boat_free(marked);
        return false;
    }
    size_t dead_count = 0;
    for (size_t i = 0; i < n; i++) {
        if (!marked[i]) {
            dead[dead_count++] = boat_graph_get_node_at_index(graph, i);
        }
    }
    for (size_t i = 0; i < dead_count; i++) {
        boat_graph_safe_remove_node(graph, dead[i]);
    }

    boat_free(dead);
    boat_free(marked);
    return true;
}

// Remove duplicate forward edges between the same (source, target) pair.
BOAT_API bool boat_graph_remove_duplicate_edges(boat_graph_t* graph) {
    if (!graph) return false;
    const size_t ne = boat_graph_edge_count(graph);
    if (ne == 0) return true;
    boat_edge_t** dup =
        (boat_edge_t**)boat_malloc(ne * sizeof(boat_edge_t*), boat_graph_device(graph));
    if (!dup) return false;
    size_t dup_count = 0;
    for (size_t i = 0; i < ne; i++) {
        const boat_edge_t* ei = boat_graph_get_edge_at_index(graph, i);
        if (!ei || boat_edge_direction(ei) != BOAT_EDGE_DIRECTION_FORWARD) continue;
        for (size_t j = 0; j < i; j++) {
            const boat_edge_t* ej = boat_graph_get_edge_at_index(graph, j);
            if (!ej) continue;
            if (boat_edge_source(ej) == boat_edge_source(ei) &&
                boat_edge_target(ej) == boat_edge_target(ei)) {
                dup[dup_count++] = (boat_edge_t*)ei;
                break;
            }
        }
    }
    for (size_t i = 0; i < dup_count; i++) {
        boat_graph_remove_edge(graph, dup[i]);
    }
    boat_free(dup);
    return true;
}
