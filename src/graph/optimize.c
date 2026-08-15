// optimize.c - Dynamic graph optimizations: dead-node elimination and
// idempotent edge cleanup.
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/graph.h>
#include <boat/memory.h>
#include <stdlib.h>

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
