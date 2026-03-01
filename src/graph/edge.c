// edge.c - Edge management for computational graph
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/graph.h>
#include <boat/memory.h>
#include <string.h>
#include <stdlib.h>
#include "graph_private.h"

// Internal edge structure (already defined in node.c, but we need access)
struct boat_edge_t {
    boat_node_t* from;            // Source node
    boat_node_t* to;              // Target node
    boat_edge_direction_t direction; // Edge direction
};

// Edge creation and destruction (non-static versions)
boat_edge_t* boat_edge_create(const boat_node_t* from, const boat_node_t* to,
                              boat_edge_direction_t direction) {
    if (!from || !to) {
        return NULL;
    }

    boat_edge_t* edge = boat_malloc(sizeof(boat_edge_t), BOAT_DEVICE_CPU);
    if (!edge) {
        return NULL;
    }

    edge->from = from;
    edge->to = to;
    edge->direction = direction;

    // Increment reference counts
    boat_node_ref(from);
    boat_node_ref(to);

    return edge;
}

void boat_edge_free(const boat_edge_t* edge) {
    if (!edge) {
        return;
    }

    // Decrement reference counts
    boat_node_unref(edge->from);
    boat_node_unref(edge->to);

    boat_free(edge);
}

// Edge properties (public API implementations)
boat_node_t* boat_edge_source(const boat_edge_t* edge) {
    return edge ? edge->from : NULL;
}

boat_node_t* boat_edge_target(const boat_edge_t* edge) {
    return edge ? edge->to : NULL;
}

boat_edge_direction_t boat_edge_direction(const boat_edge_t* edge) {
    return edge ? edge->direction : BOAT_EDGE_DIRECTION_FORWARD;
}

// Edge comparison (internal)
bool boat_edge_equal(const boat_edge_t* a, const boat_edge_t* b) {
    if (!a || !b) return false;
    return a->from == b->from && a->to == b->to && a->direction == b->direction;
}


boat_edge_list_t* boat_edge_list_create() {
    boat_edge_list_t* list = boat_malloc(sizeof(boat_edge_list_t), BOAT_DEVICE_CPU);
    if (!list) return NULL;

    list->edges = NULL;
    list->capacity = 0;
    list->count = 0;
    return list;
}

void boat_edge_list_free(boat_edge_list_t* list) {
    if (!list) return;
    // Note: we don't free edges themselves, graph owns them
    boat_free(list->edges);
    boat_free(list);
}

bool boat_edge_list_add(boat_edge_list_t* list, const boat_edge_t* edge) {
    if (!list || !edge) return false;

    if (list->count >= list->capacity) {
        size_t new_capacity = list->capacity == 0 ? 8 : list->capacity * 2;
        boat_edge_t** new_edges = boat_realloc(list->edges,
                                               new_capacity * sizeof(boat_edge_t*),
                                               BOAT_DEVICE_CPU);
        if (!new_edges) return false;

        list->edges = new_edges;
        list->capacity = new_capacity;
    }

    list->edges[list->count++] = edge;
    return true;
}

bool boat_edge_list_remove(boat_edge_list_t* list, const boat_edge_t* edge) {
    if (!list || !edge) return false;

    for (size_t i = 0; i < list->count; i++) {
        if (list->edges[i] == edge) {
            // Shift remaining elements
            for (size_t j = i; j < list->count - 1; j++) {
                list->edges[j] = list->edges[j + 1];
            }
            list->count--;
            return true;
        }
    }
    return false;
}

bool boat_edge_list_contains(const boat_edge_list_t* list, const boat_edge_t* edge) {
    if (!list || !edge) return false;
    for (size_t i = 0; i < list->count; i++) {
        if (list->edges[i] == edge) return true;
    }
    return false;
}

size_t boat_edge_list_count(const boat_edge_list_t* list) {
    return list ? list->count : 0;
}

boat_edge_t* boat_edge_list_get(const boat_edge_list_t* list, size_t index) {
    if (!list || index >= list->count) return NULL;
    return list->edges[index];
}