// node.c - Graph node implementation for computational graph
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/graph.h>
#include <boat/memory.h>
#include <string.h>
#include <stdlib.h>
#include "graph_private.h"

// Internal node structure
struct boat_node_t {
    size_t id;                    // Unique node identifier
    void* data;                   // User data associated with node
    boat_node_type_t type;        // Node type
    void (*free_fn)(void*); // Function to free user data
    size_t ref_count;             // Reference count
};



// Node creation and destruction
static boat_node_t* boat_node_create(void* data, boat_node_type_t type,
                                     void (*free_fn)(void*)) {
    boat_node_t* node = boat_malloc(sizeof(boat_node_t), BOAT_DEVICE_CPU);
    if (!node) {
        return NULL;
    }

    static size_t next_global_id = 1;
    node->id = next_global_id++;
    node->data = data;
    node->type = type;
    node->free_fn = free_fn;
    node->ref_count = 1;

    return node;
}

void boat_node_free(boat_node_t* node) {
    if (!node) {
        return;
    }

    if (--node->ref_count == 0) {
        // Free user data if free function is provided
        if (node->free_fn && node->data) {
            node->free_fn(node->data);
        }
        boat_free(node);
    }
}

void boat_node_ref(boat_node_t* node) {
    if (node) {
        node->ref_count++;
    }
}

void boat_node_unref(boat_node_t* node) {
    boat_node_free(node);
}

// Node properties
size_t boat_graph_node_id(const boat_node_t* node) {
    return node ? node->id : 0;
}

void* boat_node_data(const boat_node_t* node) {
    if (!node) {
        return NULL;
    }
    return node->data;
}

boat_node_type_t boat_node_type(const boat_node_t* node) {
    return node ? node->type : BOAT_NODE_TYPE_VARIABLE;
}

// Node operations for graph
boat_node_t* boat_graph_add_node(boat_graph_t* graph, void* data,
                                 boat_node_type_t type,
                                 void (*free_fn)(void*)) {
    if (!graph) {
        return NULL;
    }

    // Create node
    boat_node_t* node = boat_node_create(data, type, free_fn);
    if (!node) {
        return NULL;
    }

    // Ensure capacity for new node (including adjacency lists)
    if (!ensure_node_capacity(graph, graph->node_count + 1)) {
        return NULL;
    }

    // Assign node ID from graph
    node->id = graph->next_node_id++;
    graph->nodes[graph->node_count++] = node;

    return node;
}

void boat_graph_remove_node(boat_graph_t* graph, boat_node_t* node) {
    if (!graph || !node) {
        return;
    }

    // Find node in array
    size_t index = 0;
    bool found = false;
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == node) {
            index = i;
            found = true;
            break;
        }
    }

    if (!found) {
        return;
    }

    // Remove node from array (shift remaining elements)
    for (size_t i = index; i < graph->node_count - 1; i++) {
        graph->nodes[i] = graph->nodes[i + 1];
    }
    graph->node_count--;

    // Free the node
    boat_node_free(node);
}

boat_node_t* boat_graph_get_node(const boat_graph_t* graph, size_t id) {
    if (!graph) {
        return NULL;
    }

    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] && boat_graph_node_id(graph->nodes[i]) == id) {
            return graph->nodes[i];
        }
    }

    return NULL;
}



// Graph creation and management (basic implementation)
boat_graph_t* boat_graph_create() {
    return boat_graph_create_with_device(BOAT_DEVICE_CPU);
}

boat_graph_t* boat_graph_create_with_device(boat_device_t device) {
    boat_graph_t* graph = boat_malloc(sizeof(boat_graph_t), BOAT_DEVICE_CPU);
    if (!graph) {
        return NULL;
    }

    graph->nodes = NULL;
    graph->node_capacity = 0;
    graph->node_count = 0;
    graph->next_node_id = 1;

    graph->edges = NULL;
    graph->edge_capacity = 0;
    graph->edge_count = 0;

    graph->outgoing = NULL;
    graph->incoming = NULL;

    graph->checkpointing_enabled = false;
    graph->checkpoint_nodes = NULL;

    graph->device = device;
    graph->in_batch_mode = false;

    return graph;
}

// Device management
boat_device_t boat_graph_device(const boat_graph_t* graph) {
    return graph ? graph->device : BOAT_DEVICE_CPU;
}

void boat_graph_set_device(boat_graph_t* graph, boat_device_t device) {
    if (!graph) return;
    graph->device = device;
    // Note: Changing device doesn't move existing data
    // Call boat_graph_to_device() to actually move data
}

void boat_graph_free(boat_graph_t* graph) {
    if (!graph) {
        return;
    }

    // Free all edges first
    for (size_t i = 0; i < graph->edge_count; i++) {
        boat_edge_free(graph->edges[i]);
    }
    boat_free(graph->edges);

    // Free adjacency lists
    if (graph->outgoing) {
        for (size_t i = 0; i < graph->node_capacity; i++) {
            if (graph->outgoing[i]) {
                boat_free(graph->outgoing[i]->edges);
                boat_free(graph->outgoing[i]);
            }
        }
        boat_free(graph->outgoing);
    }
    if (graph->incoming) {
        for (size_t i = 0; i < graph->node_capacity; i++) {
            if (graph->incoming[i]) {
                boat_free(graph->incoming[i]->edges);
                boat_free(graph->incoming[i]);
            }
        }
        boat_free(graph->incoming);
    }

    // Free checkpoint nodes array
    if (graph->checkpoint_nodes) {
        boat_free(graph->checkpoint_nodes);
    }

    // Free all nodes
    for (size_t i = 0; i < graph->node_count; i++) {
        boat_node_free(graph->nodes[i]);
    }

    // Free nodes array
    boat_free(graph->nodes);

    // Free graph structure
    boat_free(graph);
}

// Graph properties
size_t boat_graph_node_count(const boat_graph_t* graph) {
    return graph ? graph->node_count : 0;
}

// Note: Edge operations would require additional data structures
// This is a basic implementation focusing on nodes

// Utility functions for node operations
bool boat_node_is_variable(const boat_node_t* node) {
    return node && node->type == BOAT_NODE_TYPE_VARIABLE;
}

bool boat_node_is_operation(const boat_node_t* node) {
    return node && node->type == BOAT_NODE_TYPE_OPERATION;
}

bool boat_node_is_constant(const boat_node_t* node) {
    return node && node->type == BOAT_NODE_TYPE_CONSTANT;
}

bool boat_node_is_placeholder(const boat_node_t* node) {
    return node && node->type == BOAT_NODE_TYPE_PLACEHOLDER;
}

bool boat_node_is_output(const boat_node_t* node) {
    return node && node->type == BOAT_NODE_TYPE_OUTPUT;
}

// Node data management
void boat_node_set_data(boat_node_t* node, void* data, void (*free_fn)(void*)) {
    if (!node) {
        return;
    }

    // Free old data if exists
    if (node->free_fn && node->data) {
        node->free_fn(node->data);
    }

    node->data = data;
    node->free_fn = free_fn;
}

// Node type conversion
const char* boat_node_type_name(boat_node_type_t type) {
    switch (type) {
        case BOAT_NODE_TYPE_VARIABLE: return "VARIABLE";
        case BOAT_NODE_TYPE_OPERATION: return "OPERATION";
        case BOAT_NODE_TYPE_CONSTANT: return "CONSTANT";
        case BOAT_NODE_TYPE_PLACEHOLDER: return "PLACEHOLDER";
        case BOAT_NODE_TYPE_OUTPUT: return "OUTPUT";
        default: return "UNKNOWN";
    }
}