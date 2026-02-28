// graph_private.h - Private definitions for graph implementation
// Copyright (c) 2026 Boat Framework Authors

#ifndef BOAT_GRAPH_PRIVATE_H
#define BOAT_GRAPH_PRIVATE_H

#include <boat/graph.h>
#include <boat/memory.h>

// Node reference counting (implemented in node.c)
void boat_node_ref(boat_node_t* node);
void boat_node_unref(boat_node_t* node);
void boat_node_free(boat_node_t* node);

// Graph structure definition (must match node.c)
struct boat_graph_t {
    boat_node_t** nodes;          // Array of nodes
    size_t node_capacity;         // Capacity of nodes array
    size_t node_count;            // Current number of nodes
    size_t next_node_id;          // Next available node ID

    // Edge storage
    struct boat_edge_t** edges;   // Array of all edges
    size_t edge_capacity;         // Capacity of edges array
    size_t edge_count;            // Current number of edges

    // Adjacency lists
    struct boat_edge_list_t** outgoing;  // Outgoing edges per node
    struct boat_edge_list_t** incoming;  // Incoming edges per node

    // Gradient checkpointing
    bool checkpointing_enabled;
    bool* checkpoint_nodes;       // Array marking checkpoint nodes (size = node_capacity)

    // Device management
    boat_device_t device;         // Default device for graph operations

    // Batch modification state
    bool in_batch_mode;           // Whether batch modifications are active
};

// Edge list structure for adjacency lists
typedef struct boat_edge_list_t {
    struct boat_edge_t** edges;   // Array of edge pointers
    size_t capacity;              // Capacity of edges array
    size_t count;                 // Current number of edges
} boat_edge_list_t;

// Edge list operations (implemented in edge.c)
boat_edge_list_t* boat_edge_list_create();
void boat_edge_list_free(boat_edge_list_t* list);
bool boat_edge_list_add(boat_edge_list_t* list, struct boat_edge_t* edge);
bool boat_edge_list_remove(boat_edge_list_t* list, struct boat_edge_t* edge);
bool boat_edge_list_contains(const boat_edge_list_t* list, struct boat_edge_t* edge);
size_t boat_edge_list_count(const boat_edge_list_t* list);
struct boat_edge_t* boat_edge_list_get(const boat_edge_list_t* list, size_t index);

// Edge creation (implemented in edge.c)
struct boat_edge_t* boat_edge_create(boat_node_t* from, boat_node_t* to,
                                     boat_edge_direction_t direction);
void boat_edge_free(struct boat_edge_t* edge);

// Graph capacity helpers (implemented in graph.c)
bool ensure_node_capacity(boat_graph_t* graph, size_t needed_capacity);
bool ensure_edge_capacity(boat_graph_t* graph, size_t needed_capacity);

#endif // BOAT_GRAPH_PRIVATE_H