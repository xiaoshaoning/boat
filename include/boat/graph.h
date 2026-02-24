// graph.h - Computational graph for automatic differentiation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_GRAPH_H
#define BOAT_GRAPH_H

#include <stddef.h>
#include <stdbool.h>
#include <boat/tensor.h>
#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct boat_graph_t boat_graph_t;
typedef struct boat_node_t boat_node_t;
typedef struct boat_edge_t boat_edge_t;

// Node types for computational graph
typedef enum {
    BOAT_NODE_TYPE_VARIABLE,    // Variable node (leaf)
    BOAT_NODE_TYPE_OPERATION,   // Operation node
    BOAT_NODE_TYPE_CONSTANT,    // Constant node
    BOAT_NODE_TYPE_PLACEHOLDER, // Placeholder node (input)
    BOAT_NODE_TYPE_OUTPUT       // Output node
} boat_node_type_t;

// Node type utilities
BOAT_API const char* boat_node_type_name(boat_node_type_t type);

// Edge direction
typedef enum {
    BOAT_EDGE_DIRECTION_FORWARD,  // Forward edge (data flow)
    BOAT_EDGE_DIRECTION_BACKWARD  // Backward edge (gradient flow)
} boat_edge_direction_t;

// Graph creation and management
BOAT_API boat_graph_t* boat_graph_create();
BOAT_API boat_graph_t* boat_graph_create_with_device(boat_device_t device);
BOAT_API void boat_graph_free(boat_graph_t* graph);
BOAT_API boat_graph_t* boat_graph_copy(const boat_graph_t* graph);

// Node operations
BOAT_API boat_node_t* boat_graph_add_node(boat_graph_t* graph, void* data, boat_node_type_t type,
                                 void (*free_fn)(void*));
BOAT_API void boat_graph_remove_node(boat_graph_t* graph, boat_node_t* node);
BOAT_API boat_node_t* boat_graph_get_node(const boat_graph_t* graph, size_t id);
BOAT_API size_t boat_graph_node_id(const boat_node_t* node);
BOAT_API void* boat_node_data(const boat_node_t* node);
BOAT_API boat_node_type_t boat_node_type(const boat_node_t* node);

// Edge operations
BOAT_API boat_edge_t* boat_graph_add_edge(boat_graph_t* graph, boat_node_t* from, boat_node_t* to,
                                 boat_edge_direction_t direction);
BOAT_API void boat_graph_remove_edge(boat_graph_t* graph, boat_edge_t* edge);
BOAT_API boat_node_t* boat_edge_source(const boat_edge_t* edge);
BOAT_API boat_node_t* boat_edge_target(const boat_edge_t* edge);
BOAT_API boat_edge_direction_t boat_edge_direction(const boat_edge_t* edge);

// Device management
BOAT_API boat_device_t boat_graph_device(const boat_graph_t* graph);
BOAT_API void boat_graph_set_device(boat_graph_t* graph, boat_device_t device);
BOAT_API bool boat_graph_to_device(boat_graph_t* graph, boat_device_t device);
BOAT_API size_t boat_graph_device_memory_usage(const boat_graph_t* graph, boat_device_t device);

// Graph topology
BOAT_API size_t boat_graph_node_count(const boat_graph_t* graph);
BOAT_API size_t boat_graph_edge_count(const boat_graph_t* graph);
BOAT_API size_t boat_graph_in_degree(const boat_graph_t* graph, const boat_node_t* node);
BOAT_API size_t boat_graph_out_degree(const boat_graph_t* graph, const boat_node_t* node);
BOAT_API boat_node_t* boat_graph_get_node_at_index(const boat_graph_t* graph, size_t index);
BOAT_API boat_edge_t* boat_graph_get_edge_at_index(const boat_graph_t* graph, size_t index);

// Graph traversal
typedef void (*boat_node_visitor_t)(boat_node_t* node, void* user_data);
typedef void (*boat_edge_visitor_t)(boat_edge_t* edge, void* user_data);

BOAT_API void boat_graph_dfs(const boat_graph_t* graph, boat_node_t* start,
                    boat_node_visitor_t pre_visit, boat_node_visitor_t post_visit,
                    void* user_data);
BOAT_API void boat_graph_bfs(const boat_graph_t* graph, boat_node_t* start,
                    boat_node_visitor_t visit, void* user_data);
BOAT_API void boat_graph_topological_sort(const boat_graph_t* graph, boat_node_t** sorted_nodes,
                                 size_t* count);

// Graph properties
BOAT_API bool boat_graph_is_acyclic(const boat_graph_t* graph);
BOAT_API bool boat_graph_is_connected(const boat_graph_t* graph);
BOAT_API bool boat_graph_has_path(const boat_graph_t* graph, boat_node_t* from, boat_node_t* to);

// Subgraph operations
BOAT_API boat_graph_t* boat_graph_subgraph(const boat_graph_t* graph, boat_node_t** nodes,
                                  size_t node_count);
BOAT_API void boat_graph_merge(boat_graph_t* dest, const boat_graph_t* src);

// Graph visualization
BOAT_API void boat_graph_print(const boat_graph_t* graph);
BOAT_API char* boat_graph_to_dot(const boat_graph_t* graph);

// Gradient checkpointing
BOAT_API void boat_graph_enable_checkpointing(boat_graph_t* graph, bool enabled);
BOAT_API bool boat_graph_checkpointing_enabled(const boat_graph_t* graph);
BOAT_API void boat_graph_mark_checkpoint(boat_graph_t* graph, boat_node_t* node);
BOAT_API bool boat_graph_is_checkpoint(const boat_graph_t* graph, const boat_node_t* node);

// Dynamic graph modifications
BOAT_API void boat_graph_validate(const boat_graph_t* graph);
BOAT_API bool boat_graph_can_add_edge(const boat_graph_t* graph, boat_node_t* from, boat_node_t* to);
BOAT_API bool boat_graph_can_remove_node(const boat_graph_t* graph, boat_node_t* node);

// Real-time graph modification during training
BOAT_API boat_edge_t* boat_graph_safe_add_edge(boat_graph_t* graph, boat_node_t* from, boat_node_t* to,
                                     boat_edge_direction_t direction);
BOAT_API bool boat_graph_safe_remove_node(boat_graph_t* graph, boat_node_t* node);
BOAT_API bool boat_graph_safe_replace_node(boat_graph_t* graph, boat_node_t* old_node, boat_node_t* new_node);
BOAT_API void boat_graph_batch_modifications(boat_graph_t* graph, bool begin);

// Node migration between graphs
BOAT_API bool boat_graph_migrate_node(boat_graph_t* dest_graph, boat_graph_t* src_graph, boat_node_t* node);

// Graph optimizations
BOAT_API void boat_graph_optimize(boat_graph_t* graph, unsigned int optimization_flags);
BOAT_API void boat_graph_eliminate_common_subexpressions(boat_graph_t* graph);
BOAT_API void boat_graph_eliminate_dead_code(boat_graph_t* graph);
BOAT_API void boat_graph_fold_constants(boat_graph_t* graph);
BOAT_API void boat_graph_simplify(boat_graph_t* graph);

// Optimization flags
#define BOAT_OPTIMIZE_NONE          0x00
#define BOAT_OPTIMIZE_CSE           0x01  // Common subexpression elimination
#define BOAT_OPTIMIZE_DCE           0x02  // Dead code elimination
#define BOAT_OPTIMIZE_CONSTANT_FOLD 0x04  // Constant folding
#define BOAT_OPTIMIZE_SIMPLIFY      0x08  // Graph simplification
#define BOAT_OPTIMIZE_ALL           0x0F  // All optimizations

// Computational graph specific functions (for autodiff)
BOAT_API boat_graph_t* boat_computation_graph_create();
BOAT_API void boat_computation_graph_forward(boat_graph_t* graph);
BOAT_API void boat_computation_graph_backward(boat_graph_t* graph);
BOAT_API void boat_computation_graph_clear_gradients(boat_graph_t* graph);

#ifdef __cplusplus
}
#endif

#endif // BOAT_GRAPH_H