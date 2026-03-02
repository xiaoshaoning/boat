// graph.c - Graph algorithms and operations
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/graph.h>
#include <boat/memory.h>
#include <boat/tensor.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "graph_private.h"

// Helper function to remove all edges connected to a node
static void remove_all_edges_for_node(const boat_graph_t* graph, const boat_node_t* node) {
    if (!graph || !node) return;

    // Create a list of edges to remove (we can't modify while iterating)
    size_t edges_to_remove_count = 0;
    struct boat_edge_t** edges_to_remove = boat_malloc(graph->edge_count * sizeof(struct boat_edge_t*), graph->device);
    if (!edges_to_remove) return;

    for (size_t i = 0; i < graph->edge_count; i++) {
        struct boat_edge_t* edge = graph->edges[i];
        if (!edge) continue;

        if (boat_edge_source(edge) == node || boat_edge_target(edge) == node) {
            edges_to_remove[edges_to_remove_count++] = edge;
        }
    }

    // Remove collected edges
    for (size_t i = 0; i < edges_to_remove_count; i++) {
        boat_graph_remove_edge(graph, edges_to_remove[i]);
    }

    boat_free(edges_to_remove);
}

// Helper function to ensure adjacency lists capacity
bool ensure_node_capacity(boat_graph_t* graph, size_t needed_capacity) {
    if (needed_capacity <= graph->node_capacity) {
        return true;
    }

    size_t new_capacity = graph->node_capacity == 0 ? 8 : graph->node_capacity * 2;
    while (new_capacity < needed_capacity) {
        new_capacity *= 2;
    }

    // Resize nodes array
    boat_node_t** new_nodes = boat_realloc(graph->nodes,
                                           new_capacity * sizeof(boat_node_t*),
                                           graph->device);
    if (!new_nodes) return false;
    graph->nodes = new_nodes;

    // Resize outgoing array
    boat_edge_list_t** new_outgoing = boat_realloc(graph->outgoing,
                                                   new_capacity * sizeof(boat_edge_list_t*),
                                                   graph->device);
    if (!new_outgoing) return false;
    graph->outgoing = new_outgoing;

    // Resize incoming array
    boat_edge_list_t** new_incoming = boat_realloc(graph->incoming,
                                                   new_capacity * sizeof(boat_edge_list_t*),
                                                   graph->device);
    if (!new_incoming) return false;
    graph->incoming = new_incoming;

    // Initialize new slots to NULL
    for (size_t i = graph->node_capacity; i < new_capacity; i++) {
        graph->nodes[i] = NULL;
        graph->outgoing[i] = NULL;
        graph->incoming[i] = NULL;
    }

    graph->node_capacity = new_capacity;
    return true;
}

// Helper function to ensure edge array capacity
bool ensure_edge_capacity(const boat_graph_t* graph, size_t needed_capacity) {
    if (needed_capacity <= graph->edge_capacity) {
        return true;
    }

    size_t new_capacity = graph->edge_capacity == 0 ? 8 : graph->edge_capacity * 2;
    while (new_capacity < needed_capacity) {
        new_capacity *= 2;
    }

    struct boat_edge_t** new_edges = boat_realloc(graph->edges,
                                                  new_capacity * sizeof(struct boat_edge_t*),
                                                  graph->device);
    if (!new_edges) return false;
    graph->edges = new_edges;
    graph->edge_capacity = new_capacity;
    return true;
}


// Edge operations
boat_edge_t* boat_graph_add_edge(boat_graph_t* graph, const boat_node_t* from, const boat_node_t* to,
                                 boat_edge_direction_t direction) {
    if (!graph || !from || !to) {
        return NULL;
    }

    // Find node indices (position in nodes array)
    size_t from_index = SIZE_MAX;
    size_t to_index = SIZE_MAX;
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == from) from_index = i;
        if (graph->nodes[i] == to) to_index = i;
        if (from_index != SIZE_MAX && to_index != SIZE_MAX) break;
    }
    if (from_index == SIZE_MAX || to_index == SIZE_MAX) {
        return NULL; // Nodes not in this graph
    }

    // Create edge
    struct boat_edge_t* edge = boat_edge_create(from, to, direction);
    if (!edge) {
        return NULL;
    }

    // Add to edges array
    if (!ensure_edge_capacity(graph, graph->edge_count + 1)) {
        boat_edge_free(edge);
        return NULL;
    }
    graph->edges[graph->edge_count++] = edge;

    // Ensure adjacency lists exist
    if (!graph->outgoing[from_index]) {
        graph->outgoing[from_index] = boat_edge_list_create();
        if (!graph->outgoing[from_index]) {
            // Rollback edge addition
            graph->edge_count--;
            boat_edge_free(edge);
            return NULL;
        }
    }
    if (!graph->incoming[to_index]) {
        graph->incoming[to_index] = boat_edge_list_create();
        if (!graph->incoming[to_index]) {
            // Rollback
            if (boat_edge_list_count(graph->outgoing[from_index]) == 0) {
                boat_edge_list_free(graph->outgoing[from_index]);
                graph->outgoing[from_index] = NULL;
            }
            graph->edge_count--;
            boat_edge_free(edge);
            return NULL;
        }
    }

    // Add to adjacency lists
    if (!boat_edge_list_add(graph->outgoing[from_index], edge) ||
        !boat_edge_list_add(graph->incoming[to_index], edge)) {
        // Rollback
        boat_edge_list_remove(graph->outgoing[from_index], edge);
        boat_edge_list_remove(graph->incoming[to_index], edge);
        if (boat_edge_list_count(graph->outgoing[from_index]) == 0) {
            boat_edge_list_free(graph->outgoing[from_index]);
            graph->outgoing[from_index] = NULL;
        }
        if (boat_edge_list_count(graph->incoming[to_index]) == 0) {
            boat_edge_list_free(graph->incoming[to_index]);
            graph->incoming[to_index] = NULL;
        }
        graph->edge_count--;
        boat_edge_free(edge);
        return NULL;
    }

    return edge;
}

void boat_graph_remove_edge(boat_graph_t* graph, const boat_edge_t* edge) {
    if (!graph || !edge) return;

    // Find edge in edges array
    size_t edge_index = SIZE_MAX;
    for (size_t i = 0; i < graph->edge_count; i++) {
        if (graph->edges[i] == edge) {
            edge_index = i;
            break;
        }
    }
    if (edge_index == SIZE_MAX) return; // Edge not in graph

    // Find source and target node indices
    size_t from_index = SIZE_MAX;
    size_t to_index = SIZE_MAX;
    const boat_node_t* from = boat_edge_source(edge);
    const boat_node_t* to = boat_edge_target(edge);
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == from) from_index = i;
        if (graph->nodes[i] == to) to_index = i;
        if (from_index != SIZE_MAX && to_index != SIZE_MAX) break;
    }

    // Remove from adjacency lists if nodes are in graph
    if (from_index != SIZE_MAX && graph->outgoing[from_index]) {
        boat_edge_list_remove(graph->outgoing[from_index], edge);
        if (boat_edge_list_count(graph->outgoing[from_index]) == 0) {
            boat_edge_list_free(graph->outgoing[from_index]);
            graph->outgoing[from_index] = NULL;
        }
    }
    if (to_index != SIZE_MAX && graph->incoming[to_index]) {
        boat_edge_list_remove(graph->incoming[to_index], edge);
        if (boat_edge_list_count(graph->incoming[to_index]) == 0) {
            boat_edge_list_free(graph->incoming[to_index]);
            graph->incoming[to_index] = NULL;
        }
    }

    // Remove from edges array (swap with last element)
    graph->edges[edge_index] = graph->edges[graph->edge_count - 1];
    graph->edge_count--;

    // Free edge
    boat_edge_free(edge);
}

// Graph topology functions
size_t boat_graph_edge_count(const boat_graph_t* graph) {
    return graph ? graph->edge_count : 0;
}

size_t boat_graph_in_degree(const boat_graph_t* graph, const boat_node_t* node) {
    if (!graph || !node) return 0;

    // Find node index
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == node) {
            return graph->incoming[i] ? boat_edge_list_count(graph->incoming[i]) : 0;
        }
    }
    return 0;
}

size_t boat_graph_out_degree(const boat_graph_t* graph, const boat_node_t* node) {
    if (!graph || !node) return 0;

    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == node) {
            return graph->outgoing[i] ? boat_edge_list_count(graph->outgoing[i]) : 0;
        }
    }
    return 0;
}

// Graph traversal algorithms
void boat_graph_dfs(const boat_graph_t* graph, const boat_node_t* start,
                    boat_node_visitor_t pre_visit, boat_node_visitor_t post_visit,
                    void* user_data) {
    if (!graph || !start) return;

    // Find start index
    size_t start_index = SIZE_MAX;
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == start) {
            start_index = i;
            break;
        }
    }
    if (start_index == SIZE_MAX) return;

    bool* visited = boat_calloc(graph->node_count * sizeof(bool), BOAT_DEVICE_CPU);
    if (!visited) return;

    // Use iterative DFS with stack
    size_t* stack = boat_malloc(graph->node_count * sizeof(size_t), BOAT_DEVICE_CPU);
    if (!stack) {
        boat_free(visited);
        return;
    }
    size_t stack_top = 0;
    stack[stack_top++] = start_index;

    while (stack_top > 0) {
        size_t current_index = stack[--stack_top];
        if (visited[current_index]) {
            if (post_visit) {
                post_visit(graph->nodes[current_index], user_data);
            }
            continue;
        }

        visited[current_index] = true;
        if (pre_visit) {
            pre_visit(graph->nodes[current_index], user_data);
        }

        // Push node again for post-visit
        stack[stack_top++] = current_index;

        // Push all adjacent nodes (outgoing edges)
        if (graph->outgoing[current_index]) {
            size_t edge_count = boat_edge_list_count(graph->outgoing[current_index]);
            for (size_t i = 0; i < edge_count; i++) {
                const struct boat_edge_t* edge = boat_edge_list_get(graph->outgoing[current_index], i);
                const boat_node_t* neighbor = boat_edge_target(edge);
                // Find neighbor index
                for (size_t j = 0; j < graph->node_count; j++) {
                    if (graph->nodes[j] == neighbor && !visited[j]) {
                        stack[stack_top++] = j;
                        break;
                    }
                }
            }
        }
    }

    boat_free(stack);
    boat_free(visited);
}

void boat_graph_bfs(const boat_graph_t* graph, const boat_node_t* start,
                    boat_node_visitor_t visit, void* user_data) {
    if (!graph || !start || !visit) return;

    size_t start_index = SIZE_MAX;
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == start) {
            start_index = i;
            break;
        }
    }
    if (start_index == SIZE_MAX) return;

    bool* visited = boat_calloc(graph->node_count * sizeof(bool), BOAT_DEVICE_CPU);
    if (!visited) return;

    size_t* queue = boat_malloc(graph->node_count * sizeof(size_t), BOAT_DEVICE_CPU);
    if (!queue) {
        boat_free(visited);
        return;
    }
    size_t front = 0, rear = 0;

    queue[rear++] = start_index;
    visited[start_index] = true;

    while (front < rear) {
        size_t current_index = queue[front++];
        visit(graph->nodes[current_index], user_data);

        if (graph->outgoing[current_index]) {
            size_t edge_count = boat_edge_list_count(graph->outgoing[current_index]);
            for (size_t i = 0; i < edge_count; i++) {
                const struct boat_edge_t* edge = boat_edge_list_get(graph->outgoing[current_index], i);
                const boat_node_t* neighbor = boat_edge_target(edge);
                for (size_t j = 0; j < graph->node_count; j++) {
                    if (graph->nodes[j] == neighbor && !visited[j]) {
                        visited[j] = true;
                        queue[rear++] = j;
                        break;
                    }
                }
            }
        }
    }

    boat_free(queue);
    boat_free(visited);
}

void boat_graph_topological_sort(const boat_graph_t* graph, boat_node_t** sorted_nodes,
                                 size_t* count) {
    if (!graph || !sorted_nodes || !count) return;

    // Kahn's algorithm
    size_t n = graph->node_count;
    size_t* in_degree = boat_calloc(n * sizeof(size_t), BOAT_DEVICE_CPU);
    if (!in_degree) return;

    // Compute in-degrees
    for (size_t i = 0; i < n; i++) {
        if (graph->incoming[i]) {
            in_degree[i] = boat_edge_list_count(graph->incoming[i]);
        }
    }

    size_t* queue = boat_malloc(n * sizeof(size_t), BOAT_DEVICE_CPU);
    if (!queue) {
        boat_free(in_degree);
        return;
    }
    size_t front = 0, rear = 0;

    // Enqueue nodes with in-degree 0
    for (size_t i = 0; i < n; i++) {
        if (in_degree[i] == 0) {
            queue[rear++] = i;
        }
    }

    size_t sorted_count = 0;
    while (front < rear) {
        size_t current = queue[front++];
        sorted_nodes[sorted_count++] = graph->nodes[current];

        if (graph->outgoing[current]) {
            size_t edge_count = boat_edge_list_count(graph->outgoing[current]);
            for (size_t i = 0; i < edge_count; i++) {
                const struct boat_edge_t* edge = boat_edge_list_get(graph->outgoing[current], i);
                const boat_node_t* neighbor = boat_edge_target(edge);
                // Find neighbor index
                for (size_t j = 0; j < n; j++) {
                    if (graph->nodes[j] == neighbor) {
                        if (--in_degree[j] == 0) {
                            queue[rear++] = j;
                        }
                        break;
                    }
                }
            }
        }
    }

    *count = sorted_count;

    boat_free(queue);
    boat_free(in_degree);
}

// Graph properties
static bool dfs_cycle(const boat_graph_t* graph, bool* visited, bool* rec_stack, size_t n, size_t v) {
    if (!visited[v]) {
        visited[v] = true;
        rec_stack[v] = true;

        if (graph->outgoing[v]) {
            size_t edge_count = boat_edge_list_count(graph->outgoing[v]);
            for (size_t i = 0; i < edge_count; i++) {
                const struct boat_edge_t* edge = boat_edge_list_get(graph->outgoing[v], i);
                const boat_node_t* neighbor = boat_edge_target(edge);
                size_t neighbor_idx = SIZE_MAX;
                for (size_t j = 0; j < n; j++) {
                    if (graph->nodes[j] == neighbor) {
                        neighbor_idx = j;
                        break;
                    }
                }
                if (neighbor_idx != SIZE_MAX) {
                    if (!visited[neighbor_idx] && dfs_cycle(graph, visited, rec_stack, n, neighbor_idx)) {
                        return true;
                    } else if (rec_stack[neighbor_idx]) {
                        return true;
                    }
                }
            }
        }
    }
    rec_stack[v] = false;
    return false;
}

bool boat_graph_is_acyclic(const boat_graph_t* graph) {
    if (!graph) return true;

    size_t n = graph->node_count;
    bool* visited = boat_calloc(n * sizeof(bool), BOAT_DEVICE_CPU);
    bool* rec_stack = boat_calloc(n * sizeof(bool), BOAT_DEVICE_CPU);
    if (!visited || !rec_stack) {
        boat_free(visited);
        boat_free(rec_stack);
        return true;
    }


    bool has_cycle = false;
    for (size_t i = 0; i < n; i++) {
        if (!visited[i]) {
            if (dfs_cycle(graph, visited, rec_stack, n, i)) {
                has_cycle = true;
                break;
            }
        }
    }

    boat_free(visited);
    boat_free(rec_stack);
    return !has_cycle;
}

bool boat_graph_is_connected(const boat_graph_t* graph) {
    if (!graph || graph->node_count == 0) return true;

    // Use BFS from first node
    bool* visited = boat_calloc(graph->node_count * sizeof(bool), BOAT_DEVICE_CPU);
    if (!visited) return false;

    size_t* queue = boat_malloc(graph->node_count * sizeof(size_t), BOAT_DEVICE_CPU);
    if (!queue) {
        boat_free(visited);
        return false;
    }
    size_t front = 0, rear = 0;

    queue[rear++] = 0;
    visited[0] = true;
    size_t visited_count = 1;

    while (front < rear) {
        size_t current = queue[front++];
        if (graph->outgoing[current]) {
            size_t edge_count = boat_edge_list_count(graph->outgoing[current]);
            for (size_t i = 0; i < edge_count; i++) {
                const struct boat_edge_t* edge = boat_edge_list_get(graph->outgoing[current], i);
                const boat_node_t* neighbor = boat_edge_target(edge);
                for (size_t j = 0; j < graph->node_count; j++) {
                    if (graph->nodes[j] == neighbor && !visited[j]) {
                        visited[j] = true;
                        visited_count++;
                        queue[rear++] = j;
                        break;
                    }
                }
            }
        }
        if (graph->incoming[current]) {
            size_t edge_count = boat_edge_list_count(graph->incoming[current]);
            for (size_t i = 0; i < edge_count; i++) {
                const struct boat_edge_t* edge = boat_edge_list_get(graph->incoming[current], i);
                const boat_node_t* neighbor = boat_edge_source(edge);
                for (size_t j = 0; j < graph->node_count; j++) {
                    if (graph->nodes[j] == neighbor && !visited[j]) {
                        visited[j] = true;
                        visited_count++;
                        queue[rear++] = j;
                        break;
                    }
                }
            }
        }
    }

    boat_free(queue);
    bool connected = (visited_count == graph->node_count);
    boat_free(visited);
    return connected;
}

bool boat_graph_has_path(const boat_graph_t* graph, const boat_node_t* from, const boat_node_t* to) {
    if (!graph || !from || !to) return false;

    size_t from_index = SIZE_MAX, to_index = SIZE_MAX;
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == from) from_index = i;
        if (graph->nodes[i] == to) to_index = i;
        if (from_index != SIZE_MAX && to_index != SIZE_MAX) break;
    }
    if (from_index == SIZE_MAX || to_index == SIZE_MAX) return false;

    // Simple BFS
    bool* visited = boat_calloc(graph->node_count * sizeof(bool), BOAT_DEVICE_CPU);
    if (!visited) return false;

    size_t* queue = boat_malloc(graph->node_count * sizeof(size_t), BOAT_DEVICE_CPU);
    if (!queue) {
        boat_free(visited);
        return false;
    }
    size_t front = 0, rear = 0;

    queue[rear++] = from_index;
    visited[from_index] = true;

    while (front < rear) {
        size_t current = queue[front++];
        if (current == to_index) {
            boat_free(queue);
            boat_free(visited);
            return true;
        }

        if (graph->outgoing[current]) {
            size_t edge_count = boat_edge_list_count(graph->outgoing[current]);
            for (size_t i = 0; i < edge_count; i++) {
                const struct boat_edge_t* edge = boat_edge_list_get(graph->outgoing[current], i);
                const boat_node_t* neighbor = boat_edge_target(edge);
                for (size_t j = 0; j < graph->node_count; j++) {
                    if (graph->nodes[j] == neighbor && !visited[j]) {
                        visited[j] = true;
                        queue[rear++] = j;
                        break;
                    }
                }
            }
        }
    }

    boat_free(queue);
    boat_free(visited);
    return false;
}

// Graph copy operations
boat_graph_t* boat_graph_copy(const boat_graph_t* graph) {
    if (!graph) return NULL;

    // Create new graph with same device
    boat_graph_t* copy = boat_graph_create_with_device(graph->device);
    if (!copy) return NULL;

    // Copy basic properties
    copy->next_node_id = graph->next_node_id;
    copy->checkpointing_enabled = graph->checkpointing_enabled;

    // Ensure capacity matches source graph
    if (!ensure_node_capacity(copy, graph->node_capacity)) {
        boat_graph_free(copy);
        return NULL;
    }
    if (!ensure_edge_capacity(copy, graph->edge_capacity)) {
        boat_graph_free(copy);
        return NULL;
    }

    // Create mapping from original nodes to copied nodes
    boat_node_t** node_map = boat_calloc(graph->node_count * sizeof(boat_node_t*), BOAT_DEVICE_CPU);
    if (!node_map) {
        boat_graph_free(copy);
        return NULL;
    }

    // Copy all nodes
    for (size_t i = 0; i < graph->node_count; i++) {
        const boat_node_t* orig_node = graph->nodes[i];
        if (!orig_node) continue;

        // Create a new node in the copy graph with the same data
        // This performs a shallow copy of the data (same pointer)
        // Note: We pass NULL as free_fn because the original graph owns the data
        boat_node_t* copy_node = boat_graph_add_node(copy, boat_node_data(orig_node),
                                                     boat_node_type(orig_node), NULL);
        if (!copy_node) {
            // Cleanup
            for (size_t j = 0; j < i; j++) {
                if (node_map[j]) {
                    // Remove node from graph (graph owns it, so just clear mapping)
                }
            }
            boat_free(node_map);
            boat_graph_free(copy);
            return NULL;
        }

        // Note: boat_graph_add_node assigns a new ID from copy->next_node_id
        // We don't preserve the original ID since edges reference nodes by pointer, not ID
        node_map[i] = copy_node;
    }

    // Copy all edges
    for (size_t i = 0; i < graph->edge_count; i++) {
        const struct boat_edge_t* orig_edge = graph->edges[i];
        if (!orig_edge) continue;

        // Find source and target indices in original graph
        size_t from_idx = SIZE_MAX, to_idx = SIZE_MAX;
        const boat_node_t* edge_from = boat_edge_source(orig_edge);
        const boat_node_t* edge_to = boat_edge_target(orig_edge);
        for (size_t j = 0; j < graph->node_count; j++) {
            if (graph->nodes[j] == edge_from) from_idx = j;
            if (graph->nodes[j] == edge_to) to_idx = j;
            if (from_idx != SIZE_MAX && to_idx != SIZE_MAX) break;
        }

        if (from_idx == SIZE_MAX || to_idx == SIZE_MAX) continue;
        if (!node_map[from_idx] || !node_map[to_idx]) continue;

        // Create new edge between copied nodes
        struct boat_edge_t* copy_edge = boat_edge_create(
            node_map[from_idx],
            node_map[to_idx],
            boat_edge_direction(orig_edge)
        );
        if (!copy_edge) {
            // Cleanup
            boat_free(node_map);
            boat_graph_free(copy);
            return NULL;
        }

        // Add edge to copy graph
        if (!ensure_edge_capacity(copy, copy->edge_count + 1)) {
            boat_edge_free(copy_edge);
            boat_free(node_map);
            boat_graph_free(copy);
            return NULL;
        }
        copy->edges[copy->edge_count++] = copy_edge;

        // Add to adjacency lists
        if (!copy->outgoing[from_idx]) {
            copy->outgoing[from_idx] = boat_edge_list_create();
            if (!copy->outgoing[from_idx]) {
                boat_edge_free(copy_edge);
                boat_free(node_map);
                boat_graph_free(copy);
                return NULL;
            }
        }
        if (!copy->incoming[to_idx]) {
            copy->incoming[to_idx] = boat_edge_list_create();
            if (!copy->incoming[to_idx]) {
                boat_edge_free(copy_edge);
                boat_free(node_map);
                boat_graph_free(copy);
                return NULL;
            }
        }

        if (!boat_edge_list_add(copy->outgoing[from_idx], copy_edge) ||
            !boat_edge_list_add(copy->incoming[to_idx], copy_edge)) {
            boat_edge_free(copy_edge);
            boat_free(node_map);
            boat_graph_free(copy);
            return NULL;
        }
    }

    // Copy checkpoint nodes array if checkpointing is enabled
    if (graph->checkpointing_enabled && graph->checkpoint_nodes) {
        copy->checkpoint_nodes = boat_calloc(copy->node_capacity * sizeof(bool), copy->device);
        if (!copy->checkpoint_nodes) {
            boat_free(node_map);
            boat_graph_free(copy);
            return NULL;
        }
        // Copy checkpoint markers
        for (size_t i = 0; i < graph->node_count; i++) {
            copy->checkpoint_nodes[i] = graph->checkpoint_nodes[i];
        }
    }

    boat_free(node_map);
    return copy;
}

// Subgraph operations
boat_graph_t* boat_graph_subgraph(const boat_graph_t* graph, boat_node_t** nodes,
                                  size_t node_count) {
    if (!graph || !nodes || node_count == 0) return NULL;

    // Create new graph for subgraph with same device
    boat_graph_t* subgraph = boat_graph_create_with_device(graph->device);
    if (!subgraph) return NULL;

    // Map from original node to index in nodes array
    bool* node_in_subgraph = boat_calloc(graph->node_count * sizeof(bool), BOAT_DEVICE_CPU);
    if (!node_in_subgraph) {
        boat_graph_free(subgraph);
        return NULL;
    }

    // Map from original node index to subgraph node
    boat_node_t** node_map = boat_calloc(graph->node_count * sizeof(boat_node_t*), BOAT_DEVICE_CPU);
    if (!node_map) {
        boat_free(node_in_subgraph);
        boat_graph_free(subgraph);
        return NULL;
    }

    // First pass: identify which nodes are in the subgraph
    for (size_t i = 0; i < node_count; i++) {
        const boat_node_t* node = nodes[i];
        if (!node) continue;

        // Find this node in the original graph
        size_t node_idx = SIZE_MAX;
        for (size_t j = 0; j < graph->node_count; j++) {
            if (graph->nodes[j] == node) {
                node_idx = j;
                break;
            }
        }
        if (node_idx == SIZE_MAX) continue; // Node not in original graph

        node_in_subgraph[node_idx] = true;
    }

    // Second pass: create nodes in subgraph
    for (size_t i = 0; i < graph->node_count; i++) {
        if (!node_in_subgraph[i]) continue;

        const boat_node_t* orig_node = graph->nodes[i];
        // Create a new node in subgraph with the same data (shallow copy)
        // Note: NULL free_fn because original graph owns the data
        boat_node_t* copy_node = boat_graph_add_node(subgraph, boat_node_data(orig_node),
                                                     boat_node_type(orig_node), NULL);
        if (!copy_node) {
            // Cleanup
            for (size_t j = 0; j < i; j++) {
                if (node_map[j]) {
                    // Node is owned by subgraph, will be freed when subgraph is freed
                }
            }
            boat_free(node_map);
            boat_free(node_in_subgraph);
            boat_graph_free(subgraph);
            return NULL;
        }
        node_map[i] = copy_node;
    }

    // Third pass: copy edges between nodes in subgraph
    for (size_t i = 0; i < graph->edge_count; i++) {
        const struct boat_edge_t* orig_edge = graph->edges[i];
        if (!orig_edge) continue;

        // Find source and target indices
        size_t from_idx = SIZE_MAX, to_idx = SIZE_MAX;
        const boat_node_t* edge_from = boat_edge_source(orig_edge);
        const boat_node_t* edge_to = boat_edge_target(orig_edge);
        for (size_t j = 0; j < graph->node_count; j++) {
            if (graph->nodes[j] == edge_from) from_idx = j;
            if (graph->nodes[j] == edge_to) to_idx = j;
            if (from_idx != SIZE_MAX && to_idx != SIZE_MAX) break;
        }

        // Check if both endpoints are in subgraph
        if (from_idx == SIZE_MAX || to_idx == SIZE_MAX) continue;
        if (!node_in_subgraph[from_idx] || !node_in_subgraph[to_idx]) continue;
        if (!node_map[from_idx] || !node_map[to_idx]) continue;

        // Create edge in subgraph
        struct boat_edge_t* copy_edge = boat_edge_create(
            node_map[from_idx],
            node_map[to_idx],
            boat_edge_direction(orig_edge)
        );
        if (!copy_edge) {
            // Cleanup
            boat_free(node_map);
            boat_free(node_in_subgraph);
            boat_graph_free(subgraph);
            return NULL;
        }

        // Add edge to subgraph
        if (!ensure_edge_capacity(subgraph, subgraph->edge_count + 1)) {
            boat_edge_free(copy_edge);
            for (size_t j = 0; j < graph->node_count; j++) {
                // Node is owned by subgraph, will be freed with boat_graph_free
            }
            boat_free(node_map);
            boat_free(node_in_subgraph);
            boat_graph_free(subgraph);
            return NULL;
        }
        subgraph->edges[subgraph->edge_count++] = copy_edge;

        // Ensure adjacency lists exist
        // Find indices in subgraph arrays (not same as original indices)
        size_t sub_from_idx = SIZE_MAX, sub_to_idx = SIZE_MAX;
        for (size_t j = 0; j < subgraph->node_count; j++) {
            if (subgraph->nodes[j] == node_map[from_idx]) sub_from_idx = j;
            if (subgraph->nodes[j] == node_map[to_idx]) sub_to_idx = j;
            if (sub_from_idx != SIZE_MAX && sub_to_idx != SIZE_MAX) break;
        }

        if (sub_from_idx == SIZE_MAX || sub_to_idx == SIZE_MAX) {
            boat_edge_free(copy_edge);
            continue;
        }

        if (!subgraph->outgoing[sub_from_idx]) {
            subgraph->outgoing[sub_from_idx] = boat_edge_list_create();
            if (!subgraph->outgoing[sub_from_idx]) {
                boat_edge_free(copy_edge);
                for (size_t j = 0; j < graph->node_count; j++) {
                    // Node is owned by subgraph, will be freed with boat_graph_free
                }
                boat_free(node_map);
                boat_free(node_in_subgraph);
                boat_graph_free(subgraph);
                return NULL;
            }
        }
        if (!subgraph->incoming[sub_to_idx]) {
            subgraph->incoming[sub_to_idx] = boat_edge_list_create();
            if (!subgraph->incoming[sub_to_idx]) {
                boat_edge_free(copy_edge);
                for (size_t j = 0; j < graph->node_count; j++) {
                    // Node is owned by subgraph, will be freed with boat_graph_free
                }
                boat_free(node_map);
                boat_free(node_in_subgraph);
                boat_graph_free(subgraph);
                return NULL;
            }
        }

        // Add to adjacency lists
        if (!boat_edge_list_add(subgraph->outgoing[sub_from_idx], copy_edge) ||
            !boat_edge_list_add(subgraph->incoming[sub_to_idx], copy_edge)) {
            boat_edge_free(copy_edge);
            for (size_t j = 0; j < graph->node_count; j++) {
                // Node is owned by subgraph, will be freed with boat_graph_free
            }
            boat_free(node_map);
            boat_free(node_in_subgraph);
            boat_graph_free(subgraph);
            return NULL;
        }
    }

    // Copy checkpointing state if applicable
    subgraph->checkpointing_enabled = graph->checkpointing_enabled;
    if (graph->checkpointing_enabled && graph->checkpoint_nodes) {
        // We would need to map checkpoint nodes, but for simplicity,
        // subgraph checkpointing starts fresh
        // Could implement mapping if needed
    }

    boat_free(node_map);
    boat_free(node_in_subgraph);
    return subgraph;
}

void boat_graph_merge(boat_graph_t* dest, const boat_graph_t* src) {
    if (!dest || !src) return;

    // We need to merge nodes and edges from src into dest
    // This is a complex operation that needs to handle:
    // 1. Node ID conflicts (rename nodes if needed)
    // 2. Edge conflicts (skip or merge)
    // 3. Data ownership (transfer or copy)

    // For Phase 1, implement a basic merge that copies all nodes and edges
    // This will create duplicates if nodes/edges already exist

    // Create a mapping from src node indices to dest node indices
    size_t* node_index_map = boat_calloc(src->node_count, sizeof(size_t));
    if (!node_index_map) return;

    // First, copy all nodes from src to dest
    for (size_t i = 0; i < src->node_count; i++) {
        const boat_node_t* src_node = src->nodes[i];
        if (!src_node) continue;

        // Check if a node with same ID already exists in dest
        bool node_exists = false;
        for (size_t j = 0; j < dest->node_count; j++) {
            if (dest->nodes[j] && boat_graph_node_id(dest->nodes[j]) == boat_graph_node_id(src_node)) {
                node_exists = true;
                node_index_map[i] = j;
                break;
            }
        }

        if (node_exists) {
            // Node already exists, skip copying
            continue;
        }

        // Create a copy of the node in dest graph (shallow copy)
        const boat_node_t* copy_node = boat_graph_add_node(dest, boat_node_data(src_node),
                                                     boat_node_type(src_node), NULL);
        if (!copy_node) {
            // Cleanup: nodes added to dest are owned by dest and will be freed with it
            boat_free(node_index_map);
            return;
        }

        // Find the index of the newly added node in dest
        size_t new_index = SIZE_MAX;
        for (size_t j = 0; j < dest->node_count; j++) {
            if (dest->nodes[j] == copy_node) {
                new_index = j;
                break;
            }
        }
        if (new_index == SIZE_MAX) {
            // Should not happen, but handle gracefully
            boat_free(node_index_map);
            return;
        }
        node_index_map[i] = new_index;
    }

    // Next, copy edges from src to dest
    for (size_t i = 0; i < src->edge_count; i++) {
        const struct boat_edge_t* src_edge = src->edges[i];
        if (!src_edge) continue;

        // Find source and target indices in src
        size_t src_from_idx = SIZE_MAX, src_to_idx = SIZE_MAX;
        const boat_node_t* edge_from = boat_edge_source(src_edge);
        const boat_node_t* edge_to = boat_edge_target(src_edge);
        for (size_t j = 0; j < src->node_count; j++) {
            if (src->nodes[j] == edge_from) src_from_idx = j;
            if (src->nodes[j] == edge_to) src_to_idx = j;
            if (src_from_idx != SIZE_MAX && src_to_idx != SIZE_MAX) break;
        }

        if (src_from_idx == SIZE_MAX || src_to_idx == SIZE_MAX) continue;

        // Map to dest indices
        size_t dest_from_idx = node_index_map[src_from_idx];
        size_t dest_to_idx = node_index_map[src_to_idx];
        if (dest_from_idx == SIZE_MAX || dest_to_idx == SIZE_MAX) continue;

        const boat_node_t* dest_from = dest->nodes[dest_from_idx];
        const boat_node_t* dest_to = dest->nodes[dest_to_idx];

        // Check if edge already exists in dest
        bool edge_exists = false;
        if (dest->outgoing[dest_from_idx]) {
            size_t edge_count = boat_edge_list_count(dest->outgoing[dest_from_idx]);
            for (size_t j = 0; j < edge_count; j++) {
                const struct boat_edge_t* existing_edge = boat_edge_list_get(dest->outgoing[dest_from_idx], j);
                if (existing_edge &&
                    boat_edge_source(existing_edge) == dest_from &&
                    boat_edge_target(existing_edge) == dest_to &&
                    boat_edge_direction(existing_edge) == boat_edge_direction(src_edge)) {
                    edge_exists = true;
                    break;
                }
            }
        }

        if (edge_exists) continue;

        // Create new edge
        struct boat_edge_t* new_edge = boat_edge_create(dest_from, dest_to, boat_edge_direction(src_edge));
        if (!new_edge) {
            // Continue with other edges
            continue;
        }

        // Add to dest
        if (!ensure_edge_capacity(dest, dest->edge_count + 1)) {
            boat_edge_free(new_edge);
            continue;
        }
        dest->edges[dest->edge_count++] = new_edge;

        // Ensure adjacency lists exist
        if (!dest->outgoing[dest_from_idx]) {
            dest->outgoing[dest_from_idx] = boat_edge_list_create();
        }
        if (!dest->incoming[dest_to_idx]) {
            dest->incoming[dest_to_idx] = boat_edge_list_create();
        }

        if (dest->outgoing[dest_from_idx] && dest->incoming[dest_to_idx]) {
            boat_edge_list_add(dest->outgoing[dest_from_idx], new_edge);
            boat_edge_list_add(dest->incoming[dest_to_idx], new_edge);
        }
    }

    // Merge checkpointing state
    if (src->checkpointing_enabled) {
        dest->checkpointing_enabled = true;
        // Would need to merge checkpoint_nodes arrays
    }

    boat_free(node_index_map);
}

// Graph validation
void boat_graph_validate(const boat_graph_t* graph) {
    if (!graph) return;

    // Basic sanity checks
    if (graph->node_count > graph->node_capacity) {
        fprintf(stderr, "Graph validation error: node_count > node_capacity\n");
    }
    if (graph->edge_count > graph->edge_capacity) {
        fprintf(stderr, "Graph validation error: edge_count > edge_capacity\n");
    }

    // Check nodes array
    for (size_t i = 0; i < graph->node_count; i++) {
        if (!graph->nodes[i]) {
            fprintf(stderr, "Graph validation error: node %zu is NULL\n", i);
        }
    }

    // Check edges array
    for (size_t i = 0; i < graph->edge_count; i++) {
        const struct boat_edge_t* edge = graph->edges[i];
        if (!edge) {
            fprintf(stderr, "Graph validation error: edge %zu is NULL\n", i);
            continue;
        }

        const boat_node_t* from = boat_edge_source(edge);
        const boat_node_t* to = boat_edge_target(edge);
        if (!from || !to) {
            fprintf(stderr, "Graph validation error: edge %zu has NULL source or target\n", i);
        }

        // Verify edge is properly registered in adjacency lists
        bool found_in_outgoing = false;
        bool found_in_incoming = false;

        // Find source node index
        size_t from_idx = SIZE_MAX;
        for (size_t j = 0; j < graph->node_count; j++) {
            if (graph->nodes[j] == from) {
                from_idx = j;
                break;
            }
        }

        if (from_idx != SIZE_MAX && graph->outgoing[from_idx]) {
            found_in_outgoing = boat_edge_list_contains(graph->outgoing[from_idx], edge);
        }

        // Find target node index
        size_t to_idx = SIZE_MAX;
        for (size_t j = 0; j < graph->node_count; j++) {
            if (graph->nodes[j] == to) {
                to_idx = j;
                break;
            }
        }

        if (to_idx != SIZE_MAX && graph->incoming[to_idx]) {
            found_in_incoming = boat_edge_list_contains(graph->incoming[to_idx], edge);
        }

        if (!found_in_outgoing) {
            fprintf(stderr, "Graph validation error: edge %zu not found in outgoing list of source node\n", i);
        }
        if (!found_in_incoming) {
            fprintf(stderr, "Graph validation error: edge %zu not found in incoming list of target node\n", i);
        }
    }

    // Check adjacency lists consistency
    for (size_t i = 0; i < graph->node_capacity; i++) {
        if (i < graph->node_count) {
            // Node exists, adjacency lists may exist
            if (graph->outgoing[i]) {
                size_t edge_count = boat_edge_list_count(graph->outgoing[i]);
                for (size_t j = 0; j < edge_count; j++) {
                    const struct boat_edge_t* edge = boat_edge_list_get(graph->outgoing[i], j);
                    if (!edge) {
                        fprintf(stderr, "Graph validation error: NULL edge in outgoing list of node %zu\n", i);
                        continue;
                    }
                    // Verify edge is in edges array
                    bool found_in_edges = false;
                    for (size_t k = 0; k < graph->edge_count; k++) {
                        if (graph->edges[k] == edge) {
                            found_in_edges = true;
                            break;
                        }
                    }
                    if (!found_in_edges) {
                        fprintf(stderr, "Graph validation error: edge in outgoing list of node %zu not in edges array\n", i);
                    }
                }
            }
            if (graph->incoming[i]) {
                size_t edge_count = boat_edge_list_count(graph->incoming[i]);
                for (size_t j = 0; j < edge_count; j++) {
                    const struct boat_edge_t* edge = boat_edge_list_get(graph->incoming[i], j);
                    if (!edge) {
                        fprintf(stderr, "Graph validation error: NULL edge in incoming list of node %zu\n", i);
                        continue;
                    }
                    // Verify edge is in edges array
                    bool found_in_edges = false;
                    for (size_t k = 0; k < graph->edge_count; k++) {
                        if (graph->edges[k] == edge) {
                            found_in_edges = true;
                            break;
                        }
                    }
                    if (!found_in_edges) {
                        fprintf(stderr, "Graph validation error: edge in incoming list of node %zu not in edges array\n", i);
                    }
                }
            }
        } else {
            // Beyond node_count, adjacency lists should be NULL
            if (graph->outgoing[i] || graph->incoming[i]) {
                fprintf(stderr, "Graph validation error: adjacency lists exist for index %zu beyond node_count\n", i);
            }
        }
    }

    // Check checkpoint nodes array if checkpointing is enabled
    if (graph->checkpointing_enabled) {
        if (!graph->checkpoint_nodes) {
            fprintf(stderr, "Graph validation error: checkpointing enabled but checkpoint_nodes is NULL\n");
        } else {
            // Checkpoint nodes array should be at least node_capacity in size
            // (allocated with node_capacity)
        }
    }
}

bool boat_graph_can_add_edge(const boat_graph_t* graph, const boat_node_t* from, const boat_node_t* to) {
    if (!graph || !from || !to) return false;

    // Check if both nodes are in the graph
    bool from_in_graph = false;
    bool to_in_graph = false;
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == from) from_in_graph = true;
        if (graph->nodes[i] == to) to_in_graph = true;
        if (from_in_graph && to_in_graph) break;
    }

    if (!from_in_graph || !to_in_graph) return false;

    // Check if edge would create a cycle (for forward edges only)
    // For computational graphs, we typically want acyclic graphs
    // We can skip this check for backward edges or allow cycles in some cases
    // For now, just check if edge already exists
    // Find from index
    size_t from_idx = SIZE_MAX;
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == from) {
            from_idx = i;
            break;
        }
    }

    if (from_idx == SIZE_MAX) return false;

    if (graph->outgoing[from_idx]) {
        size_t edge_count = boat_edge_list_count(graph->outgoing[from_idx]);
        for (size_t i = 0; i < edge_count; i++) {
            const struct boat_edge_t* edge = boat_edge_list_get(graph->outgoing[from_idx], i);
            if (boat_edge_target(edge) == to) {
                // Edge already exists
                return false;
            }
        }
    }

    // Additional checks could be added:
    // - Check for self-loops (from == to)
    // - Check for duplicate edges with same direction
    // - Check if graph would become cyclic (for forward edges)

    return true;
}

bool boat_graph_can_remove_node(const boat_graph_t* graph, const boat_node_t* node) {
    if (!graph || !node) return false;

    // Check if node is in graph
    bool node_in_graph = false;
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == node) {
            node_in_graph = true;
            break;
        }
    }
    if (!node_in_graph) return false;

    // Check if node has any incoming or outgoing edges
    // In computational graphs, we may want to prevent removal of nodes with edges
    // Or automatically remove edges first
    // For now, allow removal only if node has no edges
    for (size_t i = 0; i < graph->edge_count; i++) {
        const struct boat_edge_t* edge = graph->edges[i];
        if (!edge) continue;
        if (boat_edge_source(edge) == node || boat_edge_target(edge) == node) {
            // Node has at least one edge
            return false;
        }
    }

    return true;
}

// Gradient checkpointing
void boat_graph_enable_checkpointing(boat_graph_t* graph, bool enabled) {
    if (!graph) return;

    graph->checkpointing_enabled = enabled;

    if (enabled) {
        // Allocate checkpoint nodes array if not already allocated
        if (!graph->checkpoint_nodes) {
            graph->checkpoint_nodes = boat_calloc(graph->node_capacity * sizeof(bool), graph->device);
        }
    } else {
        // Free checkpoint nodes array if allocated
        if (graph->checkpoint_nodes) {
            boat_free(graph->checkpoint_nodes);
            graph->checkpoint_nodes = NULL;
        }
    }
}

bool boat_graph_checkpointing_enabled(const boat_graph_t* graph) {
    return graph ? graph->checkpointing_enabled : false;
}

void boat_graph_mark_checkpoint(const boat_graph_t* graph, const boat_node_t* node) {
    if (!graph || !node || !graph->checkpoint_nodes) return;

    // Find node index
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == node) {
            graph->checkpoint_nodes[i] = true;
            break;
        }
    }
}

bool boat_graph_is_checkpoint(const boat_graph_t* graph, const boat_node_t* node) {
    if (!graph || !node || !graph->checkpoint_nodes) return false;

    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == node) {
            return graph->checkpoint_nodes[i];
        }
    }
    return false;
}

// Graph visualization
void boat_graph_print(const boat_graph_t* graph) {
    if (!graph) return;

    printf("Graph with %zu nodes and %zu edges:\n",
           graph->node_count, graph->edge_count);
    for (size_t i = 0; i < graph->node_count; i++) {
        printf("  Node %zu (id=%zu): ", i, boat_graph_node_id(graph->nodes[i]));
        if (graph->outgoing[i]) {
            size_t edge_count = boat_edge_list_count(graph->outgoing[i]);
            if (edge_count > 0) {
                printf("-> {");
                for (size_t j = 0; j < edge_count; j++) {
                    const struct boat_edge_t* edge = boat_edge_list_get(graph->outgoing[i], j);
                    printf(" %zu", boat_graph_node_id(boat_edge_target(edge)));
                    if (j < edge_count - 1) printf(",");
                }
                printf(" }");
            }
        }
        printf("\n");
    }
}

boat_node_t* boat_graph_get_node_at_index(const boat_graph_t* graph, size_t index) {
    if (!graph || index >= graph->node_count) {
        return NULL;
    }
    return graph->nodes[index];
}

boat_edge_t* boat_graph_get_edge_at_index(const boat_graph_t* graph, size_t index) {
    if (!graph || index >= graph->edge_count) {
        return NULL;
    }
    return graph->edges[index];
}

char* boat_graph_to_dot(const boat_graph_t* graph) {
    if (!graph) return NULL;

    // Calculate required buffer size
    size_t buffer_size = 512; // Base size for graph header/footer
    buffer_size += graph->node_count * 128; // Per node
    buffer_size += graph->edge_count * 128; // Per edge

    char* dot = boat_malloc(buffer_size, BOAT_DEVICE_CPU);
    if (!dot) return NULL;

    // Start DOT graph
    size_t pos = snprintf(dot, buffer_size,
        "digraph computation_graph {\n"
        "  rankdir=TB;\n"
        "  node [shape=record, fontname=\"Courier\", fontsize=10];\n"
        "  edge [fontname=\"Courier\", fontsize=8];\n\n");

    // Add nodes
    for (size_t i = 0; i < graph->node_count; i++) {
        const boat_node_t* node = graph->nodes[i];
        if (!node) continue;

        size_t node_id = boat_graph_node_id(node);
        boat_node_type_t type = boat_node_type(node);
        const char* type_name = boat_node_type_name(type);

        // Create node label
        pos += snprintf(dot + pos, buffer_size - pos,
            "  node%zu [label=\"{%s|ID: %zu}\"];\n",
            node_id, type_name, node_id);
    }

    // Add edges
    for (size_t i = 0; i < graph->edge_count; i++) {
        const struct boat_edge_t* edge = graph->edges[i];
        if (!edge) continue;

        const boat_node_t* from = boat_edge_source(edge);
        const boat_node_t* to = boat_edge_target(edge);
        boat_edge_direction_t direction = boat_edge_direction(edge);

        if (!from || !to) continue;

        size_t from_id = boat_graph_node_id(from);
        size_t to_id = boat_graph_node_id(to);

        const char* dir_label = (direction == BOAT_EDGE_DIRECTION_FORWARD) ? "forward" : "backward";
        const char* dir_color = (direction == BOAT_EDGE_DIRECTION_FORWARD) ? "blue" : "red";

        pos += snprintf(dot + pos, buffer_size - pos,
            "  node%zu -> node%zu [label=\"%s\", color=\"%s\", style=\"solid\"];\n",
            from_id, to_id, dir_label, dir_color);
    }

    // Close graph
    pos += snprintf(dot + pos, buffer_size - pos, "}\n");

    (void)pos; // Suppress unused variable warning

    return dot;
}

// Real-time graph modification during training
boat_edge_t* boat_graph_safe_add_edge(const boat_graph_t* graph, const boat_node_t* from, const boat_node_t* to,
                                     boat_edge_direction_t direction) {
    if (!graph || !from || !to) return NULL;

    // Validate the edge addition
    if (!boat_graph_can_add_edge(graph, from, to)) {
        return NULL;
    }

    // For forward edges, check if adding would create a cycle
    if (direction == BOAT_EDGE_DIRECTION_FORWARD) {
        // Check if there's already a path from 'to' to 'from'
        // If yes, adding edge from->to would create a cycle
        if (boat_graph_has_path(graph, to, from)) {
            // Would create a cycle
            return NULL;
        }
    }

    // Add the edge
    return boat_graph_add_edge(graph, from, to, direction);
}

bool boat_graph_safe_remove_node(const boat_graph_t* graph, const boat_node_t* node) {
    if (!graph || !node) return false;

    // Check if node can be removed (no edges)
    if (!boat_graph_can_remove_node(graph, node)) {
        // Automatically remove all edges connected to this node
        remove_all_edges_for_node(graph, node);

        // Now check again if node can be removed
        // (should be true since edges are removed)
        if (!boat_graph_can_remove_node(graph, node)) {
            return false;
        }
    }

    // Remove the node
    boat_graph_remove_node(graph, node);
    return true;
}

bool boat_graph_safe_replace_node(const boat_graph_t* graph, const boat_node_t* old_node, const boat_node_t* new_node) {
    if (!graph || !old_node || !new_node) return false;

    // Check if old_node is in graph
    bool old_in_graph = false;
    bool new_in_graph = false;
    size_t old_index = SIZE_MAX;
    size_t new_index = SIZE_MAX;
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == old_node) {
            old_in_graph = true;
            old_index = i;
        }
        if (graph->nodes[i] == new_node) {
            new_in_graph = true;
            new_index = i;
        }
        if (old_in_graph && new_in_graph) break;
    }
    if (!old_in_graph) return false;

    // If new_node is not in graph, add it
    if (!new_in_graph) {
        // Add new_node to graph with same data as old_node
        // This is a shallow copy - actual data ownership needs consideration
        const boat_node_t* added_node = boat_graph_add_node(graph, boat_node_data(old_node),
                                                     boat_node_type(old_node), NULL);
        if (!added_node) return false;
        new_node = added_node;
        new_in_graph = true;
    }

    // Collect all edges involving old_node
    size_t edges_count = 0;
    struct boat_edge_t** old_edges = boat_malloc(graph->edge_count * sizeof(struct boat_edge_t*), graph->device);
    if (!old_edges) return false;

    for (size_t i = 0; i < graph->edge_count; i++) {
        struct boat_edge_t* edge = graph->edges[i];
        if (!edge) continue;

        const boat_node_t* source = boat_edge_source(edge);
        const boat_node_t* target = boat_edge_target(edge);

        if (source == old_node || target == old_node) {
            old_edges[edges_count++] = edge;
        }
    }

    // For each old edge, create a new edge with old_node replaced by new_node
    for (size_t i = 0; i < edges_count; i++) {
        const struct boat_edge_t* old_edge = old_edges[i];
        const boat_node_t* source = boat_edge_source(old_edge);
        const boat_node_t* target = boat_edge_target(old_edge);
        boat_edge_direction_t direction = boat_edge_direction(old_edge);

        // Determine new source and target
        const boat_node_t* new_source = (source == old_node) ? new_node : source;
        const boat_node_t* new_target = (target == old_node) ? new_node : target;

        // Create new edge
        const struct boat_edge_t* new_edge = boat_edge_create(new_source, new_target, direction);
        if (!new_edge) {
            // Cleanup: free old_edges array, but keep changes made so far
            boat_free(old_edges);
            return false;
        }

        // Add new edge to graph
        if (!boat_graph_add_edge(graph, new_source, new_target, direction)) {
            boat_edge_free(new_edge);
            boat_free(old_edges);
            return false;
        }
    }

    // Remove old edges
    for (size_t i = 0; i < edges_count; i++) {
        boat_graph_remove_edge(graph, old_edges[i]);
    }

    // Remove old node if desired (optional)
    // boat_graph_remove_node(graph, old_node);

    boat_free(old_edges);
    return true;
}

void boat_graph_batch_modifications(boat_graph_t* graph, bool begin) {
    if (!graph) return;

    if (begin) {
        graph->in_batch_mode = true;
        // Disable automatic validation during batch modifications
    } else {
        graph->in_batch_mode = false;
        // Apply any queued modifications and validate graph integrity
        boat_graph_validate(graph);
    }
}

// Cross-device communication and optimization
bool boat_graph_to_device(boat_graph_t* graph, boat_device_t device) {
    if (!graph) return false;

    if (graph->device == device) {
        // Already on target device
        return true;
    }

    // For Phase 2, we just update the graph's device field
    // Actual tensor data movement would require traversing nodes and moving tensors
    // This is a placeholder implementation
    graph->device = device;

    // Note: In a full implementation, we would:
    // 1. Traverse all nodes in the graph
    // 2. For each node with tensor data, move tensor to new device
    // 3. Update any device-specific resources

    return true;
}

size_t boat_graph_device_memory_usage(const boat_graph_t* graph, boat_device_t device) {
    if (!graph) return 0;

    size_t total_memory = 0;

    // For Phase 2, we return a placeholder value
    // In a full implementation, we would:
    // 1. Traverse all nodes in the graph
    // 2. For each node with tensor data on the specified device, add tensor size
    // 3. Include graph structure memory allocated on the device

    // Placeholder: return 0 for now
    return total_memory;
}

// Graph optimization functions

void boat_graph_optimize(const boat_graph_t* graph, unsigned int optimization_flags) {
    if (!graph) return;

    // Apply optimizations based on flags
    if (optimization_flags & BOAT_OPTIMIZE_CSE) {
        boat_graph_eliminate_common_subexpressions(graph);
    }
    if (optimization_flags & BOAT_OPTIMIZE_DCE) {
        boat_graph_eliminate_dead_code(graph);
    }
    if (optimization_flags & BOAT_OPTIMIZE_CONSTANT_FOLD) {
        boat_graph_fold_constants(graph);
    }
    if (optimization_flags & BOAT_OPTIMIZE_SIMPLIFY) {
        boat_graph_simplify(graph);
    }
}

void boat_graph_eliminate_common_subexpressions(const boat_graph_t* graph) {
    if (!graph) return;

    // Phase 3: Common subexpression elimination
    // This is a placeholder implementation
    // In a full implementation, we would:
    // 1. Identify identical operation subgraphs
    // 2. Merge duplicate computations
    // 3. Redirect edges to shared nodes

    // For now, just validate the graph after potential modifications
    if (!graph->in_batch_mode) {
        boat_graph_validate(graph);
    }
}

void boat_graph_eliminate_dead_code(const boat_graph_t* graph) {
    if (!graph) return;

    // Phase 3: Dead code elimination
    // This is a placeholder implementation
    // In a full implementation, we would:
    // 1. Mark all nodes reachable from output nodes
    // 2. Remove unmarked nodes (dead code)
    // 3. Clean up edges connected to removed nodes

    // For now, just validate the graph after potential modifications
    if (!graph->in_batch_mode) {
        boat_graph_validate(graph);
    }
}

void boat_graph_fold_constants(const boat_graph_t* graph) {
    if (!graph) return;

    // Phase 3: Constant folding
    // This is a placeholder implementation
    // In a full implementation, we would:
    // 1. Identify operations with constant inputs
    // 2. Pre-compute results at graph construction time
    // 3. Replace operation nodes with constant nodes

    // For now, just validate the graph after potential modifications
    if (!graph->in_batch_mode) {
        boat_graph_validate(graph);
    }
}

void boat_graph_simplify(const boat_graph_t* graph) {
    if (!graph) return;

    // Phase 3: Graph simplification
    // This is a placeholder implementation
    // In a full implementation, we would:
    // 1. Apply algebraic simplifications (e.g., x*1 = x, x+0 = x)
    // 2. Remove identity operations
    // 3. Combine consecutive operations where possible

    // For now, just validate the graph after potential modifications
    if (!graph->in_batch_mode) {
        boat_graph_validate(graph);
    }
}

// Node migration between graphs
bool boat_graph_migrate_node(const boat_graph_t* dest_graph, const boat_graph_t* src_graph, const boat_node_t* node) {
    if (!dest_graph || !src_graph || !node) {
        return false;
    }

    // Check if node is in source graph
    bool node_in_src = false;
    size_t src_index = SIZE_MAX;
    for (size_t i = 0; i < src_graph->node_count; i++) {
        if (src_graph->nodes[i] == node) {
            node_in_src = true;
            src_index = i;
            break;
        }
    }

    if (!node_in_src) {
        return false;
    }

    // Check if node is already in destination graph
    for (size_t i = 0; i < dest_graph->node_count; i++) {
        if (dest_graph->nodes[i] == node) {
            return true; // Already migrated
        }
    }


    // Remove node from source graph (but don't free it)
    // We need to remove from nodes array and adjust adjacency lists
    // First, remove all edges connected to this node from source graph
    // This is necessary because edges reference nodes by pointer
    // We'll collect edges to remove
    size_t edges_to_remove_count = 0;
    struct boat_edge_t** edges_to_remove = boat_malloc(src_graph->edge_count * sizeof(struct boat_edge_t*), src_graph->device);
    if (!edges_to_remove) {
        return false;
    }

    for (size_t i = 0; i < src_graph->edge_count; i++) {
        const struct boat_edge_t* edge = src_graph->edges[i];
        if (!edge) continue;
        const boat_node_t* source = boat_edge_source(edge);
        const boat_node_t* target = boat_edge_target(edge);
        if (source == node || target == node) {
            edges_to_remove[edges_to_remove_count++] = edge;
        }
    }

    // Remove collected edges
    for (size_t i = 0; i < edges_to_remove_count; i++) {
        boat_graph_remove_edge(src_graph, edges_to_remove[i]);
    }
    boat_free(edges_to_remove);

    // Remove node from source graph's nodes array
    for (size_t i = src_index; i < src_graph->node_count - 1; i++) {
        src_graph->nodes[i] = src_graph->nodes[i + 1];
    }
    src_graph->node_count--;

    // Adjust adjacency list arrays (shift elements after removed index)
    if (src_graph->outgoing) {
        for (size_t i = src_index; i < src_graph->node_capacity - 1; i++) {
            src_graph->outgoing[i] = src_graph->outgoing[i + 1];
        }
        src_graph->outgoing[src_graph->node_capacity - 1] = NULL;
    }
    if (src_graph->incoming) {
        for (size_t i = src_index; i < src_graph->node_capacity - 1; i++) {
            src_graph->incoming[i] = src_graph->incoming[i + 1];
        }
        src_graph->incoming[src_graph->node_capacity - 1] = NULL;
    }

    // Add node to destination graph
    if (!ensure_node_capacity(dest_graph, dest_graph->node_count + 1)) {
        // TODO: should we restore node to source graph?
        return false;
    }

    // Add node to destination graph's nodes array
    dest_graph->nodes[dest_graph->node_count++] = node;

    // Update node's ID? Keep original ID for now
    // Note: boat_graph_add_node assigns new ID, but we're reusing existing node
    // We need to ensure ID uniqueness. For simplicity, keep existing ID.
    // If destination graph already has a node with same ID, we may need to reassign.
    // For now, assume IDs are unique across graphs.

    return true;
}