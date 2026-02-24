// executor.c - Graph execution engine for computational graph
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/graph.h>
#include <boat/memory.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "graph_private.h"

// Internal forward declarations
typedef boat_tensor_t* (*boat_forward_fn_t)(size_t num_inputs, boat_tensor_t** inputs, void* user_data);
typedef void (*boat_backward_fn_t)(size_t num_inputs, boat_tensor_t** inputs, boat_tensor_t* output,
                                   boat_tensor_t* output_grad, boat_tensor_t** input_grads, void* user_data);

struct boat_computation_node_data {
    boat_forward_fn_t forward_fn;
    boat_backward_fn_t backward_fn;
    void* user_data;
    boat_tensor_t* output;
    boat_tensor_t* gradient;
};

boat_graph_t* boat_computation_graph_create() {
    boat_graph_t* graph = boat_graph_create();
    if (!graph) return NULL;
    // No extra initialization needed for now
    return graph;
}

void boat_computation_graph_forward(boat_graph_t* graph) {
    if (!graph) return;

    // Topological sort to get execution order
    size_t node_count = boat_graph_node_count(graph);
    boat_node_t** sorted_nodes = boat_malloc(node_count * sizeof(boat_node_t*), BOAT_DEVICE_CPU);
    if (!sorted_nodes) return;

    size_t sorted_count = 0;
    boat_graph_topological_sort(graph, sorted_nodes, &sorted_count);

    // Execute nodes in topological order
    for (size_t i = 0; i < sorted_count; i++) {
        boat_node_t* node = sorted_nodes[i];
        boat_node_type_t node_type = boat_node_type(node);

        // Handle different node types
        if (node_type == BOAT_NODE_TYPE_OPERATION) {
            struct boat_computation_node_data* data = boat_node_data(node);
            if (!data || !data->forward_fn) {
                continue;
            }

            // Collect inputs from predecessor nodes (nodes with edges to this node)
            // Count incoming edges with forward direction
            size_t num_inputs = 0;
            for (size_t j = 0; j < graph->edge_count; j++) {
                struct boat_edge_t* edge = graph->edges[j];
                if (!edge) continue;
                if (boat_edge_target(edge) == node &&
                    boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                    num_inputs++;
                }
            }

            if (num_inputs == 0) {
                // No inputs, call forward with NULL inputs
                data->output = data->forward_fn(0, NULL, data->user_data);
                continue;
            }

            // Allocate array for input tensors
            boat_tensor_t** inputs = boat_malloc(num_inputs * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
            if (!inputs) continue;

            // Fill input array
            size_t input_idx = 0;
            for (size_t j = 0; j < graph->edge_count; j++) {
                struct boat_edge_t* edge = graph->edges[j];
                if (!edge) continue;
                if (boat_edge_target(edge) == node &&
                    boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                    boat_node_t* source_node = boat_edge_source(edge);
                    struct boat_computation_node_data* source_data = boat_node_data(source_node);
                    if (source_data && source_data->output) {
                        inputs[input_idx++] = source_data->output;
                    } else {
                        // Input not available
                        boat_free(inputs);
                        inputs = NULL;
                        break;
                    }
                }
            }

            if (!inputs) continue;

            // Call forward function
            data->output = data->forward_fn(num_inputs, inputs, data->user_data);
            boat_free(inputs);

        } else if (node_type == BOAT_NODE_TYPE_VARIABLE || node_type == BOAT_NODE_TYPE_CONSTANT) {
            // Variables and constants already have their data
            // Nothing to compute
        } else if (node_type == BOAT_NODE_TYPE_PLACEHOLDER) {
            // Placeholder nodes need external input
            // For now, do nothing
        } else if (node_type == BOAT_NODE_TYPE_OUTPUT) {
            // Output nodes are typically just references to other nodes
            // Their value should come from an incoming edge
        }
    }

    boat_memory_free(sorted_nodes);
}

void boat_computation_graph_backward(boat_graph_t* graph) {
    if (!graph) return;

    // Reverse topological order
    size_t node_count = boat_graph_node_count(graph);
    boat_node_t** sorted_nodes = boat_malloc(node_count * sizeof(boat_node_t*), BOAT_DEVICE_CPU);
    if (!sorted_nodes) return;

    size_t sorted_count = 0;
    boat_graph_topological_sort(graph, sorted_nodes, &sorted_count);

    // Execute backward pass in reverse order
    for (size_t i = sorted_count; i > 0; i--) {
        boat_node_t* node = sorted_nodes[i - 1];
        boat_node_type_t node_type = boat_node_type(node);

        if (node_type == BOAT_NODE_TYPE_OPERATION) {
            struct boat_computation_node_data* data = boat_node_data(node);
            if (!data || !data->backward_fn) {
                continue;
            }

            // Collect gradient from successor nodes (nodes this node feeds into)
            // For simplicity, assume single output gradient
            boat_tensor_t* output_grad = NULL;

            // Find outgoing edges with forward direction
            for (size_t j = 0; j < graph->edge_count; j++) {
                struct boat_edge_t* edge = graph->edges[j];
                if (!edge) continue;
                if (boat_edge_source(edge) == node &&
                    boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                    boat_node_t* target_node = boat_edge_target(edge);
                    struct boat_computation_node_data* target_data = boat_node_data(target_node);
                    if (target_data && target_data->gradient) {
                        // Use the first gradient we find
                        // In proper implementation, we would sum gradients from multiple outputs
                        output_grad = target_data->gradient;
                        break;
                    }
                }
            }

            // If no output gradient found and this is an output node, use default
            if (!output_grad && node_type == BOAT_NODE_TYPE_OUTPUT) {
                // For output nodes, we might want to propagate a gradient of 1
                // This would require creating an all-ones tensor matching output shape
                // For now, skip
                continue;
            }

            // Collect inputs from predecessor nodes
            size_t num_inputs = 0;
            for (size_t j = 0; j < graph->edge_count; j++) {
                struct boat_edge_t* edge = graph->edges[j];
                if (!edge) continue;
                if (boat_edge_target(edge) == node &&
                    boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                    num_inputs++;
                }
            }

            if (num_inputs == 0) {
                // No inputs, call backward with NULL
                data->backward_fn(0, NULL, data->output, output_grad, NULL, data->user_data);
                continue;
            }

            // Allocate arrays for inputs and input gradients
            boat_tensor_t** inputs = boat_malloc(num_inputs * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
            boat_tensor_t** input_grads = boat_malloc(num_inputs * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
            if (!inputs || !input_grads) {
                boat_free(inputs);
                boat_free(input_grads);
                continue;
            }

            // Fill input array and initialize input gradients to NULL
            size_t input_idx = 0;
            for (size_t j = 0; j < graph->edge_count; j++) {
                struct boat_edge_t* edge = graph->edges[j];
                if (!edge) continue;
                if (boat_edge_target(edge) == node &&
                    boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                    boat_node_t* source_node = boat_edge_source(edge);
                    struct boat_computation_node_data* source_data = boat_node_data(source_node);
                    if (source_data && source_data->output) {
                        inputs[input_idx] = source_data->output;
                        input_grads[input_idx] = NULL;  // To be filled by backward function
                        input_idx++;
                    } else {
                        // Input not available
                        boat_free(inputs);
                        boat_free(input_grads);
                        inputs = NULL;
                        input_grads = NULL;
                        break;
                    }
                }
            }

            if (!inputs || !input_grads) continue;

            // Call backward function
            data->backward_fn(num_inputs, inputs, data->output, output_grad, input_grads, data->user_data);

            // Distribute computed gradients back to input nodes
            input_idx = 0;
            for (size_t j = 0; j < graph->edge_count; j++) {
                struct boat_edge_t* edge = graph->edges[j];
                if (!edge) continue;
                if (boat_edge_target(edge) == node &&
                    boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                    boat_node_t* source_node = boat_edge_source(edge);
                    struct boat_computation_node_data* source_data = boat_node_data(source_node);
                    if (source_data && input_idx < num_inputs) {
                        // If this input node expects gradient (e.g., is a variable),
                        // store or accumulate the gradient
                        if (source_data->gradient) {
                            // Already has gradient, accumulate
                            // For now, just replace (should be +=)
                            source_data->gradient = input_grads[input_idx];
                        } else {
                            source_data->gradient = input_grads[input_idx];
                        }
                        input_idx++;
                    }
                }
            }

            boat_free(inputs);
            boat_free(input_grads);

        } else if (node_type == BOAT_NODE_TYPE_VARIABLE) {
            // Variable nodes store gradients computed during backward pass
            // Nothing to compute here
        }
    }

    boat_memory_free(sorted_nodes);
}

void boat_computation_graph_clear_gradients(boat_graph_t* graph) {
    if (!graph) return;

    size_t node_count = boat_graph_node_count(graph);
    for (size_t i = 0; i < node_count; i++) {
        boat_node_t* node = boat_graph_get_node_at_index(graph, i);
        boat_node_type_t node_type = boat_node_type(node);

        if (node_type == BOAT_NODE_TYPE_VARIABLE || node_type == BOAT_NODE_TYPE_OPERATION) {
            struct boat_computation_node_data* data = boat_node_data(node);
            if (data && data->gradient) {
                // Free the gradient tensor
                boat_tensor_unref(data->gradient);
                data->gradient = NULL;
            }
        }
    }
}

// Helper function to add operation node with forward/backward functions
boat_node_t* boat_computation_graph_add_operation(boat_graph_t* graph,
                                                  boat_forward_fn_t forward_fn,
                                                  boat_backward_fn_t backward_fn,
                                                  void* user_data) {
    if (!graph) return NULL;

    struct boat_computation_node_data* data = boat_malloc(sizeof(struct boat_computation_node_data), BOAT_DEVICE_CPU);
    if (!data) return NULL;

    data->forward_fn = forward_fn;
    data->backward_fn = backward_fn;
    data->user_data = user_data;
    data->output = NULL;
    data->gradient = NULL;

    boat_node_t* node = boat_graph_add_node(graph, data, BOAT_NODE_TYPE_OPERATION, boat_memory_free);
    if (!node) {
        boat_memory_free(data);
        return NULL;
    }

    return node;
}

// Helper function to add variable node with initial tensor
boat_node_t* boat_computation_graph_add_variable(boat_graph_t* graph,
                                                 boat_tensor_t* tensor) {
    if (!graph || !tensor) return NULL;

    struct boat_computation_node_data* data = boat_malloc(sizeof(struct boat_computation_node_data), BOAT_DEVICE_CPU);
    if (!data) return NULL;

    data->forward_fn = NULL;
    data->backward_fn = NULL;
    data->user_data = NULL;
    data->output = tensor;
    data->gradient = NULL;

    boat_node_t* node = boat_graph_add_node(graph, data, BOAT_NODE_TYPE_VARIABLE, boat_memory_free);
    if (!node) {
        boat_memory_free(data);
        return NULL;
    }

    return node;
}