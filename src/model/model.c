// model.c - Model definition and serialization
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/model.h>
#include <boat/memory.h>
#include <boat/tensor.h>
#include <boat/graph.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Model structure
struct boat_model_t {
    boat_graph_t* graph;          // Computational graph representing the model
    char* name;                   // Model name (optional)
    boat_device_t device;         // Device where model is stored
    bool trainable;               // Whether model is in training mode
    void* user_data;              // User-defined data (optional)
    void (*free_user_data)(void*); // Function to free user_data
    // Layers storage (temporary until integrated with graph)
    boat_layer_t** layers;        // Array of layer pointers
    boat_node_t** nodes;          // Array of graph node pointers (parallel to layers)
    size_t layer_count;           // Number of layers
    size_t layer_capacity;        // Capacity of layers array
};

// Layer structure is now defined in model.h

// Model creation and management
boat_model_t* boat_model_create() {
    boat_model_t* model = boat_malloc(sizeof(boat_model_t), BOAT_DEVICE_CPU);
    if (!model) {
        return NULL;
    }

    // Create a new computational graph for the model
    model->graph = boat_graph_create();
    if (!model->graph) {
        boat_free(model);
        return NULL;
    }

    model->name = NULL;
    model->device = BOAT_DEVICE_CPU;
    model->trainable = true;
    model->user_data = NULL;
    model->free_user_data = NULL;

    // Initialize layers storage
    model->layers = NULL;
    model->nodes = NULL;
    model->layer_count = 0;
    model->layer_capacity = 0;

    return model;
}

boat_model_t* boat_model_create_with_graph(boat_graph_t* graph) {
    if (!graph) {
        return NULL;
    }

    boat_model_t* model = boat_malloc(sizeof(boat_model_t), BOAT_DEVICE_CPU);
    if (!model) {
        return NULL;
    }

    model->graph = graph;
    // Note: graph ownership is transferred to model
    model->name = NULL;
    model->device = boat_graph_device(graph);
    model->trainable = true;
    model->user_data = NULL;
    model->free_user_data = NULL;

    // Initialize layers storage
    model->layers = NULL;
    model->nodes = NULL;
    model->layer_count = 0;
    model->layer_capacity = 0;

    return model;
}

void boat_model_free(boat_model_t* model) {
    if (!model) {
        return;
    }

    // Free layers
    if (model->layers) {
        for (size_t i = 0; i < model->layer_count; i++) {
            boat_layer_t* layer = model->layers[i];
            if (layer) {
                // Use layer operations if available
                if (layer->ops && layer->ops->free) {
                    layer->ops->free(layer);
                } else {
                    // Fallback: free layer wrapper only
                    boat_free(layer);
                }
            }
        }
        boat_free(model->layers);
    }

    // Free nodes array (nodes themselves are owned by the graph)
    if (model->nodes) {
        boat_free(model->nodes);
    }

    if (model->graph) {
        boat_graph_free(model->graph);
    }

    if (model->name) {
        boat_free(model->name);
    }

    if (model->user_data && model->free_user_data) {
        model->free_user_data(model->user_data);
    }

    boat_free(model);
}

// Graph access
boat_graph_t* boat_model_graph(const boat_model_t* model) {
    return model ? model->graph : NULL;
}

void boat_model_set_graph(boat_model_t* model, boat_graph_t* graph) {
    if (!model || !graph) {
        return;
    }

    if (model->graph) {
        boat_graph_free(model->graph);
    }

    model->graph = graph;
    // Note: graph ownership is transferred to model
    model->device = boat_graph_device(graph);
}

// Model operations
boat_tensor_t* boat_model_forward(boat_model_t* model, const boat_tensor_t* input) {
    if (!model || !input) return NULL;
    if (model->layer_count == 0) return NULL;

    // If graph is empty (no nodes), fall back to sequential execution
    if (boat_graph_node_count(model->graph) == 0) {
        boat_tensor_t* current = NULL;
        boat_tensor_t* next = NULL;

        // Process first layer
        boat_layer_t* first_layer = model->layers[0];
        if (!first_layer || !first_layer->ops || !first_layer->ops->forward) {
            return NULL;
        }
        current = first_layer->ops->forward(first_layer, input);
        if (!current) return NULL;

        // Process remaining layers
        for (size_t i = 1; i < model->layer_count; i++) {
            boat_layer_t* layer = model->layers[i];
            if (!layer || !layer->ops || !layer->ops->forward) {
                boat_tensor_free(current);
                return NULL;
            }
            next = layer->ops->forward(layer, current);
            boat_tensor_free(current);
            if (!next) return NULL;
            current = next;
        }

        return current;
    }

    // Use computational graph for forward propagation
    boat_graph_t* graph = model->graph;

    // Get topological sort of nodes
    size_t node_count = boat_graph_node_count(graph);
    boat_node_t** sorted_nodes = boat_malloc(node_count * sizeof(boat_node_t*), BOAT_DEVICE_CPU);
    if (!sorted_nodes) {
        return NULL;
    }

    size_t sorted_count = 0;
    boat_graph_topological_sort(graph, sorted_nodes, &sorted_count);

    // Map from node to output tensor
    boat_tensor_t** node_outputs = boat_calloc(node_count * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    if (!node_outputs) {
        boat_free(sorted_nodes);
        return NULL;
    }

    // Map from node to layer (create index mapping)
    // Build map from node pointer to layer index
    // Since model->nodes array parallels model->layers, we can search
    boat_tensor_t* final_output = NULL;

    // Process nodes in topological order
    for (size_t i = 0; i < sorted_count; i++) {
        boat_node_t* node = sorted_nodes[i];

        // Find layer index for this node
        size_t layer_index = SIZE_MAX;
        for (size_t j = 0; j < model->layer_count; j++) {
            if (model->nodes[j] == node) {
                layer_index = j;
                break;
            }
        }

        if (layer_index == SIZE_MAX) {
            // Node not found in model nodes (could be input/output node)
            // Skip for now
            continue;
        }

        boat_layer_t* layer = model->layers[layer_index];
        if (!layer || !layer->ops || !layer->ops->forward) {
            // Cleanup and return error
            for (size_t k = 0; k < node_count; k++) {
                if (node_outputs[k]) {
                    boat_tensor_free(node_outputs[k]);
                }
            }
            boat_free(node_outputs);
            boat_free(sorted_nodes);
            return NULL;
        }

        // Collect input tensors from predecessor nodes
        // Count incoming edges with forward direction
        size_t num_inputs = 0;
        for (size_t j = 0; j < boat_graph_edge_count(graph); j++) {
            boat_edge_t* edge = boat_graph_get_edge_at_index(graph, j);
            if (!edge) continue;
            if (boat_edge_target(edge) == node &&
                boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                num_inputs++;
            }
        }

        boat_tensor_t* layer_input = NULL;

        if (num_inputs == 0) {
            // First layer in graph - use external input
            layer_input = (boat_tensor_t*)input;  // Cast away const for API compatibility
        } else if (num_inputs == 1) {
            // Single input - typical for sequential models
            for (size_t j = 0; j < boat_graph_edge_count(graph); j++) {
                boat_edge_t* edge = boat_graph_get_edge_at_index(graph, j);
                if (!edge) continue;
                if (boat_edge_target(edge) == node &&
                    boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                    boat_node_t* source_node = boat_edge_source(edge);
                    // Find source node index
                    size_t source_idx = SIZE_MAX;
                    for (size_t k = 0; k < model->layer_count; k++) {
                        if (model->nodes[k] == source_node) {
                            source_idx = k;
                            break;
                        }
                    }
                    if (source_idx != SIZE_MAX && source_idx < node_count) {
                        layer_input = node_outputs[source_idx];
                    }
                    break;
                }
            }
        } else {
            // Multiple inputs - not yet supported for simple sequential models
            // For now, use first input
            for (size_t j = 0; j < boat_graph_edge_count(graph); j++) {
                boat_edge_t* edge = boat_graph_get_edge_at_index(graph, j);
                if (!edge) continue;
                if (boat_edge_target(edge) == node &&
                    boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                    boat_node_t* source_node = boat_edge_source(edge);
                    size_t source_idx = SIZE_MAX;
                    for (size_t k = 0; k < model->layer_count; k++) {
                        if (model->nodes[k] == source_node) {
                            source_idx = k;
                            break;
                        }
                    }
                    if (source_idx != SIZE_MAX && source_idx < node_count) {
                        layer_input = node_outputs[source_idx];
                        break;
                    }
                }
            }
        }

        if (!layer_input && num_inputs > 0) {
            // Input not available
            for (size_t k = 0; k < node_count; k++) {
                if (node_outputs[k]) {
                    boat_tensor_free(node_outputs[k]);
                }
            }
            boat_free(node_outputs);
            boat_free(sorted_nodes);
            return NULL;
        }

        // Call layer forward
        boat_tensor_t* output = layer->ops->forward(layer, layer_input);
        if (!output) {
            // Cleanup
            for (size_t k = 0; k < node_count; k++) {
                if (node_outputs[k]) {
                    boat_tensor_free(node_outputs[k]);
                }
            }
            boat_free(node_outputs);
            boat_free(sorted_nodes);
            return NULL;
        }

        // Store output
        node_outputs[layer_index] = output;

        // If this is the last layer (no outgoing edges), it's the final output
        bool has_outgoing = false;
        for (size_t j = 0; j < boat_graph_edge_count(graph); j++) {
            boat_edge_t* edge = boat_graph_get_edge_at_index(graph, j);
            if (!edge) continue;
            if (boat_edge_source(edge) == node &&
                boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                has_outgoing = true;
                break;
            }
        }

        if (!has_outgoing) {
            final_output = output;
        }
    }

    // Cleanup temporary arrays (but keep final output)
    boat_free(sorted_nodes);
    boat_free(node_outputs);

    return final_output;
}

boat_tensor_t* boat_model_backward(boat_model_t* model, const boat_tensor_t* grad_output) {
    (void)model;
    (void)grad_output;
    // TODO: Implement backward pass through computational graph
    return NULL;
}

void boat_model_update(boat_model_t* model, float learning_rate) {
    (void)model;
    (void)learning_rate;
    // TODO: Implement parameter update
}

// Model serialization
boat_model_t* boat_model_load(const char* filename) {
    (void)filename;
    // TODO: Implement model loading from file
    return NULL;
}

bool boat_model_save(const boat_model_t* model, const char* filename) {
    (void)model;
    (void)filename;
    // TODO: Implement model saving to file
    return false;
}

// User data management
void* boat_model_get_user_data(const boat_model_t* model) {
    return model ? model->user_data : NULL;
}

void boat_model_set_user_data(boat_model_t* model, void* user_data, void (*free_fn)(void*)) {
    if (!model) {
        return;
    }

    // Free existing user data if any
    if (model->user_data && model->free_user_data) {
        model->free_user_data(model->user_data);
    }

    model->user_data = user_data;
    model->free_user_data = free_fn;
}

// Get layer count
size_t boat_model_layer_count(const boat_model_t* model) {
    return model ? model->layer_count : 0;
}

// Add layer to model
void boat_model_add_layer(boat_model_t* model, boat_layer_t* layer) {
    if (!model || !layer) {
        return;
    }

    // Expand layers and nodes arrays if needed
    if (model->layer_count >= model->layer_capacity) {
        size_t new_capacity = model->layer_capacity == 0 ? 4 : model->layer_capacity * 2;

        // Reallocate layers array
        boat_layer_t** new_layers = boat_realloc(model->layers, new_capacity * sizeof(boat_layer_t*), BOAT_DEVICE_CPU);
        if (!new_layers) {
            fprintf(stderr, "Failed to expand layers array\n");
            return;
        }
        model->layers = new_layers;

        // Reallocate nodes array
        boat_node_t** new_nodes = boat_realloc(model->nodes, new_capacity * sizeof(boat_node_t*), BOAT_DEVICE_CPU);
        if (!new_nodes) {
            fprintf(stderr, "Failed to expand nodes array\n");
            // Note: layers array already reallocated, but this is an error state
            return;
        }
        model->nodes = new_nodes;
        model->layer_capacity = new_capacity;
    }

    // Create graph node for this layer
    boat_node_t* node = boat_graph_add_node(model->graph, layer, BOAT_NODE_TYPE_OPERATION, NULL);
    if (!node) {
        fprintf(stderr, "Failed to create graph node for layer\n");
        return;
    }

    // Connect to previous node if exists
    if (model->layer_count > 0) {
        boat_node_t* prev_node = model->nodes[model->layer_count - 1];
        if (prev_node) {
            boat_edge_t* edge = boat_graph_add_edge(model->graph, prev_node, node, BOAT_EDGE_DIRECTION_FORWARD);
            if (!edge) {
                fprintf(stderr, "Warning: Failed to add edge between layer nodes\n");
            } else {
                printf("Created edge from previous layer node to current layer node\n");
            }
        }
    }

    // Store layer and node
    model->layers[model->layer_count] = layer;
    model->nodes[model->layer_count] = node;
    model->layer_count++;

    printf("Added layer to model with graph node, total layers: %zu\n", model->layer_count);
}
