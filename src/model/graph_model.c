// graph_model.c - Graph-based model implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/model.h>
#include <boat/memory.h>
#include <boat/tensor.h>
#include <boat/graph.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Graph model private structure
typedef struct {
    boat_model_t* model;         // Base model
    boat_node_t* input_node;     // Input node in the graph
    boat_node_t* output_node;    // Output node in the graph
    boat_tensor_t** parameters;  // Array of parameter tensors
    size_t parameter_count;      // Number of parameters
    size_t parameter_capacity;   // Capacity of parameters array
} boat_graph_model_private_t;

// Graph model creation
boat_model_t* boat_graph_model_create(boat_graph_t* graph, boat_node_t* input_node, boat_node_t* output_node) {
    if (!graph || !input_node || !output_node) {
        return NULL;
    }

    // Create base model
    boat_model_t* model = boat_model_create_with_graph(graph);
    if (!model) {
        return NULL;
    }

    // Allocate private data
    boat_graph_model_private_t* private = boat_malloc(sizeof(boat_graph_model_private_t), BOAT_DEVICE_CPU);
    if (!private) {
        boat_model_free(model);
        return NULL;
    }

    private->model = model;
    private->input_node = input_node;
    private->output_node = output_node;
    private->parameters = NULL;
    private->parameter_count = 0;
    private->parameter_capacity = 0;

    // Store private data in model's user_data field
    boat_model_set_user_data(model, private, boat_memory_free);

    return model;
}

// Add parameter to graph model
void boat_graph_model_add_parameter(boat_model_t* model, boat_tensor_t* parameter) {
    if (!model || !parameter) {
        return;
    }

    boat_graph_model_private_t* private = (boat_graph_model_private_t*)boat_model_get_user_data(model);
    if (!private) {
        return;
    }

    // Resize parameters array if needed
    if (private->parameter_count >= private->parameter_capacity) {
        size_t new_capacity = private->parameter_capacity == 0 ? 8 : private->parameter_capacity * 2;
        boat_tensor_t** new_params = boat_realloc(private->parameters, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
        if (!new_params) {
            return;
        }

        private->parameters = new_params;
        private->parameter_capacity = new_capacity;
    }

    // Add parameter
    private->parameters[private->parameter_count] = parameter;
    boat_tensor_ref(parameter);
    private->parameter_count++;
}