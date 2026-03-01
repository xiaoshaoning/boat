// sequential.c - Sequential model implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/model.h>
#include <boat/memory.h>
#include <boat/tensor.h>
#include <boat/graph.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Sequential model private structure
typedef struct {
    boat_layer_t** layers;       // Array of layers
    size_t layer_count;          // Number of layers
    size_t layer_capacity;       // Capacity of layers array
} boat_sequential_model_private_t;

// Sequential model creation
boat_sequential_model_t* boat_sequential_create() {
    // Create base model
    boat_model_t* model = boat_model_create();
    if (!model) {
        return NULL;
    }

    // Allocate private data
    boat_sequential_model_private_t* private = boat_malloc(sizeof(boat_sequential_model_private_t), BOAT_DEVICE_CPU);
    if (!private) {
        boat_model_free(model);
        return NULL;
    }

    private->layers = NULL;
    private->layer_count = 0;
    private->layer_capacity = 0;

    // Store private data in model's user_data field
    boat_model_set_user_data(model, private, boat_memory_free);

    return (boat_sequential_model_t*)model;
}

// Add layer to sequential model
void boat_sequential_add(boat_sequential_model_t* model, boat_layer_t* layer) {
    if (!model || !layer) {
        return;
    }

    boat_sequential_model_private_t* private = (boat_sequential_model_private_t*)boat_model_get_user_data((boat_model_t*)model);
    if (!private) {
        return;
    }

    // Resize layers array if needed
    if (private->layer_count >= private->layer_capacity) {
        size_t new_capacity = private->layer_capacity == 0 ? 4 : private->layer_capacity * 2;
        boat_layer_t** new_layers = boat_realloc(private->layers, new_capacity * sizeof(boat_layer_t*), BOAT_DEVICE_CPU);
        if (!new_layers) {
            return;
        }

        private->layers = new_layers;
        private->layer_capacity = new_capacity;
    }

    // Add layer
    private->layers[private->layer_count] = layer;
    private->layer_count++;
}