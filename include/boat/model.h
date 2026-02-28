// model.h - Neural network model definition and serialization
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#ifndef BOAT_MODEL_H
#define BOAT_MODEL_H

#include <stdbool.h>
#include "tensor.h"
#include "graph.h"

#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct boat_model_t boat_model_t;
typedef struct boat_layer_ops_t boat_layer_ops_t;

// Layer structure
typedef struct boat_layer_t {
    void* data;                    // Pointer to layer-specific data (boat_dense_layer_t, etc.)
    const boat_layer_ops_t* ops;   // Layer operations (optional, can be NULL)
} boat_layer_t;

// Layer interface
struct boat_layer_ops_t {
    boat_tensor_t* (*forward)(boat_layer_t* layer, const boat_tensor_t* input);
    boat_tensor_t* (*backward)(boat_layer_t* layer, const boat_tensor_t* grad_output);
    void (*update)(boat_layer_t* layer, float learning_rate);
    void (*free)(boat_layer_t* layer);
};

// Model creation and management
BOAT_API boat_model_t* boat_model_create();
BOAT_API boat_model_t* boat_model_create_with_graph(boat_graph_t* graph);
BOAT_API void boat_model_free(boat_model_t* model);

// Graph access
BOAT_API boat_graph_t* boat_model_graph(const boat_model_t* model);
BOAT_API void boat_model_set_graph(boat_model_t* model, boat_graph_t* graph);

// User data management
BOAT_API void* boat_model_get_user_data(const boat_model_t* model);
BOAT_API void boat_model_set_user_data(boat_model_t* model, void* user_data, void (*free_fn)(void*));

// Layer management
BOAT_API void boat_model_add_layer(boat_model_t* model, boat_layer_t* layer);
BOAT_API size_t boat_model_layer_count(const boat_model_t* model);

// Model operations
BOAT_API boat_tensor_t* boat_model_forward(boat_model_t* model, const boat_tensor_t* input);
BOAT_API boat_tensor_t* boat_model_backward(boat_model_t* model, const boat_tensor_t* grad_output);
BOAT_API void boat_model_update(boat_model_t* model, float learning_rate);

// Model serialization
BOAT_API boat_model_t* boat_model_load(const char* filename);
BOAT_API bool boat_model_save(const boat_model_t* model, const char* filename);

// Sequential model (simplified API)
typedef boat_model_t boat_sequential_model_t;
BOAT_API boat_sequential_model_t* boat_sequential_create();
BOAT_API void boat_sequential_add(boat_sequential_model_t* model, boat_layer_t* layer);

#ifdef __cplusplus
}
#endif

#endif // BOAT_MODEL_H