// gru.c - GRU layer implementation
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat/layers.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>

// GRU layer structure
struct boat_gru_layer_t {
    size_t input_size;
    size_t hidden_size;
    size_t num_layers;
    bool bidirectional;
    float dropout;

    // Parameters (weights and biases)
    boat_tensor_t* weight_ih;   // Input-hidden weights
    boat_tensor_t* weight_hh;   // Hidden-hidden weights
    boat_tensor_t* bias_ih;     // Input-hidden biases
    boat_tensor_t* bias_hh;     // Hidden-hidden biases

    // Internal state
    boat_tensor_t* hidden_state;
};

// Create GRU layer
BOAT_API boat_gru_layer_t* BOAT_CALL boat_gru_layer_create(size_t input_size, size_t hidden_size,
                                        size_t num_layers, bool bidirectional,
                                        float dropout) {
    boat_gru_layer_t* layer = (boat_gru_layer_t*)boat_malloc(sizeof(boat_gru_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }

    layer->input_size = input_size;
    layer->hidden_size = hidden_size;
    layer->num_layers = num_layers;
    layer->bidirectional = bidirectional;
    layer->dropout = dropout;

    // Initialize parameters to NULL (will be set by model loading)
    layer->weight_ih = NULL;
    layer->weight_hh = NULL;
    layer->bias_ih = NULL;
    layer->bias_hh = NULL;
    layer->hidden_state = NULL;

    return layer;
}

// Free GRU layer
BOAT_API void BOAT_CALL boat_gru_layer_free(boat_gru_layer_t* layer) {
    if (!layer) return;

    if (layer->weight_ih) boat_tensor_unref(layer->weight_ih);
    if (layer->weight_hh) boat_tensor_unref(layer->weight_hh);
    if (layer->bias_ih) boat_tensor_unref(layer->bias_ih);
    if (layer->bias_hh) boat_tensor_unref(layer->bias_hh);
    if (layer->hidden_state) boat_tensor_unref(layer->hidden_state);

    boat_free(layer);
}

// Forward pass (placeholder)
BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_forward(boat_gru_layer_t* layer, const boat_tensor_t* input) {
    (void)layer;
    (void)input;
    // TODO: Implement GRU forward pass
    return NULL;
}

// Backward pass (placeholder)
BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_backward(boat_gru_layer_t* layer, const boat_tensor_t* grad_output) {
    (void)layer;
    (void)grad_output;
    // TODO: Implement GRU backward pass
    return NULL;
}

// Update parameters (placeholder)
BOAT_API void BOAT_CALL boat_gru_layer_update(boat_gru_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // TODO: Implement parameter update
}

// Parameter setters for model loading
BOAT_API void BOAT_CALL boat_gru_layer_set_weight_ih(boat_gru_layer_t* layer, boat_tensor_t* weight) {
    if (layer->weight_ih) boat_tensor_unref(layer->weight_ih);
    layer->weight_ih = weight;
    if (weight) boat_tensor_ref(weight);
}

BOAT_API void BOAT_CALL boat_gru_layer_set_weight_hh(boat_gru_layer_t* layer, boat_tensor_t* weight) {
    if (layer->weight_hh) boat_tensor_unref(layer->weight_hh);
    layer->weight_hh = weight;
    if (weight) boat_tensor_ref(weight);
}

BOAT_API void BOAT_CALL boat_gru_layer_set_bias_ih(boat_gru_layer_t* layer, boat_tensor_t* bias) {
    if (layer->bias_ih) boat_tensor_unref(layer->bias_ih);
    layer->bias_ih = bias;
    if (bias) boat_tensor_ref(bias);
}

BOAT_API void BOAT_CALL boat_gru_layer_set_bias_hh(boat_gru_layer_t* layer, boat_tensor_t* bias) {
    if (layer->bias_hh) boat_tensor_unref(layer->bias_hh);
    layer->bias_hh = bias;
    if (bias) boat_tensor_ref(bias);
}