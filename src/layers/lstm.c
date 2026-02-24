// lstm.c - LSTM layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>

// LSTM layer structure
struct boat_lstm_layer_t {
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
    boat_tensor_t* cell_state;
};

// Create LSTM layer
boat_lstm_layer_t* boat_lstm_layer_create(size_t input_size, size_t hidden_size,
                                          size_t num_layers, bool bidirectional,
                                          float dropout) {
    boat_lstm_layer_t* layer = (boat_lstm_layer_t*)boat_malloc(sizeof(boat_lstm_layer_t), BOAT_DEVICE_CPU);
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
    layer->cell_state = NULL;

    return layer;
}

// Free LSTM layer
void boat_lstm_layer_free(boat_lstm_layer_t* layer) {
    if (!layer) return;

    if (layer->weight_ih) boat_tensor_unref(layer->weight_ih);
    if (layer->weight_hh) boat_tensor_unref(layer->weight_hh);
    if (layer->bias_ih) boat_tensor_unref(layer->bias_ih);
    if (layer->bias_hh) boat_tensor_unref(layer->bias_hh);
    if (layer->hidden_state) boat_tensor_unref(layer->hidden_state);
    if (layer->cell_state) boat_tensor_unref(layer->cell_state);

    boat_free(layer);
}

// Forward pass (placeholder)
boat_tensor_t* boat_lstm_layer_forward(boat_lstm_layer_t* layer, const boat_tensor_t* input) {
    (void)layer;
    (void)input;
    // TODO: Implement LSTM forward pass
    return NULL;
}

// Backward pass (placeholder)
boat_tensor_t* boat_lstm_layer_backward(boat_lstm_layer_t* layer, const boat_tensor_t* grad_output) {
    (void)layer;
    (void)grad_output;
    // TODO: Implement LSTM backward pass
    return NULL;
}

// Update parameters (placeholder)
void boat_lstm_layer_update(boat_lstm_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // TODO: Implement parameter update
}

// Parameter setters for model loading
void boat_lstm_layer_set_weight_ih(boat_lstm_layer_t* layer, boat_tensor_t* weight) {
    if (layer->weight_ih) boat_tensor_unref(layer->weight_ih);
    layer->weight_ih = weight;
    if (weight) boat_tensor_ref(weight);
}

void boat_lstm_layer_set_weight_hh(boat_lstm_layer_t* layer, boat_tensor_t* weight) {
    if (layer->weight_hh) boat_tensor_unref(layer->weight_hh);
    layer->weight_hh = weight;
    if (weight) boat_tensor_ref(weight);
}

void boat_lstm_layer_set_bias_ih(boat_lstm_layer_t* layer, boat_tensor_t* bias) {
    if (layer->bias_ih) boat_tensor_unref(layer->bias_ih);
    layer->bias_ih = bias;
    if (bias) boat_tensor_ref(bias);
}

void boat_lstm_layer_set_bias_hh(boat_lstm_layer_t* layer, boat_tensor_t* bias) {
    if (layer->bias_hh) boat_tensor_unref(layer->bias_hh);
    layer->bias_hh = bias;
    if (bias) boat_tensor_ref(bias);
}