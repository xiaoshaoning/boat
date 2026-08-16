// activation_layers.c - Tanh and Sigmoid activation layers
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Tanh layer: y = tanh(x)
// ---------------------------------------------------------------------------

struct boat_tanh_layer_t {
    boat_tensor_t* cache_input; // Input from forward pass (for backward)
};

BOAT_API boat_tanh_layer_t* BOAT_CALL boat_tanh_layer_create(void) {
    boat_tanh_layer_t* layer =
        (boat_tanh_layer_t*)boat_malloc(sizeof(boat_tanh_layer_t), BOAT_DEVICE_CPU);
    if (!layer) return NULL;
    layer->cache_input = NULL;
    return layer;
}

BOAT_API void BOAT_CALL boat_tanh_layer_free(boat_tanh_layer_t* layer) {
    if (!layer) return;
    if (layer->cache_input) boat_tensor_unref(layer->cache_input);
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_tanh_layer_forward(boat_tanh_layer_t* layer,
                                                          const boat_tensor_t* input) {
    if (!layer || !input) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[TanhLayer] NULL input or layer\n");
        return NULL;
    }
    if (layer->cache_input) {
        boat_tensor_unref(layer->cache_input);
        layer->cache_input = NULL;
    }
    boat_tensor_t* result = boat_tanh(input);
    if (!result) return NULL;
    layer->cache_input = (boat_tensor_t*)input;
    boat_tensor_ref(layer->cache_input);
    return result;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_tanh_layer_backward(boat_tanh_layer_t* layer,
                                                           const boat_tensor_t* grad_output) {
    if (!layer || !grad_output || !layer->cache_input) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[TanhLayer] NULL input, gradient, or missing cache\n");
        return NULL;
    }
    // d/dx tanh(x) = 1 - tanh(x)^2
    boat_tensor_t* t = boat_tanh(layer->cache_input);
    if (!t) return NULL;
    boat_tensor_t* t2 = boat_mul(t, t);
    boat_tensor_free(t);
    if (!t2) return NULL;
    boat_tensor_t* d = boat_add_scalar(boat_mul_scalar(t2, -1.0), 1.0);
    boat_tensor_free(t2);
    if (!d) return NULL;
    boat_tensor_t* grad = boat_mul(d, grad_output);
    boat_tensor_free(d);
    return grad;
}

BOAT_API void BOAT_CALL boat_tanh_layer_update(boat_tanh_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
}

// ---------------------------------------------------------------------------
// Sigmoid layer: y = 1 / (1 + exp(-x))
// ---------------------------------------------------------------------------

struct boat_sigmoid_layer_t {
    boat_tensor_t* cache_input; // Input from forward pass (for backward)
};

BOAT_API boat_sigmoid_layer_t* BOAT_CALL boat_sigmoid_layer_create(void) {
    boat_sigmoid_layer_t* layer =
        (boat_sigmoid_layer_t*)boat_malloc(sizeof(boat_sigmoid_layer_t), BOAT_DEVICE_CPU);
    if (!layer) return NULL;
    layer->cache_input = NULL;
    return layer;
}

BOAT_API void BOAT_CALL boat_sigmoid_layer_free(boat_sigmoid_layer_t* layer) {
    if (!layer) return;
    if (layer->cache_input) boat_tensor_unref(layer->cache_input);
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_sigmoid_layer_forward(boat_sigmoid_layer_t* layer,
                                                             const boat_tensor_t* input) {
    if (!layer || !input) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[SigmoidLayer] NULL input or layer\n");
        return NULL;
    }
    if (layer->cache_input) {
        boat_tensor_unref(layer->cache_input);
        layer->cache_input = NULL;
    }
    boat_tensor_t* result = boat_sigmoid(input);
    if (!result) return NULL;
    layer->cache_input = (boat_tensor_t*)input;
    boat_tensor_ref(layer->cache_input);
    return result;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_sigmoid_layer_backward(boat_sigmoid_layer_t* layer,
                                                              const boat_tensor_t* grad_output) {
    if (!layer || !grad_output || !layer->cache_input) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[SigmoidLayer] NULL input, gradient, or missing cache\n");
        return NULL;
    }
    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    boat_tensor_t* s = boat_sigmoid(layer->cache_input);
    if (!s) return NULL;
    boat_tensor_t* d = boat_add_scalar(boat_mul_scalar(s, -1.0), 1.0);
    if (!d) {
        boat_tensor_free(s);
        return NULL;
    }
    boat_tensor_t* d2 = boat_mul(s, d);
    boat_tensor_free(s);
    boat_tensor_free(d);
    if (!d2) return NULL;
    boat_tensor_t* grad = boat_mul(d2, grad_output);
    boat_tensor_free(d2);
    return grad;
}

BOAT_API void BOAT_CALL boat_sigmoid_layer_update(boat_sigmoid_layer_t* layer,
                                                  float learning_rate) {
    (void)layer;
    (void)learning_rate;
}
