// dropout.c - Dropout layer (inverted dropout)
//
// Training mode: each element is kept with probability (1 - p) and scaled by
// 1/(1 - p) (inverted dropout, so the expected activation is unchanged).
// The mask is cached for the backward pass. Inference mode: identity.
//
// The mask uses rand() (like the dense/conv initializers); callers wanting
// reproducible runs should srand() first.
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <stdlib.h>
#include <string.h>

struct boat_dropout_layer_t {
    float probability;       /* drop probability in [0, 1] */
    bool training;           /* training mode: mask + scale */
    boat_tensor_t* cache_mask; /* per-element mask for the backward pass */
};

BOAT_API boat_dropout_layer_t* BOAT_CALL boat_dropout_layer_create(float probability) {
    boat_dropout_layer_t* layer =
        (boat_dropout_layer_t*)boat_malloc(sizeof(boat_dropout_layer_t), BOAT_DEVICE_CPU);
    if (!layer) return NULL;
    layer->probability = probability < 0.0f ? 0.0f : (probability > 1.0f ? 1.0f : probability);
    layer->training = false;
    layer->cache_mask = NULL;
    return layer;
}

BOAT_API void BOAT_CALL boat_dropout_layer_free(boat_dropout_layer_t* layer) {
    if (!layer) return;
    if (layer->cache_mask) boat_tensor_free(layer->cache_mask);
    boat_free(layer);
}

BOAT_API void BOAT_CALL boat_dropout_layer_set_training(boat_dropout_layer_t* layer,
                                                        bool training) {
    if (layer) layer->training = training;
}

BOAT_API bool BOAT_CALL boat_dropout_layer_get_training(const boat_dropout_layer_t* layer) {
    return layer ? layer->training : false;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_dropout_layer_forward(const boat_dropout_layer_t* layer,
                                                             const boat_tensor_t* input) {
    if (!layer || !input) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[DropoutLayer] NULL input or layer\n");
        return NULL;
    }
    /* Inference (or disabled): identity. */
    if (!layer->training || layer->probability <= 0.0f) return boat_tensor_clone(input);

    boat_tensor_t* out = boat_tensor_clone(input);
    if (!out) return NULL;
    float keep = 1.0f - layer->probability;
    float scale = 1.0f / keep;
    size_t ne = boat_tensor_nelements(out);
    float* od = (float*)boat_tensor_data(out);

    boat_tensor_t* mask = boat_tensor_create_like(input);
    if (!mask) {
        boat_tensor_free(out);
        return NULL;
    }
    float* md = (float*)boat_tensor_data(mask);
    for (size_t i = 0; i < ne; i++) {
        float m = ((float)rand() / (float)RAND_MAX) < keep ? 1.0f : 0.0f;
        md[i] = m;
        od[i] *= m * scale;
    }

    boat_dropout_layer_t* self = (boat_dropout_layer_t*)layer;
    if (self->cache_mask) boat_tensor_free(self->cache_mask);
    self->cache_mask = mask;
    return out;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_dropout_layer_backward(boat_dropout_layer_t* layer,
                                                              const boat_tensor_t* grad_output) {
    if (!layer || !grad_output) return NULL;
    if (!layer->training || layer->probability <= 0.0f || !layer->cache_mask)
        return boat_tensor_clone(grad_output);

    boat_tensor_t* grad_input = boat_tensor_clone(grad_output);
    if (!grad_input) return NULL;
    float keep = 1.0f - layer->probability;
    float scale = 1.0f / keep;
    size_t ne = boat_tensor_nelements(grad_input);
    float* gd = (float*)boat_tensor_data(grad_input);
    const float* md = (const float*)boat_tensor_data(layer->cache_mask);
    for (size_t i = 0; i < ne; i++) gd[i] *= md[i] * scale;
    return grad_input;
}
