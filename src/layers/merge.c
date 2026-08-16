// merge.c - Merge layers: concatenation and element-wise addition (multi-input)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Concatenation layer: joins N inputs along `dim`.
// ---------------------------------------------------------------------------

struct boat_concat_layer_t {
    int64_t dim; // 0-based join axis; negative counts from the last dim
};

BOAT_API boat_concat_layer_t* BOAT_CALL boat_concat_layer_create(int64_t dim) {
    boat_concat_layer_t* layer =
        (boat_concat_layer_t*)boat_malloc(sizeof(boat_concat_layer_t), BOAT_DEVICE_CPU);
    if (!layer) return NULL;
    layer->dim = dim;
    return layer;
}

BOAT_API void BOAT_CALL boat_concat_layer_free(boat_concat_layer_t* layer) {
    if (!layer) return;
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_concat_layer_forward_many(boat_concat_layer_t* layer,
                                                                 const boat_layer_input_t* inputs,
                                                                 size_t n_inputs) {
    if (!layer || n_inputs == 0 || !inputs) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[ConcatLayer] NULL input or layer\n");
        return NULL;
    }
    const boat_tensor_t* first = inputs[0].t;
    if (!first) return NULL;

    // Normalize the join axis (negative dims count from the last dimension).
    size_t ndim = boat_tensor_ndim(first);
    int64_t dim = layer->dim;
    if (dim < 0) dim += (int64_t)ndim;
    if (dim < 0 || (size_t)dim >= ndim) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ConcatLayer] dim %lld out of range for a %zu-D tensor\n",
                        (long long)layer->dim, ndim);
        return NULL;
    }

    // A single input is an identity join.
    if (n_inputs == 1) return boat_tensor_clone(first);

    const boat_tensor_t** ts =
        (const boat_tensor_t**)boat_malloc(n_inputs * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    if (!ts) return NULL;
    for (size_t i = 0; i < n_inputs; i++)
        ts[i] = inputs[i].t;

    boat_tensor_t* out = boat_tensor_concatenate(ts, n_inputs, (size_t)dim);
    boat_free(ts);
    return out;
}

// ---------------------------------------------------------------------------
// Addition layer: element-wise sum of N inputs (broadcasting).
// ---------------------------------------------------------------------------

struct boat_add_layer_t {
    int placeholder; // stateless
};

BOAT_API boat_add_layer_t* BOAT_CALL boat_add_layer_create(void) {
    boat_add_layer_t* layer =
        (boat_add_layer_t*)boat_malloc(sizeof(boat_add_layer_t), BOAT_DEVICE_CPU);
    if (!layer) return NULL;
    layer->placeholder = 0;
    return layer;
}

BOAT_API void BOAT_CALL boat_add_layer_free(boat_add_layer_t* layer) {
    if (!layer) return;
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_add_layer_forward_many(boat_add_layer_t* layer,
                                                              const boat_layer_input_t* inputs,
                                                              size_t n_inputs) {
    if (!layer || n_inputs == 0 || !inputs) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[AddLayer] NULL input or layer\n");
        return NULL;
    }
    // A single input is an identity sum.
    if (n_inputs == 1) return boat_tensor_clone(inputs[0].t);

    boat_tensor_t* acc = boat_add(inputs[0].t, inputs[1].t);
    for (size_t i = 2; i < n_inputs && acc; i++) {
        boat_tensor_t* next = boat_add(acc, inputs[i].t);
        boat_tensor_free(acc);
        acc = next;
    }
    return acc;
}
