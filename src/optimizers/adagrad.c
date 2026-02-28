// adagrad.c - Adagrad optimizer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <stddef.h>
#include <boat/optimizers.h>
#define BOAT_DEVICE_T_DEFINED
#include <boat/tensor.h>
#include <boat/memory.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#endif

// Adagrad optimizer state structure
typedef struct boat_adagrad_state_t {
    boat_optimizer_type_t type;
    float learning_rate;
    float epsilon;

    // Parameter and gradient arrays
    boat_tensor_t** params;
    boat_tensor_t** grads;

    // Sum of squared gradients
    boat_tensor_t** sum_square_grad;

    size_t num_params;
    size_t capacity;
} boat_adagrad_state_t;

// Internal function declarations
static void adagrad_expand_capacity(boat_adagrad_state_t* state);
static void adagrad_update_parameter(boat_adagrad_state_t* state, size_t idx);

// Create Adagrad optimizer
BOAT_API boat_optimizer_t* boat_adagrad_optimizer_create(float learning_rate,
                                                float epsilon) {
    // Parameter validation
    if (learning_rate <= 0.0f) {
        return NULL;
    }
    if (epsilon <= 0.0f) {
        return NULL;
    }

    // Allocate optimizer state
    boat_adagrad_state_t* state = (boat_adagrad_state_t*)boat_malloc(sizeof(boat_adagrad_state_t), BOAT_DEVICE_CPU);
    if (!state) {
        return NULL;
    }

    // Initialize state
    state->type = BOAT_OPTIMIZER_ADAGRAD;
    state->learning_rate = learning_rate;
    state->epsilon = epsilon;
    state->num_params = 0;
    state->capacity = 16;  // Initial capacity

    // Allocate arrays
    state->params = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    state->grads = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    state->sum_square_grad = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);

    if (!state->params || !state->grads || !state->sum_square_grad) {
        if (state->params) boat_free(state->params);
        if (state->grads) boat_free(state->grads);
        if (state->sum_square_grad) boat_free(state->sum_square_grad);
        boat_free(state);
        return NULL;
    }

    // Initialize arrays to NULL
    for (size_t i = 0; i < state->capacity; i++) {
        state->params[i] = NULL;
        state->grads[i] = NULL;
        state->sum_square_grad[i] = NULL;
    }

    return (boat_optimizer_t*)state;
}

// Add a parameter to the optimizer
void adagrad_optimizer_add_parameter(boat_optimizer_t* optimizer,
                                  boat_tensor_t* param,
                                  boat_tensor_t* grad) {
    if (!optimizer || !param || !grad) {
        return;
    }

    boat_adagrad_state_t* state = (boat_adagrad_state_t*)optimizer;

    // Check if we need to expand capacity
    if (state->num_params >= state->capacity) {
        adagrad_expand_capacity(state);
    }

    size_t idx = state->num_params;

    // Store parameter and gradient
    state->params[idx] = param;
    state->grads[idx] = grad;

    // Create sum of squared gradients tensor with same shape as parameter
    const int64_t* shape = boat_tensor_shape(param);
    size_t ndim = boat_tensor_ndim(param);
    boat_dtype_t dtype = boat_tensor_dtype(param);

    state->sum_square_grad[idx] = boat_tensor_create(shape, ndim, dtype, BOAT_DEVICE_CPU);

    if (state->sum_square_grad[idx]) {
        // Initialize sum of squared gradients to zero
        float* sum_square_grad_data = (float*)boat_tensor_data(state->sum_square_grad[idx]);
        size_t num_elements = boat_tensor_nelements(state->sum_square_grad[idx]);

        for (size_t i = 0; i < num_elements; i++) {
            sum_square_grad_data[i] = 0.0f;
        }
    }

    state->num_params++;
}

// Expand capacity of optimizer state arrays
static void adagrad_expand_capacity(boat_adagrad_state_t* state) {
    size_t new_capacity = state->capacity * 2;
    if (new_capacity == 0) {
        new_capacity = 16;
    }

    // Reallocate arrays
    boat_tensor_t** new_params = (boat_tensor_t**)boat_realloc(
        state->params, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    boat_tensor_t** new_grads = (boat_tensor_t**)boat_realloc(
        state->grads, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    boat_tensor_t** new_sum_square_grad = (boat_tensor_t**)boat_realloc(
        state->sum_square_grad, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);

    if (!new_params || !new_grads || !new_sum_square_grad) {
        // Free newly allocated arrays if any failed
        if (new_params != state->params) boat_free(new_params);
        if (new_grads != state->grads) boat_free(new_grads);
        if (new_sum_square_grad != state->sum_square_grad) boat_free(new_sum_square_grad);
        return;
    }

    // Update state
    state->params = new_params;
    state->grads = new_grads;
    state->sum_square_grad = new_sum_square_grad;

    // Initialize new entries to NULL
    for (size_t i = state->capacity; i < new_capacity; i++) {
        state->params[i] = NULL;
        state->grads[i] = NULL;
        state->sum_square_grad[i] = NULL;
    }

    state->capacity = new_capacity;
}

// Update a single parameter
static void adagrad_update_parameter(boat_adagrad_state_t* state, size_t idx) {
    if (idx >= state->num_params) {
        return;
    }

    boat_tensor_t* param = state->params[idx];
    boat_tensor_t* grad = state->grads[idx];
    boat_tensor_t* sum_square_grad_tensor = state->sum_square_grad[idx];

    if (!param || !grad || !sum_square_grad_tensor) {
        return;
    }

    // Get data pointers
    float* param_data = (float*)boat_tensor_data(param);
    float* grad_data = (float*)boat_tensor_data(grad);
    float* sum_square_grad_data = (float*)boat_tensor_data(sum_square_grad_tensor);

    size_t num_elements = boat_tensor_nelements(param);

    // Update each element
    for (size_t i = 0; i < num_elements; i++) {
        float g = grad_data[i];

        // Accumulate squared gradient
        sum_square_grad_data[i] += g * g;

        // Update parameter: param -= learning_rate * g / sqrt(sum_square_grad + epsilon)
        param_data[i] -= state->learning_rate * g / (sqrtf(sum_square_grad_data[i]) + state->epsilon);
    }
}

// Perform optimization step
void adagrad_optimizer_step(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return;
    }

    boat_adagrad_state_t* state = (boat_adagrad_state_t*)optimizer;

    // Update all parameters
    for (size_t i = 0; i < state->num_params; i++) {
        adagrad_update_parameter(state, i);
    }
}

// Zero out all gradients
void adagrad_optimizer_zero_grad(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return;
    }

    boat_adagrad_state_t* state = (boat_adagrad_state_t*)optimizer;

    for (size_t i = 0; i < state->num_params; i++) {
        boat_tensor_t* grad = state->grads[i];
        if (!grad) {
            continue;
        }

        float* grad_data = (float*)boat_tensor_data(grad);
        size_t num_elements = boat_tensor_nelements(grad);

        for (size_t j = 0; j < num_elements; j++) {
            grad_data[j] = 0.0f;
        }
    }
}

// Free optimizer resources
void adagrad_optimizer_free(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return;
    }

    boat_adagrad_state_t* state = (boat_adagrad_state_t*)optimizer;

    // Free sum of squared gradients tensors
    for (size_t i = 0; i < state->num_params; i++) {
        if (state->sum_square_grad[i]) {
            boat_tensor_unref(state->sum_square_grad[i]);
        }
    }

    // Free arrays
    boat_free(state->params);
    boat_free(state->grads);
    boat_free(state->sum_square_grad);

    // Free state
    boat_free(state);
}

// Get current learning rate from Adagrad optimizer
float adagrad_optimizer_get_learning_rate(const boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return 0.0f;
    }
    const boat_adagrad_state_t* state = (const boat_adagrad_state_t*)optimizer;
    return state->learning_rate;
}

// Set learning rate for Adagrad optimizer
void adagrad_optimizer_set_learning_rate(boat_optimizer_t* optimizer, float learning_rate) {
    if (!optimizer) {
        return;
    }
    boat_adagrad_state_t* state = (boat_adagrad_state_t*)optimizer;
    state->learning_rate = learning_rate;
}