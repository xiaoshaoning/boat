// sgd.c - SGD optimizer with momentum and Nesterov support
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <stddef.h>
#include <boat.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// SGD optimizer state structure
typedef struct boat_sgd_state_t {
    boat_optimizer_type_t type;
    float learning_rate;
    float momentum;
    int use_nesterov;

    // Parameter and gradient arrays
    boat_tensor_t** params;
    boat_tensor_t** grads;

    // Velocity buffer for momentum
    boat_tensor_t** velocity;

    size_t num_params;
    size_t capacity;
} boat_sgd_state_t;

// Internal function declarations
static void sgd_expand_capacity(boat_sgd_state_t* state);
static void sgd_update_parameter(boat_sgd_state_t* state, size_t idx);

// Create SGD optimizer
BOAT_API boat_optimizer_t* boat_sgd_optimizer_create(float learning_rate,
                                            float momentum) {
    // Parameter validation
    if (learning_rate <= 0.0f) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[SGD] Learning rate must be positive\n");
        return NULL;
    }
    if (momentum < 0.0f) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[SGD] Momentum must be non-negative\n");
        return NULL;
    }

    // Allocate optimizer state
    boat_sgd_state_t* state = (boat_sgd_state_t*)boat_malloc(sizeof(boat_sgd_state_t), BOAT_DEVICE_CPU);
    if (!state) {
        boat_set_errorf(BOAT_ERROR_OUT_OF_MEMORY, "[SGD] Failed to allocate optimizer state\n");
        return NULL;
    }

    // Initialize state
    state->type = BOAT_OPTIMIZER_SGD;
    state->learning_rate = learning_rate;
    state->momentum = momentum;
    state->use_nesterov = 0;
    state->num_params = 0;
    state->capacity = 16;  // Initial capacity

    // Allocate arrays
    state->params = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    state->grads = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    state->velocity = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);

    if (!state->params || !state->grads || !state->velocity) {
        boat_set_errorf(BOAT_ERROR_OUT_OF_MEMORY, "[SGD] Failed to allocate optimizer arrays\n");
        if (state->params) boat_free(state->params);
        if (state->grads) boat_free(state->grads);
        if (state->velocity) boat_free(state->velocity);
        boat_free(state);
        return NULL;
    }

    // Initialize arrays to NULL
    for (size_t i = 0; i < state->capacity; i++) {
        state->params[i] = NULL;
        state->grads[i] = NULL;
        state->velocity[i] = NULL;
    }

    return (boat_optimizer_t*)state;
}

// Enable or disable Nesterov momentum
BOAT_API void boat_sgd_set_nesterov(boat_optimizer_t* optimizer, int use_nesterov) {
    if (!optimizer) return;
    boat_sgd_state_t* state = (boat_sgd_state_t*)optimizer;
    state->use_nesterov = use_nesterov ? 1 : 0;
}

// Add a parameter to the optimizer
void sgd_optimizer_add_parameter(boat_optimizer_t* optimizer,
                              boat_tensor_t* param,
                              boat_tensor_t* grad) {
    if (!optimizer || !param || !grad) {
        return;
    }

    boat_sgd_state_t* state = (boat_sgd_state_t*)optimizer;

    // Check if we need to expand capacity
    if (state->num_params >= state->capacity) {
        sgd_expand_capacity(state);
    }

    size_t idx = state->num_params;

    // Store parameter and gradient
    state->params[idx] = param;
    state->grads[idx] = grad;

    // Create velocity tensor with same shape as parameter
    const int64_t* shape = boat_tensor_shape(param);
    size_t ndim = boat_tensor_ndim(param);
    boat_dtype_t dtype = boat_tensor_dtype(param);

    state->velocity[idx] = boat_tensor_create(shape, ndim, dtype, BOAT_DEVICE_CPU);

    if (state->velocity[idx]) {
        // Initialize velocity to zero
        float* vel_data = (float*)boat_tensor_data(state->velocity[idx]);
        size_t num_elements = boat_tensor_nelements(state->velocity[idx]);

        for (size_t i = 0; i < num_elements; i++) {
            vel_data[i] = 0.0f;
        }
    }

    state->num_params++;
}

// Expand capacity of optimizer state arrays
static void sgd_expand_capacity(boat_sgd_state_t* state) {
    size_t new_capacity = state->capacity * 2;
    if (new_capacity == 0) {
        new_capacity = 16;
    }

    // Reallocate arrays
    boat_tensor_t** new_params = (boat_tensor_t**)boat_realloc(
        state->params, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    boat_tensor_t** new_grads = (boat_tensor_t**)boat_realloc(
        state->grads, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    boat_tensor_t** new_velocity = (boat_tensor_t**)boat_realloc(
        state->velocity, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);

    if (!new_params || !new_grads || !new_velocity) {
        // Free newly allocated arrays if any failed
        if (new_params != state->params) boat_free(new_params);
        if (new_grads != state->grads) boat_free(new_grads);
        if (new_velocity != state->velocity) boat_free(new_velocity);
        return;
    }

    // Update state
    state->params = new_params;
    state->grads = new_grads;
    state->velocity = new_velocity;

    // Initialize new entries to NULL
    for (size_t i = state->capacity; i < new_capacity; i++) {
        state->params[i] = NULL;
        state->grads[i] = NULL;
        state->velocity[i] = NULL;
    }

    state->capacity = new_capacity;
}

// Update a single parameter with momentum and optional Nesterov
static void sgd_update_parameter(boat_sgd_state_t* state, size_t idx) {
    if (idx >= state->num_params) {
        return;
    }

    const boat_tensor_t* param = state->params[idx];
    const boat_tensor_t* grad = state->grads[idx];
    const boat_tensor_t* vel_tensor = state->velocity[idx];

    if (!param || !grad || !vel_tensor) {
        return;
    }

    // Get data pointers
    float* param_data = (float*)boat_tensor_data(param);
    const float* grad_data = (const float*)boat_tensor_data(grad);
    float* vel_data = (float*)boat_tensor_data(vel_tensor);

    size_t num_elements = boat_tensor_nelements(param);

    float lr = state->learning_rate;
    float momentum = state->momentum;

    if (momentum > 0.0f) {
        if (state->use_nesterov) {
            // Nesterov momentum update:
            // v = momentum * v + g
            // param -= lr * (g + momentum * v)
            for (size_t i = 0; i < num_elements; i++) {
                float g = grad_data[i];
                float v_prev = vel_data[i];
                vel_data[i] = momentum * v_prev + g;
                param_data[i] -= lr * (g + momentum * vel_data[i]);
            }
        } else {
            // Standard momentum update:
            // v = momentum * v + g
            // param -= lr * v
            for (size_t i = 0; i < num_elements; i++) {
                float g = grad_data[i];
                vel_data[i] = momentum * vel_data[i] + g;
                param_data[i] -= lr * vel_data[i];
            }
        }
    } else {
        // Vanilla SGD (no momentum): param -= lr * g
        for (size_t i = 0; i < num_elements; i++) {
            param_data[i] -= lr * grad_data[i];
        }
    }
}

// Perform optimization step
void sgd_optimizer_step(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return;
    }

    boat_sgd_state_t* state = (boat_sgd_state_t*)optimizer;

    for (size_t i = 0; i < state->num_params; i++) {
        sgd_update_parameter(state, i);
    }
}

// Zero out all gradients
void sgd_optimizer_zero_grad(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return;
    }

    boat_sgd_state_t* state = (boat_sgd_state_t*)optimizer;

    for (size_t i = 0; i < state->num_params; i++) {
        const boat_tensor_t* grad = state->grads[i];
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
void sgd_optimizer_free(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return;
    }

    boat_sgd_state_t* state = (boat_sgd_state_t*)optimizer;

    // Free velocity tensors
    for (size_t i = 0; i < state->num_params; i++) {
        if (state->velocity[i]) {
            boat_tensor_unref(state->velocity[i]);
        }
    }

    // Free arrays
    boat_free(state->params);
    boat_free(state->grads);
    boat_free(state->velocity);

    // Free state
    boat_free(state);
}

// Get current learning rate from SGD optimizer
float sgd_optimizer_get_learning_rate(const boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return 0.0f;
    }
    const boat_sgd_state_t* state = (const boat_sgd_state_t*)optimizer;
    return state->learning_rate;
}

// Set learning rate for SGD optimizer
void sgd_optimizer_set_learning_rate(boat_optimizer_t* optimizer, float learning_rate) {
    if (!optimizer) {
        return;
    }
    boat_sgd_state_t* state = (boat_sgd_state_t*)optimizer;
    state->learning_rate = learning_rate;
}
