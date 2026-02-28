// rmsprop.c - RMSprop optimizer implementation
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

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

// RMSprop optimizer state structure
typedef struct boat_rmsprop_state_t {
    boat_optimizer_type_t type;
    float learning_rate;
    float alpha;
    float epsilon;

    // Parameter and gradient arrays
    boat_tensor_t** params;
    boat_tensor_t** grads;

    // Running average of squared gradients
    boat_tensor_t** square_avg;

    size_t num_params;
    size_t capacity;
} boat_rmsprop_state_t;

// Internal function declarations
static void rmsprop_expand_capacity(boat_rmsprop_state_t* state);
static void rmsprop_update_parameter(boat_rmsprop_state_t* state, size_t idx);

// Create RMSprop optimizer
BOAT_API boat_optimizer_t* boat_rmsprop_optimizer_create(float learning_rate,
                                                float alpha,
                                                float epsilon) {
    // Debug
    printf("RMSprop create called: lr=%f, alpha=%f, eps=%f\n", learning_rate, alpha, epsilon);
    fprintf(stderr, "RMSprop create: lr=%f, alpha=%f, eps=%f\n", learning_rate, alpha, epsilon);
    fprintf(stderr, "BOAT_DEVICE_CPU = %d\n", BOAT_DEVICE_CPU);
    fprintf(stderr, "sizeof(boat_rmsprop_state_t) = %zu\n", sizeof(boat_rmsprop_state_t));

    // Parameter validation
    if (learning_rate <= 0.0f) {
        fprintf(stderr, "Validation failed: learning_rate <= 0\n");
        return NULL;
    }
    if (alpha <= 0.0f || alpha >= 1.0f) {
        fprintf(stderr, "Validation failed: alpha=%f not in (0,1)\n", alpha);
        return NULL;
    }
    if (epsilon <= 0.0f) {
        fprintf(stderr, "Validation failed: epsilon <= 0\n");
        return NULL;
    }

    // Allocate optimizer state
    boat_rmsprop_state_t* state = (boat_rmsprop_state_t*)boat_malloc(sizeof(boat_rmsprop_state_t), BOAT_DEVICE_CPU);
    fprintf(stderr, "boat_malloc returned: %p\n", state);
    if (!state) {
        fprintf(stderr, "boat_malloc failed\n");
        return NULL;
    }

    // Initialize state
    state->type = BOAT_OPTIMIZER_RMSPROP;
    state->learning_rate = learning_rate;
    state->alpha = alpha;
    state->epsilon = epsilon;
    state->num_params = 0;
    state->capacity = 16;  // Initial capacity

    // Allocate arrays
    state->params = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    state->grads = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    state->square_avg = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);

    if (!state->params || !state->grads || !state->square_avg) {
        if (state->params) boat_free(state->params);
        if (state->grads) boat_free(state->grads);
        if (state->square_avg) boat_free(state->square_avg);
        boat_free(state);
        return NULL;
    }

    // Initialize arrays to NULL
    for (size_t i = 0; i < state->capacity; i++) {
        state->params[i] = NULL;
        state->grads[i] = NULL;
        state->square_avg[i] = NULL;
    }

    return (boat_optimizer_t*)state;
}

// Add a parameter to the optimizer
void rmsprop_optimizer_add_parameter(boat_optimizer_t* optimizer,
                                  boat_tensor_t* param,
                                  boat_tensor_t* grad) {
    if (!optimizer || !param || !grad) {
        return;
    }

    boat_rmsprop_state_t* state = (boat_rmsprop_state_t*)optimizer;

    // Check if we need to expand capacity
    if (state->num_params >= state->capacity) {
        rmsprop_expand_capacity(state);
    }

    size_t idx = state->num_params;

    // Store parameter and gradient
    state->params[idx] = param;
    state->grads[idx] = grad;

    // Create squared average tensor with same shape as parameter
    const int64_t* shape = boat_tensor_shape(param);
    size_t ndim = boat_tensor_ndim(param);
    boat_dtype_t dtype = boat_tensor_dtype(param);

    state->square_avg[idx] = boat_tensor_create(shape, ndim, dtype, BOAT_DEVICE_CPU);

    if (state->square_avg[idx]) {
        // Initialize squared average to zero
        float* square_avg_data = (float*)boat_tensor_data(state->square_avg[idx]);
        size_t num_elements = boat_tensor_nelements(state->square_avg[idx]);

        for (size_t i = 0; i < num_elements; i++) {
            square_avg_data[i] = 0.0f;
        }
    }

    state->num_params++;
}

// Expand capacity of optimizer state arrays
static void rmsprop_expand_capacity(boat_rmsprop_state_t* state) {
    size_t new_capacity = state->capacity * 2;

    // Reallocate arrays
    boat_tensor_t** new_params = (boat_tensor_t**)boat_realloc(
        state->params, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    boat_tensor_t** new_grads = (boat_tensor_t**)boat_realloc(
        state->grads, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    boat_tensor_t** new_square_avg = (boat_tensor_t**)boat_realloc(
        state->square_avg, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);

    if (!new_params || !new_grads || !new_square_avg) {
        // Free newly allocated arrays if any failed
        if (new_params != state->params) boat_free(new_params);
        if (new_grads != state->grads) boat_free(new_grads);
        if (new_square_avg != state->square_avg) boat_free(new_square_avg);
        return;
    }

    // Update state
    state->params = new_params;
    state->grads = new_grads;
    state->square_avg = new_square_avg;

    // Initialize new entries to NULL
    for (size_t i = state->capacity; i < new_capacity; i++) {
        state->params[i] = NULL;
        state->grads[i] = NULL;
        state->square_avg[i] = NULL;
    }

    state->capacity = new_capacity;
}

// Update a single parameter
static void rmsprop_update_parameter(boat_rmsprop_state_t* state, size_t idx) {
    if (idx >= state->num_params) {
        return;
    }

    boat_tensor_t* param = state->params[idx];
    boat_tensor_t* grad = state->grads[idx];
    boat_tensor_t* square_avg_tensor = state->square_avg[idx];

    if (!param || !grad || !square_avg_tensor) {
        return;
    }

    // Get data pointers
    float* param_data = (float*)boat_tensor_data(param);
    float* grad_data = (float*)boat_tensor_data(grad);
    float* square_avg_data = (float*)boat_tensor_data(square_avg_tensor);

    size_t num_elements = boat_tensor_nelements(param);

    // Update each element
    for (size_t i = 0; i < num_elements; i++) {
        float g = grad_data[i];

        // Update running average of squared gradients
        square_avg_data[i] = state->alpha * square_avg_data[i] + (1.0f - state->alpha) * g * g;

        // Update parameter
        param_data[i] -= state->learning_rate * g / (sqrtf(square_avg_data[i]) + state->epsilon);
    }
}

// Perform optimization step
void rmsprop_optimizer_step(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return;
    }

    boat_rmsprop_state_t* state = (boat_rmsprop_state_t*)optimizer;

    // Update all parameters
    for (size_t i = 0; i < state->num_params; i++) {
        rmsprop_update_parameter(state, i);
    }
}

// Zero out all gradients
void rmsprop_optimizer_zero_grad(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return;
    }

    boat_rmsprop_state_t* state = (boat_rmsprop_state_t*)optimizer;

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
void rmsprop_optimizer_free(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return;
    }

    boat_rmsprop_state_t* state = (boat_rmsprop_state_t*)optimizer;

    // Free squared average tensors
    for (size_t i = 0; i < state->num_params; i++) {
        if (state->square_avg[i]) {
            boat_tensor_unref(state->square_avg[i]);
        }
    }

    // Free arrays
    boat_free(state->params);
    boat_free(state->grads);
    boat_free(state->square_avg);

    // Free state
    boat_free(state);
}

// Get current learning rate from RMSprop optimizer
float rmsprop_optimizer_get_learning_rate(const boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return 0.0f;
    }
    const boat_rmsprop_state_t* state = (const boat_rmsprop_state_t*)optimizer;
    return state->learning_rate;
}

// Set learning rate for RMSprop optimizer
void rmsprop_optimizer_set_learning_rate(boat_optimizer_t* optimizer, float learning_rate) {
    if (!optimizer) {
        return;
    }
    boat_rmsprop_state_t* state = (boat_rmsprop_state_t*)optimizer;
    state->learning_rate = learning_rate;
}