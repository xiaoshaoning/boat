// adam.c - Adam optimizer implementation
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

#ifdef BOAT_WITH_CUDA
#include <boat/cuda_runtime.h>
#endif
#include <boat.h>

// Adam optimizer state structure
typedef struct boat_adam_state_t {
    boat_optimizer_type_t type;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int64_t timestep;

    // Parameter and gradient arrays
    boat_tensor_t** params;
    boat_tensor_t** grads;

    // First moment estimates
    boat_tensor_t** m;
    // Second moment estimates
    boat_tensor_t** v;

    size_t num_params;
    size_t capacity;
} boat_adam_state_t;

// Internal function declarations
static void adam_expand_capacity(boat_adam_state_t* state);
static void adam_update_parameter(boat_adam_state_t* state, size_t idx);

// Create Adam optimizer
BOAT_API boat_optimizer_t* boat_adam_optimizer_create(float learning_rate,
                                             float beta1, float beta2,
                                             float epsilon) {
    // Validate hyperparameters
    if (learning_rate <= 0.0f) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Adam] Learning rate must be positive\n");
        return NULL;
    }
    if (beta1 <= 0.0f || beta1 >= 1.0f) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Adam] beta1 must be in [0, 1)\n");
        return NULL;
    }
    if (beta2 <= 0.0f || beta2 >= 1.0f) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Adam] beta2 must be in [0, 1)\n");
        return NULL;
    }

    // Allocate optimizer state
    boat_adam_state_t* state = (boat_adam_state_t*)boat_malloc(sizeof(boat_adam_state_t), BOAT_DEVICE_CPU);
    if (!state) {
        boat_set_errorf(BOAT_ERROR_OUT_OF_MEMORY, "[Adam] Failed to allocate optimizer state\n");
        return NULL;
    }

    // Initialize state
    state->type = BOAT_OPTIMIZER_ADAM;
    state->learning_rate = learning_rate;
    state->beta1 = beta1;
    state->beta2 = beta2;
    state->epsilon = epsilon;
    state->timestep = 0;
    state->num_params = 0;
    state->capacity = 16;  // Initial capacity

    // Allocate arrays
    state->params = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    state->grads = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    state->m = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    state->v = (boat_tensor_t**)boat_malloc(state->capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);

    if (!state->params || !state->grads || !state->m || !state->v) {
        boat_set_errorf(BOAT_ERROR_OUT_OF_MEMORY, "[Adam] Failed to allocate optimizer state\n");
        if (state->params) boat_free(state->params);
        if (state->grads) boat_free(state->grads);
        if (state->m) boat_free(state->m);
        if (state->v) boat_free(state->v);
        boat_free(state);
        return NULL;
    }

    // Initialize arrays to NULL
    memset(state->params, 0, state->capacity * sizeof(boat_tensor_t*));
    memset(state->grads, 0, state->capacity * sizeof(boat_tensor_t*));
    memset(state->m, 0, state->capacity * sizeof(boat_tensor_t*));
    memset(state->v, 0, state->capacity * sizeof(boat_tensor_t*));

    return (boat_optimizer_t*)state;
}

// Add a parameter to the optimizer
void adam_optimizer_add_parameter(boat_optimizer_t* optimizer,
                                  boat_tensor_t* param,
                                  boat_tensor_t* grad) {
    if (!optimizer || !param || !grad) {
        return;
    }

    boat_adam_state_t* state = (boat_adam_state_t*)optimizer;

    // Check if we need to expand capacity
    if (state->num_params >= state->capacity) {
        adam_expand_capacity(state);
    }

    size_t idx = state->num_params;

    // Store parameter and gradient
    state->params[idx] = param;
    state->grads[idx] = grad;

    // Create moment estimate tensors with same shape and device as parameter
    const int64_t* shape = boat_tensor_shape(param);
    size_t ndim = boat_tensor_ndim(param);
    boat_dtype_t dtype = boat_tensor_dtype(param);
    boat_device_t device = boat_tensor_device(param);

    state->m[idx] = boat_tensor_create(shape, ndim, dtype, device);
    state->v[idx] = boat_tensor_create(shape, ndim, dtype, device);

    if (state->m[idx] && state->v[idx]) {
        // Initialize moment estimates to zero (device-aware)
        float* m_data = (float*)boat_tensor_data(state->m[idx]);
        size_t m_nbytes = boat_tensor_nbytes(state->m[idx]);
        boat_memory_set(m_data, 0, m_nbytes, device);
        float* v_data = (float*)boat_tensor_data(state->v[idx]);
        size_t v_nbytes = boat_tensor_nbytes(state->v[idx]);
        boat_memory_set(v_data, 0, v_nbytes, device);
    }

    state->num_params++;
}

// Expand capacity of optimizer state arrays
static void adam_expand_capacity(boat_adam_state_t* state) {
    size_t new_capacity = state->capacity * 2;

    // Reallocate arrays
    boat_tensor_t** new_params = (boat_tensor_t**)boat_realloc(
        state->params, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    boat_tensor_t** new_grads = (boat_tensor_t**)boat_realloc(
        state->grads, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    boat_tensor_t** new_m = (boat_tensor_t**)boat_realloc(
        state->m, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    boat_tensor_t** new_v = (boat_tensor_t**)boat_realloc(
        state->v, new_capacity * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);

    if (!new_params || !new_grads || !new_m || !new_v) {
        // Free newly allocated arrays if any failed
        if (new_params != state->params) boat_free(new_params);
        if (new_grads != state->grads) boat_free(new_grads);
        if (new_m != state->m) boat_free(new_m);
        if (new_v != state->v) boat_free(new_v);
        return;
    }

    // Update state
    state->params = new_params;
    state->grads = new_grads;
    state->m = new_m;
    state->v = new_v;

    // Initialize new entries to NULL
    for (size_t i = state->capacity; i < new_capacity; i++) {
        state->params[i] = NULL;
        state->grads[i] = NULL;
        state->m[i] = NULL;
        state->v[i] = NULL;
    }

    state->capacity = new_capacity;
}

// Update a single parameter
static void adam_update_parameter(boat_adam_state_t* state, size_t idx) {
    if (idx >= state->num_params) {
        return;
    }

    const boat_tensor_t* param = state->params[idx];
    const boat_tensor_t* grad = state->grads[idx];
    const boat_tensor_t* m_tensor = state->m[idx];
    const boat_tensor_t* v_tensor = state->v[idx];

    if (!param || !grad || !m_tensor || !v_tensor) {
        return;
    }

    // Get data pointers
    float* param_data = (float*)boat_tensor_data(param);
    const float* grad_data = (const float*)boat_tensor_data(grad);
    float* m_data = (float*)boat_tensor_data(m_tensor);
    float* v_data = (float*)boat_tensor_data(v_tensor);

    size_t num_elements = boat_tensor_nelements(param);

    // Precompute bias correction factors
    float beta1_pow_t = powf(state->beta1, (float)(state->timestep + 1));
    float beta2_pow_t = powf(state->beta2, (float)(state->timestep + 1));

#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(param) == BOAT_DEVICE_CUDA) {
        boat_cuda_adam_update_f32(param_data, grad_data, m_data, v_data,
                                   lr, state->beta1, state->beta2,
                                   beta1_pow_t, beta2_pow_t, state->epsilon,
                                   num_elements);
        return;
    }
#endif

    // Update each element (CPU)
    for (size_t i = 0; i < num_elements; i++) {
        float g = grad_data[i];

        // Update biased first moment estimate
        m_data[i] = state->beta1 * m_data[i] + (1.0f - state->beta1) * g;

        // Update biased second raw moment estimate
        v_data[i] = state->beta2 * v_data[i] + (1.0f - state->beta2) * g * g;

        // Compute bias-corrected moment estimates
        float m_hat = m_data[i] / (1.0f - beta1_pow_t);
        float v_hat = v_data[i] / (1.0f - beta2_pow_t);

        // Update parameter
        param_data[i] -= state->learning_rate * m_hat / (sqrtf(v_hat) + state->epsilon);
    }
}

// Perform optimization step
void adam_optimizer_step(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return;
    }

    boat_adam_state_t* state = (boat_adam_state_t*)optimizer;

    // Update all parameters
    for (size_t i = 0; i < state->num_params; i++) {
        adam_update_parameter(state, i);
    }

    // Increment timestep
    state->timestep++;
}

// Zero out all gradients
void adam_optimizer_zero_grad(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return;
    }

    boat_adam_state_t* state = (boat_adam_state_t*)optimizer;

    for (size_t i = 0; i < state->num_params; i++) {
        const boat_tensor_t* grad = state->grads[i];
        if (!grad) {
            continue;
        }

        float* grad_data = (float*)boat_tensor_data(grad);
        size_t nbytes = boat_tensor_nbytes(grad);
        boat_device_t device = boat_tensor_device(grad);
        boat_memory_set(grad_data, 0, nbytes, device);
    }
}

// Free optimizer resources
void adam_optimizer_free(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return;
    }

    boat_adam_state_t* state = (boat_adam_state_t*)optimizer;

    // Free moment estimate tensors
    for (size_t i = 0; i < state->num_params; i++) {
        if (state->m[i]) {
            boat_tensor_unref(state->m[i]);
        }
        if (state->v[i]) {
            boat_tensor_unref(state->v[i]);
        }
    }

    // Free arrays
    boat_free(state->params);
    boat_free(state->grads);
    boat_free(state->m);
    boat_free(state->v);

    // Free state
    boat_free(state);
}

// Get current learning rate from Adam optimizer
float adam_optimizer_get_learning_rate(const boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return 0.0f;
    }
    const boat_adam_state_t* state = (const boat_adam_state_t*)optimizer;
    return state->learning_rate;
}

// Set learning rate for Adam optimizer
void adam_optimizer_set_learning_rate(boat_optimizer_t* optimizer, float learning_rate) {
    if (!optimizer) {
        return;
    }
    boat_adam_state_t* state = (boat_adam_state_t*)optimizer;
    state->learning_rate = learning_rate;
}