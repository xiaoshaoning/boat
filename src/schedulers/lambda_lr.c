// lambda_lr.c - Lambda learning rate scheduler implementation
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <stdlib.h>
#include <math.h>
#include <string.h>

// Undefine potentially conflicting Windows macros
#ifdef device
#undef device
#endif

#include <boat/schedulers.h>
#include <boat/memory.h>
#include "schedulers_internal.h"

// Create LambdaLR scheduler
BOAT_API boat_scheduler_t* boat_lambda_lr_scheduler_create(float base_learning_rate,
                                                           float (*lambda_fn)(int step, float base_lr)) {
    // Validate hyperparameters
    if (base_learning_rate <= 0.0f) {
        return NULL;
    }
    if (!lambda_fn) {
        return NULL;
    }

    // Allocate scheduler state
    boat_lambda_lr_state_t* state = (boat_lambda_lr_state_t*)boat_malloc(sizeof(boat_lambda_lr_state_t), BOAT_DEVICE_CPU);
    if (!state) {
        return NULL;
    }

    // Initialize state
    state->header.type = BOAT_SCHEDULER_LAMBDA_LR;
    state->header.base_learning_rate = base_learning_rate;
    state->header.last_learning_rate = base_learning_rate;
    state->header.step_count = 0;
    state->lambda_fn = lambda_fn;

    return (boat_scheduler_t*)state;
}

// LambdaLR scheduler step function
void lambda_lr_scheduler_step(boat_scheduler_t* scheduler) {
    if (!scheduler) return;

    boat_lambda_lr_state_t* state = (boat_lambda_lr_state_t*)scheduler;

    // Increment step count
    state->header.step_count++;

    // Calculate new learning rate using custom lambda function
    // Note: step_count is 1-indexed for the lambda function
    float new_lr = state->lambda_fn(state->header.step_count, state->header.base_learning_rate);

    // Ensure learning rate is valid (non-negative)
    if (new_lr < 0.0f) {
        new_lr = 0.0f;
    }

    // Update last learning rate
    state->header.last_learning_rate = new_lr;
}

// LambdaLR scheduler reset function
void lambda_lr_scheduler_reset(boat_scheduler_t* scheduler) {
    if (!scheduler) return;

    boat_lambda_lr_state_t* state = (boat_lambda_lr_state_t*)scheduler;

    // Reset step count and learning rate
    state->header.step_count = 0;
    state->header.last_learning_rate = state->header.base_learning_rate;
}

// LambdaLR scheduler get last learning rate function
float lambda_lr_scheduler_get_last_lr(const boat_scheduler_t* scheduler) {
    if (!scheduler) return 0.0f;

    const boat_lambda_lr_state_t* state = (const boat_lambda_lr_state_t*)scheduler;
    return state->header.last_learning_rate;
}

// LambdaLR scheduler get next learning rate function
float lambda_lr_scheduler_get_next_lr(const boat_scheduler_t* scheduler) {
    if (!scheduler) return 0.0f;

    const boat_lambda_lr_state_t* state = (const boat_lambda_lr_state_t*)scheduler;

    // Calculate what the learning rate will be after the next step
    int next_step = state->header.step_count + 1;
    float next_lr = state->lambda_fn(next_step, state->header.base_learning_rate);

    // Ensure learning rate is valid (non-negative)
    if (next_lr < 0.0f) {
        next_lr = 0.0f;
    }

    return next_lr;
}

// LambdaLR scheduler free function
void lambda_lr_scheduler_free(boat_scheduler_t* scheduler) {
    if (!scheduler) return;

    // State was allocated as a single block, just free it
    boat_free(scheduler);
}