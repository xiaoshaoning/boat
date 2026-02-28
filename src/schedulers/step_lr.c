// step_lr.c - Step learning rate scheduler implementation
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

// Create StepLR scheduler
BOAT_API boat_scheduler_t* boat_step_lr_scheduler_create(float base_learning_rate,
                                                         int step_size,
                                                         float gamma) {
    // Validate hyperparameters
    if (base_learning_rate <= 0.0f) {
        return NULL;
    }
    if (step_size <= 0) {
        return NULL;
    }
    if (gamma <= 0.0f) {
        return NULL;
    }

    // Allocate scheduler state
    boat_step_lr_state_t* state = (boat_step_lr_state_t*)boat_malloc(sizeof(boat_step_lr_state_t), BOAT_DEVICE_CPU);
    if (!state) {
        return NULL;
    }

    // Initialize state
    state->header.type = BOAT_SCHEDULER_STEP_LR;
    state->header.base_learning_rate = base_learning_rate;
    state->header.last_learning_rate = base_learning_rate;
    state->header.step_count = 0;
    state->step_size = step_size;
    state->gamma = gamma;

    return (boat_scheduler_t*)state;
}

// StepLR scheduler step function
void step_lr_scheduler_step(boat_scheduler_t* scheduler) {
    if (!scheduler) return;

    boat_step_lr_state_t* state = (boat_step_lr_state_t*)scheduler;

    // Increment step count
    state->header.step_count++;

    // Calculate new learning rate: lr = base_lr * gamma^floor(step / step_size)
    int steps = state->header.step_count - 1; // 0-indexed for calculation
    int decay_steps = steps / state->step_size;
    float new_lr = state->header.base_learning_rate * powf(state->gamma, (float)decay_steps);

    // Update last learning rate
    state->header.last_learning_rate = new_lr;
}

// StepLR scheduler reset function
void step_lr_scheduler_reset(boat_scheduler_t* scheduler) {
    if (!scheduler) return;

    boat_step_lr_state_t* state = (boat_step_lr_state_t*)scheduler;

    // Reset step count and learning rate
    state->header.step_count = 0;
    state->header.last_learning_rate = state->header.base_learning_rate;
}

// StepLR scheduler get last learning rate function
float step_lr_scheduler_get_last_lr(const boat_scheduler_t* scheduler) {
    if (!scheduler) return 0.0f;

    const boat_step_lr_state_t* state = (const boat_step_lr_state_t*)scheduler;
    return state->header.last_learning_rate;
}

// StepLR scheduler get next learning rate function
float step_lr_scheduler_get_next_lr(const boat_scheduler_t* scheduler) {
    if (!scheduler) return 0.0f;

    const boat_step_lr_state_t* state = (const boat_step_lr_state_t*)scheduler;

    // Calculate what the learning rate will be after the next step
    int steps = state->header.step_count; // Current step count (0-indexed for next step)
    int decay_steps = steps / state->step_size;
    float next_lr = state->header.base_learning_rate * powf(state->gamma, (float)decay_steps);

    return next_lr;
}

// StepLR scheduler free function
void step_lr_scheduler_free(boat_scheduler_t* scheduler) {
    if (!scheduler) return;

    // State was allocated as a single block, just free it
    boat_free(scheduler);
}