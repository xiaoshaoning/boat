// cosine_annealing.c - Cosine annealing learning rate scheduler implementation
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

// Create CosineAnnealing scheduler
BOAT_API boat_scheduler_t* boat_cosine_annealing_scheduler_create(float base_learning_rate,
                                                                  int T_max,
                                                                  float eta_min) {
    // Validate hyperparameters
    if (base_learning_rate <= 0.0f) {
        return NULL;
    }
    if (T_max <= 0) {
        return NULL;
    }
    if (eta_min < 0.0f) {
        return NULL;
    }
    if (eta_min >= base_learning_rate) {
        // eta_min should be less than base_learning_rate
        return NULL;
    }

    // Allocate scheduler state
    boat_cosine_annealing_state_t* state = (boat_cosine_annealing_state_t*)boat_malloc(sizeof(boat_cosine_annealing_state_t), BOAT_DEVICE_CPU);
    if (!state) {
        return NULL;
    }

    // Initialize state
    state->header.type = BOAT_SCHEDULER_COSINE_ANNEALING;
    state->header.base_learning_rate = base_learning_rate;
    state->header.last_learning_rate = base_learning_rate;
    state->header.step_count = 0;
    state->T_max = T_max;
    state->eta_min = eta_min;

    return (boat_scheduler_t*)state;
}

// CosineAnnealing scheduler step function
void cosine_annealing_scheduler_step(boat_scheduler_t* scheduler) {
    if (!scheduler) return;

    boat_cosine_annealing_state_t* state = (boat_cosine_annealing_state_t*)scheduler;

    // Increment step count
    state->header.step_count++;

    // Calculate new learning rate using cosine annealing formula:
    // lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(π * step / T_max))
    // Note: step_count is 1-indexed for calculation
    float step = (float)state->header.step_count;
    float T_max_f = (float)state->T_max;

    // Clamp step to T_max to handle restart behavior
    if (step > T_max_f) {
        step = T_max_f;
    }

    float cos_factor = cosf(3.14159265358979323846f * step / T_max_f);
    float new_lr = state->eta_min + 0.5f * (state->header.base_learning_rate - state->eta_min) * (1.0f + cos_factor);

    // Update last learning rate
    state->header.last_learning_rate = new_lr;
}

// CosineAnnealing scheduler reset function
void cosine_annealing_scheduler_reset(boat_scheduler_t* scheduler) {
    if (!scheduler) return;

    boat_cosine_annealing_state_t* state = (boat_cosine_annealing_state_t*)scheduler;

    // Reset step count and learning rate
    state->header.step_count = 0;
    state->header.last_learning_rate = state->header.base_learning_rate;
}

// CosineAnnealing scheduler get last learning rate function
float cosine_annealing_scheduler_get_last_lr(const boat_scheduler_t* scheduler) {
    if (!scheduler) return 0.0f;

    const boat_cosine_annealing_state_t* state = (const boat_cosine_annealing_state_t*)scheduler;
    return state->header.last_learning_rate;
}

// CosineAnnealing scheduler get next learning rate function
float cosine_annealing_scheduler_get_next_lr(const boat_scheduler_t* scheduler) {
    if (!scheduler) return 0.0f;

    const boat_cosine_annealing_state_t* state = (const boat_cosine_annealing_state_t*)scheduler;

    // Calculate what the learning rate will be after the next step
    float step = (float)(state->header.step_count + 1); // Next step
    float T_max_f = (float)state->T_max;

    // Clamp step to T_max to handle restart behavior
    if (step > T_max_f) {
        step = T_max_f;
    }

    float cos_factor = cosf(3.14159265358979323846f * step / T_max_f);
    float next_lr = state->eta_min + 0.5f * (state->header.base_learning_rate - state->eta_min) * (1.0f + cos_factor);

    return next_lr;
}

// CosineAnnealing scheduler free function
void cosine_annealing_scheduler_free(boat_scheduler_t* scheduler) {
    if (!scheduler) return;

    // State was allocated as a single block, just free it
    boat_free(scheduler);
}