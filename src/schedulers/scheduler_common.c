// scheduler_common.c - Common scheduler functions and utilities
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <stdlib.h>
#include <math.h>
#include <string.h>

// Undefine potentially conflicting Windows macros
#ifdef device
#undef device
#endif

#include <boat/schedulers.h>
#include <boat/optimizers.h>
#include <boat/memory.h>
#include "schedulers_internal.h"

// Forward declarations for scheduler-specific functions
void step_lr_scheduler_step(boat_scheduler_t* scheduler);
void step_lr_scheduler_reset(boat_scheduler_t* scheduler);
float step_lr_scheduler_get_last_lr(const boat_scheduler_t* scheduler);
float step_lr_scheduler_get_next_lr(const boat_scheduler_t* scheduler);
void step_lr_scheduler_free(boat_scheduler_t* scheduler);

void cosine_annealing_scheduler_step(boat_scheduler_t* scheduler);
void cosine_annealing_scheduler_reset(boat_scheduler_t* scheduler);
float cosine_annealing_scheduler_get_last_lr(const boat_scheduler_t* scheduler);
float cosine_annealing_scheduler_get_next_lr(const boat_scheduler_t* scheduler);
void cosine_annealing_scheduler_free(boat_scheduler_t* scheduler);

void lambda_lr_scheduler_step(boat_scheduler_t* scheduler);
void lambda_lr_scheduler_reset(boat_scheduler_t* scheduler);
float lambda_lr_scheduler_get_last_lr(const boat_scheduler_t* scheduler);
float lambda_lr_scheduler_get_next_lr(const boat_scheduler_t* scheduler);
void lambda_lr_scheduler_free(boat_scheduler_t* scheduler);

// Helper function to get scheduler type
static boat_scheduler_type_t get_scheduler_type(const boat_scheduler_t* scheduler) {
    if (!scheduler) {
        return BOAT_SCHEDULER_STEP_LR; // Default
    }
    const boat_scheduler_header_t* header = (const boat_scheduler_header_t*)scheduler;
    return header->type;
}

// Generic scheduler step function (dispatches to specific implementation)
BOAT_API void boat_scheduler_step(boat_scheduler_t* scheduler) {
    if (!scheduler) return;

    switch (get_scheduler_type(scheduler)) {
        case BOAT_SCHEDULER_STEP_LR:
            step_lr_scheduler_step(scheduler);
            break;
        case BOAT_SCHEDULER_COSINE_ANNEALING:
            cosine_annealing_scheduler_step(scheduler);
            break;
        case BOAT_SCHEDULER_LAMBDA_LR:
            lambda_lr_scheduler_step(scheduler);
            break;
        default:
            // Unknown scheduler type, do nothing
            break;
    }
}

// Generic scheduler reset function
BOAT_API void boat_scheduler_reset(boat_scheduler_t* scheduler) {
    if (!scheduler) return;

    switch (get_scheduler_type(scheduler)) {
        case BOAT_SCHEDULER_STEP_LR:
            step_lr_scheduler_reset(scheduler);
            break;
        case BOAT_SCHEDULER_COSINE_ANNEALING:
            cosine_annealing_scheduler_reset(scheduler);
            break;
        case BOAT_SCHEDULER_LAMBDA_LR:
            lambda_lr_scheduler_reset(scheduler);
            break;
        default:
            // Unknown scheduler type, do nothing
            break;
    }
}

// Generic scheduler get last learning rate function
BOAT_API float boat_scheduler_get_last_lr(const boat_scheduler_t* scheduler) {
    if (!scheduler) return 0.0f;

    switch (get_scheduler_type(scheduler)) {
        case BOAT_SCHEDULER_STEP_LR:
            return step_lr_scheduler_get_last_lr(scheduler);
        case BOAT_SCHEDULER_COSINE_ANNEALING:
            return cosine_annealing_scheduler_get_last_lr(scheduler);
        case BOAT_SCHEDULER_LAMBDA_LR:
            return lambda_lr_scheduler_get_last_lr(scheduler);
        default:
            return 0.0f;
    }
}

// Generic scheduler get next learning rate function
BOAT_API float boat_scheduler_get_next_lr(const boat_scheduler_t* scheduler) {
    if (!scheduler) return 0.0f;

    switch (get_scheduler_type(scheduler)) {
        case BOAT_SCHEDULER_STEP_LR:
            return step_lr_scheduler_get_next_lr(scheduler);
        case BOAT_SCHEDULER_COSINE_ANNEALING:
            return cosine_annealing_scheduler_get_next_lr(scheduler);
        case BOAT_SCHEDULER_LAMBDA_LR:
            return lambda_lr_scheduler_get_next_lr(scheduler);
        default:
            return 0.0f;
    }
}

// Generic scheduler free function
BOAT_API void boat_scheduler_free(boat_scheduler_t* scheduler) {
    if (!scheduler) return;

    switch (get_scheduler_type(scheduler)) {
        case BOAT_SCHEDULER_STEP_LR:
            step_lr_scheduler_free(scheduler);
            break;
        case BOAT_SCHEDULER_COSINE_ANNEALING:
            cosine_annealing_scheduler_free(scheduler);
            break;
        case BOAT_SCHEDULER_LAMBDA_LR:
            lambda_lr_scheduler_free(scheduler);
            break;
        default:
            // Unknown scheduler type, just free the memory
            boat_free(scheduler);
            break;
    }
}

// Convenience function to update optimizer learning rate
BOAT_API void boat_scheduler_update_optimizer(const boat_scheduler_t* scheduler,
                                              boat_optimizer_t* optimizer) {
    if (!scheduler || !optimizer) return;

    float lr = boat_scheduler_get_last_lr(scheduler);
    if (lr > 0.0f) {
        boat_optimizer_set_learning_rate(optimizer, lr);
    }
}