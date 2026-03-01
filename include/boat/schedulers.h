// schedulers.h - Learning rate schedulers for deep learning framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_SCHEDULERS_H
#define BOAT_SCHEDULERS_H

#include "export.h"
#include "optimizers.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scheduler types
typedef enum {
    BOAT_SCHEDULER_STEP_LR,           // Step learning rate scheduler
    BOAT_SCHEDULER_COSINE_ANNEALING,  // Cosine annealing scheduler
    BOAT_SCHEDULER_LAMBDA_LR,         // Lambda learning rate scheduler
} boat_scheduler_type_t;

// Scheduler structure (opaque)
typedef struct boat_scheduler_t boat_scheduler_t;

// Create schedulers
BOAT_API boat_scheduler_t* boat_step_lr_scheduler_create(float base_learning_rate,
                                                         int step_size,
                                                         float gamma);

BOAT_API boat_scheduler_t* boat_cosine_annealing_scheduler_create(float base_learning_rate,
                                                                  int T_max,
                                                                  float eta_min);

BOAT_API boat_scheduler_t* boat_lambda_lr_scheduler_create(float base_learning_rate,
                                                           float (*lambda_fn)(int step, float base_lr));

// Scheduler operations
BOAT_API void boat_scheduler_step(boat_scheduler_t* scheduler);
BOAT_API void boat_scheduler_reset(boat_scheduler_t* scheduler);
BOAT_API float boat_scheduler_get_last_lr(const boat_scheduler_t* scheduler);
BOAT_API float boat_scheduler_get_next_lr(const boat_scheduler_t* scheduler);
BOAT_API void boat_scheduler_free(boat_scheduler_t* scheduler);

// Convenience function to update optimizer learning rate
BOAT_API void boat_scheduler_update_optimizer(const boat_scheduler_t* scheduler,
                                              boat_optimizer_t* optimizer);

#ifdef __cplusplus
}
#endif

#endif // BOAT_SCHEDULERS_H