// schedulers_internal.h - Internal scheduler structures and declarations
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#ifndef BOAT_SCHEDULERS_INTERNAL_H
#define BOAT_SCHEDULERS_INTERNAL_H

#include <boat/schedulers.h>

// Generic scheduler header structure (must be first field in all scheduler states)
typedef struct boat_scheduler_header_t {
    boat_scheduler_type_t type;
    float base_learning_rate;  // Initial learning rate
    float last_learning_rate;  // Last computed learning rate
    int step_count;            // Step counter
} boat_scheduler_header_t;

// StepLR scheduler state structure
typedef struct boat_step_lr_state_t {
    boat_scheduler_header_t header;
    int step_size;            // Period of learning rate decay
    float gamma;              // Multiplicative factor of learning rate decay
} boat_step_lr_state_t;

// CosineAnnealing scheduler state structure
typedef struct boat_cosine_annealing_state_t {
    boat_scheduler_header_t header;
    int T_max;                // Maximum number of iterations
    float eta_min;            // Minimum learning rate
} boat_cosine_annealing_state_t;

// LambdaLR scheduler state structure
typedef struct boat_lambda_lr_state_t {
    boat_scheduler_header_t header;
    float (*lambda_fn)(int step, float base_lr);  // Custom lambda function
} boat_lambda_lr_state_t;

#endif // BOAT_SCHEDULERS_INTERNAL_H