// optimizers.h - Optimization algorithms
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#ifndef BOAT_OPTIMIZERS_H
#define BOAT_OPTIMIZERS_H

#include "export.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optimizer types
typedef enum {
    BOAT_OPTIMIZER_SGD,
    BOAT_OPTIMIZER_ADAM,
    BOAT_OPTIMIZER_RMSPROP,
    BOAT_OPTIMIZER_ADAGRAD
} boat_optimizer_type_t;

// Optimizer structure (opaque)
typedef struct boat_optimizer_t boat_optimizer_t;

// Create optimizers
BOAT_API boat_optimizer_t* boat_sgd_optimizer_create(float learning_rate, float momentum);
BOAT_API boat_optimizer_t* boat_adam_optimizer_create(float learning_rate, float beta1, float beta2, float epsilon);
BOAT_API boat_optimizer_t* boat_rmsprop_optimizer_create(float learning_rate, float alpha, float epsilon);
BOAT_API boat_optimizer_t* boat_adagrad_optimizer_create(float learning_rate, float epsilon);

// Optimizer operations
BOAT_API void boat_optimizer_step(boat_optimizer_t* optimizer);
BOAT_API void boat_optimizer_zero_grad(boat_optimizer_t* optimizer);
BOAT_API void boat_optimizer_free(boat_optimizer_t* optimizer);

// Parameter registration
BOAT_API void boat_optimizer_add_parameter(boat_optimizer_t* optimizer,
                                           boat_tensor_t* param,
                                           boat_tensor_t* grad);

// Learning rate access
BOAT_API float boat_optimizer_get_learning_rate(const boat_optimizer_t* optimizer);
BOAT_API void boat_optimizer_set_learning_rate(boat_optimizer_t* optimizer, float learning_rate);

#ifdef __cplusplus
}
#endif

#endif // BOAT_OPTIMIZERS_H