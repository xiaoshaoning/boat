// loss.h - Loss functions
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_LOSS_H
#define BOAT_LOSS_H

#ifdef __cplusplus
extern "C" {
#endif

// Loss function types
typedef enum {
    BOAT_LOSS_MSE,
    BOAT_LOSS_CROSS_ENTROPY,
    BOAT_LOSS_HUBER
} boat_loss_type_t;

// Loss structure (opaque)
typedef struct boat_loss_t boat_loss_t;

// Create loss functions
boat_loss_t* boat_mse_loss_create();
boat_loss_t* boat_cross_entropy_loss_create();
boat_loss_t* boat_huber_loss_create(float delta);

// Loss operations
float boat_loss_compute(boat_loss_t* loss, const void* predictions, const void* targets);
void boat_loss_free(boat_loss_t* loss);

#ifdef __cplusplus
}
#endif

#endif // BOAT_LOSS_H