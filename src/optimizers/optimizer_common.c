// optimizer_common.c - Common optimizer functions and utilities
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <stdlib.h>

// Undefine potentially conflicting Windows macros
#ifdef device
#undef device
#endif

// Forward declaration of boat_tensor_t (opaque structure)
typedef struct boat_tensor_t boat_tensor_t;

#include <boat/optimizers.h>
// #include <boat/tensor.h>  // Not needed for this file
#include <boat/memory.h>

// Generic optimizer header structure (must be first field in all optimizer states)
typedef struct boat_optimizer_header_t {
    boat_optimizer_type_t type;
} boat_optimizer_header_t;

// SGD optimizer state structure
typedef struct boat_sgd_state_t {
    boat_optimizer_header_t header;
    float learning_rate;
    float momentum;
    boat_tensor_t** params;
    boat_tensor_t** grads;
    boat_tensor_t** velocity;  // For momentum
    size_t num_params;
    size_t capacity;
} boat_sgd_state_t;

// RMSprop optimizer state structure
typedef struct boat_rmsprop_state_t {
    boat_optimizer_header_t header;
    float learning_rate;
    float alpha;
    float epsilon;
    boat_tensor_t** params;
    boat_tensor_t** grads;
    boat_tensor_t** square_avg;  // Running average of squared gradients
    size_t num_params;
    size_t capacity;
} boat_rmsprop_state_t;

// Adagrad optimizer state structure
typedef struct boat_adagrad_state_t {
    boat_optimizer_header_t header;
    float learning_rate;
    float epsilon;
    boat_tensor_t** params;
    boat_tensor_t** grads;
    boat_tensor_t** sum_square_grad;  // Sum of squared gradients
    size_t num_params;
    size_t capacity;
} boat_adagrad_state_t;

// Create SGD optimizer (stub implementation)
BOAT_API boat_optimizer_t* boat_sgd_optimizer_create(float learning_rate, float momentum) {
    // TODO: Implement SGD optimizer
    (void)learning_rate;
    (void)momentum;
    return NULL;
}

// Create RMSprop optimizer (implemented in rmsprop.c)
// boat_optimizer_t* boat_rmsprop_optimizer_create(float learning_rate, float alpha, float epsilon);

// Optimizer implementations (defined in respective files)
void adam_optimizer_add_parameter(boat_optimizer_t* optimizer, boat_tensor_t* param, boat_tensor_t* grad);
void adam_optimizer_step(boat_optimizer_t* optimizer);
void adam_optimizer_zero_grad(boat_optimizer_t* optimizer);
void adam_optimizer_free(boat_optimizer_t* optimizer);

void rmsprop_optimizer_add_parameter(boat_optimizer_t* optimizer, boat_tensor_t* param, boat_tensor_t* grad);
void rmsprop_optimizer_step(boat_optimizer_t* optimizer);
void rmsprop_optimizer_zero_grad(boat_optimizer_t* optimizer);
void rmsprop_optimizer_free(boat_optimizer_t* optimizer);

void adagrad_optimizer_add_parameter(boat_optimizer_t* optimizer, boat_tensor_t* param, boat_tensor_t* grad);
void adagrad_optimizer_step(boat_optimizer_t* optimizer);
void adagrad_optimizer_zero_grad(boat_optimizer_t* optimizer);
void adagrad_optimizer_free(boat_optimizer_t* optimizer);

// Learning rate access functions
float adam_optimizer_get_learning_rate(const boat_optimizer_t* optimizer);
void adam_optimizer_set_learning_rate(boat_optimizer_t* optimizer, float learning_rate);

float rmsprop_optimizer_get_learning_rate(const boat_optimizer_t* optimizer);
void rmsprop_optimizer_set_learning_rate(boat_optimizer_t* optimizer, float learning_rate);

float adagrad_optimizer_get_learning_rate(const boat_optimizer_t* optimizer);
void adagrad_optimizer_set_learning_rate(boat_optimizer_t* optimizer, float learning_rate);

// Generic optimizer step function (dispatches to specific implementation)
// Implemented in adam.c, sgd.c, and rmsprop.c

// Generic zero gradient function
// Implemented in adam.c, sgd.c, and rmsprop.c

// Generic optimizer free function
// Implemented in adam.c, sgd.c, and rmsprop.c

// Helper function to get optimizer type
static boat_optimizer_type_t get_optimizer_type(boat_optimizer_t* optimizer) {
    if (!optimizer) {
        return BOAT_OPTIMIZER_SGD; // Default
    }
    boat_optimizer_header_t* header = (boat_optimizer_header_t*)optimizer;
    return header->type;
}

// Generic optimizer functions
BOAT_API void boat_optimizer_add_parameter(boat_optimizer_t* optimizer,
                                  boat_tensor_t* param,
                                  boat_tensor_t* grad) {
    if (!optimizer) return;

    switch (get_optimizer_type(optimizer)) {
        case BOAT_OPTIMIZER_ADAM:
            adam_optimizer_add_parameter(optimizer, param, grad);
            break;
        case BOAT_OPTIMIZER_RMSPROP:
            rmsprop_optimizer_add_parameter(optimizer, param, grad);
            break;
        case BOAT_OPTIMIZER_ADAGRAD:
            adagrad_optimizer_add_parameter(optimizer, param, grad);
            break;
        case BOAT_OPTIMIZER_SGD:
        default:
            // TODO: Implement SGD
            break;
    }
}

BOAT_API void boat_optimizer_step(boat_optimizer_t* optimizer) {
    if (!optimizer) return;

    switch (get_optimizer_type(optimizer)) {
        case BOAT_OPTIMIZER_ADAM:
            adam_optimizer_step(optimizer);
            break;
        case BOAT_OPTIMIZER_RMSPROP:
            rmsprop_optimizer_step(optimizer);
            break;
        case BOAT_OPTIMIZER_ADAGRAD:
            adagrad_optimizer_step(optimizer);
            break;
        case BOAT_OPTIMIZER_SGD:
        default:
            // TODO: Implement SGD
            break;
    }
}

BOAT_API void boat_optimizer_zero_grad(boat_optimizer_t* optimizer) {
    if (!optimizer) return;

    switch (get_optimizer_type(optimizer)) {
        case BOAT_OPTIMIZER_ADAM:
            adam_optimizer_zero_grad(optimizer);
            break;
        case BOAT_OPTIMIZER_RMSPROP:
            rmsprop_optimizer_zero_grad(optimizer);
            break;
        case BOAT_OPTIMIZER_ADAGRAD:
            adagrad_optimizer_zero_grad(optimizer);
            break;
        case BOAT_OPTIMIZER_SGD:
        default:
            // TODO: Implement SGD
            break;
    }
}

BOAT_API void boat_optimizer_free(boat_optimizer_t* optimizer) {
    if (!optimizer) return;

    switch (get_optimizer_type(optimizer)) {
        case BOAT_OPTIMIZER_ADAM:
            adam_optimizer_free(optimizer);
            break;
        case BOAT_OPTIMIZER_RMSPROP:
            rmsprop_optimizer_free(optimizer);
            break;
        case BOAT_OPTIMIZER_ADAGRAD:
            adagrad_optimizer_free(optimizer);
            break;
        case BOAT_OPTIMIZER_SGD:
        default:
            // TODO: Implement SGD
            break;
    }
}

// Generic optimizer get learning rate function
BOAT_API float boat_optimizer_get_learning_rate(const boat_optimizer_t* optimizer) {
    if (!optimizer) return 0.0f;

    switch (get_optimizer_type((boat_optimizer_t*)optimizer)) {
        case BOAT_OPTIMIZER_ADAM:
            return adam_optimizer_get_learning_rate(optimizer);
        case BOAT_OPTIMIZER_RMSPROP:
            return rmsprop_optimizer_get_learning_rate(optimizer);
        case BOAT_OPTIMIZER_ADAGRAD:
            return adagrad_optimizer_get_learning_rate(optimizer);
        case BOAT_OPTIMIZER_SGD:
        default:
            // TODO: Implement SGD
            return 0.0f;
    }
}

// Generic optimizer set learning rate function
BOAT_API void boat_optimizer_set_learning_rate(boat_optimizer_t* optimizer, float learning_rate) {
    if (!optimizer) return;

    switch (get_optimizer_type(optimizer)) {
        case BOAT_OPTIMIZER_ADAM:
            adam_optimizer_set_learning_rate(optimizer, learning_rate);
            break;
        case BOAT_OPTIMIZER_RMSPROP:
            rmsprop_optimizer_set_learning_rate(optimizer, learning_rate);
            break;
        case BOAT_OPTIMIZER_ADAGRAD:
            adagrad_optimizer_set_learning_rate(optimizer, learning_rate);
            break;
        case BOAT_OPTIMIZER_SGD:
        default:
            // TODO: Implement SGD
            break;
    }
}