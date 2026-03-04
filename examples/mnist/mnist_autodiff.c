// mnist_autodiff.c - MNIST digit recognition with Boat autodiff
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#define BOAT_STATIC_BUILD
// Force disable debug output
#undef DEBUG_LEVEL
#define DEBUG_LEVEL 0

// Force disable all Boat debug output
#undef BOAT_DEBUG
#define BOAT_DEBUG 0

#undef DEBUG
#define DEBUG 0
#include <boat.h>
#include <boat/autodiff.h>
#include <boat/tensor.h>
#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/optimizers.h>
#include <boat/schedulers.h>
#include <boat/loss.h>
#include <boat/memory.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>

#ifdef _WIN32
#include <io.h>
#define F_OK 0
#define access _access
#else
#include <unistd.h>
#endif

// Debug output control
// Levels: 0 = none, 1 = errors only, 2 = warnings, 3 = info, 4 = debug
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0  // Default to no debug output for performance
#endif

// Debug printing macros
#if DEBUG_LEVEL >= 4
#define DEBUG_PRINT(fmt, ...) DEBUG_PRINT("" fmt, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) ((void)0)
#endif

#if DEBUG_LEVEL >= 3
#define INFO_PRINT(fmt, ...) fprintf(stderr, "INFO: " fmt, ##__VA_ARGS__)
#else
#define INFO_PRINT(fmt, ...) ((void)0)
#endif

#if DEBUG_LEVEL >= 2
#define WARN_PRINT(fmt, ...) fprintf(stderr, "WARNING: " fmt, ##__VA_ARGS__)
#else
#define WARN_PRINT(fmt, ...) ((void)0)
#endif

#if DEBUG_LEVEL >= 1
#define ERROR_PRINT(fmt, ...) fprintf(stderr, "ERROR: " fmt, ##__VA_ARGS__)
#else
#define ERROR_PRINT(fmt, ...) ((void)0)
#endif

// Variable reuse helpers for performance optimization
typedef struct {
    boat_variable_t* input_var;      // Reusable input variable (no grad required)
    boat_variable_t* target_var;     // Reusable target variable (no grad required)
    boat_variable_t* logits_var;     // Reusable logits variable (from forward pass)
    boat_variable_t* loss_var;       // Reusable loss variable
} variable_pool_t;

// Create a variable pool for reuse
static variable_pool_t* create_variable_pool() {
    variable_pool_t* pool = malloc(sizeof(variable_pool_t));
    if (!pool) return NULL;

    pool->input_var = NULL;
    pool->target_var = NULL;
    pool->logits_var = NULL;
    pool->loss_var = NULL;

    return pool;
}

// Free variable pool
static void free_variable_pool(const variable_pool_t* pool) {
    if (!pool) return;

    if (pool->input_var) boat_variable_free(pool->input_var);
    if (pool->target_var) boat_variable_free(pool->target_var);
    if (pool->logits_var) boat_variable_free(pool->logits_var);
    if (pool->loss_var) boat_variable_free(pool->loss_var);

    free(pool);
}

// Reset variable data if variable exists, otherwise create new variable
static boat_variable_t* get_or_reset_variable(const variable_pool_t* pool, boat_variable_t** var_ptr,
                                              boat_tensor_t* tensor, bool requires_grad) {
    if (!pool || !var_ptr || !tensor) return NULL;

    if (*var_ptr) {
        // Try to reuse existing variable
        if (boat_variable_reset_data(*var_ptr, tensor)) {
            return *var_ptr;
        } else {
            // Reset failed, create new variable
            boat_variable_free(*var_ptr);
            *var_ptr = boat_variable_create(tensor, requires_grad);
            return *var_ptr;
        }
    } else {
        // Create new variable
        *var_ptr = boat_variable_create(tensor, requires_grad);
        return *var_ptr;
    }
}

// IDX file format loader (standard MNIST format)
boat_tensor_t* load_idx_file(const char* filename, boat_dtype_t dtype) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open IDX file %s\n", filename);
        return NULL;
    }

    // Read magic number (big-endian)
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    // Convert from big-endian to host byte order
    magic = ((magic >> 24) & 0xFF) |
            ((magic >> 8) & 0xFF00) |
            ((magic << 8) & 0xFF0000) |
            ((magic << 24) & 0xFF000000);

    // Check magic number
    uint8_t expected_dtype = (magic >> 8) & 0xFF;
    uint8_t ndim = magic & 0xFF;

    if (expected_dtype != 0x08) { // 0x08 = unsigned byte
        fprintf(stderr, "Error: IDX file %s has unsupported data type 0x%02x\n", filename, expected_dtype);
        fclose(f);
        return NULL;
    }

    // Read dimensions
    uint32_t* dims = malloc(ndim * sizeof(uint32_t));
    if (!dims) {
        fclose(f);
        return NULL;
    }

    for (uint8_t i = 0; i < ndim; i++) {
        uint32_t dim_be;
        if (fread(&dim_be, sizeof(uint32_t), 1, f) != 1) {
            free(dims);
            fclose(f);
            return NULL;
        }
        // Convert from big-endian to host byte order
        dims[i] = ((dim_be >> 24) & 0xFF) |
                 ((dim_be >> 8) & 0xFF00) |
                 ((dim_be << 8) & 0xFF0000) |
                 ((dim_be << 24) & 0xFF000000);
        DEBUG_PRINT("dims[%d] = %u (0x%08x)\n", i, dims[i], dims[i]);
    }

    // Calculate total elements
    size_t total_elements = 1;
    for (uint8_t i = 0; i < ndim; i++) {
        total_elements *= dims[i];
    }

    // Allocate data buffer
    void* data = malloc(total_elements);
    if (!data) {
        free(dims);
        fclose(f);
        return NULL;
    }

    // Read data (unsigned bytes)
    if (fread(data, 1, total_elements, f) != total_elements) {
        free(data);
        free(dims);
        fclose(f);
        return NULL;
    }

    fclose(f);

    // Convert dims to int64_t for tensor creation
    int64_t* shape = malloc(ndim * sizeof(int64_t));
    if (!shape) {
        free(data);
        free(dims);
        return NULL;
    }
    for (uint8_t i = 0; i < ndim; i++) {
        shape[i] = (int64_t)dims[i];
    }

    // Create tensor (images are uint8, will be normalized later)
    boat_tensor_t* tensor = boat_tensor_from_data(shape, ndim, dtype, data);

    fprintf(stderr, "Loaded IDX file %s: shape=[", filename);
    for (uint8_t i = 0; i < ndim; i++) {
        fprintf(stderr, "%lld%s", (long long)shape[i], i == ndim-1 ? "]\n" : ", ");
    }

    free(data);
    free(dims);
    free(shape);

    return tensor;
}

// Convert IDX uint8 images to normalized float32 tensor with channel dimension
static boat_tensor_t* idx_images_to_float32(boat_tensor_t* idx_tensor) {
    if (!idx_tensor) return NULL;

    const int64_t* shape = boat_tensor_shape(idx_tensor);
    size_t ndim = boat_tensor_ndim(idx_tensor);
    boat_dtype_t dtype = boat_tensor_dtype(idx_tensor);

    // IDX images are [N, 28, 28] or [N, 1, 28, 28] depending on source
    // We need [N, 1, 28, 28] for the model
    int64_t new_shape[4];
    if (ndim == 3) {
        // [N, 28, 28] -> [N, 1, 28, 28]
        new_shape[0] = shape[0];
        new_shape[1] = 1;
        new_shape[2] = shape[1];
        new_shape[3] = shape[2];
        ndim = 4;
    } else if (ndim == 4) {
        // Already [N, C, H, W], just use as is
        new_shape[0] = shape[0];
        new_shape[1] = shape[1];
        new_shape[2] = shape[2];
        new_shape[3] = shape[3];
    } else {
        fprintf(stderr, "Error: Unexpected image tensor shape with %zu dimensions\n", ndim);
        return NULL;
    }

    // Create output tensor
    boat_tensor_t* output = boat_tensor_create(new_shape, ndim, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!output) {
        fprintf(stderr, "Error: Failed to create output tensor\n");
        return NULL;
    }

    // Get data pointers
    float* output_data = (float*)boat_tensor_data(output);
    size_t total_elements = boat_tensor_nelements(output);

    // Normalize from [0, 255] to [0, 1] and convert to float32
    if (dtype == BOAT_DTYPE_UINT8) {
        const uint8_t* input_data = (const uint8_t*)boat_tensor_const_data(idx_tensor);
        for (size_t i = 0; i < total_elements; i++) {
            output_data[i] = input_data[i] / 255.0f;
        }
    } else if (dtype == BOAT_DTYPE_FLOAT32) {
        // Assume values are in [0, 255] range (as stored by mnist_data.py)
        const float* input_data = (const float*)boat_tensor_const_data(idx_tensor);
        for (size_t i = 0; i < total_elements; i++) {
            output_data[i] = input_data[i] / 255.0f;
        }
    } else {
        fprintf(stderr, "Error: Unsupported dtype for image conversion\n");
        boat_tensor_unref(output);
        return NULL;
    }

    fprintf(stderr, "Converted images: [%lld, %lld, %lld, %lld] -> float32 normalized\n",
            new_shape[0], new_shape[1], new_shape[2], new_shape[3]);

    return output;
}

// Original binary format loader
boat_tensor_t* load_tensor_binary(const char* filename, boat_dtype_t dtype) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }

    // Read number of dimensions
    uint32_t ndim;
    if (fread(&ndim, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    // Read shape
    int64_t* shape = malloc(sizeof(int64_t) * ndim);
    if (!shape) {
        fclose(f);
        return NULL;
    }

    for (uint32_t i = 0; i < ndim; i++) {
        uint32_t dim;
        if (fread(&dim, sizeof(uint32_t), 1, f) != 1) {
            free(shape);
            fclose(f);
            return NULL;
        }
        shape[i] = (int64_t)dim;
    }

    // Calculate total elements
    size_t total_elements = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }

    // Allocate data buffer
    size_t element_size = boat_dtype_size(dtype);
    void* data = malloc(total_elements * element_size);
    if (!data) {
        free(shape);
        fclose(f);
        return NULL;
    }

    // Read data
    if (fread(data, element_size, total_elements, f) != total_elements) {
        free(data);
        free(shape);
        fclose(f);
        return NULL;
    }

    fclose(f);

    // Create tensor
    boat_tensor_t* tensor = boat_tensor_from_data(shape, ndim, dtype, data);
    free(data);
    free(shape);

    return tensor;
}

// Model structure using autodiff
typedef struct {
    // Layers
    boat_conv_layer_t* conv1;
    boat_conv_layer_t* conv2;
    boat_pool_layer_t* pool1;
    boat_pool_layer_t* pool2;
    boat_dense_layer_t* fc1;
    boat_dense_layer_t* fc2;
    boat_flatten_layer_t* flatten;

    // Optimizer
    boat_optimizer_t* optimizer;
    float current_beta1;           // Current beta1 (momentum) parameter for Adam
    float current_beta2;           // Current beta2 (RMSprop) parameter for Adam

    // Learning rate scheduler
    boat_scheduler_t* scheduler;

    // Reusable variables for performance optimization
    boat_variable_t* reusable_input_var;
    boat_variable_t* reusable_target_var;
} mnist_model_t;

// Helper function to create variable from tensor
static boat_variable_t* tensor_to_variable(boat_tensor_t* tensor, bool requires_grad) {
    return boat_variable_create(tensor, requires_grad);
}

// Helper function to compute cross-entropy loss between predictions (logits) and labels
// predictions: variable with shape (batch, 10), logits (before softmax)
// labels: tensor with shape (batch) containing class indices (0-9)
// Returns loss variable
static boat_variable_t* cross_entropy_loss(const boat_variable_t* predictions, boat_tensor_t* labels) {
    DEBUG_PRINT("cross_entropy_loss: entered\n");
    // Get prediction tensor
    boat_tensor_t* pred_tensor = boat_variable_data(predictions);
    const int64_t* shape = boat_tensor_shape(pred_tensor);
    size_t batch_size = shape[0];
    size_t num_classes = shape[1];
    DEBUG_PRINT("cross_entropy_loss: batch_size=%zu, num_classes=%zu\n", batch_size, num_classes);
    DEBUG_PRINT("cross_entropy_loss: pred_tensor dtype=%d\n", boat_tensor_dtype(pred_tensor));

    // Create one-hot encoding of labels
    boat_tensor_t* one_hot_tensor = boat_tensor_create((int64_t[]){batch_size, num_classes}, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!one_hot_tensor) {
        DEBUG_PRINT("cross_entropy_loss: failed to create one-hot tensor\n");
        return NULL;
    }
    DEBUG_PRINT("cross_entropy_loss: one_hot_tensor created, dtype=%d\n", boat_tensor_dtype(one_hot_tensor));
    float* one_hot_data = (float*)boat_tensor_data(one_hot_tensor);
    memset(one_hot_data, 0, batch_size * num_classes * sizeof(float));

    const uint8_t* label_data = (const uint8_t*)boat_tensor_const_data(labels);
    for (size_t i = 0; i < batch_size; i++) {
        uint8_t label = label_data[i];
        if (label < num_classes) {
            one_hot_data[i * num_classes + label] = 1.0f;
        }
    }

    // Convert one-hot tensor to variable (no gradient required)
    boat_variable_t* one_hot_var = boat_variable_create(one_hot_tensor, false);
    boat_tensor_unref(one_hot_tensor);
    if (!one_hot_var) {
        DEBUG_PRINT("cross_entropy_loss: failed to create one-hot variable\n");
        return NULL;
    }

    // Compute log softmax of predictions (axis=1)
    boat_variable_t* log_softmax = boat_var_log_softmax(predictions, 1);
    if (!log_softmax) {
        DEBUG_PRINT("cross_entropy_loss: failed to compute log_softmax\n");
        boat_variable_free(one_hot_var);
        return NULL;
    }
    DEBUG_PRINT("cross_entropy_loss: log_softmax variable created at %p\n", (void*)log_softmax);
    boat_tensor_t* log_softmax_tensor = boat_variable_data(log_softmax);
    DEBUG_PRINT("cross_entropy_loss: log_softmax tensor dtype=%d\n", boat_tensor_dtype(log_softmax_tensor));

    // Debug: print variable info before multiplication
    boat_tensor_t* one_hot_tensor_debug = boat_variable_data(one_hot_var);
    boat_tensor_t* log_softmax_tensor_debug = boat_variable_data(log_softmax);
    DEBUG_PRINT("cross_entropy_loss: one_hot dtype=%d, shape=[%ld, %ld]\n",
            boat_tensor_dtype(one_hot_tensor_debug),
            boat_tensor_shape(one_hot_tensor_debug)[0],
            boat_tensor_shape(one_hot_tensor_debug)[1]);
    DEBUG_PRINT("cross_entropy_loss: log_softmax dtype=%d, shape=[%ld, %ld]\n",
            boat_tensor_dtype(log_softmax_tensor_debug),
            boat_tensor_shape(log_softmax_tensor_debug)[0],
            boat_tensor_shape(log_softmax_tensor_debug)[1]);

    // Element-wise multiplication: one_hot * log_softmax
    DEBUG_PRINT("cross_entropy_loss: before boat_var_mul, one_hot dtype=%d, log_softmax dtype=%d\n",
            boat_tensor_dtype(one_hot_tensor_debug),
            boat_tensor_dtype(log_softmax_tensor_debug));
    fflush(stderr);
    boat_variable_t* multiplied = boat_var_mul(one_hot_var, log_softmax);
    if (!multiplied) {
        DEBUG_PRINT("cross_entropy_loss: failed to multiply\n");
        boat_variable_free(one_hot_var);
        boat_variable_free(log_softmax);
        return NULL;
    }

    // Sum over all elements (both batch and class dimensions)
    boat_variable_t* sum = boat_var_sum(multiplied, NULL, 0, false);
    if (!sum) {
        DEBUG_PRINT("cross_entropy_loss: failed to sum\n");
        boat_variable_free(one_hot_var);
        boat_variable_free(log_softmax);
        boat_variable_free(multiplied);
        return NULL;
    }

    // Negate and divide by batch size
    // Create constant -1.0 / batch_size variable
    boat_tensor_t* neg_factor_tensor = boat_tensor_create((int64_t[]){1}, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!neg_factor_tensor) {
        DEBUG_PRINT("cross_entropy_loss: failed to create factor tensor\n");
        boat_variable_free(one_hot_var);
        boat_variable_free(log_softmax);
        boat_variable_free(multiplied);
        boat_variable_free(sum);
        return NULL;
    }
    float* factor_data = (float*)boat_tensor_data(neg_factor_tensor);
    *factor_data = -1.0f / batch_size;
    DEBUG_PRINT("cross_entropy_loss: created neg_factor_tensor dtype=%d, value=%f\n",
            boat_tensor_dtype(neg_factor_tensor), *factor_data);
    boat_variable_t* factor_var = boat_variable_create(neg_factor_tensor, false);
    boat_tensor_unref(neg_factor_tensor);
    if (factor_var) {
        boat_tensor_t* factor_tensor = boat_variable_data(factor_var);
        DEBUG_PRINT("cross_entropy_loss: factor_var created, tensor dtype=%d\n",
                boat_tensor_dtype(factor_tensor));
    }
    if (!factor_var) {
        DEBUG_PRINT("cross_entropy_loss: failed to create factor variable\n");
        boat_variable_free(one_hot_var);
        boat_variable_free(log_softmax);
        boat_variable_free(multiplied);
        boat_variable_free(sum);
        return NULL;
    }

    // Multiply sum by factor to get average negative log likelihood
    boat_variable_t* loss_var = boat_var_mul(factor_var, sum);

    // Cleanup intermediate variables
    boat_variable_free(one_hot_var);
    boat_variable_free(log_softmax);
    boat_variable_free(multiplied);
    boat_variable_free(sum);
    boat_variable_free(factor_var);

    DEBUG_PRINT("cross_entropy_loss: returning loss_var=%p\n", (void*)loss_var);
    return loss_var;
}

mnist_model_t* create_mnist_model(float learning_rate) {
    DEBUG_PRINT("create_mnist_model: entering\n");
    mnist_model_t* model = malloc(sizeof(mnist_model_t));
    if (!model) return NULL;
    DEBUG_PRINT("create_mnist_model: model allocated at %p\n", model);

    // Initialize reusable variables to NULL
    model->reusable_input_var = NULL;
    model->reusable_target_var = NULL;

    // Create layers
    DEBUG_PRINT("create_mnist_model: creating conv1\n");
    model->conv1 = boat_conv_layer_create(1, 32, 3, 1, 1);  // 1->32 channels, 3x3 kernel, stride=1, padding=1
    DEBUG_PRINT("create_mnist_model: conv1 = %p\n", model->conv1);
    DEBUG_PRINT("create_mnist_model: creating pool1\n");
    model->pool1 = boat_pool_layer_create(2, 2, 0);         // 2x2 max pool, stride=2
    DEBUG_PRINT("create_mnist_model: pool1 = %p\n", model->pool1);

    DEBUG_PRINT("create_mnist_model: creating conv2\n");
    model->conv2 = boat_conv_layer_create(32, 64, 3, 1, 1); // 32->64 channels
    DEBUG_PRINT("create_mnist_model: conv2 = %p\n", model->conv2);
    DEBUG_PRINT("create_mnist_model: creating pool2\n");
    model->pool2 = boat_pool_layer_create(2, 2, 0);
    DEBUG_PRINT("create_mnist_model: pool2 = %p\n", model->pool2);

    DEBUG_PRINT("create_mnist_model: creating flatten\n");
    model->flatten = boat_flatten_layer_create();
    DEBUG_PRINT("create_mnist_model: flatten = %p\n", model->flatten);

    DEBUG_PRINT("create_mnist_model: creating fc1\n");
    model->fc1 = boat_dense_layer_create(7*7*64, 128, true); // After 2 poolings: 28->14->7
    DEBUG_PRINT("create_mnist_model: fc1 = %p\n", model->fc1);
    DEBUG_PRINT("create_mnist_model: creating fc2\n");
    model->fc2 = boat_dense_layer_create(128, 10, true);
    DEBUG_PRINT("create_mnist_model: fc2 = %p\n", model->fc2);

    // Check for creation errors
    DEBUG_PRINT("create_mnist_model: checking layer creation\n");
    if (!model->conv1 || !model->conv2 || !model->pool1 || !model->pool2 ||
        !model->flatten || !model->fc1 || !model->fc2) {
        fprintf(stderr, "Error: Failed to create one or more layers\n");
        free(model);
        return NULL;
    }
    DEBUG_PRINT("create_mnist_model: all layers created successfully\n");

    // Create optimizer
    DEBUG_PRINT("create_mnist_model: creating optimizer\n");
    model->current_beta1 = 0.9f;
    model->current_beta2 = 0.999f;
    model->optimizer = boat_adam_optimizer_create(learning_rate, model->current_beta1, model->current_beta2, 1e-8f);
    DEBUG_PRINT("create_mnist_model: optimizer = %p\n", model->optimizer);
    if (!model->optimizer) {
        fprintf(stderr, "Error: Failed to create optimizer\n");
        free(model);
        return NULL;
    }
    DEBUG_PRINT("create_mnist_model: optimizer created successfully (beta1=%.3f, beta2=%.3f)\n",
            model->current_beta1, model->current_beta2);

    // Create learning rate scheduler (CosineAnnealing with warm-up)
    DEBUG_PRINT("create_mnist_model: creating scheduler\n");
    // Cosine annealing scheduler: T_max=10 epochs (will be adjusted in training loop), eta_min=1% of base LR
    model->scheduler = boat_cosine_annealing_scheduler_create(learning_rate, 10, learning_rate * 0.01f);
    DEBUG_PRINT("create_mnist_model: scheduler = %p\n", model->scheduler);
    if (!model->scheduler) {
        fprintf(stderr, "Warning: Failed to create scheduler, continuing without scheduler\n");
    } else {
        DEBUG_PRINT("create_mnist_model: cosine annealing scheduler created successfully (T_max=10, eta_min=%.6f)\n", learning_rate * 0.01f);
    }

    // Register parameters to optimizer
    DEBUG_PRINT("create_mnist_model: registering parameters\n");
    // Conv1 weight and bias
    DEBUG_PRINT("create_mnist_model: getting conv1 weight\n");
    boat_tensor_t* conv1_weight = boat_conv_layer_get_weight(model->conv1);
    boat_tensor_t* conv1_grad_weight = boat_conv_layer_get_grad_weight(model->conv1);
    DEBUG_PRINT("create_mnist_model: conv1_weight=%p, conv1_grad_weight=%p\n", conv1_weight, conv1_grad_weight);
    if (conv1_weight && conv1_grad_weight) {
        boat_optimizer_add_parameter(model->optimizer, conv1_weight, conv1_grad_weight);
    }
    boat_tensor_t* conv1_bias = boat_conv_layer_get_bias(model->conv1);
    boat_tensor_t* conv1_grad_bias = boat_conv_layer_get_grad_bias(model->conv1);
    if (conv1_bias && conv1_grad_bias) {
        boat_optimizer_add_parameter(model->optimizer, conv1_bias, conv1_grad_bias);
    }

    // Conv2 weight and bias
    boat_tensor_t* conv2_weight = boat_conv_layer_get_weight(model->conv2);
    boat_tensor_t* conv2_grad_weight = boat_conv_layer_get_grad_weight(model->conv2);
    if (conv2_weight && conv2_grad_weight) {
        boat_optimizer_add_parameter(model->optimizer, conv2_weight, conv2_grad_weight);
    }
    boat_tensor_t* conv2_bias = boat_conv_layer_get_bias(model->conv2);
    boat_tensor_t* conv2_grad_bias = boat_conv_layer_get_grad_bias(model->conv2);
    if (conv2_bias && conv2_grad_bias) {
        boat_optimizer_add_parameter(model->optimizer, conv2_bias, conv2_grad_bias);
    }

    // FC1 weight and bias
    boat_tensor_t* fc1_weight = boat_dense_layer_get_weight(model->fc1);
    boat_tensor_t* fc1_grad_weight = boat_dense_layer_get_grad_weight(model->fc1);
    if (fc1_weight && fc1_grad_weight) {
        boat_optimizer_add_parameter(model->optimizer, fc1_weight, fc1_grad_weight);
    }
    boat_tensor_t* fc1_bias = boat_dense_layer_get_bias(model->fc1);
    boat_tensor_t* fc1_grad_bias = boat_dense_layer_get_grad_bias(model->fc1);
    if (fc1_bias && fc1_grad_bias) {
        boat_optimizer_add_parameter(model->optimizer, fc1_bias, fc1_grad_bias);
    }

    // FC2 weight and bias
    boat_tensor_t* fc2_weight = boat_dense_layer_get_weight(model->fc2);
    boat_tensor_t* fc2_grad_weight = boat_dense_layer_get_grad_weight(model->fc2);
    if (fc2_weight && fc2_grad_weight) {
        boat_optimizer_add_parameter(model->optimizer, fc2_weight, fc2_grad_weight);
    }
    boat_tensor_t* fc2_bias = boat_dense_layer_get_bias(model->fc2);
    boat_tensor_t* fc2_grad_bias = boat_dense_layer_get_grad_bias(model->fc2);
    if (fc2_bias && fc2_grad_bias) {
        boat_optimizer_add_parameter(model->optimizer, fc2_bias, fc2_grad_bias);
    }

    printf("MNIST model created successfully with autodiff\n");
    printf("Architecture:\n");
    printf("  Input: 1x28x28\n");
    printf("  Conv2D(1->32, 3x3, padding=1) -> ReLU -> MaxPool2D(2x2)\n");
    printf("  Conv2D(32->64, 3x3, padding=1) -> ReLU -> MaxPool2D(2x2)\n");
    printf("  Flatten -> Dense(3136->128) -> ReLU -> Dense(128->10) -> Softmax\n");

    return model;
}

void free_mnist_model(mnist_model_t* model) {
    if (!model) return;

    if (model->conv1) boat_conv_layer_free(model->conv1);
    if (model->conv2) boat_conv_layer_free(model->conv2);
    if (model->pool1) boat_pool_layer_free(model->pool1);
    if (model->pool2) boat_pool_layer_free(model->pool2);
    if (model->fc1) boat_dense_layer_free(model->fc1);
    if (model->fc2) boat_dense_layer_free(model->fc2);
    if (model->flatten) boat_flatten_layer_free(model->flatten);
    if (model->optimizer) boat_optimizer_free(model->optimizer);
    if (model->scheduler) boat_scheduler_free(model->scheduler);

    // Free reusable variables
    if (model->reusable_input_var) boat_variable_free(model->reusable_input_var);
    if (model->reusable_target_var) boat_variable_free(model->reusable_target_var);

    free(model);
}

// Forward pass using autodiff variables
// input: variable with shape (batch, 1, 28, 28)
// Returns: variable with shape (batch, 10) (logits before softmax)
// Forward pass using layer functions (non-autodiff version)
boat_tensor_t* forward_pass_layer(mnist_model_t* model, boat_tensor_t* input) {
    DEBUG_PRINT("forward_pass_layer: entered, input=%p\n", (void*)input);
    boat_tensor_t* x = input;

    // Conv1 -> ReLU -> Pool1
    DEBUG_PRINT("forward_pass_layer: calling boat_conv_layer_forward\n");
    x = boat_conv_layer_forward(model->conv1, x);
    DEBUG_PRINT("forward_pass_layer: boat_conv_layer_forward returned %p\n", (void*)x);
    if (!x) return NULL;

    DEBUG_PRINT("forward_pass_layer: calling boat_relu\n");
    x = boat_relu(x);
    DEBUG_PRINT("forward_pass_layer: boat_relu returned %p\n", (void*)x);
    if (!x) return NULL;

    DEBUG_PRINT("forward_pass_layer: calling boat_pool_layer_forward\n");
    x = boat_pool_layer_forward(model->pool1, x);
    DEBUG_PRINT("forward_pass_layer: boat_pool_layer_forward returned %p\n", (void*)x);
    if (!x) return NULL;

    // Conv2 -> ReLU -> Pool2
    DEBUG_PRINT("forward_pass_layer: calling boat_conv_layer_forward\n");
    x = boat_conv_layer_forward(model->conv2, x);
    DEBUG_PRINT("forward_pass_layer: boat_conv_layer_forward returned %p\n", (void*)x);
    if (!x) return NULL;

    DEBUG_PRINT("forward_pass_layer: calling boat_relu\n");
    x = boat_relu(x);
    DEBUG_PRINT("forward_pass_layer: boat_relu returned %p\n", (void*)x);
    if (!x) return NULL;

    DEBUG_PRINT("forward_pass_layer: calling boat_pool_layer_forward\n");
    x = boat_pool_layer_forward(model->pool2, x);
    DEBUG_PRINT("forward_pass_layer: boat_pool_layer_forward returned %p\n", (void*)x);
    if (!x) return NULL;

    // Flatten
    DEBUG_PRINT("forward_pass_layer: calling boat_flatten_layer_forward\n");
    x = boat_flatten_layer_forward(model->flatten, x);
    DEBUG_PRINT("forward_pass_layer: boat_flatten_layer_forward returned %p\n", (void*)x);
    if (!x) return NULL;

    // FC1 -> ReLU (using dense layer forward)
    DEBUG_PRINT("forward_pass_layer: calling boat_dense_layer_forward for fc1\n");
    x = boat_dense_layer_forward(model->fc1, x);
    DEBUG_PRINT("forward_pass_layer: boat_dense_layer_forward fc1 returned %p\n", (void*)x);
    if (!x) return NULL;

    DEBUG_PRINT("forward_pass_layer: calling boat_relu\n");
    x = boat_relu(x);
    DEBUG_PRINT("forward_pass_layer: boat_relu returned %p\n", (void*)x);
    if (!x) return NULL;

    // FC2 (output logits, no activation)
    DEBUG_PRINT("forward_pass_layer: calling boat_dense_layer_forward for fc2\n");
    x = boat_dense_layer_forward(model->fc2, x);
    DEBUG_PRINT("forward_pass_layer: boat_dense_layer_forward fc2 returned %p\n", (void*)x);

    return x;
}

boat_variable_t* forward_pass(mnist_model_t* model, const boat_variable_t* input) {
    DEBUG_PRINT("forward_pass: entered, input=%p\n", (void*)input);

    // Autodiff forward pass using variable operations
    boat_variable_t* x = input;

    // Conv1 -> ReLU -> Pool1
    x = boat_var_conv(x, model->conv1);
    x = boat_var_relu(x);
    x = boat_var_pool(x, model->pool1);

    // Conv2 -> ReLU -> Pool2
    x = boat_var_conv(x, model->conv2);
    x = boat_var_relu(x);
    x = boat_var_pool(x, model->pool2);

    // Flatten (autodiff version doesn't need layer object)
    x = boat_var_flatten(x);

    // FC1 -> ReLU
    x = boat_var_dense(x, model->fc1);
    x = boat_var_relu(x);

    // FC2 (output logits, no activation)
    x = boat_var_dense(x, model->fc2);

    DEBUG_PRINT("forward_pass: returning %p\n", (void*)x);
    return x;
}

// Model checkpoint functions
// Save model weights to file
// Static helper function to save a tensor to file
static bool save_tensor_to_file(FILE* f, boat_tensor_t* tensor) {
    if (!tensor) {
        // Write marker for NULL tensor
        uint32_t is_null = 1;
        fwrite(&is_null, sizeof(uint32_t), 1, f);
        return true;
    }

    uint32_t is_null = 0;
    fwrite(&is_null, sizeof(uint32_t), 1, f);

    // Get tensor properties
    const int64_t* shape = boat_tensor_shape(tensor);
    size_t ndim = boat_tensor_ndim(tensor);
    boat_dtype_t dtype = boat_tensor_dtype(tensor);
    size_t nbytes = boat_tensor_nbytes(tensor);
    const void* data = boat_tensor_const_data(tensor);

    // Write ndim
    uint32_t ndim_u32 = (uint32_t)ndim;
    fwrite(&ndim_u32, sizeof(uint32_t), 1, f);

    // Write shape
    for (size_t i = 0; i < ndim; i++) {
        uint32_t dim = (uint32_t)shape[i];
        fwrite(&dim, sizeof(uint32_t), 1, f);
    }

    // Write dtype as uint32
    uint32_t dtype_u32 = (uint32_t)dtype;
    fwrite(&dtype_u32, sizeof(uint32_t), 1, f);

    // Write data
    fwrite(data, 1, nbytes, f);

    return true;
}

// Static helper function to load a tensor from file
static boat_tensor_t* load_tensor_from_file(FILE* f) {
    uint32_t is_null;
    if (fread(&is_null, sizeof(uint32_t), 1, f) != 1) {
        return NULL;
    }

    if (is_null) {
        return NULL; // NULL tensor marker
    }

    // Read ndim
    uint32_t ndim_u32;
    if (fread(&ndim_u32, sizeof(uint32_t), 1, f) != 1) {
        return NULL;
    }
    size_t ndim = (size_t)ndim_u32;

    // Read shape
    int64_t* shape = malloc(sizeof(int64_t) * ndim);
    if (!shape) {
        return NULL;
    }

    for (size_t i = 0; i < ndim; i++) {
        uint32_t dim;
        if (fread(&dim, sizeof(uint32_t), 1, f) != 1) {
            free(shape);
            return NULL;
        }
        shape[i] = (int64_t)dim;
    }

    // Read dtype
    uint32_t dtype_u32;
    if (fread(&dtype_u32, sizeof(uint32_t), 1, f) != 1) {
        free(shape);
        return NULL;
    }
    boat_dtype_t dtype = (boat_dtype_t)dtype_u32;

    // Calculate total elements
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }

    size_t element_size = boat_dtype_size(dtype);
    size_t nbytes = total_elements * element_size;

    // Allocate and read data
    void* data = malloc(nbytes);
    if (!data) {
        free(shape);
        return NULL;
    }

    if (fread(data, 1, nbytes, f) != nbytes) {
        free(data);
        free(shape);
        return NULL;
    }

    // Create tensor
    boat_tensor_t* tensor = boat_tensor_from_data(shape, ndim, dtype, data);

    free(data);
    free(shape);
    return tensor;
}

// Save model weights to file
bool save_mnist_model(const mnist_model_t* model, const char* filename) {
    if (!model || !filename) return false;

    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return false;
    }

    // Write magic number and version
    uint32_t magic = 0x4D4E4953; // 'MNIS' for MNIST model
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);

    // Save conv1 weight and bias
    boat_tensor_t* conv1_weight = boat_conv_layer_get_weight(model->conv1);
    boat_tensor_t* conv1_bias = boat_conv_layer_get_bias(model->conv1);
    save_tensor_to_file(f, conv1_weight);
    save_tensor_to_file(f, conv1_bias);

    // Save conv2 weight and bias
    boat_tensor_t* conv2_weight = boat_conv_layer_get_weight(model->conv2);
    boat_tensor_t* conv2_bias = boat_conv_layer_get_bias(model->conv2);
    save_tensor_to_file(f, conv2_weight);
    save_tensor_to_file(f, conv2_bias);

    // Save fc1 weight and bias
    boat_tensor_t* fc1_weight = boat_dense_layer_get_weight(model->fc1);
    boat_tensor_t* fc1_bias = boat_dense_layer_get_bias(model->fc1);
    save_tensor_to_file(f, fc1_weight);
    save_tensor_to_file(f, fc1_bias);

    // Save fc2 weight and bias
    boat_tensor_t* fc2_weight = boat_dense_layer_get_weight(model->fc2);
    boat_tensor_t* fc2_bias = boat_dense_layer_get_bias(model->fc2);
    save_tensor_to_file(f, fc2_weight);
    save_tensor_to_file(f, fc2_bias);

    fclose(f);
    printf("Model saved to %s\n", filename);
    return true;
}

// Load model weights from file
bool load_mnist_model(mnist_model_t* model, const char* filename) {
    if (!model || !filename) return false;

    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file %s for reading\n", filename);
        return false;
    }

    // Read and verify magic number and version
    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fread(&version, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }

    if (magic != 0x4D4E4953) {
        fprintf(stderr, "Error: Invalid model file (bad magic number)\n");
        fclose(f);
        return false;
    }

    if (version != 1) {
        fprintf(stderr, "Error: Unsupported model version %u\n", version);
        fclose(f);
        return false;
    }

    // Load conv1 weight and bias
    boat_tensor_t* conv1_weight = load_tensor_from_file(f);
    boat_tensor_t* conv1_bias = load_tensor_from_file(f);
    if (conv1_weight) boat_conv_layer_set_weight(model->conv1, conv1_weight);
    if (conv1_bias) boat_conv_layer_set_bias(model->conv1, conv1_bias);
    if (conv1_weight) boat_tensor_unref(conv1_weight);
    if (conv1_bias) boat_tensor_unref(conv1_bias);

    // Load conv2 weight and bias
    boat_tensor_t* conv2_weight = load_tensor_from_file(f);
    boat_tensor_t* conv2_bias = load_tensor_from_file(f);
    if (conv2_weight) boat_conv_layer_set_weight(model->conv2, conv2_weight);
    if (conv2_bias) boat_conv_layer_set_bias(model->conv2, conv2_bias);
    if (conv2_weight) boat_tensor_unref(conv2_weight);
    if (conv2_bias) boat_tensor_unref(conv2_bias);

    // Load fc1 weight and bias
    boat_tensor_t* fc1_weight = load_tensor_from_file(f);
    boat_tensor_t* fc1_bias = load_tensor_from_file(f);
    if (fc1_weight) boat_dense_layer_set_weight(model->fc1, fc1_weight);
    if (fc1_bias) boat_dense_layer_set_bias(model->fc1, fc1_bias);
    if (fc1_weight) boat_tensor_unref(fc1_weight);
    if (fc1_bias) boat_tensor_unref(fc1_bias);

    // Load fc2 weight and bias
    boat_tensor_t* fc2_weight = load_tensor_from_file(f);
    boat_tensor_t* fc2_bias = load_tensor_from_file(f);
    if (fc2_weight) boat_dense_layer_set_weight(model->fc2, fc2_weight);
    if (fc2_bias) boat_dense_layer_set_bias(model->fc2, fc2_bias);
    if (fc2_weight) boat_tensor_unref(fc2_weight);
    if (fc2_bias) boat_tensor_unref(fc2_bias);

    fclose(f);
    printf("Model loaded from %s\n", filename);
    return true;
}

// Helper function to get or reset reusable variable in model
static boat_variable_t* get_reusable_input_variable(mnist_model_t* model, boat_tensor_t* tensor) {
    if (!model || !tensor) return NULL;

    if (model->reusable_input_var) {
        if (boat_variable_reset_data(model->reusable_input_var, tensor)) {
            return model->reusable_input_var;
        } else {
            // Reset failed, create new variable
            boat_variable_free(model->reusable_input_var);
            model->reusable_input_var = boat_variable_create(tensor, false);
            return model->reusable_input_var;
        }
    } else {
        // Create new variable
        model->reusable_input_var = boat_variable_create(tensor, false);
        return model->reusable_input_var;
    }
}

static boat_variable_t* get_reusable_target_variable(mnist_model_t* model, boat_tensor_t* tensor) {
    if (!model || !tensor) return NULL;

    if (model->reusable_target_var) {
        if (boat_variable_reset_data(model->reusable_target_var, tensor)) {
            return model->reusable_target_var;
        } else {
            // Reset failed, create new variable
            boat_variable_free(model->reusable_target_var);
            model->reusable_target_var = boat_variable_create(tensor, false);
            return model->reusable_target_var;
        }
    } else {
        // Create new variable
        model->reusable_target_var = boat_variable_create(tensor, false);
        return model->reusable_target_var;
    }
}

// Compute accuracy
float compute_accuracy(const boat_variable_t* predictions, boat_tensor_t* labels) {
    boat_tensor_t* pred_tensor = boat_variable_data(predictions);
    const float* pred_data = (const float*)boat_tensor_const_data(pred_tensor);
    const uint8_t* label_data = (const uint8_t*)boat_tensor_const_data(labels);

    const int64_t* shape = boat_tensor_shape(pred_tensor);
    size_t batch_size = shape[0];
    size_t num_classes = shape[1];

    int correct = 0;
    for (size_t i = 0; i < batch_size; i++) {
        int pred_class = 0;
        float max_val = pred_data[i * num_classes];
        for (size_t j = 1; j < num_classes; j++) {
            if (pred_data[i * num_classes + j] > max_val) {
                max_val = pred_data[i * num_classes + j];
                pred_class = j;
            }
        }
        if (pred_class == label_data[i]) {
            correct++;
        }
    }

    return (float)correct / batch_size;
}

// Compute gradient statistics for model monitoring
typedef struct {
    float total_norm;      // L2 norm of all gradients
    float max_grad;        // Maximum absolute gradient value
    float min_grad;        // Minimum absolute gradient value (non-zero)
    float mean_grad;       // Mean absolute gradient value
    float grad_clip_ratio; // Ratio of gradients that would be clipped at threshold
    int param_count;       // Number of parameters with gradients
    int clipped_count;     // Number of parameters that would be clipped
    int nan_inf_count;     // Number of NaN/Inf gradient values detected
} gradient_stats_t;

// Collect gradient statistics from model layers
static gradient_stats_t compute_gradient_stats(mnist_model_t* model, float clip_threshold) {
    gradient_stats_t stats = {0};

    if (!model) return stats;

    // We'll track statistics across all parameters
    float sum_squares = 0.0f;
    float sum_abs = 0.0f;
    float max_val = 0.0f;
    float min_val = FLT_MAX;
    int total_params = 0;
    int clipped_params = 0;
    int nan_inf_params = 0;

    // Helper macro to process gradient tensor
    #define PROCESS_GRADIENT_TENSOR(grad_tensor) \
        if (grad_tensor) { \
            const float* grad_data = (const float*)boat_tensor_const_data(grad_tensor); \
            size_t num_elements = boat_tensor_nbytes(grad_tensor) / sizeof(float); \
            total_params += num_elements; \
            for (size_t i = 0; i < num_elements; i++) { \
                float g = grad_data[i]; \
                /* Check for NaN or Inf */ \
                if (isnan(g) || isinf(g)) { \
                    nan_inf_params++; \
                    continue; /* Skip NaN/Inf values in statistics */ \
                } \
                float abs_g = fabsf(g); \
                sum_squares += g * g; \
                sum_abs += abs_g; \
                if (abs_g > max_val) max_val = abs_g; \
                if (abs_g > 0 && abs_g < min_val) min_val = abs_g; \
                if (clip_threshold > 0 && abs_g > clip_threshold) clipped_params++; \
            } \
        }

    // Process gradients from all layers
    if (model->conv1) {
        PROCESS_GRADIENT_TENSOR(boat_conv_layer_get_grad_weight(model->conv1));
        PROCESS_GRADIENT_TENSOR(boat_conv_layer_get_grad_bias(model->conv1));
    }
    if (model->conv2) {
        PROCESS_GRADIENT_TENSOR(boat_conv_layer_get_grad_weight(model->conv2));
        PROCESS_GRADIENT_TENSOR(boat_conv_layer_get_grad_bias(model->conv2));
    }
    if (model->fc1) {
        PROCESS_GRADIENT_TENSOR(boat_dense_layer_get_grad_weight(model->fc1));
        PROCESS_GRADIENT_TENSOR(boat_dense_layer_get_grad_bias(model->fc1));
    }
    if (model->fc2) {
        PROCESS_GRADIENT_TENSOR(boat_dense_layer_get_grad_weight(model->fc2));
        PROCESS_GRADIENT_TENSOR(boat_dense_layer_get_grad_bias(model->fc2));
    }

    #undef PROCESS_GRADIENT_TENSOR

    // Compute final statistics
    stats.total_norm = sqrtf(sum_squares);
    stats.max_grad = max_val;
    stats.min_grad = (min_val == FLT_MAX) ? 0.0f : min_val;
    stats.mean_grad = total_params > 0 ? sum_abs / total_params : 0.0f;
    stats.grad_clip_ratio = total_params > 0 ? (float)clipped_params / total_params : 0.0f;
    stats.param_count = total_params;
    stats.clipped_count = clipped_params;
    stats.nan_inf_count = nan_inf_params;

    return stats;
}

// Apply gradient clipping to model parameters
static void apply_gradient_clipping(mnist_model_t* model, float clip_threshold) {
    if (!model || clip_threshold <= 0) return;

    // Helper macro to clip gradient tensor
    #define CLIP_GRADIENT_TENSOR(grad_tensor) \
        if (grad_tensor) { \
            float* grad_data = (float*)boat_tensor_data(grad_tensor); \
            size_t num_elements = boat_tensor_nbytes(grad_tensor) / sizeof(float); \
            for (size_t i = 0; i < num_elements; i++) { \
                float g = grad_data[i]; \
                float norm = fabsf(g); \
                if (norm > clip_threshold) { \
                    grad_data[i] = g * clip_threshold / norm; \
                } \
            } \
        }

    if (model->conv1) {
        CLIP_GRADIENT_TENSOR(boat_conv_layer_get_grad_weight(model->conv1));
        CLIP_GRADIENT_TENSOR(boat_conv_layer_get_grad_bias(model->conv1));
    }
    if (model->conv2) {
        CLIP_GRADIENT_TENSOR(boat_conv_layer_get_grad_weight(model->conv2));
        CLIP_GRADIENT_TENSOR(boat_conv_layer_get_grad_bias(model->conv2));
    }
    if (model->fc1) {
        CLIP_GRADIENT_TENSOR(boat_dense_layer_get_grad_weight(model->fc1));
        CLIP_GRADIENT_TENSOR(boat_dense_layer_get_grad_bias(model->fc1));
    }
    if (model->fc2) {
        CLIP_GRADIENT_TENSOR(boat_dense_layer_get_grad_weight(model->fc2));
        CLIP_GRADIENT_TENSOR(boat_dense_layer_get_grad_bias(model->fc2));
    }

    #undef CLIP_GRADIENT_TENSOR
}

// Training stability analysis functions
typedef struct {
    float loss_mean;          // Mean loss over recent epochs
    float loss_stddev;        // Standard deviation of loss
    float loss_smoothness;    // Smoothness metric (lower = smoother)
    float val_improvement;    // Validation improvement ratio
} stability_stats_t;

// Provide hyperparameter tuning recommendations based on monitoring data
static void provide_tuning_recommendations(const gradient_stats_t* grad_stats,
                                          const stability_stats_t* stability_stats,
                                          float current_lr,
                                          double epoch_time,
                                          int batch_size,
                                          int epoch,
                                          int total_epochs) {
    printf("             tuning suggestions: ");
    bool has_suggestion = false;

    // Check for NaN/Inf gradients
    if (grad_stats->nan_inf_count > 0) {
        printf("%sWARNING: %d NaN/Inf gradient values detected, training may be unstable",
               has_suggestion ? "; " : "", grad_stats->nan_inf_count);
        has_suggestion = true;
    }

    // Check gradient magnitude
    if (grad_stats->total_norm > 10.0f) {
        printf("%sgradient norm high (%.2f), consider reducing learning rate or applying gradient clipping",
               has_suggestion ? "; " : "", grad_stats->total_norm);
        has_suggestion = true;
    } else if (grad_stats->total_norm < 0.01f) {
        printf("%sgradient norm low (%.4f), consider increasing learning rate or checking initialization",
               has_suggestion ? "; " : "", grad_stats->total_norm);
        has_suggestion = true;
    }

    // Check gradient clipping ratio
    if (grad_stats->grad_clip_ratio > 0.05f) { // More than 5% gradients would be clipped
        printf("%shigh gradient clipping ratio (%.1f%%), consider reducing learning rate",
               has_suggestion ? "; " : "", grad_stats->grad_clip_ratio * 100.0f);
        has_suggestion = true;
    }

    // Check loss stability
    if (stability_stats->loss_stddev > 0.1f && epoch > 5) {
        printf("%shigh loss variability (std=%.3f), consider reducing learning rate or increasing batch size",
               has_suggestion ? "; " : "", stability_stats->loss_stddev);
        has_suggestion = true;
    }

    // Check loss smoothness
    if (stability_stats->loss_smoothness > 0.05f && epoch > 5) {
        printf("%sloss changes abruptly (smoothness=%.3f), consider reducing learning rate",
               has_suggestion ? "; " : "", stability_stats->loss_smoothness);
        has_suggestion = true;
    }

    // Check validation improvement
    if (epoch > 5 && stability_stats->val_improvement < 0.001f) {
        printf("%slittle validation improvement (%.3f%%), consider reducing learning rate or applying early stopping",
               has_suggestion ? "; " : "", stability_stats->val_improvement * 100.0f);
        has_suggestion = true;
    }

    // Check epoch time for batch size optimization
    if (epoch_time > 2.0 && batch_size < 256) {
        printf("%sepoch time long (%.1fs), consider increasing batch size for better throughput",
               has_suggestion ? "; " : "", epoch_time);
        has_suggestion = true;
    } else if (epoch_time < 0.5 && batch_size > 32) {
        printf("%sepoch time short (%.1fs), consider decreasing batch size for better gradient estimate",
               has_suggestion ? "; " : "", epoch_time);
        has_suggestion = true;
    }

    // Learning rate specific suggestions
    if (current_lr > 0.01f) {
        printf("%slearning rate high (%.6f), consider reducing for stable training",
               has_suggestion ? "; " : "", current_lr);
        has_suggestion = true;
    } else if (current_lr < 1e-6f) {
        printf("%slearning rate very low (%.6f), consider increasing",
               has_suggestion ? "; " : "", current_lr);
        has_suggestion = true;
    }

    if (!has_suggestion) {
        printf("training appears stable, continue current settings");
    }
    printf("\n");
}

// Reinforcement Learning agent for hyperparameter tuning
#define RL_NUM_STATES 18  // 3 (grad_norm) * 3 (stability) * 2 (multi_obj_score)
#define RL_LR_ACTIONS 5   // Learning rate adjustment factors
#define RL_BS_ACTIONS 3   // Batch size adjustment factors
#define RL_MOM_ACTIONS 3  // Momentum adjustment factors
#define RL_NUM_ACTIONS (RL_LR_ACTIONS * RL_BS_ACTIONS * RL_MOM_ACTIONS)  // Total action combinations
#define RL_LEARNING_RATE 0.1f
#define RL_DISCOUNT_FACTOR 0.9f
#define RL_EXPLORATION_RATE 0.2f

// Adaptive action space constants
#define RL_TRAINING_STAGES 3
#define RL_EARLY_STAGE 0
#define RL_MID_STAGE 1
#define RL_LATE_STAGE 2
#define RL_EARLY_STAGE_THRESHOLD 0.33f  // First 33% of training
#define RL_MID_STAGE_THRESHOLD 0.66f    // Next 33% of training

// Stage-dependent exploration decay factors
#define RL_EXPLORATION_DECAY_EARLY 0.995f  // Slow decay in early stage
#define RL_EXPLORATION_DECAY_MID   0.990f  // Medium decay in mid stage
#define RL_EXPLORATION_DECAY_LATE  0.985f  // Faster decay in late stage

// Uncertainty-based action selection constants
#define RL_UCB_EXPLORATION_CONSTANT 2.0f  // UCB exploration constant

typedef struct {
    float q_table[RL_NUM_STATES][RL_NUM_ACTIONS];
    float exploration_rate;
    float min_exploration_rate;
    float exploration_decay;
    float learning_rate;
    float discount_factor;
    int last_state;
    int last_action;
    float previous_multi_objective_score;
    bool has_previous_score;
    // Adaptive action space tracking
    int current_stage;                     // Current training stage (0=early, 1=mid, 2=late)
    bool action_mask[RL_NUM_ACTIONS];      // Mask of available actions for current stage
    int num_available_actions;             // Number of available actions
    int available_actions[RL_NUM_ACTIONS]; // List of available action indices
    // Uncertainty-based action selection tracking
    int state_visits[RL_NUM_STATES];                    // Visit counts for each state
    int action_visits[RL_NUM_STATES][RL_NUM_ACTIONS];   // Visit counts for each state-action pair
} rl_agent_t;

// RL action definitions
static const float rl_lr_factors[RL_LR_ACTIONS] = {0.5f, 0.8f, 1.0f, 1.2f, 2.0f};
static const float rl_bs_factors[RL_BS_ACTIONS] = {0.5f, 1.0f, 2.0f};
static const float rl_mom_factors[RL_MOM_ACTIONS] = {0.9f, 1.0f, 1.1f};
static const char* rl_lr_descriptions[RL_LR_ACTIONS] = {
    "halve learning rate",
    "slightly reduce learning rate",
    "keep learning rate unchanged",
    "slightly increase learning rate",
    "double learning rate"
};
static const char* rl_bs_descriptions[RL_BS_ACTIONS] = {
    "halve batch size",
    "keep batch size unchanged",
    "double batch size"
};
static const char* rl_mom_descriptions[RL_MOM_ACTIONS] = {
    "reduce momentum",
    "keep momentum unchanged",
    "increase momentum"
};

// Action decoding helper functions
static inline int decode_lr_action(int action) {
    return action % RL_LR_ACTIONS;
}
static inline int decode_bs_action(int action) {
    return (action / RL_LR_ACTIONS) % RL_BS_ACTIONS;
}
static inline int decode_mom_action(int action) {
    return (action / (RL_LR_ACTIONS * RL_BS_ACTIONS)) % RL_MOM_ACTIONS;
}
static inline float get_lr_factor(int action) {
    return rl_lr_factors[decode_lr_action(action)];
}
static inline float get_bs_factor(int action) {
    return rl_bs_factors[decode_bs_action(action)];
}
static inline float get_mom_factor(int action) {
    return rl_mom_factors[decode_mom_action(action)];
}
static inline void get_action_description(int action, char* buffer, size_t buffer_size) {
    int lr_a = decode_lr_action(action);
    int bs_a = decode_bs_action(action);
    int mom_a = decode_mom_action(action);
    snprintf(buffer, buffer_size, "LR: %s; BS: %s; Mom: %s",
             rl_lr_descriptions[lr_a], rl_bs_descriptions[bs_a], rl_mom_descriptions[mom_a]);
}

// Adaptive action space helper functions
static inline int rl_agent_get_stage(int epoch, int total_epochs) {
    if (total_epochs <= 0) return RL_EARLY_STAGE;

    float epoch_ratio = (float)epoch / (float)total_epochs;
    if (epoch_ratio < RL_EARLY_STAGE_THRESHOLD) {
        return RL_EARLY_STAGE;
    } else if (epoch_ratio < RL_MID_STAGE_THRESHOLD) {
        return RL_MID_STAGE;
    } else {
        return RL_LATE_STAGE;
    }
}

static inline bool rl_agent_is_action_available(int action, int stage) {
    // Early stage: only allow moderate adjustments (no extreme changes)
    // Disable extreme learning rate changes (halving and doubling) and batch size doubling
    int lr_action = decode_lr_action(action);
    int bs_action = decode_bs_action(action);

    switch (stage) {
        case RL_EARLY_STAGE:
            // Disable extreme LR changes (indices 0 and 4: halve and double)
            // Disable batch size doubling (index 2)
            // Keep all momentum adjustments
            if (lr_action == 0 || lr_action == 4) return false;  // No halving/doubling LR
            if (bs_action == 2) return false;                    // No doubling batch size
            return true;

        case RL_MID_STAGE:
            // Allow all LR adjustments except halving (too aggressive for mid-stage)
            // Allow all batch size adjustments
            // Allow all momentum adjustments
            if (lr_action == 0) return false;  // No halving LR
            return true;

        case RL_LATE_STAGE:
            // Allow all actions in late stage
            return true;

        default:
            return true;
    }
}

static void rl_agent_update_action_mask(rl_agent_t* agent, int stage) {
    if (!agent) return;

    agent->current_stage = stage;
    agent->num_available_actions = 0;

    // Initialize all actions as unavailable
    for (int a = 0; a < RL_NUM_ACTIONS; a++) {
        agent->action_mask[a] = false;
    }

    // Build list of available actions for this stage
    for (int a = 0; a < RL_NUM_ACTIONS; a++) {
        if (rl_agent_is_action_available(a, stage)) {
            agent->action_mask[a] = true;
            agent->available_actions[agent->num_available_actions] = a;
            agent->num_available_actions++;
        }
    }

    // Ensure at least one action is available (fallback to default no-change action)
    if (agent->num_available_actions == 0) {
        // Default: no change (action index 2 for LR, 1 for BS, 1 for Mom)
        int default_action = 2 + RL_LR_ACTIONS * 1 + (RL_LR_ACTIONS * RL_BS_ACTIONS) * 1;
        agent->action_mask[default_action] = true;
        agent->available_actions[0] = default_action;
        agent->num_available_actions = 1;
    }
}

// Initialize RL agent
static rl_agent_t rl_agent_init() {
    rl_agent_t agent;
    agent.exploration_rate = RL_EXPLORATION_RATE;
    agent.min_exploration_rate = 0.01f;
    agent.exploration_decay = 0.995f;
    agent.learning_rate = RL_LEARNING_RATE;
    agent.discount_factor = RL_DISCOUNT_FACTOR;
    agent.last_state = -1;
    agent.last_action = -1;
    agent.previous_multi_objective_score = 0.0f;
    agent.has_previous_score = false;

    // Initialize adaptive action space fields
    agent.current_stage = RL_EARLY_STAGE;  // Start in early stage
    agent.num_available_actions = 0;
    // Initialize action mask (will be updated when stage is known)
    for (int a = 0; a < RL_NUM_ACTIONS; a++) {
        agent.action_mask[a] = true;  // Initially all actions available
        agent.available_actions[a] = a;
    }
    agent.num_available_actions = RL_NUM_ACTIONS;

    // Initialize Q-table with small random values
    for (int s = 0; s < RL_NUM_STATES; s++) {
        for (int a = 0; a < RL_NUM_ACTIONS; a++) {
            agent.q_table[s][a] = 0.0f; // Start with zero, exploration will fill
        }
    }

    // Initialize uncertainty tracking fields
    for (int s = 0; s < RL_NUM_STATES; s++) {
        agent.state_visits[s] = 0;
        for (int a = 0; a < RL_NUM_ACTIONS; a++) {
            agent.action_visits[s][a] = 0;
        }
    }

    return agent;
}

// Map continuous metrics to discrete state
static int rl_agent_get_state(float grad_norm, float loss_stddev, float multi_objective_score) {
    int grad_state;
    if (grad_norm < 0.01f) grad_state = 0;
    else if (grad_norm <= 10.0f) grad_state = 1;
    else grad_state = 2;

    int stability_state;
    if (loss_stddev < 0.05f) stability_state = 0;
    else if (loss_stddev <= 0.1f) stability_state = 1;
    else stability_state = 2;

    int score_state = multi_objective_score < 0.5f ? 0 : 1;

    return (grad_state * 3 + stability_state) * 2 + score_state;
}

// Select action using epsilon-greedy policy with adaptive action space
static int rl_agent_select_action(rl_agent_t* agent, int state) {
    if (!agent) return 2; // Default: no change (action index 2)

    // Exploration: random action from available actions
    if ((float)rand() / RAND_MAX < agent->exploration_rate) {
        // If no available actions (should not happen), return default
        if (agent->num_available_actions == 0) return 2;

        // Select random action from available actions
        int random_idx = rand() % agent->num_available_actions;
        return agent->available_actions[random_idx];
    }

    // Exploitation: choose available action using UCB (Upper Confidence Bound)
    int best_action = -1;
    float best_ucb_score = -1e9f;  // Very small initial value

    // Calculate total visits for this state (add 1 to avoid log(0))
    int state_visits_total = agent->state_visits[state] + 1;
    float log_state_visits = log((float)state_visits_total);

    // Find best action based on UCB score
    for (int i = 0; i < agent->num_available_actions; i++) {
        int a = agent->available_actions[i];
        int action_visits = agent->action_visits[state][a] + 1; // Add 1 to avoid division by zero
        float exploration_bonus = RL_UCB_EXPLORATION_CONSTANT * sqrt(log_state_visits / (float)action_visits);
        float ucb_score = agent->q_table[state][a] + exploration_bonus;

        if (best_action == -1 || ucb_score > best_ucb_score) {
            best_ucb_score = ucb_score;
            best_action = a;
        }
    }

    // Fallback: if no available action found (should not happen), return default
    if (best_action == -1) {
        return 2; // Default: no change
    }

    return best_action;
}

// Update Q-table using Q-learning
static void rl_agent_update(rl_agent_t* agent, int new_state, float reward) {
    if (!agent || agent->last_state == -1 || agent->last_action == -1) {
        return;
    }

    // Q-learning update
    float old_q = agent->q_table[agent->last_state][agent->last_action];

    // Find max Q for new state among available actions
    float max_q_new = -1e9f;  // Very small initial value
    bool found_available = false;

    // Search through available actions for this stage
    for (int i = 0; i < agent->num_available_actions; i++) {
        int a = agent->available_actions[i];
        if (agent->q_table[new_state][a] > max_q_new) {
            max_q_new = agent->q_table[new_state][a];
            found_available = true;
        }
    }

    // Fallback: if no available actions (should not happen), use all actions
    if (!found_available) {
        max_q_new = agent->q_table[new_state][0];
        for (int a = 1; a < RL_NUM_ACTIONS; a++) {
            if (agent->q_table[new_state][a] > max_q_new) {
                max_q_new = agent->q_table[new_state][a];
            }
        }
    }

    // Update Q-value
    agent->q_table[agent->last_state][agent->last_action] =
        old_q + agent->learning_rate * (reward + agent->discount_factor * max_q_new - old_q);

    // Update visit counts for uncertainty-based exploration
    agent->state_visits[agent->last_state]++;
    agent->action_visits[agent->last_state][agent->last_action]++;
}

// Auto-tuning decision structure
typedef struct {
    float lr_adjustment_factor;    // Learning rate adjustment (e.g., 0.5 for halving)
    float new_clip_threshold;      // New gradient clipping threshold
    bool apply_clipping;           // Whether to apply gradient clipping
    float batch_size_factor;       // Batch size adjustment factor
    float beta1_adjustment_factor; // Beta1 (momentum) adjustment factor
    float beta2_adjustment_factor; // Beta2 (RMSprop) adjustment factor
    bool should_reduce_lr;         // Explicit flag to reduce learning rate
    bool should_increase_lr;       // Explicit flag to increase learning rate
    char explanation[256];         // Explanation of the decision
} auto_tuning_decision_t;

// Global RL agent (shared across training)
static rl_agent_t rl_agent;
static bool rl_agent_initialized = false;

// Generate auto-tuning decisions based on monitoring data
static auto_tuning_decision_t auto_tuning_strategy(const gradient_stats_t* grad_stats,
                                                  const stability_stats_t* stability_stats,
                                                  float current_lr,
                                                  double epoch_time,
                                                  int batch_size,
                                                  int epoch,
                                                  int total_epochs,
                                                  float current_clip_threshold,
                                                  float throughput_sps,
                                                  float memory_usage_mb,
                                                  float validation_accuracy) {
    auto_tuning_decision_t decision = {
        .lr_adjustment_factor = 1.0f,
        .new_clip_threshold = current_clip_threshold,
        .apply_clipping = false,
        .batch_size_factor = 1.0f,
        .beta1_adjustment_factor = 1.0f,
        .beta2_adjustment_factor = 1.0f,
        .should_reduce_lr = false,
        .should_increase_lr = false,
        .explanation = ""
    };

    // Calculate multi-objective score (needed for RL agent and multi-objective optimization)
    const float throughput_target = 5000.0f;
    const float memory_target = 200.0f;
    const float accuracy_target = 0.95f;

    float throughput_score = throughput_sps > 0 ? fminf(throughput_sps / throughput_target, 1.0f) : 0.0f;
    float memory_score = memory_usage_mb > 0 ? fmaxf(0.0f, 1.0f - (memory_usage_mb / memory_target)) : 1.0f;
    float accuracy_score = validation_accuracy / accuracy_target;

    // Dynamic weights based on training stage
    float weight_throughput = 0.3f;
    float weight_memory = 0.3f;
    float weight_accuracy = 0.4f;

    if (epoch < total_epochs / 3) {
        weight_throughput = 0.4f; weight_memory = 0.4f; weight_accuracy = 0.2f;
    } else if (epoch > total_epochs * 2 / 3) {
        weight_throughput = 0.2f; weight_memory = 0.2f; weight_accuracy = 0.6f;
    }

    float multi_objective_score = weight_throughput * throughput_score +
                                  weight_memory * memory_score +
                                  weight_accuracy * accuracy_score;

    // Reinforcement Learning tuning integration (iteration 21)
    if (!rl_agent_initialized) {
        rl_agent = rl_agent_init();
        rl_agent_initialized = true;
        printf("             RL agent initialized\n");
    }

    // Adaptive action space integration (iteration 23)
    // Update RL agent stage and action mask based on current training progress
    int rl_stage = rl_agent_get_stage(epoch, total_epochs);
    if (rl_agent.current_stage != rl_stage) {
        rl_agent_update_action_mask(&rl_agent, rl_stage);
        printf("             RL agent stage updated: %s (epoch %d/%d, available actions: %d/%d)\n",
               rl_stage == RL_EARLY_STAGE ? "early" : (rl_stage == RL_MID_STAGE ? "mid" : "late"),
               epoch + 1, total_epochs, rl_agent.num_available_actions, RL_NUM_ACTIONS);
    }

    // Get current state for RL agent
    int rl_current_state = rl_agent_get_state(grad_stats->total_norm, stability_stats->loss_stddev, multi_objective_score);

    // Select action using RL agent
    int rl_action = rl_agent_select_action(&rl_agent, rl_current_state);
    float rl_lr_factor = get_lr_factor(rl_action);
    float rl_bs_factor = get_bs_factor(rl_action);
    float rl_mom_factor = get_mom_factor(rl_action);

    // Apply RL-based hyperparameter adjustments
    char rl_description[256];
    get_action_description(rl_action, rl_description, sizeof(rl_description));

    // Learning rate adjustment
    if (rl_lr_factor != 1.0f && !decision.should_reduce_lr && !decision.should_increase_lr) {
        decision.lr_adjustment_factor = rl_lr_factor;
        if (rl_lr_factor < 1.0f) decision.should_reduce_lr = true;
        if (rl_lr_factor > 1.0f) decision.should_increase_lr = true;
    }

    // Batch size adjustment (if not already adjusted by rules)
    if (rl_bs_factor != 1.0f && decision.batch_size_factor == 1.0f) {
        decision.batch_size_factor = rl_bs_factor;
    }

    // Momentum adjustment (if not already adjusted by rules)
    if (rl_mom_factor != 1.0f && decision.beta1_adjustment_factor == 1.0f) {
        decision.beta1_adjustment_factor = rl_mom_factor;
        decision.beta2_adjustment_factor = rl_mom_factor; // Adjust both beta1 and beta2 together
    }

    // Add RL explanation if any adjustment was applied
    if (rl_lr_factor != 1.0f || rl_bs_factor != 1.0f || rl_mom_factor != 1.0f) {
        if (strlen(decision.explanation) > 0) strcat(decision.explanation, "; ");
        strcat(decision.explanation, "RL-based: ");
        strcat(decision.explanation, rl_description);
    }

    // Store state and action for next update (reward will be calculated later)
    rl_agent.last_state = rl_current_state;
    rl_agent.last_action = rl_action;

    // Default explanation
    strcpy(decision.explanation, "No adjustment needed");

    // 1. Learning rate adjustments based on gradient statistics
    if (grad_stats->total_norm > 10.0f) {
        decision.lr_adjustment_factor = 0.5f; // Reduce learning rate by half
        decision.should_reduce_lr = true;
        snprintf(decision.explanation, sizeof(decision.explanation),
                "High gradient norm (%.2f), reducing learning rate", grad_stats->total_norm);
    } else if (grad_stats->total_norm < 0.01f) {
        decision.lr_adjustment_factor = 2.0f; // Double learning rate
        decision.should_increase_lr = true;
        snprintf(decision.explanation, sizeof(decision.explanation),
                "Low gradient norm (%.4f), increasing learning rate", grad_stats->total_norm);
    }

    // 2. Gradient clipping threshold adjustment
    if (grad_stats->grad_clip_ratio > 0.05f) {
        // If more than 5% gradients would be clipped, reduce threshold
        decision.new_clip_threshold = current_clip_threshold * 0.8f; // Reduce by 20%
        decision.apply_clipping = true;
        if (strlen(decision.explanation) > 0) {
            strcat(decision.explanation, "; ");
        }
        char clip_msg[128];
        snprintf(clip_msg, sizeof(clip_msg),
                "High clipping ratio (%.1f%%), reducing clip threshold to %.2f",
                grad_stats->grad_clip_ratio * 100.0f, decision.new_clip_threshold);
        strcat(decision.explanation, clip_msg);
    } else if (grad_stats->grad_clip_ratio < 0.01f && current_clip_threshold < 10.0f) {
        // If very few gradients would be clipped, increase threshold
        decision.new_clip_threshold = current_clip_threshold * 1.2f; // Increase by 20%
        if (strlen(decision.explanation) > 0) {
            strcat(decision.explanation, "; ");
        }
        char clip_msg[128];
        snprintf(clip_msg, sizeof(clip_msg),
                "Low clipping ratio (%.1f%%), increasing clip threshold to %.2f",
                grad_stats->grad_clip_ratio * 100.0f, decision.new_clip_threshold);
        strcat(decision.explanation, clip_msg);
    }

    // 3. Training stability adjustments
    if (stability_stats->loss_stddev > 0.1f && epoch > 5) {
        // High loss variability - reduce learning rate
        if (!decision.should_reduce_lr && !decision.should_increase_lr) {
            decision.lr_adjustment_factor = 0.7f;
            decision.should_reduce_lr = true;
        } else if (decision.should_increase_lr) {
            // If we were going to increase LR but have instability, cancel increase
            decision.lr_adjustment_factor = 1.0f;
            decision.should_increase_lr = false;
        }
        if (strlen(decision.explanation) > 0) {
            strcat(decision.explanation, "; ");
        }
        char stability_msg[128];
        snprintf(stability_msg, sizeof(stability_msg),
                "High loss variability (std=%.3f), adjusting learning rate",
                stability_stats->loss_stddev);
        strcat(decision.explanation, stability_msg);
    }

    // 4. Batch size optimization based on epoch time
    if (epoch_time > 2.0 && batch_size < 256) {
        decision.batch_size_factor = 1.5f; // Increase batch size by 50%
        if (strlen(decision.explanation) > 0) {
            strcat(decision.explanation, "; ");
        }
        char batch_msg[128];
        snprintf(batch_msg, sizeof(batch_msg),
                "Long epoch time (%.1fs), increasing batch size", epoch_time);
        strcat(decision.explanation, batch_msg);
    } else if (epoch_time < 0.5 && batch_size > 32) {
        decision.batch_size_factor = 0.67f; // Reduce batch size by 33%
        if (strlen(decision.explanation) > 0) {
            strcat(decision.explanation, "; ");
        }
        char batch_msg[128];
        snprintf(batch_msg, sizeof(batch_msg),
                "Short epoch time (%.1fs), decreasing batch size", epoch_time);
        strcat(decision.explanation, batch_msg);
    }

    // 5. Learning rate magnitude check
    if (current_lr > 0.01f && !decision.should_increase_lr) {
        // Learning rate is too high, reduce it
        if (!decision.should_reduce_lr) {
            decision.lr_adjustment_factor = 0.5f;
            decision.should_reduce_lr = true;
        }
        if (strlen(decision.explanation) > 0) {
            strcat(decision.explanation, "; ");
        }
        char lr_msg[128];
        snprintf(lr_msg, sizeof(lr_msg),
                "Learning rate high (%.6f), reducing", current_lr);
        strcat(decision.explanation, lr_msg);
    } else if (current_lr < 1e-6f && !decision.should_reduce_lr) {
        // Learning rate is too low, increase it
        if (!decision.should_increase_lr) {
            decision.lr_adjustment_factor = 2.0f;
            decision.should_increase_lr = true;
        }
        if (strlen(decision.explanation) > 0) {
            strcat(decision.explanation, "; ");
        }
        char lr_msg[128];
        snprintf(lr_msg, sizeof(lr_msg),
                "Learning rate very low (%.6f), increasing", current_lr);
        strcat(decision.explanation, lr_msg);
    }

    // 6. Momentum parameter optimization based on gradient statistics and stability
    if (grad_stats->total_norm > 5.0f && stability_stats->loss_stddev > 0.05f) {
        // High gradient norm and unstable training: reduce momentum to adapt faster
        decision.beta1_adjustment_factor = 0.8f; // Reduce beta1 by 20%
        if (strlen(decision.explanation) > 0) {
            strcat(decision.explanation, "; ");
        }
        char beta_msg[128];
        snprintf(beta_msg, sizeof(beta_msg),
                "High gradient norm (%.2f) and instability (std=%.3f), reducing momentum (beta1)",
                grad_stats->total_norm, stability_stats->loss_stddev);
        strcat(decision.explanation, beta_msg);
    } else if (grad_stats->total_norm < 0.1f && stability_stats->loss_stddev < 0.02f) {
        // Low gradient norm and stable training: increase momentum for smoother updates
        decision.beta1_adjustment_factor = 1.2f; // Increase beta1 by 20%
        if (strlen(decision.explanation) > 0) {
            strcat(decision.explanation, "; ");
        }
        char beta_msg[128];
        snprintf(beta_msg, sizeof(beta_msg),
                "Low gradient norm (%.4f) and stable training (std=%.3f), increasing momentum (beta1)",
                grad_stats->total_norm, stability_stats->loss_stddev);
        strcat(decision.explanation, beta_msg);
    }

    // Beta2 (RMSprop) adjustment based on gradient clipping ratio
    if (grad_stats->grad_clip_ratio > 0.1f) {
        // Many gradients would be clipped: reduce beta2 to give more weight to recent gradients
        decision.beta2_adjustment_factor = 0.9f; // Reduce beta2 by 10%
        if (strlen(decision.explanation) > 0) {
            strcat(decision.explanation, "; ");
        }
        char beta2_msg[128];
        snprintf(beta2_msg, sizeof(beta2_msg),
                "High clipping ratio (%.1f%%), reducing beta2 for faster adaptation",
                grad_stats->grad_clip_ratio * 100.0f);
        strcat(decision.explanation, beta2_msg);
    } else if (grad_stats->grad_clip_ratio < 0.01f && grad_stats->total_norm > 0.5f) {
        // Few gradients clipped but decent gradient norm: increase beta2 for stability
        decision.beta2_adjustment_factor = 1.1f; // Increase beta2 by 10%
        if (strlen(decision.explanation) > 0) {
            strcat(decision.explanation, "; ");
        }
        char beta2_msg[128];
        snprintf(beta2_msg, sizeof(beta2_msg),
                "Low clipping ratio (%.1f%%) with good gradient flow, increasing beta2 for stability",
                grad_stats->grad_clip_ratio * 100.0f);
        strcat(decision.explanation, beta2_msg);
    }

    // Multi-objective optimization (iteration 20): balance throughput, memory, accuracy
    // (using pre-calculated scores from above)
    {
        // Apply multi-objective adjustments
        if (multi_objective_score < 0.5f) {
            // Poor overall performance, consider broader adjustments
            if (throughput_score < 0.3f && memory_score > 0.7f) {
                // Low throughput but good memory: increase batch size
                decision.batch_size_factor *= 1.2f;
                if (strlen(decision.explanation) > 0) strcat(decision.explanation, "; ");
                strcat(decision.explanation, "multi-objective: low throughput, increasing batch size");
            }
            if (memory_score < 0.3f) {
                // High memory usage: reduce batch size
                decision.batch_size_factor *= 0.8f;
                if (strlen(decision.explanation) > 0) strcat(decision.explanation, "; ");
                strcat(decision.explanation, "multi-objective: high memory usage, reducing batch size");
            }
            if (accuracy_score < 0.5f && epoch > 5) {
                // Low accuracy: adjust learning rate
                decision.lr_adjustment_factor *= 1.1f;
                if (strlen(decision.explanation) > 0) strcat(decision.explanation, "; ");
                strcat(decision.explanation, "multi-objective: low accuracy, increasing learning rate");
            }
        }

        // Log multi-objective metrics (debug)
        if (epoch % 5 == 0) {
            printf("             multi-objective: throughput=%.0f sps (score=%.2f), memory=%.1f MB (score=%.2f), accuracy=%.2f%% (score=%.2f), combined=%.2f\n",
                   throughput_sps, throughput_score, memory_usage_mb, memory_score,
                   validation_accuracy * 100.0f, accuracy_score, multi_objective_score);
        }
    }

    // RL agent update with reward based on multi-objective score improvement
    if (rl_agent_initialized && rl_agent.has_previous_score && rl_agent.last_state != -1 && rl_agent.last_action != -1) {
        // Reward is improvement in multi-objective score
        float reward = multi_objective_score - rl_agent.previous_multi_objective_score;
        // Recompute current state (may have changed due to other adjustments)
        int rl_new_state = rl_agent_get_state(grad_stats->total_norm, stability_stats->loss_stddev, multi_objective_score);
        // Update Q-table
        rl_agent_update(&rl_agent, rl_new_state, reward);
        // Decay exploration rate with stage-dependent decay factor
        float stage_decay_factor = RL_EXPLORATION_DECAY_MID; // Default
        switch (rl_agent.current_stage) {
            case RL_EARLY_STAGE: stage_decay_factor = RL_EXPLORATION_DECAY_EARLY; break;
            case RL_MID_STAGE:   stage_decay_factor = RL_EXPLORATION_DECAY_MID;   break;
            case RL_LATE_STAGE:  stage_decay_factor = RL_EXPLORATION_DECAY_LATE;  break;
        }
        rl_agent.exploration_rate *= stage_decay_factor;
        if (rl_agent.exploration_rate < rl_agent.min_exploration_rate) {
            rl_agent.exploration_rate = rl_agent.min_exploration_rate;
        }
        // Log RL update occasionally
        if (epoch % 10 == 0) {
            printf("             RL update: state=%d, action=%d, reward=%.3f, new Q=%.3f, exploration=%.3f\n",
                   rl_agent.last_state, rl_agent.last_action, reward,
                   rl_agent.q_table[rl_agent.last_state][rl_agent.last_action],
                   rl_agent.exploration_rate);
        }
    }

    // Update previous score for next epoch
    rl_agent.previous_multi_objective_score = multi_objective_score;
    rl_agent.has_previous_score = true;

    // Ensure adjustment factor is within reasonable bounds
    if (decision.lr_adjustment_factor < 0.1f) decision.lr_adjustment_factor = 0.1f;
    if (decision.lr_adjustment_factor > 5.0f) decision.lr_adjustment_factor = 5.0f;

    // Ensure clip threshold is within reasonable bounds
    if (decision.new_clip_threshold < 0.1f) decision.new_clip_threshold = 0.1f;
    if (decision.new_clip_threshold > 50.0f) decision.new_clip_threshold = 50.0f;

    // Ensure momentum adjustment factors are within reasonable bounds
    if (decision.beta1_adjustment_factor < 0.5f) decision.beta1_adjustment_factor = 0.5f;
    if (decision.beta1_adjustment_factor > 1.5f) decision.beta1_adjustment_factor = 1.5f;
    if (decision.beta2_adjustment_factor < 0.8f) decision.beta2_adjustment_factor = 0.8f;
    if (decision.beta2_adjustment_factor > 1.2f) decision.beta2_adjustment_factor = 1.2f;

    return decision;
}

// Recreate optimizer with new beta1/beta2 parameters while preserving registered parameters
static bool recreate_optimizer_with_new_betas(mnist_model_t* model, float new_beta1, float new_beta2) {
    if (!model || !model->optimizer) return false;

    // Get current learning rate
    float current_lr = boat_optimizer_get_learning_rate(model->optimizer);

    // Store old optimizer
    boat_optimizer_t* old_optimizer = model->optimizer;

    // Create new optimizer with new betas
    boat_optimizer_t* new_optimizer = boat_adam_optimizer_create(current_lr, new_beta1, new_beta2, 1e-8f);
    if (!new_optimizer) {
        fprintf(stderr, "Warning: Failed to recreate optimizer with new betas (beta1=%.3f, beta2=%.3f)\n",
                new_beta1, new_beta2);
        return false;
    }

    // Re-register all parameters from the model layers
    // Conv1 weight and bias
    boat_tensor_t* conv1_weight = boat_conv_layer_get_weight(model->conv1);
    boat_tensor_t* conv1_grad_weight = boat_conv_layer_get_grad_weight(model->conv1);
    if (conv1_weight && conv1_grad_weight) {
        boat_optimizer_add_parameter(new_optimizer, conv1_weight, conv1_grad_weight);
    }
    boat_tensor_t* conv1_bias = boat_conv_layer_get_bias(model->conv1);
    boat_tensor_t* conv1_grad_bias = boat_conv_layer_get_grad_bias(model->conv1);
    if (conv1_bias && conv1_grad_bias) {
        boat_optimizer_add_parameter(new_optimizer, conv1_bias, conv1_grad_bias);
    }

    // Conv2 weight and bias
    boat_tensor_t* conv2_weight = boat_conv_layer_get_weight(model->conv2);
    boat_tensor_t* conv2_grad_weight = boat_conv_layer_get_grad_weight(model->conv2);
    if (conv2_weight && conv2_grad_weight) {
        boat_optimizer_add_parameter(new_optimizer, conv2_weight, conv2_grad_weight);
    }
    boat_tensor_t* conv2_bias = boat_conv_layer_get_bias(model->conv2);
    boat_tensor_t* conv2_grad_bias = boat_conv_layer_get_grad_bias(model->conv2);
    if (conv2_bias && conv2_grad_bias) {
        boat_optimizer_add_parameter(new_optimizer, conv2_bias, conv2_grad_bias);
    }

    // FC1 weight and bias
    boat_tensor_t* fc1_weight = boat_dense_layer_get_weight(model->fc1);
    boat_tensor_t* fc1_grad_weight = boat_dense_layer_get_grad_weight(model->fc1);
    if (fc1_weight && fc1_grad_weight) {
        boat_optimizer_add_parameter(new_optimizer, fc1_weight, fc1_grad_weight);
    }
    boat_tensor_t* fc1_bias = boat_dense_layer_get_bias(model->fc1);
    boat_tensor_t* fc1_grad_bias = boat_dense_layer_get_grad_bias(model->fc1);
    if (fc1_bias && fc1_grad_bias) {
        boat_optimizer_add_parameter(new_optimizer, fc1_bias, fc1_grad_bias);
    }

    // FC2 weight and bias
    boat_tensor_t* fc2_weight = boat_dense_layer_get_weight(model->fc2);
    boat_tensor_t* fc2_grad_weight = boat_dense_layer_get_grad_weight(model->fc2);
    if (fc2_weight && fc2_grad_weight) {
        boat_optimizer_add_parameter(new_optimizer, fc2_weight, fc2_grad_weight);
    }
    boat_tensor_t* fc2_bias = boat_dense_layer_get_bias(model->fc2);
    boat_tensor_t* fc2_grad_bias = boat_dense_layer_get_grad_bias(model->fc2);
    if (fc2_bias && fc2_grad_bias) {
        boat_optimizer_add_parameter(new_optimizer, fc2_bias, fc2_grad_bias);
    }

    // Replace optimizer
    model->optimizer = new_optimizer;
    model->current_beta1 = new_beta1;
    model->current_beta2 = new_beta2;

    // Free old optimizer
    boat_optimizer_free(old_optimizer);

    printf("             auto-tuning: optimizer recreated with new beta1=%.3f, beta2=%.3f\n",
           new_beta1, new_beta2);
    return true;
}

// Apply auto-tuning decisions to the model
static void apply_auto_tuning_decisions(mnist_model_t* model,
                                       const auto_tuning_decision_t* decision,
                                       float* current_lr,
                                       float* current_clip_threshold,
                                       size_t* batch_size,
                                       size_t* num_batches,
                                       size_t train_samples) {
    if (!model || !decision) return;

    // 1. Apply learning rate adjustment
    if (decision->lr_adjustment_factor != 1.0f && model->optimizer) {
        float old_lr = boat_optimizer_get_learning_rate(model->optimizer);
        float new_lr = old_lr * decision->lr_adjustment_factor;
        boat_optimizer_set_learning_rate(model->optimizer, new_lr);
        *current_lr = new_lr;
        printf("             auto-tuning: learning rate adjusted from %.6f to %.6f (factor: %.2f)\n",
               old_lr, new_lr, decision->lr_adjustment_factor);
    }

    // 2. Update gradient clipping threshold
    if (decision->new_clip_threshold != *current_clip_threshold) {
        printf("             auto-tuning: gradient clip threshold changed from %.2f to %.2f\n",
               *current_clip_threshold, decision->new_clip_threshold);
        *current_clip_threshold = decision->new_clip_threshold;
    }

    // 3. Apply gradient clipping if requested
    if (decision->apply_clipping && model->optimizer) {
        apply_gradient_clipping(model, *current_clip_threshold);
        printf("             auto-tuning: gradient clipping applied with threshold %.2f\n",
               *current_clip_threshold);
    }

    // 4. Batch size adjustment with dynamic training loop restructuring
    if (decision->batch_size_factor != 1.0f) {
        size_t new_batch_size = (size_t)(*batch_size * decision->batch_size_factor);
        // Clamp to reasonable range
        if (new_batch_size < 8) new_batch_size = 8;
        if (new_batch_size > 512) new_batch_size = 512;

        if (new_batch_size != *batch_size) {
            printf("             auto-tuning: batch size changed: %zu -> %zu (factor: %.2f)\n",
                   *batch_size, new_batch_size, decision->batch_size_factor);

            // Update batch size
            size_t old_batch_size = *batch_size;
            *batch_size = new_batch_size;

            // Recompute number of batches based on new batch size
            if (train_samples > 0 && num_batches) {
                *num_batches = (train_samples + new_batch_size - 1) / new_batch_size;
                printf("             auto-tuning: number of batches updated: %zu\n", *num_batches);
            }

            // Apply learning rate scaling rule (linear scaling with batch size)
            // Only apply if learning rate hasn't been adjusted by other rules
            if (decision->lr_adjustment_factor == 1.0f && model->optimizer) {
                float lr_scale_factor = (float)new_batch_size / (float)old_batch_size;
                float old_lr = boat_optimizer_get_learning_rate(model->optimizer);
                float new_lr = old_lr * lr_scale_factor;
                boat_optimizer_set_learning_rate(model->optimizer, new_lr);
                *current_lr = new_lr;
                printf("             auto-tuning: learning rate scaled with batch size: %.6f -> %.6f (factor: %.2f)\n",
                       old_lr, new_lr, lr_scale_factor);
            }
        }
    }

    // 5. Momentum parameter adjustment
    if ((decision->beta1_adjustment_factor != 1.0f || decision->beta2_adjustment_factor != 1.0f) && model->optimizer) {
        float new_beta1 = model->current_beta1 * decision->beta1_adjustment_factor;
        float new_beta2 = model->current_beta2 * decision->beta2_adjustment_factor;

        // Clamp to valid ranges for Adam
        if (new_beta1 < 0.1f) new_beta1 = 0.1f;
        if (new_beta1 > 0.999f) new_beta1 = 0.999f;
        if (new_beta2 < 0.1f) new_beta2 = 0.1f;
        if (new_beta2 > 0.999f) new_beta2 = 0.999f;

        if (new_beta1 != model->current_beta1 || new_beta2 != model->current_beta2) {
            if (recreate_optimizer_with_new_betas(model, new_beta1, new_beta2)) {
                printf("             auto-tuning: momentum parameters adjusted: beta1 %.3f->%.3f (factor: %.2f), beta2 %.3f->%.3f (factor: %.2f)\n",
                       model->current_beta1, new_beta1, decision->beta1_adjustment_factor,
                       model->current_beta2, new_beta2, decision->beta2_adjustment_factor);
            }
        }
    }

    // 6. Log the overall explanation
    if (strlen(decision->explanation) > 0 &&
        strcmp(decision->explanation, "No adjustment needed") != 0) {
        printf("             auto-tuning rationale: %s\n", decision->explanation);
    }
}

// Profile layer-wise forward pass performance
static void profile_layer_performance(mnist_model_t* model, boat_tensor_t* sample_input) {
    printf("             layer performance analysis:\n");

    // Create variable from input tensor
    boat_variable_t* x = boat_variable_create(sample_input, false);
    if (!x) {
        printf("               failed to create input variable\n");
        return;
    }

    // Layer names for reporting
    const char* layer_names[] = {
        "conv1", "relu1", "pool1",
        "conv2", "relu2", "pool2",
        "flatten", "fc1", "relu3", "fc2"
    };
    float layer_times[10] = {0};

    clock_t start, end;

    // Conv1
    start = clock();
    boat_variable_t* x_conv1 = boat_var_conv(x, model->conv1);
    end = clock();
    layer_times[0] = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f; // ms

    // ReLU1
    start = clock();
    boat_variable_t* x_relu1 = boat_var_relu(x_conv1);
    end = clock();
    layer_times[1] = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f;

    // Pool1
    start = clock();
    boat_variable_t* x_pool1 = boat_var_pool(x_relu1, model->pool1);
    end = clock();
    layer_times[2] = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f;

    // Conv2
    start = clock();
    boat_variable_t* x_conv2 = boat_var_conv(x_pool1, model->conv2);
    end = clock();
    layer_times[3] = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f;

    // ReLU2
    start = clock();
    boat_variable_t* x_relu2 = boat_var_relu(x_conv2);
    end = clock();
    layer_times[4] = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f;

    // Pool2
    start = clock();
    boat_variable_t* x_pool2 = boat_var_pool(x_relu2, model->pool2);
    end = clock();
    layer_times[5] = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f;

    // Flatten
    start = clock();
    boat_variable_t* x_flat = boat_var_flatten(x_pool2);
    end = clock();
    layer_times[6] = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f;

    // FC1
    start = clock();
    boat_variable_t* x_fc1 = boat_var_dense(x_flat, model->fc1);
    end = clock();
    layer_times[7] = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f;

    // ReLU3
    start = clock();
    boat_variable_t* x_relu3 = boat_var_relu(x_fc1);
    end = clock();
    layer_times[8] = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f;

    // FC2 (output)
    start = clock();
    boat_variable_t* x_fc2 = boat_var_dense(x_relu3, model->fc2);
    end = clock();
    layer_times[9] = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f;

    // Clean up intermediate variables (except final output)
    boat_variable_free(x_conv1);
    boat_variable_free(x_relu1);
    boat_variable_free(x_pool1);
    boat_variable_free(x_conv2);
    boat_variable_free(x_relu2);
    boat_variable_free(x_pool2);
    boat_variable_free(x_flat);
    boat_variable_free(x_fc1);
    boat_variable_free(x_relu3);
    boat_variable_free(x_fc2);
    boat_variable_free(x);

    // Print results
    float total_time = 0.0f;
    for (int i = 0; i < 10; i++) {
        total_time += layer_times[i];
    }

    for (int i = 0; i < 10; i++) {
        float percentage = total_time > 0 ? (layer_times[i] / total_time) * 100.0f : 0.0f;
        printf("               %-8s: %6.2f ms (%5.1f%%)\n", layer_names[i], layer_times[i], percentage);
    }
    printf("               total forward pass: %.2f ms\n", total_time);
}

// Compute loss statistics from recent history
static stability_stats_t compute_stability_stats(const float* losses, const float* val_losses,
                                                 int current_epoch, int window_size) {
    stability_stats_t stats = {0};

    if (current_epoch < 1 || window_size < 1) return stats;

    int start_epoch = current_epoch - window_size;
    if (start_epoch < 0) start_epoch = 0;
    int count = current_epoch - start_epoch + 1;

    // Compute mean and standard deviation
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = start_epoch; i <= current_epoch; i++) {
        float loss = losses[i];
        sum += loss;
        sum_sq += loss * loss;
    }
    stats.loss_mean = sum / count;
    float variance = (sum_sq / count) - (stats.loss_mean * stats.loss_mean);
    stats.loss_stddev = variance > 0 ? sqrtf(variance) : 0.0f;

    // Compute smoothness (average absolute difference between consecutive losses)
    float total_diff = 0.0f;
    int diff_count = 0;
    for (int i = start_epoch + 1; i <= current_epoch; i++) {
        total_diff += fabsf(losses[i] - losses[i-1]);
        diff_count++;
    }
    stats.loss_smoothness = diff_count > 0 ? total_diff / diff_count : 0.0f;

    // Compute validation improvement
    if (val_losses) {
        float current_val = val_losses[current_epoch];
        float prev_val = val_losses[current_epoch - 1];
        stats.val_improvement = prev_val > 0 ? (prev_val - current_val) / prev_val : 0.0f;
    }

    return stats;
}

// Early stopping monitoring
typedef struct {
    int patience;           // Number of epochs to wait for improvement
    float min_delta;        // Minimum improvement to reset patience
    float best_val_loss;    // Best validation loss so far
    int epochs_no_improve;  // Epochs without improvement
    bool stop_flag;         // Whether to stop training
} early_stopping_t;

// Initialize early stopping monitor
static early_stopping_t early_stopping_init(int patience, float min_delta) {
    early_stopping_t monitor = {0};
    monitor.patience = patience;
    monitor.min_delta = min_delta;
    monitor.best_val_loss = INFINITY;
    monitor.epochs_no_improve = 0;
    monitor.stop_flag = false;
    return monitor;
}

// Update early stopping monitor with new validation loss
static void early_stopping_update(early_stopping_t* monitor, float val_loss, int epoch) {
    if (!monitor) return;

    if (val_loss < monitor->best_val_loss - monitor->min_delta) {
        // Improvement detected
        monitor->best_val_loss = val_loss;
        monitor->epochs_no_improve = 0;
        if (epoch >= 0) {
            printf("             early stopping: best validation loss improved to %.4f (epoch %d)\n",
                   val_loss, epoch + 1);
        }
    } else {
        // No improvement
        monitor->epochs_no_improve++;
        if (monitor->epochs_no_improve >= monitor->patience) {
            monitor->stop_flag = true;
            printf("             early stopping: patience exhausted (%d epochs without improvement)\n",
                   monitor->patience);
        }
    }
}

// Resource usage prediction
typedef struct {
    double avg_epoch_time;      // Average epoch time in seconds
    double time_remaining;      // Estimated time remaining in seconds
    double projected_mem_peak;  // Projected peak memory usage (MB)
} resource_prediction_t;

// Predict resource usage based on training history
static resource_prediction_t predict_resources(const double* epoch_times, int current_epoch,
                                               int total_epochs, const boat_memory_stats_t* mem_stats,
                                               int window_size) {
    resource_prediction_t pred = {0};

    if (current_epoch < 1 || total_epochs <= current_epoch) return pred;

    // Compute average epoch time from recent epochs
    int start_epoch = current_epoch - window_size;
    if (start_epoch < 0) start_epoch = 0;
    int count = current_epoch - start_epoch + 1;

    double total_time = 0.0;
    for (int i = start_epoch; i <= current_epoch; i++) {
        total_time += epoch_times[i];
    }
    pred.avg_epoch_time = total_time / count;

    // Predict remaining time
    int epochs_remaining = total_epochs - (current_epoch + 1);
    pred.time_remaining = pred.avg_epoch_time * epochs_remaining;

    // Predict memory usage (simple linear projection)
    if (mem_stats) {
        // Use current allocated memory as baseline
        double peak_mem = mem_stats->peak_allocated_bytes / (1024.0 * 1024.0);
        pred.projected_mem_peak = peak_mem * 1.1; // Conservative estimate
    }

    return pred;
}

// Evaluate model on a dataset, returning loss and accuracy
void evaluate_model(mnist_model_t* model, boat_tensor_t* images, boat_tensor_t* labels,
                    float* out_loss, float* out_accuracy) {
    if (!model || !images || !labels || !out_loss || !out_accuracy) return;

    const int64_t* shape = boat_tensor_shape(images);
    size_t num_samples = shape[0];
    size_t batch_size = 32;  // Evaluation batch size (reduced for memory)
    size_t num_batches = (num_samples + batch_size - 1) / batch_size;

    float total_loss = 0.0f;
    int total_correct = 0;

    for (size_t batch = 0; batch < num_batches; batch++) {
        size_t start_idx = batch * batch_size;
        size_t end_idx = start_idx + batch_size;
        if (end_idx > num_samples) end_idx = num_samples;
        size_t actual_batch_size = end_idx - start_idx;

        // Slice batch
        boat_tensor_t* batch_images = boat_tensor_slice(images,
            (size_t[]){start_idx, 0, 0, 0},
            (size_t[]){end_idx, 1, 28, 28}, NULL);
        boat_tensor_t* batch_labels = boat_tensor_slice(labels,
            (size_t[]){start_idx},
            (size_t[]){end_idx}, NULL);

        if (!batch_images || !batch_labels) continue;

        // Forward pass
        boat_variable_t* input_var = tensor_to_variable(batch_images, false);
        if (!input_var) {
            fprintf(stderr, "ERROR evaluate_model: tensor_to_variable failed for batch_images=%p\n", (void*)batch_images);
            boat_tensor_unref(batch_images);
            boat_tensor_unref(batch_labels);
            continue;
        }
        boat_variable_t* logits = forward_pass(model, input_var);
        if (!logits) {
            fprintf(stderr, "ERROR evaluate_model: forward_pass failed for input_var=%p\n", (void*)input_var);
            boat_variable_free(input_var);
            boat_tensor_unref(batch_images);
            boat_tensor_unref(batch_labels);
            continue;
        }

        // Compute loss for this batch
        boat_variable_t* loss_var = cross_entropy_loss(logits, batch_labels);
        boat_tensor_t* loss_tensor = boat_variable_data(loss_var);
        float batch_loss = *(float*)boat_tensor_data(loss_tensor);
        total_loss += batch_loss * actual_batch_size;  // loss is average per sample

        // Compute accuracy
        float batch_acc = compute_accuracy(logits, batch_labels);
        total_correct += (int)(batch_acc * actual_batch_size);

        // Cleanup
        boat_variable_free(input_var);
        boat_variable_free(logits);
        boat_variable_free(loss_var);
        boat_tensor_unref(batch_images);
        boat_tensor_unref(batch_labels);
    }

    *out_loss = total_loss / num_samples;
    *out_accuracy = (float)total_correct / num_samples;
}

// Training function for one batch
// Returns loss value
float train_batch(mnist_model_t* model, boat_tensor_t* batch_images, boat_tensor_t* batch_labels,
                  float learning_rate, bool zero_grad) {
    DEBUG_PRINT("train_batch: entered, batch_images=%p, batch_labels=%p\n",
            (void*)batch_images, (void*)batch_labels);
    (void)learning_rate; // Parameter not used in this implementation
    // Convert batch data to variables using reusable variables
    boat_variable_t* input_var = get_reusable_input_variable(model, batch_images);
    boat_variable_t* target_var = get_reusable_target_variable(model, batch_labels);

    if (!input_var || !target_var) {
        fprintf(stderr, "Error: Failed to get reusable variables\n");
        return 0.0f;
    }
    DEBUG_PRINT("train_batch: got variables input_var=%p, target_var=%p\n",
            (void*)input_var, (void*)target_var);

    // Forward pass
    DEBUG_PRINT("train_batch: calling forward_pass\n");
    boat_variable_t* logits = forward_pass(model, input_var);
    DEBUG_PRINT("train_batch: forward_pass returned logits=%p\n", (void*)logits);

    // Compute loss
    DEBUG_PRINT("train_batch: calling cross_entropy_loss\n");
    boat_variable_t* loss_var = cross_entropy_loss(logits, batch_labels);
    DEBUG_PRINT("train_batch: cross_entropy_loss returned loss_var=%p\n", (void*)loss_var);

    // Backward pass
    DEBUG_PRINT("train_batch: calling boat_variable_backward_full\n");
    boat_variable_backward_full(loss_var);
    DEBUG_PRINT("train_batch: boat_variable_backward_full returned\n");

    // Optimizer step
    boat_optimizer_step(model->optimizer);

    // Clear computation graph to free memory after gradients are applied
    boat_autodiff_clear_computation_graph();

    // Zero gradients for next iteration (if requested)
    if (zero_grad) {
        boat_optimizer_zero_grad(model->optimizer);
    }

    // Get loss value
    boat_tensor_t* loss_tensor = boat_variable_data(loss_var);
    float loss_value = *(float*)boat_tensor_data(loss_tensor);

    // Cleanup - don't free input_var and target_var as they are reusable
    boat_variable_free(logits);
    boat_variable_free(loss_var);

    return loss_value;
}

int main(int argc, char* argv[]) {
    printf("=== MNIST Digit Recognition with Boat Autodiff ===\n");
    DEBUG_PRINT("main() started\n");

    // Parse command line arguments
    const char* checkpoint_file = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            checkpoint_file = argv[i + 1];
            i++; // Skip next argument
            printf("Will load model from checkpoint: %s\n", checkpoint_file);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--load <checkpoint_file>]\n", argv[0]);
            printf("       --load <file>   Load model weights from checkpoint file\n");
            return 0;
        }
    }

    // Use binary format files from data directory
    // Support environment variable USE_FULL_DATA to switch to full dataset
    const char* use_full = getenv("USE_FULL_DATA");
    const char* train_images_file = use_full ? "data/train_images.bin" : "data/train_images_small.bin";
    const char* train_labels_file = use_full ? "data/train_labels.bin" : "data/train_labels_small.bin";
    const char* test_images_file = use_full ? "data/test_images.bin" : "data/test_images_small.bin";
    const char* test_labels_file = use_full ? "data/test_labels.bin" : "data/test_labels_small.bin";

    printf("Loading training data from %s...\n", train_images_file);
    // Load binary files
    boat_tensor_t* train_images_raw = load_tensor_binary(train_images_file, BOAT_DTYPE_FLOAT32);
    boat_tensor_t* train_labels = load_tensor_binary(train_labels_file, BOAT_DTYPE_UINT8);
    boat_tensor_t* test_images_raw = load_tensor_binary(test_images_file, BOAT_DTYPE_FLOAT32);
    boat_tensor_t* test_labels = load_tensor_binary(test_labels_file, BOAT_DTYPE_UINT8);

    // Convert images to normalized float32 with channel dimension
    boat_tensor_t* train_images = idx_images_to_float32(train_images_raw);
    boat_tensor_t* test_images = idx_images_to_float32(test_images_raw);

    // Free the intermediate raw tensors
    if (train_images_raw) boat_tensor_unref(train_images_raw);
    if (test_images_raw) boat_tensor_unref(test_images_raw);

    DEBUG_PRINT("Data loaded: train_images=%p, train_labels=%p, test_images=%p, test_labels=%p\n",
            train_images, train_labels, test_images, test_labels);
    if (!train_images || !train_labels || !test_images || !test_labels) {
        fprintf(stderr, "Error loading data files\n");
        return 1;
    }

    DEBUG_PRINT("Before boat_tensor_shape\n");
    const int64_t* train_shape = boat_tensor_shape(train_images);
    DEBUG_PRINT("After boat_tensor_shape, shape=%p\n", train_shape);
    DEBUG_PRINT("train_shape[0] = %lld\n", (long long)train_shape[0]);
    size_t train_samples = train_shape[0];
    fprintf(stderr, "Training samples: %zu\n", train_samples);

    // Split training data into training and validation sets
    // For small datasets, use 20% for validation, but at least 1 sample
    size_t val_size = train_samples / 5;  // 20% for validation
    if (val_size < 1) val_size = 1;
    if (val_size > train_samples) val_size = train_samples;
    size_t train_size = train_samples - val_size;

    fprintf(stderr, "Splitting data: %zu training, %zu validation\n", train_size, val_size);

    // Create validation set (first val_size samples)
    boat_tensor_t* val_images = boat_tensor_slice(train_images,
        (size_t[]){0, 0, 0, 0},
        (size_t[]){val_size, 1, 28, 28}, NULL);
    boat_tensor_t* val_labels = boat_tensor_slice(train_labels,
        (size_t[]){0},
        (size_t[]){val_size}, NULL);

    // Create training subset (remaining samples)
    boat_tensor_t* train_images_subset = boat_tensor_slice(train_images,
        (size_t[]){val_size, 0, 0, 0},
        (size_t[]){train_samples, 1, 28, 28}, NULL);
    boat_tensor_t* train_labels_subset = boat_tensor_slice(train_labels,
        (size_t[]){val_size},
        (size_t[]){train_samples}, NULL);

    if (!val_images || !val_labels || !train_images_subset || !train_labels_subset) {
        fprintf(stderr, "Error: Failed to split data into training and validation sets\n");
        return 1;
    }

    // Replace original training tensors with subset
    boat_tensor_unref(train_images);
    boat_tensor_unref(train_labels);
    train_images = train_images_subset;
    train_labels = train_labels_subset;
    train_samples = train_size;  // Update train samples count

    DEBUG_PRINT("Data split complete\n");

    // Data standardization: subtract mean, divide by std
    // Compute mean and std from training set
    fprintf(stderr, "Computing mean and std from training set...\n");
    float* train_data = (float*)boat_tensor_data(train_images);
    size_t train_total_pixels = boat_tensor_nelements(train_images);

    double sum = 0.0;
    for (size_t i = 0; i < train_total_pixels; i++) {
        sum += train_data[i];
    }
    float mean = (float)(sum / train_total_pixels);

    double sum_sq = 0.0;
    for (size_t i = 0; i < train_total_pixels; i++) {
        float diff = train_data[i] - mean;
        sum_sq += diff * diff;
    }
    float std = (float)sqrt(sum_sq / train_total_pixels);
    if (std < 1e-8) std = 1.0f;  // Avoid division by zero

    fprintf(stderr, "Training set stats: mean=%.6f, std=%.6f\n", mean, std);

    // Standardize training set
    for (size_t i = 0; i < train_total_pixels; i++) {
        train_data[i] = (train_data[i] - mean) / std;
    }

    // Standardize validation set
    float* val_data = (float*)boat_tensor_data(val_images);
    size_t val_total_pixels = boat_tensor_nelements(val_images);
    for (size_t i = 0; i < val_total_pixels; i++) {
        val_data[i] = (val_data[i] - mean) / std;
    }

    // Standardize test set
    float* test_data = (float*)boat_tensor_data(test_images);
    size_t test_total_pixels = boat_tensor_nelements(test_images);
    for (size_t i = 0; i < test_total_pixels; i++) {
        test_data[i] = (test_data[i] - mean) / std;
    }

    fprintf(stderr, "Data standardization complete\n");

    // Create contiguous copies of training data for efficient slicing
    fprintf(stderr, "Creating contiguous copies of training data...\n");
    boat_tensor_t* train_images_contig = boat_tensor_create_like(train_images);
    boat_tensor_t* train_labels_contig = boat_tensor_create_like(train_labels);

    if (!train_images_contig || !train_labels_contig) {
        fprintf(stderr, "Error: Failed to create contiguous training data copies\n");
        return 1;
    }

    // Copy data
    size_t train_images_bytes = boat_tensor_nbytes(train_images);
    size_t train_labels_bytes = boat_tensor_nbytes(train_labels);
    memcpy(boat_tensor_data(train_images_contig), boat_tensor_const_data(train_images), train_images_bytes);
    memcpy(boat_tensor_data(train_labels_contig), boat_tensor_const_data(train_labels), train_labels_bytes);

    // Replace view tensors with contiguous copies
    boat_tensor_unref(train_images);
    boat_tensor_unref(train_labels);
    train_images = train_images_contig;
    train_labels = train_labels_contig;

    // Also create contiguous copies of validation data
    boat_tensor_t* val_images_contig = boat_tensor_create_like(val_images);
    boat_tensor_t* val_labels_contig = boat_tensor_create_like(val_labels);

    if (!val_images_contig || !val_labels_contig) {
        fprintf(stderr, "Error: Failed to create contiguous validation data copies\n");
        return 1;
    }

    size_t val_images_bytes = boat_tensor_nbytes(val_images);
    size_t val_labels_bytes = boat_tensor_nbytes(val_labels);
    memcpy(boat_tensor_data(val_images_contig), boat_tensor_const_data(val_images), val_images_bytes);
    memcpy(boat_tensor_data(val_labels_contig), boat_tensor_const_data(val_labels), val_labels_bytes);

    boat_tensor_unref(val_images);
    boat_tensor_unref(val_labels);
    val_images = val_images_contig;
    val_labels = val_labels_contig;

    fprintf(stderr, "Contiguous training and validation data created successfully\n");
    DEBUG_PRINT("Before create_mnist_model\n");

    // Create model
    mnist_model_t* model = create_mnist_model(0.001f);
    DEBUG_PRINT("main: model created at %p\n", model);
    if (!model) {
        fprintf(stderr, "Failed to create model\n");
        return 1;
    }

    // Load checkpoint if specified
    if (checkpoint_file) {
        printf("Loading model weights from %s...\n", checkpoint_file);
        if (!load_mnist_model(model, checkpoint_file)) {
            fprintf(stderr, "Warning: Failed to load checkpoint, starting from scratch\n");
        } else {
            printf("Checkpoint loaded successfully\n");
        }
    }

    // Training parameters
    int epochs = 1;
    size_t batch_size = 32;
    size_t num_batches = (train_samples + batch_size - 1) / batch_size;

    DEBUG_PRINT("\nStarting training...\n");
    printf("Epochs: %d, Batch size: %zu, Learning rate: 0.001\n", epochs, batch_size);

    // Arrays to track training history
    float* train_losses = malloc(epochs * sizeof(float));
    float* train_accuracies = malloc(epochs * sizeof(float));
    float* val_losses = malloc(epochs * sizeof(float));
    float* val_accuracies = malloc(epochs * sizeof(float));
    float* learning_rates = malloc(epochs * sizeof(float));
    float* gradient_norms = malloc(epochs * sizeof(float));
    float* gradient_maxes = malloc(epochs * sizeof(float));
    float* gradient_clip_ratios = malloc(epochs * sizeof(float));
    double* epoch_times = malloc(epochs * sizeof(double));
    if (!train_losses || !train_accuracies || !val_losses || !val_accuracies || !learning_rates ||
        !gradient_norms || !gradient_maxes || !gradient_clip_ratios || !epoch_times) {
        fprintf(stderr, "Error: Failed to allocate memory for training history\n");
        return 1;
    }

    // Open training history file for real-time monitoring
    FILE* history_file = fopen("training_history.csv", "w");
    if (history_file) {
        fprintf(history_file, "epoch,timestamp,train_loss,train_accuracy,val_loss,val_accuracy,learning_rate,grad_norm,grad_max,grad_clip_ratio,epoch_time,loss_stddev,val_improvement\n");
    } else {
        fprintf(stderr, "Warning: Could not open training_history.csv for writing\n");
    }

    // Initialize early stopping monitor
    early_stopping_t early_stop = early_stopping_init(5, 0.001f); // 5 epochs patience, 0.001 min delta

    // Auto-tuning parameters
    float current_clip_threshold = 5.0f; // Initial gradient clipping threshold
    // Decision quality evaluation
    float loss_before_decision = -1.0f; // Loss before last tuning decision
    float decision_effect_score = 0.0f; // Cumulative score of decision effectiveness

    // Training loop with checkpointing
    // Reset memory statistics before training
    boat_memory_reset_stats();
    float best_val_accuracy = 0.0f;
    int best_epoch = -1;
    // Note: warm-up is disabled (set warmup_epochs > 0 if needed in the future)
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Apply learning rate warm-up (disabled when warmup_epochs = 0)
        float current_lr = boat_optimizer_get_learning_rate(model->optimizer);
        // Note: warmup_epochs is 0, so warm-up is disabled
        // If warm-up is needed in the future, set warmup_epochs > 0
        learning_rates[epoch] = current_lr;

        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        int epoch_total = 0;

        clock_t start_time = clock();

        // Simple batch iteration (non-randomized for simplicity)
        for (size_t batch = 0; batch < num_batches; batch++) {
            bool is_last_batch = (batch == num_batches - 1);
            size_t start_idx = batch * batch_size;
            size_t end_idx = start_idx + batch_size;

            // In a real implementation, we would slice tensors
            // For simplicity, we'll process one sample per batch
            if (start_idx >= train_samples) break;

            // Create batch slice
            // Calculate actual batch size for this iteration (last batch might be smaller)
            size_t actual_batch_size = batch_size;
            if (end_idx > train_samples) {
                actual_batch_size = train_samples - start_idx;
                end_idx = train_samples;
            }

            boat_tensor_t* batch_images = boat_tensor_slice(train_images,
                (size_t[]){start_idx, 0, 0, 0},
                (size_t[]){end_idx, 1, 28, 28}, NULL);
            boat_tensor_t* batch_labels = boat_tensor_slice(train_labels,
                (size_t[]){start_idx},
                (size_t[]){end_idx}, NULL);

            if (!batch_images || !batch_labels) {
                fprintf(stderr, "Failed to slice batch\n");
                continue;
            }

            // Train on batch
            DEBUG_PRINT("training loop: calling train_batch for batch %zu\n", batch);
            fflush(stderr);
            float loss = train_batch(model, batch_images, batch_labels, 0.001f, !is_last_batch);
            DEBUG_PRINT("training loop: train_batch returned loss=%.4f\n", loss);
            epoch_loss += loss * actual_batch_size; // loss is average per sample

            // Compute accuracy for this batch
            boat_variable_t* input_var = tensor_to_variable(batch_images, false);
            boat_variable_t* logits = forward_pass(model, input_var);
            float acc = compute_accuracy(logits, batch_labels);
            epoch_correct += (int)(acc * actual_batch_size); // acc is percentage correct
            epoch_total += actual_batch_size;

            // Cleanup
            boat_variable_free(input_var);
            boat_variable_free(logits);
            boat_tensor_unref(batch_images);
            boat_tensor_unref(batch_labels);

            if (batch % 100 == 0 && batch > 0) {
                printf("  Batch %zu/%zu, Loss: %.4f\n", batch, num_batches, loss);
            }
        }

        clock_t end_time = clock();
        double epoch_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        // Compute gradient statistics after the last batch (gradients still exist)
        gradient_stats_t grad_stats = compute_gradient_stats(model, current_clip_threshold); // clip threshold = 5.0
        gradient_norms[epoch] = grad_stats.total_norm;
        gradient_maxes[epoch] = grad_stats.max_grad;
        gradient_clip_ratios[epoch] = grad_stats.grad_clip_ratio;
        epoch_times[epoch] = epoch_time;

        // Apply gradient clipping if needed (legacy rule, may be overridden by auto-tuning)
        if (grad_stats.grad_clip_ratio > 0.1f) { // If more than 10% gradients would be clipped
            apply_gradient_clipping(model, current_clip_threshold);
            printf("             gradient clipping applied (%.1f%% > 10%% threshold, threshold=%.2f)\n",
                   grad_stats.grad_clip_ratio * 100.0f, current_clip_threshold);
        }

        // Zero gradients manually (since last batch didn't zero them)
        boat_optimizer_zero_grad(model->optimizer);

        float epoch_accuracy = epoch_total > 0 ? (float)epoch_correct / epoch_total : 0.0f;
        float avg_loss = epoch_total > 0 ? epoch_loss / epoch_total : 0.0f;

        double samples_per_second = epoch_time > 1e-6 ? (double)epoch_total / epoch_time : 0.0;
        double avg_batch_time = epoch_time > 1e-6 ? epoch_time / num_batches : 0.0;
        double batches_per_second = epoch_time > 1e-6 ? num_batches / epoch_time : 0.0;
        printf("Epoch %d/%d: time=%.2fs, loss=%.4f, accuracy=%.2f%%, throughput=%.0f samples/s (%.2f batches/s, %.3fs/batch), lr=%.6f, grad_norm=%.4f\n",
               epoch + 1, epochs, epoch_time, avg_loss, epoch_accuracy * 100.0f, samples_per_second, batches_per_second, avg_batch_time, current_lr, grad_stats.total_norm);

        // Store training metrics
        train_losses[epoch] = avg_loss;
        train_accuracies[epoch] = epoch_accuracy;

        // Evaluate on validation set
        float val_loss, val_accuracy;
        evaluate_model(model, val_images, val_labels, &val_loss, &val_accuracy);
        val_losses[epoch] = val_loss;
        val_accuracies[epoch] = val_accuracy;

        printf("             validation loss=%.4f, accuracy=%.2f%%\n",
               val_loss, val_accuracy * 100.0f);

        // Update early stopping monitor
        early_stopping_update(&early_stop, val_loss, epoch);

        // Memory statistics
        boat_memory_stats_t mem_stats = boat_memory_get_stats();
        printf("             memory: allocated=%.2f MB, blocks=%zu, peak=%.2f MB\n",
               mem_stats.allocated_bytes / (1024.0 * 1024.0),
               mem_stats.allocated_blocks,
               mem_stats.peak_allocated_bytes / (1024.0 * 1024.0));

        // Gradient statistics
        printf("             gradients: norm=%.4f, max=%.4f, clip_ratio=%.2f%%, nan_inf=%d\n",
               grad_stats.total_norm, grad_stats.max_grad, grad_stats.grad_clip_ratio * 100.0f, grad_stats.nan_inf_count);

        // Save checkpoint if this is the best model so far
        if (val_accuracy > best_val_accuracy) {
            best_val_accuracy = val_accuracy;
            best_epoch = epoch;
            char checkpoint_filename[256];
            snprintf(checkpoint_filename, sizeof(checkpoint_filename), "mnist_checkpoint_epoch%03d.boat", epoch + 1);
            if (save_mnist_model(model, checkpoint_filename)) {
                printf("             saved checkpoint to %s (accuracy improved)\n", checkpoint_filename);
            }

            // Also save as best model
            if (save_mnist_model(model, "mnist_best.boat")) {
                printf("             saved as best model mnist_best.boat\n");
            }
        }

        // Save regular checkpoint every 2 epochs
        if (epoch % 2 == 1) {
            char checkpoint_filename[256];
            snprintf(checkpoint_filename, sizeof(checkpoint_filename), "mnist_checkpoint_latest.boat");
            if (save_mnist_model(model, checkpoint_filename)) {
                printf("             saved latest checkpoint to %s\n", checkpoint_filename);
            }
        }

        // Update learning rate scheduler
        if (model->scheduler) {
            boat_scheduler_step(model->scheduler);
            boat_scheduler_update_optimizer(model->scheduler, model->optimizer);
        }

        // Compute training stability statistics
        stability_stats_t stability = compute_stability_stats(train_losses, val_losses, epoch, 5); // window size = 5

        // Provide hyperparameter tuning recommendations
        provide_tuning_recommendations(&grad_stats, &stability, current_lr, epoch_time, batch_size, epoch, epochs);

        // Apply auto-tuning strategy (iteration 17)
        // Compute multi-objective metrics for iteration 20
        float throughput_sps = epoch_time > 1e-6 ? (float)(batch_size * num_batches) / (float)epoch_time : 0.0f;
        boat_memory_stats_t current_mem_stats = boat_memory_get_stats();
        float memory_usage_mb = current_mem_stats.allocated_bytes / (1024.0f * 1024.0f);
        float validation_accuracy = val_accuracies[epoch];

        auto_tuning_decision_t tuning_decision = auto_tuning_strategy(
            &grad_stats, &stability, current_lr, epoch_time, (int)batch_size, epoch, epochs, current_clip_threshold,
            throughput_sps, memory_usage_mb, validation_accuracy);

        // Evaluate previous decision effect (if any)
        if (loss_before_decision >= 0.0f) {
            float loss_change = avg_loss - loss_before_decision; // Negative means improvement
            decision_effect_score += (loss_change < 0 ? 1.0f : -1.0f); // Simple scoring
            printf("             decision quality: loss change %.4f -> %.4f (delta: %.4f), cumulative score: %.1f\n",
                   loss_before_decision, avg_loss, loss_change, decision_effect_score);
        }
        // Store current loss for next decision evaluation
        loss_before_decision = avg_loss;

        // Apply the tuning decisions
        apply_auto_tuning_decisions(model, &tuning_decision, &current_lr, &current_clip_threshold, &batch_size, &num_batches, train_samples);

        // Update learning rate in the array (in case it was changed by auto-tuning)
        learning_rates[epoch] = current_lr;

        // Write training history for real-time monitoring with enhanced metrics
        if (history_file) {
            fprintf(history_file, "%d,%lld,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    epoch + 1,
                    (long long)time(NULL),
                    train_losses[epoch],
                    train_accuracies[epoch],
                    val_losses[epoch],
                    val_accuracies[epoch],
                    learning_rates[epoch],
                    gradient_norms[epoch],
                    gradient_maxes[epoch],
                    gradient_clip_ratios[epoch],
                    epoch_times[epoch],
                    stability.loss_stddev,
                    stability.val_improvement);
            fflush(history_file);  // Ensure data is written immediately
        }

        // Resource usage prediction (display every 5 epochs)
        if (epoch % 5 == 0 && epoch > 0) {
            boat_memory_stats_t local_mem_stats = boat_memory_get_stats();
            resource_prediction_t resource_pred = predict_resources(epoch_times, epoch, epochs, &local_mem_stats, 5);
            printf("             resource prediction: %.1f s/epoch, %.1f s remaining, projected peak memory: %.2f MB\n",
                   resource_pred.avg_epoch_time, resource_pred.time_remaining, resource_pred.projected_mem_peak);
        }

        // Layer performance analysis (display every 10 epochs)
        if (epoch % 10 == 0 && epoch > 0) {
            // Use a small batch from training data for profiling
            if (train_samples > 0) {
                // Take first sample for profiling
                boat_tensor_t* sample = boat_tensor_slice(train_images,
                    (size_t[]){0, 0, 0, 0},
                    (size_t[]){1, 1, 28, 28}, NULL);
                if (sample) {
                    profile_layer_performance(model, sample);
                    boat_tensor_unref(sample);
                }
            }
        }

        // Check early stopping
        if (early_stop.stop_flag) {
            printf("\nEarly stopping triggered at epoch %d\n", epoch + 1);
            break;
        }
    }

    // Close training history file
    if (history_file) {
        fclose(history_file);
        printf("\nTraining history saved to training_history.csv\n");
    }

    // Print final memory statistics
    boat_memory_stats_t final_stats = boat_memory_get_stats();
    printf("\nFinal memory statistics:\n");
    printf("  Allocated: %.2f MB in %zu blocks\n", final_stats.allocated_bytes / (1024.0 * 1024.0), final_stats.allocated_blocks);
    printf("  Freed: %.2f MB in %zu blocks\n", final_stats.freed_bytes / (1024.0 * 1024.0), final_stats.freed_blocks);
    printf("  Peak allocated: %.2f MB\n", final_stats.peak_allocated_bytes / (1024.0 * 1024.0));

    // Save final model
    if (save_mnist_model(model, "mnist_final.boat")) {
        printf("Final model saved to mnist_final.boat\n");
    }

    // Print best model info
    if (best_epoch >= 0) {
        printf("\nBest model: epoch %d, validation accuracy: %.2f%%\n",
               best_epoch + 1, best_val_accuracy * 100.0f);
    }

    // Evaluation on test set
    printf("\nEvaluating on test set...\n");
    const int64_t* test_shape = boat_tensor_shape(test_images);
    size_t test_samples = test_shape[0];

    int test_correct = 0;
    size_t test_batch_size = 32; // Reduced batch size for evaluation (memory)
    size_t test_num_batches = (test_samples + test_batch_size - 1) / test_batch_size;

    for (size_t batch = 0; batch < test_num_batches; batch++) {
        size_t start_idx = batch * test_batch_size;
        size_t end_idx = start_idx + test_batch_size;
        if (end_idx > test_samples) end_idx = test_samples;
        size_t actual_batch_size = end_idx - start_idx;

        boat_tensor_t* batch_images = boat_tensor_slice(test_images,
            (size_t[]){start_idx, 0, 0, 0},
            (size_t[]){end_idx, 1, 28, 28}, NULL);
        boat_tensor_t* batch_labels = boat_tensor_slice(test_labels,
            (size_t[]){start_idx},
            (size_t[]){end_idx}, NULL);

        if (!batch_images || !batch_labels) continue;

        boat_variable_t* input_var = tensor_to_variable(batch_images, false);
        boat_variable_t* logits = forward_pass(model, input_var);
        float acc = compute_accuracy(logits, batch_labels);
        test_correct += (int)(acc * actual_batch_size);

        boat_variable_free(input_var);
        boat_variable_free(logits);
        boat_tensor_unref(batch_images);
        boat_tensor_unref(batch_labels);
    }

    float test_accuracy = (float)test_correct / test_samples;
    printf("Test accuracy: %.2f%% (%d/%zu)\n", test_accuracy * 100.0f, test_correct, test_samples);

    // Cleanup
    // Free training history arrays
    free(train_losses);
    free(train_accuracies);
    free(val_losses);
    free(val_accuracies);
    free(learning_rates);
    free(gradient_norms);
    free(gradient_maxes);
    free(gradient_clip_ratios);
    free(epoch_times);

    // Free validation tensors
    if (val_images) boat_tensor_unref(val_images);
    if (val_labels) boat_tensor_unref(val_labels);

    free_mnist_model(model);
    boat_tensor_unref(train_images);
    boat_tensor_unref(train_labels);
    boat_tensor_unref(test_images);
    boat_tensor_unref(test_labels);

    printf("\nDone!\n");
    return 0;
}