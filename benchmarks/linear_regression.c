// linear_regression.c - Linear regression benchmark for optimizer comparison
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/optimizers.h>
#include <boat/schedulers.h>
#include <boat/tensor.h>
#include <boat/ops.h>
#include <boat/autodiff.h>
#include <boat/loss.h>
#include <boat/memory.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

// Structure to hold benchmark results
typedef struct {
    const char* optimizer_name;
    const char* scheduler_name;
    int steps_to_converge;
    float final_loss;
    double training_time_ms;
} benchmark_result_t;

// Generate synthetic linear regression data: y = Wx + b + noise
static void generate_linear_data(
    boat_tensor_t** x_out, boat_tensor_t** y_out,
    int num_samples, int input_dim, int output_dim,
    float noise_std
) {
    // Create input tensor: [num_samples, input_dim]
    int64_t x_shape[] = {num_samples, input_dim};
    *x_out = boat_tensor_create(x_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    // Create output tensor: [num_samples, output_dim]
    int64_t y_shape[] = {num_samples, output_dim};
    *y_out = boat_tensor_create(y_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    if (!*x_out || !*y_out) {
        printf("Failed to allocate data tensors\n");
        return;
    }

    float* x_data = (float*)boat_tensor_data(*x_out);
    float* y_data = (float*)boat_tensor_data(*y_out);

    // Generate random weights and bias (true parameters)
    float* W_true = (float*)malloc(input_dim * output_dim * sizeof(float));
    float* b_true = (float*)malloc(output_dim * sizeof(float));

    // Check for allocation failure
    if (!W_true || !b_true) {
        printf("Failed to allocate true parameters\n");
        if (W_true) free(W_true);
        if (b_true) free(b_true);
        boat_tensor_unref(*x_out);
        boat_tensor_unref(*y_out);
        *x_out = NULL;
        *y_out = NULL;
        return;
    }

    srand(42); // Fixed seed for reproducibility

    for (int i = 0; i < input_dim * output_dim; i++) {
        W_true[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f; // Uniform [-1, 1]
    }

    for (int i = 0; i < output_dim; i++) {
        b_true[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    // Generate data
    for (int i = 0; i < num_samples; i++) {
        // Generate random input
        for (int j = 0; j < input_dim; j++) {
            x_data[i * input_dim + j] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        }

        // Compute output: y = Wx + b
        for (int k = 0; k < output_dim; k++) {
            float dot = 0.0f;
            for (int j = 0; j < input_dim; j++) {
                dot += W_true[k * input_dim + j] * x_data[i * input_dim + j];
            }
            float y_true = dot + b_true[k];

            // Add Gaussian noise
            float noise = ((rand() / (float)RAND_MAX) * 2.0f - 1.0f) * noise_std;
            y_data[i * output_dim + k] = y_true + noise;
        }
    }

    free(W_true);
    free(b_true);
}

// Simple linear model: y = xW + b
static boat_variable_t* linear_model(
    const boat_variable_t* x,
    boat_variable_t* W,
    boat_variable_t* b
) {
    // x shape: [batch, input_dim]
    // W shape: [input_dim, output_dim]
    // b shape: [output_dim]
    // Output: x * W + b

    boat_variable_t* xW = boat_var_matmul(x, W);
    boat_variable_t* y_pred = boat_var_add(xW, b);

    boat_variable_free(xW); // Free intermediate variable

    return y_pred;
}

// Run training with given optimizer and scheduler
static benchmark_result_t run_training(
    const char* optimizer_name,
    boat_optimizer_t* optimizer,
    boat_scheduler_t* scheduler,
    boat_variable_t* W,
    const boat_variable_t* b,
    boat_tensor_t* x_data,
    boat_tensor_t* y_data,
    int max_steps,
    float loss_threshold
) {
    benchmark_result_t result = {0};
    result.optimizer_name = optimizer_name;
    result.scheduler_name = scheduler ? "with scheduler" : "no scheduler";

    clock_t start_time = clock();

    // MSE loss using automatic differentiation: mean((y_pred - y_true)^2)
    // We'll compute this in the training loop using variable operations

    // Convert data to variables
    boat_variable_t* x_var = boat_variable_create(x_data, false); // No gradient needed for input
    boat_variable_t* y_true_var = boat_variable_create(y_data, false); // No gradient needed for target

    // Training loop
    for (int step = 0; step < max_steps; step++) {
        // Zero gradients before forward pass
        boat_variable_zero_grad(W);
        boat_variable_zero_grad(b);

        // Forward pass
        boat_variable_t* y_pred_var = linear_model(x_var, W, b);

        // Compute MSE loss and gradient using automatic differentiation
        // loss = mean((y_pred - y_true)^2)
        // gradient w.r.t y_pred = 2 * (y_pred - y_true) / n

        boat_tensor_t* y_pred_tensor = boat_variable_data(y_pred_var);
        boat_tensor_t* y_true_tensor = boat_variable_data(y_true_var);

        // Compute y_pred - y_true
        boat_tensor_t* diff = boat_sub(y_pred_tensor, y_true_tensor);
        if (!diff) {
            boat_variable_free(y_pred_var);
            continue;
        }

        // Compute (y_pred - y_true)^2
        boat_tensor_t* diff_squared = boat_mul(diff, diff);
        boat_tensor_unref(diff);
        if (!diff_squared) {
            boat_variable_free(y_pred_var);
            continue;
        }

        // Compute mean: sum(diff_squared) / n_elements
        size_t n_elements = boat_tensor_nelements(diff_squared);
        boat_tensor_t* loss_sum = boat_sum(diff_squared, NULL, 0, false); // Sum all elements
        float loss = 0.0f;
        if (loss_sum) {
            const float* loss_data = (const float*)boat_tensor_const_data(loss_sum);
            loss = loss_data[0] / n_elements;
            boat_tensor_unref(loss_sum);
        }
        boat_tensor_unref(diff_squared);

        // Check convergence
        if (loss < loss_threshold) {
            result.steps_to_converge = step + 1;
            result.final_loss = loss;
            boat_variable_free(y_pred_var);
            break;
        }

        // Compute gradient of loss w.r.t y_pred: 2 * (y_pred - y_true) / n_elements
        boat_tensor_t* diff_for_grad = boat_sub(y_pred_tensor, y_true_tensor);
        if (diff_for_grad) {
            // Scale by 2 / n_elements
            float scale = 2.0f / n_elements;
            boat_tensor_t* grad_y_pred = boat_mul_scalar(diff_for_grad, scale);
            boat_tensor_unref(diff_for_grad);

            if (grad_y_pred) {
                // Perform automatic differentiation backward pass
                boat_variable_backward(y_pred_var, grad_y_pred);
                boat_tensor_unref(grad_y_pred);
            }
        }

        // Update optimizer with computed gradients
        boat_optimizer_step(optimizer);

        // Update scheduler if provided
        if (scheduler) {
            boat_scheduler_step(scheduler);
            boat_scheduler_update_optimizer(scheduler, optimizer);
        }

        // Clean up
        boat_variable_free(y_pred_var);

        // Print progress every 100 steps
        if (step % 100 == 0) {
            printf("  Step %d, Loss: %.6f\n", step, loss);
        }

        // If we reach max steps, record final loss
        if (step == max_steps - 1) {
            result.steps_to_converge = max_steps;
            result.final_loss = loss;
        }
    }

    clock_t end_time = clock();
    result.training_time_ms = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000.0;

    // Cleanup
    boat_variable_free(x_var);
    boat_variable_free(y_true_var);

    return result;
}

// Main benchmark function
void run_optimizer_benchmark() {
    printf("Optimizer Performance Benchmark\n");
    printf("===============================\n\n");

    // Configuration
    const int num_samples = 1000;
    const int input_dim = 10;
    const int output_dim = 1;
    const float noise_std = 0.1f;
    const int max_steps = 1000;
    const float loss_threshold = 0.01f;
    const float learning_rate = 0.01f;

    // Generate synthetic data
    printf("Generating synthetic linear regression data...\n");
    boat_tensor_t* x_data = NULL;
    boat_tensor_t* y_data = NULL;
    generate_linear_data(&x_data, &y_data, num_samples, input_dim, output_dim, noise_std);

    if (!x_data || !y_data) {
        printf("Failed to generate data\n");
        return;
    }

    printf("Data shape: x=%lldx%lld, y=%lldx%lld\n\n",
           boat_tensor_shape(x_data)[0], boat_tensor_shape(x_data)[1],
           boat_tensor_shape(y_data)[0], boat_tensor_shape(y_data)[1]);

    // Initialize parameters (will be reset for each optimizer)
    int64_t W_shape[] = {input_dim, output_dim};
    int64_t b_shape[] = {output_dim};

    // Benchmark results array
    benchmark_result_t results[10];
    int result_count = 0;

    // Test 1: Adam optimizer (no scheduler)
    {
        printf("1. Adam optimizer (no scheduler)\n");

        // Create parameters
        boat_variable_t* W = boat_variable_create_with_shape(W_shape, 2, BOAT_DTYPE_FLOAT32, true);
        boat_variable_t* b = boat_variable_create_with_shape(b_shape, 1, BOAT_DTYPE_FLOAT32, true);

        // Initialize with small random values
        float* W_data = (float*)boat_tensor_data(boat_variable_data(W));
        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        srand(123);
        for (int i = 0; i < input_dim * output_dim; i++) {
            W_data[i] = (rand() / (float)RAND_MAX) * 0.1f;
        }
        for (int i = 0; i < output_dim; i++) {
            b_data[i] = (rand() / (float)RAND_MAX) * 0.1f;
        }

        // Create optimizer
        boat_optimizer_t* optimizer = boat_adam_optimizer_create(learning_rate, 0.9f, 0.999f, 1e-8f);

        // Register parameters
        boat_optimizer_add_parameter(optimizer, boat_variable_data(W), boat_variable_grad(W));
        boat_optimizer_add_parameter(optimizer, boat_variable_data(b), boat_variable_grad(b));

        // Run training
        results[result_count] = run_training(
            "Adam", optimizer, NULL, W, b, x_data, y_data, max_steps, loss_threshold);
        result_count++;

        // Cleanup
        boat_optimizer_free(optimizer);
        boat_variable_free(W);
        boat_variable_free(b);
    }

    // Test 2: Adam optimizer with StepLR scheduler
    {
        printf("\n2. Adam optimizer with StepLR scheduler\n");

        // Create parameters
        boat_variable_t* W = boat_variable_create_with_shape(W_shape, 2, BOAT_DTYPE_FLOAT32, true);
        boat_variable_t* b = boat_variable_create_with_shape(b_shape, 1, BOAT_DTYPE_FLOAT32, true);

        // Initialize with small random values
        float* W_data = (float*)boat_tensor_data(boat_variable_data(W));
        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        srand(123);
        for (int i = 0; i < input_dim * output_dim; i++) {
            W_data[i] = (rand() / (float)RAND_MAX) * 0.1f;
        }
        for (int i = 0; i < output_dim; i++) {
            b_data[i] = (rand() / (float)RAND_MAX) * 0.1f;
        }

        // Create optimizer
        boat_optimizer_t* optimizer = boat_adam_optimizer_create(learning_rate, 0.9f, 0.999f, 1e-8f);

        // Register parameters
        boat_optimizer_add_parameter(optimizer, boat_variable_data(W), boat_variable_grad(W));
        boat_optimizer_add_parameter(optimizer, boat_variable_data(b), boat_variable_grad(b));

        // Create scheduler
        boat_scheduler_t* scheduler = boat_step_lr_scheduler_create(learning_rate, 200, 0.5f);

        // Run training
        results[result_count] = run_training(
            "Adam+StepLR", optimizer, scheduler, W, b, x_data, y_data, max_steps, loss_threshold);
        result_count++;

        // Cleanup
        boat_scheduler_free(scheduler);
        boat_optimizer_free(optimizer);
        boat_variable_free(W);
        boat_variable_free(b);
    }

    // Test 3: RMSprop optimizer
    {
        printf("\n3. RMSprop optimizer (no scheduler)\n");

        // Create parameters
        boat_variable_t* W = boat_variable_create_with_shape(W_shape, 2, BOAT_DTYPE_FLOAT32, true);
        boat_variable_t* b = boat_variable_create_with_shape(b_shape, 1, BOAT_DTYPE_FLOAT32, true);

        // Initialize with small random values
        float* W_data = (float*)boat_tensor_data(boat_variable_data(W));
        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        srand(123);
        for (int i = 0; i < input_dim * output_dim; i++) {
            W_data[i] = (rand() / (float)RAND_MAX) * 0.1f;
        }
        for (int i = 0; i < output_dim; i++) {
            b_data[i] = (rand() / (float)RAND_MAX) * 0.1f;
        }

        // Create optimizer
        boat_optimizer_t* optimizer = boat_rmsprop_optimizer_create(learning_rate, 0.99f, 1e-8f);

        // Register parameters
        boat_optimizer_add_parameter(optimizer, boat_variable_data(W), boat_variable_grad(W));
        boat_optimizer_add_parameter(optimizer, boat_variable_data(b), boat_variable_grad(b));

        // Run training
        results[result_count] = run_training(
            "RMSprop", optimizer, NULL, W, b, x_data, y_data, max_steps, loss_threshold);
        result_count++;

        // Cleanup
        boat_optimizer_free(optimizer);
        boat_variable_free(W);
        boat_variable_free(b);
    }

    // Test 4: Adagrad optimizer
    {
        printf("\n4. Adagrad optimizer (no scheduler)\n");

        // Create parameters
        boat_variable_t* W = boat_variable_create_with_shape(W_shape, 2, BOAT_DTYPE_FLOAT32, true);
        boat_variable_t* b = boat_variable_create_with_shape(b_shape, 1, BOAT_DTYPE_FLOAT32, true);

        // Initialize with small random values
        float* W_data = (float*)boat_tensor_data(boat_variable_data(W));
        float* b_data = (float*)boat_tensor_data(boat_variable_data(b));
        srand(123);
        for (int i = 0; i < input_dim * output_dim; i++) {
            W_data[i] = (rand() / (float)RAND_MAX) * 0.1f;
        }
        for (int i = 0; i < output_dim; i++) {
            b_data[i] = (rand() / (float)RAND_MAX) * 0.1f;
        }

        // Create optimizer
        boat_optimizer_t* optimizer = boat_adagrad_optimizer_create(learning_rate, 1e-8f);

        // Register parameters
        boat_optimizer_add_parameter(optimizer, boat_variable_data(W), boat_variable_grad(W));
        boat_optimizer_add_parameter(optimizer, boat_variable_data(b), boat_variable_grad(b));

        // Run training
        results[result_count] = run_training(
            "Adagrad", optimizer, NULL, W, b, x_data, y_data, max_steps, loss_threshold);
        result_count++;

        // Cleanup
        boat_optimizer_free(optimizer);
        boat_variable_free(W);
        boat_variable_free(b);
    }

    // Print summary
    printf("\n\nBenchmark Results Summary\n");
    printf("=========================\n");
    printf("%-20s %-20s %-15s %-15s %-15s\n",
           "Optimizer", "Scheduler", "Steps to Converge", "Final Loss", "Time (ms)");
    printf("%-20s %-20s %-15s %-15s %-15s\n",
           "---------", "---------", "-----------------", "----------", "---------");

    for (int i = 0; i < result_count; i++) {
        printf("%-20s %-20s %-15d %-15.6f %-15.2f\n",
               results[i].optimizer_name,
               results[i].scheduler_name,
               results[i].steps_to_converge,
               results[i].final_loss,
               results[i].training_time_ms);
    }

    // Cleanup data
    boat_tensor_unref(x_data);
    boat_tensor_unref(y_data);
}

int main() {
    printf("Boat Framework Optimizer Benchmark\n");
    printf("===================================\n\n");

    run_optimizer_benchmark();

    return 0;
}