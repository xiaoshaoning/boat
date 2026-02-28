// scheduler_usage.c - Example of using learning rate schedulers with optimizers
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/optimizers.h>
#include <boat/schedulers.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdio.h>
#include <math.h>

// Custom lambda function for LambdaLR scheduler
static float custom_lambda(int step, float base_lr) {
    // Decay learning rate exponentially: lr = base_lr * exp(-0.1 * step)
    return base_lr * expf(-0.1f * (float)step);
}

int main() {
    printf("Learning Rate Scheduler Usage Example\n");
    printf("=====================================\n\n");

    // Example 1: StepLR scheduler with Adam optimizer
    {
        printf("Example 1: StepLR with Adam\n");
        printf("----------------------------\n");

        // Create Adam optimizer with initial learning rate 0.01
        boat_optimizer_t* optimizer = boat_adam_optimizer_create(0.01f, 0.9f, 0.999f, 1e-8f);
        if (!optimizer) {
            printf("Failed to create optimizer\n");
            return 1;
        }

        // Create StepLR scheduler: decay by 0.5 every 10 steps
        boat_scheduler_t* scheduler = boat_step_lr_scheduler_create(0.01f, 10, 0.5f);
        if (!scheduler) {
            printf("Failed to create scheduler\n");
            boat_optimizer_free(optimizer);
            return 1;
        }

        printf("Initial learning rate: %f\n", boat_optimizer_get_learning_rate(optimizer));
        printf("Training for 25 steps:\n");

        // Simulate training loop
        for (int step = 1; step <= 25; step++) {
            // In a real training loop, you would:
            // 1. Compute forward pass
            // 2. Compute backward pass
            // 3. Call boat_optimizer_step(optimizer)
            // 4. Update learning rate with scheduler

            // Update scheduler
            boat_scheduler_step(scheduler);

            // Update optimizer learning rate
            boat_scheduler_update_optimizer(scheduler, optimizer);

            // Print learning rate every 5 steps
            if (step % 5 == 0 || step <= 3) {
                float current_lr = boat_optimizer_get_learning_rate(optimizer);
                printf("  Step %2d: learning rate = %f\n", step, current_lr);
            }
        }

        boat_scheduler_free(scheduler);
        boat_optimizer_free(optimizer);
        printf("\n");
    }

    // Example 2: CosineAnnealing scheduler with RMSprop optimizer
    {
        printf("Example 2: CosineAnnealing with RMSprop\n");
        printf("----------------------------------------\n");

        // Create RMSprop optimizer
        boat_optimizer_t* optimizer = boat_rmsprop_optimizer_create(0.05f, 0.99f, 1e-8f);
        if (!optimizer) {
            printf("Failed to create optimizer\n");
            return 1;
        }

        // Create CosineAnnealing scheduler: anneal from 0.05 to 0.005 over 20 steps
        boat_scheduler_t* scheduler = boat_cosine_annealing_scheduler_create(0.05f, 20, 0.005f);
        if (!scheduler) {
            printf("Failed to create scheduler\n");
            boat_optimizer_free(optimizer);
            return 1;
        }

        printf("Initial learning rate: %f\n", boat_optimizer_get_learning_rate(optimizer));
        printf("Training for 20 steps (one full cosine cycle):\n");

        for (int step = 1; step <= 20; step++) {
            boat_scheduler_step(scheduler);
            boat_scheduler_update_optimizer(scheduler, optimizer);

            float current_lr = boat_optimizer_get_learning_rate(optimizer);
            printf("  Step %2d: learning rate = %f\n", step, current_lr);
        }

        printf("Final learning rate after 20 steps: %f\n", boat_optimizer_get_learning_rate(optimizer));
        printf("\n");

        boat_scheduler_free(scheduler);
        boat_optimizer_free(optimizer);
    }

    // Example 3: LambdaLR scheduler with Adagrad optimizer
    {
        printf("Example 3: LambdaLR with Adagrad\n");
        printf("---------------------------------\n");

        // Create Adagrad optimizer
        boat_optimizer_t* optimizer = boat_adagrad_optimizer_create(0.1f, 1e-8f);
        if (!optimizer) {
            printf("Failed to create optimizer\n");
            return 1;
        }

        // Create LambdaLR scheduler with custom exponential decay function
        boat_scheduler_t* scheduler = boat_lambda_lr_scheduler_create(0.1f, custom_lambda);
        if (!scheduler) {
            printf("Failed to create scheduler\n");
            boat_optimizer_free(optimizer);
            return 1;
        }

        printf("Initial learning rate: %f\n", boat_optimizer_get_learning_rate(optimizer));
        printf("Training for 10 steps with exponential decay:\n");

        for (int step = 1; step <= 10; step++) {
            boat_scheduler_step(scheduler);
            boat_scheduler_update_optimizer(scheduler, optimizer);

            float current_lr = boat_optimizer_get_learning_rate(optimizer);
            float expected_lr = 0.1f * expf(-0.1f * (float)step);
            printf("  Step %2d: learning rate = %f (expected: %f)\n", step, current_lr, expected_lr);
        }

        printf("\n");

        boat_scheduler_free(scheduler);
        boat_optimizer_free(optimizer);
    }

    // Example 4: Manual scheduler usage (without automatic update)
    {
        printf("Example 4: Manual scheduler usage\n");
        printf("----------------------------------\n");

        boat_optimizer_t* optimizer = boat_adam_optimizer_create(0.01f, 0.9f, 0.999f, 1e-8f);
        boat_scheduler_t* scheduler = boat_step_lr_scheduler_create(0.01f, 5, 0.8f);

        printf("Manual learning rate adjustment:\n");
        printf("  Initial optimizer LR: %f\n", boat_optimizer_get_learning_rate(optimizer));

        // Manual pattern: update scheduler, get new LR, set it on optimizer
        for (int i = 0; i < 3; i++) {
            boat_scheduler_step(scheduler);
            float new_lr = boat_scheduler_get_last_lr(scheduler);
            boat_optimizer_set_learning_rate(optimizer, new_lr);
            printf("  After %d scheduler steps: optimizer LR = %f\n", i + 1, new_lr);
        }

        // Check next LR without taking a step
        float next_lr = boat_scheduler_get_next_lr(scheduler);
        printf("  Next LR (without step): %f\n", next_lr);

        boat_scheduler_free(scheduler);
        boat_optimizer_free(optimizer);
        printf("\n");
    }

    printf("All examples completed successfully!\n");
    return 0;
}