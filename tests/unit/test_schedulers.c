// test_schedulers.c - Learning rate scheduler unit tests
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat/schedulers.h>
#include <boat/optimizers.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

// Helper function to compare floating point numbers with tolerance
static int float_equal(float a, float b, float tolerance) {
    return fabsf(a - b) <= tolerance;
}

// Lambda function for testing LambdaLR scheduler
static float test_lambda_fn(int step, float base_lr) {
    // Simple decay: lr = base_lr * (0.95)^step
    return base_lr * powf(0.95f, (float)step);
}

int main() {
    printf("Testing learning rate schedulers...\n");

    // Test 1: StepLR scheduler creation and basic functionality
    {
        printf("  Testing StepLR scheduler...\n");
        boat_scheduler_t* scheduler = boat_step_lr_scheduler_create(0.1f, 10, 0.5f);
        assert(scheduler != NULL);

        // Test invalid parameters
        boat_scheduler_t* invalid1 = boat_step_lr_scheduler_create(0.0f, 10, 0.5f);
        assert(invalid1 == NULL);
        boat_scheduler_t* invalid2 = boat_step_lr_scheduler_create(0.1f, 0, 0.5f);
        assert(invalid2 == NULL);
        boat_scheduler_t* invalid3 = boat_step_lr_scheduler_create(0.1f, 10, 0.0f);
        assert(invalid3 == NULL);

        // Initial learning rate should be base_learning_rate
        float lr = boat_scheduler_get_last_lr(scheduler);
        assert(float_equal(lr, 0.1f, 1e-6f));

        // Take 9 steps (should not decay yet)
        for (int i = 0; i < 9; i++) {
            boat_scheduler_step(scheduler);
            lr = boat_scheduler_get_last_lr(scheduler);
            assert(float_equal(lr, 0.1f, 1e-6f));
        }

        // 10th step should decay
        boat_scheduler_step(scheduler);
        lr = boat_scheduler_get_last_lr(scheduler);
        assert(float_equal(lr, 0.1f * 0.5f, 1e-6f));

        // 20th step should decay again
        for (int i = 0; i < 10; i++) {
            boat_scheduler_step(scheduler);
        }
        lr = boat_scheduler_get_last_lr(scheduler);
        assert(float_equal(lr, 0.1f * 0.5f * 0.5f, 1e-6f));

        // Test reset
        boat_scheduler_reset(scheduler);
        lr = boat_scheduler_get_last_lr(scheduler);
        assert(float_equal(lr, 0.1f, 1e-6f));

        boat_scheduler_free(scheduler);
        printf("    StepLR tests passed\n");
    }

    // Test 2: CosineAnnealing scheduler
    {
        printf("  Testing CosineAnnealing scheduler...\n");
        boat_scheduler_t* scheduler = boat_cosine_annealing_scheduler_create(0.1f, 20, 0.01f);
        assert(scheduler != NULL);

        // Test invalid parameters
        boat_scheduler_t* invalid1 = boat_cosine_annealing_scheduler_create(0.0f, 20, 0.01f);
        assert(invalid1 == NULL);
        boat_scheduler_t* invalid2 = boat_cosine_annealing_scheduler_create(0.1f, 0, 0.01f);
        assert(invalid2 == NULL);
        boat_scheduler_t* invalid3 = boat_cosine_annealing_scheduler_create(0.1f, 20, -0.01f);
        assert(invalid3 == NULL);
        boat_scheduler_t* invalid4 = boat_cosine_annealing_scheduler_create(0.1f, 20, 0.1f);
        assert(invalid4 == NULL); // eta_min >= base_lr

        // Initial learning rate should be base_learning_rate
        float lr = boat_scheduler_get_last_lr(scheduler);
        assert(float_equal(lr, 0.1f, 1e-6f));

        // Take steps and verify cosine annealing curve
        for (int i = 0; i < 20; i++) {
            boat_scheduler_step(scheduler);
            lr = boat_scheduler_get_last_lr(scheduler);

            // Manual calculation for verification
            float step = (float)(i + 1);
            float expected = 0.01f + 0.5f * (0.1f - 0.01f) * (1.0f + cosf(3.14159265358979323846f * step / 20.0f));

            assert(float_equal(lr, expected, 1e-6f));
        }

        // After T_max steps, learning rate should be eta_min (with some floating point tolerance)
        lr = boat_scheduler_get_last_lr(scheduler);
        assert(float_equal(lr, 0.01f, 1e-5f));

        // Test reset
        boat_scheduler_reset(scheduler);
        lr = boat_scheduler_get_last_lr(scheduler);
        assert(float_equal(lr, 0.1f, 1e-6f));

        boat_scheduler_free(scheduler);
        printf("    CosineAnnealing tests passed\n");
    }

    // Test 3: LambdaLR scheduler
    {
        printf("  Testing LambdaLR scheduler...\n");
        boat_scheduler_t* scheduler = boat_lambda_lr_scheduler_create(0.1f, test_lambda_fn);
        assert(scheduler != NULL);

        // Test invalid parameters
        boat_scheduler_t* invalid1 = boat_lambda_lr_scheduler_create(0.0f, test_lambda_fn);
        assert(invalid1 == NULL);
        boat_scheduler_t* invalid2 = boat_lambda_lr_scheduler_create(0.1f, NULL);
        assert(invalid2 == NULL);

        // Initial learning rate should be base_learning_rate
        float lr = boat_scheduler_get_last_lr(scheduler);
        assert(float_equal(lr, 0.1f, 1e-6f));

        // Take steps and verify custom lambda function is used
        for (int i = 0; i < 10; i++) {
            boat_scheduler_step(scheduler);
            lr = boat_scheduler_get_last_lr(scheduler);

            float expected = 0.1f * powf(0.95f, (float)(i + 1));
            assert(float_equal(lr, expected, 1e-6f));
        }

        // Test reset
        boat_scheduler_reset(scheduler);
        lr = boat_scheduler_get_last_lr(scheduler);
        assert(float_equal(lr, 0.1f, 1e-6f));

        boat_scheduler_free(scheduler);
        printf("    LambdaLR tests passed\n");
    }

    // Test 4: Scheduler integration with optimizer
    {
        printf("  Testing scheduler-optimizer integration...\n");

        // Create optimizer and scheduler
        boat_optimizer_t* optimizer = boat_adam_optimizer_create(0.1f, 0.9f, 0.999f, 1e-8f);
        assert(optimizer != NULL);

        boat_scheduler_t* scheduler = boat_step_lr_scheduler_create(0.1f, 5, 0.5f);
        assert(scheduler != NULL);

        // Verify initial learning rate
        float optimizer_lr = boat_optimizer_get_learning_rate(optimizer);
        float scheduler_lr = boat_scheduler_get_last_lr(scheduler);
        assert(float_equal(optimizer_lr, 0.1f, 1e-6f));
        assert(float_equal(scheduler_lr, 0.1f, 1e-6f));

        // Take 4 steps with scheduler, update optimizer each time
        for (int i = 0; i < 4; i++) {
            boat_scheduler_step(scheduler);
            boat_scheduler_update_optimizer(scheduler, optimizer);

            optimizer_lr = boat_optimizer_get_learning_rate(optimizer);
            scheduler_lr = boat_scheduler_get_last_lr(scheduler);
            assert(float_equal(optimizer_lr, scheduler_lr, 1e-6f));
            assert(float_equal(optimizer_lr, 0.1f, 1e-6f)); // Not decayed yet
        }

        // 5th step should decay and update optimizer
        boat_scheduler_step(scheduler);
        boat_scheduler_update_optimizer(scheduler, optimizer);

        optimizer_lr = boat_optimizer_get_learning_rate(optimizer);
        scheduler_lr = boat_scheduler_get_last_lr(scheduler);
        assert(float_equal(optimizer_lr, scheduler_lr, 1e-6f));
        assert(float_equal(optimizer_lr, 0.1f * 0.5f, 1e-6f));

        // Test direct learning rate setting
        boat_optimizer_set_learning_rate(optimizer, 0.05f);
        optimizer_lr = boat_optimizer_get_learning_rate(optimizer);
        assert(float_equal(optimizer_lr, 0.05f, 1e-6f));

        boat_optimizer_free(optimizer);
        boat_scheduler_free(scheduler);
        printf("    Integration tests passed\n");
    }

    // Test 5: Edge cases and error handling
    {
        printf("  Testing edge cases...\n");

        // NULL scheduler operations should not crash
        boat_scheduler_step(NULL);
        boat_scheduler_reset(NULL);
        boat_scheduler_free(NULL);
        float lr = boat_scheduler_get_last_lr(NULL);
        assert(float_equal(lr, 0.0f, 1e-6f));

        // NULL optimizer operations should not crash
        boat_optimizer_set_learning_rate(NULL, 0.1f);
        lr = boat_optimizer_get_learning_rate(NULL);
        assert(float_equal(lr, 0.0f, 1e-6f));

        // NULL scheduler with optimizer update
        boat_scheduler_update_optimizer(NULL, NULL);

        printf("    Edge case tests passed\n");
    }

    printf("All scheduler tests passed!\n");
    return 0;
}