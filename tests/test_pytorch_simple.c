// Simple test for PyTorch model loading
#include <stdio.h>
#include <stdlib.h>

// Include boat headers
#include "../include/boat.h"
#include "../include/boat/format/pytorch.h"

int main() {
    printf("Testing PyTorch model loading...\n");

    const char* model_path = "test_simple_model.pt";

    // Check if file exists
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        printf("Error: Model file not found: %s\n", model_path);
        printf("Make sure to run tests/generate_test_model.py first\n");
        return 1;
    }
    fclose(f);

    // Check if it's a valid PyTorch model
    bool valid = boat_pytorch_check(model_path);
    if (!valid) {
        printf("Error: File is not a valid PyTorch model\n");
        return 1;
    }
    printf("Model validation passed\n");

    // Load the model
    printf("Loading model...\n");
    boat_model_t* model = boat_pytorch_load(model_path);
    if (!model) {
        printf("Error: Failed to load model\n");
        return 1;
    }
    printf("Model loaded successfully\n");

    // Check if model has user data (parameters)
    void* user_data = boat_model_get_user_data(model);
    if (user_data) {
        printf("Model has user data (parameters loaded)\n");
    } else {
        printf("Warning: Model has no user data\n");
    }

    // Try to get model info
    // TODO: Add more detailed checks

    // Clean up
    boat_model_free(model);
    printf("Test completed successfully\n");

    return 0;
}