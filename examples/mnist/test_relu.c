// test_relu.c - Minimal test for boat_relu function
#include <boat.h>
#include <boat/tensor.h>
#include <boat/ops.h>
#include <stdio.h>

int main() {
    printf("=== Testing boat_relu function ===\n");

    // 1. Create a simple tensor
    int64_t shape[] = {2, 3};
    boat_tensor_t* tensor = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    if (!tensor) {
        printf("ERROR: Failed to create tensor\n");
        return 1;
    }
    printf("Created tensor at %p\n", (void*)tensor);

    // 2. Fill with test data [-3, -2, -1, 0, 1, 2]
    float* data = (float*)boat_tensor_data(tensor);
    for (int i = 0; i < 6; i++) {
        data[i] = (float)(i - 3);
    }
    printf("Tensor data: ");
    for (int i = 0; i < 6; i++) {
        printf("%.1f ", data[i]);
    }
    printf("\n");

    // 3. Test boat_relu
    printf("Calling boat_relu(tensor)...\n");
    printf("Function address: %p\n", (void*)boat_relu);
    boat_tensor_t* result = boat_relu(tensor);

    if (result) {
        printf("SUCCESS: boat_relu returned tensor at %p\n", (void*)result);

        // Check result values
        float* result_data = (float*)boat_tensor_data(result);
        printf("Result data: ");
        for (int i = 0; i < 6; i++) {
            printf("%.1f ", result_data[i]);
        }
        printf("\n");

        // Expected: [0, 0, 0, 0, 1, 2]
        printf("Expected:   0.0 0.0 0.0 0.0 1.0 2.0\n");

        boat_tensor_unref(result);
    } else {
        printf("FAILED: boat_relu returned NULL\n");
    }

    // 4. Cleanup
    boat_tensor_unref(tensor);

    printf("=== Test completed ===\n");
    return 0;
}