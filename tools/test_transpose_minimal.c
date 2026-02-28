#include <boat/tensor.h>
#include <boat/ops.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    // Create 2x3 matrix
    int64_t shape[] = {2, 3};
    boat_tensor_t* t = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* data = (float*)boat_tensor_data(t);
    for (int i = 0; i < 6; i++) {
        data[i] = (float)i + 1.0f;
    }
    printf("Input data: ");
    for (int i = 0; i < 6; i++) printf("%.1f ", data[i]);
    printf("\n");

    // Transpose
    boat_tensor_t* t_t = boat_transpose(t, 0, 1);
    if (!t_t) {
        printf("Transpose failed\n");
        return 1;
    }

    const int64_t* out_shape = boat_tensor_shape(t_t);
    printf("Output shape: [%lld, %lld]\n", out_shape[0], out_shape[1]);
    float* out_data = (float*)boat_tensor_data(t_t);
    printf("Output data: ");
    for (int i = 0; i < 6; i++) printf("%.1f ", out_data[i]);
    printf("\n");

    // Expected output
    float expected[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    printf("Expected:    ");
    for (int i = 0; i < 6; i++) printf("%.1f ", expected[i]);
    printf("\n");

    // Check
    int correct = 1;
    for (int i = 0; i < 6; i++) {
        if (fabs(out_data[i] - expected[i]) > 1e-6) {
            printf("Mismatch at %d: %.1f vs %.1f\n", i, out_data[i], expected[i]);
            correct = 0;
        }
    }
    printf(correct ? "PASS\n" : "FAIL\n");

    boat_tensor_unref(t_t);
    boat_tensor_unref(t);
    return 0;
}