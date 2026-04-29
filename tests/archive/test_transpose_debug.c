// Debug transpose implementation
#include <boat/tensor.h>
#include <boat/ops.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Simple 2x3 matrix
    int64_t shape[] = {2, 3};
    boat_tensor_t* t = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* data = (float*)boat_tensor_data(t);
    for (int i = 0; i < 6; i++) data[i] = (float)i + 1.0f;

    printf("Original: ");
    for (int i = 0; i < 6; i++) printf("%.1f ", data[i]);
    printf("\n");

    boat_tensor_t* t_t = boat_transpose(t, 0, 1);
    float* out = (float*)boat_tensor_data(t_t);

    printf("Transposed: ");
    for (int i = 0; i < 6; i++) printf("%.1f ", out[i]);
    printf("\n");

    // Expected: [1, 4, 2, 5, 3, 6]
    printf("Expected: 1.0 4.0 2.0 5.0 3.0 6.0\n");

    // Print mapping
    printf("\nIndex mapping:\n");
    for (int idx = 0; idx < 6; idx++) {
        int i = idx / 3;
        int j = idx % 3;
        int out_idx = j*2 + i;
        printf("  idx=%d (i=%d,j=%d) -> out_idx=%d (value %.1f)\n",
               idx, i, j, out_idx, out[out_idx]);
    }

    boat_tensor_unref(t_t);
    boat_tensor_unref(t);
    return 0;
}