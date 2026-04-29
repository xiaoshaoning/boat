// Simple test for boat_transpose implementation
#include <boat/tensor.h>
#include <boat/ops.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void test_2d_transpose() {
    printf("Testing 2D transpose...\n");
    // Create 2x3 matrix
    int64_t shape[] = {2, 3};
    boat_tensor_t* t = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* data = (float*)boat_tensor_data(t);
    // Fill with sequential values
    for (int i = 0; i < 6; i++) {
        data[i] = (float)i + 1.0f;
    }
    printf("Original matrix (2x3):\n");
    for (int i = 0; i < 2; i++) {
        printf("  ");
        for (int j = 0; j < 3; j++) {
            printf("%.1f ", data[i*3 + j]);
        }
        printf("\n");
    }

    // Transpose dimensions 0 and 1 (2D transpose)
    boat_tensor_t* t_t = boat_transpose(t, 0, 1);
    if (!t_t) {
        printf("ERROR: transpose failed\n");
        boat_tensor_unref(t);
        return;
    }

    const int64_t* out_shape = boat_tensor_shape(t_t);
    printf("Transposed shape: [%lld, %lld]\n", out_shape[0], out_shape[1]);
    float* out_data = (float*)boat_tensor_data(t_t);
    printf("Transposed matrix (3x2):\n");
    for (int i = 0; i < 3; i++) {
        printf("  ");
        for (int j = 0; j < 2; j++) {
            printf("%.1f ", out_data[i*2 + j]);
        }
        printf("\n");
    }

    // Verify values
    bool correct = true;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (fabs(data[i*3 + j] - out_data[j*2 + i]) > 1e-6) {
                printf("ERROR: mismatch at (%d,%d): %.1f vs %.1f\n",
                       i, j, data[i*3 + j], out_data[j*2 + i]);
                correct = false;
            }
        }
    }
    if (correct) {
        printf("2D transpose PASSED\n");
    } else {
        printf("2D transpose FAILED\n");
    }

    boat_tensor_unref(t_t);
    boat_tensor_unref(t);
}

void test_4d_transpose_last_two_dims() {
    printf("\nTesting 4D transpose (swap last two dimensions)...\n");
    // Shape: [batch=2, heads=2, seq_len=3, head_size=4]
    int64_t shape[] = {2, 2, 3, 4};
    boat_tensor_t* t = boat_tensor_create(shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* data = (float*)boat_tensor_data(t);
    size_t total = 2*2*3*4;
    for (size_t i = 0; i < total; i++) {
        data[i] = (float)i + 1.0f;
    }

    printf("Original shape: [%lld, %lld, %lld, %lld]\n", shape[0], shape[1], shape[2], shape[3]);

    // Transpose dimensions 2 and 3 (last two dimensions)
    boat_tensor_t* t_t = boat_transpose(t, 2, 3);
    if (!t_t) {
        printf("ERROR: transpose failed\n");
        boat_tensor_unref(t);
        return;
    }

    const int64_t* out_shape = boat_tensor_shape(t_t);
    printf("Transposed shape: [%lld, %lld, %lld, %lld]\n",
           out_shape[0], out_shape[1], out_shape[2], out_shape[3]);

    // Expected shape: [2, 2, 4, 3]
    if (out_shape[0] != 2 || out_shape[1] != 2 || out_shape[2] != 4 || out_shape[3] != 3) {
        printf("ERROR: wrong output shape\n");
        boat_tensor_unref(t_t);
        boat_tensor_unref(t);
        return;
    }

    float* out_data = (float*)boat_tensor_data(t_t);

    // Verify: for each batch and head, the last two dimensions are transposed
    bool correct = true;
    for (int b = 0; b < 2; b++) {
        for (int h = 0; h < 2; h++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    // Original index: b*2*3*4 + h*3*4 + i*4 + j
                    size_t idx_orig = ((b * 2 + h) * 3 + i) * 4 + j;
                    // Transposed index: b*2*4*3 + h*4*3 + j*3 + i
                    size_t idx_trans = ((b * 2 + h) * 4 + j) * 3 + i;
                    if (fabs(data[idx_orig] - out_data[idx_trans]) > 1e-6) {
                        printf("ERROR: mismatch at (b=%d,h=%d,i=%d,j=%d): %.1f vs %.1f\n",
                               b, h, i, j, data[idx_orig], out_data[idx_trans]);
                        correct = false;
                        if (!correct && b == 0 && h == 0 && i == 0 && j == 0) {
                            printf("First element mismatch details\n");
                        }
                    }
                }
            }
        }
    }

    if (correct) {
        printf("4D transpose PASSED\n");
    } else {
        printf("4D transpose FAILED\n");
    }

    boat_tensor_unref(t_t);
    boat_tensor_unref(t);
}

void test_transpose_identity() {
    printf("\nTesting transpose identity (transpose twice)...\n");
    int64_t shape[] = {2, 3, 4};
    boat_tensor_t* t = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* data = (float*)boat_tensor_data(t);
    size_t total = 2*3*4;
    for (size_t i = 0; i < total; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }

    // Transpose dimensions 0 and 1
    boat_tensor_t* t1 = boat_transpose(t, 0, 1);
    if (!t1) {
        printf("ERROR: first transpose failed\n");
        boat_tensor_unref(t);
        return;
    }

    // Transpose back
    boat_tensor_t* t2 = boat_transpose(t1, 0, 1);
    if (!t2) {
        printf("ERROR: second transpose failed\n");
        boat_tensor_unref(t1);
        boat_tensor_unref(t);
        return;
    }

    // Compare with original
    float* orig_data = (float*)boat_tensor_data(t);
    float* final_data = (float*)boat_tensor_data(t2);
    bool correct = true;
    for (size_t i = 0; i < total; i++) {
        if (fabs(orig_data[i] - final_data[i]) > 1e-6) {
            printf("ERROR: double transpose not identity at %zu: %.6f vs %.6f\n",
                   i, orig_data[i], final_data[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Double transpose identity test PASSED\n");
    } else {
        printf("Double transpose identity test FAILED\n");
    }

    boat_tensor_unref(t2);
    boat_tensor_unref(t1);
    boat_tensor_unref(t);
}

int main() {
    printf("=== Testing boat_transpose implementation ===\n\n");
    test_2d_transpose();
    test_4d_transpose_last_two_dims();
    test_transpose_identity();
    printf("\n=== All tests completed ===\n");
    return 0;
}