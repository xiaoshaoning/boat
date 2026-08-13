// test_inverse.c - Matrix inverse tests
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/ops.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

static double fd(double a, double b) { double d = a - b; return d < 0 ? -d : d; }

// Verify A * Ainv == identity for a single n x n float32 matrix (row-major).
static int check_identity(const float* a, const float* inv, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int k = 0; k < n; k++) s += (double)a[i * n + k] * (double)inv[k * n + j];
            if (fd(s, i == j ? 1.0 : 0.0) > 1e-4) return 0;
        }
    return 1;
}

static void test_inverse_2d(void) {
    printf("Testing 2D matrix inverse...\n");
    int64_t sh[] = {3, 3};
    float a[] = {2,1,1, 1,3,2, 1,0,0};
    boat_tensor_t* t = boat_tensor_from_data(sh, 2, BOAT_DTYPE_FLOAT32, a);
    boat_tensor_t* inv = boat_inverse(t);
    assert(inv != NULL);
    assert(boat_tensor_ndim(inv) == 2);
    assert(check_identity(a, (float*)boat_tensor_data(inv), 3));
    boat_tensor_free(t); boat_tensor_free(inv);

    // 2x2 known inverse
    int64_t sh2[] = {2, 2};
    float a2[] = {4, 7, 2, 6};
    boat_tensor_t* t2 = boat_tensor_from_data(sh2, 2, BOAT_DTYPE_FLOAT32, a2);
    boat_tensor_t* inv2 = boat_inverse(t2);
    float* d = (float*)boat_tensor_data(inv2);
    assert(fd(d[0], 0.6) < 1e-4 && fd(d[1], -0.7) < 1e-4);
    assert(fd(d[2], -0.2) < 1e-4 && fd(d[3], 0.4) < 1e-4);
    boat_tensor_free(t2); boat_tensor_free(inv2);
    printf("  OK\n");
}

static void test_inverse_batched(void) {
    printf("Testing batched matrix inverse...\n");
    int64_t sh[] = {2, 2, 2};
    float a[] = {1,2, 3,4,  2,0, 0,4};
    boat_tensor_t* t = boat_tensor_from_data(sh, 3, BOAT_DTYPE_FLOAT32, a);
    boat_tensor_t* inv = boat_inverse(t);
    assert(inv != NULL && boat_tensor_ndim(inv) == 3);
    float* d = (float*)boat_tensor_data(inv);
    assert(check_identity(a, d, 2));
    assert(check_identity(a + 4, d + 4, 2));
    boat_tensor_free(t); boat_tensor_free(inv);
    printf("  OK\n");
}

static void test_inverse_float64_and_singular(void) {
    printf("Testing float64 and singular...\n");
    int64_t sh[] = {2, 2};
    double a[] = {3, 1, 2, 2};  // inv = [[0.5,-0.25],[-0.5,0.75]]
    boat_tensor_t* t = boat_tensor_from_data(sh, 2, BOAT_DTYPE_FLOAT64, a);
    boat_tensor_t* inv = boat_inverse(t);
    double* d = (double*)boat_tensor_data(inv);
    assert(fd(d[0], 0.5) < 1e-9 && fd(d[1], -0.25) < 1e-9);
    assert(fd(d[2], -0.5) < 1e-9 && fd(d[3], 0.75) < 1e-9);
    boat_tensor_free(t); boat_tensor_free(inv);

    // singular -> NULL
    float s[] = {1, 2, 2, 4};
    boat_tensor_t* ts = boat_tensor_from_data(sh, 2, BOAT_DTYPE_FLOAT32, s);
    assert(boat_inverse(ts) == NULL);
    boat_tensor_free(ts);
    printf("  OK\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Matrix Inverse Tests ===\n\n");
    test_inverse_2d();
    test_inverse_batched();
    test_inverse_float64_and_singular();
    printf("\n=== All inverse tests passed ===\n");
    return 0;
}
