// test_reduce_autodiff.c - Reduction ops (sum/mean/max/min) with axis support
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/autodiff.h>
#include <boat/ops.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

static float fabsdiff(float a, float b) {
    float d = a - b;
    return d < 0 ? -d : d;
}

// Numerical gradient of a scalar function of x[idx].
static float numeric_grad(float* x, int idx, int n, float (*fn)(const float*, int)) {
    float eps = 1e-3f;
    float orig = x[idx];
    x[idx] = orig + eps;
    float l1 = fn(x, n);
    x[idx] = orig - eps;
    float l2 = fn(x, n);
    x[idx] = orig;
    return (l1 - l2) / (2.0f * eps);
}

// fn: compute scalar = full sum of x.
static float fn_sum(const float* x, int n) {
    float s = 0;
    for (int i = 0; i < n; i++)
        s += x[i];
    return s;
}
// fn: compute scalar = full mean of x.
static float fn_mean(const float* x, int n) {
    float s = 0;
    for (int i = 0; i < n; i++)
        s += x[i];
    return s / (float)n;
}

static void test_tensor_level_reductions(void) {
    printf("Testing tensor-level reductions...\n");
    int64_t sh[] = {2, 3};
    float d[] = {1, 2, 3, 4, 5, 6};
    boat_tensor_t* a = boat_tensor_from_data(sh, 2, BOAT_DTYPE_FLOAT32, d);

    int64_t dim1[] = {1};
    int64_t dim0[] = {0};

    boat_tensor_t* s = boat_sum(a, dim1, 1, false);
    assert(s && boat_tensor_ndim(s) == 1 && boat_tensor_shape(s)[0] == 2);
    assert(((float*)boat_tensor_data(s))[0] == 6.0f && ((float*)boat_tensor_data(s))[1] == 15.0f);
    boat_tensor_free(s);

    boat_tensor_t* m = boat_mean(a, dim0, 1, false);
    float* md = (float*)boat_tensor_data(m);
    assert(m && boat_tensor_shape(m)[0] == 3);
    assert(fabsdiff(md[0], 2.5f) < 1e-5 && fabsdiff(md[1], 3.5f) < 1e-5 &&
           fabsdiff(md[2], 4.5f) < 1e-5);
    boat_tensor_free(m);

    // Full reduction -> scalar (ndim 0)
    boat_tensor_t* mx = boat_max(a, NULL, 0, false);
    assert(mx && boat_tensor_ndim(mx) == 0 && boat_tensor_nelements(mx) == 1);
    assert(((float*)boat_tensor_data(mx))[0] == 6.0f);
    boat_tensor_free(mx);

    boat_tensor_t* mn = boat_min(a, dim1, 1, true);
    assert(mn && boat_tensor_ndim(mn) == 2);
    assert(boat_tensor_shape(mn)[0] == 2 && boat_tensor_shape(mn)[1] == 1);
    assert(((float*)boat_tensor_data(mn))[0] == 1.0f && ((float*)boat_tensor_data(mn))[1] == 4.0f);
    boat_tensor_free(mn);

    boat_tensor_free(a);
    printf("  OK\n");
}

static void test_autodiff_sum_gradient(void) {
    printf("Testing autodiff sum gradient...\n");
    int64_t sh[] = {2, 3};
    float vals[] = {1, 2, 3, 4, 5, 6};
    boat_variable_t* x = boat_variable_create_with_shape(sh, 2, BOAT_DTYPE_FLOAT32, true);
    float* xd = (float*)boat_tensor_data(boat_variable_data(x));
    for (int i = 0; i < 6; i++)
        xd[i] = vals[i];

    // Full sum (scalar). Analytic grad = all ones.
    boat_variable_t* s = boat_var_sum(x, NULL, 0, false);
    assert(s != NULL);
    assert(boat_tensor_nelements(boat_variable_data(s)) == 1);
    assert(fabsdiff(((float*)boat_tensor_data(boat_variable_data(s)))[0], 21.0f) < 1e-4);

    boat_variable_backward(s, NULL);
    float* g = (float*)boat_tensor_data(boat_variable_grad(x));
    for (int i = 0; i < 6; i++)
        assert(fabsdiff(g[i], 1.0f) < 1e-5);
    boat_variable_free(s);

    // mean over dim 0. Analytic grad = 0.5 everywhere.
    boat_variable_zero_grad(x);
    int64_t dim0[] = {0};
    boat_variable_t* m = boat_var_mean(x, dim0, 1, false);
    assert(m != NULL);
    boat_variable_backward(m, NULL);
    g = (float*)boat_tensor_data(boat_variable_grad(x));
    for (int i = 0; i < 6; i++)
        assert(fabsdiff(g[i], 0.5f) < 1e-5);
    boat_variable_free(m);

    // max over dim 1. Analytic grad routes only to argmax.
    boat_variable_zero_grad(x);
    int64_t dim1[] = {1};
    boat_variable_t* mx = boat_var_max(x, dim1, 1, false);
    assert(mx != NULL);
    boat_variable_backward(mx, NULL);
    g = (float*)boat_tensor_data(boat_variable_grad(x));
    float want[] = {0, 0, 1, 0, 0, 1};
    for (int i = 0; i < 6; i++)
        assert(fabsdiff(g[i], want[i]) < 1e-5);
    boat_variable_free(mx);

    boat_variable_free(x);
    printf("  OK\n");
}

static void test_numeric_gradient_check(void) {
    printf("Testing numerical gradient check (sum/mean)...\n");
    int64_t sh[] = {4};
    boat_variable_t* x = boat_variable_create_with_shape(sh, 1, BOAT_DTYPE_FLOAT32, true);
    float* xd = (float*)boat_tensor_data(boat_variable_data(x));
    float vals[] = {0.5f, -1.2f, 3.3f, 2.0f};
    for (int i = 0; i < 4; i++)
        xd[i] = vals[i];

    // sum: numeric vs analytic (all ones)
    boat_variable_t* s = boat_var_sum(x, NULL, 0, false);
    boat_variable_backward(s, NULL);
    float* g = (float*)boat_tensor_data(boat_variable_grad(x));
    for (int i = 0; i < 4; i++)
        assert(fabsdiff(g[i], numeric_grad(xd, i, 4, fn_sum)) < 1e-2);
    boat_variable_free(s);

    // mean: numeric vs analytic (all 0.25)
    boat_variable_zero_grad(x);
    for (int i = 0; i < 4; i++)
        xd[i] = vals[i];
    boat_variable_t* m = boat_var_mean(x, NULL, 0, false);
    boat_variable_backward(m, NULL);
    g = (float*)boat_tensor_data(boat_variable_grad(x));
    for (int i = 0; i < 4; i++)
        assert(fabsdiff(g[i], numeric_grad(xd, i, 4, fn_mean)) < 1e-2);
    boat_variable_free(m);

    boat_variable_free(x);
    printf("  OK\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Reduction Autodiff Tests ===\n\n");
    test_tensor_level_reductions();
    test_autodiff_sum_gradient();
    test_numeric_gradient_check();
    printf("\n=== All reduction tests passed ===\n");
    return 0;
}
