// simd_benchmark.c - Throughput benchmark for the SIMD kernels, reductions,
// transpose and the conv2d im2col+SGEMM path.
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0
//
// Run with OMP_NUM_THREADS=N to see the OpenMP scaling of the conv and reduce
// paths. The scalar baselines (sum/max/min/transpose) are computed by the
// same kernels without SIMD flags; the numbers here are the current build.

#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/simd.h>
#include <boat/tensor.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_sec(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

static float rnd(void) {
    return ((float)(rand() % 2001) - 1000.0f) / 100.0f;
}

static void bench_reduce_kernels(void) {
    const size_t N = 8u << 20;  // 32 MB
    float* a = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) a[i] = rnd();
    volatile float sink = 0.0f;

    double t0 = now_sec();
    for (int r = 0; r < 30; r++) sink = boat_simd_sum_reduce_f32(a, N);
    double t1 = now_sec();
    printf("sum reduce  %.1f MB x30: %.3f s  (%.1f GB/s)\n", N * 4.0 / 1e6, t1 - t0,
           30.0 * N * 4 / 1e9 / (t1 - t0));

    t0 = now_sec();
    for (int r = 0; r < 30; r++) sink = boat_simd_max_reduce_f32(a, N);
    t1 = now_sec();
    printf("max reduce  %.1f MB x30: %.3f s  (%.1f GB/s)\n", N * 4.0 / 1e6, t1 - t0,
           30.0 * N * 4 / 1e9 / (t1 - t0));
    (void)sink;
    free(a);
}

static void bench_transpose(void) {
    const size_t R = 1024, C = 1024;
    float* src = (float*)malloc(R * C * sizeof(float));
    float* dst = (float*)malloc(R * C * sizeof(float));
    for (size_t i = 0; i < R * C; i++) src[i] = rnd();
    double t0 = now_sec();
    for (int r = 0; r < 20; r++) boat_simd_transpose2d_f32(src, dst, R, C);
    double t1 = now_sec();
    printf("transpose %zux%zu x20: %.3f s  (%.1f GB/s)\n", R, C, t1 - t0,
           20.0 * R * C * 4 / 1e9 / (t1 - t0));
    free(src);
    free(dst);
}

static void bench_conv(void) {
    int batch = 16, in_ch = 64, out_ch = 64, h = 32, wi = 32, kh = 3, kw = 3, pad = 1;
    size_t nin = (size_t)batch * in_ch * h * wi;
    size_t nw = (size_t)out_ch * in_ch * kh * kw;
    float* in = (float*)malloc(nin * sizeof(float));
    float* w = (float*)malloc(nw * sizeof(float));
    float bias[64];
    for (size_t i = 0; i < nin; i++) in[i] = rnd();
    for (size_t i = 0; i < nw; i++) w[i] = rnd();
    for (int i = 0; i < 64; i++) bias[i] = 0.01f * i;

    int64_t ish[] = {batch, in_ch, h, wi}, wsh[] = {out_ch, in_ch, kh, kw}, bsh[] = {64};
    boat_tensor_t* it = boat_tensor_from_data(ish, 4, BOAT_DTYPE_FLOAT32, in);
    boat_tensor_t* wt = boat_tensor_from_data(wsh, 4, BOAT_DTYPE_FLOAT32, w);
    boat_tensor_t* bt = boat_tensor_from_data(bsh, 1, BOAT_DTYPE_FLOAT32, bias);
    boat_conv_layer_t* conv = boat_conv_layer_create(in_ch, out_ch, kh, 1, pad, 1);
    boat_conv_layer_set_weight(conv, wt);
    boat_conv_layer_set_bias(conv, bt);

    boat_tensor_t* o = boat_conv_layer_forward(conv, it);  // warmup
    boat_tensor_unref(o);
    int reps = 10;
    double t0 = now_sec();
    for (int r = 0; r < reps; r++) {
        boat_tensor_t* oo = boat_conv_layer_forward(conv, it);
        boat_tensor_unref(oo);
    }
    double t1 = now_sec();
    double flops = 2.0 * reps * batch * out_ch * (h * wi) * in_ch * kh * kw;
    printf("conv2d %dx%dx%dx%d -> %d ch 3x3 x%d: %.3f s  (%.1f GFLOP/s, im2col+sgemm)\n",
           batch, in_ch, h, wi, out_ch, reps, t1 - t0, flops / (t1 - t0) / 1e9);

    boat_tensor_unref(it);
    boat_tensor_unref(wt);
    boat_tensor_unref(bt);
    boat_conv_layer_free(conv);
    free(in);
    free(w);
}

static void bench_reduce_op(void) {
    // boat_sum over the last dim of [1024, 2048].
    const int64_t sh[] = {1024, 2048};
    float* a = (float*)malloc(1024 * 2048 * sizeof(float));
    for (size_t i = 0; i < 1024 * 2048; i++) a[i] = rnd();
    boat_tensor_t* t = boat_tensor_from_data(sh, 2, BOAT_DTYPE_FLOAT32, a);
    int64_t dims[] = {1};
    boat_tensor_t* o = boat_sum(t, dims, 1, 0);  // warmup
    boat_tensor_unref(o);
    int reps = 20;
    double t0 = now_sec();
    for (int r = 0; r < reps; r++) {
        o = boat_sum(t, dims, 1, 0);
        boat_tensor_unref(o);
    }
    double t1 = now_sec();
    printf("boat_sum last-dim [1024,2048] x%d: %.3f s  (%.1f GB/s)\n", reps, t1 - t0,
           reps * 1024.0 * 2048 * 4 / 1e9 / (t1 - t0));
    boat_tensor_unref(t);
    free(a);
}

static void bench_one(const char* name, void (*simd)(const float*, float*, size_t),
                      void (*scalar)(const float*, float*, size_t), const float* a,
                      float* buf, size_t n, int reps) {
    double t0, t1;
    volatile float sink = 0.0f;
    simd(a, buf, 64);  // warmup
    t0 = now_sec();
    for (int r = 0; r < reps; r++) simd(a, buf, n);
    t1 = now_sec();
    double ts = t1 - t0;
    t0 = now_sec();
    for (int r = 0; r < reps; r++) scalar(a, buf, n);
    t1 = now_sec();
    double tc = t1 - t0;
    for (size_t i = 0; i < n; i++) sink += buf[i];
    (void)sink;
    printf("%-8s %6zu elems x%-3d: SIMD %.3f s  scalar %.3f s  speedup %.1fx  (%.1f MB/s SIMD)\n",
           name, n, reps, ts, tc, tc / ts, reps * n * 4 / 1e6 / ts);
}

static void s_sigmoid(const float* a, float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = 1.0f / (1.0f + expf(-a[i]));
}
static void s_tanh(const float* a, float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = tanhf(a[i]);
}
static void s_silu(const float* a, float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = a[i] / (1.0f + expf(-a[i]));
}
static void s_gelu(const float* a, float* d, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float x = a[i];
        d[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    }
}
static void s_exp(const float* a, float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = expf(a[i]);
}

// Backward (derivative) kernels: scalar references. Marked no-vectorize so the
// comparison is against genuinely scalar code (these have no libm calls, so the
// compiler would otherwise auto-vectorize them).
#if defined(__clang__)
#define BOAT_SCALAR_NO_VEC _Pragma("clang loop vectorize(disable)")
#else
#define BOAT_SCALAR_NO_VEC
#endif
#if defined(__GNUC__) && !defined(__clang__)
#define BOAT_SCALAR_NO_VEC_FUNC __attribute__((optimize("no-tree-vectorize")))
#else
#define BOAT_SCALAR_NO_VEC_FUNC
#endif

static void BOAT_SCALAR_NO_VEC_FUNC s_sigmoid_bw(const float* dy, const float* y, float* d,
                                                 size_t n) {
    BOAT_SCALAR_NO_VEC
    for (size_t i = 0; i < n; i++) d[i] = dy[i] * (y[i] * (1.0f - y[i]));
}
static void BOAT_SCALAR_NO_VEC_FUNC s_tanh_bw(const float* dy, const float* y, float* d,
                                              size_t n) {
    BOAT_SCALAR_NO_VEC
    for (size_t i = 0; i < n; i++) d[i] = dy[i] * (1.0f - y[i] * y[i]);
}
static void BOAT_SCALAR_NO_VEC_FUNC s_relu_bw(const float* dy, const float* x, float* d,
                                              size_t n) {
    BOAT_SCALAR_NO_VEC
    for (size_t i = 0; i < n; i++) d[i] = x[i] > 0.0f ? dy[i] : 0.0f;
}
static void BOAT_SCALAR_NO_VEC_FUNC s_gelu_bw(const float* dy, const float* x, float* d,
                                              size_t n) {
    BOAT_SCALAR_NO_VEC
    for (size_t i = 0; i < n; i++) {
        float xv = x[i];
        float a = 0.7978845608028654f * (xv + 0.044715f * xv * xv * xv);
        float t = tanhf(a);
        d[i] = dy[i] * (0.5f * (1.0f + t) +
                        0.5f * xv * (1.0f - t * t) * 0.7978845608028654f *
                            (1.0f + 3.0f * 0.044715f * xv * xv));
    }
}

static void bench_backward_kernels(void) {
    const size_t N = 1u << 20;
    float* x = (float*)malloc(N * sizeof(float));
    float* y = (float*)malloc(N * sizeof(float));
    float* dy = (float*)malloc(N * sizeof(float));
    float* buf = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        x[i] = rnd();
        y[i] = rnd();
        dy[i] = rnd();
    }
    double t0, t1;
    volatile float sink = 0.0f;
    boat_simd_sigmoid_backward_f32(dy, y, buf, 64);
    t0 = now_sec();
    for (int r = 0; r < 10; r++) boat_simd_sigmoid_backward_f32(dy, y, buf, N);
    t1 = now_sec();
    double ts = t1 - t0;
    t0 = now_sec();
    for (int r = 0; r < 10; r++) s_sigmoid_bw(dy, y, buf, N);
    t1 = now_sec();
    printf("sigmoid_bw %zu elems x10: SIMD %.3f s  scalar %.3f s  speedup %.1fx\n", N, ts,
           t1 - t0, (t1 - t0) / ts);

    boat_simd_tanh_backward_f32(dy, y, buf, 64);
    t0 = now_sec();
    for (int r = 0; r < 10; r++) boat_simd_tanh_backward_f32(dy, y, buf, N);
    t1 = now_sec();
    ts = t1 - t0;
    t0 = now_sec();
    for (int r = 0; r < 10; r++) s_tanh_bw(dy, y, buf, N);
    t1 = now_sec();
    printf("tanh_bw    %zu elems x10: SIMD %.3f s  scalar %.3f s  speedup %.1fx\n", N, ts,
           t1 - t0, (t1 - t0) / ts);

    boat_simd_relu_backward_f32(dy, x, buf, 64);
    t0 = now_sec();
    for (int r = 0; r < 10; r++) boat_simd_relu_backward_f32(dy, x, buf, N);
    t1 = now_sec();
    ts = t1 - t0;
    t0 = now_sec();
    for (int r = 0; r < 10; r++) s_relu_bw(dy, x, buf, N);
    t1 = now_sec();
    printf("relu_bw    %zu elems x10: SIMD %.3f s  scalar %.3f s  speedup %.1fx\n", N, ts,
           t1 - t0, (t1 - t0) / ts);

    boat_simd_gelu_backward_f32(dy, x, buf, 64);
    t0 = now_sec();
    for (int r = 0; r < 10; r++) boat_simd_gelu_backward_f32(dy, x, buf, N);
    t1 = now_sec();
    ts = t1 - t0;
    t0 = now_sec();
    for (int r = 0; r < 10; r++) s_gelu_bw(dy, x, buf, N);
    t1 = now_sec();
    printf("gelu_bw    %zu elems x10: SIMD %.3f s  scalar %.3f s  speedup %.1fx\n", N, ts,
           t1 - t0, (t1 - t0) / ts);

    // Row-wise softmax backward over [256, 4096].
    const size_t rows = 256, cols = 4096;
    for (size_t i = 0; i < rows * cols; i++) {
        y[i] = rnd() * 0.1f + 0.01f;
        dy[i] = rnd();
    }
    t0 = now_sec();
    for (int r = 0; r < 5; r++) {
        boat_simd_softmax_backward_f32(dy, y, buf, rows, cols);
    }
    t1 = now_sec();
    printf("softmax_bw %zux%zu x5: %.3f s  (%.1f MB/s)\n", rows, cols, t1 - t0,
           5.0 * rows * cols * 4 / 1e6 / (t1 - t0));

    for (size_t i = 0; i < N; i++) x[i] = rnd();
    int64_t* labels = (int64_t*)malloc(rows * sizeof(int64_t));
    for (size_t r = 0; r < rows; r++) labels[r] = (int64_t)(r % cols);
    t0 = now_sec();
    for (int r = 0; r < 5; r++) {
        boat_simd_softmax_ce_backward_f32(x, labels, buf, rows, cols, 0.01f);
    }
    t1 = now_sec();
    free(labels);
    (void)sink;
    printf("softmax_ce_bw %zux%zu x5: %.3f s  (%.1f MB/s)\n", rows, cols, t1 - t0,
           5.0 * rows * cols * 4 / 1e6 / (t1 - t0));
    free(x);
    free(y);
    free(dy);
    free(buf);
}

static void bench_activations(void) {
    const size_t N = 1u << 20;  // 1M elems
    float* a = (float*)malloc(N * sizeof(float));
    float* buf = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) a[i] = rnd();
    bench_one("sigmoid", boat_simd_sigmoid_f32, s_sigmoid, a, buf, N, 10);
    bench_one("tanh", boat_simd_tanh_f32, s_tanh, a, buf, N, 10);
    bench_one("silu", boat_simd_silu_f32, s_silu, a, buf, N, 10);
    bench_one("gelu", boat_simd_gelu_f32, s_gelu, a, buf, N, 10);
    bench_one("exp", boat_simd_exp_f32, s_exp, a, buf, N, 10);

    // Row-wise softmax over [256, 4096].
    const size_t rows = 256, cols = 4096;
    for (size_t i = 0; i < rows * cols; i++) a[i] = rnd();
    double t0 = now_sec();
    for (int r = 0; r < 5; r++) boat_simd_softmax_f32(a, buf, rows, cols);
    double t1 = now_sec();
    printf("softmax  %zux%zu x5: %.3f s  (%.1f MB/s)\n", rows, cols, t1 - t0,
           5.0 * rows * cols * 4 / 1e6 / (t1 - t0));
    free(a);
    free(buf);
}

int main(void) {
    printf("=== Boat SIMD / conv / reduce benchmark ===\n");
#ifdef _OPENMP
    printf("OpenMP: ON (%d threads)\n", omp_get_max_threads());
#else
    printf("OpenMP: OFF\n");
#endif
    bench_reduce_kernels();
    bench_transpose();
    bench_conv();
    bench_reduce_op();
    bench_activations();
    bench_backward_kernels();
    printf("done.\n");
    return 0;
}
