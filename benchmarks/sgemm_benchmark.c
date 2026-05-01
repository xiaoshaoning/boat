// sgemm_benchmark.c - SGEMM micro-kernel performance benchmark
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/sgemm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if defined(_WIN32)
    #include <windows.h>
    static double get_time_ms(void) {
        LARGE_INTEGER freq, count;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&count);
        return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
    }
#else
    #include <sys/time.h>
    static double get_time_ms(void) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
    }
#endif

// Naive reference matmul for verification
static void naive_matmul(int64_t M, int64_t N, int64_t K,
                          const float* A, const float* B, float* C)
{
    memset(C, 0, (size_t)(M * N) * sizeof(float));
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t l = 0; l < K; l++) {
                sum += A[i * K + l] * B[l * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static int verify_result(int64_t M, int64_t N,
                          const float* got, const float* expected)
{
    for (int64_t i = 0; i < M * N; i++) {
        float diff = fabsf(got[i] - expected[i]);
        float max_ab = fmaxf(fabsf(got[i]), fabsf(expected[i]));
        if (!(diff <= 1e-5f + 1e-5f * max_ab)) {
            printf("    MISMATCH at [%lld]: got %f, expected %f\n",
                   (long long)i, got[i], expected[i]);
            return 1;
        }
    }
    return 0;
}

static double benchmark_sgemm(int64_t M, int64_t N, int64_t K,
                               int iterations, int warmup)
{
    float* A = (float*)malloc((size_t)(M * K) * sizeof(float));
    float* B = (float*)malloc((size_t)(K * N) * sizeof(float));
    float* C = (float*)malloc((size_t)(M * N) * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int64_t i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        memset(C, 0, (size_t)(M * N) * sizeof(float));
        boat_sgemm(M, N, K, A, B, C);
    }

    // Benchmark
    double total_ms = 0.0;
    for (int i = 0; i < iterations; i++) {
        memset(C, 0, (size_t)(M * N) * sizeof(float));
        double t0 = get_time_ms();
        boat_sgemm(M, N, K, A, B, C);
        double t1 = get_time_ms();
        total_ms += (t1 - t0);
    }

    double avg_ms = total_ms / iterations;
    double gflops = (double)(2 * M * N * K) / (avg_ms * 1e6);

    free(A); free(B); free(C);
    return gflops;
}

int main(void)
{
    printf("SGEMM Benchmark (GFLOP/s)\n");
    printf("==========================\n");
    printf("%8s %8s %8s  %12s  %10s\n",
           "M", "N", "K", "GFLOP/s", "Time(ms)");
    printf("------------------------------------------------\n");

    int sizes[] = {16, 32, 64, 128, 256, 512, 1024};
    int num_sizes = 7;

    // --- Verification: compare with naive matmul for small sizes ---
    printf("\nVerifying correctness...\n");
    int verified = 1;
    for (int si = 0; si < num_sizes && sizes[si] <= 128; si++) {
        int64_t n = sizes[si];
        float* A = (float*)malloc((size_t)(n * n) * sizeof(float));
        float* B = (float*)malloc((size_t)(n * n) * sizeof(float));
        float* C_sgemm = (float*)calloc((size_t)(n * n), sizeof(float));
        float* C_naive = (float*)calloc((size_t)(n * n), sizeof(float));

        srand(42);
        for (int64_t i = 0; i < n * n; i++) A[i] = (float)rand() / RAND_MAX;
        for (int64_t i = 0; i < n * n; i++) B[i] = (float)rand() / RAND_MAX;

        boat_sgemm(n, n, n, A, B, C_sgemm);
        naive_matmul(n, n, n, A, B, C_naive);

        if (verify_result(n, n, C_sgemm, C_naive)) {
            printf("  FAILED at N=%lld\n", (long long)n);
            verified = 0;
        } else {
            printf("  N=%4lld: PASS\n", (long long)n);
        }

        free(A); free(B); free(C_sgemm); free(C_naive);
    }

    if (!verified) {
        printf("\nVerification FAILED — benchmark results may be invalid.\n");
    }

    // --- Performance benchmark ---
    printf("\nPerformance results:\n");
    for (int si = 0; si < num_sizes; si++) {
        int64_t n = sizes[si];
        int iters = (n <= 64) ? 1000 : (n <= 256) ? 200 : (n <= 512) ? 50 : 20;
        int warmup = 5;

        double gflops = benchmark_sgemm(n, n, n, iters, warmup);

        // Time from GFLOPs: time(ms) = 2*n^3 / (gflops * 1e6)
        double ops = (double)(2 * n * n * n);
        double time_ms = ops / (gflops * 1e6);

        printf("%8lld %8lld %8lld  %12.2f  %10.2f\n",
               (long long)n, (long long)n, (long long)n, gflops, time_ms);
    }

    // --- Batched benchmark ---
    printf("\nBatched SGEMM:\n");
    int64_t batch_sizes[] = {1, 4, 8, 16, 32};
    int num_batch = 5;
    int64_t M = 256, N = 256, K = 256;
    for (int bi = 0; bi < num_batch; bi++) {
        int64_t b = batch_sizes[bi];
        int iters = 100;

        float* A = (float*)malloc((size_t)(b * M * K) * sizeof(float));
        float* B = (float*)malloc((size_t)(b * K * N) * sizeof(float));
        float* C = (float*)malloc((size_t)(b * M * N) * sizeof(float));
        srand(42);
        for (int64_t i = 0; i < b * M * K; i++) A[i] = (float)rand() / RAND_MAX;
        for (int64_t i = 0; i < b * K * N; i++) B[i] = (float)rand() / RAND_MAX;

        double total_ms = 0.0;
        for (int i = 0; i < iters; i++) {
            memset(C, 0, (size_t)(b * M * N) * sizeof(float));
            double t0 = get_time_ms();
            for (int64_t j = 0; j < b; j++) {
                boat_sgemm(M, N, K, A + j * M * K, B + j * K * N, C + j * M * N);
            }
            double t1 = get_time_ms();
            total_ms += (t1 - t0);
        }
        double avg_ms = total_ms / iters;
        double gflops = (double)(2 * b * M * N * K) / (avg_ms * 1e6);
        printf("  batch=%2lld  M=N=K=256  %7.2f GFLOP/s  %8.2f ms\n",
               (long long)b, gflops, avg_ms);

        free(A); free(B); free(C);
    }

    printf("\nDone.\n");
    return verified ? 0 : 1;
}
