// test_cuda_layers.c - CUDA layer kernel integration tests
#include <boat/tensor.h>
#include <boat/memory.h>
#include <boat/cuda_runtime.h>
#include <boat/ops.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// CPU reference for dense forward: C = A @ B + bias
static void cpu_dense_forward(const float* A, const float* B, const float* bias,
                               float* C, size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum + (bias ? bias[j] : 0.0f);
        }
    }
}

// CPU reference for conv2d forward (direct)
static void cpu_conv2d_forward(const float* input, const float* weight,
                                const float* bias, float* output,
                                size_t N, size_t C, size_t H, size_t W,
                                size_t OC, size_t KH, size_t KW,
                                size_t pad, size_t stride, size_t groups) {
    size_t CG = C / groups;
    size_t OCG = OC / groups;
    size_t OH = (H + 2*pad - KH) / stride + 1;
    size_t OW = (W + 2*pad - KW) / stride + 1;
    memset(output, 0, N * OC * OH * OW * sizeof(float));

    for (size_t n = 0; n < N; n++) {
        for (size_t g = 0; g < groups; g++) {
            for (size_t oc = 0; oc < OCG; oc++) {
                size_t out_c = g * OCG + oc;
                for (size_t oh = 0; oh < OH; oh++) {
                    for (size_t ow = 0; ow < OW; ow++) {
                        float sum = 0.0f;
                        for (size_t ic = 0; ic < CG; ic++) {
                            size_t in_c = g * CG + ic;
                            for (size_t kh = 0; kh < KH; kh++) {
                                for (size_t kw = 0; kw < KW; kw++) {
                                    int ih = (int)(oh * stride - pad + kh);
                                    int iw = (int)(ow * stride - pad + kw);
                                    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                                        sum += input[((n * C + in_c) * H + ih) * W + iw]
                                             * weight[((out_c * CG + ic) * KH + kh) * KW + kw];
                                    }
                                }
                            }
                        }
                        size_t out_idx = ((n * OC + out_c) * OH + oh) * OW + ow;
                        output[out_idx] = sum + (bias ? bias[out_c] : 0.0f);
                    }
                }
            }
        }
    }
}

// CPU reference for batch norm
static void cpu_batchnorm_forward(const float* input, float* output,
                                   const float* gamma, const float* beta,
                                   float* mean, float* var,
                                   size_t N, size_t C, size_t H, size_t W, float eps) {
    size_t spatial = N * H * W;
    for (size_t c = 0; c < C; c++) {
        double sum = 0.0, sum_sq = 0.0;
        for (size_t i = 0; i < spatial; i++) {
            float v = input[c * spatial + i];
            sum += v;
            sum_sq += v * v;
        }
        float mu = (float)(sum / spatial);
        float variance = (float)(sum_sq / spatial - mu * mu);
        if (variance < 0) variance = 0;
        mean[c] = mu;
        var[c] = variance;
        float inv_std = 1.0f / sqrtf(variance + eps);
        for (size_t i = 0; i < spatial; i++) {
            size_t idx = c * spatial + i;
            output[idx] = gamma[c] * (input[idx] - mu) * inv_std + beta[c];
        }
    }
}

static int check_error(const float* cpu, const float* gpu, size_t n, float tol, const char* name) {
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(cpu[i] - gpu[i]);
        float max_v = fabsf(cpu[i]) > fabsf(gpu[i]) ? fabsf(cpu[i]) : fabsf(gpu[i]);
        if (diff > tol + 1e-5f * max_v) {
            fprintf(stderr, "  %s MISMATCH at [%zu]: cpu=%f gpu=%f diff=%f\n", name, i, cpu[i], gpu[i], diff);
            return 1;
        }
    }
    return 0;
}

static float* make_host(size_t n) {
    return (float*)malloc(n * sizeof(float));
}

static void free_host(float* p) {
    free(p);
}

int main() {
    int errors = 0;
    printf("=== CUDA Layer Kernel Tests ===\n");
    fflush(stdout);

    int ndev = boat_cuda_device_count();
    if (ndev <= 0) {
        printf("No CUDA devices found -- skipping all tests.\n");
        return 0;
    }
    printf("Device count: %d\n\n", ndev);

    // =========================================================================
    // Test 1: cuBLAS matmul vs CPU SGEMM
    // =========================================================================
    printf("[Test 1] cuBLAS matmul f32...\n");
    {
        size_t M = 8, K = 16, N = 12;
        float* h_A = make_host(M * K);
        float* h_B = make_host(K * N);
        float* h_C_cpu = make_host(M * N);
        float* h_C_gpu = make_host(M * N);
        for (size_t i = 0; i < M*K; i++) h_A[i] = (float)(i % 5) * 0.1f;
        for (size_t i = 0; i < K*N; i++) h_B[i] = (float)((i+3) % 7) * 0.1f;

        float* d_A = (float*)boat_cuda_malloc(M*K*sizeof(float));
        float* d_B = (float*)boat_cuda_malloc(K*N*sizeof(float));
        float* d_C = (float*)boat_cuda_malloc(M*N*sizeof(float));
        boat_cuda_memcpy_h2d(d_A, h_A, M*K*sizeof(float));
        boat_cuda_memcpy_h2d(d_B, h_B, K*N*sizeof(float));

        cpu_dense_forward(h_A, h_B, NULL, h_C_cpu, M, K, N);
        boat_cuda_matmul_f32_cublas(d_A, d_B, d_C, M, N, K);
        boat_cuda_memcpy_d2h(h_C_gpu, d_C, M*N*sizeof(float));

        errors += check_error(h_C_cpu, h_C_gpu, M*N, 1e-4f, "Test 1");
        boat_cuda_free(d_A); boat_cuda_free(d_B); boat_cuda_free(d_C);
        free_host(h_A); free_host(h_B); free_host(h_C_cpu); free_host(h_C_gpu);
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");

    // =========================================================================
    // Test 2: cuBLAS strided batched matmul
    // =========================================================================
    printf("[Test 2] cuBLAS strided batched matmul...\n");
    {
        size_t batch = 3, M = 4, K = 8, N = 6;
        size_t total_A = batch * M * K;
        size_t total_B = batch * K * N;
        size_t total_C = batch * M * N;
        float* h_A = make_host(total_A);
        float* h_B = make_host(total_B);
        float* h_C_cpu = make_host(total_C);
        float* h_C_gpu = make_host(total_C);
        for (size_t i = 0; i < total_A; i++) h_A[i] = (float)(i % 5) * 0.1f;
        for (size_t i = 0; i < total_B; i++) h_B[i] = (float)((i+2) % 7) * 0.1f;
        float* d_A = (float*)boat_cuda_malloc(total_A * sizeof(float));
        float* d_B = (float*)boat_cuda_malloc(total_B * sizeof(float));
        float* d_C = (float*)boat_cuda_malloc(total_C * sizeof(float));
        boat_cuda_memcpy_h2d(d_A, h_A, total_A * sizeof(float));
        boat_cuda_memcpy_h2d(d_B, h_B, total_B * sizeof(float));
        for (size_t b = 0; b < batch; b++)
            cpu_dense_forward(h_A + b*M*K, h_B + b*K*N, NULL, h_C_cpu + b*M*N, M, K, N);
        boat_cuda_matmul_f32_strided_batched(d_A, d_B, d_C, M, N, K,
            batch, M*K, K*N, M*N);
        boat_cuda_memcpy_d2h(h_C_gpu, d_C, total_C * sizeof(float));
        errors += check_error(h_C_cpu, h_C_gpu, total_C, 1e-4f, "Test 2");
        boat_cuda_free(d_A); boat_cuda_free(d_B); boat_cuda_free(d_C);
        free_host(h_A); free_host(h_B); free_host(h_C_cpu); free_host(h_C_gpu);
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    fflush(stdout);

    // =========================================================================
    // Test 3: boat_matmul dispatch on CUDA tensors
    // =========================================================================
    printf("[Test 3] boat_matmul dispatch on CUDA tensors...\n");
    {
        size_t M = 8, K = 16, N = 12;
        float* h_A = make_host(M * K);
        float* h_B = make_host(K * N);
        float* h_C_cpu = make_host(M * N);
        float* h_C_gpu = make_host(M * N);
        for (size_t i = 0; i < M*K; i++) h_A[i] = (float)(i % 5) * 0.1f;
        for (size_t i = 0; i < K*N; i++) h_B[i] = (float)((i+3) % 7) * 0.1f;

        int64_t shape_A[] = {(int64_t)M, (int64_t)K};
        int64_t shape_B[] = {(int64_t)K, (int64_t)N};
        boat_tensor_t* tA = boat_tensor_from_data(shape_A, 2, BOAT_DTYPE_FLOAT32, h_A);
        boat_tensor_t* tB = boat_tensor_from_data(shape_B, 2, BOAT_DTYPE_FLOAT32, h_B);
        boat_tensor_t* tA_cuda = boat_tensor_to_device(tA, BOAT_DEVICE_CUDA);
        boat_tensor_t* tB_cuda = boat_tensor_to_device(tB, BOAT_DEVICE_CUDA);

        cpu_dense_forward(h_A, h_B, NULL, h_C_cpu, M, K, N);

        boat_tensor_t* tC_cuda = boat_matmul(tA_cuda, tB_cuda);
        if (!tC_cuda) { printf("  ERROR: boat_matmul returned NULL\n"); errors++; }
        else {
            boat_tensor_t* tC_cpu = boat_tensor_to_device(tC_cuda, BOAT_DEVICE_CPU);
            memcpy(h_C_gpu, boat_tensor_data(tC_cpu), M*N*sizeof(float));
            errors += check_error(h_C_cpu, h_C_gpu, M*N, 1e-4f, "Test 3");
            boat_tensor_unref(tC_cpu);
            boat_tensor_unref(tC_cuda);
        }
        boat_tensor_unref(tA); boat_tensor_unref(tB);
        boat_tensor_unref(tA_cuda); boat_tensor_unref(tB_cuda);
        free_host(h_A); free_host(h_B); free_host(h_C_cpu); free_host(h_C_gpu);
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    fflush(stdout);

    // =========================================================================
    // Test 4: Dense forward via boat_cuda_dense_forward_f32
    // =========================================================================
    printf("[Test 4] Dense forward CUDA...\n");
    {
        size_t B = 8, I = 16, O = 12;
        float* h_in = make_host(B * I);
        float* h_w = make_host(I * O);
        float* h_bias = make_host(O);
        float* h_out_cpu = make_host(B * O);
        float* h_out_gpu = make_host(B * O);
        for (size_t i = 0; i < B*I; i++) h_in[i] = (float)(i % 7) * 0.1f;
        for (size_t i = 0; i < I*O; i++) h_w[i] = (float)((i+2) % 5) * 0.1f;
        for (size_t i = 0; i < O; i++) h_bias[i] = (float)(i % 3) * 0.2f;

        cpu_dense_forward(h_in, h_w, h_bias, h_out_cpu, B, I, O);

        float* d_in = (float*)boat_cuda_malloc(B * I * sizeof(float));
        float* d_w = (float*)boat_cuda_malloc(I * O * sizeof(float));
        float* d_bias = (float*)boat_cuda_malloc(O * sizeof(float));
        float* d_out = (float*)boat_cuda_malloc(B * O * sizeof(float));
        boat_cuda_memcpy_h2d(d_in, h_in, B * I * sizeof(float));
        boat_cuda_memcpy_h2d(d_w, h_w, I * O * sizeof(float));
        boat_cuda_memcpy_h2d(d_bias, h_bias, O * sizeof(float));

        boat_cuda_dense_forward_f32(d_in, d_w, d_bias, d_out, B, I, O);
        boat_cuda_memcpy_d2h(h_out_gpu, d_out, B * O * sizeof(float));

        errors += check_error(h_out_cpu, h_out_gpu, B * O, 1e-4f, "Test 4");
        boat_cuda_free(d_in); boat_cuda_free(d_w); boat_cuda_free(d_bias); boat_cuda_free(d_out);
        free_host(h_in); free_host(h_w); free_host(h_bias); free_host(h_out_cpu); free_host(h_out_gpu);
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    fflush(stdout);

    // =========================================================================
    // Test 5: Dense forward via warp-level kernel
    // =========================================================================
    printf("[Test 5] Dense forward warp-level kernel...\n");
    {
        size_t B = 8, I = 32, O = 16;
        float* h_in = make_host(B * I);
        float* h_w = make_host(I * O);
        float* h_bias = make_host(O);
        float* h_out_cpu = make_host(B * O);
        float* h_out_gpu = make_host(B * O);
        for (size_t i = 0; i < B*I; i++) h_in[i] = (float)(i % 7) * 0.1f;
        for (size_t i = 0; i < I*O; i++) h_w[i] = (float)((i+2) % 5) * 0.1f;
        for (size_t i = 0; i < O; i++) h_bias[i] = (float)(i % 3) * 0.2f;

        cpu_dense_forward(h_in, h_w, h_bias, h_out_cpu, B, I, O);

        float* d_in = (float*)boat_cuda_malloc(B * I * sizeof(float));
        float* d_w = (float*)boat_cuda_malloc(I * O * sizeof(float));
        float* d_bias = (float*)boat_cuda_malloc(O * sizeof(float));
        float* d_out = (float*)boat_cuda_malloc(B * O * sizeof(float));
        boat_cuda_memcpy_h2d(d_in, h_in, B * I * sizeof(float));
        boat_cuda_memcpy_h2d(d_w, h_w, I * O * sizeof(float));
        boat_cuda_memcpy_h2d(d_bias, h_bias, O * sizeof(float));

        boat_cuda_dense_forward_warp_f32(d_in, d_w, d_bias, d_out, B, I, O);
        boat_cuda_memcpy_d2h(h_out_gpu, d_out, B * O * sizeof(float));

        errors += check_error(h_out_cpu, h_out_gpu, B * O, 1e-4f, "Test 5");
        boat_cuda_free(d_in); boat_cuda_free(d_w); boat_cuda_free(d_bias); boat_cuda_free(d_out);
        free_host(h_in); free_host(h_w); free_host(h_bias); free_host(h_out_cpu); free_host(h_out_gpu);
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    fflush(stdout);

    // =========================================================================
    // Test 6: Conv2D forward
    // =========================================================================
    printf("[Test 6] Conv2D forward CUDA...\n");
    {
        size_t N=1, C=2, H=4, W=4, OC=2, KH=3, KW=3, pad=0, stride=1, groups=1;
        size_t OH = (H + 2*pad - KH) / stride + 1;
        size_t OW = (W + 2*pad - KW) / stride + 1;
        float* h_in = make_host(N * C * H * W);
        float* h_w = make_host(OC * C * KH * KW);
        float* h_bias = make_host(OC);
        float* h_out_cpu = make_host(N * OC * OH * OW);
        float* h_out_gpu = make_host(N * OC * OH * OW);
        for (size_t i = 0; i < N*C*H*W; i++) h_in[i] = (float)(i % 7) * 0.1f;
        for (size_t i = 0; i < OC*C*KH*KW; i++) h_w[i] = (float)((i+3) % 5) * 0.1f;
        for (size_t i = 0; i < OC; i++) h_bias[i] = (float)(i) * 0.1f;

        cpu_conv2d_forward(h_in, h_w, h_bias, h_out_cpu, N, C, H, W, OC, KH, KW, pad, stride, groups);

        float* d_in = (float*)boat_cuda_malloc(N * C * H * W * sizeof(float));
        float* d_w = (float*)boat_cuda_malloc(OC * C * KH * KW * sizeof(float));
        float* d_bias = (float*)boat_cuda_malloc(OC * sizeof(float));
        float* d_out = (float*)boat_cuda_malloc(N * OC * OH * OW * sizeof(float));
        boat_cuda_memcpy_h2d(d_in, h_in, N * C * H * W * sizeof(float));
        boat_cuda_memcpy_h2d(d_w, h_w, OC * C * KH * KW * sizeof(float));
        boat_cuda_memcpy_h2d(d_bias, h_bias, OC * sizeof(float));

        boat_cuda_conv2d_forward_f32(d_in, d_w, d_bias, d_out, N, C, H, W, OC, KH, KW, pad, stride, groups);
        boat_cuda_memcpy_d2h(h_out_gpu, d_out, N * OC * OH * OW * sizeof(float));

        errors += check_error(h_out_cpu, h_out_gpu, N*OC*OH*OW, 1e-4f, "Test 6");
        boat_cuda_free(d_in); boat_cuda_free(d_w); boat_cuda_free(d_bias); boat_cuda_free(d_out);
        free_host(h_in); free_host(h_w); free_host(h_bias); free_host(h_out_cpu); free_host(h_out_gpu);
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    fflush(stdout);

    // =========================================================================
    // Test 7: Group conv2d
    // =========================================================================
    printf("[Test 7] Group Conv2D (groups=2)...\n");
    {
        size_t N=1, C=4, H=4, W=4, OC=4, KH=3, KW=3, pad=0, stride=1, groups=2;
        size_t OH = (H + 2*pad - KH) / stride + 1;
        size_t OW = (W + 2*pad - KW) / stride + 1;
        float* h_in = make_host(N * C * H * W);
        float* h_w = make_host(OC * (C/groups) * KH * KW);
        float* h_bias = make_host(OC);
        float* h_out_cpu = make_host(N * OC * OH * OW);
        float* h_out_gpu = make_host(N * OC * OH * OW);
        for (size_t i = 0; i < N*C*H*W; i++) h_in[i] = (float)(i % 7) * 0.1f;
        for (size_t i = 0; i < OC*(C/groups)*KH*KW; i++) h_w[i] = (float)((i+3) % 5) * 0.1f;
        for (size_t i = 0; i < OC; i++) h_bias[i] = (float)(i) * 0.1f;

        cpu_conv2d_forward(h_in, h_w, h_bias, h_out_cpu, N, C, H, W, OC, KH, KW, pad, stride, groups);

        float* d_in = (float*)boat_cuda_malloc(N * C * H * W * sizeof(float));
        float* d_w = (float*)boat_cuda_malloc(OC * (C/groups) * KH * KW * sizeof(float));
        float* d_bias = (float*)boat_cuda_malloc(OC * sizeof(float));
        float* d_out = (float*)boat_cuda_malloc(N * OC * OH * OW * sizeof(float));
        boat_cuda_memcpy_h2d(d_in, h_in, N * C * H * W * sizeof(float));
        boat_cuda_memcpy_h2d(d_w, h_w, OC * (C/groups) * KH * KW * sizeof(float));
        boat_cuda_memcpy_h2d(d_bias, h_bias, OC * sizeof(float));

        boat_cuda_conv2d_forward_f32(d_in, d_w, d_bias, d_out, N, C, H, W, OC, KH, KW, pad, stride, groups);
        boat_cuda_memcpy_d2h(h_out_gpu, d_out, N * OC * OH * OW * sizeof(float));

        errors += check_error(h_out_cpu, h_out_gpu, N*OC*OH*OW, 1e-4f, "Test 7");
        boat_cuda_free(d_in); boat_cuda_free(d_w); boat_cuda_free(d_bias); boat_cuda_free(d_out);
        free_host(h_in); free_host(h_w); free_host(h_bias); free_host(h_out_cpu); free_host(h_out_gpu);
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    fflush(stdout);

    // =========================================================================
    // Test 8: Batch norm forward
    // =========================================================================
    printf("[Test 8] Batchnorm forward CUDA...\n");
    {
        size_t N=2, C=3, H=4, W=4;
        float eps = 1e-5f;
        float* h_in = make_host(N * C * H * W);
        float* h_gamma = make_host(C);
        float* h_beta = make_host(C);
        float* h_out_cpu = make_host(N * C * H * W);
        float* h_out_gpu = make_host(N * C * H * W);
        float* h_mean_cpu = make_host(C);
        float* h_var_cpu = make_host(C);
        float* h_mean_gpu = make_host(C);
        float* h_var_gpu = make_host(C);
        for (size_t i = 0; i < N*C*H*W; i++) h_in[i] = (float)(i % 11) * 0.1f;
        for (size_t i = 0; i < C; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

        cpu_batchnorm_forward(h_in, h_out_cpu, h_gamma, h_beta, h_mean_cpu, h_var_cpu, N, C, H, W, eps);

        float* d_in = (float*)boat_cuda_malloc(N * C * H * W * sizeof(float));
        float* d_out = (float*)boat_cuda_malloc(N * C * H * W * sizeof(float));
        float* d_gamma = (float*)boat_cuda_malloc(C * sizeof(float));
        float* d_beta = (float*)boat_cuda_malloc(C * sizeof(float));
        float* d_mean = (float*)boat_cuda_malloc(C * sizeof(float));
        float* d_var = (float*)boat_cuda_malloc(C * sizeof(float));
        boat_cuda_memcpy_h2d(d_in, h_in, N * C * H * W * sizeof(float));
        boat_cuda_memcpy_h2d(d_gamma, h_gamma, C * sizeof(float));
        boat_cuda_memcpy_h2d(d_beta, h_beta, C * sizeof(float));

        boat_cuda_batchnorm_forward_f32(d_in, d_out, d_gamma, d_beta, d_mean, d_var, N, C, H, W, eps);
        boat_cuda_memcpy_d2h(h_out_gpu, d_out, N * C * H * W * sizeof(float));
        boat_cuda_memcpy_d2h(h_mean_gpu, d_mean, C * sizeof(float));
        boat_cuda_memcpy_d2h(h_var_gpu, d_var, C * sizeof(float));

        errors += check_error(h_out_cpu, h_out_gpu, N*C*H*W, 1e-4f, "Test 8");
        errors += check_error(h_mean_cpu, h_mean_gpu, C, 1e-4f, "Test 8 mean");
        errors += check_error(h_var_cpu, h_var_gpu, C, 1e-4f, "Test 8 var");
        boat_cuda_free(d_in); boat_cuda_free(d_out); boat_cuda_free(d_gamma);
        boat_cuda_free(d_beta); boat_cuda_free(d_mean); boat_cuda_free(d_var);
        free_host(h_in); free_host(h_gamma); free_host(h_beta);
        free_host(h_out_cpu); free_host(h_out_gpu);
        free_host(h_mean_cpu); free_host(h_var_cpu); free_host(h_mean_gpu); free_host(h_var_gpu);
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    fflush(stdout);

    // =========================================================================
    // Test 9: Fused bn+relu
    // =========================================================================
    printf("[Test 9] Fused bn+relu forward CUDA...\n");
    {
        size_t N=2, C=3, H=4, W=4;
        float eps = 1e-5f;
        float* h_in = make_host(N * C * H * W);
        float* h_gamma = make_host(C);
        float* h_beta = make_host(C);
        float* h_mean = make_host(C);
        float* h_var = make_host(C);
        float* h_out_cpu = make_host(N * C * H * W);
        float* h_out_gpu = make_host(N * C * H * W);
        for (size_t i = 0; i < N*C*H*W; i++) h_in[i] = (float)(i % 11) * 0.1f - 0.5f;
        for (size_t i = 0; i < C; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

        // CPU: bn then relu
        cpu_batchnorm_forward(h_in, h_out_cpu, h_gamma, h_beta, h_mean, h_var, N, C, H, W, eps);
        for (size_t i = 0; i < N*C*H*W; i++)
            h_out_cpu[i] = h_out_cpu[i] > 0.0f ? h_out_cpu[i] : 0.0f;

        float* d_in = (float*)boat_cuda_malloc(N * C * H * W * sizeof(float));
        float* d_out = (float*)boat_cuda_malloc(N * C * H * W * sizeof(float));
        float* d_gamma = (float*)boat_cuda_malloc(C * sizeof(float));
        float* d_beta = (float*)boat_cuda_malloc(C * sizeof(float));
        float* d_mean = (float*)boat_cuda_malloc(C * sizeof(float));
        float* d_var = (float*)boat_cuda_malloc(C * sizeof(float));
        boat_cuda_memcpy_h2d(d_in, h_in, N * C * H * W * sizeof(float));
        boat_cuda_memcpy_h2d(d_gamma, h_gamma, C * sizeof(float));
        boat_cuda_memcpy_h2d(d_beta, h_beta, C * sizeof(float));
        boat_cuda_memcpy_h2d(d_mean, h_mean, C * sizeof(float));
        boat_cuda_memcpy_h2d(d_var, h_var, C * sizeof(float));

        boat_cuda_fused_bn_relu_f32(d_in, d_out, d_gamma, d_beta, d_mean, d_var, N, C, H, W, eps);
        boat_cuda_memcpy_d2h(h_out_gpu, d_out, N * C * H * W * sizeof(float));

        errors += check_error(h_out_cpu, h_out_gpu, N*C*H*W, 1e-4f, "Test 9");
        boat_cuda_free(d_in); boat_cuda_free(d_out); boat_cuda_free(d_gamma);
        boat_cuda_free(d_beta); boat_cuda_free(d_mean); boat_cuda_free(d_var);
        free_host(h_in); free_host(h_gamma); free_host(h_beta); free_host(h_mean); free_host(h_var);
        free_host(h_out_cpu); free_host(h_out_gpu);
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    fflush(stdout);

    // =========================================================================
    // Test 10: Depthwise conv2d (groups = C)
    // =========================================================================
    printf("[Test 10] Depthwise Conv2D (groups=C=2)...\n");
    {
        size_t N=1, C=2, H=4, W=4, OC=2, KH=3, KW=3, pad=1, stride=1, groups=2;
        size_t OH = (H + 2*pad - KH) / stride + 1;
        size_t OW = (W + 2*pad - KW) / stride + 1;
        float* h_in = make_host(N * C * H * W);
        float* h_w = make_host(OC * (C/groups) * KH * KW);
        float* h_bias = make_host(OC);
        float* h_out_cpu = make_host(N * OC * OH * OW);
        float* h_out_gpu = make_host(N * OC * OH * OW);
        for (size_t i = 0; i < N*C*H*W; i++) h_in[i] = (float)(i % 7) * 0.1f;
        for (size_t i = 0; i < OC*(C/groups)*KH*KW; i++) h_w[i] = (float)((i+3) % 5) * 0.1f;
        for (size_t i = 0; i < OC; i++) h_bias[i] = (float)(i) * 0.1f;

        cpu_conv2d_forward(h_in, h_w, h_bias, h_out_cpu, N, C, H, W, OC, KH, KW, pad, stride, groups);

        float* d_in = (float*)boat_cuda_malloc(N * C * H * W * sizeof(float));
        float* d_w = (float*)boat_cuda_malloc(OC * (C/groups) * KH * KW * sizeof(float));
        float* d_bias = (float*)boat_cuda_malloc(OC * sizeof(float));
        float* d_out = (float*)boat_cuda_malloc(N * OC * OH * OW * sizeof(float));
        boat_cuda_memcpy_h2d(d_in, h_in, N * C * H * W * sizeof(float));
        boat_cuda_memcpy_h2d(d_w, h_w, OC * (C/groups) * KH * KW * sizeof(float));
        boat_cuda_memcpy_h2d(d_bias, h_bias, OC * sizeof(float));

        boat_cuda_conv2d_forward_f32(d_in, d_w, d_bias, d_out, N, C, H, W, OC, KH, KW, pad, stride, groups);
        boat_cuda_memcpy_d2h(h_out_gpu, d_out, N * OC * OH * OW * sizeof(float));

        errors += check_error(h_out_cpu, h_out_gpu, N*OC*OH*OW, 1e-4f, "Test 10");
        boat_cuda_free(d_in); boat_cuda_free(d_w); boat_cuda_free(d_bias); boat_cuda_free(d_out);
        free_host(h_in); free_host(h_w); free_host(h_bias); free_host(h_out_cpu); free_host(h_out_gpu);
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    fflush(stdout);

#ifdef BOAT_WITH_CUDNN
    // =========================================================================
    // Test 11: cuDNN Conv2D forward
    // =========================================================================
    printf("[Test 11] cuDNN Conv2D forward...\n");
    {
        size_t N=1, C=2, H=4, W=4, OC=2, KH=3, KW=3, pad=0, stride=1, groups=1;
        size_t OH = (H + 2*pad - KH) / stride + 1;
        size_t OW = (W + 2*pad - KW) / stride + 1;
        float* h_in = make_host(N * C * H * W);
        float* h_w = make_host(OC * C * KH * KW);
        float* h_bias = make_host(OC);
        float* h_out_cpu = make_host(N * OC * OH * OW);
        float* h_out_gpu = make_host(N * OC * OH * OW);
        for (size_t i = 0; i < N*C*H*W; i++) h_in[i] = (float)(i % 7) * 0.1f;
        for (size_t i = 0; i < OC*C*KH*KW; i++) h_w[i] = (float)((i+3) % 5) * 0.1f;
        for (size_t i = 0; i < OC; i++) h_bias[i] = (float)(i) * 0.1f;

        cpu_conv2d_forward(h_in, h_w, h_bias, h_out_cpu, N, C, H, W, OC, KH, KW, pad, stride, groups);

        float* d_in = (float*)boat_cuda_malloc(N * C * H * W * sizeof(float));
        float* d_w = (float*)boat_cuda_malloc(OC * C * KH * KW * sizeof(float));
        float* d_bias = (float*)boat_cuda_malloc(OC * sizeof(float));
        float* d_out = (float*)boat_cuda_malloc(N * OC * OH * OW * sizeof(float));
        boat_cuda_memcpy_h2d(d_in, h_in, N * C * H * W * sizeof(float));
        boat_cuda_memcpy_h2d(d_w, h_w, OC * C * KH * KW * sizeof(float));
        boat_cuda_memcpy_h2d(d_bias, h_bias, OC * sizeof(float));

        boat_cuda_conv2d_cudnn_forward_f32(d_in, d_w, d_bias, d_out, N, C, H, W, OC, KH, KW, pad, stride, groups);
        boat_cuda_memcpy_d2h(h_out_gpu, d_out, N * OC * OH * OW * sizeof(float));

        errors += check_error(h_out_cpu, h_out_gpu, N*OC*OH*OW, 1e-4f, "Test 11");
        boat_cuda_free(d_in); boat_cuda_free(d_w); boat_cuda_free(d_bias); boat_cuda_free(d_out);
        free_host(h_in); free_host(h_w); free_host(h_bias); free_host(h_out_cpu); free_host(h_out_gpu);
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    fflush(stdout);

    // =========================================================================
    // Test 12: cuDNN BatchNorm forward
    // =========================================================================
    printf("[Test 12] cuDNN BatchNorm forward...\n");
    {
        size_t N=2, C=3, H=4, W=4;
        float eps = 1e-5f;
        float* h_in = make_host(N * C * H * W);
        float* h_gamma = make_host(C);
        float* h_beta = make_host(C);
        float* h_out_cpu = make_host(N * C * H * W);
        float* h_out_gpu = make_host(N * C * H * W);
        float* h_mean_cpu = make_host(C);
        float* h_var_cpu = make_host(C);
        float* h_mean_gpu = make_host(C);
        float* h_var_gpu = make_host(C);
        for (size_t i = 0; i < N*C*H*W; i++) h_in[i] = (float)(i % 11) * 0.1f;
        for (size_t i = 0; i < C; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

        cpu_batchnorm_forward(h_in, h_out_cpu, h_gamma, h_beta, h_mean_cpu, h_var_cpu, N, C, H, W, eps);

        float* d_in = (float*)boat_cuda_malloc(N * C * H * W * sizeof(float));
        float* d_out = (float*)boat_cuda_malloc(N * C * H * W * sizeof(float));
        float* d_gamma = (float*)boat_cuda_malloc(C * sizeof(float));
        float* d_beta = (float*)boat_cuda_malloc(C * sizeof(float));
        float* d_mean = (float*)boat_cuda_malloc(C * sizeof(float));
        float* d_var = (float*)boat_cuda_malloc(C * sizeof(float));
        boat_cuda_memcpy_h2d(d_in, h_in, N * C * H * W * sizeof(float));
        boat_cuda_memcpy_h2d(d_gamma, h_gamma, C * sizeof(float));
        boat_cuda_memcpy_h2d(d_beta, h_beta, C * sizeof(float));

        boat_cuda_batchnorm_cudnn_forward_f32(d_in, d_out, d_gamma, d_beta, d_mean, d_var, N, C, H, W, eps);
        boat_cuda_memcpy_d2h(h_out_gpu, d_out, N * C * H * W * sizeof(float));
        boat_cuda_memcpy_d2h(h_mean_gpu, d_mean, C * sizeof(float));
        boat_cuda_memcpy_d2h(h_var_gpu, d_var, C * sizeof(float));

        errors += check_error(h_out_cpu, h_out_gpu, N*C*H*W, 1e-4f, "Test 12");
        errors += check_error(h_mean_cpu, h_mean_gpu, C, 1e-3f, "Test 12 mean");
        errors += check_error(h_var_cpu, h_var_gpu, C, 1e-3f, "Test 12 var");
        boat_cuda_free(d_in); boat_cuda_free(d_out); boat_cuda_free(d_gamma);
        boat_cuda_free(d_beta); boat_cuda_free(d_mean); boat_cuda_free(d_var);
        free_host(h_in); free_host(h_gamma); free_host(h_beta);
        free_host(h_out_cpu); free_host(h_out_gpu);
        free_host(h_mean_cpu); free_host(h_var_cpu); free_host(h_mean_gpu); free_host(h_var_gpu);
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    fflush(stdout);
#endif

    boat_cuda_cublas_destroy();
#ifdef BOAT_WITH_CUDNN
    boat_cuda_cudnn_destroy();
#endif
    printf("\n=== %s ===\n", errors == 0 ? "ALL TESTS PASSED" : "FAILED");
    return errors == 0 ? 0 : 1;
}
