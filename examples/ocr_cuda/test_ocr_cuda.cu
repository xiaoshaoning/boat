// test_ocr_cuda.cu - Unit tests for CUDA-accelerated OCR operations
// Tests each custom CUDA kernel against a CPU reference implementation.
//
// Build: part of ocr_cuda example (see CMakeLists.txt)
// Run:   ./test_ocr_cuda

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <boat/tensor.h>
#include <boat/cuda_runtime.h>

#include "ocr_cuda_kernels.cuh"
#include "cogvit_cuda.cuh"
#include "glm_cuda.cuh"

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "  [FAIL] CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                        \
    }                                                                   \
} while(0)

extern "C" cublasHandle_t boat_cuda_get_cublas_handle(void);

static int total_tests = 0, passed_tests = 0;

#define TEST(name) do { fprintf(stderr, "  TEST %s ... ", name); total_tests++; } while(0)
#define PASS() do { fprintf(stderr, "PASS\n"); passed_tests++; } while(0)
#define FAIL(msg) do { fprintf(stderr, "FAIL: %s\n", msg); return 1; } while(0)

static float rand_f32() {
    return ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
}

static int compare_arrays(const float* cpu, const float* gpu, int n,
                           float rtol, float atol) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(cpu[i] - gpu[i]);
        float denom = fmaxf(fabsf(cpu[i]), fabsf(gpu[i]));
        if (denom < 1e-8f) denom = 1.0f;
        if (diff > atol && diff / denom > rtol)
            return i; // index of first mismatch
    }
    return -1;
}

// ---------------------------------------------------------------------------
// CPU reference: matmul_bt (transposed matmul: C[M,N] = A[M,K] @ W[N,K]^T)
// ---------------------------------------------------------------------------
static void cpu_matmul_bt(float* C, const float* A, const float* W, int M, int K, int N) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[m * K + k] * W[n * K + k];
            C[m * N + n] = sum;
        }
}

static int test_matmul_bt() {
    TEST("matmul_bt");
    int M = 8, K = 32, N = 16;
    int n_cpu = M * N, n_a = M * K, n_w = N * K;

    float* A = (float*)malloc(n_a * sizeof(float));
    float* W = (float*)malloc(n_w * sizeof(float));
    float* C_cpu = (float*)malloc(n_cpu * sizeof(float));
    for (int i = 0; i < n_a; i++) A[i] = rand_f32();
    for (int i = 0; i < n_w; i++) W[i] = rand_f32();

    cpu_matmul_bt(C_cpu, A, W, M, K, N);

    float *d_A, *d_W, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, n_a * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, n_w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, n_cpu * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A, n_a * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W, n_w * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    matmul_bt_cuda(handle, d_A, d_W, d_C, M, K, N);

    float* C_gpu = (float*)malloc(n_cpu * sizeof(float));
    CUDA_CHECK(cudaMemcpy(C_gpu, d_C, n_cpu * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = compare_arrays(C_cpu, C_gpu, n_cpu, 1e-4f, 1e-5f);
    free(A); free(W); free(C_cpu); free(C_gpu);
    cudaFree(d_A); cudaFree(d_W); cudaFree(d_C);
    if (bad >= 0) FAIL("matmul_bt mismatch");
    PASS(); return 0;
}

// ---------------------------------------------------------------------------
// CPU reference: RMSNorm
// ---------------------------------------------------------------------------
static void cpu_rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

static int test_rmsnorm() {
    TEST("rmsnorm");
    int rows = 4, cols = 64;
    int n = rows * cols;
    float* x = (float*)malloc(n * sizeof(float));
    float* w = (float*)malloc(cols * sizeof(float));
    float* cpu_out = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = rand_f32();
    for (int i = 0; i < cols; i++) w[i] = rand_f32() + 0.5f;
    for (int r = 0; r < rows; r++)
        cpu_rmsnorm(cpu_out + r * cols, x + r * cols, w, cols, 1e-5f);

    float *d_x, *d_w, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w, cols * sizeof(float), cudaMemcpyHostToDevice));

    boat_cuda_rmsnorm_forward_f32(d_x, d_w, d_out, rows, cols, 1e-5f);

    float* gpu_out = (float*)malloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = compare_arrays(cpu_out, gpu_out, n, 1e-4f, 1e-5f);
    free(x); free(w); free(cpu_out); free(gpu_out);
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
    if (bad >= 0) FAIL("rmsnorm mismatch");
    PASS(); return 0;
}

// ---------------------------------------------------------------------------
// CPU reference: SiLU gate
// ---------------------------------------------------------------------------
static float cpu_silu(float x) { return x / (1.0f + expf(-x)); }

static void cpu_silu_gate(float* d, int N, int ff) {
    for (int i = 0; i < N * ff; i++)
        d[i] = cpu_silu(d[i]) * d[i + N * ff];
}

static int test_silu_gate() {
    TEST("silu_gate");
    int N = 4, ff = 32;
    int total = N * ff * 2;
    float* data = (float*)malloc(total * sizeof(float));
    for (int i = 0; i < total; i++) data[i] = rand_f32();

    float* cpu_out = (float*)malloc(2 * N * ff * sizeof(float));
    memcpy(cpu_out, data, N * ff * sizeof(float));
    memcpy(cpu_out + N * ff, data + N * ff, N * ff * sizeof(float));
    cpu_silu_gate(cpu_out, N, ff);

    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data, total * sizeof(float), cudaMemcpyHostToDevice));

    silu_gate_cuda(d_data, N, ff, 0);

    float* gpu_out = (float*)malloc(N * ff * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_out, d_data, N * ff * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = compare_arrays(cpu_out, gpu_out, N * ff, 1e-4f, 1e-5f);
    free(data); free(cpu_out); free(gpu_out);
    cudaFree(d_data);
    if (bad >= 0) FAIL("silu_gate mismatch");
    PASS(); return 0;
}

// ---------------------------------------------------------------------------
// CPU reference: patch embed (simplified with small dimensions)
// ---------------------------------------------------------------------------
static void cpu_patch_embed(float* out, const float* img, const float* w,
                             const float* b, int H, int W, int C_out, int ps) {
    int oh = H / ps, ow = W / ps;
    int in_ch = 3, temp = 2;
    for (int oc = 0; oc < C_out; oc++)
        for (int oy = 0; oy < oh; oy++)
            for (int ox = 0; ox < ow; ox++) {
                float sum = b[oc];
                for (int ic = 0; ic < in_ch; ic++)
                    for (int t = 0; t < temp; t++)
                        for (int ky = 0; ky < ps; ky++)
                            for (int kx = 0; kx < ps; kx++) {
                                int iy = oy * ps + ky, ix = ox * ps + kx;
                                if (iy < H && ix < W) {
                                    int widx = ((oc * in_ch + ic) * temp + t) * ps * ps + ky * ps + kx;
                                    sum += w[widx] * img[ic * H * W + iy * W + ix];
                                }
                            }
                out[oc * oh * ow + oy * ow + ox] = sum;
            }
}

static int test_patch_embed() {
    TEST("patch_embed");
    int C_out = 4, patch_size = 4, H = 8, W = 8;
    int oh = H / patch_size, ow = W / patch_size;
    int img_size = 3 * H * W;
    int w_size = C_out * 6 * patch_size * patch_size;

    float* img = (float*)malloc(img_size * sizeof(float));
    float* weight = (float*)malloc(w_size * sizeof(float));
    float* bias = (float*)malloc(C_out * sizeof(float));
    for (int i = 0; i < img_size; i++) img[i] = rand_f32();
    for (int i = 0; i < w_size; i++) weight[i] = rand_f32();
    for (int i = 0; i < C_out; i++) bias[i] = rand_f32();

    int out_size = C_out * oh * ow;
    float* cpu_out = (float*)malloc(out_size * sizeof(float));
    cpu_patch_embed(cpu_out, img, weight, bias, H, W, C_out, patch_size);

    float *d_img, *d_w, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_img, img_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, C_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_img, img, img_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, weight, w_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, bias, C_out * sizeof(float), cudaMemcpyHostToDevice));

    patch_embed_cuda(d_img, d_w, d_b, d_out, H, W, C_out, patch_size, 0);

    float* gpu_out = (float*)malloc(out_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = compare_arrays(cpu_out, gpu_out, out_size, 1e-4f, 1e-5f);
    free(img); free(weight); free(bias); free(cpu_out); free(gpu_out);
    cudaFree(d_img); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
    if (bad >= 0) FAIL("patch_embed mismatch");
    PASS(); return 0;
}

// ---------------------------------------------------------------------------
// CPU reference: patch reorder
// ---------------------------------------------------------------------------
static void cpu_patch_reorder(float* out, const float* inp, int C, int ph, int pw) {
    for (int c = 0; c < C; c++)
        for (int r = 0; r < ph; r++)
            for (int cw = 0; cw < pw; cw++) {
                int p = r * pw + cw;
                int hw2 = pw / 2;
                int sb = ((r / 2) * hw2 + (cw / 2)) * 4 + (r % 2) * 2 + (cw % 2);
                out[sb * C + c] = inp[c * ph * pw + p];
            }
}

static int test_patch_reorder() {
    TEST("patch_reorder");
    int C = 4, ph = 4, pw = 4;
    int N = ph * pw, total = N * C;
    float* inp = (float*)malloc(total * sizeof(float));
    for (int i = 0; i < total; i++) inp[i] = rand_f32();

    float* cpu_out = (float*)malloc(total * sizeof(float));
    cpu_patch_reorder(cpu_out, inp, C, ph, pw);

    float *d_inp, *d_out;
    CUDA_CHECK(cudaMalloc(&d_inp, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_inp, inp, total * sizeof(float), cudaMemcpyHostToDevice));

    patch_reorder_cuda(d_inp, d_out, C, ph, pw, 0);

    float* gpu_out = (float*)malloc(total * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = compare_arrays(cpu_out, gpu_out, total, 1e-6f, 1e-7f);
    free(inp); free(cpu_out); free(gpu_out);
    cudaFree(d_inp); cudaFree(d_out);
    if (bad >= 0) FAIL("patch_reorder mismatch");
    PASS(); return 0;
}

// ---------------------------------------------------------------------------
// CPU reference: downsample
// ---------------------------------------------------------------------------
static void cpu_downsample(float* out, const float* inp, const float* w,
                            const float* b, int np, int hid, int ohid) {
    int ng = np / 4;
    for (int g = 0; g < ng; g++)
        for (int oc = 0; oc < ohid; oc++) {
            float sum = b[oc];
            for (int ic = 0; ic < hid; ic++)
                for (int ky = 0; ky < 2; ky++)
                    for (int kx = 0; kx < 2; kx++)
                        sum += inp[(g * 4 + ky * 2 + kx) * hid + ic]
                             * w[(oc * hid + ic) * 4 + ky * 2 + kx];
            out[g * ohid + oc] = sum;
        }
}

static int test_downsample() {
    TEST("downsample");
    int hid = 8, ohid = 12, np = 16;
    int inp_size = np * hid, w_size = ohid * hid * 4, b_size = ohid, out_size = (np/4) * ohid;

    float* inp = (float*)malloc(inp_size * sizeof(float));
    float* weight = (float*)malloc(w_size * sizeof(float));
    float* bias = (float*)malloc(b_size * sizeof(float));
    for (int i = 0; i < inp_size; i++) inp[i] = rand_f32();
    for (int i = 0; i < w_size; i++) weight[i] = rand_f32();
    for (int i = 0; i < b_size; i++) bias[i] = rand_f32();

    float* cpu_out = (float*)malloc(out_size * sizeof(float));
    cpu_downsample(cpu_out, inp, weight, bias, np, hid, ohid);

    float *d_inp, *d_w, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_inp, inp_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, b_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_inp, inp, inp_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, weight, w_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, bias, b_size * sizeof(float), cudaMemcpyHostToDevice));

    downsample_cuda(d_inp, d_w, d_b, d_out, np, hid, ohid, 0);

    float* gpu_out = (float*)malloc(out_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = compare_arrays(cpu_out, gpu_out, out_size, 1e-4f, 1e-5f);
    free(inp); free(weight); free(bias); free(cpu_out); free(gpu_out);
    cudaFree(d_inp); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
    if (bad >= 0) FAIL("downsample mismatch");
    PASS(); return 0;
}

// ---------------------------------------------------------------------------
// CPU reference: 2D RoPE
// ---------------------------------------------------------------------------
static void cpu_rope_2d_compute(float* cosv, float* sinv, int ph, int pw, int hdim, float theta) {
    int N = ph * pw;
    for (int p = 0; p < N; p++) {
        int r = p / pw, cw = p % pw, hw2 = pw / 2;
        int sb = ((r / 2) * hw2 + (cw / 2)) * 4 + (r % 2) * 2 + (cw % 2);
        for (int i = 0; i < 16; i++) {
            float inv = 1.0f / powf(theta, (2.0f * i) / 32.0f);
            float th = r * inv, tw = cw * inv;
            cosv[sb * 64 + i] = cosf(th);       cosv[sb * 64 + 32 + i] = cosf(th);
            cosv[sb * 64 + 16 + i] = cosf(tw);  cosv[sb * 64 + 48 + i] = cosf(tw);
            sinv[sb * 64 + i] = sinf(th);       sinv[sb * 64 + 32 + i] = sinf(th);
            sinv[sb * 64 + 16 + i] = sinf(tw);  sinv[sb * 64 + 48 + i] = sinf(tw);
        }
    }
}

static void cpu_apply_rope_2d(float* q, float* k, int N, int nh, int hdim,
                               const float* cosv, const float* sinv) {
    int stride = nh * hdim;
    for (int p = 0; p < N; p++)
        for (int h = 0; h < nh; h++)
            for (int d = 0; d < hdim; d += 2) {
                int off = p * stride + h * hdim + d;
                float q0 = q[off], q1 = q[off + 1];
                float c = cosv[p * hdim + d], s = sinv[p * hdim + d];
                q[off] = q0 * c - q1 * s; q[off + 1] = q1 * c + q0 * s;
                if (k) {
                    float k0 = k[off], k1 = k[off + 1];
                    k[off] = k0 * c - k1 * s; k[off + 1] = k1 * c + k0 * s;
                }
            }
}

static int test_rope_2d() {
    TEST("rope_2d");
    int ph = 4, pw = 4, N = ph * pw, nh = 2, hdim = 64;
    int qk_size = N * nh * hdim, cos_size = N * hdim;

    float* q = (float*)malloc(qk_size * sizeof(float));
    float* k = (float*)malloc(qk_size * sizeof(float));
    float* cosv = (float*)malloc(cos_size * sizeof(float));
    float* sinv = (float*)malloc(cos_size * sizeof(float));
    for (int i = 0; i < qk_size; i++) { q[i] = rand_f32(); k[i] = rand_f32(); }

    float* q_ref = (float*)malloc(qk_size * sizeof(float));
    float* k_ref = (float*)malloc(qk_size * sizeof(float));
    memcpy(q_ref, q, qk_size * sizeof(float));
    memcpy(k_ref, k, qk_size * sizeof(float));
    cpu_rope_2d_compute(cosv, sinv, ph, pw, hdim, 10000.0f);
    cpu_apply_rope_2d(q_ref, k_ref, N, nh, hdim, cosv, sinv);

    float *d_q, *d_k, *d_cos, *d_sin;
    CUDA_CHECK(cudaMalloc(&d_q, qk_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, qk_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, cos_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, cos_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, q, qk_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k, qk_size * sizeof(float), cudaMemcpyHostToDevice));

    rope_2d_compute_cuda(d_cos, d_sin, ph, pw, hdim, 10000.0f, 0);
    apply_rope_2d_cuda(d_q, d_k, N, nh, hdim, d_cos, d_sin, 0);

    float* gpu_q = (float*)malloc(qk_size * sizeof(float));
    float* gpu_k = (float*)malloc(qk_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_q, d_q, qk_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_k, d_k, qk_size * sizeof(float), cudaMemcpyDeviceToHost));

    int bad_q = compare_arrays(q_ref, gpu_q, qk_size, 1e-4f, 1e-5f);
    int bad_k = compare_arrays(k_ref, gpu_k, qk_size, 1e-4f, 1e-5f);
    free(q); free(k); free(cosv); free(sinv); free(q_ref); free(k_ref);
    free(gpu_q); free(gpu_k);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_cos); cudaFree(d_sin);
    if (bad_q >= 0) FAIL("rope_2d Q mismatch");
    if (bad_k >= 0) FAIL("rope_2d K mismatch");
    PASS(); return 0;
}

// ---------------------------------------------------------------------------
// CPU reference: M-RoPE (matches ocr_common.h apply_rope_mrope)
// ---------------------------------------------------------------------------
static void cpu_apply_mrope(float* q, float* k, int seq_len, int nh, int nkv, int hdim,
                             float theta, const int* pt, const int* ph, const int* pw) {
    int sec_t = 16, sec_h = 24, sec_w = 24, sec_sum = 64;
    for (int p = 0; p < seq_len; p++) {
        for (int h = 0; h < nh; h++) {
            int q_off = (p * nh + h) * hdim;
            for (int s = 0; s < 2; s++) {
                int base = s * sec_sum;
                for (int i = base; i < base + sec_t; i += 2) {
                    float freq = powf(theta, -(float)i / (float)hdim);
                    float c = cosf(pt[p] * freq), sv = sinf(pt[p] * freq);
                    float x0 = q[q_off + i], x1 = q[q_off + i + 1];
                    q[q_off + i] = x0 * c - x1 * sv; q[q_off + i + 1] = x1 * c + x0 * sv;
                }
                for (int i = base + sec_t; i < base + sec_t + sec_h; i += 2) {
                    float freq = powf(theta, -(float)i / (float)hdim);
                    float c = cosf(ph[p] * freq), sv = sinf(ph[p] * freq);
                    float x0 = q[q_off + i], x1 = q[q_off + i + 1];
                    q[q_off + i] = x0 * c - x1 * sv; q[q_off + i + 1] = x1 * c + x0 * sv;
                }
                for (int i = base + sec_t + sec_h; i < base + sec_sum; i += 2) {
                    float freq = powf(theta, -(float)i / (float)hdim);
                    float c = cosf(pw[p] * freq), sv = sinf(pw[p] * freq);
                    float x0 = q[q_off + i], x1 = q[q_off + i + 1];
                    q[q_off + i] = x0 * c - x1 * sv; q[q_off + i + 1] = x1 * c + x0 * sv;
                }
            }
        }
        if (k) for (int h = 0; h < nkv; h++) {
            int k_off = (p * nkv + h) * hdim;
            for (int s = 0; s < 2; s++) {
                int base = s * sec_sum;
                for (int i = base; i < base + sec_t; i += 2) {
                    float freq = powf(theta, -(float)i / (float)hdim);
                    float c = cosf(pt[p] * freq), sv = sinf(pt[p] * freq);
                    float x0 = k[k_off + i], x1 = k[k_off + i + 1];
                    k[k_off + i] = x0 * c - x1 * sv; k[k_off + i + 1] = x1 * c + x0 * sv;
                }
                for (int i = base + sec_t; i < base + sec_t + sec_h; i += 2) {
                    float freq = powf(theta, -(float)i / (float)hdim);
                    float c = cosf(ph[p] * freq), sv = sinf(ph[p] * freq);
                    float x0 = k[k_off + i], x1 = k[k_off + i + 1];
                    k[k_off + i] = x0 * c - x1 * sv; k[k_off + i + 1] = x1 * c + x0 * sv;
                }
                for (int i = base + sec_t + sec_h; i < base + sec_sum; i += 2) {
                    float freq = powf(theta, -(float)i / (float)hdim);
                    float c = cosf(pw[p] * freq), sv = sinf(pw[p] * freq);
                    float x0 = k[k_off + i], x1 = k[k_off + i + 1];
                    k[k_off + i] = x0 * c - x1 * sv; k[k_off + i + 1] = x1 * c + x0 * sv;
                }
            }
        }
    }
}

static int test_mrope() {
    TEST("mrope");
    int seq = 4, nh = 2, nkv = 1, hdim = 128;
    int q_size = seq * nh * hdim, k_size = seq * nkv * hdim;

    float* q = (float*)malloc(q_size * sizeof(float));
    float* k = (float*)malloc(k_size * sizeof(float));
    int* pt = (int*)malloc(seq * sizeof(int));
    int* ph = (int*)malloc(seq * sizeof(int));
    int* pw = (int*)malloc(seq * sizeof(int));
    for (int i = 0; i < q_size; i++) q[i] = rand_f32();
    for (int i = 0; i < k_size; i++) k[i] = rand_f32();
    pt[0] = 0; ph[0] = 0; pw[0] = 0;
    pt[1] = 5; ph[1] = 7; pw[1] = 3;
    pt[2] = 6; ph[2] = 8; pw[2] = 4;
    pt[3] = 7; ph[3] = 7; pw[3] = 7;

    float* q_ref = (float*)malloc(q_size * sizeof(float));
    float* k_ref = (float*)malloc(k_size * sizeof(float));
    memcpy(q_ref, q, q_size * sizeof(float));
    memcpy(k_ref, k, k_size * sizeof(float));
    cpu_apply_mrope(q_ref, k_ref, seq, nh, nkv, hdim, 10000.0f, pt, ph, pw);

    float *d_q, *d_k; int *d_pt, *d_ph, *d_pw;
    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, k_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pt, seq * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ph, seq * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pw, seq * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_q, q, q_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k, k_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pt, pt, seq * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ph, ph, seq * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pw, pw, seq * sizeof(int), cudaMemcpyHostToDevice));

    apply_mrope_cuda(d_q, d_k, seq, nh, nkv, hdim, 10000.0f, d_pt, d_ph, d_pw, 0);

    float* gpu_q = (float*)malloc(q_size * sizeof(float));
    float* gpu_k = (float*)malloc(k_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_q, d_q, q_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_k, d_k, k_size * sizeof(float), cudaMemcpyDeviceToHost));

    int bad_q = compare_arrays(q_ref, gpu_q, q_size, 1e-4f, 1e-5f);
    int bad_k = compare_arrays(k_ref, gpu_k, k_size, 1e-4f, 1e-5f);
    free(q); free(k); free(pt); free(ph); free(pw);
    free(q_ref); free(k_ref); free(gpu_q); free(gpu_k);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_pt); cudaFree(d_ph); cudaFree(d_pw);
    if (bad_q >= 0) FAIL("mrope Q mismatch");
    if (bad_k >= 0) FAIL("mrope K mismatch");
    PASS(); return 0;
}

// ---------------------------------------------------------------------------
// CPU reference: batched attention (CogViT style)
// ---------------------------------------------------------------------------
static void cpu_batched_attention(float* ctx, const float* q, const float* k,
                                   const float* v, int N, int nh, int hd, float scale) {
    int stride = nh * hd;
    float* scores = (float*)malloc((size_t)nh * N * N * sizeof(float));
    for (int h = 0; h < nh; h++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int d = 0; d < hd; d++)
                    sum += q[i * stride + h * hd + d] * k[j * stride + h * hd + d];
                scores[(h * N + i) * N + j] = sum * scale;
            }
    for (int h = 0; h < nh; h++)
        for (int i = 0; i < N; i++) {
            int base = (h * N + i) * N;
            float mx = scores[base];
            for (int j = 1; j < N; j++) if (scores[base + j] > mx) mx = scores[base + j];
            float sum = 0.0f;
            for (int j = 0; j < N; j++) { scores[base + j] = expf(scores[base + j] - mx); sum += scores[base + j]; }
            float inv = 1.0f / sum;
            for (int j = 0; j < N; j++) scores[base + j] *= inv;
        }
    memset(ctx, 0, (size_t)N * stride * sizeof(float));
    for (int h = 0; h < nh; h++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float attn = scores[(h * N + i) * N + j];
                for (int d = 0; d < hd; d++)
                    ctx[i * stride + h * hd + d] += attn * v[j * stride + h * hd + d];
            }
    free(scores);
}

static int test_attention() {
    TEST("batched_attention");
    int N = 8, nh = 2, hd = 32;
    int size = N * nh * hd;
    float* q = (float*)malloc(size * sizeof(float));
    float* k = (float*)malloc(size * sizeof(float));
    float* v = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) { q[i] = rand_f32(); k[i] = rand_f32(); v[i] = rand_f32(); }

    float* ctx_cpu = (float*)malloc(size * sizeof(float));
    cpu_batched_attention(ctx_cpu, q, k, v, N, nh, hd, 1.0f / sqrtf((float)hd));

    float *d_q, *d_k, *d_v, *d_ctx;
    CUDA_CHECK(cudaMalloc(&d_q, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ctx, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, q, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v, size * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    batched_attention_cuda(handle, d_q, d_k, d_v, d_ctx, N, nh, hd, 1.0f / sqrtf((float)hd), 0);

    float* gpu_ctx = (float*)malloc(size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_ctx, d_ctx, size * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = compare_arrays(ctx_cpu, gpu_ctx, size, 1e-4f, 1e-5f);
    free(q); free(k); free(v); free(ctx_cpu); free(gpu_ctx);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_ctx);
    if (bad >= 0) FAIL("attention mismatch");
    PASS(); return 0;
}

// ---------------------------------------------------------------------------
// CPU reference: LayerNorm + GELU (fused merger)
// ---------------------------------------------------------------------------
static float cpu_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void cpu_layernorm_gelu(float* y, const float* x, const float* w, const float* b,
                                 int M, int D, float eps) {
    for (int row = 0; row < M; row++) {
        const float* rx = x + row * D;
        float sum = 0.0f;
        for (int i = 0; i < D; i++) sum += rx[i];
        float mean = sum / D;
        float vs = 0.0f;
        for (int i = 0; i < D; i++) { float d = rx[i] - mean; vs += d * d; }
        float inv = 1.0f / sqrtf(vs / D + eps);
        for (int i = 0; i < D; i++) {
            float val = (rx[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);
            y[row * D + i] = cpu_gelu(val);
        }
    }
}

static int test_layernorm_gelu() {
    TEST("layernorm_gelu");
    int M = 4, D = 32;
    int n = M * D;
    float* x = (float*)malloc(n * sizeof(float));
    float* w = (float*)malloc(D * sizeof(float));
    float* bias = (float*)malloc(D * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = rand_f32();
    for (int i = 0; i < D; i++) { w[i] = rand_f32() + 0.5f; bias[i] = rand_f32(); }

    float* cpu_out = (float*)malloc(n * sizeof(float));
    cpu_layernorm_gelu(cpu_out, x, w, bias, M, D, 1e-5f);

    float *d_x, *d_w, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w, D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, bias, D * sizeof(float), cudaMemcpyHostToDevice));

    layernorm_gelu_cuda(d_x, d_out, d_w, d_b, M, D, 1e-5f, 0);

    float* gpu_out = (float*)malloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = compare_arrays(cpu_out, gpu_out, n, 1e-4f, 1e-5f);
    free(x); free(w); free(bias); free(cpu_out); free(gpu_out);
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
    if (bad >= 0) FAIL("layernorm_gelu mismatch");
    PASS(); return 0;
}

// ---------------------------------------------------------------------------
// CPU reference: GQA decode (single step)
// ---------------------------------------------------------------------------
static void cpu_gqa_decode(float* ctx, const float* q, const float* k_cache,
                            const float* v_cache, int kv_len, int nh, int nkv, int hd) {
    int groups = nh / nkv;
    int kvs = nkv * hd;
    float scale = 1.0f / sqrtf((float)hd);
    float* scores = (float*)malloc(kv_len * sizeof(float));

    for (int h = 0; h < nh; h++) {
        int kv_h = h / groups;
        for (int j = 0; j < kv_len; j++) {
            float sum = 0.0f;
            for (int d = 0; d < hd; d++)
                sum += q[h * hd + d] * k_cache[j * kvs + kv_h * hd + d];
            scores[j] = sum * scale;
        }
        float mx = scores[0];
        for (int j = 1; j < kv_len; j++) if (scores[j] > mx) mx = scores[j];
        float sum_exp = 0.0f;
        for (int j = 0; j < kv_len; j++) { scores[j] = expf(scores[j] - mx); sum_exp += scores[j]; }
        for (int j = 0; j < kv_len; j++) scores[j] /= sum_exp;

        for (int d = 0; d < hd; d++) {
            float sum = 0.0f;
            for (int j = 0; j < kv_len; j++)
                sum += scores[j] * v_cache[j * kvs + kv_h * hd + d];
            ctx[h * hd + d] = sum;
        }
    }
    free(scores);
}

static int test_gqa_decode() {
    TEST("gqa_decode");
    int nh = 4, nkv = 2, hd = 32, kv_len = 8;
    int q_size = nh * hd, kv_size = nkv * hd;

    float* q = (float*)malloc(q_size * sizeof(float));
    float* k_cache = (float*)malloc((size_t)kv_len * kv_size * sizeof(float));
    float* v_cache = (float*)malloc((size_t)kv_len * kv_size * sizeof(float));
    float* ctx_cpu = (float*)malloc(q_size * sizeof(float));
    for (int i = 0; i < q_size; i++) q[i] = rand_f32();
    for (int i = 0; i < kv_len * kv_size; i++) { k_cache[i] = rand_f32(); v_cache[i] = rand_f32(); }

    cpu_gqa_decode(ctx_cpu, q, k_cache, v_cache, kv_len, nh, nkv, hd);

    float *d_q, *d_kc, *d_vc, *d_ctx;
    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kc, (size_t)kv_len * kv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vc, (size_t)kv_len * kv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ctx, q_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, q, q_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kc, k_cache, (size_t)kv_len * kv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vc, v_cache, (size_t)kv_len * kv_size * sizeof(float), cudaMemcpyHostToDevice));

    int groups = nh / nkv;
    int shmem = (kv_len + hd) * sizeof(float);
    int nthreads = kv_len > hd ? kv_len : hd;
    nthreads = nthreads < 256 ? 256 : (nthreads > 512 ? 512 : nthreads);

    // Direct kernel call (available via separable compilation)
    extern __global__ void gqa_decode_fused_kernel(
        const float* __restrict__, const float* __restrict__,
        const float* __restrict__, float* __restrict__,
        int, int, int, int, int);
    gqa_decode_fused_kernel<<<nh, nthreads, shmem, 0>>>(
        d_q, d_kc, d_vc, d_ctx, kv_len, nh, nkv, hd, groups);
    CUDA_CHECK(cudaGetLastError());

    float* gpu_ctx = (float*)malloc(q_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_ctx, d_ctx, q_size * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = compare_arrays(ctx_cpu, gpu_ctx, q_size, 1e-4f, 1e-5f);
    free(q); free(k_cache); free(v_cache); free(ctx_cpu); free(gpu_ctx);
    cudaFree(d_q); cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_ctx);
    if (bad >= 0) FAIL("gqa_decode mismatch");
    PASS(); return 0;
}

// ---------------------------------------------------------------------------
// CPU reference: add_bias
// ---------------------------------------------------------------------------
static int test_add_bias() {
    TEST("add_bias");
    int M = 4, N = 16, total = M * N;
    float* data = (float*)malloc(total * sizeof(float));
    float* bias = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < total; i++) data[i] = rand_f32();
    for (int i = 0; i < N; i++) bias[i] = rand_f32();

    float* ref = (float*)malloc(total * sizeof(float));
    memcpy(ref, data, total * sizeof(float));
    for (int i = 0; i < total; i++) ref[i] += bias[i % N];

    float *d_data, *d_bias;
    CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias, N * sizeof(float), cudaMemcpyHostToDevice));

    // Use boat library's add_bias wrapper
    boat_cuda_add_bias_f32(d_data, d_bias, d_data, M, N);

    float* gpu_out = (float*)malloc(total * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_out, d_data, total * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = compare_arrays(ref, gpu_out, total, 1e-6f, 1e-7f);
    free(data); free(bias); free(ref); free(gpu_out);
    cudaFree(d_data); cudaFree(d_bias);
    if (bad >= 0) FAIL("add_bias mismatch");
    PASS(); return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    fprintf(stderr, "=== OCR CUDA Unit Tests ===\n\n");

    // cublas handle init
    boat_cuda_get_cublas_handle();

    int fail = 0;
    fail |= test_matmul_bt();
    fail |= test_rmsnorm();
    fail |= test_silu_gate();
    fail |= test_patch_embed();
    fail |= test_patch_reorder();
    fail |= test_downsample();
    fail |= test_rope_2d();
    fail |= test_mrope();
    fail |= test_attention();
    fail |= test_layernorm_gelu();
    fail |= test_gqa_decode();
    fail |= test_add_bias();

    fprintf(stderr, "\n=== Results: %d/%d passed ===\n", passed_tests, total_tests);
    return fail ? 1 : 0;
}
