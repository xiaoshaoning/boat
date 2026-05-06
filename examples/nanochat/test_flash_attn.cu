// test_flash_attn.cu — standalone correctness test for flash attention kernels
// Compares GPU (warp shuffle + online softmax) against CPU reference (two-pass softmax)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "nanochat_kernels.cuh"

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                            \
                __FILE__, __LINE__, cudaGetErrorString(err));                  \
        exit(1);                                                                \
    }                                                                           \
} while(0)

// Simulate BF16 storage precision: truncate lower 16 bits of FP32 mantissa
static float quantize_bf16(float x) {
    unsigned int bits;
    memcpy(&bits, &x, sizeof(bits));
    bits &= 0xFFFF0000u;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// CPU reference: causal attention with two-pass softmax
static void cpu_prefill_attention(
    const float* q, const float* k, const float* v,
    float* out, int seq_len, int num_heads, int head_dim)
{
    int stride = num_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int q_pos = 0; q_pos < seq_len; q_pos++) {
        for (int h = 0; h < num_heads; h++) {
            int n_scores = q_pos + 1;
            float* scores = (float*)malloc((size_t)n_scores * sizeof(float));

            // Compute dot products Q[h] · K[kp] for kp=0..q_pos
            for (int kp = 0; kp <= q_pos; kp++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q[(size_t)q_pos * stride + h * head_dim + d] *
                           k[(size_t)kp   * stride + h * head_dim + d];
                }
                scores[kp] = dot * scale;
            }

            // Two-pass softmax: find max
            float mx = -1e38f;
            for (int i = 0; i < n_scores; i++)
                mx = fmaxf(mx, scores[i]);

            // Sum of exponentials
            float sum = 0.0f;
            for (int i = 0; i < n_scores; i++)
                sum += expf(scores[i] - mx);
            float inv_sum = 1.0f / sum;

            // Weighted V sum for each dimension
            for (int d = 0; d < head_dim; d++) {
                float ctxv = 0.0f;
                for (int kp = 0; kp <= q_pos; kp++) {
                    float w = expf(scores[kp] - mx) * inv_sum;
                    ctxv += w * v[(size_t)kp * stride + h * head_dim + d];
                }
                out[(size_t)q_pos * stride + h * head_dim + d] = ctxv;
            }

            free(scores);
        }
    }
}

// CPU reference: decode attention (no causal mask, flat Q)
static void cpu_decode_attention(
    const float* q, const float* k, const float* v,
    float* out, int kv_len, int num_heads, int head_dim)
{
    int stride = num_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < num_heads; h++) {
        // Compute all scores at once
        float* scores = (float*)malloc((size_t)kv_len * sizeof(float));

        for (int kp = 0; kp < kv_len; kp++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q[h * head_dim + d] * k[(size_t)kp * stride + h * head_dim + d];
            }
            scores[kp] = dot * scale;
        }

        // Two-pass softmax
        float mx = -1e38f;
        for (int i = 0; i < kv_len; i++) mx = fmaxf(mx, scores[i]);
        float sum = 0.0f;
        for (int i = 0; i < kv_len; i++) sum += expf(scores[i] - mx);
        float inv_sum = 1.0f / sum;

        // Weighted V sum per dimension
        for (int d = 0; d < head_dim; d++) {
            float ctxv = 0.0f;
            for (int kp = 0; kp < kv_len; kp++) {
                float w = expf(scores[kp] - mx) * inv_sum;
                ctxv += w * v[(size_t)kp * stride + h * head_dim + d];
            }
            out[h * head_dim + d] = ctxv;
        }

        free(scores);
    }
}

// Run one prefill test case
static int test_prefill(int seq_len, int num_heads, int head_dim, float tolerance) {
    int stride = num_heads * head_dim;
    size_t total = (size_t)seq_len * stride;

    // Host data (quantized to BF16 to match GPU input precision)
    float* h_q   = (float*)malloc(total * sizeof(float));
    float* h_k   = (float*)malloc(total * sizeof(float));
    float* h_v   = (float*)malloc(total * sizeof(float));
    float* h_ref = (float*)calloc(total, sizeof(float));
    float* h_out = (float*)calloc(total, sizeof(float));

    srand(42);
    for (size_t i = 0; i < total; i++) {
        h_q[i] = quantize_bf16(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
        h_k[i] = quantize_bf16(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
        h_v[i] = quantize_bf16(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }

    // CPU reference
    cpu_prefill_attention(h_q, h_k, h_v, h_ref, seq_len, num_heads, head_dim);

    // GPU memory
    __nv_bfloat16 *d_q, *d_k, *d_v, *d_ctx;
    float *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q,   total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k,   total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v,   total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_ctx, total * sizeof(__nv_bfloat16)));

    cudaStream_t stream = NULL;

    // Upload: host FP32 → device FP32 → device BF16
    CUDA_CHECK(cudaMemcpy(d_tmp, h_q, total * sizeof(float), cudaMemcpyHostToDevice));
    fp32_to_bf16_cuda(d_tmp, d_q, (int)total, stream);
    CUDA_CHECK(cudaMemcpy(d_tmp, h_k, total * sizeof(float), cudaMemcpyHostToDevice));
    fp32_to_bf16_cuda(d_tmp, d_k, (int)total, stream);
    CUDA_CHECK(cudaMemcpy(d_tmp, h_v, total * sizeof(float), cudaMemcpyHostToDevice));
    fp32_to_bf16_cuda(d_tmp, d_v, (int)total, stream);
    CUDA_CHECK(cudaFree(d_tmp));

    // Run GPU kernel
    float scale = 1.0f / sqrtf((float)head_dim);
    fused_prefill_attention_bf16_cuda(d_q, d_k, d_v, d_ctx,
                                       seq_len, num_heads, head_dim, scale, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy result back: device BF16 → device FP32 → host FP32
    float *d_out_f32;
    CUDA_CHECK(cudaMalloc(&d_out_f32, total * sizeof(float)));
    bf16_to_fp32_cuda(d_ctx, d_out_f32, (int)total, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_out, d_out_f32, total * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare GPU vs CPU
    int errors = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < total; i++) {
        float diff = fabsf(h_out[i] - h_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) {
            if (errors < 5)
                printf("    [%zu] GPU=%.6f CPU=%.6f diff=%.6f\n", i, h_out[i], h_ref[i], diff);
            errors++;
        }
    }

    printf("  prefill seq_len=%-5d heads=%-2d dim=%-3d  "
           "max_diff=%.2e  errors=%d/%zu  %s\n",
           seq_len, num_heads, head_dim, max_diff, errors, total,
           errors == 0 ? "OK" : "FAIL");

    // Cleanup
    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_ctx));
    CUDA_CHECK(cudaFree(d_out_f32));
    free(h_q); free(h_k); free(h_v); free(h_ref); free(h_out);

    return errors;
}

// Run one decode test case
static int test_decode(int kv_len, int num_heads, int head_dim, float tolerance) {
    int stride = num_heads * head_dim;
    size_t q_size = (size_t)num_heads * head_dim;
    size_t cache_size = (size_t)kv_len * stride;
    int total_out = (int)(num_heads * head_dim);

    // Host data (quantized to BF16)
    float* h_q     = (float*)malloc(q_size * sizeof(float));
    float* h_k     = (float*)malloc(cache_size * sizeof(float));
    float* h_v     = (float*)malloc(cache_size * sizeof(float));
    float* h_ref   = (float*)calloc(total_out, sizeof(float));
    float* h_out   = (float*)calloc(total_out, sizeof(float));

    srand(123);
    for (size_t i = 0; i < q_size; i++)
        h_q[i] = quantize_bf16(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    for (size_t i = 0; i < cache_size; i++) {
        h_k[i] = quantize_bf16(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
        h_v[i] = quantize_bf16(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }

    // CPU reference
    cpu_decode_attention(h_q, h_k, h_v, h_ref, kv_len, num_heads, head_dim);

    // GPU memory
    __nv_bfloat16 *d_q, *d_k, *d_v, *d_ctx;
    float *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, (int)(cache_size > q_size ? cache_size : q_size) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q,   q_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k,   cache_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v,   cache_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_ctx, total_out * sizeof(__nv_bfloat16)));

    cudaStream_t stream = NULL;

    CUDA_CHECK(cudaMemcpy(d_tmp, h_q, q_size * sizeof(float), cudaMemcpyHostToDevice));
    fp32_to_bf16_cuda(d_tmp, d_q, (int)q_size, stream);
    CUDA_CHECK(cudaMemcpy(d_tmp, h_k, cache_size * sizeof(float), cudaMemcpyHostToDevice));
    fp32_to_bf16_cuda(d_tmp, d_k, (int)cache_size, stream);
    CUDA_CHECK(cudaMemcpy(d_tmp, h_v, cache_size * sizeof(float), cudaMemcpyHostToDevice));
    fp32_to_bf16_cuda(d_tmp, d_v, (int)cache_size, stream);
    CUDA_CHECK(cudaFree(d_tmp));

    fused_decode_attention_bf16_cuda(d_q, d_k, d_v, d_ctx,
                                      kv_len, num_heads, head_dim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float *d_out_f32;
    CUDA_CHECK(cudaMalloc(&d_out_f32, (size_t)total_out * sizeof(float)));
    bf16_to_fp32_cuda(d_ctx, d_out_f32, total_out, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_out, d_out_f32, (size_t)total_out * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < total_out; i++) {
        float diff = fabsf(h_out[i] - h_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) {
            if (errors < 5)
                printf("    [%d] GPU=%.6f CPU=%.6f diff=%.6f\n", i, h_out[i], h_ref[i], diff);
            errors++;
        }
    }

    printf("  decode  kv_len=%-5d heads=%-2d dim=%-3d  "
           "max_diff=%.2e  errors=%d/%-4d  %s\n",
           kv_len, num_heads, head_dim, max_diff, errors, total_out,
           errors == 0 ? "OK" : "FAIL");

    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_ctx));
    CUDA_CHECK(cudaFree(d_out_f32));
    free(h_q); free(h_k); free(h_v); free(h_ref); free(h_out);

    return errors;
}

// Verify no NaN/Inf in kernel output
static int test_nan_inf(int seq_len, int num_heads, int head_dim) {
    int stride = num_heads * head_dim;
    size_t total = (size_t)seq_len * stride;

    float* h_q = (float*)malloc(total * sizeof(float));
    float* h_k = (float*)malloc(total * sizeof(float));
    float* h_v = (float*)malloc(total * sizeof(float));

    // Edge case: all zeros → softmax should be uniform
    memset(h_q, 0, total * sizeof(float));
    memset(h_k, 0, total * sizeof(float));
    memset(h_v, 0, total * sizeof(float));
    // Set V to different values to check output
    for (size_t i = 0; i < total; i++) h_v[i] = 1.0f;

    __nv_bfloat16 *d_q, *d_k, *d_v, *d_ctx;
    float *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q,   total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k,   total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v,   total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_ctx, total * sizeof(__nv_bfloat16)));

    cudaStream_t stream = NULL;
    CUDA_CHECK(cudaMemcpy(d_tmp, h_q, total * sizeof(float), cudaMemcpyHostToDevice));
    fp32_to_bf16_cuda(d_tmp, d_q, (int)total, stream);
    CUDA_CHECK(cudaMemcpy(d_tmp, h_k, total * sizeof(float), cudaMemcpyHostToDevice));
    fp32_to_bf16_cuda(d_tmp, d_k, (int)total, stream);
    CUDA_CHECK(cudaMemcpy(d_tmp, h_v, total * sizeof(float), cudaMemcpyHostToDevice));
    fp32_to_bf16_cuda(d_tmp, d_v, (int)total, stream);
    CUDA_CHECK(cudaFree(d_tmp));

    float scale = 1.0f / sqrtf((float)head_dim);
    fused_prefill_attention_bf16_cuda(d_q, d_k, d_v, d_ctx,
                                       seq_len, num_heads, head_dim, scale, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float* h_out = (float*)malloc(total * sizeof(float));
    float *d_out_f32;
    CUDA_CHECK(cudaMalloc(&d_out_f32, total * sizeof(float)));
    bf16_to_fp32_cuda(d_ctx, d_out_f32, (int)total, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_out, d_out_f32, total * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (size_t i = 0; i < total; i++) {
        if (isnan(h_out[i]) || isinf(h_out[i])) {
            printf("    NaN/Inf at [%zu]: %f\n", i, h_out[i]);
            errors++;
        }
    }
    // All zeros input → uniform attention → output should equal mean V for each head_dim
    // For each position q_pos, mean V over kp=0..q_pos
    printf("  nan/inf zero-input seq_len=%-4d heads=%-2d  errors=%d  %s\n",
           seq_len, num_heads, errors, errors == 0 ? "OK" : "FAIL");

    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_ctx));
    CUDA_CHECK(cudaFree(d_out_f32));
    free(h_q); free(h_k); free(h_v); free(h_out);

    return errors;
}

int main() {
    printf("=== Flash Attention Kernel Correctness Test ===\n");
    printf("(GPU vs CPU reference, BF16-quantized inputs, FP32 accumulation)\n\n");

    int total_errors = 0;

    // --- Prefill tests ---
    printf("--- Prefill (causal attention) ---\n");
    float tol = 1e-2f;  // BF16 output quantization (~0.8% relative)
    total_errors += test_prefill(1,     1,  128, tol);
    total_errors += test_prefill(1,     17, 128, tol);
    total_errors += test_prefill(17,    17, 128, tol);
    total_errors += test_prefill(128,   1,  128, tol);
    total_errors += test_prefill(512,   1,  128, tol);
    total_errors += test_prefill(2048,  1,  128, tol);
    total_errors += test_prefill(2048,  17, 128, tol);

    // --- Decode tests ---
    printf("\n--- Decode (non-causal, flat Q) ---\n");
    total_errors += test_decode(1,     1,  128, tol);
    total_errors += test_decode(17,    17, 128, tol);
    total_errors += test_decode(128,   1,  128, tol);
    total_errors += test_decode(2048,  1,  128, tol);
    total_errors += test_decode(2048,  17, 128, tol);

    // --- Edge cases ---
    printf("\n--- Edge cases ---\n");
    total_errors += test_nan_inf(128, 1, 128);
    total_errors += test_nan_inf(2048, 17, 128);

    printf("\n=== %s: %d total errors ===\n",
           total_errors == 0 ? "ALL PASSED" : "FAILED", total_errors);
    return total_errors;
}
