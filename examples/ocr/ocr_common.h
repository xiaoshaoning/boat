// ocr_common.h - Shared utility functions for OCR modules
#ifndef BOAT_OCR_COMMON_H
#define BOAT_OCR_COMMON_H

#include <math.h>
#include <string.h>
#include <stdlib.h>

// Inline utility functions (defined here to avoid duplication)

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

static inline void apply_rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * rms * weight[i];
}

#include <cblas.h>

// Matmul with "transposed" weight: C[M,N] = A[M,K] @ W[N,K]^T (implicitly)
// W is stored as [N, K] (out_features, in_features) row-major.
// Optimized using OpenBLAS cblas_sgemm.
static inline void matmul_bt(float* C, const float* A, const float* W, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f,
                A, K,
                W, K,
                0.0f,
                C, N);
}

// RoPE for GLM: applied to both Q (full heads) and K (KV heads)
void apply_rope_glm(float* q, float* k, int seq_len, int num_heads, int num_kv_heads,
                     int head_dim, float theta);

// M-RoPE: 3D rotary position embeddings for visual tokens
// head_dim=128 split into sections: [16(temporal), 24(height), 24(width)] × 2 repeats
// matching HuggingFace GlmOcrTextRotaryEmbedding: inv_freq has 64 values applied to 128 dims
// via repeat_interleave(2), then mrope [16,24,24] sections use t/h/w positions respectively
static inline void apply_rope_mrope(float* q, float* k, int seq_len, int num_heads, int num_kv_heads,
                                     int head_dim, float theta,
                                     const int* pos_t, const int* pos_h, const int* pos_w) {
    int sec_t = 16, sec_h = 24, sec_w = 24;
    int sec_sum = sec_t + sec_h + sec_w;  // 64
    for (int p = 0; p < seq_len; p++) {
        for (int h = 0; h < num_heads; h++) {
            int q_off = (p * num_heads + h) * head_dim;
            // First repeat: dims 0-15 temporal, 16-39 height, 40-63 width
            for (int s = 0; s < 2; s++) {
                int base = s * sec_sum;
                // temporal section
                for (int i = base; i < base + sec_t; i += 2) {
                    float freq = powf(theta, -(float)i / (float)head_dim);
                    float c = cosf(pos_t[p] * freq), s_v = sinf(pos_t[p] * freq);
                    float x0 = q[q_off + i], x1 = q[q_off + i + 1];
                    q[q_off + i] = x0 * c - x1 * s_v; q[q_off + i + 1] = x1 * c + x0 * s_v;
                }
                // height section
                for (int i = base + sec_t; i < base + sec_t + sec_h; i += 2) {
                    float freq = powf(theta, -(float)i / (float)head_dim);
                    float c = cosf(pos_h[p] * freq), s_v = sinf(pos_h[p] * freq);
                    float x0 = q[q_off + i], x1 = q[q_off + i + 1];
                    q[q_off + i] = x0 * c - x1 * s_v; q[q_off + i + 1] = x1 * c + x0 * s_v;
                }
                // width section
                for (int i = base + sec_t + sec_h; i < base + sec_sum; i += 2) {
                    float freq = powf(theta, -(float)i / (float)head_dim);
                    float c = cosf(pos_w[p] * freq), s_v = sinf(pos_w[p] * freq);
                    float x0 = q[q_off + i], x1 = q[q_off + i + 1];
                    q[q_off + i] = x0 * c - x1 * s_v; q[q_off + i + 1] = x1 * c + x0 * s_v;
                }
            }
        }
        if (k) {
            for (int h = 0; h < num_kv_heads; h++) {
                int k_off = (p * num_kv_heads + h) * head_dim;
                for (int s = 0; s < 2; s++) {
                    int base = s * sec_sum;
                    for (int i = base; i < base + sec_t; i += 2) {
                        float freq = powf(theta, -(float)i / (float)head_dim);
                        float c = cosf(pos_t[p] * freq), s_v = sinf(pos_t[p] * freq);
                        float x0 = k[k_off + i], x1 = k[k_off + i + 1];
                        k[k_off + i] = x0 * c - x1 * s_v; k[k_off + i + 1] = x1 * c + x0 * s_v;
                    }
                    for (int i = base + sec_t; i < base + sec_t + sec_h; i += 2) {
                        float freq = powf(theta, -(float)i / (float)head_dim);
                        float c = cosf(pos_h[p] * freq), s_v = sinf(pos_h[p] * freq);
                        float x0 = k[k_off + i], x1 = k[k_off + i + 1];
                        k[k_off + i] = x0 * c - x1 * s_v; k[k_off + i + 1] = x1 * c + x0 * s_v;
                    }
                    for (int i = base + sec_t + sec_h; i < base + sec_sum; i += 2) {
                        float freq = powf(theta, -(float)i / (float)head_dim);
                        float c = cosf(pos_w[p] * freq), s_v = sinf(pos_w[p] * freq);
                        float x0 = k[k_off + i], x1 = k[k_off + i + 1];
                        k[k_off + i] = x0 * c - x1 * s_v; k[k_off + i + 1] = x1 * c + x0 * s_v;
                    }
                }
            }
        }
    }
}

#endif // BOAT_OCR_COMMON_H
