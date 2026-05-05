// kernels.cu - CUDA kernel implementations for Qwen3-ASR (FP32)
#include "kernels.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t stat = call;                                         \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                \
        fprintf(stderr, "[cuBLAS] %s:%d: error %d\n",                  \
                __FILE__, __LINE__, (int)stat);                         \
        exit(1);                                                        \
    }                                                                   \
} while(0)

extern "C" cublasHandle_t boat_cuda_get_cublas_handle(void);

// ============================================================================
// cuBLAS row-major matmul: C[M,N] = A[M,K] @ B[K,N]
// ============================================================================
void matmul_f32_cuda(cublasHandle_t handle,
                     const float* A, const float* B, float* C,
                     int M, int K, int N) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

// ============================================================================
// RoPE 1D kernel — first-half/second-half split (rotate_half)
// Pairs (i, i+half) with the same cos/sin frequency.
// ============================================================================
__global__ void rope_1d_f32_kernel(float* d_Q, float* d_K,
    int T, int NH, int NKV, int HD,
    const float* d_cos, const float* d_sin,
    int pos_offset) {
    // Each thread handles one (pos, head, dim) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = HD / 2;
    int total_q = T * NH * half;
    int total = total_q + T * NKV * half;
    if (idx >= total) return;

    int pos_idx, h, d;
    float* data;
    int stride;

    if (idx < total_q) {
        // Q
        pos_idx = idx / (NH * half);
        int r = idx % (NH * half);
        h = r / half;
        d = r % half;
        data = d_Q;
        stride = NH * HD;
    } else {
        // K
        int k_idx = idx - total_q;
        pos_idx = k_idx / (NKV * half);
        int r = k_idx % (NKV * half);
        h = r / half;
        d = r % half;
        data = d_K;
        stride = NKV * HD;
    }

    int pos = pos_offset + pos_idx;
    float cos_v = d_cos[pos * half + d];
    float sin_v = d_sin[pos * half + d];

    float* v0 = &data[(size_t)pos_idx * stride + (size_t)h * HD + d];
    float* v1 = &data[(size_t)pos_idx * stride + (size_t)h * HD + d + half];
    float a = *v0;
    float b = *v1;
    *v0 = a * cos_v - b * sin_v;
    *v1 = b * cos_v + a * sin_v;
}

void rope_1d_f32_cuda(float* d_Q, float* d_K,
                      int T, int NH, int NKV, int HD,
                      const float* d_cos, const float* d_sin,
                      int pos_offset, cudaStream_t stream) {
    int half = HD / 2;
    int total = T * (NH + NKV) * half;
    const int block = 256;
    unsigned int grid = (unsigned int)((total + block - 1) / block);
    rope_1d_f32_kernel<<<grid, block, 0, stream>>>(
        d_Q, d_K, T, NH, NKV, HD, d_cos, d_sin, pos_offset);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Sinusoidal positional encoding (first-half sin, second-half cos)
// ============================================================================
__global__ void sinusoidal_pe_f32_kernel(float* d_pe, int T, int D, float base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = D / 2;
    int total = T * D;
    if (idx >= total) return;

    int pos = idx / D;
    int dim = idx % D;
    float freq = powf(base, -2.0f * (dim % half) / (float)D);
    if (dim < half)
        d_pe[idx] = sinf(pos * freq);
    else
        d_pe[idx] = cosf(pos * freq);
}

void sinusoidal_pe_f32_cuda(float* d_pe, int T, int D, float base,
                            cudaStream_t stream) {
    const int block = 256;
    unsigned int grid = (unsigned int)(((size_t)T * D + block - 1) / block);
    sinusoidal_pe_f32_kernel<<<grid, block, 0, stream>>>(d_pe, T, D, base);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Fused encoder MHA attention — no causal mask
// Grid: (T, NH), Block: HD threads
// Online softmax with warp shuffle reduction
// ============================================================================
__global__ void fused_enc_attn_f32_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int T, int NH, int HD, float scale)
{
    int q_pos = blockIdx.x;
    int h = blockIdx.y;
    if (q_pos >= T || h >= NH) return;

    extern __shared__ float shared[];
    float* warp_sums = shared;          // num_warps elements
    float* shared_score = shared + 4;   // 1 element (max 4 warps for HD=128)

    int stride = NH * HD;
    float qv = Q[(size_t)q_pos * stride + h * HD + threadIdx.x];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    float m = -INFINITY;
    float d = 0.0f;
    float acc = 0.0f;

    for (int kp = 0; kp < T; kp++) {
        const float* k_row = K + (size_t)kp * stride + h * HD;
        float prod = qv * k_row[threadIdx.x];

        // Warp shuffle tree reduction
        for (int offset = 16; offset > 0; offset >>= 1)
            prod += __shfl_xor_sync(0xFFFFFFFF, prod, offset);

        if (lane == 0) warp_sums[warp_id] = prod;
        __syncthreads();

        if (threadIdx.x == 0) {
            int num_warps = (HD + 31) / 32;
            float total = 0.0f;
            for (int w = 0; w < num_warps; w++) total += warp_sums[w];
            *shared_score = total * scale;
        }
        __syncthreads();

        float score = *shared_score;
        float m_old = m;
        m = fmaxf(m, score);
        float exp_old = expf(m_old - m);
        float exp_cur = expf(score - m);
        d = d * exp_old + exp_cur;
        acc = acc * exp_old + exp_cur * V[(size_t)kp * stride + h * HD + threadIdx.x];
    }

    O[(size_t)q_pos * stride + h * HD + threadIdx.x] = acc / d;
}

void fused_enc_attn_f32_cuda(const float* d_Q, const float* d_K,
                              const float* d_V, float* d_O,
                              int T, int NH, int HD, float scale,
                              cudaStream_t stream) {
    dim3 grid((unsigned int)T, (unsigned int)NH);
    int shared_bytes = (4 + 1) * sizeof(float);  // warp_sums[4] + shared_score
    fused_enc_attn_f32_kernel<<<grid, (unsigned int)HD, shared_bytes, stream>>>(
        d_Q, d_K, d_V, d_O, T, NH, HD, scale);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Fused GQA prefill attention — with causal mask
// Grid: (T, NH), Block: HD threads
// Online softmax with warp shuffle reduction
// ============================================================================
__global__ void fused_gqa_prefill_attn_f32_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int T, int NH, int NKV, int HD, float scale)
{
    int q_pos = blockIdx.x;
    int h = blockIdx.y;
    if (q_pos >= T || h >= NH) return;

    int kv_h = h / (NH / NKV);  // GQA group index
    extern __shared__ float shared[];
    float* warp_sums = shared;
    float* shared_score = shared + 4;

    int stride_q = NH * HD;
    int stride_kv = NKV * HD;
    float qv = Q[(size_t)q_pos * stride_q + h * HD + threadIdx.x];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    float m = -INFINITY;
    float d = 0.0f;
    float acc = 0.0f;

    for (int kp = 0; kp <= q_pos; kp++) {
        const float* k_row = K + (size_t)kp * stride_kv + kv_h * HD;
        float prod = qv * k_row[threadIdx.x];

        for (int offset = 16; offset > 0; offset >>= 1)
            prod += __shfl_xor_sync(0xFFFFFFFF, prod, offset);

        if (lane == 0) warp_sums[warp_id] = prod;
        __syncthreads();

        if (threadIdx.x == 0) {
            int num_warps = (HD + 31) / 32;
            float total = 0.0f;
            for (int w = 0; w < num_warps; w++) total += warp_sums[w];
            *shared_score = total * scale;
        }
        __syncthreads();

        float score = *shared_score;
        float m_old = m;
        m = fmaxf(m, score);
        float exp_old = expf(m_old - m);
        float exp_cur = expf(score - m);
        d = d * exp_old + exp_cur;
        acc = acc * exp_old + exp_cur * V[(size_t)kp * stride_kv + kv_h * HD + threadIdx.x];
    }

    O[(size_t)q_pos * stride_q + h * HD + threadIdx.x] = acc / d;
}

void fused_gqa_prefill_attn_f32_cuda(const float* d_Q, const float* d_K,
                                      const float* d_V, float* d_O,
                                      int T, int NH, int NKV, int HD,
                                      float scale, cudaStream_t stream) {
    dim3 grid((unsigned int)T, (unsigned int)NH);
    int shared_bytes = (4 + 1) * sizeof(float);
    fused_gqa_prefill_attn_f32_kernel<<<grid, (unsigned int)HD, shared_bytes, stream>>>(
        d_Q, d_K, d_V, d_O, T, NH, NKV, HD, scale);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Fused GQA decode attention — single query position
// Grid: NH, Block: HD threads
// Online softmax with warp shuffle reduction
// ============================================================================
__global__ void fused_gqa_decode_attn_f32_kernel(
    const float* __restrict__ Q,        // [NH*HD]
    const float* __restrict__ K_cache,  // [kv_len, NKV*HD]
    const float* __restrict__ V_cache,  // [kv_len, NKV*HD]
    float* __restrict__ O,              // [NH*HD]
    int kv_len, int NH, int NKV, int HD)
{
    int h = blockIdx.x;
    if (h >= NH) return;
    int kv_h = h / (NH / NKV);
    float scale = 1.0f / sqrtf((float)HD);

    extern __shared__ float shared[];
    float* warp_sums = shared;
    float* shared_score = shared + 4;

    float qv = Q[(size_t)h * HD + threadIdx.x];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    float m = -INFINITY;
    float d = 0.0f;
    float acc = 0.0f;

    int stride_kv = NKV * HD;
    for (int j = 0; j < kv_len; j++) {
        const float* k_row = K_cache + (size_t)j * stride_kv + kv_h * HD;
        float prod = qv * k_row[threadIdx.x];

        for (int offset = 16; offset > 0; offset >>= 1)
            prod += __shfl_xor_sync(0xFFFFFFFF, prod, offset);

        if (lane == 0) warp_sums[warp_id] = prod;
        __syncthreads();

        if (threadIdx.x == 0) {
            int num_warps = (HD + 31) / 32;
            float total = 0.0f;
            for (int w = 0; w < num_warps; w++) total += warp_sums[w];
            *shared_score = total * scale;
        }
        __syncthreads();

        float score = *shared_score;
        float m_old = m;
        m = fmaxf(m, score);
        float exp_old = expf(m_old - m);
        float exp_cur = expf(score - m);
        d = d * exp_old + exp_cur;
        acc = acc * exp_old + exp_cur * V_cache[(size_t)j * stride_kv + kv_h * HD + threadIdx.x];
    }

    O[(size_t)h * HD + threadIdx.x] = acc / d;
}

void fused_gqa_decode_attn_f32_cuda(const float* d_Q,
                                     const float* d_K_cache,
                                     const float* d_V_cache,
                                     float* d_O,
                                     int kv_len, int NH, int NKV, int HD,
                                     cudaStream_t stream) {
    int shared_bytes = (4 + 1) * sizeof(float);
    fused_gqa_decode_attn_f32_kernel<<<(unsigned int)NH, (unsigned int)HD, shared_bytes, stream>>>(
        d_Q, d_K_cache, d_V_cache, d_O, kv_len, NH, NKV, HD);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Embedding gather: out[i,:] = table[tokens[i], :]
// ============================================================================
__global__ void embed_gather_f32_kernel(const float* __restrict__ table,
                                        const int* __restrict__ tokens,
                                        float* __restrict__ out,
                                        int num_tokens, int hidden_size) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;
    const float* row = table + (size_t)tokens[t] * hidden_size;
    float* dst = out + (size_t)t * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        dst[i] = row[i];
}

void embed_gather_f32_cuda(const float* d_table, const int* d_tokens,
                           float* d_out, int num_tokens, int hidden_size,
                           cudaStream_t stream) {
    const int block = 256;
    embed_gather_f32_kernel<<<num_tokens, block, 0, stream>>>(
        d_table, d_tokens, d_out, num_tokens, hidden_size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Residual add: y[i] += x[i]
// ============================================================================
__global__ void residual_add_f32_kernel(float* y, const float* x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    y[idx] += x[idx];
}

void residual_add_f32_cuda(float* d_y, const float* d_x, int N,
                           cudaStream_t stream) {
    const int block = 256;
    unsigned int grid = (unsigned int)((N + block - 1) / block);
    residual_add_f32_kernel<<<grid, block, 0, stream>>>(d_y, d_x, N);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// SiLU (in-place): y[i] = y[i] / (1 + exp(-y[i]))
// ============================================================================
__global__ void silu_inplace_f32_kernel(float* d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    d[idx] = d[idx] / (1.0f + expf(-d[idx]));
}

void silu_inplace_f32_cuda(float* d_data, int N, cudaStream_t stream) {
    const int block = 256;
    unsigned int grid = (unsigned int)((N + block - 1) / block);
    silu_inplace_f32_kernel<<<grid, block, 0, stream>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Element-wise multiply: z[i] = x[i] * y[i]
// ============================================================================
__global__ void mul_f32_kernel(float* z, const float* x, const float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    z[idx] = x[idx] * y[idx];
}

void mul_f32_cuda(float* d_z, const float* d_x, const float* d_y, int N,
                  cudaStream_t stream) {
    const int block = 256;
    unsigned int grid = (unsigned int)((N + block - 1) / block);
    mul_f32_kernel<<<grid, block, 0, stream>>>(d_z, d_x, d_y, N);
    CUDA_CHECK(cudaGetLastError());
}
