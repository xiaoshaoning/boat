// nanochat_kernels.cu - CUDA kernel implementations for NanoChat inference
// BF16 (bfloat16) compute — matches model's trained precision format
#include "nanochat_kernels.cuh"
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
// cuBLAS transposed matmul: C[M,N] = A[M,K] @ W[N,K]^T
// ============================================================================
void matmul_bt_cuda(cublasHandle_t handle,
                    const float* A, const float* W, float* C,
                    int M, int K, int N) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K, &alpha, W, K, A, K, &beta, C, N));
}

// ============================================================================
// FP32 ↔ BF16 conversion helpers
// ============================================================================
__global__ void fp32_to_bf16_kernel(const float* __restrict__ in,
                                     __nv_bfloat16* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

__global__ void bf16_to_fp32_kernel(const __nv_bfloat16* __restrict__ in,
                                     float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

void fp32_to_bf16_cuda(const float* in, __nv_bfloat16* out, int n, cudaStream_t stream) {
    const int block = 256;
    unsigned int grid = (unsigned int)((n + block - 1) / block);
    fp32_to_bf16_kernel<<<grid, block, 0, stream>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_to_fp32_cuda(const __nv_bfloat16* in, float* out, int n, cudaStream_t stream) {
    const int block = 256;
    unsigned int grid = (unsigned int)((n + block - 1) / block);
    bf16_to_fp32_kernel<<<grid, block, 0, stream>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// BF16 transposed matmul (cuBLAS GemmEx, FP32 accumulation)
// ============================================================================
void matmul_bt_bf16_cuda(cublasHandle_t handle,
                         const __nv_bfloat16* A, const __nv_bfloat16* W, __nv_bfloat16* C,
                         int M, int K, int N) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16BF, K,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_16BF, N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void matmul_bt_bf16_out_f32_cuda(cublasHandle_t handle,
                                  const __nv_bfloat16* A, const __nv_bfloat16* W, float* C,
                                  int M, int K, int N) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16BF, K,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_32F, N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ============================================================================
// 1D Standard RoPE kernel (half-pair rotation)
// Pairs (d, d+head_dim/2) for d in [0, head_dim/2), matching HuggingFace
// ============================================================================
__global__ void rope_1d_bf16_kernel(__nv_bfloat16* d_q, __nv_bfloat16* d_k,
                                     int seq_len, int num_heads, int num_kv_heads,
                                     int head_dim, float theta,
                                     const int* d_pos) {
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * (num_heads + num_kv_heads) * half_dim;
    if (idx >= total) return;

    int q_pairs = seq_len * num_heads * half_dim;
    bool is_q = idx < q_pairs;
    int pos_idx, h, d;
    if (is_q) {
        pos_idx = idx / (num_heads * half_dim);
        int within_pos = idx % (num_heads * half_dim);
        h = within_pos / half_dim;
        d = within_pos % half_dim;
    } else {
        int k_off = idx - q_pairs;
        pos_idx = k_off / (num_kv_heads * half_dim);
        int within_pos = k_off % (num_kv_heads * half_dim);
        h = within_pos / half_dim;
        d = within_pos % half_dim;
    }

    __nv_bfloat16* data = is_q ? d_q : d_k;
    int stride = is_q ? (num_heads * head_dim) : (num_kv_heads * head_dim);
    __nv_bfloat16* v0 = &data[pos_idx * stride + h * head_dim + d];
    __nv_bfloat16* v1 = &data[pos_idx * stride + h * head_dim + d + half_dim];

    int pos = d_pos[pos_idx];
    float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    float cos_val = cosf(pos * freq);
    float sin_val = sinf(pos * freq);

    float x0 = __bfloat162float(*v0);
    float x1 = __bfloat162float(*v1);
    *v0 = __float2bfloat16(x0 * cos_val + x1 * sin_val);
    *v1 = __float2bfloat16(x1 * cos_val - x0 * sin_val);
}

void apply_rope_bf16_cuda(__nv_bfloat16* d_q, __nv_bfloat16* d_k,
                          int seq_len, int num_heads, int num_kv_heads,
                          int head_dim, float theta,
                          const int* d_pos,
                          cudaStream_t stream) {
    int half_dim = head_dim / 2;
    int total = seq_len * (num_heads + num_kv_heads) * half_dim;
    const int block = 256;
    unsigned int grid = (unsigned int)((total + block - 1) / block);
    rope_1d_bf16_kernel<<<grid, block, 0, stream>>>(
        d_q, d_k, seq_len, num_heads, num_kv_heads, head_dim, theta, d_pos);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// BF16 ReLU² activation: y = relu(x)^2
// ============================================================================
__global__ void relu2_bf16_kernel(__nv_bfloat16* d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = __bfloat162float(d[idx]);
    d[idx] = __float2bfloat16((v > 0.0f) ? (v * v) : 0.0f);
}

void relu2_bf16_cuda(__nv_bfloat16* d_data, int N, cudaStream_t stream) {
    const int block = 256;
    unsigned int grid = (unsigned int)((N + block - 1) / block);
    relu2_bf16_kernel<<<grid, block, 0, stream>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Logit softcap: y = cap * tanh(x / cap)
// ============================================================================
__global__ void softcap_kernel(float* d, int N, float cap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    d[idx] = cap * tanhf(d[idx] / cap);
}

void softcap_cuda(float* d_data, int N, float cap, cudaStream_t stream) {
    const int block = 256;
    unsigned int grid = (unsigned int)((N + block - 1) / block);
    softcap_kernel<<<grid, block, 0, stream>>>(d_data, N, cap);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// BF16 RMSNorm without learnable weight
// One block per row, uses shared memory for reduction
// ============================================================================
__global__ void rmsnorm_nw_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                        __nv_bfloat16* __restrict__ y,
                                        int cols, float eps) {
    extern __shared__ float sh[];
    int row = blockIdx.x;
    const __nv_bfloat16* row_x = x + (size_t)row * cols;
    __nv_bfloat16* row_y = y + (size_t)row * cols;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        sum_sq += __bfloat162float(row_x[i]) * __bfloat162float(row_x[i]);
    sh[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }

    float rms = rsqrtf(sh[0] / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_y[i] = __float2bfloat16(__bfloat162float(row_x[i]) * rms);
}

void rmsnorm_nw_bf16_cuda(const __nv_bfloat16* d_x, __nv_bfloat16* d_y,
                          int rows, int cols, float eps,
                          cudaStream_t stream) {
    const int block = 256;
    rmsnorm_nw_bf16_kernel<<<rows, block, block * sizeof(float), stream>>>(
        d_x, d_y, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Fused MHA prefill attention — one block per (query_pos, head)
// Grid: (seq_len, num_heads), Block: head_dim threads
// Flash-attention-style: warp shuffle dot product + online softmax
// BF16 I/O, FP32 internal computation
// ============================================================================
__global__ void fused_prefill_attn_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ ctx,
    int seq_len, int num_heads, int head_dim,
    float scale)
{
    int q_pos = blockIdx.x;
    int h     = blockIdx.y;
    int t     = threadIdx.x;
    if (q_pos >= seq_len || h >= num_heads) return;

    __shared__ float warp_sums[4];
    __shared__ float shared_score;

    int stride = num_heads * head_dim;
    float qv = __bfloat162float(q[(size_t)q_pos * stride + h * head_dim + t]);
    int lane = t & 31;
    int warp_id = t >> 5;

    // Online softmax state: running max, sum of exponentials, weighted V accumulator
    float m = -1e38f;
    float d = 0.0f;
    float o = 0.0f;

    for (int kp = 0; kp <= q_pos; kp++) {
        // Each thread: element-wise product Q_d * K_d
        float prod = qv * __bfloat162float(k[(size_t)kp * stride + h * head_dim + t]);

        // Warp shuffle tree reduction — 5 instructions, no shared memory
        for (int offset = 16; offset > 0; offset >>= 1)
            prod += __shfl_xor_sync(0xFFFFFFFF, prod, offset);
        // prod now holds the sum of this warp's 32 threads

        // Cross-warp: one thread per warp writes its partial sum
        if (lane == 0) warp_sums[warp_id] = prod;
        __syncthreads();

        // Thread 0 sums the 4 warp partials and applies scale
        if (t == 0)
            shared_score = (warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3]) * scale;
        __syncthreads();

        float score = shared_score;

        // Online softmax update: rescale previous state if new max found
        float m_old = m;
        m = fmaxf(m, score);
        float exp_old = expf(m_old - m);
        float exp_cur = expf(score - m);

        d = d * exp_old + exp_cur;
        o = o * exp_old + exp_cur * __bfloat162float(v[(size_t)kp * stride + h * head_dim + t]);
    }

    ctx[(size_t)q_pos * stride + h * head_dim + t] = __float2bfloat16(o / d);
}

void fused_prefill_attention_bf16_cuda(
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    __nv_bfloat16* d_ctx,
    int seq_len, int num_heads, int head_dim,
    float scale, cudaStream_t stream)
{
    dim3 grid((unsigned int)seq_len, (unsigned int)num_heads);
    fused_prefill_attn_bf16_kernel<<<grid, (unsigned int)head_dim, 0, stream>>>(
        d_q, d_k, d_v, d_ctx, seq_len, num_heads, head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Fused MHA decode attention — one block per head
// Grid: num_heads, Block: head_dim threads
// Flash-attention-style: warp shuffle dot product + online softmax
// BF16 I/O, FP32 internal computation
// ============================================================================
__global__ void fused_decode_attn_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ ctx,
    int kv_len, int num_heads, int head_dim,
    float scale)
{
    int h = blockIdx.x;
    int t = threadIdx.x;
    if (h >= num_heads) return;

    __shared__ float warp_sums[4];
    __shared__ float shared_score;

    int stride = num_heads * head_dim;
    float qv = __bfloat162float(q[h * head_dim + t]);
    int lane = t & 31;
    int warp_id = t >> 5;

    // Online softmax state: running max, sum of exponentials, weighted V accumulator
    float m = -1e38f;
    float d = 0.0f;
    float o = 0.0f;

    for (int kp = 0; kp < kv_len; kp++) {
        float prod = qv * __bfloat162float(k_cache[(size_t)kp * stride + h * head_dim + t]);

        // Warp shuffle tree reduction
        for (int offset = 16; offset > 0; offset >>= 1)
            prod += __shfl_xor_sync(0xFFFFFFFF, prod, offset);

        // Cross-warp sum
        if (lane == 0) warp_sums[warp_id] = prod;
        __syncthreads();

        if (t == 0)
            shared_score = (warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3]) * scale;
        __syncthreads();

        float score = shared_score;

        // Online softmax
        float m_old = m;
        m = fmaxf(m, score);
        float exp_old = expf(m_old - m);
        float exp_cur = expf(score - m);

        d = d * exp_old + exp_cur;
        o = o * exp_old + exp_cur * __bfloat162float(v_cache[(size_t)kp * stride + h * head_dim + t]);
    }

    ctx[h * head_dim + t] = __float2bfloat16(o / d);
}

void fused_decode_attention_bf16_cuda(
    const __nv_bfloat16* d_q,
    const __nv_bfloat16* d_k_cache, const __nv_bfloat16* d_v_cache,
    __nv_bfloat16* d_ctx,
    int kv_len, int num_heads, int head_dim,
    cudaStream_t stream)
{
    float scale = 1.0f / sqrtf((float)head_dim);
    fused_decode_attn_bf16_kernel<<<(unsigned int)num_heads, (unsigned int)head_dim, 0, stream>>>(
        d_q, d_k_cache, d_v_cache, d_ctx, kv_len, num_heads, head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// BF16 embedding gather: out[i, :] = table[tokens[i], :]
// ============================================================================
__global__ void embed_gather_bf16_kernel(const __nv_bfloat16* __restrict__ table,
                                          const int* __restrict__ tokens,
                                          __nv_bfloat16* __restrict__ out,
                                          int num_tokens, int hidden_size) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;
    const __nv_bfloat16* row = table + (size_t)tokens[t] * hidden_size;
    __nv_bfloat16* dst = out + (size_t)t * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        dst[i] = row[i];
}

void embed_gather_bf16_cuda(const __nv_bfloat16* d_table, const int* d_tokens,
                             __nv_bfloat16* d_out, int num_tokens, int hidden_size,
                             cudaStream_t stream) {
    const int block = 256;
    embed_gather_bf16_kernel<<<num_tokens, block, 0, stream>>>(
        d_table, d_tokens, d_out, num_tokens, hidden_size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// BF16 residual add: y[i] += x[i]
// ============================================================================
__global__ void residual_add_bf16_kernel(__nv_bfloat16* y, const __nv_bfloat16* x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    y[idx] = __float2bfloat16(__bfloat162float(y[idx]) + __bfloat162float(x[idx]));
}

void residual_add_bf16_cuda(__nv_bfloat16* d_y, const __nv_bfloat16* d_x, int N, cudaStream_t stream) {
    const int block = 256;
    unsigned int grid = (unsigned int)((N + block - 1) / block);
    residual_add_bf16_kernel<<<grid, block, 0, stream>>>(d_y, d_x, N);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// NaN scan utility
// ============================================================================
void nan_scan_cuda(const float* d_buf, int N, const char* label,
                   cudaStream_t stream) {
    float h_buf[4];
    CUDA_CHECK(cudaMemcpyAsync(h_buf, d_buf, sizeof(h_buf),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < 4 && i < N; i++) {
        if (isnan(h_buf[i]) || isinf(h_buf[i])) {
            fprintf(stderr, "[NanoChat-CUDA] NaN/Inf in %s at [%d] = %f\n",
                    label, i, h_buf[i]);
            return;
        }
    }
}
