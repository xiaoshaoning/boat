// nanochat_kernels.cu - CUDA kernel implementations for NanoChat inference
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
// 1D Standard RoPE kernel (half-pair rotation)
// Pairs (d, d+head_dim/2) for d in [0, head_dim/2), matching HuggingFace
// ============================================================================
__global__ void rope_1d_kernel(float* d_q, float* d_k,
                                int seq_len, int num_heads, int num_kv_heads,
                                int head_dim, float theta,
                                const int* d_pos) {
    // One thread per (position, head, dim_pair) — only first half of dims
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

    float* data = is_q ? d_q : d_k;
    int stride = is_q ? (num_heads * head_dim) : (num_kv_heads * head_dim);
    // Half-pair: v0 = first-half dim d, v1 = second-half dim d+half_dim
    float* v0 = &data[pos_idx * stride + h * head_dim + d];
    float* v1 = &data[pos_idx * stride + h * head_dim + d + half_dim];

    int pos = d_pos[pos_idx];
    float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    float cos_val = cosf(pos * freq);
    float sin_val = sinf(pos * freq);

    float x0 = *v0;
    float x1 = *v1;
    // HuggingFace apply_rotary_pos_emb: q*cos + rotate_half(q)*sin
    // rotate_half: [second_half, -first_half] → first_half' = q0*cos + q1*sin
    //                                              second_half' = q1*cos - q0*sin
    *v0 = x0 * cos_val + x1 * sin_val;
    *v1 = x1 * cos_val - x0 * sin_val;
}

void apply_rope_cuda(float* d_q, float* d_k,
                     int seq_len, int num_heads, int num_kv_heads,
                     int head_dim, float theta,
                     const int* d_pos,
                     cudaStream_t stream) {
    int half_dim = head_dim / 2;
    int total = seq_len * (num_heads + num_kv_heads) * half_dim;
    const int block = 256;
    unsigned int grid = (unsigned int)((total + block - 1) / block);
    rope_1d_kernel<<<grid, block, 0, stream>>>(
        d_q, d_k, seq_len, num_heads, num_kv_heads, head_dim, theta, d_pos);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// ReLU² activation: y = relu(x)^2
// ============================================================================
__global__ void relu2_kernel(float* d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = d[idx];
    d[idx] = (v > 0.0f) ? (v * v) : 0.0f;
}

void relu2_cuda(float* d_data, int N, cudaStream_t stream) {
    const int block = 256;
    unsigned int grid = (unsigned int)((N + block - 1) / block);
    relu2_kernel<<<grid, block, 0, stream>>>(d_data, N);
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
// RMSNorm without learnable weight
// One block per row, uses shared memory for reduction
// ============================================================================
__global__ void rmsnorm_nw_kernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int cols, float eps) {
    extern __shared__ float sh[];
    int row = blockIdx.x;
    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum_sq += row_x[i] * row_x[i];
    }
    sh[threadIdx.x] = sum_sq;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }

    float rms = rsqrtf(sh[0] / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_y[i] = row_x[i] * rms;
    }
}

void rmsnorm_nw_cuda(const float* d_x, float* d_y,
                     int rows, int cols, float eps,
                     cudaStream_t stream) {
    const int block = 256;
    rmsnorm_nw_kernel<<<rows, block, block * sizeof(float), stream>>>(
        d_x, d_y, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Fused MHA prefill attention — one block per (query_pos, head)
// Grid: (seq_len, num_heads), Block: head_dim threads
// Computes Q*K^T + causal softmax + PV in a single kernel
// ============================================================================
__global__ void fused_prefill_attn_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ ctx,
    int seq_len, int num_heads, int head_dim,
    float scale)
{
    int q_pos = blockIdx.x;
    int h     = blockIdx.y;
    int t     = threadIdx.x;

    if (q_pos >= seq_len || h >= num_heads) return;

    extern __shared__ float sh[];
    float* scores  = sh;                    // [0..seq_len-1]
    float* scratch = sh + seq_len;          // [seq_len..seq_len+head_dim-1]
    // sh[seq_len+head_dim] = max_val
    // sh[seq_len+head_dim+1] = sum_val

    int stride = num_heads * head_dim;

    float qv = q[(size_t)q_pos * stride + h * head_dim + t];

    // Pass 1: compute scores for all key positions 0..q_pos
    for (int kp = 0; kp <= q_pos; kp++) {
        scratch[t] = qv * k[(size_t)kp * stride + h * head_dim + t];
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (t < s) scratch[t] += scratch[t + s];
            __syncthreads();
        }

        if (t == 0) scores[kp] = scratch[0] * scale;
        __syncthreads();
    }

    // Thread 0: find max, compute softmax sum
    if (t == 0) {
        float mx = -1e38f;
        for (int i = 0; i <= q_pos; i++)
            mx = fmaxf(mx, scores[i]);
        scratch[head_dim] = mx;

        float sum = 0.0f;
        for (int i = 0; i <= q_pos; i++)
            sum += __expf(scores[i] - mx);
        scratch[head_dim + 1] = sum;
    }
    __syncthreads();

    float mx   = scratch[head_dim];
    float isum = 1.0f / scratch[head_dim + 1];

    // Pass 2: accumulate context from V
    float ctxv = 0.0f;
    for (int kp = 0; kp <= q_pos; kp++) {
        float w = __expf(scores[kp] - mx) * isum;
        ctxv += w * v[(size_t)kp * stride + h * head_dim + t];
    }

    ctx[(size_t)q_pos * stride + h * head_dim + t] = ctxv;
}

void fused_prefill_attention_cuda(
    const float* d_q, const float* d_k, const float* d_v,
    float* d_ctx,
    int seq_len, int num_heads, int head_dim,
    float scale, cudaStream_t stream)
{
    dim3 grid((unsigned int)seq_len, (unsigned int)num_heads);
    size_t shmem = (size_t)(seq_len + head_dim + 2) * sizeof(float);
    fused_prefill_attn_kernel<<<grid, (unsigned int)head_dim, shmem, stream>>>(
        d_q, d_k, d_v, d_ctx, seq_len, num_heads, head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Fused MHA decode attention — one block per head
// Grid: num_heads, Block: head_dim threads
// Computes Q*K_cache^T + softmax + PV in a single kernel
// ============================================================================
__global__ void fused_decode_attn_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ ctx,
    int kv_len, int num_heads, int head_dim,
    float scale)
{
    int h = blockIdx.x;
    int t = threadIdx.x;

    if (h >= num_heads) return;

    extern __shared__ float sh[];
    float* scores  = sh;                    // [0..kv_len-1]
    float* scratch = sh + kv_len;           // [kv_len..kv_len+head_dim-1]
    // sh[kv_len+head_dim] = max_val
    // sh[kv_len+head_dim+1] = sum_val

    int stride = num_heads * head_dim;

    float qv = q[h * head_dim + t];

    // Compute scores for all KV positions
    for (int kp = 0; kp < kv_len; kp++) {
        scratch[t] = qv * k_cache[(size_t)kp * stride + h * head_dim + t];
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (t < s) scratch[t] += scratch[t + s];
            __syncthreads();
        }

        if (t == 0) scores[kp] = scratch[0] * scale;
        __syncthreads();
    }

    // Thread 0: softmax normalization
    if (t == 0) {
        float mx = -1e38f;
        for (int i = 0; i < kv_len; i++)
            mx = fmaxf(mx, scores[i]);
        scratch[head_dim] = mx;

        float sum = 0.0f;
        for (int i = 0; i < kv_len; i++)
            sum += __expf(scores[i] - mx);
        scratch[head_dim + 1] = sum;
    }
    __syncthreads();

    float mx   = scratch[head_dim];
    float isum = 1.0f / scratch[head_dim + 1];

    // Accumulate context from V cache
    float ctxv = 0.0f;
    for (int kp = 0; kp < kv_len; kp++) {
        float w = __expf(scores[kp] - mx) * isum;
        ctxv += w * v_cache[(size_t)kp * stride + h * head_dim + t];
    }

    ctx[h * head_dim + t] = ctxv;
}

void fused_decode_attention_cuda(
    const float* d_q,
    const float* d_k_cache, const float* d_v_cache,
    float* d_ctx,
    int kv_len, int num_heads, int head_dim,
    cudaStream_t stream)
{
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t shmem = (size_t)(kv_len + head_dim + 2) * sizeof(float);
    fused_decode_attn_kernel<<<(unsigned int)num_heads, (unsigned int)head_dim, shmem, stream>>>(
        d_q, d_k_cache, d_v_cache, d_ctx, kv_len, num_heads, head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Embedding gather
// ============================================================================
__global__ void embed_gather_kernel(const float* __restrict__ table,
                                     const int* __restrict__ tokens,
                                     float* __restrict__ out,
                                     int num_tokens, int hidden_size) {
    int t = blockIdx.x;
    if (t >= num_tokens) return;
    const float* row = table + tokens[t] * hidden_size;
    float* dst = out + t * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = row[i];
    }
}

void embed_gather_cuda(const float* d_table, const int* d_tokens,
                       float* d_out, int num_tokens, int hidden_size,
                       cudaStream_t stream) {
    const int block = 256;
    embed_gather_kernel<<<num_tokens, block, 0, stream>>>(
        d_table, d_tokens, d_out, num_tokens, hidden_size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Simple GEMM kernel: C[M,N] = A[M,K] @ W[N,K]^T
// One block per row, strided loop over N
// ============================================================================
__global__ void gemm_naive_kernel(const float* __restrict__ a,
                                   const float* __restrict__ w,
                                   float* __restrict__ c,
                                   int M, int K, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    for (int col = tid; col < N; col += total_threads) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * w[col * K + k];
        }
        c[row * N + col] = sum;
    }
}

void gemm_naive_cuda(const float* d_a, const float* d_w, float* d_c,
                     int M, int K, int N, cudaStream_t stream) {
    int block = 256;
    gemm_naive_kernel<<<M, block, 0, stream>>>(d_a, d_w, d_c, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Max element-wise diff between two buffers (one block reduction)
// ============================================================================
__global__ void max_diff_kernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 int N, float* partial_max) {
    __shared__ float sh[256];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int start = bid * block_size;
    int end = min(start + block_size, N);
    float local_max = 0.0f;
    for (int i = start + tid; i < end; i += block_size) {
        float d = fabsf(a[i] - b[i]);
        if (d > local_max) local_max = d;
    }
    sh[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    if (tid == 0) partial_max[bid] = sh[0];
}

void max_diff_cuda(const float* d_a, const float* d_b, int N,
                   float* h_result, cudaStream_t stream) {
    int block = 256;
    unsigned int grid = (unsigned int)((N + block - 1) / block);
    float* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, grid * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_partial, 0, grid * sizeof(float), stream));
    max_diff_kernel<<<grid, block, 0, stream>>>(d_a, d_b, N, d_partial);
    // Second pass: reduction on host (grid is small, at most 9 blocks)
    float* h_partial = (float*)malloc(grid * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(h_partial, d_partial, grid * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float max_val = 0.0f;
    for (unsigned int i = 0; i < grid; i++) if (h_partial[i] > max_val) max_val = h_partial[i];
    *h_result = max_val;
    free(h_partial);
    CUDA_CHECK(cudaFree(d_partial));
}

// ============================================================================
// Residual add: y[i] += x[i]
// ============================================================================
__global__ void residual_add_kernel(float* y, const float* x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] += x[idx];
}

void residual_add_cuda(float* d_y, const float* d_x, int N, cudaStream_t stream) {
    const int block = 256;
    unsigned int grid = (unsigned int)((N + block - 1) / block);
    residual_add_kernel<<<grid, block, 0, stream>>>(d_y, d_x, N);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// NaN scan utility
// ============================================================================
__global__ void nan_scan_kernel(const float* d, int N, const char* label) {
    // Only thread 0 scans — labeling from device is tricky, so we just detect
}

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
