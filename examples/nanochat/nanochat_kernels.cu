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
// Prefill attention: per-head QK^T -> causal softmax -> PV (MHA, no GQA)
// ============================================================================
// Helper kernel for head extraction
static __global__ void extract_head_kernel(float* dst, const float* src,
                                            int rows, int cols,
                                            int src_stride, int src_off) {
    int r = blockIdx.x;
    int c = threadIdx.x;
    if (r >= rows || c >= cols) return;
    dst[r * cols + c] = src[r * src_stride + src_off + c];
}

static __global__ void causal_softmax_kernel(float* scores, int seq_len) {
    int i = blockIdx.x;  // query position
    if (i >= seq_len) return;
    // cuBLAS Sgemm output is column-major: scores[i + j*seq_len] = Q[i]·K[j]
    // Mask future keys (j > i)
    for (int j = i + 1; j < seq_len; j++)
        scores[i + j * seq_len] = -1e38f;
    // Find max over valid keys (j <= i)
    float mx = scores[i];
    for (int j = 1; j <= i; j++) {
        float v = scores[i + j * seq_len];
        if (v > mx) mx = v;
    }
    // Softmax numerator and sum
    float sum = 0.0f;
    for (int j = 0; j <= i; j++) {
        float v = expf(scores[i + j * seq_len] - mx);
        scores[i + j * seq_len] = v;
        sum += v;
    }
    float inv = 1.0f / sum;
    for (int j = 0; j <= i; j++)
        scores[i + j * seq_len] *= inv;
}

static __global__ void pv_kernel(const float* __restrict__ scores,
                                  const float* __restrict__ v,
                                  float* __restrict__ context,
                                  int seq_len, int head_dim,
                                  int q_off, int kv_off, int q_size, int kv_size) {
    int q = blockIdx.x;
    int d = threadIdx.x;
    if (q >= seq_len || d >= head_dim) return;
    float sum = 0.0f;
    for (int t = 0; t <= q; t++)
        sum += scores[q + t * seq_len] * v[t * kv_size + kv_off + d];
    context[q * q_size + q_off + d] = sum;
}

void prefill_attention_cuda(cublasHandle_t handle,
                            const float* d_q, const float* d_k, const float* d_v,
                            float* d_context,
                            int seq_len, int num_heads, int head_dim,
                            float scale, cudaStream_t stream) {
    int q_size = num_heads * head_dim;
    int kv_size = num_heads * head_dim; // MHA: same as q_size

    float *d_q_h, *d_k_h, *d_scores;
    CUDA_CHECK(cudaMalloc(&d_q_h, (size_t)seq_len * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_h, (size_t)seq_len * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, (size_t)seq_len * seq_len * sizeof(float)));

    for (int h = 0; h < num_heads; h++) {
        // Extract Q_h, K_h from strided buffers (MHA: kv_h == h)
        extract_head_kernel<<<seq_len, head_dim, 0, stream>>>(
            d_q_h, d_q, seq_len, head_dim, q_size, h * head_dim);
        extract_head_kernel<<<seq_len, head_dim, 0, stream>>>(
            d_k_h, d_k, seq_len, head_dim, kv_size, h * head_dim);
        CUDA_CHECK(cudaGetLastError());

        // QK^T * scale (cuBLAS column-major)
        float alpha = scale, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    seq_len, seq_len, head_dim,
                    &alpha, d_q_h, head_dim, d_k_h, head_dim,
                    &beta, d_scores, seq_len));

        // Causal mask + softmax
        causal_softmax_kernel<<<seq_len, 1, 0, stream>>>(d_scores, seq_len);
        CUDA_CHECK(cudaGetLastError());

        // PV multiply
        pv_kernel<<<seq_len, head_dim, 0, stream>>>(
            d_scores, d_v, d_context,
            seq_len, head_dim,
            h * head_dim, h * head_dim, q_size, kv_size);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(d_q_h));
    CUDA_CHECK(cudaFree(d_k_h));
    CUDA_CHECK(cudaFree(d_scores));
}

// ============================================================================
// Softmax for 1D array (single block, cooperative reduction)
// ============================================================================
__global__ void softmax_1d_kernel(float* data, int n) {
    extern __shared__ float sh[];
    int tid = threadIdx.x;

    // Find max (parallel reduction)
    float mx = -1e38f;
    for (int i = tid; i < n; i += blockDim.x) mx = fmaxf(mx, data[i]);
    sh[tid] = mx;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
    }
    __syncthreads();
    mx = sh[0];

    // Sum exp
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += expf(data[i] - mx);
    sh[tid] = sum;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) sh[tid] += sh[tid + s];
    }
    __syncthreads();
    float inv = 1.0f / sh[0];

    // Apply softmax
    for (int i = tid; i < n; i += blockDim.x) data[i] = expf(data[i] - mx) * inv;
}

// ============================================================================
// MHA decode attention using cuBLAS (per-head Sgemv)
// ============================================================================
void decode_attention_cuda(cublasHandle_t handle,
                           const float* d_q,
                           const float* d_k_cache, const float* d_v_cache,
                           float* d_context,
                           int kv_len, int num_heads, int head_dim,
                           cudaStream_t stream) {
    int q_size = num_heads * head_dim;
    float* d_scores;
    CUDA_CHECK(cudaMalloc(&d_scores, (size_t)kv_len * sizeof(float)));

    for (int h = 0; h < num_heads; h++) {
        // scores[t] = scale * sum_d Q[h,d] * K_cache[t,h,d]
        float scale = 1.0f / sqrtf((float)head_dim);
        float alpha = scale, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T,
                    head_dim, kv_len,
                    &alpha,
                    d_k_cache + (size_t)h * head_dim, q_size,
                    d_q + (size_t)h * head_dim, 1,
                    &beta,
                    d_scores, 1));

        // Softmax scores
        softmax_1d_kernel<<<1, 256, 256 * sizeof(float), stream>>>(d_scores, kv_len);
        CUDA_CHECK(cudaGetLastError());

        // ctx[h,d] = sum_t scores[t] * V_cache[t,h,d]
        // Uses cuBLAS Sgemv(OP_N): y[hd] = A[hd,kv_len] * x[kv_len]
        alpha = 1.0f;
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
                    head_dim, kv_len,
                    &alpha,
                    d_v_cache + (size_t)h * head_dim, q_size,
                    d_scores, 1,
                    &beta,
                    d_context + (size_t)h * head_dim, 1));
    }

    CUDA_CHECK(cudaFree(d_scores));
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
