// nanochat_kernels_bw.cu — Backward CUDA kernels for NanoChat training
// BF16 forward activations, FP32 gradient I/O throughout
#include "nanochat_kernels_bw.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "[CUDA-BW] %s:%d: error %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t stat = call;                                         \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                \
        fprintf(stderr, "[cuBLAS-BW] %s:%d: error %d\n",               \
                __FILE__, __LINE__, (int)stat);                         \
        exit(1);                                                        \
    }                                                                   \
} while(0)

extern "C" cublasHandle_t boat_cuda_get_cublas_handle(void);

// ============================================================================
// 1A. RMSNorm backward (no learnable weight) — BF16 input, FP32 gradient I/O
// y_i = x_i * rsqrt(mean(x^2) + eps)
// dx_i = rms * dy_i - (rms^3 / cols) * x_i * sum(dy_j * x_j)
// ============================================================================
__global__ void rmsnorm_nw_bf16_bw_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ d_y,
    float* __restrict__ d_x,
    int cols, float eps)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const __nv_bfloat16* row_x = x + (size_t)row * cols;
    const float* row_dy = d_y + (size_t)row * cols;

    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        sum_sq += __bfloat162float(row_x[i]) * __bfloat162float(row_x[i]);
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / cols + eps);
    float rms_cube = rms * rms * rms;
    __syncthreads();

    float sum_dy_x = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        sum_dy_x += row_dy[i] * __bfloat162float(row_x[i]);
    sdata[tid] = sum_dy_x;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum_dy_x_global = sdata[0];
    __syncthreads();

    float norm = 1.0f / cols;
    for (int i = tid; i < cols; i += blockDim.x)
        d_x[(size_t)row * cols + i] = rms * row_dy[i]
            - rms_cube * norm * __bfloat162float(row_x[i]) * sum_dy_x_global;
}

void rmsnorm_nw_bf16_bw_cuda(
    const __nv_bfloat16* d_x_saved, const float* d_y,
    float* d_x, int rows, int cols, float eps,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid((unsigned int)rows);
    size_t shared_mem = block * sizeof(float);
    rmsnorm_nw_bf16_bw_kernel<<<grid, block, shared_mem, stream>>>(
        d_x_saved, d_y, d_x, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 1B. ReLU² backward: dx = (x > 0) ? 2*x*dy : 0
// ============================================================================
__global__ void relu2_bf16_bw_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ d_output,
    float* __restrict__ d_input,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x_val = __bfloat162float(input[i]);
    d_input[i] = (x_val > 0.0f) ? 2.0f * x_val * d_output[i] : 0.0f;
}

void relu2_bf16_bw_cuda(
    const __nv_bfloat16* d_saved, const float* d_output,
    float* d_input, int N, cudaStream_t stream)
{
    const int block = 256;
    unsigned int grid = (unsigned int)((N + block - 1) / block);
    relu2_bf16_bw_kernel<<<grid, block, 0, stream>>>(d_saved, d_output, d_input, N);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 1C. RoPE backward — transpose rotation
// Forward:  v0' = v0*cos + v1*sin,  v1' = v1*cos - v0*sin
// Backward: dv0 = dv0'*cos - dv1'*sin,  dv1 = dv1'*cos + dv0'*sin
// ============================================================================
__global__ void rope_bf16_bw_kernel(
    const __nv_bfloat16* __restrict__ q_rope,   // saved Q after RoPE (unused, for future ref)
    const __nv_bfloat16* __restrict__ k_rope,   // saved K after RoPE (unused)
    const float* __restrict__ d_q_out,
    const float* __restrict__ d_k_out,
    float* __restrict__ d_q_in,
    float* __restrict__ d_k_in,
    int seq_len, int num_heads, int num_kv_heads,
    int head_dim, float theta,
    const int* __restrict__ d_pos)
{
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = seq_len * (num_heads + num_kv_heads) * half_dim;
    if (idx >= total_pairs) return;

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

    int pos = d_pos[pos_idx];
    float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    float cos_val = cosf(pos * freq);
    float sin_val = sinf(pos * freq);

    if (is_q) {
        int stride = num_heads * head_dim;
        int base = pos_idx * stride + h * head_dim + d;
        d_q_in[base] = d_q_out[base] * cos_val - d_q_out[base + half_dim] * sin_val;
        d_q_in[base + half_dim] = d_q_out[base + half_dim] * cos_val + d_q_out[base] * sin_val;
    } else {
        int stride = num_kv_heads * head_dim;
        int base = pos_idx * stride + h * head_dim + d;
        d_k_in[base] = d_k_out[base] * cos_val - d_k_out[base + half_dim] * sin_val;
        d_k_in[base + half_dim] = d_k_out[base + half_dim] * cos_val + d_k_out[base] * sin_val;
    }
}

void rope_bf16_bw_cuda(
    const __nv_bfloat16* d_q_rope, const __nv_bfloat16* d_k_rope,
    const float* d_q_out, const float* d_k_out,
    float* d_q_in, float* d_k_in,
    int seq_len, int num_heads, int num_kv_heads,
    int head_dim, float theta,
    const int* d_pos, cudaStream_t stream)
{
    int half_dim = head_dim / 2;
    int total = seq_len * (num_heads + num_kv_heads) * half_dim;
    const int block = 256;
    unsigned int grid = (unsigned int)((total + block - 1) / block);
    rope_bf16_bw_kernel<<<grid, block, 0, stream>>>(
        d_q_rope, d_k_rope, d_q_out, d_k_out, d_q_in, d_k_in,
        seq_len, num_heads, num_kv_heads, head_dim, theta, d_pos);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 1D. Softmax backward: dS_i = P_i * (dP_i - sum(P_j * dP_j))
// ============================================================================
__global__ void softmax_bw_bf16_kernel(
    const __nv_bfloat16* __restrict__ d_prob,
    const float* __restrict__ d_out,
    float* __restrict__ d_score,
    int rows, int cols)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* prob_row = d_prob + (size_t)row * cols;
    const float* dP_row = d_out + (size_t)row * cols;
    float* dS_row = d_score + (size_t)row * cols;

    float sum_p_dp = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        sum_p_dp += __bfloat162float(prob_row[i]) * dP_row[i];
    sdata[tid] = sum_p_dp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum_p_dp_global = sdata[0];
    __syncthreads();

    for (int i = tid; i < cols; i += blockDim.x)
        dS_row[i] = __bfloat162float(prob_row[i]) * (dP_row[i] - sum_p_dp_global);
}

void softmax_bw_bf16_cuda(
    const __nv_bfloat16* d_prob, const float* d_out,
    float* d_score, int rows, int cols,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid((unsigned int)rows);
    size_t shared_mem = block * sizeof(float);
    softmax_bw_bf16_kernel<<<grid, block, shared_mem, stream>>>(
        d_prob, d_out, d_score, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// FP32 variant of softmax backward — for training attention path where P is in FP32
__global__ void softmax_bw_f32_kernel(
    const float* __restrict__ d_prob,
    const float* __restrict__ d_out,
    float* __restrict__ d_score,
    int rows, int cols)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* prob_row = d_prob + (size_t)row * cols;
    const float* dP_row = d_out + (size_t)row * cols;
    float* dS_row = d_score + (size_t)row * cols;

    float sum_p_dp = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        sum_p_dp += prob_row[i] * dP_row[i];
    sdata[tid] = sum_p_dp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum_p_dp_global = sdata[0];
    __syncthreads();

    for (int i = tid; i < cols; i += blockDim.x)
        dS_row[i] = prob_row[i] * (dP_row[i] - sum_p_dp_global);
}

void softmax_bw_f32_cuda(
    const float* d_prob, const float* d_out,
    float* d_score, int rows, int cols,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid((unsigned int)rows);
    size_t shared_mem = block * sizeof(float);
    softmax_bw_f32_kernel<<<grid, block, shared_mem, stream>>>(
        d_prob, d_out, d_score, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 1E. Causal mask: set upper triangle of attention scores to -inf
// One block per row × head
// ============================================================================
__global__ void causal_mask_f32_kernel(float* d_scores, int seq_len, int num_heads)
{
    int row = blockIdx.x;
    int h = blockIdx.y;
    if (row >= seq_len || h >= num_heads) return;
    float* head_base = d_scores + (size_t)h * seq_len * seq_len;
    for (int j = row + 1; j < seq_len; j++)
        head_base[(size_t)row * seq_len + j] = -INFINITY;
}

void causal_mask_f32_cuda(float* d_scores, int seq_len, int num_heads,
                           cudaStream_t stream)
{
    dim3 grid((unsigned int)seq_len, (unsigned int)num_heads);
    causal_mask_f32_kernel<<<grid, 1, 0, stream>>>(d_scores, seq_len, num_heads);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 1F. Fused cross-entropy loss + softcap backward (batched over positions)
// One block per batch element. Steps:
// 1. Softmax, 2. CE loss, 3. CE grad, 4. Softcap grad
// ============================================================================
__global__ void cross_entropy_softcap_bw_kernel(
    const float* __restrict__ logits_capped,
    const int* __restrict__ targets,
    float* __restrict__ d_loss,
    float* __restrict__ d_raw_grad,
    int B, int V, float softcap)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int b = blockIdx.x;
    if (b >= B) return;

    const float* row = logits_capped + (size_t)b * V;
    float* grad_row = d_raw_grad + (size_t)b * V;
    int target = targets[b];

    // Step 1: Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = tid; i < V; i += blockDim.x)
        max_val = fmaxf(max_val, row[i]);
    sdata[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Step 2: Compute exp(x - max), sum
    float sum_exp = 0.0f;
    for (int i = tid; i < V; i += blockDim.x)
        sum_exp += expf(row[i] - row_max);
    sdata[tid] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    // Step 3: Write P to grad_row (reuse as temp), track loss
    float prob_target = 0.0f;
    for (int i = tid; i < V; i += blockDim.x) {
        float p = expf(row[i] - row_max) * inv_sum;
        if (i == target) prob_target = p;
        grad_row[i] = p;
    }

    // Step 4: CE loss
    float eps = 1e-10f;
    float local_loss = (prob_target < eps) ? -logf(eps) : -logf(prob_target);
    // Reduce loss within block, then atomicAdd to global
    sdata[tid] = local_loss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_loss, sdata[0]);
    __syncthreads();

    // Step 5: CE backward + softcap backward
    // d_raw[i] = (P[i] - (i==target)) * (1 - (capped[i]/softcap)^2)
    for (int i = tid; i < V; i += blockDim.x) {
        float p = grad_row[i];
        float ce_grad = p - (i == target ? 1.0f : 0.0f);
        float cap_ratio = row[i] / softcap;
        grad_row[i] = ce_grad * (1.0f - cap_ratio * cap_ratio);
    }
}

void cross_entropy_softcap_bw_cuda(
    const float* d_logits_capped,
    const int* d_targets,
    int B, int V, float softcap,
    float* h_loss,
    float* d_raw_grad,
    cudaStream_t stream)
{
    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_loss, 0, sizeof(float), stream));

    const int block = 256;
    dim3 grid((unsigned int)B);
    size_t shared_mem = block * sizeof(float);
    cross_entropy_softcap_bw_kernel<<<grid, block, shared_mem, stream>>>(
        d_logits_capped, d_targets, d_loss, d_raw_grad, B, V, softcap);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_loss, d_loss, sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_loss));
}

// ============================================================================
// 2A. Attention score computation (per-head cuBLAS GEMM)
// NanoChat stores Q/K in interleaved layout [T, H*d], where each head's
// data has stride H*d between positions. Use per-head cuBLAS calls with
// correct leading dimension.
//
// S_h[T,T] = Q_h[T,d] @ K_h[T,d]^T  for each head h
//
// cuBLAS col-major: Q_h_col is [d,T] with lda = H*d
// CUBLAS_OP_T on Q: op(Q) = Q_h^T = T×d
// CUBLAS_OP_N on K: op(K) = K_h_col = d×T (note: we need d×T, which is K_h^T in row-major)
// ============================================================================
void attn_scores_bf16_cuda(cublasHandle_t handle,
    const __nv_bfloat16* d_q,
    const __nv_bfloat16* d_k,
    float* d_scores,
    int seq_len, int head_dim, int num_heads,
    cudaStream_t stream)
{
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float alpha = 1.0f, beta = 0.0f;
    int hx = num_heads * head_dim;  // stride between positions = H*d

    for (int h = 0; h < num_heads; h++) {
        const __nv_bfloat16* q_h = d_q + h * head_dim;
        const __nv_bfloat16* k_h = d_k + h * head_dim;
        float* s_h = d_scores + (size_t)h * seq_len * seq_len;

        // S_h[T,T] = Q_h[T,d] @ K_h[T,d]^T
        // cuBLAS: C[T,T] = op(A)[T,d] @ op(B)[d,T]
        // op(A) = Q_h^T with A = Q_h_col [d,T], OP_T gives [T,d]
        // op(B) = K_h_col with OP_N gives [d,T]
        CUBLAS_CHECK(cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            seq_len, seq_len, head_dim,
            &alpha,
            q_h, CUDA_R_16BF, hx,
            k_h, CUDA_R_16BF, hx,
            &beta,
            s_h, CUDA_R_32F, seq_len,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

// ============================================================================
// 2B. Apply attention probabilities to values (per-head cuBLAS GEMM)
// O_h[T,d] = P_h[T,T] @ V_h[T,d]  for each head h
//
// cuBLAS col-major: V_h_col is [d,T] with lda = H*d
// P_h is [T,T] col-major (lda=T) or can be used with OP_T for P_h^T
//
// For O[T,d] = P[T,T] @ V[T,d]:
// In cuBLAS: C = op(A) @ op(B)
// op(A) = P[T,T] where P_h is [T,T] col-major (lda=T), OP_N: op = [T,T]
// op(B) = V[T,d] ? V_col is [d,T] with lda=H*d. OP_T: op = V^T = [T,d]
// C[T,d] = [T,T] @ [T,d] ... but [T,T] @ [T,d] requires inner dims T and T to match ✓
// Wait: op(A) = P [T×T] with OP_N, B = V with OP_T: op(B) = V^T [T×d]
// C[T,d] = P[T,T] @ V^T[T,d]... that's sum_k P[i,k] * V^T[k,j]... V^T[k,j] = V[j,k]
// But V is [T,d], so V[j,k] for j ∈ [0,T-1], k ∈ [0,d-1]
// Then C[i,j] = sum_k P[i,k] * V[j,k] = sum_k P[i,k] * V_orig[k % T][? row-major: V[k][j]]
// This is wrong. Let me reconsider.
//
// O_h[i,d] = sum_j P_h[i,j] * V_h[j,d]
// In col-major, we want: C[d,i] = O_h[i,d] = sum_j P_h[i,j] * V_h[j,d]
// More precisely, C_col[d,i] = sum_j V_col[d,j] * P_col[j,i] where V is [d,T] and P is [T,T]
// = (V_col @ P_col)[d,i]
// So: C = V_col @ P_col
// cuBLAS: cublasGemmEx(OP_N, OP_N, d, T, T, ...) with:
//   A = V_col [d,T], lda = H*d, OP_N: A is d×T
//   B = P_col [T,T], lda = T, OP_N: B is T×T
//   C = [d,T], ldc = d
// C[d,i] = sum_j A[d,j] * B[j,i] = sum_j V[d,j] * P[j,i] = sum_j V_h[j,d] * P_h[i,j]
// That's the transpose of O (in row-major O[i,d] = sum_j P[i,j] * V[j,d])
// C_col[d,i] = O[i,d] means C_row[i,d] = O[i,d]. PERFECT!
//
// Wait: C_col[d,i] = sum_j V_col[d,j] * P_col[j,i]
// V_col[d,j] = V_h[j][d] in row-major (V_h[j,d])
// P_col[j,i] = P_h[i][j] in row-major (P_h[i,j])... hmm.
// P is stored [H,T,T] in row-major, so P_col is [T,T] where P_col[j,i] = P_h[i,j] row-major
// V is stored [T,H*d] interleaved, V_h_col is [d,T] where V_h_col[d,j] = V_h[j,d] = V[j][h*d+d]
//
// C_col[d,i] = sum_j V[j][h*d+d] * P_h[i][j]
// C_row[i,d] = C_col[d,i] = sum_j P_h[i][j] * V[j][h*d+d]
// Which is exactly O_h[i,d] = (P_h @ V_h)[i,d] ✓✓✓
//
// O_h[i,d] is the output for position i, head h, dim d, in the [T,H*d] interleaved layout.
// But our output O is [T,H*d] interleaved! So O[i][h*d+d] = C_row[i,d] = C_col[d,i]
// That means our C matrix [d,T] stores O: C_col[d,i] = O[i][h*d+d]
// In memory: C_col is [d,T] meaning elements across i (positions) are contiguous:
// C_col[0,0..T-1] = O[0..T-1][h*d+0], C_col[1,0..T-1] = O[0..T-1][h*d+1], ...
// But this is NOT the [T,H*d] interleaved layout! In interleaved, we need:
// O[0][h*d+0], O[0][h*d+1], ..., O[0][h*d+d-1], O[1][h*d+0], ...
// In C_col (d×T): O[0][h*d+0], O[1][h*d+0], ..., O[T-1][h*d+0], O[0][h*d+1], ...
// These are different layouts!
//
// So this approach doesn't directly give us the right output layout either.
// The fundamental issue is that NanoChat's interleaved layout [T,H*d] is incompatible
// with standard matrix multiply outputs.
//
// SIMPLEST SOLUTION: write a kernel that transposes from [H,T,d] to [T,H*d]
// Or even simpler: just materialize the attention output in [H,T,d] format,
// then transpose back to [T,H*d].
//
// Actually let me just keep it simple for now and use a straightforward approach:
// For each head: compute O_h[T,d] with a cuBLAS call into a [H,T,d] temp buffer,
// then transpose the entire thing to [T,H*d].
// The transpose kernel is simple and fast.
//
// Or: use a single big GEMM approach with the right data layout.
// ============================================================================

// Transpose helper: convert [num_heads, seq_len, head_dim] to [seq_len, num_heads*head_dim]
__global__ void transpose_hsd_to_sdh_kernel(
    const __nv_bfloat16* __restrict__ src,   // [H, T, d]
    __nv_bfloat16* __restrict__ dst,          // [T, H*d]
    int T, int H, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * H * d;
    if (i >= total) return;
    int d_rem = i % d;
    int h = (i / d) % H;
    int t = i / (H * d);
    dst[(size_t)t * H * d + h * d + d_rem] = src[i];
}

// ============================================================================
// 2B. P @ V with transpose to interleaved layout
// O_h[T,d] = P_h[T,T] @ V_h[T,d] for each head, then transpose [H,T,d] -> [T,H*d]
// ============================================================================
void attn_apply_pv_bf16_cuda(cublasHandle_t handle,
    const float* d_p,
    const __nv_bfloat16* d_v,
    __nv_bfloat16* d_out,
    int seq_len, int head_dim, int num_heads,
    cudaStream_t stream)
{
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t tmp_size = (size_t)num_heads * seq_len * head_dim * sizeof(__nv_bfloat16);
    __nv_bfloat16* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size));

    float alpha = 1.0f, beta = 0.0f;
    int hx = num_heads * head_dim;

    for (int h = 0; h < num_heads; h++) {
        const __nv_bfloat16* v_h = d_v + h * head_dim;
        const float* p_h = d_p + (size_t)h * seq_len * seq_len;
        __nv_bfloat16* o_h = d_tmp + (size_t)h * seq_len * head_dim;

        // O_h[T,d]_row = P_h[T,T]_row @ V_h[T,d]_row
        // In col-major: O_h_col[d,T] = V_h_col[d,T] @ P_h_col[T,T]
        // sum_j V_h[j,d] * P_h[T,j] = (P_h @ V_h)[T,d] in row-major
        CUBLAS_CHECK(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, seq_len, seq_len,
            &alpha,
            v_h, CUDA_R_16BF, hx,
            p_h, CUDA_R_32F, seq_len,
            &beta,
            o_h, CUDA_R_16BF, head_dim,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    int total = seq_len * num_heads * head_dim;
    const int block = 256;
    unsigned int grid = (unsigned int)((total + block - 1) / block);
    transpose_hsd_to_sdh_kernel<<<grid, block, 0, stream>>>(d_tmp, d_out, seq_len, num_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_tmp));
}

// ============================================================================
// 3. Single fused kernel: scale + causal mask + softmax for attention scores
// One block per (head, row), handles all softmax operations
// ============================================================================
__global__ void attn_softmax_scale_causal_kernel(
    float* d_scores,        // [H, T, T] FP32 in/out
    int seq_len, int num_heads, float scale)
{
    int row = blockIdx.x;
    int h = blockIdx.y;
    if (row >= seq_len || h >= num_heads) return;

    float* base = d_scores + (size_t)h * seq_len * seq_len + (size_t)row * seq_len;
    int tid = threadIdx.x;

    // Causal mask: this row can attend to positions 0..row
    int limit = row + 1;

    // Pass 1: find max in attended positions
    float max_val = -INFINITY;
    for (int j = tid; j < limit; j += blockDim.x)
        max_val = fmaxf(max_val, base[j] * scale);
    // Reduce across threads (warp shuffle + shared)
    // For blockDim <= 256, use shared memory reduction
    extern __shared__ float sdata_sm[];
    sdata_sm[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata_sm[tid] = fmaxf(sdata_sm[tid], sdata_sm[tid + s]);
        __syncthreads();
    }
    float row_max = sdata_sm[0];
    __syncthreads();

    // Pass 2: compute exp(x * scale - max) and sum
    float sum_exp = 0.0f;
    for (int j = tid; j < limit; j += blockDim.x)
        sum_exp += expf(base[j] * scale - row_max);
    sdata_sm[tid] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata_sm[tid] += sdata_sm[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata_sm[0];
    __syncthreads();

    // Pass 3: normalize and write
    for (int j = tid; j < limit; j += blockDim.x)
        base[j] = expf(base[j] * scale - row_max) * inv_sum;
    // Zero out masked positions
    for (int j = tid + limit; j < seq_len; j += blockDim.x)
        base[j] = 0.0f;
}

void training_attention_bf16_fwd_cuda(cublasHandle_t handle,
    const __nv_bfloat16* d_q, const __nv_bfloat16* d_k, const __nv_bfloat16* d_v,
    __nv_bfloat16* d_out, float* d_scores, float* d_p_saved,
    int seq_len, int head_dim, int num_heads, float scale,
    cudaStream_t stream)
{
    // Step 1: S = Q @ K^T * scale (we apply scale in the softmax kernel)
    attn_scores_bf16_cuda(handle, d_q, d_k, d_scores, seq_len, head_dim, num_heads, stream);

    // Step 2: scale + causal mask + softmax in-place on d_scores
    // If d_p_saved != d_scores, we also need to copy the result
    dim3 grid((unsigned int)seq_len, (unsigned int)num_heads);
    const int block = 256;
    size_t shared_mem = block * sizeof(float);
    attn_softmax_scale_causal_kernel<<<grid, block, shared_mem, stream>>>(
        d_scores, seq_len, num_heads, scale);
    CUDA_CHECK(cudaGetLastError());

    // If d_p_saved != d_scores, copy P from d_scores to d_p_saved
    if (d_p_saved != d_scores) {
        size_t total_size = (size_t)num_heads * seq_len * seq_len * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(d_p_saved, d_scores, total_size,
                                    cudaMemcpyDeviceToDevice, stream));
    }

    // Step 3: O = P @ V
    attn_apply_pv_bf16_cuda(handle, d_scores, d_v, d_out,
                             seq_len, head_dim, num_heads, stream);
}

// ============================================================================
// 4. Training attention backward
// Inputs: saved P (FP32), Q/K/V (BF16), gradient dO (FP32)
// Outputs: dQ, dK, dV gradients (all FP32, interleaved [T, H*d] layout)
// d_workspace: [num_heads, seq_len, seq_len] FP32 temp (caller-allocated)
// ============================================================================
void training_attention_bf16_bw_cuda(cublasHandle_t handle,
    const float* d_p_saved,
    const __nv_bfloat16* d_q,
    const __nv_bfloat16* d_k,
    const __nv_bfloat16* d_v,
    const float* d_out,
    float* d_q_grad,
    float* d_k_grad,
    float* d_v_grad,
    float* d_workspace,
    int seq_len, int head_dim, int num_heads,
    cudaStream_t stream)
{
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float alpha = 1.0f, beta = 0.0f;
    int hx = num_heads * head_dim;

    // Phase A: dP = dO @ V^T and dV = P^T @ dO
    for (int h = 0; h < num_heads; h++) {
        const float*   p_h = d_p_saved + (size_t)h * seq_len * seq_len;
        const __nv_bfloat16* v_h = d_v + h * head_dim;
        const float*  do_h = d_out + h * head_dim;
        float* dv_h = d_v_grad + h * head_dim;
        float* dp_h = d_workspace + (size_t)h * seq_len * seq_len;

        // dP[T,T] = dO[T,d] @ V[T,d]^T
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            seq_len, seq_len, head_dim,
            &alpha, do_h, CUDA_R_32F, hx, v_h, CUDA_R_16BF, hx,
            &beta, dp_h, CUDA_R_32F, seq_len,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // dV[T,d] = P^T @ dO : dV_col[d,T] = dO_col[d,T] @ P_col[T,T]
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, seq_len, seq_len,
            &alpha, do_h, CUDA_R_32F, hx, p_h, CUDA_R_32F, seq_len,
            &beta, dv_h, CUDA_R_32F, hx,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Phase B: dS = softmax_bw(P, dP) in-place on d_workspace
    // d_p_saved is FP32 (from training attention forward with cuBLAS scores)
    int total_rows = num_heads * seq_len;
    softmax_bw_f32_cuda(d_p_saved, d_workspace, d_workspace,
                         total_rows, seq_len, stream);

    // Phase C: dQ = dS @ K, dK = dS^T @ Q
    for (int h = 0; h < num_heads; h++) {
        const float* ds_h = d_workspace + (size_t)h * seq_len * seq_len;
        const __nv_bfloat16* k_h = d_k + h * head_dim;
        const __nv_bfloat16* q_h = d_q + h * head_dim;
        float* dq_h = d_q_grad + h * head_dim;
        float* dk_h = d_k_grad + h * head_dim;

        // dQ[T,d] = dS[T,T] @ K[T,d]
        // cuBLAS: dQ_col[d,T] = K_col[d,T] @ dS[T,T]
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, seq_len, seq_len,
            &alpha, k_h, CUDA_R_16BF, hx, ds_h, CUDA_R_32F, seq_len,
            &beta, dq_h, CUDA_R_32F, hx,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // dK[T,d] = dS^T @ Q : dK_col[d,T] = Q_col[d,T] @ dS[T,T]
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, seq_len, seq_len,
            &alpha, q_h, CUDA_R_16BF, hx, ds_h, CUDA_R_32F, seq_len,
            &beta, dk_h, CUDA_R_32F, hx,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

// ============================================================================
// 5. Mixed-type GEMM wrapper: C[M,N] = A[M,K] @ B[N,K]^T
// Supports FP32 or BF16 for A and B, FP32 or BF16 for C
// ============================================================================
void mixed_gemm_nt_cuda(cublasHandle_t handle,
    const void* A, cudaDataType typeA,
    const void* B, cudaDataType typeB,
    void* C, cudaDataType typeC,
    int M, int K, int N,
    cudaStream_t stream)
{
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, typeB, K,
        A, typeA, K,
        &beta,
        C, typeC, N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
