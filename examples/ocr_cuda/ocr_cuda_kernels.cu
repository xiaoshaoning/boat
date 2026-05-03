// ocr_cuda_kernels.cu - CUDA kernel implementations for OCR operations
#include "ocr_cuda_kernels.cuh"
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
    // C_col[N,M] = W_col[K,N]^T * A_col[K,M]
    // W:[N,K] row => col:[K,N] ld=K, op_T => [N,K]
    // A:[M,K] row => col:[K,M] ld=K, op_N => [K,M]
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K, &alpha, W, K, A, K, &beta, C, N));
}

// ----------------------------------------------------------------------------
// Helper: add broadcast bias
// ----------------------------------------------------------------------------
__global__ void add_bias_kernel(float* out, const float* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    out[idx] += bias[idx % N];
}

// ============================================================================
// Patch embedding: conv2d with temporal merge
// Weight: [C_out, in_ch*temporal, patch, patch], Input: [3, H, W]
// ============================================================================
__global__ void patch_embed_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int H, int W, int patch_size, int C_out) {
    int oc = blockIdx.y;
    int oy = blockIdx.x / (W / patch_size);
    int ox = blockIdx.x % (W / patch_size);
    int oh = H / patch_size;
    int ow = W / patch_size;
    float sum = bias[oc];
    int in_ch = 3, temp = 2;
    for (int ic = 0; ic < in_ch; ic++)
        for (int t = 0; t < temp; t++)
            for (int ky = 0; ky < patch_size; ky++)
                for (int kx = 0; kx < patch_size; kx++) {
                    int iy = oy * patch_size + ky;
                    int ix = ox * patch_size + kx;
                    if (iy < H && ix < W) {
                        int widx = ((oc * in_ch + ic) * temp + t) * patch_size * patch_size
                                 + ky * patch_size + kx;
                        sum += weight[widx] * input[ic * H * W + iy * W + ix];
                    }
                }
    output[oc * oh * ow + oy * ow + ox] = sum;
}

void patch_embed_cuda(const float* d_input, const float* d_weight,
                       const float* d_bias, float* d_output,
                       int H, int W, int C_out, int patch_size,
                       cudaStream_t stream) {
    int oh = H / patch_size, ow = W / patch_size;
    dim3 grid((unsigned int)(oh * ow), (unsigned int)C_out);
    patch_embed_kernel<<<grid, 1, 0, stream>>>(d_input, d_weight, d_bias,
                                                 d_output, H, W, patch_size, C_out);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Patch reorder: channel-first [C,ph,pw] -> spatial-block [ph*pw, C]
// ============================================================================
__global__ void patch_reorder_kernel(const float* __restrict__ inp,
                                      float* __restrict__ out,
                                      int C, int ph, int pw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ph * pw * C;
    if (idx >= total) return;
    int c = idx % C;
    int p = idx / C;
    int r = p / pw, cw = p % pw;
    int hw2 = pw / 2;
    int sb = ((r / 2) * hw2 + (cw / 2)) * 4 + (r % 2) * 2 + (cw % 2);
    out[sb * C + c] = inp[c * ph * pw + r * pw + cw];
}

void patch_reorder_cuda(const float* d_input, float* d_output,
                         int C, int patch_h, int patch_w,
                         cudaStream_t stream) {
    int total = patch_h * patch_w * C;
    const int block = 256;
    unsigned int grid = (unsigned int)((total + block - 1) / block);
    patch_reorder_kernel<<<grid, block, 0, stream>>>(d_input, d_output,
                                                       C, patch_h, patch_w);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 2D RoPE precompute (CogViT)
// ============================================================================
__global__ void rope_2d_compute_kernel(float* d_cos, float* d_sin,
                                        int ph, int pw, int hdim, float theta) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int N = ph * pw;
    if (p >= N) return;
    int r = p / pw, cw = p % pw, hw2 = pw / 2;
    int sb = ((r / 2) * hw2 + (cw / 2)) * 4 + (r % 2) * 2 + (cw % 2);
    for (int i = 0; i < 16; i++) {
        float inv = 1.0f / powf(theta, (2.0f * i) / 32.0f);
        float th = r * inv, tw = cw * inv;
        float ch = cosf(th), sh = sinf(th), cw_v = cosf(tw), sw = sinf(tw);
        d_cos[sb * 64 + i] = ch;       d_cos[sb * 64 + 32 + i] = ch;
        d_cos[sb * 64 + 16 + i] = cw_v; d_cos[sb * 64 + 48 + i] = cw_v;
        d_sin[sb * 64 + i] = sh;       d_sin[sb * 64 + 32 + i] = sh;
        d_sin[sb * 64 + 16 + i] = sw;  d_sin[sb * 64 + 48 + i] = sw;
    }
}

void rope_2d_compute_cuda(float* d_cos, float* d_sin,
                           int patch_h, int patch_w, int head_dim,
                           float theta_base, cudaStream_t stream) {
    int N = patch_h * patch_w;
    const int block = 128;
    unsigned int grid = (unsigned int)((N + block - 1) / block);
    rope_2d_compute_kernel<<<grid, block, 0, stream>>>(d_cos, d_sin, patch_h, patch_w,
                                                         head_dim, theta_base);
    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Apply 2D RoPE (per-position-per-head 2D grid)
// Rotates pairs (d, d+half) where half = hdim/2, matching CPU's half-split RoPE.
// blockDim = half (= hdim/2 threads), each thread handles one pair.
// ----------------------------------------------------------------------------
__global__ void apply_rope_2d_kernel(float* q, float* k,
                                      int N, int nh, int hdim, int stride_qk,
                                      const float* cos, const float* sin) {
    int p = blockIdx.x, h = blockIdx.y;
    int d = threadIdx.x;
    int half = hdim / 2;
    if (p >= N || h >= nh || d >= half) return;
    int off = p * stride_qk + h * hdim;
    float q0 = q[off + d], q1 = q[off + d + half];
    float c = cos[p * hdim + d], s = sin[p * hdim + d];
    q[off + d] = q0 * c - q1 * s;
    q[off + d + half] = q1 * c + q0 * s;
    if (k) {
        float k0 = k[off + d], k1 = k[off + d + half];
        k[off + d] = k0 * c - k1 * s;
        k[off + d + half] = k1 * c + k0 * s;
    }
}

void apply_rope_2d_cuda(float* d_q, float* d_k, int N, int num_heads,
                          int head_dim, int stride_qk,
                          const float* d_cos, const float* d_sin,
                          cudaStream_t stream) {
    dim3 grid((unsigned int)N, (unsigned int)num_heads);
    apply_rope_2d_kernel<<<grid, head_dim / 2, 0, stream>>>(d_q, d_k, N, num_heads,
                                                              head_dim, stride_qk, d_cos, d_sin);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// M-RoPE for GLM: 3D rotational embeddings
// head_dim=128, sections: [16t, 24h, 24w] × 2 repeats = 128 dims
// 64 threads, each handles one non-overlapping pair (di, di+1)
// Pairs per section per repeat: T=8, H=12, W=12 (32 pairs/repeat, 64 total)
// ============================================================================
__global__ void apply_mrope_kernel(float* q, float* k,
                                    int seq_len, int nh, int nkv, int hdim,
                                    float theta, const int* pt, const int* ph, const int* pw) {
    int p = blockIdx.x;
    if (p >= seq_len) return;
    int t = threadIdx.x; // pair index 0..63
    if (t >= 64) return;

    // Map pair index to (repeat, section, start_dim)
    int repeat = t / 32;  // 0 or 1
    int within = t % 32;  // 0..31 within the repeat
    int di;  // start of the dimension pair
    const int* pos;  // which position array to use
    if (within < 8) {
        di = repeat * 64 + 2 * within;        // dims 0,2..14 or 64,66..78
        pos = pt;
    } else if (within < 20) {
        di = repeat * 64 + 16 + 2 * (within - 8); // dims 16,18..38 or 80,82..102
        pos = ph;
    } else {
        di = repeat * 64 + 40 + 2 * (within - 20); // dims 40,42..62 or 104,106..126
        pos = pw;
    }

    float fr = powf(theta, -(float)di / (float)hdim);
    float c = cosf(pos[p] * fr), s = sinf(pos[p] * fr);

    // Process Q (all heads)
    for (int h = 0; h < nh; h++) {
        int off = (p * nh + h) * hdim + di;
        float x0 = q[off], x1 = q[off + 1];
        q[off] = x0 * c - x1 * s; q[off + 1] = x1 * c + x0 * s;
    }

    // Process K (num_kv_heads heads)
    if (k) for (int h = 0; h < nkv; h++) {
        int off = (p * nkv + h) * hdim + di;
        float x0 = k[off], x1 = k[off + 1];
        k[off] = x0 * c - x1 * s; k[off + 1] = x1 * c + x0 * s;
    }
}

void apply_mrope_cuda(float* d_q, float* d_k,
                       int seq_len, int num_heads, int num_kv_heads,
                       int head_dim, float theta,
                       const int* d_pos_t, const int* d_pos_h, const int* d_pos_w,
                       cudaStream_t stream) {
    dim3 grid((unsigned int)seq_len, 1);
    int block = head_dim / 2; // 64 threads
    apply_mrope_kernel<<<grid, block, 0, stream>>>(d_q, d_k, seq_len, num_heads,
                                                     num_kv_heads, head_dim, theta,
                                                     d_pos_t, d_pos_h, d_pos_w);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Standard 1D RoPE for GLM decoder (matches CPU apply_rope_glm)
// Rotates consecutive pairs (d, d+1) at each position.
// ============================================================================
__global__ void glm_rope_kernel(float* q, float* k,
                                 int seq_len, int nh, int nkv, int hdim,
                                 float theta, const int* d_pos) {
    int p = blockIdx.x;
    if (p >= seq_len) return;
    int pos = d_pos[p];
    int t = threadIdx.x;
    int pairs = hdim / 2;

    for (int h = 0; h < nh; h++) {
        int off = (p * nh + h) * hdim;
        for (int i = t; i < pairs; i += blockDim.x) {
            int d = i * 2;
            float freq = powf(theta, -(float)d / (float)hdim);
            float c = cosf(pos * freq), s = sinf(pos * freq);
            float x0 = q[off + d], x1 = q[off + d + 1];
            q[off + d] = x0 * c - x1 * s;
            q[off + d + 1] = x1 * c + x0 * s;
        }
    }

    if (k) for (int h = 0; h < nkv; h++) {
        int off = (p * nkv + h) * hdim;
        for (int i = t; i < pairs; i += blockDim.x) {
            int d = i * 2;
            float freq = powf(theta, -(float)d / (float)hdim);
            float c = cosf(pos * freq), s = sinf(pos * freq);
            float x0 = k[off + d], x1 = k[off + d + 1];
            k[off + d] = x0 * c - x1 * s;
            k[off + d + 1] = x1 * c + x0 * s;
        }
    }
}

void glm_rope_cuda(float* d_q, float* d_k,
                    int seq_len, int num_heads, int num_kv_heads,
                    int head_dim, float theta,
                    const int* d_pos,
                    cudaStream_t stream) {
    int block = head_dim / 2;
    if (block < 32) block = 32;
    glm_rope_kernel<<<(unsigned int)seq_len, block, 0, stream>>>(
        d_q, d_k, seq_len, num_heads, num_kv_heads, head_dim, theta, d_pos);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Downsample: 4 consecutive patches -> 1 output token
// ============================================================================
__global__ void downsample_kernel(const float* __restrict__ inp,
                                   const float* __restrict__ w,
                                   const float* __restrict__ b,
                                   float* __restrict__ out,
                                   int np, int hid, int ohid) {
    int g = blockIdx.x, oc = blockIdx.y;
    float sum = b[oc];
    for (int ic = 0; ic < hid; ic++)
        for (int ky = 0; ky < 2; ky++)
            for (int kx = 0; kx < 2; kx++)
                sum += inp[(g * 4 + ky * 2 + kx) * hid + ic]
                     * w[(oc * hid + ic) * 4 + ky * 2 + kx];
    out[g * ohid + oc] = sum;
}

void downsample_cuda(const float* d_input, const float* d_weight,
                      const float* d_bias, float* d_output,
                      int num_patches, int hidden_size, int out_hidden_size,
                      cudaStream_t stream) {
    int ng = num_patches / 4;
    dim3 grid((unsigned int)ng, (unsigned int)out_hidden_size);
    downsample_kernel<<<grid, 1, 0, stream>>>(d_input, d_weight, d_bias, d_output,
                                                num_patches, hidden_size, out_hidden_size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// SiLU gate activation for interleaved layout (CogViT blocks)
// Input: [N, 2*ff_dim] interleaved row-major: [gate_0..gate_{ff-1}, up_0..up_{ff-1}] per row
// Output: d[0..N*ff) = silu(gate) * up (contiguous for downstream matmul)
// ============================================================================
__global__ void silu_gate_kernel(float* d, int N, int ff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * ff) return;
    int row = idx / ff;
    int col = idx % ff;
    float g = d[row * 2 * ff + col];
    float u = d[row * 2 * ff + ff + col];
    d[idx] = (g / (1.0f + expf(-g))) * u;
}

// ============================================================================
// SiLU gate activation for contiguous layout (merger)
// Input: gate at d[0..M*ff), up at d[M*ff..2M*ff) (contiguous blocks)
// Output: d[0..M*ff) = silu(gate) * up (in-place in the gate block)
// ============================================================================
__global__ void merger_silu_gate_kernel(float* d, int M, int ff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * ff) return;
    float g = d[idx];
    float u = d[M * ff + idx];
    d[idx] = (g / (1.0f + expf(-g))) * u;
}

void silu_gate_cuda(float* d_data, int N, int ff_dim, cudaStream_t stream) {
    int total = N * ff_dim;
    const int block = 256;
    unsigned int grid = (unsigned int)((total + block - 1) / block);
    silu_gate_kernel<<<grid, block, 0, stream>>>(d_data, N, ff_dim);
    CUDA_CHECK(cudaGetLastError());
}

void merger_silu_gate_cuda(float* d_data, int M, int ff_dim, cudaStream_t stream) {
    int total = M * ff_dim;
    const int block = 256;
    unsigned int grid = (unsigned int)((total + block - 1) / block);
    merger_silu_gate_kernel<<<grid, block, 0, stream>>>(d_data, M, ff_dim);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// LayerNorm + GELU (fused for merger)
// y = GELU(LayerNorm(x, weight, bias))
// ============================================================================
__global__ void layernorm_gelu_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       const float* __restrict__ w,
                                       const float* __restrict__ b,
                                       int M, int D, float eps) {
    extern __shared__ float sdat[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const float* rx = x + row * D;
    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) sum += rx[i];
    sdat[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdat[tid] += sdat[tid + s];
        __syncthreads();
    }
    float mean = sdat[0] / D;
    __syncthreads();
    float vs = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) { float d = rx[i] - mean; vs += d * d; }
    sdat[tid] = vs;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdat[tid] += sdat[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sdat[0] / D + eps);
    __syncthreads();
    for (int i = tid; i < D; i += blockDim.x) {
        float v = (rx[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);
        y[row * D + i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

void layernorm_gelu_cuda(const float* d_x, float* d_y,
                          const float* d_weight, const float* d_bias,
                          int M, int D, float eps,
                          cudaStream_t stream) {
    const int block = 256;
    dim3 grid((unsigned int)M);
    size_t smem = block * sizeof(float);
    layernorm_gelu_kernel<<<grid, block, smem, stream>>>(d_x, d_y, d_weight, d_bias, M, D, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Batched attention: QK^T -> softmax -> PV (used in CogViT)
// ============================================================================
__global__ void batched_attn_scores_kernel(const float* __restrict__ q,
                                            const float* __restrict__ k,
                                            float* __restrict__ scores,
                                            int N, int nh, int hd, int stride_qk, float scale) {
    int h = blockIdx.x, i = blockIdx.y;
    if (h >= nh || i >= N) return;
    int base = (h * N + i) * N;
    int qi = i * stride_qk + h * hd;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float sum = 0.0f;
        int kj = j * stride_qk + h * hd;
        for (int d = 0; d < hd; d++)
            sum += q[qi + d] * k[kj + d];
        scores[base + j] = sum * scale;
    }
}

__global__ void batched_attn_softmax_kernel(float* scores, int N, int nh) {
    int h = blockIdx.x, i = blockIdx.y;
    if (h >= nh || i >= N) return;
    int base = (h * N + i) * N;
    float mx = scores[base];
    for (int j = 1; j < N; j++) if (scores[base + j] > mx) mx = scores[base + j];
    float sum = 0.0f;
    for (int j = 0; j < N; j++) { scores[base + j] = expf(scores[base + j] - mx); sum += scores[base + j]; }
    float inv = 1.0f / sum;
    for (int j = 0; j < N; j++) scores[base + j] *= inv;
}

__global__ void batched_attn_ctx_kernel(const float* __restrict__ scores,
                                          const float* __restrict__ v,
                                          float* __restrict__ ctx,
                                          int N, int nh, int hd, int stride_v) {
    int h = blockIdx.x, i = blockIdx.y, d = threadIdx.x;
    if (h >= nh || i >= N || d >= hd) return;
    int out_stride = nh * hd;
    float sum = 0.0f;
    for (int j = 0; j < N; j++)
        sum += scores[(h * N + i) * N + j] * v[j * stride_v + h * hd + d];
    ctx[i * out_stride + h * hd + d] = sum;
}

void batched_attention_cuda(cublasHandle_t handle,
                             const float* d_q, const float* d_k, const float* d_v,
                             float* d_context, int N, int num_heads, int head_dim,
                             int stride_qkv, float scale, cudaStream_t stream) {
    float* d_scores;
    size_t sz = (size_t)num_heads * N * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_scores, sz));
    dim3 g2d((unsigned int)num_heads, (unsigned int)N);
    batched_attn_scores_kernel<<<g2d, 256, 0, stream>>>(d_q, d_k, d_scores, N, num_heads, head_dim, stride_qkv, scale);
    CUDA_CHECK(cudaGetLastError());
    batched_attn_softmax_kernel<<<g2d, 1, 0, stream>>>(d_scores, N, num_heads);
    CUDA_CHECK(cudaGetLastError());
    batched_attn_ctx_kernel<<<g2d, head_dim, 0, stream>>>(d_scores, d_v, d_context, N, num_heads, head_dim, stride_qkv);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_scores));
}

// ============================================================================
// NaN scan utility: checks buffer for NaN/Inf, prints first occurrence
// ============================================================================
void nan_scan_cuda(const float* d_buf, int N, const char* label, cudaStream_t stream) {
    (void)stream;
    if (N <= 0) return;
    float* h_buf = (float*)malloc((size_t)N * sizeof(float));
    cudaMemcpy(h_buf, d_buf, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        if (isnan(h_buf[i])) {
            fprintf(stderr, "  [NAN_SCAN] %s: first NaN at index %d (value=%f)\n", label, i, h_buf[i]);
            free(h_buf);
            return;
        }
    }
    fprintf(stderr, "  [NAN_SCAN] %s: all %d elements clean\n", label, N);
    free(h_buf);
}

