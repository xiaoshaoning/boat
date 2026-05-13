// swin.cu - CUDA kernels for Swin Transformer
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>
#include <stdint.h>

// =========================================================================
// Window partition: [B, H, W, C] -> [B*nh*nw, ws*ws, C]
// =========================================================================
__global__ void window_partition_kernel(
    const float* src, float* dst,
    int B, int H, int W, int C, int ws)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * W * C;
    if (idx >= total) return;

    // Reconstruct coordinates
    int c = idx % C;
    int w_pos = (idx / C) % W;
    int h_pos = (idx / (C * W)) % H;
    int b = idx / (C * W * H);

    int nh = H / ws;
    int nw = W / ws;
    int win_h = h_pos / ws;
    int win_w = w_pos / ws;
    int win_idx = (b * nh + win_h) * nw + win_w;
    int local_h = h_pos % ws;
    int local_w = w_pos % ws;

    int dst_idx = ((win_idx * ws + local_h) * ws + local_w) * C + c;
    dst[dst_idx] = src[idx];
}

extern "C" void boat_cuda_swin_window_partition_f32(
    const float* src, float* dst,
    int B, int H, int W, int C, int ws)
{
    int total = B * H * W * C;
    int block = 256;
    int grid = (total + block - 1) / block;
    window_partition_kernel<<<grid, block>>>(src, dst, B, H, W, C, ws);
}

// =========================================================================
// Window reverse: [B*nh*nw, ws*ws, C] -> [B, H, W, C]
// =========================================================================
__global__ void window_reverse_kernel(
    const float* src, float* dst,
    int B, int H, int W, int C, int ws)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * W * C;
    if (idx >= total) return;

    int c = idx % C;
    int w_pos = (idx / C) % W;
    int h_pos = (idx / (C * W)) % H;
    int b = idx / (C * W * H);

    int nh = H / ws;
    int nw = W / ws;
    int win_h = h_pos / ws;
    int win_w = w_pos / ws;
    int win_idx = (b * nh + win_h) * nw + win_w;
    int local_h = h_pos % ws;
    int local_w = w_pos % ws;

    int src_idx = ((win_idx * ws + local_h) * ws + local_w) * C + c;
    dst[idx] = src[src_idx];
}

extern "C" void boat_cuda_swin_window_reverse_f32(
    const float* src, float* dst,
    int B, int H, int W, int C, int ws)
{
    int total = B * H * W * C;
    int block = 256;
    int grid = (total + block - 1) / block;
    window_reverse_kernel<<<grid, block>>>(src, dst, B, H, W, C, ws);
}

// =========================================================================
// Cyclic shift: torch.roll on (H, W) dims
// =========================================================================
__global__ void cyclic_shift_kernel(
    const float* src, float* dst,
    int B, int H, int W, int C, int sh_h, int sh_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * W * C;
    if (idx >= total) return;

    int c = idx % C;
    int w_pos = (idx / C) % W;
    int h_pos = (idx / (C * W)) % H;
    int b = idx / (C * W * H);

    // For forward shift (SW-MSA): dest at (h,w) gets src at (h+sh, w+sh) (wrapping)
    int src_h = (h_pos + sh_h) % H;
    int src_w = (w_pos + sh_w) % W;
    int src_idx = ((b * H + src_h) * W + src_w) * C + c;
    dst[idx] = src[src_idx];
}

extern "C" void boat_cuda_swin_cyclic_shift_f32(
    const float* src, float* dst,
    int B, int H, int W, int C, int shift, int reverse)
{
    int sh = reverse ? shift : -shift;
    int sh_h = ((sh % H) + H) % H;
    int sh_w = ((sh % W) + W) % W;
    int total = B * H * W * C;
    int block = 256;
    int grid = (total + block - 1) / block;
    cyclic_shift_kernel<<<grid, block>>>(src, dst, B, H, W, C, sh_h, sh_w);
}

// =========================================================================
// Batched matmul with scale: scores = Q @ K^T * scale, where
// Q: [batch, N, D], K: [batch, D, N], scores: [batch, N, N]
// =========================================================================
__global__ void batched_matmul_scale_kernel(
    const float* Q, const float* K, float* scores,
    int batch, int N, int D, float scale)
{
    // Each block handles one output position in one batch
    int b = blockIdx.x;
    int ni = blockIdx.y;
    int nj = threadIdx.x;

    if (b >= batch || ni >= N || nj >= N) return;

    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        sum += Q[b * N * D + ni * D + d] * K[b * D * N + d * N + nj];
    }
    scores[b * N * N + ni * N + nj] = sum * scale;
}

extern "C" void boat_cuda_window_attn_scores_f32(
    const float* Q, const float* K, float* scores,
    int batch, int N, int D, float scale)
{
    dim3 grid(batch, N);
    int block = N;
    batched_matmul_scale_kernel<<<grid, block>>>(Q, K, scores, batch, N, D, scale);
}

// =========================================================================
// Add relative position bias to attention scores
// scores: [num_windows, num_heads, N, N]
// bias_tbl: [(2*ws-1)^2, num_heads]
// rpi: [N, N] int64
// =========================================================================
__global__ void add_rel_pos_bias_kernel(
    float* scores, const float* bias_tbl, const int64_t* rpi,
    int num_windows, int num_heads, int N)
{
    int wi = blockIdx.x;
    int h = blockIdx.y;
    int ni = blockIdx.z;
    int nj = threadIdx.x;

    if (wi >= num_windows || h >= num_heads || ni >= N || nj >= N) return;

    int bias_idx = (int)rpi[ni * N + nj];
    int s_idx = ((wi * num_heads + h) * N + ni) * N + nj;
    scores[s_idx] += bias_tbl[bias_idx * num_heads + h];
}

extern "C" void boat_cuda_add_rel_pos_bias_f32(
    float* scores, const float* bias_tbl,
    const int64_t* rpi, int num_windows, int num_heads, int N)
{
    dim3 grid(num_windows, num_heads, N);
    int block = N;
    add_rel_pos_bias_kernel<<<grid, block>>>(scores, bias_tbl, rpi, num_windows, num_heads, N);
}

// =========================================================================
// Batched attn @ V: out = attn @ V
// attn: [batch, N, N], V: [batch, N, D], out: [batch, N, D]
// =========================================================================
__global__ void attn_apply_kernel(
    const float* attn, const float* V, float* out,
    int batch, int N, int D)
{
    int b = blockIdx.x;
    int ni = blockIdx.y;
    int d = threadIdx.x;

    if (b >= batch || ni >= N || d >= D) return;

    float sum = 0.0f;
    for (int nj = 0; nj < N; nj++) {
        sum += attn[b * N * N + ni * N + nj] * V[b * N * D + nj * D + d];
    }
    out[b * N * D + ni * D + d] = sum;
}

extern "C" void boat_cuda_window_attn_apply_f32(
    const float* attn, const float* V, float* out,
    int batch, int N, int D)
{
    dim3 grid(batch, N);
    int block = D;
    attn_apply_kernel<<<grid, block>>>(attn, V, out, batch, N, D);
}

// =========================================================================
// Patch embed: Conv2d with kernel=ps, stride=ps, no padding
// input: [N, C, H, W], weight: [embed_dim, C, ps, ps], bias: [embed_dim]
// output: [N, Hp, Wp, embed_dim] where Hp=H/ps, Wp=W/ps
// =========================================================================
__global__ void patch_embed_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C, int H, int W, int embed_dim, int ps, int Hp, int Wp)
{
    int oc = blockIdx.x;
    int hi = blockIdx.y;
    int wi = blockIdx.z;
    int n = threadIdx.x;

    if (n >= N || hi >= Hp || wi >= Wp || oc >= embed_dim) return;

    float sum = bias ? bias[oc] : 0.0f;
    for (int ic = 0; ic < C; ic++) {
        for (int i = 0; i < ps; i++) {
            for (int j = 0; j < ps; j++) {
                int in_idx = ((n * C + ic) * H + hi * ps + i) * W + wi * ps + j;
                int w_idx = ((oc * C + ic) * ps + i) * ps + j;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }
    int out_idx = ((n * Hp + hi) * Wp + wi) * embed_dim + oc;
    output[out_idx] = sum;
}

extern "C" void boat_cuda_swin_patch_embed_f32(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C, int H, int W, int embed_dim, int ps)
{
    int Hp = H / ps, Wp = W / ps;
    int block = N;
    if (block > 256) block = 256;
    dim3 grid(embed_dim, Hp, Wp);
    patch_embed_kernel<<<grid, block>>>(input, weight, bias, output,
        N, C, H, W, embed_dim, ps, Hp, Wp);
}

// =========================================================================
// Generic batched matmul with scale (for decoder attention)
// C[b,m,n] = sum_k A[b,m,k] * B[b,k,n] * scale
// =========================================================================
__global__ void batched_matmul_generic_kernel(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K, float scale)
{
    int b = blockIdx.x;
    int m = blockIdx.y;
    int n = threadIdx.x;

    if (b >= batch || m >= M || n >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[b * M * K + m * K + k] * B[b * K * N + k * N + n];
    }
    C[b * M * N + m * N + n] = sum * scale;
}

extern "C" void boat_cuda_batched_matmul_scale_f32(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K, float scale)
{
    dim3 grid(batch, M);
    int block = N;
    batched_matmul_generic_kernel<<<grid, block>>>(A, B, C, batch, M, N, K, scale);
}

// =========================================================================
// Add causal mask: -inf for future positions
// scores: [batch, T, L], step_offset: position offset for KV cache
// =========================================================================
__global__ void add_causal_mask_kernel(
    float* scores, int batch, int T, int L, int step_offset)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int l = threadIdx.x;

    if (b >= batch || t >= T || l >= L) return;

    int global_t = step_offset + t;
    if (l > global_t) {
        scores[b * T * L + t * L + l] = -INFINITY;
    }
}

extern "C" void boat_cuda_add_causal_mask_f32(
    float* scores, int batch, int T, int L, int step_offset)
{
    dim3 grid(batch, T);
    int block = L;
    add_causal_mask_kernel<<<grid, block>>>(scores, batch, T, L, step_offset);
}

// =========================================================================
// KV cache append: copy new K,V into the rolling cache
// src: [B, H, T, head_dim], dst: [B, H, cache_max, head_dim]
// step: position to start writing at
// =========================================================================
__global__ void kv_cache_append_kernel(
    const float* src, float* dst,
    int B, int H, int T, int head_dim,
    int cache_max, int step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * T * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int t = (idx / head_dim) % T;
    int h = (idx / (T * head_dim)) % H;
    int b = idx / (H * T * head_dim);

    int src_idx = ((b * H + h) * T + t) * head_dim + d;
    int dst_idx = ((b * H + h) * cache_max + step + t) * head_dim + d;
    dst[dst_idx] = src[src_idx];
}

extern "C" void boat_cuda_kv_cache_append_f32(
    const float* src, float* dst,
    int B, int H, int T, int head_dim,
    int cache_max, int step)
{
    int total = B * H * T * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    kv_cache_append_kernel<<<grid, block>>>(src, dst, B, H, T, head_dim, cache_max, step);
}

// =========================================================================
// KV cache extract: copy from strided cache to contiguous output tensor
// cache: [B, H, cache_max, head_dim], dst: [B, H, L, head_dim]
// L = current cached length
// =========================================================================
__global__ void kv_cache_extract_kernel(
    const float* cache, float* dst,
    int B, int H, int L, int head_dim,
    int cache_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * L * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int t = (idx / head_dim) % L;
    int h = (idx / (L * head_dim)) % H;
    int b = idx / (H * L * head_dim);

    int src_idx = ((b * H + h) * cache_max + t) * head_dim + d;
    dst[idx] = cache[src_idx];
}

extern "C" void boat_cuda_kv_cache_extract_f32(
    const float* cache, float* dst,
    int B, int H, int L, int head_dim,
    int cache_max)
{
    int total = B * H * L * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    kv_cache_extract_kernel<<<grid, block>>>(cache, dst, B, H, L, head_dim, cache_max);
}
