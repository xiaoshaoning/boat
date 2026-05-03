// cogvit_cuda.cu - CUDA-accelerated CogViT vision encoder
#include "cogvit_cuda.cuh"
#include "ocr_cuda_kernels.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

extern "C" cublasHandle_t boat_cuda_get_cublas_handle(void);

// Forward declarations for kernels defined in ocr_cuda_kernels.cu (separable compilation)
extern __global__ void add_bias_kernel(float* out, const float* bias, int M, int N);
// Forward declaration for kernel defined later in this file
__global__ void add_residual_kernel(float* y, const float* x, int N);

// Helper: load a weight tensor from safetensors and transfer to GPU
static float* load_weight_to_gpu(safetensors_t* st, const char* name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) { fprintf(stderr, "[CogViT-CUDA] Weight not found: %s\n", name); return NULL; }
    boat_tensor_t* t = safetensors_load_tensor(st, idx, 0);
    if (!t) { fprintf(stderr, "[CogViT-CUDA] Failed to load: %s\n", name); return NULL; }
    size_t nbytes = boat_tensor_nbytes(t);
    float* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, nbytes));
    CUDA_CHECK(cudaMemcpy(d_ptr, boat_tensor_data(t), nbytes, cudaMemcpyHostToDevice));
    boat_tensor_unref(t);
    return d_ptr;
}

// ----------------------------------------------------------------------------
// Per-head RMSNorm for interleaved Q/K layout [N, D] = [N, num_heads * head_dim]
// Each (token, head) pair is normalized independently across head_dim.
// Grid: dim3(N, num_heads), Block: head_dim/2 threads (one per dimension pair).
// Shared memory: blockDim.x * sizeof(float).
// Uses grid-stride loop over tokens to handle N > gridDim.x.
// ----------------------------------------------------------------------------
__global__ void per_head_rmsnorm_kernel(float* data, const float* weight,
                                         int N, int tot_dim, int hd, float eps) {
    int t0 = blockIdx.x;
    int h = blockIdx.y;
    if (h >= tot_dim / hd) return;
    extern __shared__ float sdat[];
    int tid = threadIdx.x;
    int pairs = hd / 2;

    for (int t = t0; t < N; t += gridDim.x) {
        float* row = data + t * tot_dim + h * hd;

        float sum = 0.0f;
        for (int p = tid; p < pairs; p += blockDim.x) {
            float x0 = row[p * 2], x1 = row[p * 2 + 1];
            sum += x0 * x0 + x1 * x1;
        }
        sdat[tid] = sum;
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            __syncthreads();
            if (tid < s) sdat[tid] += sdat[tid + s];
        }
        __syncthreads();
        float rms = rsqrtf(sdat[0] / hd + eps);

        for (int p = tid; p < pairs; p += blockDim.x) {
            int d0 = p * 2, d1 = p * 2 + 1;
            row[d0] = row[d0] * rms * weight[d0];
            row[d1] = row[d1] * rms * weight[d1];
        }
        __syncthreads();
    }
}

// ============================================================================
// CogViT block forward on GPU
// All buffers are device pointers
// ============================================================================
static void cogvit_block_gpu(cublasHandle_t handle,
                              float* d_x,           // [N, D] input/output
                              int N, int D, int num_heads, int head_dim,
                              const float* d_norm1_w,
                              const float* d_qkv_w, const float* d_qkv_b,
                              const float* d_qn_w, const float* d_kn_w,
                              const float* d_proj_w, const float* d_proj_b,
                              const float* d_norm2_w,
                              const float* d_gw, const float* d_gb,
                              const float* d_uw, const float* d_ub,
                              const float* d_dw, const float* d_db,
                              const float* d_cos, const float* d_sin,
                              float* d_tmp,  // working buffer [N * (3*D + ff_dim + max(N,head_dim))]
                              cudaStream_t stream) {
    int q_size = num_heads * head_dim; // D
    int kv_size = num_heads * head_dim; // CogViT has equal Q/KV heads
    int ff_dim = 4096;
    int groups = num_heads / num_heads; // = 1, CogViT uses full attention

    // Buffer layout in d_tmp:
    float* d_normed = d_tmp;
    float* d_qkv    = d_tmp + N * D;
    float* d_q      = d_qkv;
    float* d_k      = d_qkv + N * D;
    float* d_v      = d_qkv + N * 2 * D;
    float* d_attn_ctx = d_qkv + N * 3 * D; // reuse after QKV is consumed
    // For FFN: reuse d_qkv buffer after attention
    float* d_ff = d_qkv; // reuse
    float* d_mlp_out = d_qkv + N * ff_dim; // reuse

    // ---- Step 1: Pre-attention RMSNorm ----
    boat_cuda_rmsnorm_forward_f32(d_x, d_norm1_w, d_normed, N, D, 1e-5f);

    // ---- Step 2: QKV projection ----
    // QKV = normed @ qkv_w^T + qkv_b, qkv_w is [3*D, D]
    matmul_bt_cuda(handle, d_normed, d_qkv_w, d_qkv, N, D, 3 * D);
    // Add bias
    {
        int total = N * 3 * D;
        const int block = 256;
        unsigned int grid = (unsigned int)((total + block - 1) / block);
        add_bias_kernel<<<grid, block, 0, stream>>>(d_qkv, d_qkv_b, N, 3 * D);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Step 3: Per-head QK-Norm (interleaved QKV layout, stride=3*D) ----
    {
        int pairs = head_dim / 2; // 32
        int shmem = pairs * sizeof(float);
        int stride_qkv = 3 * D;
        dim3 qk_grid((unsigned int)N, (unsigned int)num_heads);
        // Q: at d_qkv offset 0, stride = 3*D between tokens
        per_head_rmsnorm_kernel<<<qk_grid, pairs, shmem, stream>>>(
            d_qkv, d_qn_w, N, stride_qkv, head_dim, 1e-5f);
        CUDA_CHECK(cudaGetLastError());
        // K: at d_qkv offset D, stride = 3*D between tokens
        per_head_rmsnorm_kernel<<<qk_grid, pairs, shmem, stream>>>(
            d_qkv + D, d_kn_w, N, stride_qkv, head_dim, 1e-5f);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Step 4: 2D RoPE (interleaved QKV stride) ----
    apply_rope_2d_cuda(d_qkv, d_qkv + D, N, num_heads, head_dim, 3 * D, d_cos, d_sin, stream);

    // ---- Step 5: Attention (interleaved QKV stride) ----
    float scale = 1.0f / sqrtf((float)head_dim);
    batched_attention_cuda(handle, d_qkv, d_qkv + D, d_qkv + 2 * D,
                           d_attn_ctx, N, num_heads, head_dim, 3 * D, scale, stream);

    // ---- Step 6: Output projection ----
    matmul_bt_cuda(handle, d_attn_ctx, d_proj_w, d_ff, N, D, D);
    {
        int total = N * D;
        const int block = 256;
        unsigned int grid = (unsigned int)((total + block - 1) / block);
        add_bias_kernel<<<grid, block, 0, stream>>>(d_ff, d_proj_b, N, D);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Step 7: Post-self-attention RMSNorm + residual ----
    // attn_out = RMSNorm(attn_out) + x (residual)
    // Use d_normed as temp for the RMSNorm'd output
    boat_cuda_rmsnorm_forward_f32(d_ff, d_norm1_w, d_normed, N, D, 1e-5f);
    // Wait - the post-self-attn norm has its own weight. Let me fix this.
    // Actually looking at the CPU code:
    // post_self_attn_layernorm_weight is the weight for post-self-attention.
    // But in the CPU CogViT code (cogvit.c), it uses norm1_w for pre-attention
    // and norm_2_w for pre-MLP. There is no post-self-attn norm in CogViT!
    // The GLM has post-self-attn norm, but CogViT doesn't.
    // In CogViT: residual -> norm1 -> attn -> residual add -> norm2 -> MLP -> residual add
    // So d_norm1_w is used only for pre-attention norm.
    // The post-self-attention RMSNorm from GLM is not present in CogViT.
    // (This comment is just for clarity - the code is correct as-is)
    float* d_attn_out = d_normed;
    // Residual: x[i] = x[i] + attn_out[i]
    {
        int total = N * D;
        const int block = 256;
        unsigned int grid = (unsigned int)((total + block - 1) / block);
        add_residual_kernel<<<grid, block, 0, stream>>>(d_x, d_ff, N * D);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Step 8: Pre-MLP RMSNorm ----
    boat_cuda_rmsnorm_forward_f32(d_x, d_norm2_w, d_normed, N, D, 1e-5f);

    // ---- Step 9: SiLU FFN ----
    // gate_up = normed @ gate_up_w^T + gate_up_b  [N, 2*ff_dim]
    matmul_bt_cuda(handle, d_normed, d_gw, d_ff, N, D, 2 * ff_dim);
    {
        int total = N * 2 * ff_dim;
        const int block = 256;
        unsigned int grid = (unsigned int)((total + block - 1) / block);
        add_bias_kernel<<<grid, block, 0, stream>>>(d_ff, d_gb, N, 2 * ff_dim);
        CUDA_CHECK(cudaGetLastError());
    }

    // SiLU gate: first N*ff_dim = silu(gate) * up
    silu_gate_cuda(d_ff, N, ff_dim, stream);

    // down out = gate(silu*up) @ down_w^T (d_ff now has N*ff_dim after silu_gate)
    matmul_bt_cuda(handle, d_ff, d_dw, d_mlp_out, N, ff_dim, D);
    {
        int total = N * D;
        const int block = 256;
        unsigned int grid = (unsigned int)((total + block - 1) / block);
        add_bias_kernel<<<grid, block, 0, stream>>>(d_mlp_out, d_db, N, D);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Step 10: Residual add (MLP) ----
    {
        int total = N * D;
        const int block = 256;
        unsigned int grid = (unsigned int)((total + block - 1) / block);
        add_residual_kernel<<<grid, block, 0, stream>>>(d_x, d_mlp_out, N * D);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Helper kernel: y += x (element-wise)
__global__ void add_residual_kernel(float* y, const float* x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] += x[idx];
}

// ============================================================================
// CogViT forward
// ============================================================================
boat_tensor_t* cogvit_cuda_forward(cogvit_cuda_model_t* m, const boat_tensor_t* image) {
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    cudaStream_t stream = 0; // default stream

    const float* img_h = (const float*)boat_tensor_data(image);
    const int64_t* img_shape = boat_tensor_shape(image);
    int img_h_dim = (int)img_shape[2];
    int img_w = (int)img_shape[3];
    int ph = img_h_dim / COGVIT_PATCH_SIZE;
    int pw = img_w / COGVIT_PATCH_SIZE;
    int N = ph * pw; // num_patches

    fprintf(stderr, "[CogViT-CUDA] Image: %dx%d, patches: %dx%d = %d\n",
            img_h_dim, img_w, ph, pw, N);

    // ---- Allocate device memory ----
    float *d_img = NULL, *d_patch = NULL, *d_hidden = NULL, *d_cos = NULL, *d_sin = NULL;
    float *d_tmp = NULL;
    size_t img_bytes = (size_t)3 * img_h_dim * img_w * sizeof(float);
    size_t patch_bytes = (size_t)COGVIT_HIDDEN_SIZE * N * sizeof(float);
    // Max temp buffer usage: d_ff (= d_qkv reuse) at offset N*D needs N*2*ff_dim for gate/up,
    // extending to N*(D + 2*ff_dim) = N*9216 (for D=1024, ff_dim=4096).
    // Old formula N*(3*D + ff_dim) = N*7168 was too small.
    size_t tmp_bytes = (size_t)N * (COGVIT_HIDDEN_SIZE + 2 * COGVIT_INTERMEDIATE_SIZE) * sizeof(float);
    size_t cos_bytes = (size_t)N * COGVIT_HEAD_DIM * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_img, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_patch, patch_bytes));
    CUDA_CHECK(cudaMalloc(&d_hidden, patch_bytes));
    CUDA_CHECK(cudaMalloc(&d_cos, cos_bytes));
    CUDA_CHECK(cudaMalloc(&d_sin, cos_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));

    // Transfer image to GPU
    CUDA_CHECK(cudaMemcpy(d_img, img_h, img_bytes, cudaMemcpyHostToDevice));

    // ---- Step 1: Patch embedding ----
    patch_embed_cuda(d_img, m->d_patch_embed_weight, m->d_patch_embed_bias,
                     d_patch, img_h_dim, img_w, COGVIT_HIDDEN_SIZE, COGVIT_PATCH_SIZE, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // DIAG: patch embed full tensor norm
    {
        size_t nelem = (size_t)N * COGVIT_HIDDEN_SIZE;
        float* h_buf = (float*)malloc(nelem * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_buf, d_patch, nelem * sizeof(float), cudaMemcpyDeviceToHost));
        double nrm = 0; for (size_t j = 0; j < nelem; j++) nrm += (double)h_buf[j]*h_buf[j];
        fprintf(stderr, "[CogViT-DIAG] patch_embed full_norm=%.2f per_token=%.4f\n",
                sqrt(nrm), sqrt(nrm / N));
        fprintf(stderr, "[CogViT-DIAG] patch_embed[0]=%.4f %.4f %.4f %.4f\n",
                h_buf[0], h_buf[1], h_buf[2], h_buf[3]);
        fprintf(stderr, "[CogViT-DIAG] patch_embed[1]=%.4f %.4f %.4f %.4f (token 1)\n",
                h_buf[COGVIT_HIDDEN_SIZE], h_buf[COGVIT_HIDDEN_SIZE+1],
                h_buf[COGVIT_HIDDEN_SIZE+2], h_buf[COGVIT_HIDDEN_SIZE+3]);
        free(h_buf);
    }
    fprintf(stderr, "[CogViT-CUDA] Patch embed done\n");

    // ---- Step 2: Patch reorder ----
    patch_reorder_cuda(d_patch, d_hidden, COGVIT_HIDDEN_SIZE, ph, pw, stream);
    // DIAG: after reorder (first 4 values of token 0)
    {
        float h4[8];
        CUDA_CHECK(cudaMemcpy(h4, d_hidden, 8*sizeof(float), cudaMemcpyDeviceToHost));
        fprintf(stderr, "[CogViT-DIAG] after_reorder[0]=%.4f %.4f %.4f %.4f\n",
                h4[0], h4[1], h4[2], h4[3]);
    }
    fprintf(stderr, "[CogViT-CUDA] Patch reorder done\n");

    // ---- Step 3: Precompute 2D RoPE ----
    rope_2d_compute_cuda(d_cos, d_sin, ph, pw, COGVIT_HEAD_DIM, 10000.0f, stream);
    fprintf(stderr, "[CogViT-CUDA] RoPE computed\n");

    // ---- Step 4: 24 transformer blocks ----
    for (int l = 0; l < COGVIT_NUM_LAYERS; l++) {
        if (l % 4 == 0) fprintf(stderr, "[CogViT-CUDA] Block %d/%d\n", l, COGVIT_NUM_LAYERS);
        cogvit_block_gpu(handle, d_hidden, N, COGVIT_HIDDEN_SIZE, COGVIT_NUM_HEADS, COGVIT_HEAD_DIM,
                          m->blocks[l].d_norm1_weight,
                          m->blocks[l].d_attn_qkv_weight, m->blocks[l].d_attn_qkv_bias,
                          m->blocks[l].d_attn_q_norm_weight, m->blocks[l].d_attn_k_norm_weight,
                          m->blocks[l].d_attn_proj_weight, m->blocks[l].d_attn_proj_bias,
                          m->blocks[l].d_norm2_weight,
                          m->blocks[l].d_mlp_gate_proj_weight, m->blocks[l].d_mlp_gate_proj_bias,
                          m->blocks[l].d_mlp_up_proj_weight, m->blocks[l].d_mlp_up_proj_bias,
                          m->blocks[l].d_mlp_down_proj_weight, m->blocks[l].d_mlp_down_proj_bias,
                          d_cos, d_sin, d_tmp, stream);
    }
    fprintf(stderr, "[CogViT-CUDA] All blocks done\n");

    // DIAG: after blocks (full norm + first token first 4 values)
    {
        size_t nelem = (size_t)N * COGVIT_HIDDEN_SIZE;
        float* h_buf = (float*)malloc(nelem * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_buf, d_hidden, nelem * sizeof(float), cudaMemcpyDeviceToHost));
        double nrm = 0; for (size_t j = 0; j < nelem; j++) nrm += (double)h_buf[j]*h_buf[j];
        fprintf(stderr, "[CogViT-DIAG] after_blocks full_norm=%.2f per_token=%.4f\n",
                sqrt(nrm), sqrt(nrm / N));
        fprintf(stderr, "[CogViT-DIAG] after_blocks[0]=%.4f %.4f %.4f %.4f\n",
                h_buf[0], h_buf[1], h_buf[2], h_buf[3]);
        free(h_buf);
    }

    // ---- Step 5: Post-layernorm ----
    boat_cuda_rmsnorm_forward_f32(d_hidden, m->d_post_layernorm_weight, d_hidden, N, COGVIT_HIDDEN_SIZE, 1e-5f);

    // DIAG: after post-layernorm
    {
        size_t nelem = (size_t)N * COGVIT_HIDDEN_SIZE;
        float* h_buf = (float*)malloc(nelem * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_buf, d_hidden, nelem * sizeof(float), cudaMemcpyDeviceToHost));
        double nrm = 0; for (size_t j = 0; j < nelem; j++) nrm += (double)h_buf[j]*h_buf[j];
        fprintf(stderr, "[CogViT-DIAG] post_ln full_norm=%.2f per_token=%.4f\n",
                sqrt(nrm), sqrt(nrm / N));
        fprintf(stderr, "[CogViT-DIAG] post_ln[0]=%.4f %.4f %.4f %.4f\n",
                h_buf[0], h_buf[1], h_buf[2], h_buf[3]);
        free(h_buf);
    }

    // ---- Step 6: Downsample ----
    int num_groups = N / 4;
    float* d_downsampled;
    CUDA_CHECK(cudaMalloc(&d_downsampled, (size_t)num_groups * COGVIT_OUT_HIDDEN_SIZE * sizeof(float)));
    downsample_cuda(d_hidden, m->d_downsample_weight, m->d_downsample_bias,
                     d_downsampled, N, COGVIT_HIDDEN_SIZE, COGVIT_OUT_HIDDEN_SIZE, stream);
    // DIAG: after downsample
    {
        int M2 = num_groups;
        size_t nelem = (size_t)M2 * COGVIT_OUT_HIDDEN_SIZE;
        float* h_buf = (float*)malloc(nelem * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_buf, d_downsampled, nelem * sizeof(float), cudaMemcpyDeviceToHost));
        double nrm = 0; for (size_t j = 0; j < nelem; j++) nrm += (double)h_buf[j]*h_buf[j];
        fprintf(stderr, "[CogViT-DIAG] downsample full_norm=%.2f per_token=%.4f\n",
                sqrt(nrm), sqrt(nrm / M2));
        fprintf(stderr, "[CogViT-DIAG] downsample[0]=%.4f %.4f %.4f %.4f\n",
                h_buf[0], h_buf[1], h_buf[2], h_buf[3]);
        free(h_buf);
    }
    fprintf(stderr, "[CogViT-CUDA] Downsample done: %d -> %d tokens\n", N, num_groups);

    // ---- Step 7: Merger ----
    int M = num_groups;
    int D = COGVIT_OUT_HIDDEN_SIZE;
    int ff_dim = 4608;

    // Temp buffers for merger
    float *d_merger_tmp, *d_merger_out;
    CUDA_CHECK(cudaMalloc(&d_merger_tmp, (size_t)M * ff_dim * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_merger_out, (size_t)M * D * sizeof(float)));

    // Step 7a: proj = x @ proj_w^T  [M, D]
    matmul_bt_cuda(handle, d_downsampled, m->d_merger_proj_weight, d_merger_out, M, D, D);

    // Step 7b: LayerNorm + GELU
    layernorm_gelu_cuda(d_merger_out, d_merger_out,
                         m->d_merger_post_norm_weight, m->d_merger_post_norm_bias,
                         M, D, 1e-5f, stream);

    // Step 7c-d: gate = proj @ gate_w^T, up = proj @ up_w^T
    // Use d_merger_tmp as [M, 2*ff_dim]
    matmul_bt_cuda(handle, d_merger_out, m->d_merger_gate_proj_weight,
                   d_merger_tmp, M, D, ff_dim);
    matmul_bt_cuda(handle, d_merger_out, m->d_merger_up_proj_weight,
                   d_merger_tmp + M * ff_dim, M, D, ff_dim);

    // Step 7e: gate = silu(gate) * up
    // d_merger_tmp layout: [0..M*ff_dim) = gate, [M*ff_dim..2M*ff_dim) = up (contiguous blocks)
    merger_silu_gate_cuda(d_merger_tmp, M, ff_dim, stream);

    // Step 7f: down projection: out = gate_result @ down_w^T
    matmul_bt_cuda(handle, d_merger_tmp, m->d_merger_down_proj_weight,
                   d_merger_out, M, ff_dim, D);

    // DIAG: after merger
    {
        size_t nelem = (size_t)M * D;
        float* h_buf = (float*)malloc(nelem * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_buf, d_merger_out, nelem * sizeof(float), cudaMemcpyDeviceToHost));
        double nrm = 0; for (size_t j = 0; j < nelem; j++) nrm += (double)h_buf[j]*h_buf[j];
        fprintf(stderr, "[CogViT-DIAG] merger full_norm=%.2f per_token=%.4f\n",
                sqrt(nrm), sqrt(nrm / M));
        fprintf(stderr, "[CogViT-DIAG] merger[0]=%.4f %.4f %.4f %.4f\n",
                h_buf[0], h_buf[1], h_buf[2], h_buf[3]);
        free(h_buf);
    }
    fprintf(stderr, "[CogViT-CUDA] Merger done\n");

    // ---- Copy result to host ----
    int64_t shape[] = { 1, M, D };
    boat_tensor_t* result = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (result) {
        CUDA_CHECK(cudaMemcpy(boat_tensor_data(result), d_merger_out,
                               (size_t)M * D * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_patch));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_downsampled));
    CUDA_CHECK(cudaFree(d_merger_tmp));
    CUDA_CHECK(cudaFree(d_merger_out));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return result;
}

// ============================================================================
// Model loading
// ============================================================================
int cogvit_cuda_load(cogvit_cuda_model_t* m, safetensors_t* st) {
    memset(m, 0, sizeof(*m));

    m->d_patch_embed_weight = load_weight_to_gpu(st, "model.visual.patch_embed.proj.weight");
    m->d_patch_embed_bias   = load_weight_to_gpu(st, "model.visual.patch_embed.proj.bias");
    m->d_post_layernorm_weight = load_weight_to_gpu(st, "model.visual.post_layernorm.weight");

    if (!m->d_patch_embed_weight || !m->d_patch_embed_bias) {
        fprintf(stderr, "[CogViT-CUDA] Missing patch_embed weights\n");
        return 0;
    }

    for (int l = 0; l < COGVIT_NUM_LAYERS; l++) {
        char name[256];
        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm1.weight", l);
        m->blocks[l].d_norm1_weight = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.qkv.weight", l);
        m->blocks[l].d_attn_qkv_weight = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.qkv.bias", l);
        m->blocks[l].d_attn_qkv_bias = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.q_norm.weight", l);
        m->blocks[l].d_attn_q_norm_weight = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.k_norm.weight", l);
        m->blocks[l].d_attn_k_norm_weight = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.proj.weight", l);
        m->blocks[l].d_attn_proj_weight = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.proj.bias", l);
        m->blocks[l].d_attn_proj_bias = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm2.weight", l);
        m->blocks[l].d_norm2_weight = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.gate_proj.weight", l);
        m->blocks[l].d_mlp_gate_proj_weight = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.gate_proj.bias", l);
        m->blocks[l].d_mlp_gate_proj_bias = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.up_proj.weight", l);
        m->blocks[l].d_mlp_up_proj_weight = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.up_proj.bias", l);
        m->blocks[l].d_mlp_up_proj_bias = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.down_proj.weight", l);
        m->blocks[l].d_mlp_down_proj_weight = load_weight_to_gpu(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.down_proj.bias", l);
        m->blocks[l].d_mlp_down_proj_bias = load_weight_to_gpu(st, name);

        if (!m->blocks[l].d_attn_qkv_weight || !m->blocks[l].d_attn_proj_weight) {
            fprintf(stderr, "[CogViT-CUDA] Missing attn weights in block %d\n", l);
            return 0;
        }
    }

    m->d_downsample_weight = load_weight_to_gpu(st, "model.visual.downsample.weight");
    m->d_downsample_bias   = load_weight_to_gpu(st, "model.visual.downsample.bias");

    m->d_merger_gate_proj_weight = load_weight_to_gpu(st, "model.visual.merger.gate_proj.weight");
    m->d_merger_up_proj_weight   = load_weight_to_gpu(st, "model.visual.merger.up_proj.weight");
    m->d_merger_down_proj_weight = load_weight_to_gpu(st, "model.visual.merger.down_proj.weight");
    m->d_merger_proj_weight      = load_weight_to_gpu(st, "model.visual.merger.proj.weight");
    m->d_merger_post_norm_weight = load_weight_to_gpu(st, "model.visual.merger.post_projection_norm.weight");
    m->d_merger_post_norm_bias   = load_weight_to_gpu(st, "model.visual.merger.post_projection_norm.bias");

    fprintf(stderr, "[CogViT-CUDA] Loaded: 24 blocks, %d hidden (weights on GPU)\n",
            COGVIT_HIDDEN_SIZE);
    return 1;
}

static void free_gpu(float* p) { if (p) cudaFree(p); }

void cogvit_cuda_free(cogvit_cuda_model_t* m) {
    if (!m) return;
    free_gpu(m->d_patch_embed_weight); free_gpu(m->d_patch_embed_bias);
    free_gpu(m->d_post_layernorm_weight);
    for (int l = 0; l < COGVIT_NUM_LAYERS; l++) {
        free_gpu(m->blocks[l].d_norm1_weight);
        free_gpu(m->blocks[l].d_attn_qkv_weight); free_gpu(m->blocks[l].d_attn_qkv_bias);
        free_gpu(m->blocks[l].d_attn_q_norm_weight); free_gpu(m->blocks[l].d_attn_k_norm_weight);
        free_gpu(m->blocks[l].d_attn_proj_weight); free_gpu(m->blocks[l].d_attn_proj_bias);
        free_gpu(m->blocks[l].d_norm2_weight);
        free_gpu(m->blocks[l].d_mlp_gate_proj_weight); free_gpu(m->blocks[l].d_mlp_gate_proj_bias);
        free_gpu(m->blocks[l].d_mlp_up_proj_weight); free_gpu(m->blocks[l].d_mlp_up_proj_bias);
        free_gpu(m->blocks[l].d_mlp_down_proj_weight); free_gpu(m->blocks[l].d_mlp_down_proj_bias);
    }
    free_gpu(m->d_downsample_weight); free_gpu(m->d_downsample_bias);
    free_gpu(m->d_merger_gate_proj_weight); free_gpu(m->d_merger_up_proj_weight);
    free_gpu(m->d_merger_down_proj_weight); free_gpu(m->d_merger_proj_weight);
    free_gpu(m->d_merger_post_norm_weight); free_gpu(m->d_merger_post_norm_bias);
    memset(m, 0, sizeof(*m));
}
