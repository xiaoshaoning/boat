// cogvit.c - CogViT vision encoder forward pass
#include "cogvit.h"
#include "ocr_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ========== Patch embedding ==========
// Weight: [1024, 3, 2, 14, 14] for temporal_patch_size=2
// For a single image, we sum over temporal dim or reshape
static void patch_embed_forward(float* output, const float* input,
                                 const float* weight, const float* bias,
                                 int C, int H, int W, int patch_size) {
    // input: [1, 3, H, W] in CHW
    // weight: [1024, 6, 14, 14] treated as [1024, 6, 14, 14]
    // output: [1, 1024, H/14, W/14]
    int out_h = H / patch_size;
    int out_w = W / patch_size;
    int in_ch = 3;
    int temp_merge = 2;  // temporal_patch_size

    // For 2D image: patch_size=P, stride=P, kernel=[C*T, 3, P, P]
    // We treat weight [1024, 3, 2, 14, 14] as [1024, 6, 14, 14] by merging channels
    // For image input [3, 336, 336], we duplicate along temporal dim

    int k_size = patch_size;

    for (int oc = 0; oc < C; oc++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                float sum = bias[oc];
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int t = 0; t < temp_merge; t++) {
                        for (int ky = 0; ky < k_size; ky++) {
                            for (int kx = 0; kx < k_size; kx++) {
                                int iy = oy * patch_size + ky;
                                int ix = ox * patch_size + kx;
                                if (iy < H && ix < W) {
                                    float img_val = input[ic * H * W + iy * W + ix];
                                    int widx = oc * (in_ch * temp_merge * k_size * k_size)
                                             + ic * (temp_merge * k_size * k_size)
                                             + t * (k_size * k_size)
                                             + ky * k_size + kx;
                                    sum += weight[widx] * img_val;
                                }
                            }
                        }
                    }
                }
                output[oc * out_h * out_w + oy * out_w + ox] = sum;
            }
        }
    }
}

// ========== Multi-head attention with QK-Norm and 2D RoPE ==========
static void attention_forward(float* out, const float* x, int N, int D,
                               const float* qkv_w, const float* qkv_b,
                               const float* q_norm_w, const float* k_norm_w,
                               const float* proj_w, const float* proj_b,
                               int num_heads,
                               const float* cos, const float* sin) {
    int head_dim = D / num_heads;
    // QKV projection
    float* qkv = (float*)malloc(N * 3 * D * sizeof(float));
    matmul_bt(qkv, x, qkv_w, N, D, 3 * D);
    for (int i = 0; i < N * 3 * D; i++) qkv[i] += qkv_b[i % (3 * D)];

    float* q_base = qkv;
    float* k_base = qkv + D;
    float* v_base = qkv + 2 * D;

    // Apply QK-Norm (per-head RMSNorm) and 2D RoPE
    for (int i = 0; i < N; i++) {
        for (int h = 0; h < num_heads; h++) {
            int off = i * 3 * D + h * head_dim;
            apply_rmsnorm(q_base + off, q_base + off, q_norm_w, head_dim, 1e-5f);
            apply_rmsnorm(k_base + off, k_base + off, k_norm_w, head_dim, 1e-5f);
            // 2D RoPE: q_rot = q*cos + rotate_half(q)*sin (full head_dim rotation)
            if (cos) {
                int half = head_dim / 2;
                for (int d = 0; d < half; d++) {
                    float q0 = q_base[off + d], q1 = q_base[off + d + half];
                    float c = cos[i * head_dim + d], s = sin[i * head_dim + d];
                    q_base[off + d] = q0 * c - q1 * s;
                    q_base[off + d + half] = q1 * c + q0 * s;
                }
                for (int d = 0; d < half; d++) {
                    float k0 = k_base[off + d], k1 = k_base[off + d + half];
                    float c = cos[i * head_dim + d], s = sin[i * head_dim + d];
                    k_base[off + d] = k0 * c - k1 * s;
                    k_base[off + d + half] = k1 * c + k0 * s;
                }
            }
        }
    }

    // Process attention per head using OpenBLAS sgemm for QK^T and PV
    float scale = 1.0f / sqrtf((float)head_dim);
    float* context = (float*)calloc(N * D, sizeof(float));
    float* scores = (float*)malloc(N * N * sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        int h_off = h * head_dim;

        // scores[N,N] = (Q_h @ K_h^T) * scale
        // Q_h: [N, head_dim] at stride 3*D, offset h_off
        // K_h: [N, head_dim] at stride 3*D, offset h_off
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, N, head_dim,
                    scale,
                    q_base + h_off, 3 * D,
                    k_base + h_off, 3 * D,
                    0.0f,
                    scores, N);

        // Softmax each row (manual loop — O(N²) with no dot products)
        for (int i = 0; i < N; i++) {
            float* si = scores + i * N;
            float max_val = si[0];
            for (int j = 1; j < N; j++)
                if (si[j] > max_val) max_val = si[j];
            float ssum = 0.0f;
            for (int j = 0; j < N; j++) { si[j] = expf(si[j] - max_val); ssum += si[j]; }
            float inv_ssum = 1.0f / ssum;
            for (int j = 0; j < N; j++) si[j] *= inv_ssum;
        }

        // context_h = softmax_scores @ V_h
        // V_h: [N, head_dim] at stride 3*D, offset h_off
        // context_h: [N, head_dim] at stride D, offset h_off
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, head_dim, N,
                    1.0f,
                    scores, N,
                    v_base + h_off, 3 * D,
                    0.0f,
                    context + h_off, D);
    }

    // Output projection
    matmul_bt(out, context, proj_w, N, D, D);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            out[i * D + j] += proj_b[j];

    free(qkv);
    free(context);
    free(scores);
}

// ========== CogViT block forward ==========
static void block_forward(float* hidden, int N, int D,
                           const float* norm1_w,
                           const float* qkv_w, const float* qkv_b,
                           const float* qn_w, const float* kn_w,
                           const float* proj_w, const float* proj_b,
                           const float* norm2_w,
                           const float* gate_w, const float* gate_b,
                           const float* up_w, const float* up_b,
                           const float* down_w, const float* down_b,
                           int num_heads,
                           const float* cos, const float* sin) {
    // Store residual
    float* residual = (float*)malloc(N * D * sizeof(float));
    memcpy(residual, hidden, N * D * sizeof(float));

    // RMSNorm before attention
    for (int i = 0; i < N; i++)
        apply_rmsnorm(hidden + i * D, residual + i * D, norm1_w, D, 1e-5f);

    // Attention
    float* attn_out = (float*)malloc(N * D * sizeof(float));
    attention_forward(attn_out, hidden, N, D, qkv_w, qkv_b, qn_w, kn_w, proj_w, proj_b, num_heads, cos, sin);

    // Residual add
    for (int i = 0; i < N * D; i++) hidden[i] = residual[i] + attn_out[i];
    free(attn_out);

    // Store residual again
    memcpy(residual, hidden, N * D * sizeof(float));

    // RMSNorm before MLP
    for (int i = 0; i < N; i++)
        apply_rmsnorm(hidden + i * D, residual + i * D, norm2_w, D, 1e-5f);

    // SiLU FFN: gate = silu(x @ gate_w.T + gate_b), up = x @ up_w.T + up_b, out = (gate * up) @ down_w.T + down_b
    int ff_dim = 4096;
    float* gate = (float*)malloc(N * ff_dim * sizeof(float));
    float* up = (float*)malloc(N * ff_dim * sizeof(float));

    matmul_bt(gate, hidden, gate_w, N, D, ff_dim);
    for (int i = 0; i < N * ff_dim; i++) gate[i] += gate_b[i % ff_dim];
    for (int i = 0; i < N * ff_dim; i++) gate[i] = silu(gate[i]);

    matmul_bt(up, hidden, up_w, N, D, ff_dim);
    for (int i = 0; i < N * ff_dim; i++) up[i] += up_b[i % ff_dim];

    float* ff_out = (float*)malloc(N * ff_dim * sizeof(float));
    for (int i = 0; i < N * ff_dim; i++) ff_out[i] = gate[i] * up[i];

    float* mlp_out = (float*)malloc(N * D * sizeof(float));
    matmul_bt(mlp_out, ff_out, down_w, N, ff_dim, D);
    for (int i = 0; i < N * D; i++) mlp_out[i] += down_b[i % D];

    // Residual add
    for (int i = 0; i < N * D; i++) hidden[i] = residual[i] + mlp_out[i];

    free(residual);
    free(gate);
    free(up);
    free(ff_out);
    free(mlp_out);
}

// ========== CogViT forward ==========
boat_tensor_t* cogvit_forward(cogvit_model_t* m, const boat_tensor_t* image) {
    const float* img = (const float*)boat_tensor_const_data(image);
    const int64_t* img_shape = boat_tensor_shape(image);
    int img_h = (int)img_shape[2];
    int img_w = (int)img_shape[3];
    int patch_h = img_h / COGVIT_PATCH_SIZE;
    int patch_w = img_w / COGVIT_PATCH_SIZE;
    int num_patches = patch_h * patch_w;

    // 1. Patch embedding
    float* hidden = (float*)malloc(COGVIT_HIDDEN_SIZE * num_patches * sizeof(float));
    const float* pe_w = (const float*)boat_tensor_const_data(m->patch_embed_weight);
    const float* pe_b = (const float*)boat_tensor_const_data(m->patch_embed_bias);
    patch_embed_forward(hidden, img, pe_w, pe_b, COGVIT_HIDDEN_SIZE, img_h, img_w, COGVIT_PATCH_SIZE);

    // Reorder hidden from row-major to spatial-2x2-block order to match Python's image processor output.
    // Python pixel_values are arranged by the image processor as:
    //   view(B, grid_t, tp, C, gh//ms, ms, ph, gw//ms, ms, pw).permute(0,1,4,7,5,8,3,2,6,9)
    // This groups patches into 2x2 spatial blocks.
    // The C patch_embed produces row-major order: patch (r,c) at index r*patch_w + c.
    // We reorder: spatial-block index = (r/2 * w/2 + c/2) * 4 + (r%2)*2 + (c%2)
    {
        float* reordered = (float*)malloc(COGVIT_HIDDEN_SIZE * num_patches * sizeof(float));
        int hw2 = patch_w / 2;
        for (int r = 0; r < patch_h; r++) {
            for (int c = 0; c < patch_w; c++) {
                int sb_idx = ((r / 2) * hw2 + (c / 2)) * 4 + (r % 2) * 2 + (c % 2);
                int rm_base = r * patch_w + c;
                for (int oc = 0; oc < COGVIT_HIDDEN_SIZE; oc++)
                    reordered[sb_idx * COGVIT_HIDDEN_SIZE + oc] = hidden[oc * num_patches + rm_base];
            }
        }
        memcpy(hidden, reordered, COGVIT_HIDDEN_SIZE * num_patches * sizeof(float));
        free(reordered);
    }

    // Debug: check norms after patch embedding
    {
        float nrm = 0; for (int j = 0; j < num_patches * COGVIT_HIDDEN_SIZE; j++) nrm += hidden[j]*hidden[j];
        fprintf(stderr, "[DEBUG] patch_embed norm: %.2f (per-token: %.4f)\n",
                sqrtf(nrm), sqrtf(nrm / num_patches));
        fprintf(stderr, "[DEBUG] patch_embed[0]:");
        for (int j = 0; j < 4; j++) fprintf(stderr, " %.4f", hidden[j]);
        fprintf(stderr, "\n");
    }

    // 1b. Precompute 2D RoPE (cos/sin) for vision attention in spatial-block order.
    // hidden is now in spatial-block order: index p corresponds to block position
    // (gy = p/(4*w/2), gx = (p/4)%(w/2), my = (p%4)/2, mx = p%2) → grid pos (r=gy*2+my, c=gx*2+mx).
    // We iterate grid positions and compute RoPE at the correct spatial-block index.
    float* cos_buf = (float*)malloc(num_patches * COGVIT_HEAD_DIM * sizeof(float));
    float* sin_buf = (float*)malloc(num_patches * COGVIT_HEAD_DIM * sizeof(float));
    float inv_freq[16];
    for (int i = 0; i < 16; i++)
        inv_freq[i] = 1.0f / powf(10000.0f, (2.0f * i) / 32.0f);
    {
        int hw2 = patch_w / 2;
        for (int r = 0; r < patch_h; r++) {
            for (int c = 0; c < patch_w; c++) {
                int p = ((r / 2) * hw2 + (c / 2)) * 4 + (r % 2) * 2 + (c % 2);
                for (int i = 0; i < 16; i++) {
                    float th = r * inv_freq[i];
                    float tw = c * inv_freq[i];
                    float c_h = cosf(th), s_h = sinf(th);
                    float c_w = cosf(tw), s_w = sinf(tw);
                    // Python layout: [h0..h15, w0..w15, h0..h15, w0..w15]
                    cos_buf[p * 64 + i] = c_h;       cos_buf[p * 64 + 32 + i] = c_h;
                    sin_buf[p * 64 + i] = s_h;       sin_buf[p * 64 + 32 + i] = s_h;
                    cos_buf[p * 64 + 16 + i] = c_w;  cos_buf[p * 64 + 48 + i] = c_w;
                    sin_buf[p * 64 + 16 + i] = s_w;  sin_buf[p * 64 + 48 + i] = s_w;
                }
            }
        }
    }

    // 2. 24 transformer blocks
    for (int l = 0; l < COGVIT_NUM_LAYERS; l++) {
        if (l % 4 == 0) fprintf(stderr, "[INFO] CogViT block %d/%d\n", l, COGVIT_NUM_LAYERS);
        block_forward(hidden, num_patches, COGVIT_HIDDEN_SIZE,
                      (const float*)boat_tensor_const_data(m->blocks[l].norm1_weight),
                      (const float*)boat_tensor_const_data(m->blocks[l].attn_qkv_weight),
                      (const float*)boat_tensor_const_data(m->blocks[l].attn_qkv_bias),
                      (const float*)boat_tensor_const_data(m->blocks[l].attn_q_norm_weight),
                      (const float*)boat_tensor_const_data(m->blocks[l].attn_k_norm_weight),
                      (const float*)boat_tensor_const_data(m->blocks[l].attn_proj_weight),
                      (const float*)boat_tensor_const_data(m->blocks[l].attn_proj_bias),
                      (const float*)boat_tensor_const_data(m->blocks[l].norm2_weight),
                      (const float*)boat_tensor_const_data(m->blocks[l].mlp_gate_proj_weight),
                      (const float*)boat_tensor_const_data(m->blocks[l].mlp_gate_proj_bias),
                      (const float*)boat_tensor_const_data(m->blocks[l].mlp_up_proj_weight),
                      (const float*)boat_tensor_const_data(m->blocks[l].mlp_up_proj_bias),
                      (const float*)boat_tensor_const_data(m->blocks[l].mlp_down_proj_weight),
                      (const float*)boat_tensor_const_data(m->blocks[l].mlp_down_proj_bias),
                      COGVIT_NUM_HEADS,
                      cos_buf, sin_buf);
    }
    free(cos_buf);
    free(sin_buf);

    // 3. Post-layer norm
    // Debug: check norms after blocks (before post-ln)
    {
        float nrm = 0; for (int j = 0; j < num_patches * COGVIT_HIDDEN_SIZE; j++) nrm += hidden[j]*hidden[j];
        fprintf(stderr, "[DEBUG] after blocks norm: %.2f (per-token: %.4f)\n",
                sqrtf(nrm), sqrtf(nrm / num_patches));
    }

    const float* post_ln_w = (const float*)boat_tensor_const_data(m->post_layernorm_weight);
    for (int i = 0; i < num_patches; i++) {
        float* h = hidden + i * COGVIT_HIDDEN_SIZE;
        float ss = 0.0f;
        for (int j = 0; j < COGVIT_HIDDEN_SIZE; j++) ss += h[j] * h[j];
        float rms = 1.0f / sqrtf(ss / COGVIT_HIDDEN_SIZE + 1e-5f);
        for (int j = 0; j < COGVIT_HIDDEN_SIZE; j++) h[j] *= rms * post_ln_w[j];
    }

    // Debug: check norms after post-layernorm
    {
        float nrm = 0; for (int j = 0; j < num_patches * COGVIT_HIDDEN_SIZE; j++) nrm += hidden[j]*hidden[j];
        fprintf(stderr, "[DEBUG] post-ln norm: %.2f (per-token: %.4f)\n",
                sqrtf(nrm), sqrtf(nrm / num_patches));
        // Check first few values
        fprintf(stderr, "[DEBUG] post-ln[0]:");
        for (int j = 0; j < 4; j++) fprintf(stderr, " %.4f", hidden[j]);
        fprintf(stderr, "\n");
    }

    // 4. Downsample: group 4 consecutive patches → conv2d 2x2
    // Python: hidden.view(-1, 2, 2, dim).permute(0, 3, 1, 2) → Conv2d(1024, 1536, 2, 2)
    //   groups 4 CONSECUTIVE patches, NOT spatial 2x2 blocks.
    // Weight: [1536, 1024, 2, 2], Bias: [1536]
    int num_groups = num_patches / 4;
    int ds_tokens = num_groups;

    const float* ds_w_data = (const float*)boat_tensor_const_data(m->downsample_weight);
    const float* ds_b_data = (const float*)boat_tensor_const_data(m->downsample_bias);

    float* downsampled = (float*)calloc(ds_tokens * COGVIT_OUT_HIDDEN_SIZE, sizeof(float));

    // Each group of 4 consecutive patches [g*4+0..g*4+3] → 1 output token.
    // Weight access: w[oc, ic, ky, kx] maps to patch[g*4 + ky*2 + kx]
    for (int g = 0; g < num_groups; g++) {
        for (int oc = 0; oc < COGVIT_OUT_HIDDEN_SIZE; oc++) {
            float sum = ds_b_data[oc];
            for (int ic = 0; ic < COGVIT_HIDDEN_SIZE; ic++) {
                for (int ky = 0; ky < COGVIT_SPATIAL_MERGE_SIZE; ky++) {
                    for (int kx = 0; kx < COGVIT_SPATIAL_MERGE_SIZE; kx++) {
                        int pi = g * 4 + ky * 2 + kx;
                        sum += hidden[pi * COGVIT_HIDDEN_SIZE + ic]
                             * ds_w_data[oc * COGVIT_HIDDEN_SIZE * 4 + ic * 4 + ky * 2 + kx];
                    }
                }
            }
            downsampled[g * COGVIT_OUT_HIDDEN_SIZE + oc] = sum;
        }
    }
    free(hidden);

    // Debug: check norms after downsample
    {
        float nrm = 0; for (int j = 0; j < ds_tokens * COGVIT_OUT_HIDDEN_SIZE; j++) nrm += downsampled[j]*downsampled[j];
        fprintf(stderr, "[DEBUG] downsample norm: %.2f (per-token: %.4f)\n",
                sqrtf(nrm), sqrtf(nrm / ds_tokens));
        fprintf(stderr, "[DEBUG] downsample[0]:");
        for (int j = 0; j < 4; j++) fprintf(stderr, " %.4f", downsampled[j]);
        fprintf(stderr, "\n");
    }

    // 5. Merger: refine visual tokens with MLP
    // Python order: proj → LayerNorm → GELU → gate_proj(SiLU) * up_proj → down_proj
    // Weight shapes:
    //   proj: [1536, 1536], post_norm: [1536], bias: [1536]
    //   gate_proj: [4608, 1536], up_proj: [4608, 1536], down_proj: [1536, 4608]
    int M = ds_tokens;
    int D = COGVIT_OUT_HIDDEN_SIZE;
    int ff_dim = 4608;

    const float* gate_w = (const float*)boat_tensor_const_data(m->merger_gate_proj_weight);
    const float* up_w = (const float*)boat_tensor_const_data(m->merger_up_proj_weight);
    const float* down_w = (const float*)boat_tensor_const_data(m->merger_down_proj_weight);
    const float* proj_w = (const float*)boat_tensor_const_data(m->merger_proj_weight);
    const float* pn_w = (const float*)boat_tensor_const_data(m->merger_post_norm_weight);
    const float* pn_b = (const float*)boat_tensor_const_data(m->merger_post_norm_bias);

    // Step 1: proj = x @ proj_w.T  [M, D]
    float* tmp = (float*)malloc(M * D * sizeof(float));
    matmul_bt(tmp, downsampled, proj_w, M, D, D);

    // Step 2: LayerNorm (with bias) + GELU
    float* hidden_m = (float*)malloc(M * D * sizeof(float));
    for (int i = 0; i < M; i++) {
        float mean = 0, var = 0;
        for (int j = 0; j < D; j++) { mean += tmp[i * D + j]; }
        mean /= D;
        for (int j = 0; j < D; j++) { float d = tmp[i * D + j] - mean; var += d * d; }
        var /= D;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int j = 0; j < D; j++) {
            float val = (tmp[i * D + j] - mean) * inv_std * pn_w[j] + pn_b[j];
            hidden_m[i * D + j] = 0.5f * val * (1.0f + erff(val / 1.41421356237f));
        }
    }
    free(tmp);

    // Step 3-4: gate = silu(x @ gate_w.T), up = x @ up_w.T
    float* gate = (float*)malloc(M * ff_dim * sizeof(float));
    matmul_bt(gate, hidden_m, gate_w, M, D, ff_dim);
    for (int i = 0; i < M * ff_dim; i++) gate[i] = silu(gate[i]);

    float* up = (float*)malloc(M * ff_dim * sizeof(float));
    matmul_bt(up, hidden_m, up_w, M, D, ff_dim);
    free(hidden_m);

    // Step 5: ff_out = gate * up
    for (int i = 0; i < M * ff_dim; i++) up[i] *= gate[i];
    free(gate);

    // Step 6: out = ff_out @ down_w.T  [M, D]
    float* out = (float*)malloc(M * D * sizeof(float));
    matmul_bt(out, up, down_w, M, ff_dim, D);
    free(up);

    // Debug: check norms after merger
    {
        float nrm = 0; for (int j = 0; j < M * D; j++) nrm += out[j]*out[j];
        fprintf(stderr, "[DEBUG] merger norm: %.2f (per-token: %.4f)\n",
                sqrtf(nrm), sqrtf(nrm / M));
        fprintf(stderr, "[DEBUG] merger[0]:");
        for (int j = 0; j < 4; j++) fprintf(stderr, " %.4f", out[j]);
        fprintf(stderr, "\n");
        float wn = 0; for (int j = 0; j < D; j++) wn += pn_w[j]*pn_w[j];
        float bn = 0; for (int j = 0; j < D; j++) bn += pn_b[j]*pn_b[j];
        fprintf(stderr, "[DEBUG] merger_post_norm: weight_rms=%.4f bias_rms=%.4f\n",
                sqrtf(wn/D), sqrtf(bn/D));
    }

    free(downsampled);

    // Create output tensor [1, M, 1536] where M = (H/28)*(W/28)
    int64_t shape[] = { 1, M, D };
    boat_tensor_t* result = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (result) {
        float* rdata = (float*)boat_tensor_data(result);
        memcpy(rdata, out, M * D * sizeof(float));
    }
    free(out);
    return result;
}

// ========== Weight loading helpers ==========
static boat_tensor_t* load_w(safetensors_t* st, const char* name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) { fprintf(stderr, "[WARN] CogViT weight not found: %s\n", name); return NULL; }
    return safetensors_load_tensor(st, idx, 0);
}

int cogvit_load(cogvit_model_t* m, safetensors_t* st) {
    memset(m, 0, sizeof(*m));

    m->patch_embed_weight = load_w(st, "model.visual.patch_embed.proj.weight");
    m->patch_embed_bias = load_w(st, "model.visual.patch_embed.proj.bias");
    m->post_layernorm_weight = load_w(st, "model.visual.post_layernorm.weight");

    if (!m->patch_embed_weight || !m->patch_embed_bias) {
        fprintf(stderr, "[ERROR] Missing CogViT patch_embed weights\n");
        return 0;
    }

    for (int l = 0; l < COGVIT_NUM_LAYERS; l++) {
        char name[256];

        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm1.weight", l);
        m->blocks[l].norm1_weight = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.qkv.weight", l);
        m->blocks[l].attn_qkv_weight = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.qkv.bias", l);
        m->blocks[l].attn_qkv_bias = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.q_norm.weight", l);
        m->blocks[l].attn_q_norm_weight = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.k_norm.weight", l);
        m->blocks[l].attn_k_norm_weight = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.proj.weight", l);
        m->blocks[l].attn_proj_weight = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.attn.proj.bias", l);
        m->blocks[l].attn_proj_bias = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.norm2.weight", l);
        m->blocks[l].norm2_weight = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.gate_proj.weight", l);
        m->blocks[l].mlp_gate_proj_weight = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.gate_proj.bias", l);
        m->blocks[l].mlp_gate_proj_bias = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.up_proj.weight", l);
        m->blocks[l].mlp_up_proj_weight = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.up_proj.bias", l);
        m->blocks[l].mlp_up_proj_bias = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.down_proj.weight", l);
        m->blocks[l].mlp_down_proj_weight = load_w(st, name);

        snprintf(name, sizeof(name), "model.visual.blocks.%d.mlp.down_proj.bias", l);
        m->blocks[l].mlp_down_proj_bias = load_w(st, name);

        if (!m->blocks[l].attn_qkv_weight || !m->blocks[l].attn_proj_weight) {
            fprintf(stderr, "[ERROR] Missing attention weights for visual block %d\n", l);
            return 0;
        }
    }

    // Downsample
    m->downsample_weight = load_w(st, "model.visual.downsample.weight");
    m->downsample_bias = load_w(st, "model.visual.downsample.bias");

    // Merger
    m->merger_gate_proj_weight = load_w(st, "model.visual.merger.gate_proj.weight");
    m->merger_up_proj_weight = load_w(st, "model.visual.merger.up_proj.weight");
    m->merger_down_proj_weight = load_w(st, "model.visual.merger.down_proj.weight");
    m->merger_proj_weight = load_w(st, "model.visual.merger.proj.weight");
    m->merger_post_norm_weight = load_w(st, "model.visual.merger.post_projection_norm.weight");
    m->merger_post_norm_bias = load_w(st, "model.visual.merger.post_projection_norm.bias");

    fprintf(stderr, "[INFO] CogViT loaded: 24 blocks, %d hidden\n",
            COGVIT_HIDDEN_SIZE);
    return 1;
}

static void free_t(boat_tensor_t* t) { if (t) boat_tensor_unref(t); }

void cogvit_free(cogvit_model_t* m) {
    if (!m) return;
    free_t(m->patch_embed_weight); free_t(m->patch_embed_bias);
    free_t(m->post_layernorm_weight);
    for (int l = 0; l < COGVIT_NUM_LAYERS; l++) {
        free_t(m->blocks[l].norm1_weight);
        free_t(m->blocks[l].attn_qkv_weight); free_t(m->blocks[l].attn_qkv_bias);
        free_t(m->blocks[l].attn_q_norm_weight); free_t(m->blocks[l].attn_k_norm_weight);
        free_t(m->blocks[l].attn_proj_weight); free_t(m->blocks[l].attn_proj_bias);
        free_t(m->blocks[l].norm2_weight);
        free_t(m->blocks[l].mlp_gate_proj_weight); free_t(m->blocks[l].mlp_gate_proj_bias);
        free_t(m->blocks[l].mlp_up_proj_weight); free_t(m->blocks[l].mlp_up_proj_bias);
        free_t(m->blocks[l].mlp_down_proj_weight); free_t(m->blocks[l].mlp_down_proj_bias);
    }
    free_t(m->downsample_weight); free_t(m->downsample_bias);
    free_t(m->merger_gate_proj_weight); free_t(m->merger_up_proj_weight);
    free_t(m->merger_down_proj_weight); free_t(m->merger_proj_weight);
    free_t(m->merger_post_norm_weight); free_t(m->merger_post_norm_bias);
    memset(m, 0, sizeof(*m));
}
