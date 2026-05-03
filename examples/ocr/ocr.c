// ocr.c - GLM-OCR inference engine
// Loads GLM-OCR model from safetensors, processes images, generates text
//
// Build:
//   cd build && cmake .. -DBOAT_WITH_EXAMPLES=ON && make
// Usage:
//   ocr <model_dir> <image_path> [prompt]
//
// Example:
//   ocr D:/huggingface/GLM-OCR test.png "请描述这张图片"

#include <boat.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../common/safetensors.h"
#include "tokenizer.h"
#include "image.h"
#include "sampling.h"
#include "cogvit.h"
#include "glm.h"
#include "ocr_common.h"

// Default prompt for OCR
#define DEFAULT_PROMPT "Please read the text in this image"

#define MAX_GEN_TOKENS 200

int main(int argc, char** argv) {
    int fast_mode = 0;
    // Scan for --fast flag anywhere in args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fast") == 0) {
            fast_mode = 1;
            // Shift remaining args left to remove --fast
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1];
            argc--;
            break;
        }
    }
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_dir> <image_path> [prompt] [--fast]\n", argv[0]);
        return 1;
    }

    const char* model_dir = argv[1];
    const char* image_path = argv[2];
    const char* prompt_text = (argc > 3) ? argv[3] : DEFAULT_PROMPT;

    // Build paths
    char model_path[1024];
    char tokenizer_path[1024];
    snprintf(model_path, sizeof(model_path), "%s/model.safetensors", model_dir);
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", model_dir);

    srand((unsigned int)time(NULL));

    fprintf(stderr, "[INFO] GLM-OCR Inference Engine\n");
    fprintf(stderr, "[INFO] Model: %s\n", model_path);
    fprintf(stderr, "[INFO] Image: %s\n", image_path);
    fprintf(stderr, "[INFO] Prompt: %s\n", prompt_text);

    // ======== 1. Load tokenizer ========
    fprintf(stderr, "[INFO] Loading tokenizer...\n");
    ocr_tokenizer_t tokenizer;
    if (!ocr_tokenizer_init(&tokenizer, tokenizer_path)) {
        fprintf(stderr, "[FATAL] Failed to load tokenizer\n");
        return 1;
    }

    // ======== 2. Open safetensors ========
    fprintf(stderr, "[INFO] Opening model weights...\n");
    safetensors_t st;
    if (!safetensors_open(&st, model_path)) {
        fprintf(stderr, "[FATAL] Failed to open model.safetensors\n");
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    // ======== 3. Load CogViT model ========
    fprintf(stderr, "[INFO] Loading CogViT vision encoder...\n");
    cogvit_model_t cogvit;
    if (!cogvit_load(&cogvit, &st)) {
        fprintf(stderr, "[FATAL] Failed to load CogViT\n");
        safetensors_close(&st);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    // ======== 4. Load GLM model ========
    fprintf(stderr, "[INFO] Loading GLM decoder...\n");
    glm_model_t glm;
    if (!glm_load(&glm, &st)) {
        fprintf(stderr, "[FATAL] Failed to load GLM\n");
        cogvit_free(&cogvit);
        safetensors_close(&st);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    // Done with safetensors file (weights are loaded into tensors)
    safetensors_close(&st);

    // ======== 5. Get image dimensions and compute target size ========
    fprintf(stderr, "[INFO] Getting image dimensions...\n");
    int img_w, img_h;
    if (!ocr_image_get_dimensions(image_path, &img_w, &img_h)) {
        fprintf(stderr, "[FATAL] Failed to get image dimensions\n");
        glm_free(&glm);
        cogvit_free(&cogvit);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    int target_w, target_h;
    if (fast_mode) {
        // Fast mode: use smaller target (~336px) for quick testing
        int max_dim = img_w > img_h ? img_w : img_h;
        float scale = 336.0f / max_dim;
        int sw = (int)(img_w * scale);
        int sh = (int)(img_h * scale);
        if (sw < 56) sw = 56;
        if (sh < 56) sh = 56;
        target_w = ((sw + 14) / 28) * 28;
        target_h = ((sh + 14) / 28) * 28;
    } else {
        ocr_compute_target_size(img_w, img_h, &target_w, &target_h);
    }
    int vis_grid_h = target_h / 28;
    int vis_grid_w = target_w / 28;
    int num_vis_tokens = vis_grid_h * vis_grid_w;
    fprintf(stderr, "[INFO] Image: %dx%d -> target: %dx%d, visual grid: %dx%d = %d tokens\n",
            img_w, img_h, target_w, target_h, vis_grid_h, vis_grid_w, num_vis_tokens);

    // ======== 6. Load and preprocess image ========
    fprintf(stderr, "[INFO] Loading image...\n");
    float mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
    float std[3] = { 0.26862954f, 0.26130258f, 0.27577711f };
    boat_tensor_t* image = ocr_image_load(image_path, target_w, target_h, mean, std);
    if (!image) {
        fprintf(stderr, "[FATAL] Failed to load image\n");
        glm_free(&glm);
        cogvit_free(&cogvit);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    // ======== 7. Encode image with CogViT ========
    boat_tensor_t* visual_tokens = NULL;
    if (fast_mode) {
        fprintf(stderr, "[INFO] FAST MODE: using random visual tokens\n");
        // Create random visual tokens to bypass CogViT
        int64_t vt_shape[] = { 1, num_vis_tokens, COGVIT_OUT_HIDDEN_SIZE };
        visual_tokens = boat_tensor_create(vt_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (visual_tokens) {
            float* vt_data = (float*)boat_tensor_data(visual_tokens);
            for (int i = 0; i < num_vis_tokens * COGVIT_OUT_HIDDEN_SIZE; i++)
                vt_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
    } else {
        fprintf(stderr, "[INFO] Encoding image with CogViT...\n");
        visual_tokens = cogvit_forward(&cogvit, image);
    }
    boat_tensor_unref(image);
    if (!visual_tokens) {
        fprintf(stderr, "[FATAL] CogViT forward failed\n");
        glm_free(&glm);
        cogvit_free(&cogvit);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    const float* vis_data = (const float*)boat_tensor_const_data(visual_tokens);
    // Verify token count matches expectation
    const int64_t* vis_shape = boat_tensor_shape(visual_tokens);
    int actual_vis_tokens = (int)vis_shape[1];
    if (actual_vis_tokens != num_vis_tokens) {
        fprintf(stderr, "[WARN] Expected %d visual tokens but CogViT produced %d\n",
                num_vis_tokens, actual_vis_tokens);
        num_vis_tokens = actual_vis_tokens;
    }
    fprintf(stderr, "[INFO] Got %d visual tokens from CogViT\n", num_vis_tokens);

    // ======== 8. Prepare decoder input ========
    // Build input: [gMASK, sop, user, img_start, image×N, img_end, prompt_tokens..., assistant]
    // Visual features from CogViT replace the <|image|> token embeddings
    int prompt_text_len = (int)strlen(prompt_text);
    int* prompt_ids = ocr_tokenizer_encode(&tokenizer, prompt_text, prompt_text_len, &prompt_text_len);
    if (!prompt_ids) {
        fprintf(stderr, "[FATAL] Tokenization failed\n");
        boat_tensor_unref(visual_tokens);
        glm_free(&glm);
        cogvit_free(&cogvit);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    // Build input_ids: [gMASK, sop, user, \n, img_start, image×N, img_end, prompt_tokens..., assistant, \n]
    int input_len = 1 + 1 + 1 + 1 + 1 + num_vis_tokens + 1 + prompt_text_len + 1 + 1;
    int* input_ids = (int*)malloc(input_len * sizeof(int));
    int pos = 0;
    input_ids[pos++] = tokenizer.gmask_id;          // [gMASK] at position 0
    input_ids[pos++] = tokenizer.sop_id;            // <sop> at position 1
    input_ids[pos++] = tokenizer.user_role_id;      // <|user|> at position 2
    input_ids[pos++] = tokenizer.newline_id;        // \n at position 3
    input_ids[pos++] = tokenizer.img_start_id;       // <|begin_of_image|> at position 4
    for (int i = 0; i < num_vis_tokens; i++)         // <|image|> × N at positions 5..N+4
        input_ids[pos++] = tokenizer.image_token_id;
    input_ids[pos++] = tokenizer.img_end_id;         // <|end_of_image|>
    memcpy(input_ids + pos, prompt_ids, prompt_text_len * sizeof(int));
    pos += prompt_text_len;
    input_ids[pos++] = tokenizer.assistant_role_id;  // <|assistant|>
    input_ids[pos++] = tokenizer.newline_id;          // \n at the end

    // Debug: print tokenized prompt
    int dbg_cnt = prompt_text_len < 30 ? prompt_text_len : 30;
    fprintf(stderr, "[DEBUG] Prompt '%s' -> %d tokens:", prompt_text, prompt_text_len);
    for (int i = 0; i < dbg_cnt; i++) fprintf(stderr, " %d", prompt_ids[i]);
    fprintf(stderr, "\n");
    free(prompt_ids);

    // ======== 9. Prefill: combine embeddings ========
    fprintf(stderr, "[INFO] Running prefill (%d tokens)...\n", input_len);

    // Create combined hidden states: embed text tokens, then replace image positions with visual features
    float* prefill_hidden = (float*)malloc(input_len * GLM_HIDDEN_SIZE * sizeof(float));
    const float* embed_w = (const float*)boat_tensor_const_data(glm.embed_tokens_weight);

    for (int i = 0; i < input_len; i++) {
        int id = input_ids[i];
        if (id < 0 || id >= GLM_VOCAB_SIZE) id = tokenizer.unk_id;
        memcpy(prefill_hidden + i * GLM_HIDDEN_SIZE,
               embed_w + id * GLM_HIDDEN_SIZE, GLM_HIDDEN_SIZE * sizeof(float));
    }

    // Replace <|image|> token embeddings (positions 4..N+3) with visual features from CogViT
    int vis_start = 5;  // position of first <|image|> token in input_ids
    memcpy(prefill_hidden + vis_start * GLM_HIDDEN_SIZE, vis_data,
           num_vis_tokens * GLM_HIDDEN_SIZE * sizeof(float));
    boat_tensor_unref(visual_tokens);

    // Debug: check input embeddings
    fprintf(stderr, "[DEBUG] sop embed:");
    for (int j = 0; j < 4; j++) fprintf(stderr, " %.6f", prefill_hidden[j]);
    fprintf(stderr, " ... last:");
    for (int j = GLM_HIDDEN_SIZE - 4; j < GLM_HIDDEN_SIZE; j++) fprintf(stderr, " %.6f", prefill_hidden[j]);
    float ns = 0; for (int j = 0; j < GLM_HIDDEN_SIZE; j++) ns += prefill_hidden[j]*prefill_hidden[j];
    fprintf(stderr, " norm=%.4f\n", sqrtf(ns));
    fprintf(stderr, "[DEBUG] first vis:");
    for (int j = vis_start*GLM_HIDDEN_SIZE; j < vis_start*GLM_HIDDEN_SIZE+4; j++) fprintf(stderr, " %.6f", prefill_hidden[j]);
    float nv = 0; for (int j = vis_start*GLM_HIDDEN_SIZE; j < (vis_start+1)*GLM_HIDDEN_SIZE; j++) nv += prefill_hidden[j]*prefill_hidden[j];
    fprintf(stderr, " norm=%.4f\n", sqrtf(nv));

    int total_prefill = input_len;
    int prefill_pos_end;  // for decode position continuation

    // Compute M-RoPE positions once (matching HuggingFace get_rope_index)
    // Text tokens get sequential positions. Vision tokens get positions offset by
    // current_pos, and text counter advances by max(grid_h, grid_w) after vision group.
    int* pos_t = (int*)malloc(total_prefill * sizeof(int));
    int* pos_h = (int*)malloc(total_prefill * sizeof(int));
    int* pos_w = (int*)malloc(total_prefill * sizeof(int));
    {
        int cur = 0, ti = 0;
        for (; ti < vis_start; ti++)
            pos_t[ti] = pos_h[ti] = pos_w[ti] = cur++;
        for (; ti < vis_start + num_vis_tokens; ti++) {
            int img_idx = ti - vis_start;
            int row = img_idx / vis_grid_w;
            int col = img_idx % vis_grid_w;
            pos_t[ti] = cur;
            pos_h[ti] = cur + row;
            pos_w[ti] = cur + col;
        }
        cur += vis_grid_h > vis_grid_w ? vis_grid_h : vis_grid_w;
        for (; ti < total_prefill; ti++)
            pos_t[ti] = pos_h[ti] = pos_w[ti] = cur++;
        prefill_pos_end = cur;
    }

    // Run decoder layers on the full prefill sequence (without KV cache for initial fill)
    // We compute layer-by-layer to build up the KV cache
    for (int l = 0; l < GLM_NUM_LAYERS; l++) {
        const glm_layer_weights_t* layer = &glm.layers[l];
        glm_kv_cache_t* cache = &glm.kv_caches[l];

        float* residual = (float*)malloc(total_prefill * GLM_HIDDEN_SIZE * sizeof(float));
        memcpy(residual, prefill_hidden, total_prefill * GLM_HIDDEN_SIZE * sizeof(float));

        // Pre-attention RMSNorm
        const float* in_ln = (const float*)boat_tensor_const_data(layer->input_layernorm_weight);
        for (int i = 0; i < total_prefill; i++)
            apply_rmsnorm(prefill_hidden + i * GLM_HIDDEN_SIZE,
                          residual + i * GLM_HIDDEN_SIZE, in_ln, GLM_HIDDEN_SIZE, 1e-5f);

        // GQA attention (prefill without KV cache, but store results)
        int q_size = GLM_NUM_HEADS * GLM_HEAD_DIM;
        int kv_size = GLM_NUM_KV_HEADS * GLM_HEAD_DIM;
        int groups = GLM_NUM_HEADS / GLM_NUM_KV_HEADS;

        const float* q_w = (const float*)boat_tensor_const_data(layer->q_proj_weight);
        const float* k_w = (const float*)boat_tensor_const_data(layer->k_proj_weight);
        const float* v_w = (const float*)boat_tensor_const_data(layer->v_proj_weight);
        const float* o_w = (const float*)boat_tensor_const_data(layer->o_proj_weight);

        float* q = (float*)malloc(total_prefill * q_size * sizeof(float));
        float* k = (float*)malloc(total_prefill * kv_size * sizeof(float));
        float* v = (float*)malloc(total_prefill * kv_size * sizeof(float));

        matmul_bt(q, prefill_hidden, q_w, total_prefill, GLM_HIDDEN_SIZE, q_size);
        matmul_bt(k, prefill_hidden, k_w, total_prefill, GLM_HIDDEN_SIZE, kv_size);
        matmul_bt(v, prefill_hidden, v_w, total_prefill, GLM_HIDDEN_SIZE, kv_size);

        apply_rope_mrope(q, k, total_prefill, GLM_NUM_HEADS, GLM_NUM_KV_HEADS, GLM_HEAD_DIM,
                         GLM_ROPE_THETA, pos_t, pos_h, pos_w);

        // Store K, V in KV cache
        float* k_cache_data = (float*)boat_tensor_data(cache->k_cache);
        float* v_cache_data = (float*)boat_tensor_data(cache->v_cache);
        for (int s = 0; s < total_prefill; s++) {
            memcpy(k_cache_data + s * kv_size, k + s * kv_size, kv_size * sizeof(float));
            memcpy(v_cache_data + s * kv_size, v + s * kv_size, kv_size * sizeof(float));
        }
        cache->seq_len = total_prefill;

        // GQA attention scores with causal mask (per-head scores)
        // scores layout: [seq_len, num_heads, seq_len]
        float* scores = (float*)malloc(total_prefill * GLM_NUM_HEADS * total_prefill * sizeof(float));
        for (int i = 0; i < total_prefill; i++) {
            for (int h = 0; h < GLM_NUM_HEADS; h++) {
                int kv_h = h / groups;
                for (int j = 0; j <= i; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < GLM_HEAD_DIM; d++) {
                        sum += q[i * q_size + h * GLM_HEAD_DIM + d]
                             * k[j * kv_size + kv_h * GLM_HEAD_DIM + d];
                    }
                    scores[(i * GLM_NUM_HEADS + h) * total_prefill + j] = sum / sqrtf((float)GLM_HEAD_DIM);
                }
                for (int j = i + 1; j < total_prefill; j++)
                    scores[(i * GLM_NUM_HEADS + h) * total_prefill + j] = -INFINITY;
            }
        }

        // Softmax per head
        for (int i = 0; i < total_prefill; i++) {
            for (int h = 0; h < GLM_NUM_HEADS; h++) {
                int base = (i * GLM_NUM_HEADS + h) * total_prefill;
                float max_val = scores[base];
                for (int j = 1; j <= i; j++)
                    if (scores[base + j] > max_val)
                        max_val = scores[base + j];
                float sum = 0.0f;
                for (int j = 0; j <= i; j++) {
                    scores[base + j] = expf(scores[base + j] - max_val);
                    sum += scores[base + j];
                }
                for (int j = 0; j <= i; j++)
                    scores[base + j] /= sum;
            }
        }

        // Weighted sum of values (per-head context)
        float* context = (float*)calloc(total_prefill * q_size, sizeof(float));
        for (int i = 0; i < total_prefill; i++) {
            for (int h = 0; h < GLM_NUM_HEADS; h++) {
                int kv_h = h / groups;
                for (int j = 0; j <= i; j++) {
                    float attn = scores[(i * GLM_NUM_HEADS + h) * total_prefill + j];
                    for (int d = 0; d < GLM_HEAD_DIM; d++) {
                        context[i * q_size + h * GLM_HEAD_DIM + d] += attn * v[j * kv_size + kv_h * GLM_HEAD_DIM + d];
                    }
                }
            }
        }
        free(q); free(k); free(v); free(scores);

        // Output projection
        float* attn_out = (float*)malloc(total_prefill * GLM_HIDDEN_SIZE * sizeof(float));
        matmul_bt(attn_out, context, o_w, total_prefill, q_size, GLM_HIDDEN_SIZE);
        free(context);

        // Post-self-attention RMSNorm
        const float* psa_ln = (const float*)boat_tensor_const_data(layer->post_self_attn_layernorm_weight);
        for (int i = 0; i < total_prefill; i++)
            apply_rmsnorm(attn_out + i * GLM_HIDDEN_SIZE, attn_out + i * GLM_HIDDEN_SIZE, psa_ln, GLM_HIDDEN_SIZE, 1e-5f);

        // Residual add
        for (int i = 0; i < total_prefill * GLM_HIDDEN_SIZE; i++)
            prefill_hidden[i] = residual[i] + attn_out[i];
        free(attn_out);

        memcpy(residual, prefill_hidden, total_prefill * GLM_HIDDEN_SIZE * sizeof(float));

        // Pre-MLP RMSNorm
        const float* pa_ln = (const float*)boat_tensor_const_data(layer->post_attention_layernorm_weight);
        for (int i = 0; i < total_prefill; i++)
            apply_rmsnorm(prefill_hidden + i * GLM_HIDDEN_SIZE, residual + i * GLM_HIDDEN_SIZE, pa_ln, GLM_HIDDEN_SIZE, 1e-5f);

        // SiLU FFN
        const float* gate_up_w = (const float*)boat_tensor_const_data(layer->gate_up_proj_weight);
        const float* down_w = (const float*)boat_tensor_const_data(layer->down_proj_weight);
        int ff_dim = GLM_INTERMEDIATE_SIZE;

        float* gate_up = (float*)malloc(total_prefill * 2 * ff_dim * sizeof(float));
        matmul_bt(gate_up, prefill_hidden, gate_up_w, total_prefill, GLM_HIDDEN_SIZE, 2 * ff_dim);
        for (int i = 0; i < total_prefill; i++) {
            for (int j = 0; j < ff_dim; j++) {
                float g = gate_up[i * 2 * ff_dim + j];
                float u = gate_up[i * 2 * ff_dim + ff_dim + j];
                gate_up[i * ff_dim + j] = silu(g) * u;
            }
        }

        float* mlp_out = (float*)malloc(total_prefill * GLM_HIDDEN_SIZE * sizeof(float));
        matmul_bt(mlp_out, gate_up, down_w, total_prefill, ff_dim, GLM_HIDDEN_SIZE);
        free(gate_up);

        // Post-MLP RMSNorm
        const float* pm_ln = (const float*)boat_tensor_const_data(layer->post_mlp_layernorm_weight);
        for (int i = 0; i < total_prefill; i++)
            apply_rmsnorm(mlp_out + i * GLM_HIDDEN_SIZE, mlp_out + i * GLM_HIDDEN_SIZE, pm_ln, GLM_HIDDEN_SIZE, 1e-5f);

        // Residual add
        for (int i = 0; i < total_prefill * GLM_HIDDEN_SIZE; i++)
            prefill_hidden[i] = residual[i] + mlp_out[i];
        free(residual);
        free(mlp_out);

        fprintf(stderr, "[INFO] Layer %d/%d prefill complete\n", l + 1, GLM_NUM_LAYERS);
    }
    free(pos_t); free(pos_h); free(pos_w);

    // Final RMSNorm
    const float* norm_w = (const float*)boat_tensor_const_data(glm.norm_weight);
    for (int i = 0; i < total_prefill; i++)
        apply_rmsnorm(prefill_hidden + i * GLM_HIDDEN_SIZE,
                      prefill_hidden + i * GLM_HIDDEN_SIZE, norm_w, GLM_HIDDEN_SIZE, 1e-5f);

    // Compute logits for the last position
    // Note: lm_head.weight is UNTIED from embed_tokens.weight (tie_word_embeddings=False),
    // so we must use lm_head.weight, not the embedding weights
    const float* lm_w = glm.lm_head_weight ?
        (const float*)boat_tensor_const_data(glm.lm_head_weight) :
        (const float*)boat_tensor_const_data(glm.embed_tokens_weight);

    float* logits = (float*)malloc(GLM_VOCAB_SIZE * sizeof(float));
    int last_pos = total_prefill - 1;
    for (int j = 0; j < GLM_VOCAB_SIZE; j++) {
        float sum = 0.0f;
        for (int k = 0; k < GLM_HIDDEN_SIZE; k++)
            sum += prefill_hidden[last_pos * GLM_HIDDEN_SIZE + k] * lm_w[j * GLM_HIDDEN_SIZE + k];
        logits[j] = sum;
    }

    // Debug: print top-k logits
    {
        int top_indices[10];
        float top_values[10];
        for (int i = 0; i < 10; i++) top_values[i] = -INFINITY;
        for (int j = 0; j < GLM_VOCAB_SIZE; j++) {
            for (int k = 0; k < 10; k++) {
                if (logits[j] > top_values[k]) {
                    for (int m = 9; m > k; m--) {
                        top_values[m] = top_values[m-1];
                        top_indices[m] = top_indices[m-1];
                    }
                    top_values[k] = logits[j];
                    top_indices[k] = j;
                    break;
                }
            }
        }
        fprintf(stderr, "[DEBUG] Prefill logits top-10:");
        for (int k = 0; k < 10; k++)
            fprintf(stderr, " %d(%.2f)", top_indices[k], top_values[k]);
        fprintf(stderr, "\n");
        // Decode the top token for debugging
        char* top_token_str = ocr_tokenizer_decode(&tokenizer, top_indices, 1);
        fprintf(stderr, "[DEBUG] top-1 token '%d' decodes to: '%s'\n",
                top_indices[0], top_token_str ? top_token_str : "(null)");
        free(top_token_str);
    }

    free(prefill_hidden);

    // ======== 10. Decode tokens ========
    fprintf(stderr, "[INFO] Generating...\n");

    int gen_tokens[MAX_GEN_TOKENS];
    int gen_count = 0;
    float temp = 0.1f;  // low temperature for OCR precision

    // First token from prefill logits
    int next_id = sample_topk(logits, GLM_VOCAB_SIZE, 5, temp);
    gen_tokens[gen_count++] = next_id;

    // Decode loop
    while (gen_count < MAX_GEN_TOKENS) {
        // Check for EOS (both <|endoftext|> and <|user|> per config.json)
        if (next_id == tokenizer.eos_id || next_id == tokenizer.user_role_id) break;

        // Embed the single token
        float* hidden = (float*)malloc(1 * GLM_HIDDEN_SIZE * sizeof(float));
        int id = next_id;
        if (id < 0 || id >= GLM_VOCAB_SIZE) id = tokenizer.unk_id;
        memcpy(hidden, embed_w + id * GLM_HIDDEN_SIZE, GLM_HIDDEN_SIZE * sizeof(float));

        // Single-step decode through all layers
        float norm_hidden[GLM_HIDDEN_SIZE];
        for (int l = 0; l < GLM_NUM_LAYERS; l++) {
            const glm_layer_weights_t* layer = &glm.layers[l];
            glm_kv_cache_t* cache = &glm.kv_caches[l];
            int kv_len = cache->seq_len;

            // RMSNorm
            apply_rmsnorm(norm_hidden, hidden, (const float*)boat_tensor_const_data(layer->input_layernorm_weight),
                          GLM_HIDDEN_SIZE, 1e-5f);

            // QKV projections
            int q_size = GLM_NUM_HEADS * GLM_HEAD_DIM;
            int kv_size = GLM_NUM_KV_HEADS * GLM_HEAD_DIM;
            int groups = GLM_NUM_HEADS / GLM_NUM_KV_HEADS;

            const float* q_w = (const float*)boat_tensor_const_data(layer->q_proj_weight);
            const float* k_w = (const float*)boat_tensor_const_data(layer->k_proj_weight);
            const float* v_w = (const float*)boat_tensor_const_data(layer->v_proj_weight);
            const float* o_w = (const float*)boat_tensor_const_data(layer->o_proj_weight);

            float q[2048], k[1024], v[1024];
            matmul_bt(q, norm_hidden, q_w, 1, GLM_HIDDEN_SIZE, q_size);
            matmul_bt(k, norm_hidden, k_w, 1, GLM_HIDDEN_SIZE, kv_size);
            matmul_bt(v, norm_hidden, v_w, 1, GLM_HIDDEN_SIZE, kv_size);

            // M-RoPE for decode: text tokens use position = prefill_pos_end + gen_count - 1
            int abs_pos = prefill_pos_end + gen_count - 1;
            apply_rope_mrope(q, k, 1, GLM_NUM_HEADS, GLM_NUM_KV_HEADS, GLM_HEAD_DIM, GLM_ROPE_THETA,
                             &abs_pos, &abs_pos, &abs_pos);

            // Append to KV cache
            float* k_cache_data = (float*)boat_tensor_data(cache->k_cache);
            float* v_cache_data = (float*)boat_tensor_data(cache->v_cache);
            memcpy(k_cache_data + kv_len * kv_size, k, kv_size * sizeof(float));
            memcpy(v_cache_data + kv_len * kv_size, v, kv_size * sizeof(float));
            cache->seq_len = kv_len + 1;
            int new_kv_len = kv_len + 1;

            // GQA attention with per-head scores
            float score[16 * 4096];  // [num_heads, max_kv_len]
            float max_score[16];
            for (int h = 0; h < GLM_NUM_HEADS; h++) max_score[h] = -INFINITY;

            for (int j = 0; j < new_kv_len; j++) {
                for (int h = 0; h < GLM_NUM_HEADS; h++) {
                    int kv_h = h / groups;
                    float sum = 0.0f;
                    for (int d = 0; d < GLM_HEAD_DIM; d++)
                        sum += q[h * GLM_HEAD_DIM + d] * k_cache_data[j * kv_size + kv_h * GLM_HEAD_DIM + d];
                    score[h * new_kv_len + j] = sum / sqrtf((float)GLM_HEAD_DIM);
                    if (score[h * new_kv_len + j] > max_score[h])
                        max_score[h] = score[h * new_kv_len + j];
                }
            }

            float sum_exp[16] = {0};
            for (int h = 0; h < GLM_NUM_HEADS; h++) {
                for (int j = 0; j < new_kv_len; j++) {
                    score[h * new_kv_len + j] = expf(score[h * new_kv_len + j] - max_score[h]);
                    sum_exp[h] += score[h * new_kv_len + j];
                }
            }

            float context[2048] = {0};
            for (int j = 0; j < new_kv_len; j++) {
                for (int h = 0; h < GLM_NUM_HEADS; h++) {
                    int kv_h = h / groups;
                    float attn = score[h * new_kv_len + j] / sum_exp[h];
                    for (int d = 0; d < GLM_HEAD_DIM; d++)
                        context[h * GLM_HEAD_DIM + d] += attn * v_cache_data[j * kv_size + kv_h * GLM_HEAD_DIM + d];
                }
            }

            // Output projection
            float attn_out[1536];
            matmul_bt(attn_out, context, o_w, 1, q_size, GLM_HIDDEN_SIZE);

            // Post-attention norms + residual
            const float* psa_ln = (const float*)boat_tensor_const_data(layer->post_self_attn_layernorm_weight);
            apply_rmsnorm(attn_out, attn_out, psa_ln, GLM_HIDDEN_SIZE, 1e-5f);
            for (int i = 0; i < GLM_HIDDEN_SIZE; i++) hidden[i] += attn_out[i];

            // Pre-MLP RMSNorm
            float residual_mlp[GLM_HIDDEN_SIZE];
            memcpy(residual_mlp, hidden, GLM_HIDDEN_SIZE * sizeof(float));
            const float* pa_ln = (const float*)boat_tensor_const_data(layer->post_attention_layernorm_weight);
            apply_rmsnorm(hidden, residual_mlp, pa_ln, GLM_HIDDEN_SIZE, 1e-5f);

            // SiLU FFN
            const float* gate_up_w = (const float*)boat_tensor_const_data(layer->gate_up_proj_weight);
            const float* down_w = (const float*)boat_tensor_const_data(layer->down_proj_weight);
            int ff_dim = GLM_INTERMEDIATE_SIZE;

            float gate_up_buf[9216];
            matmul_bt(gate_up_buf, hidden, gate_up_w, 1, GLM_HIDDEN_SIZE, 2 * ff_dim);
            for (int j = 0; j < ff_dim; j++) {
                float g = gate_up_buf[j];
                float u = gate_up_buf[ff_dim + j];
                gate_up_buf[j] = silu(g) * u;
            }
            float mlp_out_buf[1536];
            matmul_bt(mlp_out_buf, gate_up_buf, down_w, 1, ff_dim, GLM_HIDDEN_SIZE);

            // Post-MLP norms + residual
            const float* pm_ln = (const float*)boat_tensor_const_data(layer->post_mlp_layernorm_weight);
            apply_rmsnorm(mlp_out_buf, mlp_out_buf, pm_ln, GLM_HIDDEN_SIZE, 1e-5f);
            for (int i = 0; i < GLM_HIDDEN_SIZE; i++) hidden[i] = residual_mlp[i] + mlp_out_buf[i];
        }

        // Final RMSNorm
        apply_rmsnorm(norm_hidden, hidden, norm_w, GLM_HIDDEN_SIZE, 1e-5f);

        // LM head
        for (int j = 0; j < GLM_VOCAB_SIZE; j++) {
            float sum = 0.0f;
            for (int k = 0; k < GLM_HIDDEN_SIZE; k++)
                sum += norm_hidden[k] * lm_w[j * GLM_HIDDEN_SIZE + k];
            logits[j] = sum;
        }

        free(hidden);

        // Sample next token
        next_id = sample_topk(logits, GLM_VOCAB_SIZE, 5, temp);
        gen_tokens[gen_count++] = next_id;

        if (gen_count % 50 == 0) {
            char* partial = ocr_tokenizer_decode(&tokenizer, gen_tokens, gen_count);
            fprintf(stderr, "[INFO] Generated %d tokens: '%s'\n", gen_count, partial ? partial : "null");
            free(partial);
        }
    }

    // ======== 11. Decode and output text ========
    char* text = ocr_tokenizer_decode(&tokenizer, gen_tokens, gen_count);
    if (text) {
        // Remove <|endoftext|> if present
        char* eos_pos = strstr(text, "<|endoftext|>");
        if (eos_pos) *eos_pos = '\0';
        printf("=== OCR Result ===\n%s\n==================\n", text);
        fflush(stdout);
        free(text);
    }

    fprintf(stderr, "[INFO] Generated %d tokens\n", gen_count);

    // ======== Cleanup ========
    free(input_ids);
    free(logits);
    glm_free(&glm);
    cogvit_free(&cogvit);
    ocr_tokenizer_free(&tokenizer);

    fprintf(stderr, "[INFO] Done\n");
    return 0;
}
