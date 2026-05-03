// ocr_cuda.c - CUDA-accelerated GLM-OCR inference engine
// Processes images and generates text using GPU-accelerated CogViT + GLM.
//
// Build:
//   cd build && cmake .. -DBOAT_WITH_CUDA=ON -DBOAT_WITH_EXAMPLES=ON && cmake --build .
// Usage:
//   ocr_cuda <model_dir> <image_path> [prompt] [--fast]
//
// Example:
//   ocr_cuda D:/huggingface/GLM-OCR test.png "请描述这张图片"

#include <boat.h>
#include <boat/tensor.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

#include "../common/safetensors.h"
#include "tokenizer.h"
#include "image.h"
#include "sampling.h"
#include "cogvit_cuda.cuh"
#include "glm_cuda.cuh"

#define DEFAULT_PROMPT "Please read the text in this image"
#define MAX_GEN_TOKENS 200

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                        \
    }                                                                   \
} while(0)

// Load embedding table from safetensors for CPU-side token lookups.
// Returns a boat_tensor with shape [vocab, hidden] or NULL.
static boat_tensor_t* load_embed_table(safetensors_t* st) {
    int idx = safetensors_find(st, "model.language_model.embed_tokens.weight");
    if (idx < 0) { fprintf(stderr, "[MAIN] embed_tokens.weight not found\n"); return NULL; }
    return safetensors_load_tensor(st, idx, 0);
}

// Embed a single token ID into a pre-allocated output vector.
static void embed_one(float* out, const float* embed_w, int id, int hidden_size, int unk_id, int vocab_size) {
    if (id < 0 || id >= vocab_size) id = unk_id;
    memcpy(out, embed_w + id * hidden_size, hidden_size * sizeof(float));
}

int main(int argc, char** argv) {
    int fast_mode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fast") == 0) {
            fast_mode = 1;
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

    char model_path[1024], tokenizer_path[1024];
    snprintf(model_path, sizeof(model_path), "%s/model.safetensors", model_dir);
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", model_dir);

    srand((unsigned int)time(NULL));

    fprintf(stderr, "[INFO] GLM-OCR CUDA Inference Engine\n");
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

    // Load embedding table for CPU-side lookups
    boat_tensor_t* embed_t = load_embed_table(&st);
    if (!embed_t) {
        fprintf(stderr, "[FATAL] Failed to load embedding table\n");
        safetensors_close(&st);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }
    const float* embed_h = (const float*)boat_tensor_const_data(embed_t);

    // ======== 3. Load CogViT CUDA model ========
    fprintf(stderr, "[INFO] Loading CogViT vision encoder (CUDA)...\n");
    cogvit_cuda_model_t cogvit_cuda;
    if (!cogvit_cuda_load(&cogvit_cuda, &st)) {
        fprintf(stderr, "[FATAL] Failed to load CogViT CUDA model\n");
        boat_tensor_unref(embed_t);
        safetensors_close(&st);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    // ======== 4. Load GLM CUDA model ========
    fprintf(stderr, "[INFO] Loading GLM decoder (CUDA)...\n");
    glm_cuda_model_t glm_cuda;
    if (!glm_cuda_load(&glm_cuda, &st)) {
        fprintf(stderr, "[FATAL] Failed to load GLM CUDA model\n");
        cogvit_cuda_free(&cogvit_cuda);
        boat_tensor_unref(embed_t);
        safetensors_close(&st);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    safetensors_close(&st);
    fprintf(stderr, "[INFO] All models loaded on GPU\n");

    // ======== 5. Get image dimensions ========
    int img_w, img_h;
    if (!ocr_image_get_dimensions(image_path, &img_w, &img_h)) {
        fprintf(stderr, "[FATAL] Failed to get image dimensions\n");
        glm_cuda_free(&glm_cuda);
        cogvit_cuda_free(&cogvit_cuda);
        boat_tensor_unref(embed_t);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    int target_w, target_h;
    if (fast_mode) {
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
    float mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
    float std[3]  = { 0.26862954f, 0.26130258f, 0.27577711f };
    boat_tensor_t* image = ocr_image_load(image_path, target_w, target_h, mean, std);
    if (!image) {
        fprintf(stderr, "[FATAL] Failed to load image\n");
        glm_cuda_free(&glm_cuda);
        cogvit_cuda_free(&cogvit_cuda);
        boat_tensor_unref(embed_t);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    // ======== 7. Encode image with CogViT CUDA ========
    boat_tensor_t* visual_tokens = NULL;
    if (fast_mode) {
        fprintf(stderr, "[INFO] FAST MODE: using random visual tokens\n");
        int64_t vt_shape[] = { 1, num_vis_tokens, COGVIT_OUT_HIDDEN_SIZE };
        visual_tokens = boat_tensor_create(vt_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (visual_tokens) {
            float* vt_data = (float*)boat_tensor_data(visual_tokens);
            for (int i = 0; i < num_vis_tokens * COGVIT_OUT_HIDDEN_SIZE; i++)
                vt_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
    } else {
        fprintf(stderr, "[INFO] Encoding image with CogViT CUDA...\n");
        visual_tokens = cogvit_cuda_forward(&cogvit_cuda, image);
        // Diagnostic: check CogViT output
        if (visual_tokens) {
            const float* vt_check = (const float*)boat_tensor_const_data(visual_tokens);
            double vt_norm = 0.0;
            int n_elems = (int)boat_tensor_nelements(visual_tokens);
            for (int i = 0; i < n_elems && i < 100; i++) vt_norm += (double)vt_check[i] * (double)vt_check[i];
            fprintf(stderr, "[DIAG] CogViT output first_vals=[%.6f %.6f %.6f %.6f] partial_norm=%.6f\n",
                    vt_check[0], vt_check[1], vt_check[2], vt_check[3], sqrt(vt_norm));
            // Full norm
            vt_norm = 0.0;
            for (int i = 0; i < n_elems; i++) vt_norm += (double)vt_check[i] * (double)vt_check[i];
            fprintf(stderr, "[DIAG] CogViT output full_norm=%.4f\n", sqrt(vt_norm));
        }
    }
    boat_tensor_unref(image);
    if (!visual_tokens) {
        fprintf(stderr, "[FATAL] CogViT forward failed\n");
        glm_cuda_free(&glm_cuda);
        cogvit_cuda_free(&cogvit_cuda);
        boat_tensor_unref(embed_t);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    const float* vis_data = (const float*)boat_tensor_const_data(visual_tokens);
    const int64_t* vis_shape = boat_tensor_shape(visual_tokens);
    int actual_vis_tokens = (int)vis_shape[1];
    if (actual_vis_tokens != num_vis_tokens) {
        fprintf(stderr, "[WARN] Expected %d visual tokens but CogViT produced %d\n",
                num_vis_tokens, actual_vis_tokens);
        num_vis_tokens = actual_vis_tokens;
    }
    fprintf(stderr, "[INFO] Got %d visual tokens from CogViT CUDA\n", num_vis_tokens);

    // ======== 8. Prepare decoder input ========
    int prompt_text_len = (int)strlen(prompt_text);
    int* prompt_ids = ocr_tokenizer_encode(&tokenizer, prompt_text, prompt_text_len, &prompt_text_len);
    if (!prompt_ids) {
        fprintf(stderr, "[FATAL] Tokenization failed\n");
        boat_tensor_unref(visual_tokens);
        glm_cuda_free(&glm_cuda);
        cogvit_cuda_free(&cogvit_cuda);
        boat_tensor_unref(embed_t);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    int input_len = 1 + 1 + 1 + 1 + 1 + num_vis_tokens + 1 + prompt_text_len + 1 + 1;
    int* input_ids = (int*)malloc(input_len * sizeof(int));
    int pos = 0;
    input_ids[pos++] = tokenizer.gmask_id;
    input_ids[pos++] = tokenizer.sop_id;
    input_ids[pos++] = tokenizer.user_role_id;
    input_ids[pos++] = tokenizer.newline_id;
    input_ids[pos++] = tokenizer.img_start_id;
    for (int i = 0; i < num_vis_tokens; i++)
        input_ids[pos++] = tokenizer.image_token_id;
    input_ids[pos++] = tokenizer.img_end_id;
    memcpy(input_ids + pos, prompt_ids, prompt_text_len * sizeof(int));
    pos += prompt_text_len;
    input_ids[pos++] = tokenizer.assistant_role_id;
    input_ids[pos++] = tokenizer.newline_id;

    fprintf(stderr, "[DEBUG] Prompt '%s' -> %d tokens\n", prompt_text, prompt_text_len);
    free(prompt_ids);

    // ======== 9. Prefill: build hidden states ========
    fprintf(stderr, "[INFO] Running prefill (%d tokens)...\n", input_len);

    // Build combined hidden states on CPU
    float* prefill_hidden = (float*)malloc(input_len * GLM_HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < input_len; i++) {
        embed_one(prefill_hidden + i * GLM_HIDDEN_SIZE,
                   embed_h, input_ids[i], GLM_HIDDEN_SIZE,
                   tokenizer.unk_id, GLM_VOCAB_SIZE);
    }

    // Replace <|image|> token positions with visual features
    int vis_start = 5;
    memcpy(prefill_hidden + vis_start * GLM_HIDDEN_SIZE, vis_data,
           num_vis_tokens * GLM_HIDDEN_SIZE * sizeof(float));
    // Diagnostic: print first visual token values and norm, compare with CPU
    {
        float* first_vis = prefill_hidden + vis_start * GLM_HIDDEN_SIZE;
        double nv = 0.0;
        for (int k = 0; k < GLM_HIDDEN_SIZE; k++)
            nv += (double)first_vis[k] * (double)first_vis[k];
        fprintf(stderr, "[DIAG] first vis: %.6f %.6f %.6f %.6f norm=%.4f\n",
                first_vis[0], first_vis[1], first_vis[2], first_vis[3], sqrt(nv));
        float* last_h = prefill_hidden + (input_len - 1) * GLM_HIDDEN_SIZE;
        double nl = 0.0;
        for (int k = 0; k < GLM_HIDDEN_SIZE; k++)
            nl += (double)last_h[k] * (double)last_h[k];
        fprintf(stderr, "[DIAG] last embed: %.6f %.6f %.6f %.6f norm=%.4f\n",
                last_h[0], last_h[1], last_h[2], last_h[3], sqrt(nl));
    }
    boat_tensor_unref(visual_tokens);

    int total_prefill = input_len;
    int prefill_pos_end;

    // Compute M-RoPE positions
    int* pos_t = (int*)malloc(total_prefill * sizeof(int));
    int* pos_h = (int*)malloc(total_prefill * sizeof(int));
    int* pos_w = (int*)malloc(total_prefill * sizeof(int));
    glm_compute_rope_positions(pos_t, pos_h, pos_w,
                                total_prefill, vis_start,
                                num_vis_tokens, vis_grid_h, vis_grid_w);
    prefill_pos_end = pos_t[total_prefill - 1] + 1;

    // Copy hidden states to GPU
    float* d_hidden;
    CUDA_CHECK(cudaMalloc(&d_hidden, (size_t)total_prefill * GLM_HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_hidden, prefill_hidden,
                           (size_t)total_prefill * GLM_HIDDEN_SIZE * sizeof(float),
                           cudaMemcpyHostToDevice));
    free(prefill_hidden);

    // Run CUDA prefill through all layers
    boat_tensor_t* prefill_logits = glm_cuda_forward(&glm_cuda, d_hidden,
                                                       total_prefill, prefill_pos_end, 0,
                                                       pos_t, pos_h, pos_w);
    CUDA_CHECK(cudaFree(d_hidden));
    free(pos_t); free(pos_h); free(pos_w);

    if (!prefill_logits) {
        fprintf(stderr, "[FATAL] GLM prefill failed\n");
        free(input_ids);
        glm_cuda_free(&glm_cuda);
        cogvit_cuda_free(&cogvit_cuda);
        boat_tensor_unref(embed_t);
        ocr_tokenizer_free(&tokenizer);
        return 1;
    }

    // Debug: print top-k logits
    {
        float* logits_h = (float*)boat_tensor_data(prefill_logits);
        int top_indices[10];
        float top_values[10];
        for (int i = 0; i < 10; i++) top_values[i] = -FLT_MAX;
        for (int j = 0; j < GLM_VOCAB_SIZE; j++) {
            for (int k = 0; k < 10; k++) {
                if (logits_h[j] > top_values[k]) {
                    for (int m = 9; m > k; m--) {
                        top_values[m] = top_values[m-1];
                        top_indices[m] = top_indices[m-1];
                    }
                    top_values[k] = logits_h[j];
                    top_indices[k] = j;
                    break;
                }
            }
        }
        fprintf(stderr, "[DEBUG] Prefill logits top-10:");
        for (int k = 0; k < 10; k++)
            fprintf(stderr, " %d(%.2f)", top_indices[k], top_values[k]);
        fprintf(stderr, "\n");
        char* top_token_str = ocr_tokenizer_decode(&tokenizer, top_indices, 1);
        fprintf(stderr, "[DEBUG] top-1 token '%d' decodes to: '%s'\n",
                top_indices[0], top_token_str ? top_token_str : "(null)");
        free(top_token_str);
    }

    // ======== 10. Decode ========
    fprintf(stderr, "[INFO] Generating...\n");

    int gen_tokens[MAX_GEN_TOKENS];
    int gen_count = 0;
    float temp = 0.0f;  // greedy

    float* logits_data = (float*)boat_tensor_data(prefill_logits);
    int next_id = sample_topk(logits_data, GLM_VOCAB_SIZE, 5, temp);
    boat_tensor_unref(prefill_logits);
    gen_tokens[gen_count++] = next_id;

    // Allocate device buffer for single-token embeddings (reused per step)
    float* d_embed;
    CUDA_CHECK(cudaMalloc(&d_embed, (size_t)GLM_HIDDEN_SIZE * sizeof(float)));

    float single_embed[GLM_HIDDEN_SIZE];

    while (gen_count < MAX_GEN_TOKENS) {
        if (next_id == tokenizer.eos_id || next_id == tokenizer.user_role_id) break;

        embed_one(single_embed, embed_h, next_id, GLM_HIDDEN_SIZE,
                   tokenizer.unk_id, GLM_VOCAB_SIZE);

        CUDA_CHECK(cudaMemcpy(d_embed, single_embed,
                               (size_t)GLM_HIDDEN_SIZE * sizeof(float),
                               cudaMemcpyHostToDevice));

        int abs_pos = prefill_pos_end + gen_count - 1;
        boat_tensor_t* logits_t = glm_cuda_decode_step(&glm_cuda, d_embed, abs_pos);
        if (!logits_t) {
            fprintf(stderr, "[FATAL] Decode step failed at token %d\n", gen_count);
            break;
        }

        float* logits_h = (float*)boat_tensor_data(logits_t);
        if (gen_count <= 5) {
            int topk_idx[3]; float topk_val[3];
            for (int i = 0; i < 3; i++) topk_val[i] = -FLT_MAX;
            for (int j = 0; j < GLM_VOCAB_SIZE; j++) {
                for (int k = 0; k < 3; k++) {
                    if (logits_h[j] > topk_val[k]) {
                        for (int m = 2; m > k; m--) { topk_val[m] = topk_val[m-1]; topk_idx[m] = topk_idx[m-1]; }
                        topk_val[k] = logits_h[j]; topk_idx[k] = j;
                        break;
                    }
                }
            }
            char* t0 = ocr_tokenizer_decode(&tokenizer, topk_idx, 1);
            fprintf(stderr, "[DECODE] step %d top-3: %d(%f) %d(%f) %d(%f) t0='%s'\n",
                    gen_count, topk_idx[0], topk_val[0], topk_idx[1], topk_val[1], topk_idx[2], topk_val[2],
                    t0 ? t0 : "?");
            free(t0);
        }
        next_id = sample_topk(logits_h, GLM_VOCAB_SIZE, 5, temp);
        boat_tensor_unref(logits_t);

        gen_tokens[gen_count++] = next_id;

        if (gen_count % 50 == 0) {
            char* partial = ocr_tokenizer_decode(&tokenizer, gen_tokens, gen_count);
            fprintf(stderr, "[INFO] Generated %d tokens: '%s'\n", gen_count, partial ? partial : "null");
            free(partial);
        }
    }

    CUDA_CHECK(cudaFree(d_embed));

    // ======== 11. Decode and output ========
    char* text = ocr_tokenizer_decode(&tokenizer, gen_tokens, gen_count);
    if (text) {
        char* eos_pos = strstr(text, "<|endoftext|>");
        if (eos_pos) *eos_pos = '\0';
        printf("=== OCR Result ===\n%s\n==================\n", text);
        fflush(stdout);
        free(text);
    }

    fprintf(stderr, "[INFO] Generated %d tokens\n", gen_count);

    // ======== Cleanup ========
    free(input_ids);
    glm_cuda_free(&glm_cuda);
    cogvit_cuda_free(&cogvit_cuda);
    boat_tensor_unref(embed_t);
    ocr_tokenizer_free(&tokenizer);

    fprintf(stderr, "[INFO] Done\n");
    return 0;
}
