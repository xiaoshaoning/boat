// main.cu — Qwen3-ASR CUDA CLI entry point
#include "model.h"
#include "kernels.cuh"
#include "../qwen3-asr/config.h"
#include "../qwen3-asr/decoder.h"
#include "../qwen3-asr/tokenizer.h"
#include <boat/sampling.h>
#include <boat/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                        \
    }                                                                   \
} while(0)

static float get_time_sec(void) {
    return (float)clock() / (float)CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: qwen3asr_cuda <model_dir> <mel_binary>\n");
        return 1;
    }

    const char *model_dir = argv[1];
    const char *mel_path = argv[2];

    // ---- 1. Load weights on CPU ----
    printf("Loading weights from %s ...\n", model_dir);
    fflush(stdout);
    float t0 = get_time_sec();
    qwen3asr_weights_t *w = qwen3asr_weights_load(model_dir);
    if (!w) { fprintf(stderr, "ERROR: failed to load weights\n"); return 1; }
    printf("  done (%.2fs)\n", get_time_sec() - t0);

    // ---- 2. Upload weights to GPU ----
    printf("Uploading weights to GPU ...\n");
    t0 = get_time_sec();
    qwen3asr_cuda_model_t model;
    if (!qwen3asr_cuda_model_init(&model, w)) {
        fprintf(stderr, "ERROR: GPU model init failed\n");
        qwen3asr_weights_free(w);
        return 1;
    }
    printf("  done (%.2fs)\n", get_time_sec() - t0);

    // ---- 3. Read mel binary ----
    printf("Reading mel from %s ...\n", mel_path);
    FILE *f = fopen(mel_path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", mel_path); return 1; }

    int T_mel;
    if (fread(&T_mel, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "ERROR: failed to read mel header\n"); fclose(f); return 1;
    }
    if (T_mel <= 0 || T_mel > 10000) {
        fprintf(stderr, "ERROR: invalid mel frame count: %d\n", T_mel); fclose(f); return 1;
    }

    float *mel_data = (float*)malloc((size_t)QWEN3ASR_MEL_BINS * T_mel * sizeof(float));
    if (!mel_data) { fclose(f); return 1; }
    size_t mel_bytes = (size_t)QWEN3ASR_MEL_BINS * T_mel * sizeof(float);
    if (fread(mel_data, 1, mel_bytes, f) != mel_bytes) {
        fprintf(stderr, "ERROR: failed to read mel data\n");
        free(mel_data); fclose(f); return 1;
    }
    fclose(f);
    printf("  %d mel frames\n", T_mel);

    // Upload mel to GPU
    float *d_mel;
    CUDA_CHECK(cudaMalloc(&d_mel, mel_bytes));
    CUDA_CHECK(cudaMemcpy(d_mel, mel_data, mel_bytes, cudaMemcpyHostToDevice));
    free(mel_data);

    // ---- 4. Encoder forward on GPU ----
    printf("Running encoder (GPU) ...\n");
    t0 = get_time_sec();
    float *d_audio = qwen3asr_cuda_encoder_forward(&model, d_mel, T_mel);
    CUDA_CHECK(cudaFree(d_mel));
    if (!d_audio) { fprintf(stderr, "ERROR: encoder forward failed\n"); return 1; }

    // Get encoder output dimensions
    int T_audio;
    {
        // Infer T_audio from valid frames calculation
        int feat_len = (T_mel - 1) / 2 + 1;
        int temp = (feat_len - 1) / 2 + 1;
        T_audio = (temp - 1) / 2 + 1;
    }

    // Download encoder output to CPU
    float *audio_features = (float*)malloc((size_t)T_audio * QWEN3ASR_ENCODER_OUTPUT_DIM * sizeof(float));
    CUDA_CHECK(cudaMemcpy(audio_features, d_audio,
        (size_t)T_audio * QWEN3ASR_ENCODER_OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_audio));
    printf("  audio features: [%d, %d] (%.2fs)\n",
           T_audio, QWEN3ASR_ENCODER_OUTPUT_DIM, get_time_sec() - t0);

    // ---- 5. Build prompt (CPU) ----
    printf("Building prompt ...\n");
    t0 = get_time_sec();

    int prompt_tokens[] = QWEN3ASR_PROMPT_TOKENS;
    int prompt_len = QWEN3ASR_PROMPT_LEN;
    int H = QWEN3ASR_DECODER_HIDDEN_SIZE;

    float *embeddings = (float*)calloc((size_t)prompt_len * H, sizeof(float));
    for (int i = 0; i < prompt_len; i++) {
        if (i == QWEN3ASR_PROMPT_PLACEHOLDER_POS) continue;
        int token = prompt_tokens[i];
        if (token >= 0 && token < QWEN3ASR_VOCAB_SIZE) {
            memcpy(embeddings + (size_t)i * H,
                   w->embed_tokens + (size_t)token * H,
                   (size_t)H * sizeof(float));
        }
    }

    int before_len = QWEN3ASR_PROMPT_PLACEHOLDER_POS;
    int after_len = prompt_len - before_len - 1;
    int total_len = before_len + T_audio + after_len;

    float *merged = (float*)malloc((size_t)total_len * H * sizeof(float));
    memcpy(merged, embeddings, (size_t)before_len * H * sizeof(float));
    memcpy(merged + (size_t)before_len * H, audio_features,
           (size_t)T_audio * H * sizeof(float));
    memcpy(merged + (size_t)(before_len + T_audio) * H,
           embeddings + (size_t)(before_len + 1) * H,
           (size_t)after_len * H * sizeof(float));
    free(embeddings);
    free(audio_features);
    printf("  merged sequence: %d tokens (%.2fs)\n", total_len, get_time_sec() - t0);

    // Upload merged embeddings to GPU
    float *d_merged;
    CUDA_CHECK(cudaMalloc(&d_merged, (size_t)total_len * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_merged, merged, (size_t)total_len * H * sizeof(float),
                           cudaMemcpyHostToDevice));
    free(merged);

    // ---- 6. Decoder prefix forward on GPU ----
    printf("Running decoder prefix (GPU) ...\n");
    t0 = get_time_sec();
    float *d_logits = qwen3asr_cuda_decoder_forward(&model, d_merged, total_len);
    CUDA_CHECK(cudaFree(d_merged));
    if (!d_logits) { fprintf(stderr, "ERROR: decoder forward failed\n"); return 1; }
    printf("  prefix done (%.2fs)\n", get_time_sec() - t0);

    // ---- 7. Autoregressive decode loop ----
    printf("Generating ...\n");
    t0 = get_time_sec();

    int V = QWEN3ASR_VOCAB_SIZE;
    float *h_logits = (float*)malloc((size_t)V * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_logits, d_logits, (size_t)V * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_logits));

    int output_ids[QWEN3ASR_MAX_NEW_TOKENS];
    int n_output = 0;
    int current_pos = total_len;

    // Sampled first token
    int next_token = boat_sample_token(h_logits, V, 10, 1.0f);
    printf("  sampled first token: %d\n", next_token);

    while (next_token != QWEN3ASR_EOS_ID && next_token != 0
           && n_output < QWEN3ASR_MAX_NEW_TOKENS) {
        output_ids[n_output++] = next_token;

        // Copy embedding from GPU table to single buffer
        CUDA_CHECK(cudaMemcpy(model.d_single,
            model.d_embed_tokens + (size_t)next_token * H,
            (size_t)H * sizeof(float), cudaMemcpyDeviceToDevice));

        // Decoder step
        d_logits = qwen3asr_cuda_decoder_step(&model, model.d_single, current_pos);
        if (!d_logits) {
            fprintf(stderr, "WARN: decoder step failed at pos %d\n", current_pos);
            break;
        }
        current_pos++;

        CUDA_CHECK(cudaMemcpy(h_logits, d_logits, (size_t)V * sizeof(float),
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_logits));

        next_token = boat_sample_token(h_logits, V, 10, 1.0f);
    }

    printf("  generated %d tokens (%.2fs)\n", n_output, get_time_sec() - t0);
    printf("  token IDs: ");
    for (int i = 0; i < n_output; i++) printf("%d ", output_ids[i]);
    printf("\n");
    free(h_logits);

    // ---- 8. Decode tokens to text ----
    char *text = qwen3asr_decode_tokens(model_dir, output_ids, n_output);
    printf("\n=== Transcription ===\n");
    printf("%s\n", text ? text : "(null)");
    free(text);

    // ---- Cleanup ----
    qwen3asr_cuda_model_free(&model);
    qwen3asr_weights_free(w);
    printf("\nDone.\n");
    return 0;
}
