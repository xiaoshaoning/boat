// main.c — Qwen3-ASR CLI entry point
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0
//
// Usage: qwen3asr <model_dir> <mel_binary>
//
// model_dir: directory containing qwen3_asr_weights.safetensors
// mel_binary: file from extract_mel.py (int32 T + float[128][T])

#include "encoder.h"
#include "decoder.h"
#include "weights.h"
#include "config.h"
#include "tokenizer.h"

#include <boat/ops.h>
#include <boat/tensor.h>
#include <boat/sampling.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static float get_time_sec(void) {
    return (float)clock() / (float)CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: qwen3asr <model_dir> <mel_binary>\n");
        return 1;
    }

    const char *model_dir = argv[1];
    const char *mel_path = argv[2];

    // ---- 1. Load weights ----
    printf("Loading weights from %s ...\n", model_dir);
    fflush(stdout);
    float t0 = get_time_sec();
    qwen3asr_weights_t *w = qwen3asr_weights_load(model_dir);
    if (!w) { fprintf(stderr, "ERROR: failed to load weights\n"); return 1; }
    printf("  done (%.2fs)\n", get_time_sec() - t0);

    // ---- 2. Read mel binary ----
    printf("Reading mel from %s ...\n", mel_path);
    FILE *f = fopen(mel_path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", mel_path); return 1; }

    int T_mel;
    if (fread(&T_mel, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "ERROR: failed to read mel header\n");
        fclose(f);
        return 1;
    }
    if (T_mel <= 0 || T_mel > 10000) {
        fprintf(stderr, "ERROR: invalid mel frame count: %d\n", T_mel);
        fclose(f);
        return 1;
    }

    float *mel_data = (float*)malloc((size_t)QWEN3ASR_MEL_BINS * T_mel * sizeof(float));
    if (!mel_data) { fclose(f); return 1; }
    size_t mel_bytes = (size_t)QWEN3ASR_MEL_BINS * T_mel * sizeof(float);
    if (fread(mel_data, 1, mel_bytes, f) != mel_bytes) {
        fprintf(stderr, "ERROR: failed to read mel data\n");
        free(mel_data);
        fclose(f);
        return 1;
    }
    fclose(f);

    const int64_t mel_shape[] = {QWEN3ASR_MEL_BINS, T_mel};
    boat_tensor_t *mel = boat_tensor_from_data(mel_shape, 2, BOAT_DTYPE_FLOAT32, mel_data);
    free(mel_data);
    printf("  %d mel frames\n", T_mel);

    // ---- 3. Encoder forward ----
    printf("Running encoder ...\n");
    t0 = get_time_sec();
    qwen3asr_encoder_t *encoder = qwen3asr_encoder_create(w);
    if (!encoder) { fprintf(stderr, "ERROR: encoder create failed\n"); return 1; }

    boat_tensor_t *audio_features = qwen3asr_encoder_forward(encoder, mel);
    qwen3asr_encoder_free(encoder);
    boat_tensor_unref(mel);

    if (!audio_features) { fprintf(stderr, "ERROR: encoder forward failed\n"); return 1; }
    int T_audio = (int)boat_tensor_shape(audio_features)[0];
    printf("  audio features: [%d, 1024] (%.2fs)\n", T_audio, get_time_sec() - t0);
    // ---- 4. Build prompt with audio merged ----
    printf("Building prompt ...\n");
    t0 = get_time_sec();

    boat_tensor_t *embed_weight = boat_tensor_from_data(
        (int64_t[]){QWEN3ASR_VOCAB_SIZE, QWEN3ASR_DECODER_HIDDEN_SIZE}, 2,
        BOAT_DTYPE_FLOAT32, w->embed_tokens);

    boat_tensor_t *merged = qwen3asr_build_prompt(embed_weight, audio_features);
    boat_tensor_unref(embed_weight);
    boat_tensor_unref(audio_features);

    if (!merged) { fprintf(stderr, "ERROR: build prompt failed\n"); return 1; }
    int total_len = (int)boat_tensor_shape(merged)[0];
    printf("  merged sequence: %d tokens (%.2fs)\n", total_len, get_time_sec() - t0);

    // ---- 5. Decoder prefix forward ----
    printf("Running decoder prefix ...\n");
    t0 = get_time_sec();
    qwen3asr_decoder_t *decoder = qwen3asr_decoder_create(w);
    if (!decoder) { fprintf(stderr, "ERROR: decoder create failed\n"); return 1; }

    boat_tensor_t *logits = qwen3asr_decoder_forward(decoder, merged);
    if (!logits) { fprintf(stderr, "ERROR: decoder forward failed\n"); return 1; }
    printf("  prefix done (%.2fs)\n", get_time_sec() - t0);

    // ---- 6. Autoregressive decode loop ----
    printf("Generating ...\n");
    t0 = get_time_sec();

    int output_ids[QWEN3ASR_MAX_NEW_TOKENS];
    int n_output = 0;
    int current_pos = total_len;

    const float *logits_data = (const float*)boat_tensor_data(logits);

    // DEBUG: print top-5 tokens from prefix logits
    printf("  first logits top-5: ");
    {
        int seen[5] = {-1,-1,-1,-1,-1};
        for (int _j = 0; _j < 5; _j++) {
            int best = -1;
            for (int _i = 0; _i < QWEN3ASR_VOCAB_SIZE; _i++) {
                int skip = 0;
                for (int _k = 0; _k < _j; _k++) if (seen[_k] == _i) { skip = 1; break; }
                if (skip) continue;
                if (best < 0 || logits_data[_i] > logits_data[best]) best = _i;
            }
            seen[_j] = best;
            printf("%d(%.1f) ", best, logits_data[best]);
        }
    }
    printf("\n");

    int next_token = boat_sample_token(logits_data, QWEN3ASR_VOCAB_SIZE, 10, 1.0f);
    printf("  sampled first token: %d\n", next_token);
    boat_tensor_unref(logits);

    while (next_token != QWEN3ASR_EOS_ID && next_token != 0
           && n_output < QWEN3ASR_MAX_NEW_TOKENS) {
        output_ids[n_output++] = next_token;

        float *embed = w->embed_tokens + (size_t)next_token * QWEN3ASR_DECODER_HIDDEN_SIZE;
        boat_tensor_t *token_embed = boat_tensor_from_data(
            (int64_t[]){1, QWEN3ASR_DECODER_HIDDEN_SIZE}, 2,
            BOAT_DTYPE_FLOAT32, embed);

        boat_tensor_t *step_logits = qwen3asr_decoder_step(decoder, token_embed, current_pos);
        boat_tensor_unref(token_embed);

        if (!step_logits) {
            fprintf(stderr, "WARN: decoder step failed at pos %d\n", current_pos);
            break;
        }

        current_pos++;
        const float *sl_data = (const float*)boat_tensor_data(step_logits);
        next_token = boat_sample_token(sl_data, QWEN3ASR_VOCAB_SIZE, 10, 1.0f);
        boat_tensor_unref(step_logits);
    }

    printf("  generated %d tokens (%.2fs)\n", n_output, get_time_sec() - t0);
    printf("  token IDs: ");
    for (int i = 0; i < n_output; i++) printf("%d ", output_ids[i]);
    printf("\n");

    // ---- 7. Decode tokens to text ----
    char *text = qwen3asr_decode_tokens(model_dir, output_ids, n_output);
    printf("\n=== Transcription ===\n");
    printf("%s\n", text ? text : "(null)");
    free(text);

    // ---- Cleanup ----
    qwen3asr_decoder_free(decoder);
    boat_tensor_unref(merged);
    qwen3asr_weights_free(w);

    printf("\nDone.\n");
    return 0;
}
