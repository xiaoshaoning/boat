// nougat_decoder.h - Nougat-LaTeX decoder stack and autoregressive generation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef NOUGAT_DECODER_H
#define NOUGAT_DECODER_H

#include "nougat_model.h"
#include <boat/tokenizers/bpe.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Run the full decoder stack and generate LaTeX tokens.
// model: loaded model weights
// encoder_output: [1, S, 1024] from Swin encoder
// tokenizer: BPE tokenizer (used for special token IDs)
// max_steps: maximum generation steps
// device: device to use for computation
// out_ids: output buffer for generated token IDs (caller must free)
// out_len: number of generated tokens
//
// Returns 0 on success, -1 on failure.
int nougat_decoder_generate(
    const nougat_model_t* model,
    const boat_tensor_t* encoder_output,
    const boat_bpe_tokenizer_t* tokenizer,
    int max_steps,
    boat_device_t device,
    int32_t** out_ids,
    int* out_len);

#ifdef __cplusplus
}
#endif

#endif // NOUGAT_DECODER_H
