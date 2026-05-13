// bpe.h - BPE tokenizer (inference: decode only)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_BPE_H
#define BOAT_BPE_H

#include <stddef.h>
#include <stdint.h>
#include <boat/export.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque tokenizer handle
// ---------------------------------------------------------------------------
typedef struct boat_bpe_tokenizer_t boat_bpe_tokenizer_t;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
// Load from tokenizer.json (HuggingFace format)
BOAT_API boat_bpe_tokenizer_t* boat_bpe_tokenizer_create(
    const char* tokenizer_json_path);

BOAT_API void boat_bpe_tokenizer_free(boat_bpe_tokenizer_t* tok);

// ---------------------------------------------------------------------------
// Decode: token IDs → text
// ---------------------------------------------------------------------------
// Returns a malloc'd string — caller must free()
BOAT_API char* boat_bpe_tokenizer_decode(
    const boat_bpe_tokenizer_t* tok,
    const int32_t* ids,
    size_t n_ids);

// ---------------------------------------------------------------------------
// Encode: text → token IDs
// ---------------------------------------------------------------------------
// Returns malloc'd array — caller must free()
BOAT_API int32_t* boat_bpe_tokenizer_encode(
    const boat_bpe_tokenizer_t* tok,
    const char* text,
    size_t* out_len);

// ---------------------------------------------------------------------------
// Special token IDs
// ---------------------------------------------------------------------------
BOAT_API int32_t boat_bpe_tokenizer_bos_id(const boat_bpe_tokenizer_t* tok);
BOAT_API int32_t boat_bpe_tokenizer_eos_id(const boat_bpe_tokenizer_t* tok);
BOAT_API int32_t boat_bpe_tokenizer_pad_id(const boat_bpe_tokenizer_t* tok);
BOAT_API int32_t boat_bpe_tokenizer_unk_id(const boat_bpe_tokenizer_t* tok);
BOAT_API size_t  boat_bpe_tokenizer_vocab_size(const boat_bpe_tokenizer_t* tok);

#ifdef __cplusplus
}
#endif

#endif // BOAT_BPE_H
