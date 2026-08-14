// san.h - Simple Attention Network inference executor for the Needle .cact
// deployment weights. Implements the decode path of needle/model/decode.py:
// tied embedding, 27-layer MHC scan with gated GQA attention (RoPE + sliding
// window), Hadamard MLP, engram KV memory at the configured sites, final
// ZCRMSNorm and tied logits.

#ifndef NEEDLE_SAN_H
#define NEEDLE_SAN_H

#include <stdio.h>

#include "cact.h"
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct needle_model needle_model_t;

// Dequantize all needed cact tensors into a runnable model. Returns NULL on
// error.
needle_model_t* needle_model_load(const needle_cact_t* cact);
void needle_model_free(needle_model_t* m);

// Generate from `prompt`, streaming the incremental decode to `stream` (may
// be NULL). Returns the full generated text (malloc'd, caller frees) or NULL
// on error.
char* needle_model_generate(const needle_model_t* m, const needle_tokenizer_t* tok,
                            const char* prompt, int max_new_tokens, float temperature,
                            FILE* stream);

// Like needle_model_generate but returns the raw generated token ids
// (malloc'd, caller frees; *n_out = count).
int32_t* needle_model_generate_ids(const needle_model_t* m, const needle_tokenizer_t* tok,
                                   const char* prompt, int max_new_tokens, float temperature,
                                   size_t* n_out);

// Version string for diagnostics.
const char* needle_model_engine_version(void);

// Diagnostics: run the prompt through the model and fill `logits` (vocab
// floats) with the raw logits of the last prompt position. Returns the
// number of prompt tokens on success, or -1 on error.
int needle_model_prompt_logits(const needle_model_t* m, const needle_tokenizer_t* tok,
                               const char* prompt, float* logits, size_t logits_cap);

// Vocabulary size of the loaded model.
uint32_t needle_model_vocab(const needle_model_t* m);

#ifdef __cplusplus
}
#endif

#endif // NEEDLE_SAN_H
