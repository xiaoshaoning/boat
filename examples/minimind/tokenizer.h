// tokenizer.h - BPE tokenizer for MiniMind
#pragma once

typedef struct minimind_tokenizer_t minimind_tokenizer_t;

// Load tokenizer from a directory containing tokenizer.json.
// Returns NULL on failure.
minimind_tokenizer_t* minimind_tokenizer_load(const char* model_dir);

// Free tokenizer.
void minimind_tokenizer_free(minimind_tokenizer_t* tok);

// Encode text to token IDs. Returns number of tokens, -1 on error.
// tokens_out must be pre-allocated (max_len elements).
int minimind_tokenizer_encode(minimind_tokenizer_t* tok, const char* text,
                               int* tokens_out, int max_len);

// Decode token IDs to string. Returns malloc'd string (caller frees).
char* minimind_tokenizer_decode(minimind_tokenizer_t* tok,
                                 const int* tokens, int n_tokens);

// Get vocab size.
int minimind_tokenizer_vocab_size(minimind_tokenizer_t* tok);
