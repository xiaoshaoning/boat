// tokenizer.h - NanoChat BPE tokenizer (GPT-2 byte-level BPE)
#pragma once
#include <stddef.h>

#define NANOCHAT_VOCAB_SIZE 65536

typedef struct {
    char** tokens;          // token strings by ID
    int* lengths;           // byte length of each token
    int n;                  // vocabulary size (loaded)

    // BPE merge data
    char** merge_pairs;     // merge pair strings "A B"
    int* merge_priorities;  // priority for each pair
    int num_merges;

    // Byte-to-Unicode mapping (GPT-2 style)
    char byte_to_unicode[256][5];

    // Special token IDs
    int bos_id;             // 0
    int eos_id;             // 1
    int pad_id;             // 1

    // Added tokens (from added_tokens in tokenizer.json)
    int added_tokens_start; // 65527

    // Pre-allocated workspace for encoding
    int* work_ids;
    char** work_tokens;
    int work_cap;
} nanochat_tokenizer_t;

int nanochat_tokenizer_init(nanochat_tokenizer_t* tok, const char* vocab_path);
void nanochat_tokenizer_free(nanochat_tokenizer_t* tok);

int* nanochat_tokenizer_encode(const nanochat_tokenizer_t* tok,
                                const char* text, size_t text_len, int* out_len);
char* nanochat_tokenizer_decode(const nanochat_tokenizer_t* tok,
                                 const int* ids, int n_ids);
int nanochat_tokenizer_eos_id(const nanochat_tokenizer_t* tok);
