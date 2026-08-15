// spm.h - Minimal SentencePiece-like tokenizer using vocab.json
#ifndef BOAT_EXAMPLE_SPM_H
#define BOAT_EXAMPLE_SPM_H

#include <stddef.h>

// Tokenizer state
typedef struct {
    int* ids;          // token ID for each vocab entry (size = n)
    char** tokens;     // token strings (size = n)
    int* lengths;      // byte length of each token string (size = n)
    int n;             // vocabulary size
    int max_token_len; // longest token string in bytes

    // Sorted indices for longest-first matching
    int* sorted; // indices sorted by token length descending

    // Special IDs
    int unk_id;
    int bos_id; // </s> or BOS
    int eos_id; // </s> or EOS
} spm_tokenizer_t;

// Load tokenizer from vocab.json file.
// Returns 1 on success, 0 on failure.
int spm_init(spm_tokenizer_t* tok, const char* vocab_path);

// Free tokenizer resources
void spm_free(spm_tokenizer_t* tok);

// Encode a UTF-8 string to token IDs.
// Returns a malloc'd array of token IDs, *out_len is set.
// Caller must free() the returned array.
int* spm_encode(const spm_tokenizer_t* tok, const char* text, size_t text_len, int* out_len);

// Decode token IDs to a UTF-8 string.
// Returns a malloc'd string. Caller must free().
char* spm_decode(const spm_tokenizer_t* tok, const int* ids, int n_ids);

#endif // BOAT_EXAMPLE_SPM_H
