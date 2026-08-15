// tokenizer.h - Self-contained SentencePiece BPE tokenizer for the Needle
// .cact RAW tokenizer blob (see RefTokenizer in needle/model/export.py).

#ifndef NEEDLE_TOKENIZER_H
#define NEEDLE_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Piece types in the RAW tokenizer blob.
#define NEEDLE_TK_NORMAL 0
#define NEEDLE_TK_UNKNOWN 1
#define NEEDLE_TK_CONTROL 2
#define NEEDLE_TK_USER_DEFINED 3
#define NEEDLE_TK_BYTE 4

typedef struct {
    uint32_t n_pieces;
    uint32_t pad_id;
    uint32_t eos_id;
    uint32_t bos_id;
    uint32_t unk_id;
    uint8_t add_dummy_prefix;
    uint8_t byte_fallback;

    char** pieces;     // n_pieces UTF-8 strings (owned)
    float* scores;     // n_pieces merge scores
    uint8_t* types;    // n_pieces piece types
    uint32_t* order;   // n_pieces, piece indices sorted by piece string
    int* byte_ids;     // [256] -> piece id, or -1 (byte pieces "<0xXX>")
    uint32_t* markers; // user-defined piece ids, sorted by length desc
    uint32_t n_markers;
} needle_tokenizer_t;

// Parse a RAW tokenizer blob. Returns 0 on success.
int needle_tokenizer_init(needle_tokenizer_t* tok, const uint8_t* blob, size_t nbytes);
void needle_tokenizer_free(needle_tokenizer_t* tok);

// Encode text into token ids. Returns count, or -1 if out_ids is too small.
int needle_tokenizer_encode(const needle_tokenizer_t* tok, const char* text, int* out_ids,
                            size_t cap);

// Decode token ids into a newly allocated UTF-8 string (caller frees).
char* needle_tokenizer_decode(const needle_tokenizer_t* tok, const int* ids, size_t n);

#ifdef __cplusplus
}
#endif

#endif // NEEDLE_TOKENIZER_H
