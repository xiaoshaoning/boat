// tokenizer.h - GLM-OCR BPE tokenizer using tokenizer.json (GPT-2 byte-level BPE)
#ifndef BOAT_OCR_TOKENIZER_H
#define BOAT_OCR_TOKENIZER_H

#include <stddef.h>

#define OCR_VOCAB_SIZE 59392

typedef struct {
    char** tokens;          // token strings by ID (indexed by token id)
    int* lengths;           // byte length of each token
    int n;                  // vocabulary size

    // BPE merge data
    char** merge_pairs;     // merge pair strings "A B"
    int* merge_priorities;  // priority for each pair (lower = merged earlier)
    int num_merges;

    // Byte-to-Unicode mapping (GPT-2 style)
    char byte_to_unicode[256][5];  // byte value -> UTF-8 encoded unicode char

    // Special token IDs
    int unk_id;             // 59246 = <|endoftext|>
    int eos_id;             // 59246 = <|endoftext|>
    int sop_id;             // 59250 = <sop>
    int eop_id;             // 59251 = <eop>
    int gmask_id;           // 59248 = [gMASK]
    int user_role_id;       // 59253 = <|user|>
    int assistant_role_id;  // 59254 = <|assistant|>
    int think_id;           // 59267 = <think>
    int endthink_id;        // 59268 = </think>
    int image_token_id;     // 59280 = <|image|>
    int img_start_id;       // 59256 = <|begin_of_image|>
    int img_end_id;         // 59257 = <|end_of_image|>
    int newline_id;         // 10 = \n

    // Pre-allocated workspace for encoding
    int* work_ids;
    char** work_tokens;
    int work_cap;
} ocr_tokenizer_t;

int ocr_tokenizer_init(ocr_tokenizer_t* tok, const char* vocab_path);
void ocr_tokenizer_free(ocr_tokenizer_t* tok);

// Encode text to token IDs (caller must free returned array with free())
int* ocr_tokenizer_encode(const ocr_tokenizer_t* tok, const char* text, size_t text_len, int* out_len);

// Decode token IDs to text (caller must free returned string with free())
char* ocr_tokenizer_decode(const ocr_tokenizer_t* tok, const int* ids, int n_ids);

#endif // BOAT_OCR_TOKENIZER_H
