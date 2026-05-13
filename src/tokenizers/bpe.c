// bpe.c - BPE tokenizer implementation (decode-only for inference)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/tokenizers/bpe.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>

// =========================================================================
// Tokenizer struct
// =========================================================================

struct boat_bpe_tokenizer_t {
    // Vocab: id -> token string
    char** vocab;           // [vocab_size]
    size_t vocab_size;

    // Special token IDs
    int32_t bos_id;
    int32_t eos_id;
    int32_t pad_id;
    int32_t unk_id;

    // BPE merge ranks for encoding (optional, used by encode)
    // For decode-only, we don't need merges — just the vocab mapping.

    // Added tokens decoder (for special tokens like [START_REF], etc.)
    int32_t num_added_tokens;
    int32_t* added_ids;     // IDs of added tokens
    char** added_tokens;    // strings of added tokens
};

// =========================================================================
// Simple JSON parser for tokenizer.json
// =========================================================================

// Recursive descent JSON parser for extracting tokenizer vocab.
// We only parse the structure we need, not arbitrary JSON.

typedef struct {
    const char* data;
    size_t len;
    size_t pos;
} json_reader_t;

static void json_skip_ws(json_reader_t* jr) {
    while (jr->pos < jr->len) {
        char c = jr->data[jr->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            jr->pos++;
        } else {
            break;
        }
    }
}

static int json_peek(json_reader_t* jr) {
    json_skip_ws(jr);
    return jr->pos < jr->len ? jr->data[jr->pos] : 0;
}

static int json_next(json_reader_t* jr) {
    json_skip_ws(jr);
    if (jr->pos >= jr->len) return 0;
    return jr->data[jr->pos++];
}

static int json_expect(json_reader_t* jr, char expected) {
    int c = json_next(jr);
    return c == expected;
}

static char* json_parse_string(json_reader_t* jr) {
    if (json_next(jr) != '"') return NULL;
    size_t start = jr->pos;
    size_t end = start;
    while (end < jr->len) {
        if (jr->data[end] == '\\') {
            end += 2; // skip escaped char
        } else if (jr->data[end] == '"') {
            break;
        } else {
            end++;
        }
    }
    if (end >= jr->len) return NULL;
    size_t slen = end - start;
    char* str = (char*)malloc(slen + 1);
    if (!str) return NULL;
    size_t wi = 0;
    for (size_t i = start; i < end; i++) {
        if (jr->data[i] == '\\' && i + 1 < end) {
            i++;
            switch (jr->data[i]) {
                case 'n': str[wi++] = '\n'; break;
                case 't': str[wi++] = '\t'; break;
                case 'r': str[wi++] = '\r'; break;
                default:  str[wi++] = jr->data[i]; break;
            }
        } else {
            str[wi++] = jr->data[i];
        }
    }
    str[wi] = '\0';
    jr->pos = end + 1;
    return str;
}

static int json_parse_int(json_reader_t* jr) {
    json_skip_ws(jr);
    int neg = 0;
    if (jr->pos < jr->len && jr->data[jr->pos] == '-') { neg = 1; jr->pos++; }
    int val = 0;
    while (jr->pos < jr->len && jr->data[jr->pos] >= '0' && jr->data[jr->pos] <= '9') {
        val = val * 10 + (jr->data[jr->pos] - '0');
        jr->pos++;
    }
    return neg ? -val : val;
}

static void json_skip_value(json_reader_t* jr) {
    int c = json_peek(jr);
    if (c == '"') {
        char* s = json_parse_string(jr);
        free(s);
    } else if (c == '{') {
        json_next(jr); // skip '{'
        int depth = 1;
        while (depth > 0 && jr->pos < jr->len) {
            c = json_next(jr);
            if (c == '{') depth++;
            else if (c == '}') depth--;
        }
    } else if (c == '[') {
        json_next(jr); // skip '['
        int depth = 1;
        while (depth > 0 && jr->pos < jr->len) {
            c = json_next(jr);
            if (c == '[') depth++;
            else if (c == ']') depth--;
        }
    } else if (c == 't' || c == 'f') {
        // true/false
        while (jr->pos < jr->len && jr->data[jr->pos] >= 'a' && jr->data[jr->pos] <= 'z') jr->pos++;
    } else if (c == 'n') {
        // null
        while (jr->pos < jr->len && jr->data[jr->pos] >= 'a' && jr->data[jr->pos] <= 'z') jr->pos++;
    } else {
        // number
        while (jr->pos < jr->len) {
            c = jr->data[jr->pos];
            if ((c >= '0' && c <= '9') || c == '-' || c == '.' || c == 'e' || c == 'E') {
                jr->pos++;
            } else {
                break;
            }
        }
    }
}

// Skip to a specific key in a JSON object
static int json_skip_to_key(json_reader_t* jr, const char* key) {
    if (json_next(jr) != '{') return 0;
    while (1) {
        int c = json_peek(jr);
        if (c == '}') { json_next(jr); return 0; }
        if (c != '"') return 0;
        char* k = json_parse_string(jr);
        if (!k) return 0;
        if (!json_expect(jr, ':')) { free(k); return 0; }
        if (strcmp(k, key) == 0) { free(k); return 1; }
        json_skip_value(jr);
        free(k);
        c = json_peek(jr);
        if (c == ',') json_next(jr);
    }
}

// =========================================================================
// Implementation
// =========================================================================

boat_bpe_tokenizer_t* boat_bpe_tokenizer_create(const char* tokenizer_json_path) {
    if (!tokenizer_json_path) return NULL;

    // Read file
    FILE* f = fopen(tokenizer_json_path, "rb");
    if (!f) { fprintf(stderr, "[BPE] Cannot open: %s\n", tokenizer_json_path); return NULL; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    char* json_str = (char*)malloc(fsize + 1);
    if (!json_str) { fclose(f); return NULL; }
    size_t nread = fread(json_str, 1, fsize, f);
    fclose(f);
    json_str[nread] = '\0';

    json_reader_t jr;
    jr.data = json_str;
    jr.len = nread;
    jr.pos = 0;

    boat_bpe_tokenizer_t* tok = (boat_bpe_tokenizer_t*)boat_malloc(sizeof(boat_bpe_tokenizer_t), BOAT_DEVICE_CPU);
    if (!tok) { free(json_str); return NULL; }
    memset(tok, 0, sizeof(*tok));
    // Default special token IDs (may be overridden by added_tokens_decoder)
    tok->bos_id = 0;
    tok->eos_id = 2;
    tok->pad_id = 1;
    tok->unk_id = 3;

    // Fast-forward past the top-level JSON object; we extract vocab via strstr below.
    {
        int depth = 0, in_str = 0;
        for (size_t i = 0; i < jr.len; i++) {
            if (in_str) {
                if (json_str[i] == '\\') { i++; continue; }
                if (json_str[i] == '"') in_str = 0;
            } else {
                if (json_str[i] == '"') in_str = 1;
                else if (json_str[i] == '{') depth++;
                else if (json_str[i] == '}') { depth--; if (depth == 0) { jr.pos = i + 1; break; } }
            }
        }
    }

    free(json_str);

    // Load vocab from tokenizer.json model.vocab
    // Use strstr-based scan for the "vocab" object inside "model".
    f = fopen(tokenizer_json_path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        fsize = ftell(f);
        rewind(f);
        json_str = (char*)malloc(fsize + 1);
        if (json_str) {
            nread = fread(json_str, 1, fsize, f);
            json_str[nread] = '\0';
            jr.data = json_str;
            jr.len = nread;
            jr.pos = 0;

            // Find "model" -> "vocab" object
            // Search for "vocab" string
            const char* vocab_pos = strstr(json_str, "\"vocab\"");
            if (vocab_pos) {
                // Find the opening { after "vocab":
                const char* brace = vocab_pos;
                while (*brace && *brace != '{') brace++;
                if (*brace == '{') {
                    // Count vocab entries
                    size_t vcount = 0;
                    const char* p = brace + 1;
                    int depth = 1;
                    while (depth > 0 && *p) {
                        if (*p == '{') depth++;
                        else if (*p == '}') depth--;
                        else if (*p == ',' && depth == 1) vcount++;
                        p++;
                    }
                    vcount++; // last item before }
                    if (vcount > 0 && vcount < 200000) {
                        if (tok->vocab) {
                            for (size_t i = 0; i < tok->vocab_size; i++) {
                                if (tok->vocab[i]) free(tok->vocab[i]);
                            }
                            boat_free(tok->vocab);
                        }
                        tok->vocab_size = vcount;
                        tok->vocab = (char**)boat_malloc(sizeof(char*) * tok->vocab_size, BOAT_DEVICE_CPU);
                        if (tok->vocab) {
                            memset(tok->vocab, 0, sizeof(char*) * tok->vocab_size);

                            p = brace + 1;
                            // Parse each "token_str": id
                            while (*p) {
                                while (*p && *p != '"') p++;
                                if (!*p) break;
                                p++; // skip opening quote
                                const char* tstart = p;
                                while (*p && *p != '"') {
                                    if (*p == '\\') p++;
                                    if (*p) p++;
                                }
                                if (!*p) break;
                                // token string is from tstart to p
                                size_t tlen = p - tstart;
                                p++; // skip closing quote
                                while (*p && *p != ':') p++;
                                if (!*p) break;
                                p++; // skip :
                                while (*p && (*p == ' ' || *p == '\t')) p++;
                                int id = 0;
                                int neg = 0;
                                if (*p == '-') { neg = 1; p++; }
                                while (*p && *p >= '0' && *p <= '9') {
                                    id = id * 10 + (*p - '0');
                                    p++;
                                }
                                if (neg) id = -id;
                                while (*p && *p != ',' && *p != '}') p++;

                                if (id >= 0 && id < (int)tok->vocab_size && !tok->vocab[id]) {
                                    char* token = (char*)malloc(tlen + 1);
                                    if (token) {
                                        memcpy(token, tstart, tlen);
                                        token[tlen] = '\0';
                                        tok->vocab[id] = token;
                                    }
                                }

                                if (*p == '}') break;
                                p++; // skip comma
                            }
                        }
                    }
                }
            }
            free(json_str);
        }
        fclose(f);
    }

    // Fill any missing vocab entries
    for (size_t i = 0; i < tok->vocab_size; i++) {
        if (!tok->vocab[i]) {
            char buf[32];
            snprintf(buf, sizeof(buf), "<|%zu|>", i);
            tok->vocab[i] = strdup(buf);
        }
    }

    fprintf(stderr, "[Nougat] Tokenizer loaded: vocab=%zu, bos=%d, eos=%d, pad=%d, unk=%d\n",
            tok->vocab_size, tok->bos_id, tok->eos_id, tok->pad_id, tok->unk_id);
    return tok;
}

void boat_bpe_tokenizer_free(boat_bpe_tokenizer_t* tok) {
    if (!tok) return;
    if (tok->vocab) {
        for (size_t i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i]) free(tok->vocab[i]);
        }
        boat_free(tok->vocab);
    }
    if (tok->added_ids) boat_free(tok->added_ids);
    if (tok->added_tokens) {
        for (int32_t i = 0; i < tok->num_added_tokens; i++) {
            if (tok->added_tokens[i]) free(tok->added_tokens[i]);
        }
        boat_free(tok->added_tokens);
    }
    boat_free(tok);
}

char* boat_bpe_tokenizer_decode(const boat_bpe_tokenizer_t* tok,
                                  const int32_t* ids, size_t n_ids)
{
    if (!tok || !ids || n_ids == 0) return NULL;

    // First pass: compute total length
    size_t total_len = 0;
    for (size_t i = 0; i < n_ids; i++) {
        int32_t id = ids[i];
        if (id < 0 || (size_t)id >= tok->vocab_size) continue;
        // Skip special tokens
        if (id == tok->bos_id || id == tok->eos_id || id == tok->pad_id) continue;
        if (tok->vocab[id]) {
            total_len += strlen(tok->vocab[id]);
        }
    }

    // Also handle added tokens
    if (tok->num_added_tokens > 0 && tok->added_ids && tok->added_tokens) {
        for (size_t i = 0; i < n_ids; i++) {
            for (int32_t j = 0; j < tok->num_added_tokens; j++) {
                if (ids[i] == tok->added_ids[j] && tok->added_tokens[j]) {
                    total_len += strlen(tok->added_tokens[j]) + 2; // space padding
                }
            }
        }
    }

    // Allocate output
    size_t extra = 128;
    char* result = (char*)malloc(total_len + extra + 1);
    if (!result) return NULL;
    result[0] = '\0';
    size_t pos = 0;

    // Decode each token
    for (size_t i = 0; i < n_ids; i++) {
        int32_t id = ids[i];
        if (id < 0 || (size_t)id >= tok->vocab_size) continue;
        if (id == tok->bos_id || id == tok->eos_id || id == tok->pad_id) continue;

        const char* token = tok->vocab[id];
        if (!token) continue;

        // Handle BPE continuation markers (Ġ = space)
        // Nougat uses SentencePiece-style tokenization
        // Space is encoded as "Ġ" (U+0120, 0xC4 0xA0 in UTF-8)
        // Map it to regular space
        size_t tlen = strlen(token);
        if (tlen > 0 && (unsigned char)token[0] == 0xC4 && (unsigned char)token[1] == 0xA0) {
            // Ġ prefix = space before token
            if (pos > 0) result[pos++] = ' ';
            for (size_t j = 2; j < tlen; j++) {
                result[pos++] = token[j];
            }
        } else {
            for (size_t j = 0; j < tlen; j++) {
                result[pos++] = token[j];
            }
        }
    }

    // Handle added tokens
    if (tok->num_added_tokens > 0 && tok->added_ids && tok->added_tokens) {
        for (size_t i = 0; i < n_ids; i++) {
            for (int32_t j = 0; j < tok->num_added_tokens; j++) {
                if (ids[i] == tok->added_ids[j] && tok->added_tokens[j]) {
                    size_t tlen = strlen(tok->added_tokens[j]);
                    if (pos > 0) result[pos++] = ' ';
                    memcpy(result + pos, tok->added_tokens[j], tlen);
                    pos += tlen;
                }
            }
        }
    }

    result[pos] = '\0';
    return result;
}

int32_t* boat_bpe_tokenizer_encode(const boat_bpe_tokenizer_t* tok,
                                    const char* text, size_t* out_len)
{
    // For now: encode each character individually (placeholder)
    // Full BPE merge implementation would require the merge rules from tokenizer.json
    if (!tok || !text || !out_len) return NULL;

    size_t len = strlen(text);
    if (len == 0) {
        *out_len = 0;
        return NULL;
    }

    // Simple: look up each character/byte as a token
    // This won't produce correct BPE merges but allows basic functionality
    // For proper BPE, we need the merge rules from tokenizer.json
    int32_t* ids = (int32_t*)malloc(sizeof(int32_t) * len);
    if (!ids) { *out_len = 0; return NULL; }

    size_t count = 0;
    for (size_t i = 0; i < len; ) {
        // Try longest match first (simple greedy)
        // For utf-8 chars
        unsigned char c = (unsigned char)text[i];
        int char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        // Try the unk token as fallback
        ids[count++] = tok->unk_id;
        i += char_len;
    }

    *out_len = count;
    return ids;
}

int32_t boat_bpe_tokenizer_bos_id(const boat_bpe_tokenizer_t* tok) { return tok ? tok->bos_id : 0; }
int32_t boat_bpe_tokenizer_eos_id(const boat_bpe_tokenizer_t* tok) { return tok ? tok->eos_id : 2; }
int32_t boat_bpe_tokenizer_pad_id(const boat_bpe_tokenizer_t* tok) { return tok ? tok->pad_id : 1; }
int32_t boat_bpe_tokenizer_unk_id(const boat_bpe_tokenizer_t* tok) { return tok ? tok->unk_id : 3; }
size_t  boat_bpe_tokenizer_vocab_size(const boat_bpe_tokenizer_t* tok) { return tok ? tok->vocab_size : 0; }
