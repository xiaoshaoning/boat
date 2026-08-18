// spm.c - Minimal SentencePiece-like tokenizer using vocab.json
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include "spm.h"
#include "json.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int spm_init(spm_tokenizer_t* tok, const char* vocab_path) {
    memset(tok, 0, sizeof(*tok));

    // Read vocabulary file
    FILE* f = fopen(vocab_path, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open vocab file: %s\n", vocab_path);
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* json_data = (char*)malloc((size_t)fsize + 1);
    if (!json_data) {
        fclose(f);
        return 0;
    }
    if (fread(json_data, 1, (size_t)fsize, f) != (size_t)fsize) {
        free(json_data);
        fclose(f);
        return 0;
    }
    json_data[fsize] = '\0';
    fclose(f);

    // Parse JSON: {"tok1": id1, "tok2": id2, ...}
    json_ctx_t jctx;
    json_init(&jctx, json_data, (size_t)fsize);

    if (json_next(&jctx) != '{') {
        free(json_data);
        return 0;
    }

    // First pass: count entries
    int count = 0;
    {
        json_ctx_t c2 = jctx;
        while (1) {
            json_skip_ws(&c2);
            if (c2.pos >= c2.len || json_data[c2.pos] == '}') break;
            char* k = json_parse_string(&c2);
            if (!k) break;
            free(k);
            json_skip_ws(&c2);
            json_expect(&c2, ':');
            json_parse_int(&c2);
            json_skip_ws(&c2);
            if (c2.pos < c2.len && json_data[c2.pos] == ',') c2.pos++;
            count++;
        }
    }

    if (count == 0) {
        free(json_data);
        return 0;
    }

    // Allocate
    tok->n = count;
    tok->tokens = (char**)calloc(count, sizeof(char*));
    tok->ids = (int*)malloc(count * sizeof(int));
    tok->lengths = (int*)malloc(count * sizeof(int));
    tok->sorted = (int*)malloc(count * sizeof(int));
    if (!tok->tokens || !tok->ids || !tok->lengths || !tok->sorted) {
        free(json_data);
        spm_free(tok);
        return 0;
    }

    // Second pass: parse all entries
    int idx = 0;
    tok->unk_id = 1; // default
    tok->bos_id = 0;
    tok->eos_id = 0;
    tok->max_token_len = 0;

    while (1) {
        json_skip_ws(&jctx);
        if (jctx.pos >= jctx.len || json_data[jctx.pos] == '}') break;

        char* token_str = json_parse_string(&jctx);
        if (!token_str) break;
        json_skip_ws(&jctx);
        json_expect(&jctx, ':');
        int id = (int)json_parse_int(&jctx);

        // Trim <unk> byte order mark if present
        // SentencePiece sometimes adds \x00 or BOM to tokens
        char* clean = token_str;
        // Skip leading \x01 (SOH) byte prefix markers
        while ((unsigned char)*clean == 0x01 || *clean == '\xef' || (unsigned char)*clean == 0xbb ||
               *clean == '\xbf') {
            clean++;
        }

        tok->ids[idx] = id;
        tok->tokens[idx] = strdup(clean);
        tok->lengths[idx] = (int)strlen(clean);
        if (tok->lengths[idx] > tok->max_token_len) tok->max_token_len = tok->lengths[idx];
        tok->sorted[idx] = idx;

        if (strcmp(clean, "<unk>") == 0) tok->unk_id = id;
        if (strcmp(clean, "</s>") == 0) {
            tok->bos_id = id;
            tok->eos_id = id;
        }

        free(token_str);
        idx++;

        json_skip_ws(&jctx);
        if (jctx.pos < jctx.len && json_data[jctx.pos] == ',') jctx.pos++;
    }

    free(json_data);

    if (idx != count) {
        tok->n = idx;
    }

    // Sort indices by token length descending (longest first for greedy matching)
    // Simple insertion sort (n ~ 60K, one-time cost)
    for (int i = 0; i < tok->n; i++) {
        for (int j = i + 1; j < tok->n; j++) {
            if (tok->lengths[tok->sorted[i]] < tok->lengths[tok->sorted[j]]) {
                int tmp = tok->sorted[i];
                tok->sorted[i] = tok->sorted[j];
                tok->sorted[j] = tmp;
            }
        }
    }

    return 1;
}

void spm_free(spm_tokenizer_t* tok) {
    if (!tok) return;
    for (int i = 0; i < tok->n; i++)
        free(tok->tokens[i]);
    free(tok->tokens);
    free(tok->ids);
    free(tok->lengths);
    free(tok->sorted);
    tok->tokens = NULL;
    tok->ids = NULL;
    tok->lengths = NULL;
    tok->sorted = NULL;
    tok->n = 0;
}

// Pre-encoded U+2581 (SentencePiece space marker) in UTF-8: E2 96 81
#define SPACE_MARKER "\xE2\x96\x81"
#define SPACE_MARKER_LEN 3

int* spm_encode(const spm_tokenizer_t* tok, const char* text, size_t text_len, int* out_len) {
    if (!tok || !text) {
        *out_len = 0;
        return NULL;
    }

    // Prepare input buffer: add space prefix, replace ' ' with U+2581
    // Upper bound: text_len + prefix + overhead for space replacement
    size_t buf_cap = text_len + 16 + SPACE_MARKER_LEN * 2;
    char* buf = (char*)malloc(buf_cap);
    if (!buf) {
        *out_len = 0;
        return NULL;
    }
    size_t blen = 0;

    // Add space prefix (word start marker in SentencePiece)
    memcpy(buf + blen, SPACE_MARKER, SPACE_MARKER_LEN);
    blen += SPACE_MARKER_LEN;

    // Process input text
    for (size_t i = 0; i < text_len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == ' ') {
            // Ensure capacity
            if (blen + SPACE_MARKER_LEN >= buf_cap) {
                buf_cap *= 2;
                buf = (char*)realloc(buf, buf_cap);
                if (!buf) {
                    *out_len = 0;
                    return NULL;
                }
            }
            memcpy(buf + blen, SPACE_MARKER, SPACE_MARKER_LEN);
            blen += SPACE_MARKER_LEN;
        } else {
            if (blen + 1 >= buf_cap) {
                buf_cap *= 2;
                buf = (char*)realloc(buf, buf_cap);
                if (!buf) {
                    *out_len = 0;
                    return NULL;
                }
            }
            buf[blen++] = (char)c;
        }
    }

    // Greedy encoding
    int* ids = (int*)malloc(tok->n * 2 * sizeof(int)); // worst case: each byte is a token
    if (!ids) {
        free(buf);
        *out_len = 0;
        return NULL;
    }
    int n_ids = 0;

    size_t pos = 0;
    while (pos < blen) {
        int found = 0;
        size_t remaining = blen - pos;

        // Try tokens from longest to shortest
        for (int si = 0; si < tok->n; si++) {
            int ti = tok->sorted[si];
            int tlen = tok->lengths[ti];
            if (tlen == 0 || (size_t)tlen > remaining) continue;

            if (memcmp(buf + pos, tok->tokens[ti], (size_t)tlen) == 0) {
                ids[n_ids++] = tok->ids[ti];
                pos += (size_t)tlen;
                found = 1;
                break;
            }
        }

        if (!found) {
            // Use <unk> for this byte
            ids[n_ids++] = tok->unk_id;
            pos++;
        }
    }

    free(buf);
    *out_len = n_ids;
    return ids;
}

char* spm_decode(const spm_tokenizer_t* tok, const int* ids, int n_ids) {
    if (!tok || !ids || n_ids == 0) return NULL;

    // Upper bound: sum of all token lengths
    size_t cap = 1024;
    size_t len = 0;
    char* result = (char*)malloc(cap);
    if (!result) return NULL;

    for (int i = 0; i < n_ids; i++) {
        // Find token by ID
        char* token_str = NULL;
        for (int j = 0; j < tok->n; j++) {
            if (tok->ids[j] == ids[i]) {
                token_str = tok->tokens[j];
                break;
            }
        }
        if (!token_str) continue;

        // Skip special tokens
        if (ids[i] == tok->bos_id || ids[i] == tok->eos_id || ids[i] == tok->unk_id) continue;

        size_t tlen = strlen(token_str);

        // Handle space marker: replace with space
        if (tlen >= SPACE_MARKER_LEN && memcmp(token_str, SPACE_MARKER, SPACE_MARKER_LEN) == 0) {
            // Add space
            if (len + 1 >= cap) {
                cap *= 2;
                result = (char*)realloc(result, cap);
                if (!result) return NULL;
            }
            result[len++] = ' ';

            // Append rest of token after the space marker
            size_t rest_len = tlen - SPACE_MARKER_LEN;
            if (rest_len > 0) {
                if (len + rest_len + 1 >= cap) {
                    cap *= 2;
                    result = (char*)realloc(result, cap);
                    if (!result) return NULL;
                }
                memcpy(result + len, token_str + SPACE_MARKER_LEN, rest_len);
                len += rest_len;
            }
        } else {
            if (len + tlen + 1 >= cap) {
                cap *= 2;
                result = (char*)realloc(result, cap);
                if (!result) return NULL;
            }
            memcpy(result + len, token_str, tlen);
            len += tlen;
        }
    }

    result[len] = '\0';

    // Trim leading space
    while (len > 0 && result[0] == ' ') {
        memmove(result, result + 1, len);
        len--;
    }

    return result;
}
