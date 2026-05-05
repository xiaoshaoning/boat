// tokenizer.c — Qwen2 BPE token decoder implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0
//
// Implements GPT-2 style bytes_to_unicode BPE decoding using vocab.json.
// The vocab.json maps "unicode_string" → token_id. Each unicode character
// in the key represents one byte via the bytes_to_unicode mapping.

#include "tokenizer.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Bytes-to-unicode reverse mapping (unicode code point → byte)
// ---------------------------------------------------------------------------
static unsigned char _unicode_to_byte[65536];
static int _byte_decoder_inited = 0;

static void _build_byte_decoder(void) {
    if (_byte_decoder_inited) return;
    _byte_decoder_inited = 1;
    memset(_unicode_to_byte, 0, sizeof(_unicode_to_byte));

    int bs[256], cs[256];
    int n = 0;

    // Printable bytes that map to themselves in unicode
    for (int b = 33; b <= 126; b++) { bs[n] = b; cs[n] = b; n++; }
    for (int b = 161; b <= 172; b++) { bs[n] = b; cs[n] = b; n++; }
    for (int b = 174; b <= 255; b++) { bs[n] = b; cs[n] = b; n++; }

    // Non-printable bytes get mapped to unicode 256+
    int next = 256;
    for (int b = 0; b < 256; b++) {
        int found = 0;
        for (int i = 0; i < n; i++) {
            if (bs[i] == b) { found = 1; break; }
        }
        if (!found) {
            bs[n] = b;
            cs[n] = next++;
            n++;
        }
    }

    // Build reverse mapping: unicode code point → byte
    for (int i = 0; i < 256; i++) {
        _unicode_to_byte[cs[i]] = (unsigned char)bs[i];
    }
}

// ---------------------------------------------------------------------------
// UTF-8 decoder: decode one code point from s[*pos], advance *pos
// ---------------------------------------------------------------------------
static int utf8_decode(const unsigned char *s, size_t *pos, size_t len) {
    if (*pos >= len) return -1;
    unsigned char c = s[*pos];
    if (c < 0x80) {
        (*pos)++;
        return c;
    } else if (c < 0xE0) {
        if (*pos + 2 > len) return -1;
        int cp = (c & 0x1F) << 6;
        cp |= (s[*pos + 1] & 0x3F);
        *pos += 2;
        return cp;
    } else if (c < 0xF0) {
        if (*pos + 3 > len) return -1;
        int cp = (c & 0x0F) << 12;
        cp |= (s[*pos + 1] & 0x3F) << 6;
        cp |= (s[*pos + 2] & 0x3F);
        *pos += 3;
        return cp;
    } else {
        if (*pos + 4 > len) return -1;
        int cp = (c & 0x07) << 18;
        cp |= (s[*pos + 1] & 0x3F) << 12;
        cp |= (s[*pos + 2] & 0x3F) << 6;
        cp |= (s[*pos + 3] & 0x3F);
        *pos += 4;
        return cp;
    }
}

// ---------------------------------------------------------------------------
// Decode a single token string (vocab JSON key value, already raw bytes
// after JSON escape decoding) into actual bytes via reverse bytes_to_unicode.
// Returns malloc'd buffer with decoded bytes, *out_len set.
// ---------------------------------------------------------------------------
static unsigned char* decode_token_string(const unsigned char *token_str,
                                           size_t token_len, size_t *out_len) {
    if (!token_str || token_len == 0) {
        *out_len = 0;
        return NULL;
    }

    // Upper bound: each unicode code point → 1 byte, and worst-case each UTF-8
    // char is 1 byte → output_len <= input_len. But to be safe, allocate input_len.
    unsigned char *result = (unsigned char*)malloc(token_len + 1);
    if (!result) { *out_len = 0; return NULL; }

    size_t write_pos = 0;
    size_t read_pos = 0;

    while (read_pos < token_len) {
        size_t old_pos = read_pos;
        int cp = utf8_decode(token_str, &read_pos, token_len);
        if (cp < 0 || cp >= 65536) {
            // Invalid UTF-8, copy byte as-is (shouldn't happen)
            if (write_pos < token_len) {
                result[write_pos++] = token_str[old_pos];
            }
            continue;
        }
        unsigned char b = _unicode_to_byte[cp];
        result[write_pos++] = b;
    }

    result[write_pos] = '\0';
    *out_len = write_pos;
    return result;
}

// ---------------------------------------------------------------------------
// Load vocab.json and build id → token_string array
// Returns array of malloc'd strings indexed by token ID, or NULL on failure.
// *out_size is set to the vocab size.
// ---------------------------------------------------------------------------
static char** load_vocab(const char *vocab_path, int *out_size) {
    *out_size = 0;

    // Read entire file
    FILE *f = fopen(vocab_path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", vocab_path); return NULL; }
    fseek(f, 0, SEEK_END);
    long file_len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_len <= 0) { fclose(f); return NULL; }

    char *data = (char*)malloc((size_t)file_len + 1);
    if (!data) { fclose(f); return NULL; }
    size_t nread = fread(data, 1, (size_t)file_len, f);
    fclose(f);
    if ((long)nread != file_len) { free(data); return NULL; }
    data[file_len] = '\0';

    // First pass: count entries (count '{' at start then parse)
    // Format: {"key":value,"key2":value2,...}
    // We'll parse manually with a state machine

    // Allocate vocab array (max possible = up to 152K for Qwen2)
    int max_entries = QWEN3ASR_VOCAB_SIZE;
    char **id_to_str = (char**)calloc((size_t)max_entries, sizeof(char*));
    if (!id_to_str) { free(data); return NULL; }
    int count = 0;
    int max_id = 0;

    // Parse the JSON object
    size_t pos = 0;
    // Skip whitespace and opening '{'
    while (pos < (size_t)file_len && data[pos] != '{') pos++;
    if (data[pos] == '{') pos++;

    while (pos < (size_t)file_len) {
        // Skip whitespace and commas
        while (pos < (size_t)file_len && (data[pos] == ' ' || data[pos] == '\t' ||
               data[pos] == '\n' || data[pos] == '\r' || data[pos] == ',')) pos++;

        // Check for closing brace
        if (pos >= (size_t)file_len || data[pos] == '}') break;

        // Parse key string
        if (data[pos] != '"') break;
        pos++;  // skip opening quote

        // Extract key with JSON escape handling
        // We'll build the decoded key incrementally
        size_t key_start = pos;
        // First, find the end of the raw key (before escape processing)
        size_t key_end = pos;
        int has_escapes = 0;
        while (key_end < (size_t)file_len) {
            if (data[key_end] == '\\') {
                has_escapes = 1;
                key_end += 2;  // skip \ and next char
            } else if (data[key_end] == '"') {
                break;
            } else {
                key_end++;
            }
        }

        // Extract raw key bytes (without quotes, with escapes still encoded as in file)
        // Actually, we need to handle the escapes PROPERLY here.
        // For most keys there are no escapes. For keys with escapes (\n, \t, \\),
        // we need to interpret them.
        // Build the decoded key:
        size_t raw_len = key_end - key_start;
        char *key_raw = (char*)malloc(raw_len + 1);
        if (!key_raw) break;
        size_t kr = 0;
        size_t kp = key_start;
        while (kp < key_end && kp < (size_t)file_len) {
            if (data[kp] == '\\' && kp + 1 < (size_t)file_len) {
                char nextc = data[kp + 1];
                switch (nextc) {
                    case '"':  key_raw[kr++] = '"';  break;
                    case '\\': key_raw[kr++] = '\\'; break;
                    case '/':  key_raw[kr++] = '/';  break;
                    case 'n':  key_raw[kr++] = '\n'; break;
                    case 't':  key_raw[kr++] = '\t'; break;
                    case 'r':  key_raw[kr++] = '\r'; break;
                    case 'b':  key_raw[kr++] = '\b'; break;
                    case 'f':  key_raw[kr++] = '\f'; break;
                    case 'u': {
                        // \uXXXX hex — skip 4 hex digits, output replacement
                        kp += 5;  // skip \uXXXX
                        continue;
                    }
                    default:   key_raw[kr++] = nextc; break;
                }
                kp += 2;
            } else {
                key_raw[kr++] = data[kp++];
            }
        }
        key_raw[kr] = '\0';

        // Skip to after closing quote
        pos = (data[pos] == '"') ? pos + 1 : key_end + 1;
        while (pos < (size_t)file_len && data[pos] != '"' && data[pos] != ':') pos++;
        // We should be at ':'
        if (pos < (size_t)file_len && data[pos] == ':') pos++;
        if (pos < (size_t)file_len && data[pos] == '"') {
            // String value — skip past it (shouldn't happen in vocab.json)
            pos++;
            while (pos < (size_t)file_len) {
                if (data[pos] == '\\') pos += 2;
                else if (data[pos] == '"') { pos++; break; }
                else pos++;
            }
        } else {
            // Integer value
            int val = 0;
            int sign = 1;
            while (pos < (size_t)file_len && data[pos] <= ' ') pos++;
            if (pos < (size_t)file_len && data[pos] == '-') { sign = -1; pos++; }
            while (pos < (size_t)file_len && data[pos] >= '0' && data[pos] <= '9') {
                val = val * 10 + (data[pos] - '0');
                pos++;
            }
            val *= sign;

            if (val >= 0 && val < max_entries && id_to_str[val] == NULL) {
                id_to_str[val] = key_raw;
                if (val > max_id) max_id = val;
                count++;
            } else {
                free(key_raw);
            }
        }
    }

    free(data);
    *out_size = max_id + 1;

    if (count == 0) {
        free(id_to_str);
        return NULL;
    }

    return id_to_str;
}

// ---------------------------------------------------------------------------
// Public API: decode token IDs to text
// ---------------------------------------------------------------------------
char* qwen3asr_decode_tokens(const char *model_dir, const int *tokens, int n_tokens) {
    _build_byte_decoder();

    // Build vocab path
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);

    // Load vocab (id → token string mapping)
    int vocab_size = 0;
    char **id_to_str = load_vocab(vocab_path, &vocab_size);
    if (!id_to_str) {
        fprintf(stderr, "WARN: failed to load vocab from %s, using byte fallback\n", vocab_path);
        // Fallback: simple byte pass-through
        char *fallback = (char*)malloc((size_t)n_tokens * 4 + 1);
        if (!fallback) return NULL;
        size_t pos = 0;
        for (int i = 0; i < n_tokens && pos < (size_t)n_tokens * 4; i++) {
            // For byte tokens (0-255), use bytes_to_unicode reverse mapping
            if (tokens[i] >= 0 && tokens[i] < 256) {
                // In bytes_to_unicode, token IDs 0-93 = bytes 33-126
                // This is a rough approximation — exact mapping requires the full table
                fallback[pos++] = (char)tokens[i];
            } else {
                pos += snprintf(fallback + pos, 4, "<%d>", tokens[i]);
                if (pos > (size_t)n_tokens * 4 - 10) break;
            }
        }
        fallback[pos] = '\0';
        return fallback;
    }

    // Decode each token
    // First pass: estimate total decoded length (each token string ≤ 64 bytes typically)
    // But we'll build incrementally for simplicity
    size_t capacity = 4096;
    unsigned char *decoded = (unsigned char*)malloc(capacity);
    if (!decoded) { free(id_to_str); free(decoded); return NULL; }
    size_t decoded_len = 0;

    for (int i = 0; i < n_tokens; i++) {
        int tid = tokens[i];
        if (tid < 0 || tid >= vocab_size || !id_to_str[tid]) continue;

        const char *token_str = id_to_str[tid];
        size_t token_len = strlen(token_str);

        // Decode the token string bytes (UTF-8 → reverse bytes_to_unicode)
        size_t part_len = 0;
        unsigned char *part = decode_token_string(
            (const unsigned char*)token_str, token_len, &part_len);

        if (part && part_len > 0) {
            // Ensure capacity
            if (decoded_len + part_len >= capacity) {
                capacity = (decoded_len + part_len) * 2;
                unsigned char *new_decoded = (unsigned char*)realloc(decoded, capacity);
                if (!new_decoded) { free(part); free(decoded); free(id_to_str); return NULL; }
                decoded = new_decoded;
            }
            memcpy(decoded + decoded_len, part, part_len);
            decoded_len += part_len;
            free(part);
        }
    }

    decoded[decoded_len] = '\0';
    free(id_to_str);

    // Return the decoded text (may contain valid UTF-8 at this point)
    return (char*)decoded;
}
