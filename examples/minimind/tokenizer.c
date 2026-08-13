// tokenizer.c - Byte-Level BPE tokenizer for MiniMind
// GPT-2 style byte-level BPE, 6400 vocab, 6108 merges.
#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_VOCAB 6400
#define MAX_MERGES 6200
#define MAX_TOKEN_LEN 32

typedef struct {
    char a[MAX_TOKEN_LEN];
    char b[MAX_TOKEN_LEN];
    int rank;
} merge_t;

struct minimind_tokenizer_t {
    char vocab[MAX_VOCAB][MAX_TOKEN_LEN];
    int vocab_size;
    merge_t merges[MAX_MERGES];
    int num_merges;
    // Added tokens (matched as complete strings before BPE)
    // Tokens with IDs < 36, e.g. <|im_start|>, <|im_end|>, <think>, etc.
    char added_token_strs[128][MAX_TOKEN_LEN];
    int added_token_ids[128];
    int num_added_tokens;
    // Byte-to-unicode lookup (GPT-2 mapping)
    unsigned short byte_to_unicode[256];
    // Unicode-to-byte reverse lookup
    unsigned char unicode_to_byte[65536];
};

// GPT-2 byte-to-unicode mapping
static void build_byte_unicode_map(minimind_tokenizer_t* tok) {
    // Printable ASCII bytes map to themselves in the unicode range
    // bytes 33-126 map to unicode codepoints 33-126
    // bytes 161-172 map to 161-172
    // bytes 174-255 map to 174-255
    // Other bytes 0-32, 127-160, 173 map to 256-512 range

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) ||
            (b >= 161 && b <= 172) ||
            (b >= 174 && b <= 255)) {
            tok->byte_to_unicode[b] = (unsigned short)b;
        } else {
            tok->byte_to_unicode[b] = (unsigned short)(256 + n);
            n++;
        }
    }

    // Build reverse mapping
    memset(tok->unicode_to_byte, 0, sizeof(tok->unicode_to_byte));
    for (int b = 0; b < 256; b++) {
        tok->unicode_to_byte[tok->byte_to_unicode[b]] = (unsigned char)b;
    }
}

// Convert Unicode codepoint to UTF-8 bytes
static int codepoint_to_utf8(unsigned int cp, char* out) {
    if (cp < 0x80) { out[0] = (char)cp; out[1] = '\0'; return 1; }
    else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = '\0'; return 2;
    } else {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        out[3] = '\0'; return 3;
    }
}

// Encode text into byte-level unicode string
// Returns malloc'd string of codepoints (as UTF-8), caller frees
static char* text_to_unicode(const minimind_tokenizer_t* tok, const char* text) {
    int len = (int)strlen(text);
    // Each byte → 1 unicode char (1-3 UTF-8 bytes)
    char* out = (char*)malloc((size_t)len * 4 + 1);
    if (!out) return NULL;
    out[0] = '\0';

    // Encode text as UTF-8 bytes first, then map each byte to unicode
    for (int i = 0; i < len; i++) {
        unsigned char byte = (unsigned char)text[i];
        unsigned int cp = tok->byte_to_unicode[byte];
        char utf8[5];
        codepoint_to_utf8(cp, utf8);
        strcat(out, utf8);
    }
    return out;
}

// Decode unicode string back to bytes/text
static char* unicode_to_text(const minimind_tokenizer_t* tok, const char* unicode_str) {
    int len = (int)strlen(unicode_str);
    char* out = (char*)malloc((size_t)len + 1);
    if (!out) return NULL;
    int pos = 0;
    const char* p = unicode_str;
    while (*p) {
        unsigned int cp;
        if ((unsigned char)*p < 0x80) {
            cp = (unsigned int)(unsigned char)*p;
            p++;
        } else if (((unsigned char)*p & 0xE0) == 0xC0) {
            cp = (unsigned int)(((unsigned char)p[0] & 0x1F) << 6) |
                 (unsigned int)((unsigned char)p[1] & 0x3F);
            p += 2;
        } else {
            cp = (unsigned int)(((unsigned char)p[0] & 0x0F) << 12) |
                 (unsigned int)(((unsigned char)p[1] & 0x3F) << 6) |
                 (unsigned int)((unsigned char)p[2] & 0x3F);
            p += 3;
        }
        unsigned char byte = tok->unicode_to_byte[cp];
        out[pos++] = (char)byte;
    }
    out[pos] = '\0';
    return out;
}

// --- File reading ---
static char* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    *out_len = fread(buf, 1, sz, f);
    buf[*out_len] = '\0';
    fclose(f);
    return buf;
}

// --- JSON helpers for parsing tokenizer.json ---
static const char* find_section(const char* json, const char* key) {
    char search[256];
    int key_len = snprintf(search, sizeof(search), "\"%s\"", key);
    const char* p = json;
    while (1) {
        p = strstr(p, search);
        if (!p) return NULL;
        const char* q = p + key_len;
        while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
        if (*q == ':') {
            q++;
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
            return q;
        }
        p++;
    }
}

// Encode a unicode codepoint as UTF-8, return number of bytes written
static int encode_utf8(unsigned int cp, char* out) {
    if (cp < 0x80) {
        out[0] = (char)cp; return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
}

static char* parse_string(const char** pp) {
    const char* p = *pp;
    while (*p && *p != '"') p++;
    if (*p != '"') return NULL;
    p++; // skip opening "

    size_t max_len = strlen(p) + 1;
    char* out = (char*)malloc(max_len * 3); // worst case: every char is \\uXXXX -> 3 UTF-8 bytes
    if (!out) return NULL;

    int out_pos = 0;
    while (*p) {
        if (*p == '\\') {
            p++;
            if (*p == 'u') {
                p++; // skip 'u'
                // Parse 4 hex digits
                int cp = 0;
                for (int i = 0; i < 4; i++) {
                    char c = *p;
                    cp <<= 4;
                    if (c >= '0' && c <= '9') cp += c - '0';
                    else if (c >= 'a' && c <= 'f') cp += c - 'a' + 10;
                    else if (c >= 'A' && c <= 'F') cp += c - 'A' + 10;
                    else break;
                    p++;
                }
                out_pos += encode_utf8((unsigned int)cp, out + out_pos);
                continue;
            } else if (*p == 'n') {
                out[out_pos++] = '\n'; p++;
            } else if (*p == 't') {
                out[out_pos++] = '\t'; p++;
            } else if (*p == 'r') {
                out[out_pos++] = '\r'; p++;
            } else if (*p == '"') {
                out[out_pos++] = '"'; p++;
            } else if (*p == '\\') {
                out[out_pos++] = '\\'; p++;
            } else {
                p++; // skip unknown escape
            }
        } else if (*p == '"') {
            p++; // skip closing "
            break;
        } else {
            // Regular byte: copy through (these are ASCII printables)
            out[out_pos++] = *p++;
        }
    }

    out[out_pos] = '\0';
    *pp = p;
    return out;
}

// --- Load tokenizer.json ---
minimind_tokenizer_t* minimind_tokenizer_load(const char* model_dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);

    size_t json_len;
    char* json = read_file(path, &json_len);
    if (!json) {
        snprintf(path, sizeof(path), "%s/../tokenizer.json", model_dir);
        json = read_file(path, &json_len);
        if (!json) { fprintf(stderr, "Cannot find tokenizer.json\n"); return NULL; }
    }

    minimind_tokenizer_t* tok = (minimind_tokenizer_t*)calloc(1, sizeof(*tok));
    if (!tok) { free(json); return NULL; }

    build_byte_unicode_map(tok);

    // Parse "model" section
    const char* model = find_section(json, "model");
    if (!model || *model != '{') { printf("model section not found\n"); goto fail; }

    // Parse vocab
    const char* vocab_start = find_section(model, "vocab");
    if (!vocab_start || *vocab_start != '{') { printf("vocab not found\n"); goto fail; }

    const char* p = vocab_start + 1;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
        if (*p == '}') break;
        char* token = parse_string(&p);
        if (!token) break;
        while (*p == ' ' || *p == '\t' || *p == ':') p++;
        int id = 0;
        while (*p >= '0' && *p <= '9') { id = id * 10 + (*p - '0'); p++; }
        if (id >= 0 && id < MAX_VOCAB) {
            strncpy(tok->vocab[id], token, MAX_TOKEN_LEN - 1);
            if (id + 1 > tok->vocab_size) tok->vocab_size = id + 1;
        }
        free(token);
    }

    // Parse merges (arrays of [token_a, token_b])
    const char* merges_start = find_section(model, "merges");
    if (merges_start && *merges_start == '[') {
        p = merges_start + 1;
        int mi = 0;
        while (*p && mi < MAX_MERGES) {
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
            if (*p == ']') break;
            if (*p != '[') { p++; continue; } // skip non-array entries
            p++; // skip [

            char* token_a = parse_string(&p);
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
            char* token_b = parse_string(&p);
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ']') p++;
            if (p && *p == ',') { } // will skip in loop

            if (token_a && token_b) {
                strncpy(tok->merges[mi].a, token_a, MAX_TOKEN_LEN - 1);
                strncpy(tok->merges[mi].b, token_b, MAX_TOKEN_LEN - 1);
                tok->merges[mi].rank = mi;
                free(token_a); free(token_b);
                mi++;
            } else {
                free(token_a); free(token_b);
            }
        }
        tok->num_merges = mi;
    }

    // Parse added_tokens
    const char* added = find_section(json, "added_tokens");
    if (added && *added == '[') {
        p = added + 1;
        while (*p) {
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
            if (*p == ']') break;
            if (*p != '{') { p++; continue; }
            p++; // skip {
            int at_id = -1;
            char at_content[MAX_TOKEN_LEN] = "";
            while (*p && *p != '}') {
                while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ',' || *p == '\r') p++;
                if (*p == '}') break;
                char* key = parse_string(&p);
                if (!key) break;
                while (*p == ' ' || *p == '\t' || *p == ':') p++;

                if (strcmp(key, "id") == 0) {
                    at_id = 0;
                    while (*p >= '0' && *p <= '9') { at_id = at_id * 10 + (*p - '0'); p++; }
                } else if (strcmp(key, "content") == 0) {
                    char* val = parse_string(&p);
                    if (val) { strncpy(at_content, val, MAX_TOKEN_LEN - 1); free(val); }
                } else {
                    // Skip value
                    if (*p == '"') { free(parse_string(&p)); }
                    else if (*p == '{' || *p == '[') {
                        int depth = 0; char open = *p, close = (open == '{') ? '}' : ']';
                        while (*p) { if (*p == open) depth++; else if (*p == close && --depth == 0) { p++; break; } p++; }
                    } else { while (*p && *p != ',' && *p != '}' && *p != '\n') p++; }
                }
                free(key);
            }
            if (*p == '}') p++;
            if (at_id >= 0 && at_id < MAX_VOCAB && at_id >= tok->vocab_size) {
                strncpy(tok->vocab[at_id], at_content, MAX_TOKEN_LEN - 1);
                if (at_id + 1 > tok->vocab_size) tok->vocab_size = at_id + 1;
            }
        }
    }

    // Build added_tokens list (first 36 tokens are "added" and should be matched as-is)
    tok->num_added_tokens = 0;
    for (int i = 0; i < 36 && i < tok->vocab_size && tok->num_added_tokens < 128; i++) {
        if (tok->vocab[i][0]) {
            strncpy(tok->added_token_strs[tok->num_added_tokens], tok->vocab[i], MAX_TOKEN_LEN - 1);
            tok->added_token_ids[tok->num_added_tokens] = i;
            tok->num_added_tokens++;
        }
    }

    free(json);
    printf("Tokenizer loaded: %d vocab, %d merges, %d added tokens\n",
           tok->vocab_size, tok->num_merges, tok->num_added_tokens);
    return tok;

fail:
    free(json);
    minimind_tokenizer_free(tok);
    return NULL;
}

void minimind_tokenizer_free(minimind_tokenizer_t* tok) {
    if (!tok) return;
    free(tok);
}

// --- ID lookup ---
static int token_to_id(minimind_tokenizer_t* tok, const char* token) {
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->vocab[i][0] && strcmp(tok->vocab[i], token) == 0)
            return i;
    }
    return -1;
}

// --- Build merge rank lookup ---
// For fast pair->rank lookup during BPE encoding
#define MERGE_HASH_SIZE 65536

// --- BPE encode a text segment (no added tokens) ---
static int bpe_encode_segment(minimind_tokenizer_t* tok, const char* text,
                               int* tokens_out, int max_len) {
    if (!text || !*text) return 0;

    char* unicode_str = text_to_unicode(tok, text);
    if (!unicode_str || !unicode_str[0]) { free(unicode_str); return 0; }

    typedef struct { char s[64]; } tok_part_t;
    int u_len = (int)strlen(unicode_str);
    int max_n = u_len + 16;
    tok_part_t* parts = (tok_part_t*)malloc((size_t)max_n * sizeof(tok_part_t));
    int n = 0;
    const char* up = unicode_str;
    while (*up && n < max_n) {
        int clen;
        if ((unsigned char)*up < 0x80) clen = 1;
        else if (((unsigned char)*up & 0xE0) == 0xC0) clen = 2;
        else clen = 3;
        if (n < max_n) {
            memcpy(parts[n].s, up, (size_t)clen);
            parts[n].s[clen] = '\0';
            n++;
        }
        up += clen;
    }

    // Standard BPE loop
    while (n > 1) {
        int best_rank = tok->num_merges;
        int best_i = -1;
        for (int i = 0; i < n - 1; i++) {
            for (int mi = 0; mi < tok->num_merges; mi++) {
                if (strcmp(parts[i].s, tok->merges[mi].a) == 0 &&
                    strcmp(parts[i + 1].s, tok->merges[mi].b) == 0) {
                    if (mi < best_rank) { best_rank = mi; best_i = i; }
                    break;
                }
            }
        }
        if (best_i == -1) break;
        int al = (int)strlen(parts[best_i].s);
        int bl = (int)strlen(parts[best_i + 1].s);
        if (al + bl < 63) {
            memcpy(parts[best_i].s + al, parts[best_i + 1].s, (size_t)bl);
            parts[best_i].s[al + bl] = '\0';
        }
        for (int i = best_i + 1; i < n - 1; i++)
            memcpy(&parts[i], &parts[i + 1], sizeof(tok_part_t));
        n--;
    }

    int out_n = 0;
    for (int i = 0; i < n && out_n < max_len; i++) {
        int id = token_to_id(tok, parts[i].s);
        if (id >= 0) tokens_out[out_n++] = id;
    }
    free(parts);
    free(unicode_str);
    return out_n;
}

// --- BPE Encode with added-token pre-scanning ---
int minimind_tokenizer_encode(minimind_tokenizer_t* tok, const char* text,
                               int* tokens_out, int max_len) {
    if (!text || !*text) return 0;

    int out_n = 0;
    const char* p = text;
    while (*p && out_n < max_len) {
        // Find longest matching added token at current position
        int best_len = 0, best_id = -1;
        for (int a = 0; a < tok->num_added_tokens; a++) {
            int alen = (int)strlen(tok->added_token_strs[a]);
            if (alen > best_len && strncmp(p, tok->added_token_strs[a], alen) == 0) {
                best_len = alen;
                best_id = tok->added_token_ids[a];
            }
        }

        if (best_id >= 0) {
            // Emit added token
            tokens_out[out_n++] = best_id;
            p += best_len;
        } else {
            // Find next added token or end of string
            const char* seg_end = p;
            while (*seg_end) {
                int matched = 0;
                for (int a = 0; a < tok->num_added_tokens; a++) {
                    int alen = (int)strlen(tok->added_token_strs[a]);
                    if (strncmp(seg_end, tok->added_token_strs[a], alen) == 0) {
                        matched = 1; break;
                    }
                }
                if (matched) break;
                seg_end++;
            }
            // BPE-encode the segment [p, seg_end)
            int seg_len = (int)(seg_end - p);
            if (seg_len > 0) {
                char* seg = (char*)malloc((size_t)(seg_len + 1));
                memcpy(seg, p, (size_t)seg_len);
                seg[seg_len] = '\0';
                int n_added = bpe_encode_segment(tok, seg, tokens_out + out_n, max_len - out_n);
                out_n += n_added;
                free(seg);
            }
            p = seg_end;
        }
    }
    return out_n;
}

// --- Decode ---
char* minimind_tokenizer_decode(minimind_tokenizer_t* tok,
                                 const int* tokens, int n_tokens) {
    // Concatenate all token strings
    size_t total = 0;
    for (int i = 0; i < n_tokens; i++) {
        int tid = tokens[i];
        if (tid >= 0 && tid < tok->vocab_size && tok->vocab[tid][0]) {
            total += strlen(tok->vocab[tid]);
        }
    }

    char* unicode_str = (char*)malloc(total + 1);
    if (!unicode_str) return NULL;
    unicode_str[0] = '\0';
    for (int i = 0; i < n_tokens; i++) {
        int tid = tokens[i];
        if (tid >= 0 && tid < tok->vocab_size && tok->vocab[tid][0]) {
            strcat(unicode_str, tok->vocab[tid]);
        }
    }

    // Convert unicode back to bytes/text
    char* text = unicode_to_text(tok, unicode_str);
    free(unicode_str);
    return text;
}

int minimind_tokenizer_vocab_size(minimind_tokenizer_t* tok) {
    return tok->vocab_size;
}
