// tokenizer.c - NanoChat BPE tokenizer (GPT-2 byte-level BPE)
#include "tokenizer.h"
#include "../common/json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void init_byte_encoding(nanochat_tokenizer_t* tok) {
    int kept[256] = {0};
    for (int b = 33; b <= 126; b++) kept[b] = 1;
    for (int b = 161; b <= 172; b++) kept[b] = 1;
    for (int b = 174; b <= 255; b++) kept[b] = 1;

    int n = 0;
    for (int b = 0; b < 256; b++) {
        unsigned int cp;
        if (kept[b]) {
            cp = (unsigned int)b;
        } else {
            cp = 256 + n;
            n++;
        }
        unsigned char buf[4]; int blen = 0;
        if (cp < 0x80) {
            buf[blen++] = (unsigned char)cp;
        } else if (cp < 0x800) {
            buf[blen++] = 0xC0 | (unsigned char)(cp >> 6);
            buf[blen++] = 0x80 | (unsigned char)(cp & 0x3F);
        } else {
            buf[blen++] = 0xE0 | (unsigned char)(cp >> 12);
            buf[blen++] = 0x80 | (unsigned char)((cp >> 6) & 0x3F);
            buf[blen++] = 0x80 | (unsigned char)(cp & 0x3F);
        }
        buf[blen] = 0;
        memcpy(tok->byte_to_unicode[b], buf, blen + 1);
    }
}

static int unicode_to_byte(nanochat_tokenizer_t* tok, const char* s) {
    (void)tok;
    unsigned char c = (unsigned char)s[0];
    if (c == 0) return -1;
    unsigned int cp;
    if (c < 0x80) {
        cp = c;
    } else if (c < 0xE0) {
        cp = ((unsigned int)(s[0] & 0x1F) << 6) | (unsigned int)(s[1] & 0x3F);
    } else if (c < 0xF0) {
        cp = ((unsigned int)(s[0] & 0x0F) << 12) | ((unsigned int)(s[1] & 0x3F) << 6) | (unsigned int)(s[2] & 0x3F);
    } else {
        cp = ((unsigned int)(s[0] & 0x07) << 18) | ((unsigned int)(s[1] & 0x3F) << 12) | ((unsigned int)(s[2] & 0x3F) << 6) | ((unsigned int)(s[3] & 0x3F));
    }
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255))
        return (int)cp;
    if (cp >= 256 && cp < 256 + 256) {
        int offset = (int)(cp - 256);
        if (offset < 33) return offset;
        if (offset < 33 + 34) return 127 + (offset - 33);
        if (offset < 33 + 34 + 1) return 173;
    }
    return -1;
}

int nanochat_tokenizer_init(nanochat_tokenizer_t* tok, const char* vocab_path) {
    memset(tok, 0, sizeof(*tok));

    init_byte_encoding(tok);

    FILE* f = fopen(vocab_path, "rb");
    if (!f) { fprintf(stderr, "[NanoChat] Cannot open tokenizer file: %s\n", vocab_path); return 0; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* json_data = (char*)malloc(fsize + 1);
    if (!json_data) { fclose(f); return 0; }
    if (fread(json_data, 1, fsize, f) != (size_t)fsize) { free(json_data); fclose(f); return 0; }
    fclose(f);
    json_data[fsize] = '\0';

    json_ctx_t jctx;
    json_init(&jctx, json_data, fsize);

    if (json_next(&jctx) != '{') { free(json_data); return 0; }

    tok->tokens = (char**)calloc(NANOCHAT_VOCAB_SIZE, sizeof(char*));
    tok->lengths = (int*)calloc(NANOCHAT_VOCAB_SIZE, sizeof(int));

    // Parse top-level keys
    while (1) {
        json_skip_ws(&jctx);
        if (jctx.pos >= (size_t)fsize || json_data[jctx.pos] == '}') break;
        char* key = json_parse_string(&jctx);
        if (!key) break;
        json_skip_ws(&jctx);
        json_expect(&jctx, ':');

        if (strcmp(key, "added_tokens") == 0 && json_next(&jctx) == '[') {
            // Parse added tokens (special tokens at end of vocab)
            while (1) {
                json_skip_ws(&jctx);
                if (jctx.pos >= (size_t)fsize || json_data[jctx.pos] == ']') break;
                if (json_next(&jctx) != '{') break;
                int t_id = 0;
                char* t_content = NULL;
                while (1) {
                    json_skip_ws(&jctx);
                    if (jctx.pos >= (size_t)fsize || json_data[jctx.pos] == '}') break;
                    char* tk = json_parse_string(&jctx);
                    if (!tk) break;
                    json_skip_ws(&jctx); json_expect(&jctx, ':');
                    if (strcmp(tk, "id") == 0) {
                        t_id = (int)json_parse_int(&jctx);
                    } else if (strcmp(tk, "content") == 0) {
                        t_content = json_parse_string(&jctx);
                    } else {
                        json_skip_value(&jctx);
                    }
                    free(tk);
                    json_skip_ws(&jctx);
                    if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ',') jctx.pos++;
                }
                if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == '}') jctx.pos++;
                json_skip_ws(&jctx);
                if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ',') jctx.pos++;

                if (t_content && t_id >= 0 && t_id < NANOCHAT_VOCAB_SIZE) {
                    if (!tok->tokens[t_id]) {
                        tok->tokens[t_id] = t_content;
                        tok->lengths[t_id] = (int)strlen(t_content);
                        if (t_id >= tok->n) tok->n = t_id + 1;
                    } else {
                        free(t_content);
                    }
                } else {
                    free(t_content);
                }
            }
            if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ']') jctx.pos++;

        } else if (strcmp(key, "model") == 0) {
            json_skip_ws(&jctx);
            if (json_next(&jctx) != '{') { free(key); free(json_data); return 0; }

            // Parse model object keys
            while (1) {
                json_skip_ws(&jctx);
                if (jctx.pos >= (size_t)fsize || json_data[jctx.pos] == '}') break;
                char* mkey = json_parse_string(&jctx);
                if (!mkey) break;
                json_skip_ws(&jctx);
                json_expect(&jctx, ':');

                if (strcmp(mkey, "vocab") == 0) {
                    if (json_next(&jctx) == '{') {
                        while (1) {
                            json_skip_ws(&jctx);
                            if (jctx.pos >= (size_t)fsize || json_data[jctx.pos] == '}') break;
                            char* token_str = json_parse_string(&jctx);
                            if (!token_str) break;
                            json_skip_ws(&jctx);
                            json_expect(&jctx, ':');
                            int64_t id = json_parse_int(&jctx);
                            json_skip_ws(&jctx);
                            if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ',') jctx.pos++;

                            if (id >= 0 && id < NANOCHAT_VOCAB_SIZE && !tok->tokens[id]) {
                                tok->tokens[id] = token_str;
                                tok->lengths[id] = (int)strlen(token_str);
                                if (id >= tok->n) tok->n = (int)(id + 1);
                            } else {
                                free(token_str);
                            }
                        }
                        if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == '}') jctx.pos++;
                    }
                } else if (strcmp(mkey, "merges") == 0) {
                    if (json_next(&jctx) == '[') {
                        int cap = 4096;
                        tok->merge_pairs = (char**)malloc(cap * sizeof(char*));
                        tok->merge_priorities = (int*)malloc(cap * sizeof(int));
                        tok->num_merges = 0;

                        while (1) {
                            json_skip_ws(&jctx);
                            if (jctx.pos >= (size_t)fsize || json_data[jctx.pos] == ']') break;
                            if (json_next(&jctx) != '[') break;
                            char* first = json_parse_string(&jctx);
                            if (!first) break;
                            json_skip_ws(&jctx);
                            if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ',') jctx.pos++;
                            char* second = json_parse_string(&jctx);
                            if (!second) { free(first); break; }
                            json_skip_ws(&jctx);
                            if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ']') jctx.pos++;
                            json_skip_ws(&jctx);
                            if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ',') jctx.pos++;

                            int first_len = (int)strlen(first);
                            int second_len = (int)strlen(second);
                            int key_len = first_len + 1 + second_len;
                            char* pair = (char*)malloc(key_len + 1);
                            memcpy(pair, first, first_len);
                            pair[first_len] = ' ';
                            memcpy(pair + first_len + 1, second, second_len);
                            pair[key_len] = '\0';
                            free(first); free(second);

                            if (tok->num_merges >= cap) {
                                cap *= 2;
                                tok->merge_pairs = (char**)realloc(tok->merge_pairs, cap * sizeof(char*));
                                tok->merge_priorities = (int*)realloc(tok->merge_priorities, cap * sizeof(int));
                            }
                            tok->merge_pairs[tok->num_merges] = pair;
                            tok->merge_priorities[tok->num_merges] = tok->num_merges;
                            tok->num_merges++;
                        }
                        if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ']') jctx.pos++;
                    }
                } else {
                    json_skip_value(&jctx);
                }
                free(mkey);
                json_skip_ws(&jctx);
                if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ',') jctx.pos++;
            }
            if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == '}') jctx.pos++;
        } else {
            json_skip_value(&jctx);
        }
        free(key);
        json_skip_ws(&jctx);
        if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ',') jctx.pos++;
    }

    free(json_data);

    // Special token IDs from config.json
    tok->bos_id = 0;
    tok->eos_id = 1;
    tok->pad_id = 1;
    tok->added_tokens_start = 65527;

    // Ensure special tokens exist
    if (!tok->tokens[0]) tok->tokens[0] = strdup("<|bos|>");
    if (!tok->tokens[1]) tok->tokens[1] = strdup("<|eos|>");
    if (!tok->tokens[65527]) tok->tokens[65527] = strdup("<|bos|>");
    if (!tok->tokens[65528]) tok->tokens[65528] = strdup("<|user_start|>");
    if (!tok->tokens[65529]) tok->tokens[65529] = strdup("<|user_end|>");
    if (!tok->tokens[65530]) tok->tokens[65530] = strdup("<|assistant_start|>");
    if (!tok->tokens[65531]) tok->tokens[65531] = strdup("<|assistant_end|>");
    if (!tok->tokens[65532]) tok->tokens[65532] = strdup("<|python_start|>");
    if (!tok->tokens[65533]) tok->tokens[65533] = strdup("<|python_end|>");
    if (!tok->tokens[65534]) tok->tokens[65534] = strdup("<|output_start|>");
    if (!tok->tokens[65535]) tok->tokens[65535] = strdup("<|output_end|>");

    for (int i = 0; i < NANOCHAT_VOCAB_SIZE; i++) {
        if (tok->tokens[i]) tok->lengths[i] = (int)strlen(tok->tokens[i]);
    }

    // Allocate workspace
    tok->work_cap = 65536;
    tok->work_ids = (int*)malloc(tok->work_cap * sizeof(int));
    tok->work_tokens = (char**)malloc(tok->work_cap * sizeof(char*));

    fprintf(stderr, "[NanoChat] Tokenizer: vocab=%d/%d, merges=%d\n",
            tok->n, NANOCHAT_VOCAB_SIZE, tok->num_merges);
    return 1;
}

void nanochat_tokenizer_free(nanochat_tokenizer_t* tok) {
    if (!tok) return;
    if (tok->tokens) {
        for (int i = 0; i < NANOCHAT_VOCAB_SIZE; i++) free(tok->tokens[i]);
        free(tok->tokens);
    }
    free(tok->lengths);
    if (tok->merge_pairs) {
        for (int i = 0; i < tok->num_merges; i++) free(tok->merge_pairs[i]);
        free(tok->merge_pairs);
    }
    free(tok->merge_priorities);
    free(tok->work_ids);
    free(tok->work_tokens);
    memset(tok, 0, sizeof(*tok));
}

int nanochat_tokenizer_eos_id(const nanochat_tokenizer_t* tok) {
    return tok->eos_id;
}

static int find_merge_priority(const nanochat_tokenizer_t* tok, const char* a, const char* b) {
    int alen = (int)strlen(a);
    int blen = (int)strlen(b);
    int key_len = alen + 1 + blen;
    char* key = (char*)malloc(key_len + 1);
    if (!key) return -1;
    memcpy(key, a, alen);
    key[alen] = ' ';
    memcpy(key + alen + 1, b, blen);
    key[key_len] = '\0';

    for (int i = 0; i < tok->num_merges; i++) {
        if (strcmp(key, tok->merge_pairs[i]) == 0) {
            free(key);
            return i;
        }
    }
    free(key);
    return -1;
}

static int text_to_bytes_unicode(const nanochat_tokenizer_t* tok, const char* text,
                                  size_t text_len, char* out, int out_cap) {
    int out_pos = 0;
    for (size_t i = 0; i < text_len && out_pos < out_cap - 8; ) {
        unsigned char c = (unsigned char)text[i];
        int seq_len;
        if (c < 0x80) seq_len = 1;
        else if (c < 0xE0) seq_len = 2;
        else if (c < 0xF0) seq_len = 3;
        else seq_len = 4;

        unsigned int cp;
        if (seq_len == 1) cp = c;
        else if (seq_len == 2) cp = ((unsigned int)(text[i] & 0x1F) << 6) | (unsigned int)(text[i + 1] & 0x3F);
        else if (seq_len == 3) cp = ((unsigned int)(text[i] & 0x0F) << 12) | ((unsigned int)(text[i + 1] & 0x3F) << 6) | (unsigned int)(text[i + 2] & 0x3F);
        else cp = ((unsigned int)(text[i] & 0x07) << 18) | ((unsigned int)(text[i + 1] & 0x3F) << 12) | ((unsigned int)(text[i + 2] & 0x3F) << 6) | (unsigned int)(text[i + 3] & 0x3F);

        unsigned char bytes[4]; int nbytes;
        if (cp < 0x80) { bytes[0] = cp; nbytes = 1; }
        else if (cp < 0x800) { bytes[0] = 0xC0 | (cp >> 6); bytes[1] = 0x80 | (cp & 0x3F); nbytes = 2; }
        else if (cp < 0x10000) { bytes[0] = 0xE0 | (cp >> 12); bytes[1] = 0x80 | ((cp >> 6) & 0x3F); bytes[2] = 0x80 | (cp & 0x3F); nbytes = 3; }
        else { bytes[0] = 0xF0 | (cp >> 18); bytes[1] = 0x80 | ((cp >> 12) & 0x3F); bytes[2] = 0x80 | ((cp >> 6) & 0x3F); bytes[3] = 0x80 | (cp & 0x3F); nbytes = 4; }

        for (int j = 0; j < nbytes && out_pos < out_cap - 5; j++) {
            const char* mapped = tok->byte_to_unicode[bytes[j]];
            int mlen = (int)strlen(mapped);
            memcpy(out + out_pos, mapped, mlen);
            out_pos += mlen;
        }
        i += seq_len;
    }
    out[out_pos] = '\0';
    return out_pos;
}

static char* unicode_to_text(const nanochat_tokenizer_t* tok, const char* s) {
    size_t len = strlen(s);
    unsigned char* bytes = (unsigned char*)malloc(len + 1);
    int nbytes = 0;
    size_t pos = 0;
    while (pos < len) {
        unsigned char c = (unsigned char)s[pos];
        int seq_len;
        if (c < 0x80) seq_len = 1;
        else if (c < 0xE0) seq_len = 2;
        else if (c < 0xF0) seq_len = 3;
        else seq_len = 4;
        if (pos + seq_len > len) break;
        char tmp[16];
        memcpy(tmp, s + pos, seq_len);
        tmp[seq_len] = '\0';
        int byte_val = unicode_to_byte((nanochat_tokenizer_t*)tok, tmp);
        if (byte_val >= 0 && byte_val <= 255) {
            bytes[nbytes++] = (unsigned char)byte_val;
        }
        pos += seq_len;
    }
    char* result = (char*)malloc(nbytes + 1);
    // Skip leading NULL bytes (e.g. from BOS token)
    int skip = 0;
    while (skip < nbytes && bytes[skip] == 0) skip++;
    int out_len = nbytes - skip;
    memcpy(result, bytes + skip, out_len);
    result[out_len] = '\0';
    free(bytes);
    return result;
}

static int find_best_pair(const nanochat_tokenizer_t* tok, const char** tokens, int n, int* out_priority) {
    int best_priority = tok->num_merges;
    int best_idx = -1;
    for (int i = 0; i < n - 1; i++) {
        int pri = find_merge_priority(tok, tokens[i], tokens[i + 1]);
        if (pri >= 0 && pri < best_priority) {
            best_priority = pri;
            best_idx = i;
        }
    }
    *out_priority = best_priority;
    return best_idx;
}

int* nanochat_tokenizer_encode(const nanochat_tokenizer_t* tok, const char* text,
                                 size_t text_len, int* out_len) {
    char* unicode_text = (char*)malloc(text_len * 8 + 1);
    if (!unicode_text) { *out_len = 0; return NULL; }
    int unicode_len = text_to_bytes_unicode(tok, text, text_len, unicode_text, text_len * 8);

    // Pre-tokenization: split on whitespace (GPT-2 style)
    int max_words = 65536;
    int* word_starts = (int*)malloc(max_words * sizeof(int));
    int* word_ends = (int*)malloc(max_words * sizeof(int));
    int num_words = 0;

    int i = 0;
    while (i < unicode_len && num_words < max_words) {
        if (unicode_text[i] == ' ' || unicode_text[i] == '\t' ||
            unicode_text[i] == '\n' || unicode_text[i] == '\r') {
            int ws_start = i;
            while (i < unicode_len && (unicode_text[i] == ' ' || unicode_text[i] == '\t' ||
                   unicode_text[i] == '\n' || unicode_text[i] == '\r')) i++;
            if (i < unicode_len) {
                int word_start = ws_start;
                while (i < unicode_len && unicode_text[i] != ' ' && unicode_text[i] != '\t' &&
                       unicode_text[i] != '\n' && unicode_text[i] != '\r') i++;
                word_starts[num_words] = word_start;
                word_ends[num_words] = i;
                num_words++;
            }
        } else {
            word_starts[num_words] = i;
            while (i < unicode_len && unicode_text[i] != ' ' && unicode_text[i] != '\t' &&
                   unicode_text[i] != '\n' && unicode_text[i] != '\r') i++;
            word_ends[num_words] = i;
            num_words++;
        }
    }

    int result_cap = unicode_len + num_words + 256;
    int* result = (int*)malloc(result_cap * sizeof(int));
    int result_len = 0;

    for (int w = 0; w < num_words; w++) {
        int wlen = word_ends[w] - word_starts[w];
        const char* wstart = unicode_text + word_starts[w];

        int ntokens = 0, pos = 0;
        while (pos < wlen) {
            unsigned char c = (unsigned char)wstart[pos];
            int seq_len;
            if (c < 0x80) seq_len = 1;
            else if (c < 0xE0) seq_len = 2;
            else if (c < 0xF0) seq_len = 3;
            else seq_len = 4;
            if (pos + seq_len > wlen) break;
            if (ntokens >= tok->work_cap) break;
            memcpy(tok->work_tokens[ntokens] = (char*)malloc(seq_len + 1), wstart + pos, seq_len);
            tok->work_tokens[ntokens][seq_len] = '\0';
            ntokens++;
            pos += seq_len;
        }

        if (ntokens == 0) continue;

        // Apply BPE merges
        int changed;
        do {
            changed = 0;
            int best_priority;
            int best_idx = find_best_pair(tok, (const char**)tok->work_tokens, ntokens, &best_priority);
            if (best_idx < 0) break;

            int merged_len = (int)(strlen(tok->work_tokens[best_idx]) + strlen(tok->work_tokens[best_idx + 1]));
            char* merged = (char*)malloc(merged_len + 1);
            strcpy(merged, tok->work_tokens[best_idx]);
            strcat(merged, tok->work_tokens[best_idx + 1]);

            free(tok->work_tokens[best_idx]);
            tok->work_tokens[best_idx] = merged;

            for (int j = best_idx + 1; j < ntokens - 1; j++)
                tok->work_tokens[j] = tok->work_tokens[j + 1];
            ntokens--;
            changed = 1;
        } while (changed);

        for (int j = 0; j < ntokens; j++) {
            int found = -1;
            for (int id = 0; id < tok->n; id++) {
                if (tok->tokens[id] && strcmp(tok->tokens[id], tok->work_tokens[j]) == 0) {
                    found = id;
                    break;
                }
            }
            if (found >= 0) {
                result[result_len++] = found;
            } else {
                // Fall back to BPE character-level: store the raw unicode bytes as tokens
                // For byte-level BPE, map each unicode char to its byte equivalent
                const char* wtok = tok->work_tokens[j];
                size_t wtok_len = strlen(wtok);
                for (size_t k = 0; k < wtok_len; ) {
                    unsigned char c = (unsigned char)wtok[k];
                    int seq_len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
                    if (k + seq_len > wtok_len) break;
                    char tmp[16];
                    memcpy(tmp, wtok + k, seq_len);
                    tmp[seq_len] = '\0';
                    int byte_val = unicode_to_byte((nanochat_tokenizer_t*)tok, tmp);
                    if (byte_val >= 0 && byte_val < NANOCHAT_VOCAB_SIZE && tok->tokens[byte_val]) {
                        result[result_len++] = byte_val;
                    }
                    k += seq_len;
                }
            }
            free(tok->work_tokens[j]);
        }
    }

    free(unicode_text);
    free(word_starts);
    free(word_ends);

    *out_len = result_len;
    return result;
}

char* nanochat_tokenizer_decode(const nanochat_tokenizer_t* tok, const int* ids, int n_ids) {
    size_t total = 0;
    for (int i = 0; i < n_ids; i++) {
        if (ids[i] >= 0 && ids[i] < NANOCHAT_VOCAB_SIZE && tok->tokens[ids[i]]) {
            total += strlen(tok->tokens[ids[i]]);
        }
    }

    if (total == 0) {
        char* empty = (char*)malloc(1);
        if (empty) empty[0] = '\0';
        return empty;
    }

    char* concat = (char*)malloc(total + 1);
    if (!concat) return NULL;
    int pos = 0;
    for (int i = 0; i < n_ids; i++) {
        if (ids[i] >= 0 && ids[i] < NANOCHAT_VOCAB_SIZE && tok->tokens[ids[i]]) {
            int len = tok->lengths[ids[i]];
            memcpy(concat + pos, tok->tokens[ids[i]], len);
            pos += len;
        }
    }
    concat[pos] = '\0';

    char* text = unicode_to_text(tok, concat);
    free(concat);
    return text;
}
