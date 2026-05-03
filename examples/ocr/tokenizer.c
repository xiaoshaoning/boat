// tokenizer.c - GLM-OCR BPE tokenizer (GPT-2 byte-level BPE)
// Loads tokenizer.json from HuggingFace format
#include "tokenizer.h"
#include "../common/json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Initialize byte-to-unicode mapping (GPT-2 style, matching HuggingFace tokenizers)
// bytes 33-126 (ASCII printable) and 161-172, 174-255 (Latin-1 supplement) kept as-is
// All other bytes (0-32, 127-160, 173) mapped to U+0100 onward
static void init_byte_encoding(ocr_tokenizer_t* tok) {
    // Step 1: determine which bytes are "kept" (same codepoint)
    int kept[256] = {0};
    for (int b = 33; b <= 126; b++) kept[b] = 1;       // ASCII printable
    for (int b = 161; b <= 172; b++) kept[b] = 1;      // Latin-1 ¡-¬
    for (int b = 174; b <= 255; b++) kept[b] = 1;      // Latin-1 ®-ÿ

    int n = 0;
    for (int b = 0; b < 256; b++) {
        unsigned int cp;
        if (kept[b]) {
            cp = (unsigned int)b;
        } else {
            cp = 256 + n;
            n++;
        }
        // Encode codepoint as UTF-8
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

// Simple hash for string keys
static unsigned int str_hash(const char* s) {
    unsigned int h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s; h *= 16777619u; }
    return h;
}

// Build a reverse mapping from unicode string back to byte value using codepoint
// Instead of string compare, extract the codepoint of the first character and reverse-map
static int unicode_to_byte(ocr_tokenizer_t* tok, const char* s) {
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
    // Reverse map: bytes 33-126, 161-172, 174-255 kept as-is
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255))
        return (int)cp;
    // Remaining bytes: first byte 0 (cp=256), byte 1 (cp=257), ..., byte 32 (cp=288), byte 127 (cp=289), etc.
    if (cp >= 256 && cp < 256 + 256) {
        int offset = (int)(cp - 256);
        // List of non-kept bytes in order: 0..32 (33), 127..160 (34), 173 (1) = 68 total
        if (offset < 33) return offset;             // bytes 0-32
        if (offset < 33 + 34) return 127 + (offset - 33);  // bytes 127-160
        if (offset < 33 + 34 + 1) return 173;       // byte 173
    }
    return -1;
}

int ocr_tokenizer_init(ocr_tokenizer_t* tok, const char* vocab_path) {
    memset(tok, 0, sizeof(*tok));

    // Initialize byte encoding table
    init_byte_encoding(tok);

    // Read tokenizer.json
    FILE* f = fopen(vocab_path, "rb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open tokenizer file: %s\n", vocab_path); return 0; }
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

    // Skip to model object
    if (json_next(&jctx) != '{') { free(json_data); return 0; }

    // Find "model" key and enter its object
    if (!json_find_key(&jctx, "model")) { free(json_data); return 0; }
    char* mod_key = json_parse_string(&jctx);
    free(mod_key);
    json_skip_ws(&jctx);
    if (json_next(&jctx) != ':') { free(json_data); return 0; }
    if (json_next(&jctx) != '{') { free(json_data); return 0; }

    // Parse model object keys
    tok->tokens = (char**)calloc(OCR_VOCAB_SIZE, sizeof(char*));
    tok->lengths = (int*)calloc(OCR_VOCAB_SIZE, sizeof(int));

    while (1) {
        json_skip_ws(&jctx);
        if (jctx.pos >= (size_t)fsize || json_data[jctx.pos] == '}') break;
        char* key = json_parse_string(&jctx);
        if (!key) break;
        json_skip_ws(&jctx);
        json_expect(&jctx, ':');

        if (strcmp(key, "vocab") == 0) {
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

                    if (id >= 0 && id < OCR_VOCAB_SIZE && !tok->tokens[id]) {
                        tok->tokens[id] = token_str;
                        tok->lengths[id] = (int)strlen(token_str);
                        if (id >= tok->n) tok->n = (int)(id + 1);
                    } else {
                        free(token_str);
                    }
                }
                if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == '}') jctx.pos++;
            }
        } else if (strcmp(key, "merges") == 0) {
            if (json_next(&jctx) == '[') {
                int cap = 1024;
                tok->merge_pairs = (char**)malloc(cap * sizeof(char*));
                tok->merge_priorities = (int*)malloc(cap * sizeof(int));
                tok->num_merges = 0;

                while (1) {
                    json_skip_ws(&jctx);
                    if (jctx.pos >= (size_t)fsize || json_data[jctx.pos] == ']') break;
                    // Merges are stored as arrays: ["token1", "token2"]
                    if (json_next(&jctx) != '[') { fprintf(stderr, "[DEBUG] merge: expected '[' at pos %zu\n", jctx.pos); break; }
                    char* first = json_parse_string(&jctx);
                    if (!first) { fprintf(stderr, "[DEBUG] merge: failed to parse first token at pos %zu\n", jctx.pos); break; }
                    json_skip_ws(&jctx);
                    if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ',') jctx.pos++;
                    char* second = json_parse_string(&jctx);
                    if (!second) { fprintf(stderr, "[DEBUG] merge: failed to parse second token at pos %zu\n", jctx.pos); free(first); break; }
                    // Consume closing ']'
                    json_skip_ws(&jctx);
                    if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ']') jctx.pos++;
                    // Consume comma after the array if present
                    json_skip_ws(&jctx);
                    if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ',') jctx.pos++;

                    // Build "first second" string as the merge key
                    int first_len = (int)strlen(first);
                    int second_len = (int)strlen(second);
                    int key_len = first_len + 1 + second_len;
                    char* key = (char*)malloc(key_len + 1);
                    memcpy(key, first, first_len);
                    key[first_len] = ' ';
                    memcpy(key + first_len + 1, second, second_len);
                    key[key_len] = '\0';
                    free(first);
                    free(second);

                    if (tok->num_merges >= cap) {
                        cap *= 2;
                        tok->merge_pairs = (char**)realloc(tok->merge_pairs, cap * sizeof(char*));
                        tok->merge_priorities = (int*)realloc(tok->merge_priorities, cap * sizeof(int));
                    }
                    tok->merge_pairs[tok->num_merges] = key;
                    tok->merge_priorities[tok->num_merges] = tok->num_merges;
                    tok->num_merges++;
                }
                fprintf(stderr, "[DEBUG] Merges parsed: %d\n", tok->num_merges);
                if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ']') jctx.pos++;
            }
        } else {
            json_skip_value(&jctx);
        }
        free(key);
        json_skip_ws(&jctx);
        if (jctx.pos < (size_t)fsize && json_data[jctx.pos] == ',') jctx.pos++;
    }

    free(json_data);

    // Set special token IDs
    tok->unk_id = 59246;  // <|endoftext|> serves as UNK
    tok->eos_id = 59246;  // <|endoftext|>
    tok->sop_id = 59250;  // <sop>
    tok->eop_id = 59251;  // <eop>
    tok->image_token_id = 59280;
    tok->img_start_id = 59256;
    tok->img_end_id = 59257;
    tok->gmask_id = 59248;
    tok->user_role_id = 59253;
    tok->assistant_role_id = 59254;
    tok->think_id = 59267;
    tok->endthink_id = 59268;
    tok->newline_id = 10;

    // Ensure special tokens exist in vocab
    if (!tok->tokens[59246]) tok->tokens[59246] = strdup("<|endoftext|>");
    if (!tok->tokens[59250]) tok->tokens[59250] = strdup("<sop>");
    if (!tok->tokens[59251]) tok->tokens[59251] = strdup("<eop>");
    if (!tok->tokens[59256]) tok->tokens[59256] = strdup("<|begin_of_image|>");
    if (!tok->tokens[59257]) tok->tokens[59257] = strdup("<|end_of_image|>");
    if (!tok->tokens[59280]) tok->tokens[59280] = strdup("<|image|>");
    if (!tok->tokens[59248]) tok->tokens[59248] = strdup("[gMASK]");
    if (!tok->tokens[59253]) tok->tokens[59253] = strdup("<|user|>");
    if (!tok->tokens[59254]) tok->tokens[59254] = strdup("<|assistant|>");
    if (!tok->tokens[59267]) tok->tokens[59267] = strdup("<think>");
    if (!tok->tokens[59268]) tok->tokens[59268] = strdup("</think>");

    for (int i = 0; i < OCR_VOCAB_SIZE; i++) {
        if (tok->tokens[i]) tok->lengths[i] = (int)strlen(tok->tokens[i]);
    }

    // Allocate workspace for encoding
    tok->work_cap = 65536;
    tok->work_ids = (int*)malloc(tok->work_cap * sizeof(int));
    tok->work_tokens = (char**)malloc(tok->work_cap * sizeof(char*));

    fprintf(stderr, "[INFO] Tokenizer: vocab=%d/%d, merges=%d\n", tok->n, OCR_VOCAB_SIZE, tok->num_merges);
    return 1;
}

void ocr_tokenizer_free(ocr_tokenizer_t* tok) {
    if (!tok) return;
    if (tok->tokens) {
        for (int i = 0; i < OCR_VOCAB_SIZE; i++) free(tok->tokens[i]);
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

// Find merge priority for a pair of tokens (returns -1 if not mergeable)
static int find_merge_priority(const ocr_tokenizer_t* tok, const char* a, const char* b) {
    // Build pair key: "A B"
    int alen = (int)strlen(a);
    int blen = (int)strlen(b);
    int key_len = alen + 1 + blen;
    char* key = (char*)malloc(key_len + 1);
    if (!key) return -1;
    memcpy(key, a, alen);
    key[alen] = ' ';
    memcpy(key + alen + 1, b, blen);
    key[key_len] = '\0';

    // Linear scan through merges (106K entries - acceptable for one-time scan per batch)
    // In practice, this could be optimized with a hash map
    int best = -1;
    for (int i = 0; i < tok->num_merges; i++) {
        if (strcmp(key, tok->merge_pairs[i]) == 0) {
            best = i;
            break;
        }
    }
    free(key);
    return best;
}

// Convert a text segment to its byte-level unicode representation
// Returns number of bytes in output (not counting null terminator)
static int text_to_bytes_unicode(const ocr_tokenizer_t* tok, const char* text, size_t text_len,
                                  char* out, int out_cap) {
    int out_pos = 0;
    for (size_t i = 0; i < text_len && out_pos < out_cap - 8; ) {
        unsigned char c = (unsigned char)text[i];
        // Determine UTF-8 sequence length
        int seq_len;
        if (c < 0x80) seq_len = 1;
        else if (c < 0xE0) seq_len = 2;
        else if (c < 0xF0) seq_len = 3;
        else seq_len = 4;

        // Extract codepoint
        unsigned int cp = 0;
        if (seq_len == 1) cp = c;
        else if (seq_len == 2) cp = ((unsigned int)(text[i] & 0x1F) << 6) | (unsigned int)(text[i + 1] & 0x3F);
        else if (seq_len == 3) cp = ((unsigned int)(text[i] & 0x0F) << 12) | ((unsigned int)(text[i + 1] & 0x3F) << 6) | (unsigned int)(text[i + 2] & 0x3F);
        else cp = ((unsigned int)(text[i] & 0x07) << 18) | ((unsigned int)(text[i + 1] & 0x3F) << 12) | ((unsigned int)(text[i + 2] & 0x3F) << 6) | (unsigned int)(text[i + 3] & 0x3F);

        // Encode to UTF-8 bytes then map each byte to unicode char
        unsigned char bytes[4];
        int nbytes;
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

// Decode byte-encoded unicode back to original text
static char* unicode_to_text(const ocr_tokenizer_t* tok, const char* s) {
    // First pass: build byte sequence
    size_t len = strlen(s);
    unsigned char* bytes = (unsigned char*)malloc(len + 1);
    int nbytes = 0;

    // Try to map each unicode token back to a byte
    // We need to split s into individual unicode characters and map each
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

        // Try to map back to byte using fixed reverse mapping
        int byte_val = unicode_to_byte(tok, tmp);
        if (byte_val >= 0 && byte_val <= 255) {
            bytes[nbytes++] = (unsigned char)byte_val;
        }
        pos += seq_len;
    }

    // Convert byte sequence to UTF-8 string (it's already UTF-8 since original bytes were UTF-8)
    char* result = (char*)malloc(nbytes + 1);
    memcpy(result, bytes, nbytes);
    result[nbytes] = '\0';
    free(bytes);
    return result;
}

// Find the best BPE merge pair in the current token sequence
// Returns the index of the first token in the best pair, or -1 if no merge possible
static int find_best_pair(const ocr_tokenizer_t* tok, const char** tokens, int n, int* out_priority) {
    int best_priority = tok->num_merges;  // higher than any actual priority
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

int* ocr_tokenizer_encode(const ocr_tokenizer_t* tok, const char* text, size_t text_len, int* out_len) {
    // Convert text to byte-level unicode representation
    char* unicode_text = (char*)malloc(text_len * 8 + 1);
    if (!unicode_text) { *out_len = 0; return NULL; }
    int unicode_len = text_to_bytes_unicode(tok, text, text_len, unicode_text, text_len * 8);

    // Simple pre-tokenization: split on whitespace, keeping leading space with non-first words
    // Matches GPT-2 behavior where " read" becomes ["Ġread"] (space prefixed to non-first word)
    // Store as array of (start, end) ranges
    int max_words = 65536;
    int* word_starts = (int*)malloc(max_words * sizeof(int));
    int* word_ends = (int*)malloc(max_words * sizeof(int));
    int num_words = 0;

    int i = 0;
    while (i < unicode_len && num_words < max_words) {
        // If this position is whitespace (space, tab, newline, cr), include it as part of next word
        if (unicode_text[i] == ' ' || unicode_text[i] == '\t' ||
            unicode_text[i] == '\n' || unicode_text[i] == '\r') {
            // Include leading whitespace with the next word
            int ws_start = i;
            while (i < unicode_len && (unicode_text[i] == ' ' || unicode_text[i] == '\t' ||
                   unicode_text[i] == '\n' || unicode_text[i] == '\r')) i++;
            // If more text follows, include whitespace as prefix
            if (i < unicode_len) {
                // Find end of word
                int word_start = ws_start;  // include preceding whitespace
                while (i < unicode_len && unicode_text[i] != ' ' && unicode_text[i] != '\t' &&
                       unicode_text[i] != '\n' && unicode_text[i] != '\r') i++;
                word_starts[num_words] = word_start;
                word_ends[num_words] = i;
                num_words++;
            }
            // If no more text, trailing whitespace is ignored (GPT-2 behavior)
        } else {
            // No leading whitespace (first word)
            word_starts[num_words] = i;
            while (i < unicode_len && unicode_text[i] != ' ' && unicode_text[i] != '\t' &&
                   unicode_text[i] != '\n' && unicode_text[i] != '\r') i++;
            word_ends[num_words] = i;
            num_words++;
        }
    }

    // Allocate result (upper bound)
    int result_cap = unicode_len + num_words + 256;
    int* result = (int*)malloc(result_cap * sizeof(int));
    int result_len = 0;

    // Process each word through BPE
    for (int w = 0; w < num_words; w++) {
        int wlen = word_ends[w] - word_starts[w];
        const char* wstart = unicode_text + word_starts[w];

        // Start with character-level tokens
        int ntokens = 0;
        int pos = 0;
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

            // Merge best pair
            int merged_len = (int)(strlen(tok->work_tokens[best_idx]) + strlen(tok->work_tokens[best_idx + 1]));
            char* merged = (char*)malloc(merged_len + 1);
            strcpy(merged, tok->work_tokens[best_idx]);
            strcat(merged, tok->work_tokens[best_idx + 1]);

            free(tok->work_tokens[best_idx]);
            tok->work_tokens[best_idx] = merged;

            // Shift remaining tokens
            for (int j = best_idx + 1; j < ntokens - 1; j++) {
                tok->work_tokens[j] = tok->work_tokens[j + 1];
            }
            ntokens--;
            changed = 1;
        } while (changed);

        // Look up each token in vocab
        for (int j = 0; j < ntokens; j++) {
            // Try direct vocab lookup
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
                // Use UNK token
                result[result_len++] = tok->unk_id;
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

char* ocr_tokenizer_decode(const ocr_tokenizer_t* tok, const int* ids, int n_ids) {
    // First pass: concatenate all token strings
    size_t total = 0;
    for (int i = 0; i < n_ids; i++) {
        if (ids[i] >= 0 && ids[i] < OCR_VOCAB_SIZE && tok->tokens[ids[i]]) {
            total += strlen(tok->tokens[ids[i]]);
        }
    }

    char* concat = (char*)malloc(total + 1);
    if (!concat) return NULL;
    int pos = 0;
    for (int i = 0; i < n_ids; i++) {
        if (ids[i] >= 0 && ids[i] < OCR_VOCAB_SIZE && tok->tokens[ids[i]]) {
            int len = tok->lengths[ids[i]];
            memcpy(concat + pos, tok->tokens[ids[i]], len);
            pos += len;
        }
    }
    concat[pos] = '\0';

    // Convert byte-encoded unicode back to text
    char* text = unicode_to_text(tok, concat);
    free(concat);
    return text;
}
