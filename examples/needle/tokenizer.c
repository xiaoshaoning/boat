// tokenizer.c - SentencePiece BPE tokenizer reading the Needle .cact RAW
// tokenizer blob. Reference encoder/decoder: RefTokenizer in
// needle/model/export.py.
//
// Blob layout: header "<IIIIIBBH" (n_pieces, pad, eos, bos, unk, dummy
// prefix flag, byte-fallback flag, pad), then n_pieces records "<fBH"
// (score, type, surface_len) + surface UTF-8 bytes.

#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// U+2581 LOWER ONE EIGHTH BLOCK, the SentencePiece meta-space character.
static const char META_SPACE[] = "\xE2\x96\x81";

typedef struct {
    const char* s;
    uint32_t id;
} entry_t;

static int entry_cmp(const void* a, const void* b) {
    return strcmp(((const entry_t*)a)->s, ((const entry_t*)b)->s);
}

static uint16_t rd_u16(const uint8_t* p) {
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}
static uint32_t rd_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static float rd_f32(const uint8_t* p) {
    uint32_t v = rd_u32(p);
    float f;
    memcpy(&f, &v, sizeof(f));
    return f;
}

// Length of the UTF-8 sequence for the leading byte (0 = invalid/continuation).
static int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 0;
}

int needle_tokenizer_init(needle_tokenizer_t* tok, const uint8_t* blob, size_t nbytes) {
    memset(tok, 0, sizeof(*tok));
    if (nbytes < 24) return -1;
    tok->n_pieces = rd_u32(blob + 0);
    tok->pad_id = rd_u32(blob + 4);
    tok->eos_id = rd_u32(blob + 8);
    tok->bos_id = rd_u32(blob + 12);
    tok->unk_id = rd_u32(blob + 16);
    tok->add_dummy_prefix = blob[20] != 0;
    tok->byte_fallback = blob[21] != 0;

    tok->pieces = (char**)calloc(tok->n_pieces, sizeof(char*));
    tok->scores = (float*)malloc(tok->n_pieces * sizeof(float));
    tok->types = (uint8_t*)malloc(tok->n_pieces);
    tok->order = (uint32_t*)malloc(tok->n_pieces * sizeof(uint32_t));
    tok->byte_ids = (int*)malloc(256 * sizeof(int));
    tok->markers = (uint32_t*)malloc(tok->n_pieces * sizeof(uint32_t));
    if (!tok->pieces || !tok->scores || !tok->types || !tok->order || !tok->byte_ids ||
        !tok->markers) {
        needle_tokenizer_free(tok);
        return -1;
    }
    for (int i = 0; i < 256; i++)
        tok->byte_ids[i] = -1;

    const uint8_t* p = blob + 24;
    uint32_t n_markers = 0;
    for (uint32_t i = 0; i < tok->n_pieces; i++) {
        if ((size_t)(p - blob) + 7 > nbytes) {
            needle_tokenizer_free(tok);
            return -1;
        }
        float score = rd_f32(p);
        uint8_t type = p[4];
        uint16_t len = rd_u16(p + 5);
        p += 7;
        if ((size_t)(p - blob) + len > nbytes) {
            needle_tokenizer_free(tok);
            return -1;
        }
        char* s = (char*)malloc((size_t)len + 1);
        if (!s) {
            needle_tokenizer_free(tok);
            return -1;
        }
        memcpy(s, p, len);
        s[len] = 0;
        tok->pieces[i] = s;
        tok->scores[i] = score;
        tok->types[i] = type;
        p += len;
        if (type == NEEDLE_TK_BYTE) {
            // "<0xXX>"
            if (len == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x') {
                unsigned int v;
                if (sscanf(s + 3, "%02x", &v) == 1 && v < 256) {
                    tok->byte_ids[v] = (int)i;
                }
            }
        }
        if (type == NEEDLE_TK_USER_DEFINED) {
            tok->markers[n_markers++] = i;
        }
    }
    tok->n_markers = n_markers;

    // Sort marker ids by piece length, descending (longest match first).
    for (uint32_t i = 1; i < n_markers; i++) {
        uint32_t v = tok->markers[i];
        size_t vl = strlen(tok->pieces[v]);
        uint32_t j = i;
        while (j > 0 && strlen(tok->pieces[tok->markers[j - 1]]) < vl) {
            tok->markers[j] = tok->markers[j - 1];
            j--;
        }
        tok->markers[j] = v;
    }

    // Sorted index by piece string.
    {
        entry_t* entries = (entry_t*)malloc(tok->n_pieces * sizeof(entry_t));
        if (!entries) {
            needle_tokenizer_free(tok);
            return -1;
        }
        for (uint32_t i = 0; i < tok->n_pieces; i++) {
            entries[i].s = tok->pieces[i];
            entries[i].id = i;
        }
        qsort(entries, tok->n_pieces, sizeof(entry_t), entry_cmp);
        for (uint32_t i = 0; i < tok->n_pieces; i++)
            tok->order[i] = entries[i].id;
        free(entries);
    }
    return 0;
}

void needle_tokenizer_free(needle_tokenizer_t* tok) {
    if (tok->pieces) {
        for (uint32_t i = 0; i < tok->n_pieces; i++)
            free(tok->pieces[i]);
    }
    free(tok->pieces);
    free(tok->scores);
    free(tok->types);
    free(tok->order);
    free(tok->byte_ids);
    free(tok->markers);
    memset(tok, 0, sizeof(*tok));
}

// Binary search for a piece by its exact UTF-8 bytes (up to 512).
static int find_piece(const needle_tokenizer_t* t, const char* s, size_t len) {
    char tmp[512];
    if (len >= sizeof(tmp)) return -1;
    memcpy(tmp, s, len);
    tmp[len] = 0;
    uint32_t lo = 0, hi = t->n_pieces;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        int c = strcmp(tmp, t->pieces[t->order[mid]]);
        if (c == 0) return (int)t->order[mid];
        if (c < 0) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return -1;
}

// One BPE symbol = a span [start, len) into the escaped buffer.
typedef struct {
    uint32_t start;
    uint32_t len;
} sym_t;

// Run BPE merges over `n` symbols, mutating the array in place. Returns the
// final symbol count (<= n).
static size_t bpe_merge(const needle_tokenizer_t* t, const char* esc, sym_t* syms, size_t n) {
    for (;;) {
        int best_j = -1;
        float best_score = 0.0f;
        for (size_t j = 0; j + 1 < n; j++) {
            const char* key = esc + syms[j].start;
            size_t key_len = (size_t)syms[j].len + syms[j + 1].len;
            int id = find_piece(t, key, key_len);
            if (id >= 0 && (best_j < 0 || t->scores[id] > best_score)) {
                best_j = (int)j;
                best_score = t->scores[id];
            }
        }
        if (best_j < 0) break;
        syms[best_j].len += syms[best_j + 1].len;
        for (size_t k = (size_t)best_j + 1; k + 1 < n; k++) {
            syms[k] = syms[k + 1];
        }
        n--;
    }
    return n;
}

static void bpe_flush(const needle_tokenizer_t* t, const char* esc, sym_t* syms, size_t n, int* out,
                      size_t cap, size_t* count) {
    n = bpe_merge(t, esc, syms, n);
    for (size_t k = 0; k < n; k++) {
        const char* s = esc + syms[k].start;
        size_t len = syms[k].len;
        int id = find_piece(t, s, len);
        if (id < 0) {
            if (t->byte_fallback) {
                for (size_t b = 0; b < len; b++) {
                    int bid = t->byte_ids[(unsigned char)s[b]];
                    if (bid >= 0 && *count < cap) out[(*count)++] = bid;
                }
            } else if (*count < cap) {
                out[(*count)++] = (int)t->unk_id;
            }
        } else if (*count < cap) {
            out[(*count)++] = id;
        }
    }
}

int needle_tokenizer_encode(const needle_tokenizer_t* tok, const char* text, int* out_ids,
                            size_t cap) {
    if (!text || text[0] == 0) return 0; // matches RefTokenizer / spm native
    // Escape: ' ' -> META_SPACE; prepend META_SPACE if add_dummy_prefix.
    size_t src_len = strlen(text);
    size_t n_meta = 0;
    for (size_t i = 0; i < src_len; i++) {
        if (text[i] == ' ') n_meta++;
    }
    size_t esc_cap = src_len + n_meta * 2 + 1; // ' ' grows by 2 bytes
    if (tok->add_dummy_prefix) esc_cap += 3;
    char* esc = (char*)malloc(esc_cap + 1);
    if (!esc) return -1;
    char* w = esc;
    if (tok->add_dummy_prefix) {
        memcpy(w, META_SPACE, 3);
        w += 3;
    }
    for (size_t i = 0; i < src_len; i++) {
        if (text[i] == ' ') {
            memcpy(w, META_SPACE, 3);
            w += 3;
        } else {
            *w++ = text[i];
        }
    }
    *w = 0;
    size_t esc_len = (size_t)(w - esc);

    // Symbols: one per code point; merges only ever join adjacent spans so a
    // single fixed array sized by the byte length is sufficient.
    sym_t* syms = (sym_t*)malloc(esc_len * sizeof(sym_t));
    if (!syms) {
        free(esc);
        return -1;
    }
    size_t count = 0;
    size_t nsym = 0; // chars accumulated since the last marker
    size_t i = 0;
    while (i < esc_len) {
        // Longest marker first.
        int marker_id = -1;
        size_t marker_len = 0;
        for (uint32_t m = 0; m < tok->n_markers; m++) {
            const char* piece = tok->pieces[tok->markers[m]];
            size_t plen = strlen(piece);
            if (i + plen <= esc_len && strncmp(esc + i, piece, plen) == 0) {
                marker_id = (int)tok->markers[m];
                marker_len = plen;
                break;
            }
        }
        if (marker_id >= 0) {
            bpe_flush(tok, esc, syms, nsym, out_ids, cap, &count);
            if (count < cap) out_ids[count++] = marker_id;
            nsym = 0;
            i += marker_len;
            continue;
        }
        int clen = utf8_char_len((unsigned char)esc[i]);
        if (clen <= 0) clen = 1;
        syms[nsym].start = (uint32_t)i;
        syms[nsym].len = (uint32_t)clen;
        nsym++;
        i += (size_t)clen;
    }
    bpe_flush(tok, esc, syms, nsym, out_ids, cap, &count);

    free(syms);
    free(esc);
    return (int)count;
}

char* needle_tokenizer_decode(const needle_tokenizer_t* tok, const int* ids, size_t n) {
    size_t cap = 256;
    char* buf = (char*)malloc(cap);
    size_t len = 0;
    if (!buf) return NULL;
    for (size_t i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || (uint32_t)id >= tok->n_pieces) continue;
        uint8_t type = tok->types[id];
        if (type == NEEDLE_TK_CONTROL || type == NEEDLE_TK_UNKNOWN) continue;
        const char* piece = tok->pieces[id];
        if (type == NEEDLE_TK_BYTE) {
            unsigned int b;
            if (sscanf(piece, "<0x%02x>", &b) == 1) {
                if (len + 1 > cap) {
                    cap *= 2;
                    char* nb = (char*)realloc(buf, cap);
                    if (!nb) {
                        free(buf);
                        return NULL;
                    }
                    buf = nb;
                }
                buf[len++] = (char)b;
            }
            continue;
        }
        size_t plen = strlen(piece);
        if (len + plen > cap) {
            while (len + plen > cap)
                cap *= 2;
            char* nb = (char*)realloc(buf, cap);
            if (!nb) {
                free(buf);
                return NULL;
            }
            buf = nb;
        }
        memcpy(buf + len, piece, plen);
        len += plen;
    }
    // META_SPACE bytes -> ' '; drop the dummy leading space if applicable.
    char* out = (char*)malloc(len + 1);
    if (!out) {
        free(buf);
        return NULL;
    }
    size_t o = 0;
    size_t i2 = 0;
    while (i2 < len) {
        if (i2 + 3 <= len && memcmp(buf + i2, META_SPACE, 3) == 0) {
            out[o++] = ' ';
            i2 += 3;
        } else {
            out[o++] = buf[i2++];
        }
    }
    out[o] = 0;
    free(buf);
    if (tok->add_dummy_prefix && o > 0 && out[0] == ' ') {
        memmove(out, out + 1, o); // drop the dummy leading space
    }
    return out;
}
