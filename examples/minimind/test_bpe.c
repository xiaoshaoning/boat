// test_bpe.c - Debug BPE merge for 你好
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    minimind_tokenizer_t* tok = minimind_tokenizer_load("./weights");
    if (!tok) {
        printf("Token load fail\n");
        return 1;
    }

    // Check first 10 merges (a, b as hex)
    printf("First 10 merges loaded:\n");
    for (int i = 0; i < 10; i++) {
        printf("  merge[%d]: a=", i);
        for (char* p = tok->merges[i].a; *p; p++)
            printf("%02x ", (unsigned char)*p);
        printf(" b=");
        for (char* p = tok->merges[i].b; *p; p++)
            printf("%02x ", (unsigned char)*p);
        printf("\n");
    }

    // Check: what does text_to_unicode produce for "你好"?
    printf("\n--- text_to_unicode test ---\n");
    // Manually apply byte-to-unicode
    // My C mapping: (33-126) || (161-172) || (174-255) -> self, else 256+n
    const char* text = "\xe4\xbd\xa0\xe5\xa5\xbd"; // 你好 as raw bytes
    int text_len = 6;
    printf("Input bytes: ");
    for (int i = 0; i < text_len; i++)
        printf("%02x ", (unsigned char)text[i]);
    printf("\n");

    // Build byte_to_unicode
    {
        unsigned short byte_to_unicode[256];
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
                byte_to_unicode[b] = (unsigned short)b;
            else
                byte_to_unicode[b] = (unsigned short)(256 + n++);
        }

        // Encode
        char unicode_buf[256];
        int pos = 0;
        for (int i = 0; i < text_len && pos < 250; i++) {
            unsigned int cp = byte_to_unicode[(unsigned char)text[i]];
            if (cp < 0x80) {
                unicode_buf[pos++] = (char)cp;
            } else if (cp < 0x800) {
                unicode_buf[pos++] = (char)(0xC0 | (cp >> 6));
                unicode_buf[pos++] = (char)(0x80 | (cp & 0x3F));
            } else {
                unicode_buf[pos++] = (char)(0xE0 | (cp >> 12));
                unicode_buf[pos++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                unicode_buf[pos++] = (char)(0x80 | (cp & 0x3F));
            }
        }
        unicode_buf[pos] = '\0';

        printf("Unicode output (%d bytes): ", pos);
        for (int i = 0; i < pos; i++)
            printf("%02x ", (unsigned char)unicode_buf[i]);
        printf("\n");

        // Split into individual chars
        char parts[32][8];
        int n_parts = 0;
        const char* u = unicode_buf;
        while (*u && n_parts < 32) {
            int clen;
            if ((unsigned char)*u < 0x80)
                clen = 1;
            else if (((unsigned char)*u & 0xE0) == 0xC0)
                clen = 2;
            else
                clen = 3;
            memcpy(parts[n_parts], u, clen);
            parts[n_parts][clen] = '\0';
            n_parts++;
            u += clen;
        }

        printf("Initial parts (%d):\n", n_parts);
        for (int i = 0; i < n_parts; i++) {
            printf("  [%d]: ", i);
            for (char* p = parts[i]; *p; p++)
                printf("%02x ", (unsigned char)*p);
            printf("\n");
        }

        // Apply BPE merges (debug first few)
        int merge_count = 0;
        while (n_parts > 1) {
            int best_rank = tok->num_merges;
            int best_i = -1;
            for (int i = 0; i < n_parts - 1; i++) {
                for (int mi = 0; mi < tok->num_merges; mi++) {
                    if (strcmp(parts[i], tok->merges[mi].a) == 0 &&
                        strcmp(parts[i + 1], tok->merges[mi].b) == 0) {
                        if (mi < best_rank) {
                            best_rank = mi;
                            best_i = i;
                        }
                        break;
                    }
                }
            }
            if (best_i == -1) break;

            int al = (int)strlen(parts[best_i]);
            int bl = (int)strlen(parts[best_i + 1]);
            if (al + bl < 7) {
                memcpy(parts[best_i] + al, parts[best_i + 1], bl);
                parts[best_i][al + bl] = '\0';
            }
            for (int i = best_i + 1; i < n_parts - 1; i++)
                memcpy(parts[i], parts[i + 1], 8);
            n_parts--;
            merge_count++;

            if (merge_count <= 10) {
                printf("  merge #%d rank=%d: merged[%d]=", merge_count, best_rank, best_i);
                for (char* p = parts[best_i]; *p; p++)
                    printf("%02x ", (unsigned char)*p);
                printf(" (a=");
                for (char* p = tok->merges[best_rank].a; *p; p++)
                    printf("%02x ", (unsigned char)*p);
                printf("b=");
                for (char* p = tok->merges[best_rank].b; *p; p++)
                    printf("%02x ", (unsigned char)*p);
                printf(")\n");
            }
        }
        printf("Total merges applied: %d, final parts: %d\n", merge_count, n_parts);
        for (int i = 0; i < n_parts; i++) {
            printf("  final[%d]: ", i);
            for (char* p = parts[i]; *p; p++)
                printf("%02x ", (unsigned char)*p);
            // Look up in vocab
            int found = 0;
            for (int v = 0; v < tok->vocab_size; v++) {
                if (tok->vocab[v][0] && strcmp(tok->vocab[v], parts[i]) == 0) {
                    printf(" -> id=%d", v);
                    found = 1;
                    break;
                }
            }
            if (!found) printf(" -> NOT FOUND");
            printf("\n");
        }
    }

    minimind_tokenizer_free(tok);
    return 0;
}
