// test_unicode.c - Minimal byte-to-unicode test
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// GPT-2 byte-to-unicode mapping
static void build_map(unsigned short* btou) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
            btou[b] = (unsigned short)b;
        else
            btou[b] = (unsigned short)(256 + (n++));
    }
}

static void test_text(const char* name, const char* utf8_text) {
    unsigned short btou[256];
    build_map(btou);

    int text_len = (int)strlen(utf8_text);
    printf("\n=== %s ===\n", name);
    printf("UTF-8 text: '%s'\n", utf8_text);
    printf("Raw bytes (%d): ", text_len);
    for (int i = 0; i < text_len; i++)
        printf("%02x ", (unsigned char)utf8_text[i]);
    printf("\n");

    // Convert to unicode chars
    printf("Unicode codepoints: ");
    for (int i = 0; i < text_len; i++) {
        unsigned int cp = btou[(unsigned char)utf8_text[i]];
        printf("U+%04X ", cp);
    }
    printf("\n");

    // Encode as UTF-8
    char unicode_utf8[256];
    int pos = 0;
    for (int i = 0; i < text_len && pos < 250; i++) {
        unsigned int cp = btou[(unsigned char)utf8_text[i]];
        if (cp < 0x80) {
            unicode_utf8[pos++] = (char)cp;
        } else if (cp < 0x800) {
            unicode_utf8[pos++] = (char)(0xC0 | (cp >> 6));
            unicode_utf8[pos++] = (char)(0x80 | (cp & 0x3F));
        } else {
            unicode_utf8[pos++] = (char)(0xE0 | (cp >> 12));
            unicode_utf8[pos++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            unicode_utf8[pos++] = (char)(0x80 | (cp & 0x3F));
        }
    }
    unicode_utf8[pos] = '\0';
    printf("Unicode UTF-8 bytes (%d): ", pos);
    for (int i = 0; i < pos; i++)
        printf("%02x ", (unsigned char)unicode_utf8[i]);
    printf("\n");

    // Split into individual characters
    printf("Split chars:\n");
    const char* p = unicode_utf8;
    int idx = 0;
    while (*p) {
        int clen;
        if ((unsigned char)*p < 0x80)
            clen = 1;
        else if (((unsigned char)*p & 0xE0) == 0xC0)
            clen = 2;
        else
            clen = 3;
        printf("  char[%d]: ", idx);
        for (int j = 0; j < clen; j++)
            printf("%02x ", (unsigned char)p[j]);
        // Show as codepoint
        unsigned int cp;
        if (clen == 1)
            cp = (unsigned char)p[0];
        else if (clen == 2)
            cp = ((unsigned char)p[0] & 0x1F) << 6 | ((unsigned char)p[1] & 0x3F);
        else
            cp = ((unsigned char)p[0] & 0x0F) << 12 | ((unsigned char)p[1] & 0x3F) << 6 |
                 ((unsigned char)p[2] & 0x3F);
        printf(" (U+%04X)\n", cp);
        p += clen;
        idx++;
    }
}

int main() {
    test_text("你好", "\xe4\xbd\xa0\xe5\xa5\xbd"); // 你好 as raw bytes
    test_text("user\\n", "user\n");
    test_text("newline only", "\n");
    return 0;
}
