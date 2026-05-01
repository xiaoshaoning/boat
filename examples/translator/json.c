// json.c - Minimal JSON parser
#include "json.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

void json_init(json_ctx_t* ctx, const char* json_str, size_t len) {
    ctx->data = (char*)json_str;
    ctx->len = len;
    ctx->pos = 0;
}

void json_skip_ws(json_ctx_t* ctx) {
    while (ctx->pos < ctx->len) {
        char c = ctx->data[ctx->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            ctx->pos++;
        } else {
            break;
        }
    }
}

char json_peek(json_ctx_t* ctx) {
    json_skip_ws(ctx);
    return ctx->pos < ctx->len ? ctx->data[ctx->pos] : '\0';
}

char json_next(json_ctx_t* ctx) {
    json_skip_ws(ctx);
    return ctx->pos < ctx->len ? ctx->data[ctx->pos++] : '\0';
}

char* json_parse_string(json_ctx_t* ctx) {
    if (json_next(ctx) != '"') return NULL;
    size_t start = ctx->pos;
    size_t cap = 64, len = 0;
    char* result = (char*)malloc(cap);
    if (!result) return NULL;

    while (ctx->pos < ctx->len) {
        char c = ctx->data[ctx->pos++];
        if (c == '"') {
            result[len] = '\0';
            return result;
        }
        if (c == '\\') {
            if (ctx->pos >= ctx->len) { free(result); return NULL; }
            char esc = ctx->data[ctx->pos++];
            switch (esc) {
                case '"':  c = '"'; break;
                case '\\': c = '\\'; break;
                case '/':  c = '/'; break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case 'u': {
                    // Decode \uXXXX to UTF-8
                    if (ctx->pos + 4 > ctx->len) { free(result); return NULL; }
                    unsigned int cp = 0;
                    for (int i = 0; i < 4; i++) {
                        char hex = ctx->data[ctx->pos++];
                        cp <<= 4;
                        if (hex >= '0' && hex <= '9')      cp |= (hex - '0');
                        else if (hex >= 'a' && hex <= 'f') cp |= (hex - 'a' + 10);
                        else if (hex >= 'A' && hex <= 'F') cp |= (hex - 'A' + 10);
                        else { free(result); return NULL; }
                    }
                    // Encode codepoint to UTF-8
                    if (cp < 0x80) {
                        if (len + 1 >= cap) { cap *= 2; result = realloc(result, cap); if (!result) return NULL; }
                        result[len++] = (char)cp;
                    } else if (cp < 0x800) {
                        if (len + 2 >= cap) { cap *= 2; result = realloc(result, cap); if (!result) return NULL; }
                        result[len++] = (char)(0xC0 | (cp >> 6));
                        result[len++] = (char)(0x80 | (cp & 0x3F));
                    } else {
                        if (len + 3 >= cap) { cap *= 2; result = realloc(result, cap); if (!result) return NULL; }
                        result[len++] = (char)(0xE0 | (cp >> 12));
                        result[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        result[len++] = (char)(0x80 | (cp & 0x3F));
                    }
                    continue; // skip the result[len++] = c at the bottom
                }
                default: break;
            }
        }
        if (len + 1 >= cap) {
            cap *= 2;
            char* tmp = (char*)realloc(result, cap);
            if (!tmp) { free(result); return NULL; }
            result = tmp;
        }
        result[len++] = c;
    }
    free(result);
    return NULL;
}

int64_t json_parse_int(json_ctx_t* ctx) {
    json_skip_ws(ctx);
    int64_t val = 0;
    int neg = 0;
    if (ctx->pos < ctx->len && ctx->data[ctx->pos] == '-') {
        neg = 1;
        ctx->pos++;
    }
    while (ctx->pos < ctx->len && isdigit((unsigned char)ctx->data[ctx->pos])) {
        val = val * 10 + (ctx->data[ctx->pos++] - '0');
    }
    return neg ? -val : val;
}

double json_parse_number(json_ctx_t* ctx) {
    json_skip_ws(ctx);
    const char* start = ctx->data + ctx->pos;
    if (ctx->pos < ctx->len && ctx->data[ctx->pos] == '-') ctx->pos++;
    while (ctx->pos < ctx->len && isdigit((unsigned char)ctx->data[ctx->pos])) ctx->pos++;
    if (ctx->pos < ctx->len && ctx->data[ctx->pos] == '.') {
        ctx->pos++;
        while (ctx->pos < ctx->len && isdigit((unsigned char)ctx->data[ctx->pos])) ctx->pos++;
    }
    if (ctx->pos < ctx->len && (ctx->data[ctx->pos] == 'e' || ctx->data[ctx->pos] == 'E')) {
        ctx->pos++;
        if (ctx->pos < ctx->len && (ctx->data[ctx->pos] == '-' || ctx->data[ctx->pos] == '+')) ctx->pos++;
        while (ctx->pos < ctx->len && isdigit((unsigned char)ctx->data[ctx->pos])) ctx->pos++;
    }
    return strtod(start, NULL);
}

int json_expect(json_ctx_t* ctx, char c) {
    return json_next(ctx) == c;
}

int json_find_key(json_ctx_t* ctx, const char* key) {
    json_skip_ws(ctx);
    size_t saved = ctx->pos;

    // We expect to be inside an object. Look for "key":
    while (ctx->pos < ctx->len) {
        json_skip_ws(ctx);
        if (ctx->pos >= ctx->len || ctx->data[ctx->pos] == '}') {
            ctx->pos = saved;
            return 0;
        }
        char* k = json_parse_string(ctx);
        if (!k) { ctx->pos = saved; return 0; }
        json_skip_ws(ctx);
        if (!json_expect(ctx, ':')) { free(k); ctx->pos = saved; return 0; }

        if (strcmp(k, key) == 0) {
            free(k);
            return 1;
        }
        free(k);
        json_skip_value(ctx);
        json_skip_ws(ctx);
        if (ctx->pos < ctx->len && ctx->data[ctx->pos] == ',') {
            ctx->pos++;
        }
    }

    ctx->pos = saved;
    return 0;
}

void json_skip_value(json_ctx_t* ctx) {
    json_skip_ws(ctx);
    if (ctx->pos >= ctx->len) return;
    char c = ctx->data[ctx->pos];
    if (c == '"') {
        json_parse_string(ctx);
    } else if (c == '{') {
        ctx->pos++; // skip '{'
        int depth = 1;
        while (ctx->pos < ctx->len && depth > 0) {
            char cc = ctx->data[ctx->pos++];
            if (cc == '{') depth++;
            else if (cc == '}') depth--;
            else if (cc == '"') {
                // skip string
                while (ctx->pos < ctx->len) {
                    if (ctx->data[ctx->pos] == '\\') ctx->pos += 2;
                    else if (ctx->data[ctx->pos++] == '"') break;
                }
            }
        }
    } else if (c == '[') {
        ctx->pos++;
        int depth = 1;
        while (ctx->pos < ctx->len && depth > 0) {
            char cc = ctx->data[ctx->pos++];
            if (cc == '[') depth++;
            else if (cc == ']') depth--;
            else if (cc == '"') {
                while (ctx->pos < ctx->len) {
                    if (ctx->data[ctx->pos] == '\\') ctx->pos += 2;
                    else if (ctx->data[ctx->pos++] == '"') break;
                }
            }
        }
    } else {
        // number, true, false, null
        while (ctx->pos < ctx->len) {
            char cc = ctx->data[ctx->pos];
            if (cc == ',' || cc == '}' || cc == ']' || cc == ' ' || cc == '\t' || cc == '\n' || cc == '\r') break;
            ctx->pos++;
        }
    }
}
