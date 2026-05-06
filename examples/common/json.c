// json.c - Minimal JSON parser implementation
#include "json.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

void json_init(json_ctx_t* ctx, const char* json_str, size_t len) {
    ctx->data = (char*)json_str;
    ctx->len = len;
    ctx->pos = 0;
}

void json_skip_ws(json_ctx_t* ctx) {
    while (ctx->pos < ctx->len) {
        char c = ctx->data[ctx->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ctx->pos++;
        else break;
    }
}

char json_peek(json_ctx_t* ctx) {
    json_skip_ws(ctx);
    return (ctx->pos < ctx->len) ? ctx->data[ctx->pos] : '\0';
}

char json_next(json_ctx_t* ctx) {
    json_skip_ws(ctx);
    return (ctx->pos < ctx->len) ? ctx->data[ctx->pos++] : '\0';
}

char* json_parse_string(json_ctx_t* ctx) {
    if (json_next(ctx) != '"') return NULL;
    size_t start = ctx->pos;
    while (ctx->pos < ctx->len) {
        if (ctx->data[ctx->pos] == '\\') ctx->pos += 2;
        else if (ctx->data[ctx->pos] == '"') break;
        else ctx->pos++;
    }
    if (ctx->pos >= ctx->len) return NULL;
    size_t len = ctx->pos - start;
    char* s = (char*)malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, ctx->data + start, len);
    s[len] = '\0';
    ctx->pos++;
    return s;
}

int64_t json_parse_int(json_ctx_t* ctx) {
    json_skip_ws(ctx);
    int64_t val = 0;
    int sign = 1;
    if (ctx->pos < ctx->len && ctx->data[ctx->pos] == '-') { sign = -1; ctx->pos++; }
    while (ctx->pos < ctx->len && ctx->data[ctx->pos] >= '0' && ctx->data[ctx->pos] <= '9') {
        val = val * 10 + (ctx->data[ctx->pos] - '0');
        ctx->pos++;
    }
    return sign * val;
}

double json_parse_number(json_ctx_t* ctx) {
    json_skip_ws(ctx);
    char* end;
    double val = strtod(ctx->data + ctx->pos, &end);
    ctx->pos = (size_t)(end - ctx->data);
    return val;
}

int json_expect(json_ctx_t* ctx, char c) {
    json_skip_ws(ctx);
    if (ctx->pos < ctx->len && ctx->data[ctx->pos] == c) { ctx->pos++; return 1; }
    return 0;
}

int json_find_key(json_ctx_t* ctx, const char* key) {
    int depth = 0;
    while (ctx->pos < ctx->len) {
        json_skip_ws(ctx);
        if (ctx->pos >= ctx->len) break;
        char c = ctx->data[ctx->pos];
        if (c == '{' || c == '[') { depth++; ctx->pos++; }
        else if (c == '}' || c == ']') { depth--; ctx->pos++; }
        else if (c == '"') {
            size_t save = ctx->pos;
            char* k = json_parse_string(ctx);
            if (!k) break;
            json_skip_ws(ctx);
            int found = (ctx->pos < ctx->len && ctx->data[ctx->pos] == ':');
            if (found && strcmp(k, key) == 0) {
                ctx->pos = save;
                free(k);
                return 1;
            }
            free(k);
            if (found) { ctx->pos++; json_skip_value(ctx); }
        }
        else ctx->pos++;
    }
    return 0;
}

void json_skip_value(json_ctx_t* ctx) {
    json_skip_ws(ctx);
    if (ctx->pos >= ctx->len) return;
    char c = ctx->data[ctx->pos];
    if (c == '{') {
        int depth = 0;
        while (ctx->pos < ctx->len) {
            char cc = ctx->data[ctx->pos++];
            if (cc == '{') depth++;
            else if (cc == '}') { if (--depth == 0) return; }
            else if (cc == '"') { while (ctx->pos < ctx->len) { if (ctx->data[ctx->pos] == '\\') ctx->pos += 2; else if (ctx->data[ctx->pos++] == '"') break; } }
        }
    } else if (c == '[') {
        int depth = 0;
        while (ctx->pos < ctx->len) {
            char cc = ctx->data[ctx->pos++];
            if (cc == '[') depth++;
            else if (cc == ']') { if (--depth == 0) return; }
            else if (cc == '"') { while (ctx->pos < ctx->len) { if (ctx->data[ctx->pos] == '\\') ctx->pos += 2; else if (ctx->data[ctx->pos++] == '"') break; } }
        }
    } else if (c == '"') {
        ctx->pos++;
        while (ctx->pos < ctx->len) { if (ctx->data[ctx->pos] == '\\') ctx->pos += 2; else if (ctx->data[ctx->pos++] == '"') break; }
    } else {
        while (ctx->pos < ctx->len) {
            char cc = ctx->data[ctx->pos];
            if (cc == ',' || cc == '}' || cc == ']' || cc == ':') break;
            ctx->pos++;
        }
    }
}
