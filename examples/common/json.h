// json.h - Minimal JSON parser for safetensors headers and config files
#ifndef BOAT_EXAMPLE_JSON_H
#define BOAT_EXAMPLE_JSON_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char* data;
    size_t len;
    size_t pos;
} json_ctx_t;

void json_init(json_ctx_t* ctx, const char* json_str, size_t len);
void json_skip_ws(json_ctx_t* ctx);
char json_peek(json_ctx_t* ctx);
char json_next(json_ctx_t* ctx);
char* json_parse_string(json_ctx_t* ctx);
int64_t json_parse_int(json_ctx_t* ctx);
double json_parse_number(json_ctx_t* ctx);
int json_expect(json_ctx_t* ctx, char c);
int json_find_key(json_ctx_t* ctx, const char* key);
void json_skip_value(json_ctx_t* ctx);

#endif // BOAT_EXAMPLE_JSON_H
