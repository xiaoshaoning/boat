// json.h - Minimal JSON parser for safetensors header and vocab.json
#ifndef BOAT_EXAMPLE_JSON_H
#define BOAT_EXAMPLE_JSON_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char* data;
    size_t len;
    size_t pos;
} json_ctx_t;

// Initialize JSON parser context
void json_init(json_ctx_t* ctx, const char* json_str, size_t len);

// Skip whitespace
void json_skip_ws(json_ctx_t* ctx);

// Peek current character
char json_peek(json_ctx_t* ctx);

// Read next non-whitespace character
char json_next(json_ctx_t* ctx);

// Parse a JSON string (allocates with malloc, caller frees)
char* json_parse_string(json_ctx_t* ctx);

// Parse a JSON number (returns as int64 or double)
int64_t json_parse_int(json_ctx_t* ctx);
double json_parse_number(json_ctx_t* ctx);

// Expect a specific character
int json_expect(json_ctx_t* ctx, char c);

// Find a specific key at the current object level and position after it.
// Returns 1 if found, 0 if not.
int json_find_key(json_ctx_t* ctx, const char* key);

// Skip one JSON value
void json_skip_value(json_ctx_t* ctx);

#endif // BOAT_EXAMPLE_JSON_H
