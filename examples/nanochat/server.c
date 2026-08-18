// server.c - NanoChat LLM Serving API
// Platform-sockets HTTP server with OpenAI-compatible /v1/chat/completions
// and SSE token streaming.
#include "server.h"
#include "engine.h"
#include "tokenizer.h"
#include "config.h"
#include "../common/json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

#ifdef _WIN32
#define strncasecmp _strnicmp
#endif

// ---------------------------------------------------------------------------
// Platform socket and threading abstraction
// ---------------------------------------------------------------------------
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
typedef SRWLOCK mutex_t;
#define INVALID_SOCKET_VAL INVALID_SOCKET
#define SOCKET_ERR(e) ((e) == SOCKET_ERROR)
#define SOCKET_CLOSE(s) closesocket(s)
#define SOCKET_LAST_ERR() WSAGetLastError()
#define MUTEX_INIT(m) InitializeSRWLock(m)
#define MUTEX_LOCK(m) AcquireSRWLockExclusive(m)
#define MUTEX_UNLOCK(m) ReleaseSRWLockExclusive(m)
#define MUTEX_DESTROY(m) ((void)0)
static int platform_init(void) {
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0 ? 0 : -1;
}
static void platform_cleanup(void) {
    WSACleanup();
}
#define SHUT_RDWR SD_BOTH
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <signal.h>
typedef int socket_t;
typedef pthread_mutex_t mutex_t;
#define INVALID_SOCKET_VAL (-1)
#define SOCKET_ERR(e) ((e) < 0)
#define SOCKET_CLOSE(s) close(s)
#define SOCKET_LAST_ERR() errno
#define MUTEX_INIT(m) pthread_mutex_init(m, NULL)
#define MUTEX_LOCK(m) pthread_mutex_lock(m)
#define MUTEX_UNLOCK(m) pthread_mutex_unlock(m)
#define MUTEX_DESTROY(m) pthread_mutex_destroy(m)
static int platform_init(void) {
    signal(SIGPIPE, SIG_IGN);
    return 0;
}
static void platform_cleanup(void) {
}
#endif

// ---------------------------------------------------------------------------
// Server options
// ---------------------------------------------------------------------------
static int g_max_tokens = 512;
static float g_default_temp = 0.7f;
static int g_default_top_k = 40;

void nanochat_server_set_max_tokens(int max_tokens) {
    g_max_tokens = max_tokens;
}
void nanochat_server_set_default_temperature(float temp) {
    g_default_temp = temp;
}
void nanochat_server_set_default_top_k(int top_k) {
    g_default_top_k = top_k;
}

// ---------------------------------------------------------------------------
// HTTP I/O helpers
// ---------------------------------------------------------------------------
// Safe send of a buffer. Returns 0 on success, -1 on connection close/error.
static int send_all(socket_t fd, const char* data, size_t len) {
    while (len > 0) {
#ifdef _WIN32
        int n = send(fd, data, (int)len, 0);
        if (n == SOCKET_ERROR) return -1;
#else
        ssize_t n = write(fd, data, len);
        if (n < 0) return -1;
#endif
        data += n;
        len -= (size_t)n;
    }
    return 0;
}

static int send_str(socket_t fd, const char* s) {
    return send_all(fd, s, strlen(s));
}

// Read a line (up to \n) from socket into buf[0..bufsize-1].
// Returns number of bytes read (including \n) or -1 on error/close.
static int read_line(socket_t fd, char* buf, int bufsize) {
    int total = 0;
    while (total < bufsize - 1) {
#ifdef _WIN32
        char c;
        int n = recv(fd, &c, 1, 0);
        if (n <= 0) return -1;
#else
        char c;
        ssize_t n = read(fd, &c, 1);
        if (n <= 0) return -1;
#endif
        buf[total++] = c;
        if (c == '\n') break;
    }
    buf[total] = '\0';
    return total;
}

// Read exactly n bytes. Returns 0 on success, -1 on error.
static int read_exact(socket_t fd, char* buf, size_t n) {
    while (n > 0) {
#ifdef _WIN32
        int r = recv(fd, buf, (int)n, 0);
        if (r <= 0) return -1;
#else
        ssize_t r = read(fd, buf, n);
        if (r <= 0) return -1;
#endif
        buf += r;
        n -= (size_t)r;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// JSON response construction helpers (dynamic string builder)
// ---------------------------------------------------------------------------
typedef struct {
    char* data;
    size_t len;
    size_t cap;
} string_buf_t;

static void sb_init(string_buf_t* sb) {
    sb->cap = 4096;
    sb->len = 0;
    sb->data = (char*)malloc(sb->cap);
    if (sb->data) sb->data[0] = '\0';
}

static void sb_free(string_buf_t* sb) {
    free(sb->data);
    sb->data = NULL;
}

static void sb_grow(string_buf_t* sb, size_t need) {
    if (sb->len + need < sb->cap) return;
    while (sb->len + need >= sb->cap)
        sb->cap *= 2;
    sb->data = (char*)realloc(sb->data, sb->cap);
}

static void sb_puts(string_buf_t* sb, const char* s) {
    size_t n = strlen(s);
    sb_grow(sb, n + 1);
    memcpy(sb->data + sb->len, s, n + 1);
    sb->len += n;
}

static void sb_putf(string_buf_t* sb, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    sb_grow(sb, (size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(sb->data + sb->len, (size_t)n + 1, fmt, ap);
    va_end(ap);
    sb->len += (size_t)n;
}

// JSON-escape a string and append it in quotes.
static void sb_put_json_str(string_buf_t* sb, const char* s) {
    sb_grow(sb, strlen(s) * 2 + 3);
    sb->data[sb->len++] = '"';
    for (const char* p = s; *p; p++) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
        case '"':
            sb->data[sb->len++] = '\\';
            sb->data[sb->len++] = '"';
            break;
        case '\\':
            sb->data[sb->len++] = '\\';
            sb->data[sb->len++] = '\\';
            break;
        case '\n':
            sb->data[sb->len++] = '\\';
            sb->data[sb->len++] = 'n';
            break;
        case '\r':
            sb->data[sb->len++] = '\\';
            sb->data[sb->len++] = 'r';
            break;
        case '\t':
            sb->data[sb->len++] = '\\';
            sb->data[sb->len++] = 't';
            break;
        default:
            if (c < 0x20) {
                sb_grow(sb, 7);
                sb->len += (size_t)snprintf(sb->data + sb->len, 7, "\\u%04x", c);
            } else {
                sb->data[sb->len++] = c;
            }
            break;
        }
    }
    sb->data[sb->len++] = '"';
    sb->data[sb->len] = '\0';
}

// ---------------------------------------------------------------------------
// HTTP request structure
// ---------------------------------------------------------------------------
typedef struct {
    char method[16];
    char path[1024];
    char body[65536];
    size_t body_len;
} http_request_t;

// Parse a single HTTP request from the socket.
// Returns 0 on success, -1 on error/close.
static int parse_http_request(socket_t fd, http_request_t* req) {
    memset(req, 0, sizeof(*req));

    // Request line
    char line[2048];
    if (read_line(fd, line, sizeof(line)) <= 0) return -1;

    // Parse method + path
    char ver[64];
    if (sscanf(line, "%15s %1023s %63s", req->method, req->path, ver) < 2) return -1;

    // Headers
    size_t content_length = 0;
    while (1) {
        if (read_line(fd, line, sizeof(line)) <= 0) return -1;
        if (line[0] == '\r' || line[0] == '\n') break; // end of headers

        // Check for content-length
        if (strncasecmp(line, "content-length:", 15) == 0) {
            content_length = (size_t)atol(line + 15);
        }
    }

    // Read body if present
    if (content_length > 0) {
        if (content_length >= sizeof(req->body)) content_length = sizeof(req->body) - 1;
        if (read_exact(fd, req->body, content_length) < 0) return -1;
        req->body[content_length] = '\0';
        req->body_len = content_length;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Chat history builder (adapts engine.cu logic for API messages)
// ---------------------------------------------------------------------------
typedef struct {
    int* ids;
    int len;
} token_array_t;

static token_array_t tokenize_text(nanochat_tokenizer_t* tok, const char* text) {
    token_array_t r = {NULL, 0};
    if (!text || !*text) return r;
    size_t text_len = strlen(text);
    r.ids = nanochat_tokenizer_encode(tok, text, text_len, &r.len);
    return r;
}

// Build prompt token array from an array of message strings.
//   messages_layout[n][0] = role ('u' for user, 'a' for assistant)
//   messages_layout[n][1] = content string
// The final block appends ASSISTANT_START (model will fill).
static int* build_prompt_tokens(nanochat_tokenizer_t* tok, const char** roles,
                                const char** contents, int num_messages, int* out_len) {
    // First pass: count tokens
    int total = 0;
    for (int i = 0; i < num_messages; i++) {
        if (roles[i][0] == 'u' || roles[i][0] == 's') {
            total += 1; // USER_START
            token_array_t ta = tokenize_text(tok, contents[i]);
            total += ta.len;
            total += 1; // USER_END
            free(ta.ids);
        } else if (roles[i][0] == 'a') {
            total += 1; // ASSISTANT_START
            token_array_t ta = tokenize_text(tok, contents[i]);
            total += ta.len;
            total += 1; // ASSISTANT_END
            free(ta.ids);
        }
    }
    total += 1; // ASSISTANT_START for model response
    if (total <= 0) {
        *out_len = 0;
        return NULL;
    }

    int* tokens = (int*)malloc((size_t)total * sizeof(int));
    int pos = 0;

    for (int i = 0; i < num_messages; i++) {
        if (roles[i][0] == 'u' || roles[i][0] == 's') {
            tokens[pos++] = NANOCHAT_TOKEN_USER_START;
            token_array_t ta = tokenize_text(tok, contents[i]);
            if (ta.ids && ta.len > 0) {
                memcpy(tokens + pos, ta.ids, (size_t)ta.len * sizeof(int));
                pos += ta.len;
            }
            free(ta.ids);
            tokens[pos++] = NANOCHAT_TOKEN_USER_END;
        } else if (roles[i][0] == 'a') {
            tokens[pos++] = NANOCHAT_TOKEN_ASSISTANT_START;
            token_array_t ta = tokenize_text(tok, contents[i]);
            if (ta.ids && ta.len > 0) {
                memcpy(tokens + pos, ta.ids, (size_t)ta.len * sizeof(int));
                pos += ta.len;
            }
            free(ta.ids);
            tokens[pos++] = NANOCHAT_TOKEN_ASSISTANT_END;
        }
    }
    tokens[pos++] = NANOCHAT_TOKEN_ASSISTANT_START;

    *out_len = pos;
    return tokens;
}

// ---------------------------------------------------------------------------
// Streaming callback: sends SSE data chunks
// ---------------------------------------------------------------------------
typedef struct {
    socket_t fd;
    int conn_id;
    int cancelled;
} stream_ctx_t;

static void stream_callback(const char* text, void* user_data) {
    stream_ctx_t* ctx = (stream_ctx_t*)user_data;
    if (ctx->cancelled) return;

    // Build SSE data frame: {"id":"...","object":"chat.completion.chunk",...}
    char buf[8192];
    int n = snprintf(buf, sizeof(buf),
                     "data: {\"id\":\"chatcmpl-%d\",\"object\":\"chat.completion.chunk\","
                     "\"created\":%ld,\"model\":\"nanochat\","
                     "\"choices\":[{\"index\":0,\"delta\":{\"content\":",
                     ctx->conn_id, (long)time(NULL));

    // Append JSON-escaped text
    n += snprintf(buf + n, (n < (int)sizeof(buf)) ? (size_t)((int)sizeof(buf) - n) : 0, "\"");
    for (const char* p = text; *p && n < (int)sizeof(buf) - 20; p++) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
        case '"':
            buf[n++] = '\\';
            buf[n++] = '"';
            break;
        case '\\':
            buf[n++] = '\\';
            buf[n++] = '\\';
            break;
        case '\n':
            buf[n++] = '\\';
            buf[n++] = 'n';
            break;
        case '\r':
            buf[n++] = '\\';
            buf[n++] = 'r';
            break;
        case '\t':
            buf[n++] = '\\';
            buf[n++] = 't';
            break;
        default:
            if (c < 0x20) {
                n += snprintf(buf + n, (size_t)((int)sizeof(buf) - n), "\\u%04x", c);
            } else {
                buf[n++] = c;
            }
            break;
        }
    }
    int remaining = (int)sizeof(buf) - n;
    if (remaining > 5) {
        n += snprintf(buf + n, (size_t)remaining, "},\"finish_reason\":null}],\"usage\":null}\n\n");
    }

    if (send_all(ctx->fd, buf, (size_t)n) < 0) {
        ctx->cancelled = 1;
    }
}

// ---------------------------------------------------------------------------
// Handler for /v1/chat/completions
// ---------------------------------------------------------------------------
static void handle_chat_completions(socket_t fd, const http_request_t* req, nanochat_engine_t* eng,
                                    int conn_id) {
    int max_tokens = g_max_tokens;
    float temperature = g_default_temp;
    int top_k = g_default_top_k;
    int streaming = 0;
    int num_messages = 0;
    const char* roles[64];
    const char* contents[64];

    // Parse JSON body
    json_ctx_t jctx;
    json_init(&jctx, req->body, req->body_len);

    // Expect object
    json_skip_ws(&jctx);
    if (json_next(&jctx) != '{') {
        send_str(
            fd,
            "HTTP/1.1 400 Bad Request\r\nContent-Length: 27\r\n\r\n{\"error\":\"invalid JSON\"}");
        return;
    }

    while (1) {
        json_skip_ws(&jctx);
        char c = (jctx.pos < jctx.len) ? jctx.data[jctx.pos] : '\0';
        if (c == '}' || c == '\0') break;

        char* key = json_parse_string(&jctx);
        if (!key) break;
        json_skip_ws(&jctx);
        if (jctx.pos < jctx.len && jctx.data[jctx.pos] == ':') jctx.pos++;

        if (strcmp(key, "messages") == 0) {
            json_skip_ws(&jctx);
            if (json_next(&jctx) == '[') {
                // Parse message array
                while (1) {
                    json_skip_ws(&jctx);
                    if (jctx.pos >= jctx.len) break;
                    if (jctx.data[jctx.pos] == ']') {
                        jctx.pos++;
                        break;
                    }
                    if (jctx.data[jctx.pos] == ',') {
                        jctx.pos++;
                        continue;
                    }

                    if (jctx.data[jctx.pos] == '{') {
                        json_next(&jctx); // skip {

                        char* role = NULL;
                        char* content = NULL;
                        while (1) {
                            json_skip_ws(&jctx);
                            if (jctx.pos >= jctx.len) break;
                            if (jctx.data[jctx.pos] == '}') {
                                jctx.pos++;
                                break;
                            }

                            char* mk = json_parse_string(&jctx);
                            if (!mk) break;
                            json_skip_ws(&jctx);
                            if (jctx.pos < jctx.len && jctx.data[jctx.pos] == ':') jctx.pos++;

                            if (strcmp(mk, "role") == 0) {
                                role = json_parse_string(&jctx);
                            } else if (strcmp(mk, "content") == 0) {
                                content = json_parse_string(&jctx);
                            } else {
                                json_skip_value(&jctx);
                            }
                            free(mk);
                            json_skip_ws(&jctx);
                            if (jctx.pos < jctx.len && jctx.data[jctx.pos] == ',') jctx.pos++;
                        }

                        if (role && content && num_messages < 64) {
                            roles[num_messages] = role;
                            contents[num_messages] = content;
                            num_messages++;
                        } else {
                            free(role);
                            free(content);
                        }
                    } else
                        break;
                }
            }
        } else if (strcmp(key, "max_tokens") == 0) {
            max_tokens = (int)json_parse_int(&jctx);
            if (max_tokens <= 0) max_tokens = g_max_tokens;
            if (max_tokens > 1024) max_tokens = 1024;
        } else if (strcmp(key, "temperature") == 0) {
            temperature = (float)json_parse_number(&jctx);
            if (temperature < 0.0f) temperature = 0.0f;
        } else if (strcmp(key, "top_k") == 0) {
            top_k = (int)json_parse_int(&jctx);
            if (top_k <= 0) top_k = 1;
        } else if (strcmp(key, "stream") == 0) {
            json_skip_ws(&jctx);
            if (jctx.pos + 4 <= jctx.len && (strncmp(jctx.data + jctx.pos, "true", 4) == 0)) {
                streaming = 1;
                jctx.pos += 4;
            } else {
                jctx.pos += 5; // skip "false"
            }
        } else {
            json_skip_value(&jctx);
        }

        free(key);
        if (jctx.pos >= jctx.len) break;
        json_skip_ws(&jctx);
        if (jctx.pos < jctx.len && jctx.data[jctx.pos] == ',') jctx.pos++;
    }

    // Validate messages
    if (num_messages == 0) {
        send_str(fd, "HTTP/1.1 400 Bad Request\r\nContent-Length: 32\r\n\r\n{\"error\":\"no "
                     "messages provided\"}");
        return;
    }

    // Build prompt tokens from messages
    int prompt_len;
    int* prompt_tokens =
        build_prompt_tokens(eng->tokenizer, roles, contents, num_messages, &prompt_len);

    // Clean up parsed message strings
    for (int i = 0; i < num_messages; i++) {
        free((char*)roles[i]);
        free((char*)contents[i]);
    }

    if (!prompt_tokens || prompt_len == 0) {
        send_str(
            fd,
            "HTTP/1.1 400 Bad Request\r\nContent-Length: 27\r\n\r\n{\"error\":\"empty prompt\"}");
        return;
    }

    // Generate response ID
    char id_str[64];
    snprintf(id_str, sizeof(id_str), "chatcmpl-%d", conn_id);
    long created = (long)time(NULL);

    if (streaming) {
        // ---- SSE streaming ----
        const char* sse_headers = "HTTP/1.1 200 OK\r\n"
                                  "Content-Type: text/event-stream\r\n"
                                  "Cache-Control: no-cache\r\n"
                                  "Connection: keep-alive\r\n"
                                  "Access-Control-Allow-Origin: *\r\n"
                                  "\r\n";
        if (send_str(fd, sse_headers) < 0) {
            free(prompt_tokens);
            return;
        }

        // Send role chunk
        char role_chunk[1024];
        snprintf(role_chunk, sizeof(role_chunk),
                 "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                 "\"created\":%ld,\"model\":\"nanochat\","
                 "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":"
                 "null}]}\n\n",
                 id_str, created);
        if (send_str(fd, role_chunk) < 0) {
            free(prompt_tokens);
            return;
        }

        // Generate with streaming callback
        stream_ctx_t sctx;
        sctx.fd = fd;
        sctx.conn_id = conn_id;
        sctx.cancelled = 0;

        char* full = nanochat_generate_from_tokens(eng, prompt_tokens, prompt_len, max_tokens,
                                                   temperature, top_k, stream_callback, &sctx);
        free(full);

        // Send finish chunk with token usage if we have it
        // (We don't track token counts at this level, so usage is partial)
        char finish_chunk[1024];
        snprintf(finish_chunk, sizeof(finish_chunk),
                 "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                 "\"created\":%ld,\"model\":\"nanochat\","
                 "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                 id_str, created);
        send_str(fd, finish_chunk);

        send_str(fd, "data: [DONE]\n\n");
    } else {
        // ---- Non-streaming: generate with token API directly ----
        char* output = nanochat_generate_from_tokens(eng, prompt_tokens, prompt_len, max_tokens,
                                                     temperature, top_k, NULL, NULL);
        prompt_tokens = NULL; // consumed by generate_from_tokens

        if (!output) {
            send_str(fd, "HTTP/1.1 500 Internal Server Error\r\nContent-Length: "
                         "29\r\n\r\n{\"error\":\"generation failed\"}");
            return;
        }

        // Build JSON response
        string_buf_t sb;
        sb_init(&sb);
        sb_putf(&sb,
                "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,"
                "\"model\":\"nanochat\",\"choices\":[{"
                "\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":",
                id_str, created);
        sb_put_json_str(&sb, output);
        sb_puts(&sb, "},\"finish_reason\":\"stop\"}],");
        sb_puts(&sb, "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}");

        char header[256];
        int hlen = snprintf(header, sizeof(header),
                            "HTTP/1.1 200 OK\r\n"
                            "Content-Type: application/json\r\n"
                            "Content-Length: %zu\r\n"
                            "Access-Control-Allow-Origin: *\r\n"
                            "\r\n",
                            sb.len);
        send_all(fd, header, (size_t)hlen);
        send_str(fd, sb.data);

        sb_free(&sb);
        free(output);
    }
}

// ---------------------------------------------------------------------------
// Handler for /v1/models
// ---------------------------------------------------------------------------
static void handle_list_models(socket_t fd) {
    const char* body = "{\"object\":\"list\",\"data\":["
                       "{\"id\":\"nanochat\",\"object\":\"model\","
                       "\"created\":0,\"owned_by\":\"boat\"}"
                       "]}";
    size_t blen = strlen(body);
    char header[256];
    int hlen = snprintf(header, sizeof(header),
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: application/json\r\n"
                        "Content-Length: %zu\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "\r\n",
                        blen);
    send_all(fd, header, (size_t)hlen);
    send_str(fd, body);
}

// ---------------------------------------------------------------------------
// Handle one HTTP connection
// ---------------------------------------------------------------------------
static void handle_connection(socket_t fd, nanochat_engine_t* eng, int conn_id) {
    http_request_t req;
    if (parse_http_request(fd, &req) < 0) return;

    // CORS preflight
    if (strcmp(req.method, "OPTIONS") == 0) {
        send_str(fd, "HTTP/1.1 204 No Content\r\n"
                     "Access-Control-Allow-Origin: *\r\n"
                     "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                     "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                     "Access-Control-Max-Age: 86400\r\n"
                     "\r\n");
        return;
    }

    if (strcmp(req.path, "/v1/chat/completions") == 0 && strcmp(req.method, "POST") == 0) {
        handle_chat_completions(fd, &req, eng, conn_id);
    } else if (strcmp(req.path, "/v1/models") == 0 && strcmp(req.method, "GET") == 0) {
        handle_list_models(fd);
    } else if (strcmp(req.path, "/health") == 0 || strcmp(req.path, "/") == 0) {
        send_str(fd, "HTTP/1.1 200 OK\r\n"
                     "Content-Type: application/json\r\n"
                     "Content-Length: 31\r\n"
                     "\r\n"
                     "{\"status\":\"ok\",\"model\":\"nanochat\"}");
    } else {
        send_str(fd, "HTTP/1.1 404 Not Found\r\n"
                     "Content-Type: application/json\r\n"
                     "Content-Length: 22\r\n"
                     "\r\n"
                     "{\"error\":\"not found\"}");
    }
}

// ---------------------------------------------------------------------------
// Main server loop
// ---------------------------------------------------------------------------
int nanochat_start_server(const char* model_dir, const char* host, int port) {
    if (platform_init() < 0) {
        fprintf(stderr, "[Server] Failed to initialize platform sockets\n");
        return -1;
    }

    // Create engine
    fprintf(stderr, "[Server] Loading NanoChat engine from: %s\n", model_dir);
    nanochat_engine_t* eng = nanochat_engine_create(model_dir);
    if (!eng) {
        fprintf(stderr, "[Server] Failed to create engine\n");
        platform_cleanup();
        return -1;
    }
    fprintf(stderr, "[Server] Engine loaded\n");

    // Create socket
    socket_t listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd == INVALID_SOCKET_VAL) {
        fprintf(stderr, "[Server] Failed to create socket\n");
        nanochat_engine_free(eng);
        platform_cleanup();
        return -1;
    }

    // Allow reuse
#ifdef _WIN32
    BOOL reuse = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
#else
    int reuse = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#endif

    // Bind
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);

    if (!host || strcmp(host, "0.0.0.0") == 0 || strcmp(host, "*") == 0) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        addr.sin_addr.s_addr = inet_addr(host);
        if (addr.sin_addr.s_addr == (unsigned)-1) {
            fprintf(stderr, "[Server] Invalid host: %s\n", host);
            SOCKET_CLOSE(listen_fd);
            nanochat_engine_free(eng);
            platform_cleanup();
            return -1;
        }
    }

    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[Server] Failed to bind to %s:%d\n", host ? host : "0.0.0.0", port);
        SOCKET_CLOSE(listen_fd);
        nanochat_engine_free(eng);
        platform_cleanup();
        return -1;
    }

    if (listen(listen_fd, 8) < 0) {
        fprintf(stderr, "[Server] Failed to listen\n");
        SOCKET_CLOSE(listen_fd);
        nanochat_engine_free(eng);
        platform_cleanup();
        return -1;
    }

    fprintf(stderr, "[Server] Listening on http://%s:%d\n", host ? host : "0.0.0.0", port);
    fprintf(stderr, "[Server] OpenAI-compatible API:\n");
    fprintf(stderr, "[Server]   POST /v1/chat/completions\n");
    fprintf(stderr, "[Server]   GET  /v1/models\n");
    fprintf(stderr, "[Server]   GET  /health\n");
    fprintf(stderr, "\n");

    int conn_id = 0;

    while (1) {
        struct sockaddr_in client_addr;
#ifdef _WIN32
        int addrlen = sizeof(client_addr);
#else
        socklen_t addrlen = sizeof(client_addr);
#endif
        socket_t client_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &addrlen);
        if (client_fd == INVALID_SOCKET_VAL) {
            fprintf(stderr, "[Server] Accept failed\n");
            continue;
        }

        // Disable Nagle for responsive streaming
#ifdef _WIN32
        BOOL nodelay = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof(nodelay));
#else
        int nodelay = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
#endif

        conn_id++;

        // Handle request
        handle_connection(client_fd, eng, conn_id);

        // Close connection
        shutdown(client_fd, SHUT_RDWR);
        SOCKET_CLOSE(client_fd);
    }

    // Cleanup (unreachable in normal operation)
    SOCKET_CLOSE(listen_fd);
    nanochat_engine_free(eng);
    platform_cleanup();
    return 0;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    const char* model_dir = NULL;
    const char* host = "127.0.0.1";
    int port = 8080;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_dir = argv[++i];
        else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc)
            host = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            nanochat_server_set_max_tokens(atoi(argv[++i]));
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc)
            nanochat_server_set_default_temperature((float)atof(argv[++i]));
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc)
            nanochat_server_set_default_top_k(atoi(argv[++i]));
        else if (!model_dir)
            model_dir = argv[i];
    }

    if (!model_dir) {
        fprintf(stderr, "Usage: nanochat_server <model_dir> [--host HOST] [--port PORT] "
                        "[--max-tokens N] [--temperature T] [--top-k K]\n");
        return 1;
    }

    fprintf(stderr, "[Server] Starting on %s:%d with model %s\n", host, port, model_dir);
    return nanochat_start_server(model_dir, host, port);
}
