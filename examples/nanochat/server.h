// server.h - NanoChat LLM Serving API (OpenAI-compatible HTTP server)
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Start the HTTP server. Blocks the calling thread indefinitely.
// Returns 0 on clean shutdown, -1 on fatal error.
int nanochat_start_server(const char* model_dir, const char* host, int port);

// Set server options before calling nanochat_start_server (optional).
void nanochat_server_set_max_tokens(int max_tokens);
void nanochat_server_set_default_temperature(float temp);
void nanochat_server_set_default_top_k(int top_k);

#ifdef __cplusplus
}
#endif
