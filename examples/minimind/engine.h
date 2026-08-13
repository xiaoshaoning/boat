// engine.h - MiniMind inference engine
#pragma once
#include "model.h"

// Generate response tokens from a prompt string.
// Returns malloc'd string of generated tokens (caller frees), or NULL on error.
// max_tokens: max new tokens to generate
// temperature: 0.0 = greedy, >0 = sampling (typical: 0.85)
// top_k: 0 = no filtering, >0 = keep top k (typical: 50)
char* minimind_generate(minimind_model_t* m, const char* prompt,
                        int max_tokens, float temperature, int top_k);

// Get the chat template formatted prompt.
// Returns malloc'd string (caller frees).
char* minimind_format_chat_prompt(const char* user_input);
