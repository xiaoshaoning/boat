// huggingface.c - Hugging Face Transformers model format loader
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/format/huggingface.h>
#include <boat/model.h>
#include <boat/layers.h>
#include <boat/layers/norm.h>
#include <boat/memory.h>
#include <boat/tensor.h>
#include <boat/graph.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

// JSON parsing with cJSON
#ifdef BOAT_USE_CJSON
#include <cjson/cJSON.h>
#else
// Minimal JSON parsing helpers when cJSON is not available
typedef void cJSON; // Dummy type
#endif

// Forward declarations for internal structures
typedef struct hf_layer_builder_t hf_layer_builder_t;

// Internal structure for Hugging Face model configuration
typedef struct {
    char* model_type;               // e.g., "bert", "gpt2", "roberta"
    int hidden_size;                // hidden size
    int num_hidden_layers;          // number of layers
    int num_attention_heads;        // attention heads
    int intermediate_size;          // feed-forward intermediate size
    int max_position_embeddings;    // max sequence length
    int vocab_size;                 // vocabulary size
    float layer_norm_eps;           // LayerNorm epsilon

    // Layer builder tracking for associating weights and biases
    hf_layer_builder_t* builders;    // Array of layer builders
    size_t builder_count;           // Number of builders
    size_t builder_capacity;        // Capacity of builders array
} hf_config_t;

// Simple layer wrapper for Hugging Face model loading
typedef struct {
    boat_tensor_t* weight;          // Weight tensor (owned by layer)
    boat_tensor_t* bias;            // Bias tensor (optional, owned by layer)
    char* layer_type;               // Layer type identifier
    void* layer_data;               // Pointer to actual layer implementation (future)
} hf_layer_wrapper_t;

// Forward declarations for static functions
static void free_config(const hf_config_t* config);
static void init_builders(const hf_config_t* config);
static boat_layer_t* create_layer_from_config(const hf_config_t* config, const char* layer_name, boat_tensor_t* weight);
static hf_layer_wrapper_t* create_layer_wrapper(const char* layer_type, boat_tensor_t* weight);
static void free_layer_wrapper(hf_layer_wrapper_t* wrapper);
static char* get_base_layer_name(const char* tensor_name);
static boat_layer_t* create_actual_layer_from_tensor(const char* base_name, const char* tensor_name, boat_tensor_t* tensor);
static char* read_file_to_string(const char* filename);

// Layer operations for dense layers
static boat_tensor_t* dense_layer_forward(const boat_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !layer->data || !input) return NULL;
    const boat_dense_layer_t* dense_layer = (const boat_dense_layer_t*)layer->data;
    return boat_dense_layer_forward(dense_layer, input);
}

static boat_tensor_t* dense_layer_backward(const boat_layer_t* layer, const boat_tensor_t* grad_output) {
    if (!layer || !layer->data || !grad_output) return NULL;
    const boat_dense_layer_t* dense_layer = (const boat_dense_layer_t*)layer->data;
    return boat_dense_layer_backward(dense_layer, grad_output);
}

static void dense_layer_update(const boat_layer_t* layer, float learning_rate) {
    if (!layer || !layer->data) return;
    const boat_dense_layer_t* dense_layer = (const boat_dense_layer_t*)layer->data;
    boat_dense_layer_update(dense_layer, learning_rate);
}

static void dense_layer_free(const boat_layer_t* layer) {
    if (!layer || !layer->data) return;

    const boat_dense_layer_t* dense_layer = (const boat_dense_layer_t*)layer->data;
    boat_dense_layer_free(dense_layer);

    // Free the layer wrapper itself
    free(layer);
}

static const boat_layer_ops_t dense_layer_ops = {
    .forward = dense_layer_forward,
    .backward = dense_layer_backward,
    .update = dense_layer_update,
    .free = dense_layer_free
};

// Layer operations for layer normalization
static boat_tensor_t* layernorm_layer_forward(const boat_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !layer->data || !input) return NULL;
    boat_layernorm_t* layernorm = (boat_layernorm_t*)layer->data;
    return boat_layernorm_forward(layernorm, input);
}

static boat_tensor_t* layernorm_layer_backward(const boat_layer_t* layer, const boat_tensor_t* grad_output) {
    if (!layer || !layer->data || !grad_output) return NULL;
    boat_layernorm_t* layernorm = (boat_layernorm_t*)layer->data;
    return boat_layernorm_backward(layernorm, grad_output);
}

static void layernorm_layer_update(const boat_layer_t* layer, float learning_rate) {
    if (!layer || !layer->data) return;
    boat_layernorm_t* layernorm = (boat_layernorm_t*)layer->data;
    boat_layernorm_update(layernorm, learning_rate);
}

static void layernorm_layer_free(const boat_layer_t* layer) {
    if (!layer || !layer->data) return;

    boat_layernorm_t* layernorm = (boat_layernorm_t*)layer->data;
    boat_layernorm_free(layernorm);

    // Free the layer wrapper itself
    free(layer);
}

static const boat_layer_ops_t layernorm_layer_ops = {
    .forward = layernorm_layer_forward,
    .backward = layernorm_layer_backward,
    .update = layernorm_layer_update,
    .free = layernorm_layer_free
};

// Layer operations for wrapper layers (layer_norm, conv, etc.)
static boat_tensor_t* wrapper_layer_forward(const boat_layer_t* layer, const boat_tensor_t* input) {
    (void)layer;
    (void)input;
    // TODO: Implement forward pass for wrapper layers
    return NULL;
}

static boat_tensor_t* wrapper_layer_backward(const boat_layer_t* layer, const boat_tensor_t* grad_output) {
    (void)layer;
    (void)grad_output;
    // TODO: Implement backward pass for wrapper layers
    return NULL;
}

static void wrapper_layer_update(const boat_layer_t* layer, float learning_rate) {
    (void)layer;
    (void)learning_rate;
    // TODO: Implement parameter update for wrapper layers
}

static void wrapper_layer_free(const boat_layer_t* layer) {
    if (!layer || !layer->data) return;

    hf_layer_wrapper_t* wrapper = (hf_layer_wrapper_t*)layer->data;
    free_layer_wrapper(wrapper);

    // Free the layer wrapper itself
    free(layer);
}

static const boat_layer_ops_t wrapper_layer_ops = {
    .forward = wrapper_layer_forward,
    .backward = wrapper_layer_backward,
    .update = wrapper_layer_update,
    .free = wrapper_layer_free
};

// Structure to track layer creation during loading
typedef struct hf_layer_builder_t {
    char* base_name;                // Base name without suffix (e.g., "lin" for "lin.weight")
    void* layer;                    // Pointer to actual Boat layer (boat_dense_layer_t, etc.)
    char* layer_type;               // Layer type ("dense", "layer_norm", etc.)
    bool has_weight;                // Whether weight tensor has been set
    bool has_bias;                  // Whether bias tensor has been set
    boat_tensor_t* weight_tensor;   // Temporary storage for weight tensor
    boat_tensor_t* bias_tensor;     // Temporary storage for bias tensor
} hf_layer_builder_t;

// Safetensors format constants
#define SAFETENSORS_MAGIC 0x7473  // "st" in little endian
#define SAFETENSORS_VERSION 1

// Safetensors data type mapping
typedef enum {
    SAFETENSORS_DTYPE_F64 = 0,
    SAFETENSORS_DTYPE_F32,
    SAFETENSORS_DTYPE_F16,
    SAFETENSORS_DTYPE_BF16,
    SAFETENSORS_DTYPE_I64,
    SAFETENSORS_DTYPE_I32,
    SAFETENSORS_DTYPE_I16,
    SAFETENSORS_DTYPE_I8,
    SAFETENSORS_DTYPE_U64,
    SAFETENSORS_DTYPE_U32,
    SAFETENSORS_DTYPE_U16,
    SAFETENSORS_DTYPE_U8,
    SAFETENSORS_DTYPE_BOOL,
    SAFETENSORS_DTYPE_UNKNOWN
} safetensors_dtype_t;


// Parse JSON configuration into hf_config_t
static hf_config_t* parse_config(const char* config_json) {
    if (!config_json) return NULL;

    hf_config_t* config = calloc(1, sizeof(hf_config_t));
    if (!config) return NULL;

    // Initialize builders array
    init_builders(config);

    // Initialize with default values
    config->hidden_size = 768;
    config->num_hidden_layers = 12;
    config->num_attention_heads = 12;
    config->intermediate_size = 3072;
    config->max_position_embeddings = 512;
    config->vocab_size = 30522;
    config->layer_norm_eps = 1e-12f;

#ifdef BOAT_USE_CJSON
    // Parse JSON using cJSON
    cJSON* root = cJSON_Parse(config_json);
    if (!root) {
        fprintf(stderr, "Failed to parse JSON configuration\n");
        free_config(config);
        return NULL;
    }

    // Extract model_type
    cJSON* model_type = cJSON_GetObjectItem(root, "model_type");
    if (model_type && cJSON_IsString(model_type)) {
        config->model_type = strdup(model_type->valuestring);
    } else {
        config->model_type = strdup("unknown");
    }

    // Extract architecture parameters
    cJSON* hidden_size = cJSON_GetObjectItem(root, "hidden_size");
    if (hidden_size && cJSON_IsNumber(hidden_size)) {
        config->hidden_size = hidden_size->valueint;
    }

    cJSON* num_hidden_layers = cJSON_GetObjectItem(root, "num_hidden_layers");
    if (num_hidden_layers && cJSON_IsNumber(num_hidden_layers)) {
        config->num_hidden_layers = num_hidden_layers->valueint;
    }

    cJSON* num_attention_heads = cJSON_GetObjectItem(root, "num_attention_heads");
    if (num_attention_heads && cJSON_IsNumber(num_attention_heads)) {
        config->num_attention_heads = num_attention_heads->valueint;
    }

    cJSON* intermediate_size = cJSON_GetObjectItem(root, "intermediate_size");
    if (intermediate_size && cJSON_IsNumber(intermediate_size)) {
        config->intermediate_size = intermediate_size->valueint;
    }

    cJSON* max_position_embeddings = cJSON_GetObjectItem(root, "max_position_embeddings");
    if (max_position_embeddings && cJSON_IsNumber(max_position_embeddings)) {
        config->max_position_embeddings = max_position_embeddings->valueint;
    }

    cJSON* vocab_size = cJSON_GetObjectItem(root, "vocab_size");
    if (vocab_size && cJSON_IsNumber(vocab_size)) {
        config->vocab_size = vocab_size->valueint;
    }

    cJSON* layer_norm_eps = cJSON_GetObjectItem(root, "layer_norm_eps");
    if (layer_norm_eps && cJSON_IsNumber(layer_norm_eps)) {
        config->layer_norm_eps = (float)layer_norm_eps->valuedouble;
    }

    cJSON_Delete(root);
    return config;
#else
    // Without cJSON, use default values and try to parse simple JSON
    // This is a very basic parser that only looks for specific patterns
    fprintf(stderr, "cJSON not available, using default configuration\n");

    // Simple string extraction for model_type
    const char* model_type_ptr = strstr(config_json, "\"model_type\"");
    if (model_type_ptr) {
        const char* colon = strstr(model_type_ptr, ":");
        if (colon) {
            const char* quote = strstr(colon, "\"");
            if (quote) {
                const char* end_quote = strstr(quote + 1, "\"");
                if (end_quote) {
                    size_t len = end_quote - (quote + 1);
                    config->model_type = malloc(len + 1);
                    if (config->model_type) {
                        strncpy(config->model_type, quote + 1, len);
                        config->model_type[len] = '\0';
                    }
                }
            }
        }
    }

    if (!config->model_type) {
        config->model_type = strdup("unknown");
    }

    return config;
#endif
}

// Layer builder management functions

// Initialize builders array
static void init_builders(const hf_config_t* config) {
    if (!config) return;
    config->builder_capacity = 16;
    config->builder_count = 0;
    config->builders = calloc(config->builder_capacity, sizeof(hf_layer_builder_t));
    if (!config->builders) {
        config->builder_capacity = 0;
    }
}

// Free builders array
static void free_builders(const hf_config_t* config) {
    if (!config || !config->builders) return;

    for (size_t i = 0; i < config->builder_count; i++) {
        hf_layer_builder_t* builder = &config->builders[i];
        free(builder->base_name);
        free(builder->layer_type);
        if (builder->weight_tensor) {
            boat_tensor_unref(builder->weight_tensor);
        }
        if (builder->bias_tensor) {
            boat_tensor_unref(builder->bias_tensor);
        }
    }
    free(config->builders);
    config->builders = NULL;
    config->builder_count = 0;
    config->builder_capacity = 0;
}

// Find or create a layer builder by base name
static hf_layer_builder_t* find_or_create_builder(const hf_config_t* config, const char* base_name, const char* layer_type) {
    if (!config || !base_name || !layer_type) return NULL;

    // First, try to find existing builder
    for (size_t i = 0; i < config->builder_count; i++) {
        hf_layer_builder_t* builder = &config->builders[i];
        if (builder->base_name && strcmp(builder->base_name, base_name) == 0) {
            return builder;
        }
    }

    // Not found, create new builder
    if (config->builder_count >= config->builder_capacity) {
        // Expand array
        size_t new_capacity = config->builder_capacity * 2;
        if (new_capacity < 16) new_capacity = 16;
        hf_layer_builder_t* new_builders = realloc(config->builders, new_capacity * sizeof(hf_layer_builder_t));
        if (!new_builders) return NULL;
        config->builders = new_builders;
        config->builder_capacity = new_capacity;

        // Initialize new entries
        for (size_t i = config->builder_count; i < new_capacity; i++) {
            memset(&config->builders[i], 0, sizeof(hf_layer_builder_t));
        }
    }

    hf_layer_builder_t* builder = &config->builders[config->builder_count];
    builder->base_name = strdup(base_name);
    builder->layer_type = strdup(layer_type);
    builder->has_weight = false;
    builder->has_bias = false;
    builder->weight_tensor = NULL;
    builder->bias_tensor = NULL;
    builder->layer = NULL;

    if (!builder->base_name || !builder->layer_type) {
        free(builder->base_name);
        free(builder->layer_type);
        return NULL;
    }

    config->builder_count++;
    return builder;
}

// Set weight tensor for a builder
static bool set_builder_weight(hf_layer_builder_t* builder, boat_tensor_t* weight) {
    if (!builder || !weight) return false;

    if (builder->weight_tensor) {
        boat_tensor_unref(builder->weight_tensor);
    }
    builder->weight_tensor = weight;
    boat_tensor_ref(weight); // Take reference
    builder->has_weight = true;
    return true;
}

// Set bias tensor for a builder
static bool set_builder_bias(hf_layer_builder_t* builder, boat_tensor_t* bias) {
    if (!builder || !bias) return false;

    // If layer already exists (created from weight), set bias directly
    if (builder->layer) {
        if (strcmp(builder->layer_type, "dense") == 0) {
            const boat_dense_layer_t* dense_layer = (const boat_dense_layer_t*)builder->layer;
            boat_dense_layer_set_bias(dense_layer, bias);
            printf("    Updated existing dense layer with bias tensor\n");
            return true;
        } else if (strcmp(builder->layer_type, "layer_norm") == 0) {
            boat_layernorm_t* layernorm = (boat_layernorm_t*)builder->layer;
            boat_layernorm_set_bias(layernorm, bias);
            printf("    Updated existing layer normalization layer with bias tensor\n");
            return true;
        }
    }

    // Otherwise, store bias tensor in builder
    if (builder->bias_tensor) {
        boat_tensor_unref(builder->bias_tensor);
    }
    builder->bias_tensor = bias;
    boat_tensor_ref(bias); // Take reference
    builder->has_bias = true;
    return true;
}

// Complete builder and create actual layer
static boat_layer_t* complete_builder(hf_layer_builder_t* builder, const hf_config_t* config) {
    if (!builder || !builder->has_weight) return NULL;

    // Handle different layer types
    if (strcmp(builder->layer_type, "dense") == 0) {
        // Get weight tensor properties
        size_t ndim = boat_tensor_ndim(builder->weight_tensor);
        const int64_t* shape = boat_tensor_shape(builder->weight_tensor);

        if (ndim != 2) {
            return NULL; // Not a weight matrix
        }

        size_t input_features = shape[0];
        size_t output_features = shape[1];

        // Create dense layer with bias flag based on whether we have bias tensor
        boat_dense_layer_t* dense_layer = boat_dense_layer_create(input_features, output_features, builder->has_bias);
        if (!dense_layer) {
            return NULL;
        }

        // Set weight
        boat_dense_layer_set_weight(dense_layer, builder->weight_tensor);

        // Set bias if available
        if (builder->has_bias && builder->bias_tensor) {
            boat_dense_layer_set_bias(dense_layer, builder->bias_tensor);
        }

        // Create layer wrapper
        boat_layer_t* layer = malloc(sizeof(boat_layer_t));
        if (!layer) {
            boat_dense_layer_free(dense_layer);
            return NULL;
        }
        layer->data = dense_layer;
        layer->ops = &dense_layer_ops;

        // Store the actual layer pointer in builder for reference
        builder->layer = dense_layer;

        // Release builder's tensor references since layer now owns them
        if (builder->weight_tensor) {
            boat_tensor_unref(builder->weight_tensor);
            builder->weight_tensor = NULL;
        }
        if (builder->bias_tensor) {
            boat_tensor_unref(builder->bias_tensor);
            builder->bias_tensor = NULL;
        }

        printf("    Created dense layer with dimensions %zu -> %zu (bias: %s)\n",
               input_features, output_features, builder->has_bias ? "yes" : "no");

        return layer;
    } else if (strcmp(builder->layer_type, "layer_norm") == 0) {
        // Layer normalization implementation
        // Get weight tensor properties (should be 1D for layer norm)
        size_t ndim = boat_tensor_ndim(builder->weight_tensor);
        const int64_t* shape = boat_tensor_shape(builder->weight_tensor);

        if (ndim != 1) {
            return NULL; // Not a layer norm weight vector
        }

        size_t normalized_shape = shape[0];

        // Create layer normalization configuration
        boat_layernorm_config_t ln_config = {
            .normalized_shape = normalized_shape,
            .eps = config ? config->layer_norm_eps : 1e-5f,
            .elementwise_affine = true,
            .use_bias = builder->has_bias
        };

        boat_layernorm_t* layernorm = boat_layernorm_create(&ln_config);
        if (!layernorm) {
            return NULL;
        }

        // Set weight
        boat_layernorm_set_weight(layernorm, builder->weight_tensor);

        // Set bias if available
        if (builder->has_bias && builder->bias_tensor) {
            boat_layernorm_set_bias(layernorm, builder->bias_tensor);
        }

        // Create layer wrapper
        boat_layer_t* layer = malloc(sizeof(boat_layer_t));
        if (!layer) {
            boat_layernorm_free(layernorm);
            return NULL;
        }
        layer->data = layernorm;
        layer->ops = &layernorm_layer_ops;

        // Store the actual layer pointer in builder for reference
        builder->layer = layernorm;

        // Release builder's tensor references since layer now owns them
        if (builder->weight_tensor) {
            boat_tensor_unref(builder->weight_tensor);
            builder->weight_tensor = NULL;
        }
        if (builder->bias_tensor) {
            boat_tensor_unref(builder->bias_tensor);
            builder->bias_tensor = NULL;
        }

        printf("    Created layer normalization layer with shape %zu (bias: %s)\n",
               normalized_shape, builder->has_bias ? "yes" : "no");

        return layer;
    } else {
        // Unknown layer type
        return NULL;
    }
}

// Free configuration
static void free_config(const hf_config_t* config) {
    if (!config) return;
    free(config->model_type);
    free_builders(config);
    free(config);
}

// Convert safetensors dtype string to enum
static safetensors_dtype_t safetensors_dtype_from_string(const char* dtype_str) {
    if (!dtype_str) return SAFETENSORS_DTYPE_UNKNOWN;

    if (strcmp(dtype_str, "F64") == 0) return SAFETENSORS_DTYPE_F64;
    if (strcmp(dtype_str, "F32") == 0) return SAFETENSORS_DTYPE_F32;
    if (strcmp(dtype_str, "F16") == 0) return SAFETENSORS_DTYPE_F16;
    if (strcmp(dtype_str, "BF16") == 0) return SAFETENSORS_DTYPE_BF16;
    if (strcmp(dtype_str, "I64") == 0) return SAFETENSORS_DTYPE_I64;
    if (strcmp(dtype_str, "I32") == 0) return SAFETENSORS_DTYPE_I32;
    if (strcmp(dtype_str, "I16") == 0) return SAFETENSORS_DTYPE_I16;
    if (strcmp(dtype_str, "I8") == 0) return SAFETENSORS_DTYPE_I8;
    if (strcmp(dtype_str, "U64") == 0) return SAFETENSORS_DTYPE_U64;
    if (strcmp(dtype_str, "U32") == 0) return SAFETENSORS_DTYPE_U32;
    if (strcmp(dtype_str, "U16") == 0) return SAFETENSORS_DTYPE_U16;
    if (strcmp(dtype_str, "U8") == 0) return SAFETENSORS_DTYPE_U8;
    if (strcmp(dtype_str, "BOOL") == 0) return SAFETENSORS_DTYPE_BOOL;

    return SAFETENSORS_DTYPE_UNKNOWN;
}

// Convert safetensors dtype to boat dtype
static boat_dtype_t boat_dtype_from_safetensors(safetensors_dtype_t sdtype) {
    switch (sdtype) {
        case SAFETENSORS_DTYPE_F64: return BOAT_DTYPE_FLOAT64;
        case SAFETENSORS_DTYPE_F32: return BOAT_DTYPE_FLOAT32;
        case SAFETENSORS_DTYPE_F16: return BOAT_DTYPE_FLOAT16;
        case SAFETENSORS_DTYPE_BF16: return BOAT_DTYPE_FLOAT16; // TODO: Add BF16 support
        case SAFETENSORS_DTYPE_I64: return BOAT_DTYPE_INT64;
        case SAFETENSORS_DTYPE_I32: return BOAT_DTYPE_INT32;
        case SAFETENSORS_DTYPE_I16: return BOAT_DTYPE_INT32; // Map to INT32
        case SAFETENSORS_DTYPE_I8: return BOAT_DTYPE_INT32;  // Map to INT32
        case SAFETENSORS_DTYPE_U64: return BOAT_DTYPE_INT64; // Map to INT64
        case SAFETENSORS_DTYPE_U32: return BOAT_DTYPE_INT32; // Map to INT32
        case SAFETENSORS_DTYPE_U16: return BOAT_DTYPE_INT32; // Map to INT32
        case SAFETENSORS_DTYPE_U8: return BOAT_DTYPE_UINT8;
        case SAFETENSORS_DTYPE_BOOL: return BOAT_DTYPE_BOOL;
        default: return BOAT_DTYPE_FLOAT32; // Default fallback
    }
}

// Simple JSON parsing helper (minimal implementation without cJSON)
static char* extract_json_string(const char* json, const char* key) {
    // TODO: Implement proper JSON parsing
    // This is a placeholder that returns NULL
    (void)json;
    (void)key;
    return NULL;
}

// Parse safetensors header
static char* parse_safetensors_header(const uint8_t* data, size_t size, size_t* header_len) {
    if (size < 8) return NULL;

    // First 8 bytes: little-endian header length
    uint64_t json_len = 0;
    memcpy(&json_len, data, 8);

    // Check bounds
    if (json_len == 0 || json_len > size - 8) {
        return NULL;
    }

    // Allocate buffer for JSON string
    char* json_str = malloc(json_len + 1);
    if (!json_str) return NULL;

    // Copy JSON data (starts at offset 8)
    memcpy(json_str, data + 8, json_len);
    json_str[json_len] = '\0';

    *header_len = 8 + json_len;
    return json_str;
}

// Check if data looks like a safetensors file
static bool is_safetensors_format(const uint8_t* data, size_t size) {
    fprintf(stderr, "is_safetensors_format: size=%zu\n", size);
    if (size < 8) {
        fprintf(stderr, "  size < 8\n");
        return false;
    }

    // Read header length
    uint64_t json_len = 0;
    memcpy(&json_len, data, 8);

    // Basic validation
    if (json_len == 0 || json_len > size - 8) {
        return false;
    }

    // Check that JSON starts with '{' and ends with '}' (allowing padding spaces)
    // This is a simple heuristic
    if (data[8] != '{') {
        fprintf(stderr, "  JSON does not start with '{'\n");
        return false;
    }
    // Find last non-space character before JSON end
    size_t pos = 8 + json_len - 1;
    while (pos > 8 && data[pos] == ' ') {
        pos--;
    }
    if (data[pos] != '}') {
        fprintf(stderr, "  JSON does not end with '}' (found 0x%02x at pos %zu)\n", data[pos], pos);
        return false;
    }

    return true;
}

// Load safetensors format weights
static bool load_safetensors(const void* data, size_t size, const hf_config_t* config, const boat_model_t* model) {
    if (!data || size == 0 || !model) return false;

    const uint8_t* file_data = (const uint8_t*)data;

    // Parse safetensors header
    size_t header_size = 0;
    char* json_header = parse_safetensors_header(file_data, size, &header_size);
    if (!json_header) {
        fprintf(stderr, "Failed to parse safetensors header\n");
        return false;
    }

    // Debug: print first 200 chars of JSON header
    printf("Safetensors JSON header (%zu bytes):\n%.*s\n", strlen(json_header),
           (int)(strlen(json_header) > 200 ? 200 : strlen(json_header)), json_header);

    int tensor_count = 0;
    bool success = false;

#ifdef BOAT_USE_CJSON
    // Parse JSON using cJSON
    cJSON* root = cJSON_Parse(json_header);
    if (!root) {
        fprintf(stderr, "Failed to parse safetensors JSON header\n");
        free(json_header);
        return false;
    }

    // Get all keys in the JSON object
    cJSON* item = NULL;
    cJSON_ArrayForEach(item, root) {
        const char* key = item->string;

        // Skip metadata fields
        if (strcmp(key, "__header__") == 0 || strcmp(key, "__metadata__") == 0) {
            continue;
        }

        // Check if this is a tensor entry (should have dtype, shape, data_offsets)
        cJSON* dtype_item = cJSON_GetObjectItem(item, "dtype");
        cJSON* shape_item = cJSON_GetObjectItem(item, "shape");
        cJSON* offsets_item = cJSON_GetObjectItem(item, "data_offsets");

        if (!dtype_item || !shape_item || !offsets_item) {
            fprintf(stderr, "Warning: tensor '%s' missing required fields\n", key);
            continue;
        }

        // Extract dtype
        if (!cJSON_IsString(dtype_item)) {
            fprintf(stderr, "Warning: tensor '%s' dtype is not a string\n", key);
            continue;
        }
        const char* dtype_str = dtype_item->valuestring;
        safetensors_dtype_t sdtype = safetensors_dtype_from_string(dtype_str);
        boat_dtype_t bdtype = boat_dtype_from_safetensors(sdtype);

        // Extract shape
        if (!cJSON_IsArray(shape_item)) {
            fprintf(stderr, "Warning: tensor '%s' shape is not an array\n", key);
            continue;
        }

        int shape_len = cJSON_GetArraySize(shape_item);
        if (shape_len < 0 || shape_len > BOAT_MAX_DIMS) {
            fprintf(stderr, "Warning: tensor '%s' invalid shape length %d\n", key, shape_len);
            continue;
        }

        // Handle scalar tensors (shape_len = 0)
        int64_t* shape = NULL;
        if (shape_len > 0) {
            shape = malloc(shape_len * sizeof(int64_t));
            if (!shape) {
                fprintf(stderr, "Memory allocation failed for shape\n");
                continue;
            }
        }

        // Only fill shape array if shape_len > 0 (scalar tensors have empty shape)
        if (shape_len > 0) {
            for (int i = 0; i < shape_len; i++) {
                cJSON* dim = cJSON_GetArrayItem(shape_item, i);
                if (cJSON_IsNumber(dim)) {
                    shape[i] = (int64_t)dim->valueint;
                } else {
                    shape[i] = 1; // Default
                }
            }
        }

        // Extract data offsets
        if (!cJSON_IsArray(offsets_item) || cJSON_GetArraySize(offsets_item) != 2) {
            fprintf(stderr, "Warning: tensor '%s' invalid data_offsets\n", key);
            free(shape);
            continue;
        }

        cJSON* offset_start = cJSON_GetArrayItem(offsets_item, 0);
        cJSON* offset_end = cJSON_GetArrayItem(offsets_item, 1);
        if (!cJSON_IsNumber(offset_start) || !cJSON_IsNumber(offset_end)) {
            fprintf(stderr, "Warning: tensor '%s' offsets not numbers\n", key);
            free(shape);
            continue;
        }

        size_t start = (size_t)offset_start->valueint;
        size_t end = (size_t)offset_end->valueint;
        if (start >= end || end > (size - header_size)) {
            fprintf(stderr, "Warning: tensor '%s' invalid offset range\n", key);
            free(shape);
            continue;
        }

        // Calculate data pointer
        size_t data_offset = header_size + start;
        size_t data_size = end - start;
        const void* tensor_data = file_data + data_offset;

        // Create boat tensor
        boat_tensor_t* tensor = boat_tensor_from_data(shape, shape_len, bdtype, tensor_data);
        free(shape);

        if (!tensor) {
            fprintf(stderr, "Warning: failed to create tensor '%s'\n", key);
            continue;
        }

        // Associate tensor with model layer
        boat_layer_t* layer = create_layer_from_config(config, key, tensor);
        if (layer) {
            // Check if this is a special marker for bias update (not an actual layer)
            if (layer == (boat_layer_t*)1) {
                // Bias tensor successfully associated with existing layer
                tensor_count++;
                printf("Associated bias tensor '%s' with existing layer\n", key);
            } else {
                // Actual layer created, add to model
                boat_model_add_layer(model, layer);
                tensor_count++;
                printf("Loaded tensor '%s' with shape [", key);
                if (shape_len > 0) {
                    for (int i = 0; i < shape_len; i++) {
                        printf("%" PRId64, shape[i]);
                        if (i < shape_len - 1) printf(", ");
                    }
                }
                printf("] dtype=%s\n", dtype_str);
            }
        } else {
            fprintf(stderr, "Warning: no layer mapping for tensor '%s'\n", key);
        }

        boat_tensor_unref(tensor); // Model now holds reference
    }

    cJSON_Delete(root);
    success = tensor_count > 0;
#else
    // Without cJSON, use simple heuristic to count tensors
    const char* ptr = json_header;
    while ((ptr = strstr(ptr, "\"dtype\"")) != NULL) {
        tensor_count++;
        ptr += 7; // Move past found string
    }

    printf("Found %d tensors in safetensors file (cJSON not available, cannot load)\n", tensor_count);
    success = false; // Can't actually load without JSON parsing
#endif

    free(json_header);
    return success;
}

// Load PyTorch .bin format weights (state_dict)
static bool load_pytorch_bin(const void* data, size_t size, const hf_config_t* config, const boat_model_t* model) {
    // TODO: Implement PyTorch .bin parsing
    // This is more complex due to Pickle format
    // Consider requiring safetensors instead
    (void)data;
    (void)size;
    (void)config;
    (void)model;
    return false;
}

// Create a simple layer wrapper for Hugging Face model loading
static hf_layer_wrapper_t* create_layer_wrapper(const char* layer_type, boat_tensor_t* weight) {
    hf_layer_wrapper_t* wrapper = malloc(sizeof(hf_layer_wrapper_t));
    if (!wrapper) {
        return NULL;
    }

    wrapper->weight = weight;
    if (weight) {
        boat_tensor_ref(weight); // Increase ref count since wrapper owns it
    }
    wrapper->bias = NULL;
    wrapper->layer_type = layer_type ? strdup(layer_type) : strdup("unknown");
    wrapper->layer_data = NULL; // For future expansion

    return wrapper;
}

// Free a layer wrapper
static void free_layer_wrapper(hf_layer_wrapper_t* wrapper) {
    if (!wrapper) return;

    if (wrapper->weight) boat_tensor_unref(wrapper->weight);
    if (wrapper->bias) boat_tensor_unref(wrapper->bias);
    free(wrapper->layer_type);
    free(wrapper);
}

// Create layers based on configuration
static boat_layer_t* create_layer_from_config(const hf_config_t* config, const char* layer_name, boat_tensor_t* weight) {
    if (!config || !layer_name || !weight) return NULL;

    // Get tensor properties for logging
    size_t ndim = boat_tensor_ndim(weight);
    const int64_t* shape = boat_tensor_shape(weight);

    printf("Creating layer for tensor: %s (shape: [", layer_name);
    for (size_t i = 0; i < ndim; i++) {
        printf("%" PRId64, shape[i]);
        if (i < ndim - 1) printf(", ");
    }
    printf("])\n");

    // Map layer names based on common Hugging Face naming patterns
    // This is a simplified mapping - real implementation would be more sophisticated

    // Check for embedding layers
    if (strstr(layer_name, "embedding") != NULL || strstr(layer_name, "embeddings") != NULL) {
        printf("  Detected embedding layer: %s\n", layer_name);
        // TODO: Create embedding layer
        return NULL;
    }

    // Check for attention layers
    if (strstr(layer_name, "attention") != NULL) {
        printf("  Detected attention layer: %s\n", layer_name);

        // Check what type of attention weight this is
        if (strstr(layer_name, "query") != NULL) {
            printf("    Query weight\n");
        } else if (strstr(layer_name, "key") != NULL) {
            printf("    Key weight\n");
        } else if (strstr(layer_name, "value") != NULL) {
            printf("    Value weight\n");
        } else if (strstr(layer_name, "output") != NULL) {
            printf("    Output weight\n");
        }

        // TODO: Create attention layer with appropriate dimensions
        // For now, return NULL as placeholder
        return NULL;
    }

    // Check for dense/linear layers
    if (strstr(layer_name, "dense") != NULL ||
        strstr(layer_name, "linear") != NULL ||
        (strstr(layer_name, "weight") != NULL && ndim == 2) ||
        (strstr(layer_name, "bias") != NULL && ndim == 1)) {

        printf("  Detected dense/linear layer parameter: %s\n", layer_name);

        // Get base layer name (e.g., "lin" from "lin.weight" or "lin.bias")
        char* base_name = get_base_layer_name(layer_name);
        if (!base_name) {
            return NULL;
        }

        // Create or find builder for this layer
        hf_layer_builder_t* builder = find_or_create_builder(config, base_name, "dense");
        free(base_name);

        if (!builder) {
            fprintf(stderr, "Failed to create/find builder for layer: %s\n", layer_name);
            return NULL;
        }

        // Check if this is a weight or bias tensor
        if (strstr(layer_name, "weight") != NULL && ndim == 2) {
            printf("    Weight matrix: %" PRId64 "x%" PRId64 "\n", shape[0], shape[1]);
            if (!set_builder_weight(builder, weight)) {
                fprintf(stderr, "Failed to set weight for builder: %s\n", layer_name);
                return NULL;
            }
        } else if (strstr(layer_name, "bias") != NULL && ndim == 1) {
            printf("    Bias vector: length %" PRId64 "\n", shape[0]);
            if (!set_builder_bias(builder, weight)) {
                fprintf(stderr, "Failed to set bias for builder: %s\n", layer_name);
                return NULL;
            }
        } else {
            // Not a weight or bias tensor, ignore
            return NULL;
        }

        // If builder now has weight (and optionally bias), complete it
        if (builder->has_weight) {
            // Check if layer already exists (created from weight)
            if (builder->layer) {
                // Layer already exists, bias was updated via set_builder_bias
                // Return a special marker to indicate success without creating new layer
                return (boat_layer_t*)1; // Special marker for successful bias update
            } else {
                // Layer doesn't exist yet, create it
                boat_layer_t* layer = complete_builder(builder, config);
                if (layer) {
                    return layer;
                } else {
                    fprintf(stderr, "Failed to complete builder for layer: %s\n", layer_name);
                    return NULL;
                }
            }
        } else {
            // Weight not yet available, return NULL for now
            return NULL;
        }
    }

    // Check for layer normalization
    if (strstr(layer_name, "layer_norm") != NULL || strstr(layer_name, "ln") != NULL) {
        printf("  Detected layer normalization: %s\n", layer_name);

        // Get base layer name (e.g., "layer_norm" from "layer_norm.weight" or "layer_norm.bias")
        char* base_name = get_base_layer_name(layer_name);
        if (!base_name) {
            return NULL;
        }

        // Create or find builder for this layer
        hf_layer_builder_t* builder = find_or_create_builder(config, base_name, "layer_norm");
        free(base_name);

        if (!builder) {
            fprintf(stderr, "Failed to create/find builder for layer: %s\n", layer_name);
            return NULL;
        }

        // Check if this is a weight or bias tensor
        if (strstr(layer_name, "weight") != NULL && ndim == 1) {
            printf("    Scale (gamma) parameter: length %" PRId64 "\n", shape[0]);
            if (!set_builder_weight(builder, weight)) {
                fprintf(stderr, "Failed to set weight for builder: %s\n", layer_name);
                return NULL;
            }
        } else if (strstr(layer_name, "bias") != NULL && ndim == 1) {
            printf("    Shift (beta) parameter: length %" PRId64 "\n", shape[0]);
            if (!set_builder_bias(builder, weight)) {
                fprintf(stderr, "Failed to set bias for builder: %s\n", layer_name);
                return NULL;
            }
        } else {
            // Not a weight or bias tensor, ignore
            return NULL;
        }

        // If builder now has weight (and optionally bias), complete it
        if (builder->has_weight) {
            // Check if layer already exists (created from weight)
            if (builder->layer) {
                // Layer already exists, bias was updated via set_builder_bias
                // Return a special marker to indicate success without creating new layer
                return (boat_layer_t*)1; // Special marker for successful bias update
            } else {
                // Layer doesn't exist yet, create it
                boat_layer_t* layer = complete_builder(builder, config);
                if (layer) {
                    return layer;
                } else {
                    fprintf(stderr, "Failed to complete builder for layer: %s\n", layer_name);
                    return NULL;
                }
            }
        } else {
            // Weight not yet available, return NULL for now
            return NULL;
        }
    }

    // Check for convolutional layers (for CNN models)
    if (strstr(layer_name, "conv") != NULL && ndim >= 3) {
        printf("  Detected convolutional layer: %s\n", layer_name);
        // Create a wrapper layer for convolutional layer
        hf_layer_wrapper_t* wrapper = create_layer_wrapper("conv", weight);
        if (!wrapper) {
            return NULL;
        }
        boat_layer_t* layer = malloc(sizeof(boat_layer_t));
        if (!layer) {
            free_layer_wrapper(wrapper);
            return NULL;
        }
        layer->data = wrapper;
        layer->ops = &wrapper_layer_ops;
        printf("    Created convolutional layer wrapper\n");
        return layer;
    }

    printf("  No specific layer mapping for: %s\n", layer_name);
    return NULL;
}

// Load Hugging Face model from directory
boat_model_t* boat_huggingface_load(const char* model_dir) {
    if (!model_dir) return NULL;

    // Construct file paths
    char config_path[1024];
    char weights_path[1024];

    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

    // Check for safetensors file first
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", model_dir);

    FILE* weights_file = fopen(weights_path, "rb");
    if (!weights_file) {
        // Try pytorch_model.bin as fallback
        snprintf(weights_path, sizeof(weights_path), "%s/pytorch_model.bin", model_dir);
        weights_file = fopen(weights_path, "rb");
        if (!weights_file) {
            fprintf(stderr, "No weight file found (tried model.safetensors and pytorch_model.bin)\n");
            return NULL;
        }
    }
    fclose(weights_file); // We'll reopen it later for reading

    printf("Loading Hugging Face model from directory: %s\n", model_dir);
    printf("Config file: %s\n", config_path);
    printf("Weights file: %s\n", weights_path);

    // 1. Read config.json into string
    char* config_json = read_file_to_string(config_path);
    if (!config_json) {
        fprintf(stderr, "Failed to read config.json\n");
        return NULL;
    }

    // 2. Read weights file into memory buffer
    FILE* fp = fopen(weights_path, "rb");
    if (!fp) {
        free(config_json);
        fprintf(stderr, "Failed to open weights file: %s\n", weights_path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long weights_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (weights_size <= 0) {
        fclose(fp);
        free(config_json);
        fprintf(stderr, "Weights file is empty or error getting size: %s\n", weights_path);
        return NULL;
    }

    uint8_t* weights_data = (uint8_t*)malloc(weights_size);
    if (!weights_data) {
        fclose(fp);
        free(config_json);
        fprintf(stderr, "Failed to allocate memory for weights data: %zu bytes\n", (size_t)weights_size);
        return NULL;
    }

    size_t bytes_read = fread(weights_data, 1, weights_size, fp);
    fclose(fp);

    if (bytes_read != (size_t)weights_size) {
        free(weights_data);
        free(config_json);
        fprintf(stderr, "Failed to read entire weights file: %s (read %zu of %ld bytes)\n",
                weights_path, bytes_read, weights_size);
        return NULL;
    }

    // 3. Call boat_huggingface_load_from_memory with the data
    boat_model_t* model = boat_huggingface_load_from_memory(config_json, weights_data, weights_size);

    // Clean up
    free(weights_data);
    free(config_json);

    if (!model) {
        fprintf(stderr, "Failed to load model from memory data\n");
    } else {
        printf("Successfully loaded model from directory: %s\n", model_dir);
    }

    return model;
}

// Load Hugging Face model from memory buffers
boat_model_t* boat_huggingface_load_from_memory(const char* config_json, const void* weights_data, size_t weights_size) {
    if (!config_json || !weights_data || weights_size == 0) {
        return NULL;
    }

    // Parse model configuration
    hf_config_t* config = parse_config(config_json);
    if (!config) {
        fprintf(stderr, "Failed to parse model configuration\n");
        return NULL;
    }

    // Create empty model
    boat_model_t* model = boat_model_create();
    if (!model) {
        fprintf(stderr, "Failed to create model\n");
        free_config(config);
        return NULL;
    }

    bool success = false;
    const uint8_t* weights = (const uint8_t*)weights_data;

    // Detect format and load weights
    if (is_safetensors_format(weights, weights_size)) {
        printf("Detected safetensors format\n");
        success = load_safetensors(weights_data, weights_size, config, model);
    } else {
        // Try to detect PyTorch .bin format
        // Simple heuristic: check for pickle magic bytes
        if (weights_size >= 2 && weights[0] == 0x80 && weights[1] == 0x04) {
            printf("Detected PyTorch .bin format (pickle)\n");
            success = load_pytorch_bin(weights_data, weights_size, config, model);
        } else {
            fprintf(stderr, "Unknown weights format\n");
        }
    }

    if (!success) {
        fprintf(stderr, "Failed to load weights\n");
        boat_model_free(model);
        free_config(config);
        return NULL;
    }

    // Model loaded successfully
    free_config(config);
    return model;
}

// Check if directory contains a valid Hugging Face model
bool boat_huggingface_check(const char* model_dir) {
    if (!model_dir) return false;

    // Check for config.json
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

    FILE* config_file = fopen(config_path, "rb");
    if (!config_file) {
        return false;
    }
    fclose(config_file);

    // Check for at least one weight file (safetensors or pytorch .bin)
    char weights_path[1024];
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", model_dir);
    FILE* weights_file = fopen(weights_path, "rb");
    if (weights_file) {
        fclose(weights_file);
        return true;
    }

    snprintf(weights_path, sizeof(weights_path), "%s/pytorch_model.bin", model_dir);
    weights_file = fopen(weights_path, "rb");
    if (weights_file) {
        fclose(weights_file);
        return true;
    }

    // No weight file found
    return false;
}

// Get model configuration information
char* boat_huggingface_get_config(const char* model_dir) {
    if (!model_dir) return NULL;

    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

    return read_file_to_string(config_path);
}

// Save model to Hugging Face format directory
bool boat_huggingface_save(const boat_model_t* model, const char* model_dir) {
    // TODO: Implement saving to Hugging Face format
    // This would involve:
    // 1. Convert Boat layers to Hugging Face configuration
    // 2. Save config.json
    // 3. Save weights in safetensors format
    (void)model;
    (void)model_dir;
    return false;
}

// Extract base layer name from tensor name (e.g., "lin.weight" -> "lin")
static char* get_base_layer_name(const char* tensor_name) {
    if (!tensor_name) return NULL;

    // Make a copy to work with
    char* base = strdup(tensor_name);
    if (!base) return NULL;

    // Remove common suffixes
    char* suffixes[] = {".weight", ".bias", ".gamma", ".beta", ".running_mean", ".running_var", ".num_batches_tracked"};
    for (size_t i = 0; i < sizeof(suffixes)/sizeof(suffixes[0]); i++) {
        char* suffix = suffixes[i];
        size_t suffix_len = strlen(suffix);
        size_t base_len = strlen(base);
        if (base_len >= suffix_len && strcmp(base + base_len - suffix_len, suffix) == 0) {
            base[base_len - suffix_len] = '\0';
            break;
        }
    }

    return base;
}

// Create actual Boat layer from tensor
static boat_layer_t* create_actual_layer_from_tensor(const char* base_name, const char* tensor_name, boat_tensor_t* tensor) {
    if (!base_name || !tensor_name || !tensor) return NULL;

    size_t ndim = boat_tensor_ndim(tensor);
    const int64_t* shape = boat_tensor_shape(tensor);

    printf("Creating actual layer for tensor: %s (base: %s, shape: [", tensor_name, base_name);
    for (size_t i = 0; i < ndim; i++) {
        printf("%" PRId64, shape[i]);
        if (i < ndim - 1) printf(", ");
    }
    printf("])\n");

    // Check for dense/linear layers (weight tensor with 2 dimensions)
    if (strstr(tensor_name, ".weight") != NULL && ndim == 2) {
        printf("  Creating dense layer for weight tensor\n");

        // Extract dimensions from weight shape
        size_t input_features = shape[0];
        size_t output_features = shape[1];

        // Create dense layer with bias (will be set later if bias tensor exists)
        boat_dense_layer_t* dense_layer = boat_dense_layer_create(input_features, output_features, true);
        if (!dense_layer) {
            fprintf(stderr, "Failed to create dense layer\n");
            return NULL;
        }

        // Set weight tensor
        boat_dense_layer_set_weight(dense_layer, tensor);

        // Create boat_layer_t wrapper
        boat_layer_t* layer = malloc(sizeof(boat_layer_t));
        if (!layer) {
            boat_dense_layer_free(dense_layer);
            return NULL;
        }
        layer->data = dense_layer;
        printf("    Created dense layer with dimensions %zu -> %zu\n", input_features, output_features);
        return layer;
    }
    // Check for bias tensor (1D) - we need to find the corresponding layer
    else if (strstr(tensor_name, ".bias") != NULL && ndim == 1) {
        printf("  Bias tensor detected, will be associated with existing layer\n");
        // Bias will be handled when the corresponding weight is processed
        return NULL;
    }
    // Check for layer normalization parameters
    else if ((strstr(tensor_name, ".weight") != NULL || strstr(tensor_name, ".bias") != NULL) &&
             (strstr(base_name, "layer_norm") != NULL || strstr(base_name, "ln") != NULL) && ndim == 1) {
        printf("  Layer normalization parameter detected\n");
        // TODO: Implement layer normalization layer creation
        return NULL;
    }

    printf("  No actual layer implementation for this tensor type\n");
    return NULL;
}

// Read file contents into a string (caller must free)
static char* read_file_to_string(const char* filename) {
    if (!filename) return NULL;

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }

    // Get file size
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size <= 0) {
        fclose(fp);
        fprintf(stderr, "File is empty or error getting size: %s\n", filename);
        return NULL;
    }

    // Allocate memory for file contents + null terminator
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        fclose(fp);
        fprintf(stderr, "Failed to allocate memory for file: %s\n", filename);
        return NULL;
    }

    // Read file contents
    size_t bytes_read = fread(buffer, 1, file_size, fp);
    fclose(fp);

    if (bytes_read != (size_t)file_size) {
        free(buffer);
        fprintf(stderr, "Failed to read entire file: %s (read %zu of %ld bytes)\n",
                filename, bytes_read, file_size);
        return NULL;
    }

    buffer[file_size] = '\0'; // Null-terminate the string
    return buffer;
}