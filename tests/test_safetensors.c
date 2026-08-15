// test_safetensors.c - Test safetensors parsing functionality
#include <boat/format/huggingface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Read entire file into memory buffer
static char* read_file(const char* filename, size_t* size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    if (file_size < 0) {
        fclose(f);
        return NULL;
    }
    fseek(f, 0, SEEK_SET);

    char* buffer = malloc(file_size + 1);
    if (!buffer) {
        fclose(f);
        return NULL;
    }

    size_t read_size = fread(buffer, 1, file_size, f);
    fclose(f);

    if (read_size != (size_t)file_size) {
        free(buffer);
        return NULL;
    }

    buffer[file_size] = '\0'; // Null-terminate for text files
    *size = file_size;
    return buffer;
}

// Build a minimal safetensors model fully in memory so the parser path is
// exercised even on CI runners that have no local model fixtures.
static int test_synthetic_model(void) {
    const char* config_json = "{\"model_type\": \"mlp\", \"hidden_size\": 2}";
    const char* header =
        "{\"dense.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}";
    const size_t header_len = strlen(header);
    const size_t data_len = 4 * sizeof(float);
    const size_t total_size = 8 + header_len + data_len;

    uint8_t* buf = (uint8_t*)malloc(total_size);
    if (!buf) {
        fprintf(stderr, "Failed to allocate synthetic safetensors buffer\n");
        return 1;
    }

    // Safetensors layout: 8-byte little-endian header length, JSON header, raw data
    uint64_t hlen = (uint64_t)header_len;
    buf[0] = (uint8_t)(hlen & 0xFF);
    buf[1] = (uint8_t)((hlen >> 8) & 0xFF);
    buf[2] = (uint8_t)((hlen >> 16) & 0xFF);
    buf[3] = (uint8_t)((hlen >> 24) & 0xFF);
    buf[4] = (uint8_t)((hlen >> 32) & 0xFF);
    buf[5] = (uint8_t)((hlen >> 40) & 0xFF);
    buf[6] = (uint8_t)((hlen >> 48) & 0xFF);
    buf[7] = (uint8_t)((hlen >> 56) & 0xFF);
    memcpy(buf + 8, header, header_len);

    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(buf + 8 + header_len, data, data_len);

    boat_model_t* model = boat_huggingface_load_from_memory(config_json, buf, total_size);
    if (!model) {
        fprintf(stderr, "Synthetic safetensors model failed to load\n");
        free(buf);
        return 1;
    }

    size_t layer_count = boat_model_layer_count(model);
    boat_model_free(model);
    free(buf);

    if (layer_count != 1) {
        fprintf(stderr, "Synthetic model: expected 1 layer, got %zu\n", layer_count);
        return 1;
    }
    return 0;
}

int main() {
    // Always run the self-contained synthetic model first
    if (test_synthetic_model() != 0) {
        return 1;
    }

    const char* model_dir = "D:/huggingface/mnist-cnn-digit-classifier";
    char config_path[1024];
    char weights_path[1024];

    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", model_dir);

    printf("Testing safetensors parsing...\n");
    printf("Config file: %s\n", config_path);
    printf("Weights file: %s\n", weights_path);

    // Read config.json
    size_t config_size = 0;
    char* config_json = read_file(config_path, &config_size);
    if (!config_json) {
        // Optional deeper check: fixtures absent (e.g. CI) - synthetic test above already ran
        printf("File-based check skipped: %s not found\n", config_path);
        return 0;
    }

    printf("Config size: %zu bytes\n", config_size);

    // Read model.safetensors
    size_t weights_size = 0;
    char* weights_data = read_file(weights_path, &weights_size);
    if (!weights_data) {
        // Optional deeper check: fixtures absent (e.g. CI) - synthetic test above already ran
        printf("File-based check skipped: %s not found\n", weights_path);
        free(config_json);
        return 0;
    }

    printf("Weights size: %zu bytes\n", weights_size);

    // Load model from memory
    printf("Calling boat_huggingface_load_from_memory...\n");
    boat_model_t* model =
        boat_huggingface_load_from_memory(config_json, weights_data, weights_size);

    if (model) {
        printf("Successfully loaded model!\n");
        // TODO: verify model structure
        boat_model_free(model);
    } else {
        fprintf(stderr, "Failed to load model\n");
    }

    free(config_json);
    free(weights_data);

    printf("Test completed.\n");
    return 0;
}