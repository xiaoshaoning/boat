// test_safetensors.c - Test safetensors parsing functionality
#include <boat/format/huggingface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main() {
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
        fprintf(stderr, "Failed to read config.json\n");
        return 1;
    }

    printf("Config size: %zu bytes\n", config_size);

    // Read model.safetensors
    size_t weights_size = 0;
    char* weights_data = read_file(weights_path, &weights_size);
    if (!weights_data) {
        fprintf(stderr, "Failed to read model.safetensors\n");
        free(config_json);
        return 1;
    }

    printf("Weights size: %zu bytes\n", weights_size);

    // Load model from memory
    printf("Calling boat_huggingface_load_from_memory...\n");
    boat_model_t* model = boat_huggingface_load_from_memory(config_json, weights_data, weights_size);

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