// nanochat_train.cu — NanoChat training executable
// Reads pre-tokenized int32 binary data and runs gradient descent.
#include "training.h"
#include "weights.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                        \
    }                                                                   \
} while(0)

static void print_usage(void) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  nanochat_train <model_dir> <data.bin> [options]\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --steps N        Total training steps (default: 1000)\n");
    fprintf(stderr, "  --warmup N       Warmup steps (default: 100)\n");
    fprintf(stderr, "  --seq-len N      Sequence length (default: 2048, max: 2048)\n");
    fprintf(stderr, "  --peak-lr F      Peak learning rate (default: 3e-4)\n");
    fprintf(stderr, "  --min-lr F       Minimum learning rate (default: 3e-5)\n");
    fprintf(stderr, "  --log N          Log interval in steps (default: 10)\n");
    fprintf(stderr, "  --save N         Save checkpoint every N steps, 0=off (default: 0)\n");
    fprintf(stderr, "  --data-offset N  Starting offset in data file (default: 0)\n");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    const char* model_dir = argv[1];
    const char* data_path = argv[2];

    // Default parameters
    int total_steps = 1000;
    int warmup_steps = 100;
    int seq_len = 2048;
    float peak_lr = 3e-4f;
    float min_lr = 3e-5f;
    int log_interval = 10;
    int save_interval = 0;
    long long data_offset = 0;

    // Parse optional flags
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            total_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc)
            seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--peak-lr") == 0 && i + 1 < argc)
            peak_lr = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--min-lr") == 0 && i + 1 < argc)
            min_lr = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--log") == 0 && i + 1 < argc)
            log_interval = atoi(argv[++i]);
        else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc)
            save_interval = atoi(argv[++i]);
        else if (strcmp(argv[i], "--data-offset") == 0 && i + 1 < argc)
            data_offset = atoll(argv[++i]);
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    if (seq_len < 2 || seq_len > NANOCHAT_MAX_SEQ_LEN) {
        fprintf(stderr, "seq_len must be in [2, %d]\n", NANOCHAT_MAX_SEQ_LEN);
        return 1;
    }

    // Validate CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 1) {
        fprintf(stderr, "No CUDA device found\n");
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "[NanoChat-Train] Device: %s\n", prop.name);

    // ------------------------------------------------------------------
    // 1. Load weights and init CUDA model
    // ------------------------------------------------------------------
    fprintf(stderr, "[NanoChat-Train] Loading weights from: %s\n", model_dir);
    nanochat_weights_t* weights = nanochat_weights_load(model_dir);
    if (!weights) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    nanochat_cuda_model_t model;
    if (!nanochat_cuda_model_init(&model, weights)) {
        fprintf(stderr, "Failed to init CUDA model\n");
        nanochat_weights_free(weights);
        return 1;
    }
    nanochat_weights_free(weights);

    // ------------------------------------------------------------------
    // 2. Allocate training buffers
    // ------------------------------------------------------------------
    if (!nanochat_cuda_model_alloc_train(&model, seq_len)) {
        fprintf(stderr, "Failed to allocate training buffers\n");
        nanochat_cuda_model_free(&model);
        return 1;
    }

    // ------------------------------------------------------------------
    // 3. Load pre-tokenized training data (int32 binary)
    // ------------------------------------------------------------------
    FILE* fp = fopen(data_path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open data file: %s\n", data_path);
        nanochat_cuda_model_free(&model);
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    long long file_bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    long long total_tokens = file_bytes / sizeof(int);

    fprintf(stderr, "[NanoChat-Train] Data file: %s (%lld tokens)\n",
            data_path, total_tokens);

    if (data_offset + seq_len > total_tokens) {
        fprintf(stderr, "Data offset %lld + seq_len %d exceeds total tokens %lld\n",
                data_offset, seq_len, total_tokens);
        fclose(fp);
        nanochat_cuda_model_free(&model);
        return 1;
    }

    // Read all tokens from offset
    long long read_start = data_offset;
    long long read_count = total_tokens - read_start;
    // Clamp to a reasonable max (e.g., enough for total_steps * seq_len)
    long long desired = (long long)total_steps * seq_len;
    if (desired < read_count) read_count = desired;

    int* h_data = (int*)malloc((size_t)read_count * sizeof(int));
    if (!h_data) {
        fprintf(stderr, "malloc failed for data buffer\n");
        fclose(fp);
        nanochat_cuda_model_free(&model);
        return 1;
    }

    fseek(fp, (long)(read_start * sizeof(int)), SEEK_SET);
    size_t items_read = fread(h_data, sizeof(int), (size_t)read_count, fp);
    fclose(fp);

    if ((long long)items_read < seq_len) {
        fprintf(stderr, "Not enough data: read %zu tokens, need at least %d\n",
                items_read, seq_len);
        free(h_data);
        nanochat_cuda_model_free(&model);
        return 1;
    }
    fprintf(stderr, "[NanoChat-Train] Loaded %zu tokens from offset %lld\n",
            items_read, data_offset);

    // Upload tokens to GPU (allocate extra for whole data buffer)
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, (size_t)items_read * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, (size_t)items_read * sizeof(int),
                           cudaMemcpyHostToDevice));
    free(h_data);

    // ------------------------------------------------------------------
    // 4. Training loop
    // ------------------------------------------------------------------
    fprintf(stderr, "\n[NanoChat-Train] Starting training: %d steps, seq_len=%d, "
            "peak_lr=%.2e\n\n", total_steps, seq_len, peak_lr);

    double total_time = 0.0;

    for (int step = 0; step < total_steps; step++) {
        // Pick a random contiguous segment within the data
        int max_start = (int)(items_read - seq_len);
        int start_pos = max_start > 0 ? (rand() % max_start) : 0;
        int* d_tokens = d_data + start_pos;

        // Compute learning rate
        float lr = nanochat_cosine_lr(step, warmup_steps, total_steps, peak_lr, min_lr);

        // Run training step
        float loss;
        cudaEvent_t start_evt, stop_evt;
        CUDA_CHECK(cudaEventCreate(&start_evt));
        CUDA_CHECK(cudaEventCreate(&stop_evt));
        CUDA_CHECK(cudaEventRecord(start_evt, 0));

        nanochat_cuda_train_step(&model, d_tokens, seq_len, lr, &loss);

        CUDA_CHECK(cudaEventRecord(stop_evt, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_evt));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));
        CUDA_CHECK(cudaEventDestroy(start_evt));
        CUDA_CHECK(cudaEventDestroy(stop_evt));
        total_time += ms;

        // Logging
        if (step % log_interval == 0 || step == total_steps - 1) {
            fprintf(stderr, "Step %5d/%d  loss=%.6f  lr=%.2e  %.0fms\n",
                    step, total_steps, loss, lr, ms);
        }

        // Check for NaN
        if (isnan(loss) || isinf(loss)) {
            fprintf(stderr, "\n[NanoChat-Train] NaN/Inf detected at step %d, aborting\n", step);
            break;
        }
    }

    fprintf(stderr, "\n[NanoChat-Train] Training complete: %.1f total, %.1f ms/step avg\n",
            total_time, total_time / total_steps);

    // ------------------------------------------------------------------
    // 5. Cleanup
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_data));
    nanochat_cuda_model_free(&model);

    fprintf(stderr, "[NanoChat-Train] Done.\n");
    return 0;
}
