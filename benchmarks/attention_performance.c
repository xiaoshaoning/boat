// attention_performance.c - Attention layer performance benchmark
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers.h>
#include <boat/layers/attention.h>
#include <boat/tensor.h>
#include <boat/ops.h>
#include <boat/memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// High-resolution timer for Windows
#ifdef _WIN32
#include <windows.h>
static double get_time_ms() {
    static LARGE_INTEGER frequency;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
}
#else
#include <sys/time.h>
static double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}
#endif

// Structure to hold benchmark configuration
typedef struct {
    size_t batch_size;
    size_t seq_len;
    size_t hidden_size;
    size_t num_heads;
    int iterations;
    bool causal_mask;
} benchmark_config_t;

// Structure to hold benchmark results
typedef struct {
    double forward_time_ms;
    double backward_time_ms;
    double total_time_ms;
    size_t total_parameters;
    size_t flops_estimate;  // Rough FLOPs estimate
    double flops_per_second; // GFLOPs/s
} benchmark_result_t;

// Generate random tensor data
static boat_tensor_t* create_random_tensor(const int64_t* shape, size_t ndim, boat_dtype_t dtype) {
    boat_tensor_t* tensor = boat_tensor_create(shape, ndim, dtype, BOAT_DEVICE_CPU);
    if (!tensor) return NULL;

    size_t num_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
        num_elements *= shape[i];
    }

    float* data = (float*)boat_tensor_data(tensor);
    for (size_t i = 0; i < num_elements; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Uniform [-1, 1]
    }

    return tensor;
}

// Estimate FLOPs for attention layer (simplified)
static size_t estimate_attention_flops(size_t batch_size, size_t seq_len, size_t hidden_size, size_t num_heads) {
    size_t head_size = hidden_size / num_heads;

    // QKV projections: 3 * batch * seq_len * hidden_size * hidden_size
    size_t qkv_proj = 3 * batch_size * seq_len * hidden_size * hidden_size;

    // Attention scores: batch * num_heads * seq_len * seq_len * head_size
    size_t scores = batch_size * num_heads * seq_len * seq_len * head_size;

    // Softmax: ~3 * batch * num_heads * seq_len * seq_len
    size_t softmax = 3 * batch_size * num_heads * seq_len * seq_len;

    // Attention output: batch * num_heads * seq_len * seq_len * head_size
    size_t attn_out = batch_size * num_heads * seq_len * seq_len * head_size;

    // Final projection: batch * seq_len * hidden_size * hidden_size
    size_t final_proj = batch_size * seq_len * hidden_size * hidden_size;

    return qkv_proj + scores + softmax + attn_out + final_proj;
}

// Direct backward pass wrapper (bypasses layer interface cache issues)
static boat_tensor_t* attention_backward_direct(boat_attention_layer_t* layer, const boat_tensor_t* grad_output) {
    boat_tensor_t* grad_query = NULL;
    boat_tensor_t* grad_key = NULL;
    boat_tensor_t* grad_value = NULL;
    if (boat_attention_backward((boat_attention_t*)layer, grad_output, &grad_query, &grad_key, &grad_value)) {
        if (grad_key) boat_tensor_free(grad_key);
        if (grad_value) boat_tensor_free(grad_value);
        return grad_query;
    }
    return NULL;
}

// Run single benchmark iteration
static benchmark_result_t run_benchmark(const benchmark_config_t* config) {
    benchmark_result_t result = {0};

    // Create attention layer
    boat_attention_layer_t* attention = boat_attention_layer_create(
        config->hidden_size,
        config->num_heads,
        0.0f,  // no dropout for benchmarking
        config->causal_mask
    );

    if (!attention) {
        fprintf(stderr, "Failed to create attention layer\n");
        return result;
    }

    // Create input tensors (query, key, value are same for self-attention)
    int64_t q_shape[] = {(int64_t)config->batch_size, (int64_t)config->seq_len, (int64_t)config->hidden_size};
    boat_tensor_t* query = create_random_tensor(q_shape, 3, BOAT_DTYPE_FLOAT32);
    boat_tensor_t* key = create_random_tensor(q_shape, 3, BOAT_DTYPE_FLOAT32);
    boat_tensor_t* value = create_random_tensor(q_shape, 3, BOAT_DTYPE_FLOAT32);

    if (!query || !key || !value) {
        fprintf(stderr, "Failed to create input tensors\n");
        boat_attention_layer_free(attention);
        if (query) boat_tensor_free(query);
        if (key) boat_tensor_free(key);
        if (value) boat_tensor_free(value);
        return result;
    }

    // Warm-up run (discard timing)
    boat_tensor_t* output = boat_attention_layer_forward(attention, query, key, value, NULL);
    if (output) {
        boat_tensor_free(output);
    }

    // Forward pass benchmark
    double forward_start = get_time_ms();
    for (int i = 0; i < config->iterations; i++) {
        output = boat_attention_layer_forward(attention, query, key, value, NULL);
        if (!output) {
            fprintf(stderr, "Forward pass failed at iteration %d\n", i);
            break;
        }
        boat_tensor_free(output);
    }
    double forward_end = get_time_ms();

    // Create gradient output for backward pass
    boat_tensor_t* grad_output = create_random_tensor(q_shape, 3, BOAT_DTYPE_FLOAT32);
    if (!grad_output) {
        fprintf(stderr, "Failed to create gradient tensor\n");
        boat_attention_layer_free(attention);
        boat_tensor_free(query);
        boat_tensor_free(key);
        boat_tensor_free(value);
        return result;
    }

    // Run one forward pass to get output for backward
    output = boat_attention_layer_forward(attention, query, key, value, NULL);
    if (!output) {
        fprintf(stderr, "Forward pass failed for backward preparation\n");
        boat_tensor_free(grad_output);
        boat_attention_layer_free(attention);
        boat_tensor_free(query);
        boat_tensor_free(key);
        boat_tensor_free(value);
        return result;
    }

    // Check if backward pass is implemented using direct backward call
    printf("[PERF DEBUG] Checking backward pass implementation\n");
    fflush(stdout);
    boat_tensor_t* test_grad = attention_backward_direct(attention, grad_output);
    bool backward_implemented = (test_grad != NULL);
    printf("[PERF DEBUG] Backward check: test_grad=%p, backward_implemented=%d\n", (void*)test_grad, backward_implemented);
    fflush(stdout);

    // Test layer interface directly
    printf("[PERF DEBUG] attention pointer: %p\n", (void*)attention);
    printf("[PERF DEBUG] grad_output pointer: %p\n", (void*)grad_output);
    printf("[PERF DEBUG] boat_attention_layer_backward function address: %p\n", (void*)boat_attention_layer_backward);
    // Try calling via function pointer
    typedef boat_tensor_t* (__cdecl *layer_backward_func_t)(boat_attention_layer_t*, const boat_tensor_t*);
    layer_backward_func_t func_ptr = (layer_backward_func_t)boat_attention_layer_backward;
    printf("[PERF DEBUG] Calling via function pointer: func_ptr=%p, attention=%p, grad_output=%p\n",
           (void*)func_ptr, (void*)attention, (void*)grad_output);
    fflush(stdout);
    boat_tensor_t* layer_grad_ptr = func_ptr(attention, grad_output);
    printf("[PERF DEBUG] Layer interface via function pointer: layer_grad_ptr=%p\n", (void*)layer_grad_ptr);
    if (layer_grad_ptr) {
        boat_tensor_free(layer_grad_ptr);
    }
    // Original direct call
    printf("[PERF DEBUG] Calling boat_attention_layer_backward directly\n");
    fflush(stdout);
    boat_tensor_t* layer_grad = boat_attention_layer_backward(attention, grad_output);
    printf("[PERF DEBUG] Layer interface check: layer_grad=%p\n", (void*)layer_grad);
    fflush(stdout);
    if (layer_grad) {
        boat_tensor_free(layer_grad);
    }

    if (test_grad) {
        boat_tensor_free(test_grad);
    }

    // Backward pass benchmark (only if implemented)
    double backward_start = 0, backward_end = 0;
    if (backward_implemented) {
        backward_start = get_time_ms();
        for (int i = 0; i < config->iterations; i++) {
            boat_tensor_t* grad = attention_backward_direct(attention, grad_output);
            if (!grad) {
                fprintf(stderr, "Backward pass failed at iteration %d\n", i);
                backward_implemented = false;
                break;
            }
            boat_tensor_free(grad);
        }
        backward_end = get_time_ms();
    }

    // Calculate results
    result.forward_time_ms = (forward_end - forward_start) / config->iterations;
    if (backward_implemented) {
        result.backward_time_ms = (backward_end - backward_start) / config->iterations;
        result.total_time_ms = result.forward_time_ms + result.backward_time_ms;
    } else {
        result.backward_time_ms = -1.0;  // Indicates not implemented
        result.total_time_ms = result.forward_time_ms;
    }

    // Estimate FLOPs
    size_t flops_per_forward = estimate_attention_flops(
        config->batch_size, config->seq_len, config->hidden_size, config->num_heads);
    result.flops_estimate = flops_per_forward;
    result.flops_per_second = (flops_per_forward / (result.forward_time_ms / 1000.0)) / 1e9; // GFLOPs/s

    // Count parameters (rough estimate: 4 * hidden_size * hidden_size for Q,K,V,O projections)
    result.total_parameters = 4 * config->hidden_size * config->hidden_size;

    // Cleanup
    boat_tensor_free(output);
    boat_tensor_free(grad_output);
    boat_tensor_free(query);
    boat_tensor_free(key);
    boat_tensor_free(value);
    boat_attention_layer_free(attention);

    return result;
}

// Print benchmark results
static void print_results(const benchmark_config_t* config, const benchmark_result_t* result) {
    printf("=== Attention Layer Performance Benchmark ===\n");
    printf("Configuration:\n");
    printf("  Batch size: %zu\n", config->batch_size);
    printf("  Sequence length: %zu\n", config->seq_len);
    printf("  Hidden size: %zu\n", config->hidden_size);
    printf("  Number of heads: %zu\n", config->num_heads);
    printf("  Causal mask: %s\n", config->causal_mask ? "true" : "false");
    printf("  Iterations: %d\n", config->iterations);
    printf("\nResults:\n");
    printf("  Forward pass: %.3f ms\n", result->forward_time_ms);
    if (result->backward_time_ms >= 0) {
        printf("  Backward pass: %.3f ms\n", result->backward_time_ms);
        printf("  Total time: %.3f ms\n", result->total_time_ms);
    } else {
        printf("  Backward pass: N/A (not implemented)\n");
        printf("  Total time: %.3f ms (forward only)\n", result->total_time_ms);
    }
    printf("  Estimated parameters: %zu\n", result->total_parameters);
    printf("  Estimated FLOPs/forward: %zu\n", result->flops_estimate);
    printf("  Estimated throughput: %.2f GFLOPs/s\n", result->flops_per_second);
    printf("\n");
}

// Main benchmark function
int main() {
    srand(42); // Fixed seed for reproducibility
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    printf("Starting attention layer performance benchmark...\n");

    // Define benchmark configurations
    benchmark_config_t configs[] = {
        // Small configuration (similar to testing)
        {2, 4, 32, 4, 100, false},
        // Medium configuration
        {4, 16, 64, 8, 50, false},
        // Larger configuration
        {8, 32, 128, 8, 20, false},
        // Causal attention
        {4, 16, 64, 8, 50, true},
    };

    int num_configs = sizeof(configs) / sizeof(configs[0]);

    for (int i = 0; i < num_configs; i++) {
        printf("\nRunning configuration %d/%d...\n", i + 1, num_configs);

        benchmark_result_t result = run_benchmark(&configs[i]);

        if (result.total_time_ms > 0) {
            print_results(&configs[i], &result);
        } else {
            fprintf(stderr, "Benchmark failed for configuration %d\n", i);
        }
    }

    printf("Benchmark completed.\n");
    return 0;
}