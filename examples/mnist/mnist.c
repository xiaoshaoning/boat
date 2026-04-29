// mnist.c - MNIST digit recognition with Boat framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <boat/tensor.h>
#include <boat/layers.h>
#include <boat/ops.h>
#include <boat/optimizers.h>
#include <boat/loss.h>
#include <boat/memory.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <io.h>
#define F_OK 0
#define access _access
#else
#include <unistd.h>
#endif

// Data loading functions
boat_tensor_t* load_tensor_binary(const char* filename, boat_dtype_t dtype) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }

    // Read number of dimensions
    uint32_t ndim;
    if (fread(&ndim, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    // Read shape
    int64_t* shape = malloc(sizeof(int64_t) * ndim);
    if (!shape) {
        fclose(f);
        return NULL;
    }

    for (uint32_t i = 0; i < ndim; i++) {
        uint32_t dim;
        if (fread(&dim, sizeof(uint32_t), 1, f) != 1) {
            free(shape);
            fclose(f);
            return NULL;
        }
        shape[i] = (int64_t)dim;
    }

    // Calculate total elements
    size_t total_elements = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }

    // Allocate data buffer
    size_t element_size = boat_dtype_size(dtype);
    void* data = malloc(total_elements * element_size);
    if (!data) {
        free(shape);
        fclose(f);
        return NULL;
    }

    // Read data
    if (fread(data, element_size, total_elements, f) != total_elements) {
        free(data);
        free(shape);
        fclose(f);
        return NULL;
    }

    fclose(f);

    // Create tensor
    boat_tensor_t* tensor = boat_tensor_from_data(shape, ndim, dtype, data);
    free(data);
    free(shape);

    return tensor;
}

// Model structure
typedef struct {
    boat_conv_layer_t* conv1;
    boat_conv_layer_t* conv2;
    boat_pool_layer_t* pool1;
    boat_pool_layer_t* pool2;
    boat_flatten_layer_t* flatten;
    boat_dense_layer_t* fc1;
    boat_dense_layer_t* fc2;
    boat_relu_layer_t* relu1;
    boat_relu_layer_t* relu2;
    boat_relu_layer_t* relu3;
    boat_softmax_layer_t* softmax;
} mnist_model_t;

mnist_model_t* create_mnist_model() {
    mnist_model_t* model = malloc(sizeof(mnist_model_t));
    if (!model) return NULL;

    // Create layers
    model->conv1 = boat_conv_layer_create(1, 32, 3, 1, 1);  // 1->32 channels, 3x3 kernel, stride=1, padding=1
    model->relu1 = boat_relu_layer_create();
    model->pool1 = boat_pool_layer_create(2, 2, 0);         // 2x2 max pool, stride=2

    model->conv2 = boat_conv_layer_create(32, 64, 3, 1, 1); // 32->64 channels
    model->relu2 = boat_relu_layer_create();
    model->pool2 = boat_pool_layer_create(2, 2, 0);

    model->flatten = boat_flatten_layer_create();
    model->fc1 = boat_dense_layer_create(7*7*64, 128, true); // After 2 poolings: 28->14->7
    model->relu3 = boat_relu_layer_create();
    model->fc2 = boat_dense_layer_create(128, 10, true);

    model->softmax = boat_softmax_layer_create(-1);  // Apply softmax on last dimension

    // Check for creation errors
    if (!model->conv1 || !model->conv2 || !model->pool1 || !model->pool2 ||
        !model->flatten || !model->fc1 || !model->fc2 ||
        !model->relu1 || !model->relu2 || !model->relu3 || !model->softmax) {
        fprintf(stderr, "Error: Failed to create one or more layers\n");
        free(model);
        return NULL;
    }

    printf("MNIST model created successfully\n");
    printf("Architecture:\n");
    printf("  Input: 1x28x28\n");
    printf("  Conv2D(1->32, 3x3, padding=1) -> ReLU -> MaxPool2D(2x2)\n");
    printf("  Conv2D(32->64, 3x3, padding=1) -> ReLU -> MaxPool2D(2x2)\n");
    printf("  Flatten -> Dense(3136->128) -> ReLU -> Dense(128->10) -> Softmax\n");

    return model;
}

void free_mnist_model(mnist_model_t* model) {
    if (!model) return;

    if (model->conv1) boat_conv_layer_free(model->conv1);
    if (model->conv2) boat_conv_layer_free(model->conv2);
    if (model->pool1) boat_pool_layer_free(model->pool1);
    if (model->pool2) boat_pool_layer_free(model->pool2);
    if (model->flatten) boat_flatten_layer_free(model->flatten);
    if (model->fc1) boat_dense_layer_free(model->fc1);
    if (model->fc2) boat_dense_layer_free(model->fc2);
    if (model->relu1) boat_relu_layer_free(model->relu1);
    if (model->relu2) boat_relu_layer_free(model->relu2);
    if (model->relu3) boat_relu_layer_free(model->relu3);
    if (model->softmax) boat_softmax_layer_free(model->softmax);

    free(model);
}

boat_tensor_t* forward_pass(mnist_model_t* model, boat_tensor_t* input) {
    boat_tensor_t* x = input;
    boat_tensor_t* tmp = NULL;

    // Conv1 -> ReLU -> Pool1
    x = boat_conv_layer_forward(model->conv1, x);
    if (!x) { fprintf(stderr, "conv1 forward failed\n"); return NULL; }
    tmp = boat_relu_layer_forward(model->relu1, x);
    if (!tmp) { fprintf(stderr, "relu1 forward failed\n"); return NULL; }
    boat_tensor_unref(x); x = tmp;
    tmp = boat_pool_layer_forward(model->pool1, x);
    if (!tmp) { fprintf(stderr, "pool1 forward failed\n"); return NULL; }
    boat_tensor_unref(x); x = tmp;

    // Conv2 -> ReLU -> Pool2
    tmp = boat_conv_layer_forward(model->conv2, x);
    if (!tmp) { fprintf(stderr, "conv2 forward failed\n"); return NULL; }
    boat_tensor_unref(x); x = tmp;
    tmp = boat_relu_layer_forward(model->relu2, x);
    if (!tmp) { fprintf(stderr, "relu2 forward failed\n"); return NULL; }
    boat_tensor_unref(x); x = tmp;
    tmp = boat_pool_layer_forward(model->pool2, x);
    if (!tmp) { fprintf(stderr, "pool2 forward failed\n"); return NULL; }
    boat_tensor_unref(x); x = tmp;

    // Flatten -> FC1 -> ReLU -> FC2 -> Softmax
    tmp = boat_flatten_layer_forward(model->flatten, x);
    if (!tmp) { fprintf(stderr, "flatten forward failed\n"); return NULL; }
    boat_tensor_unref(x); x = tmp;
    tmp = boat_dense_layer_forward(model->fc1, x);
    if (!tmp) { fprintf(stderr, "fc1 forward failed\n"); return NULL; }
    boat_tensor_unref(x); x = tmp;
    tmp = boat_relu_layer_forward(model->relu3, x);
    if (!tmp) { fprintf(stderr, "relu3 forward failed\n"); return NULL; }
    boat_tensor_unref(x); x = tmp;
    tmp = boat_dense_layer_forward(model->fc2, x);
    if (!tmp) { fprintf(stderr, "fc2 forward failed\n"); return NULL; }
    boat_tensor_unref(x); x = tmp;
    tmp = boat_softmax_layer_forward(model->softmax, x);
    if (!tmp) { fprintf(stderr, "softmax forward failed\n"); return NULL; }
    boat_tensor_unref(x); x = tmp;

    return x;
}

void backward_pass(mnist_model_t* model, boat_tensor_t* grad_output) {
    // Backward pass through layers in reverse order.
    // Convention: each backward function returns a tensor the caller owns.
    // For layer N's output (grad) going into layer N+1:
    //   out = backward_N+1(layer, grad);
    //   boat_tensor_unref(grad);  // release backward_pass's ref on input
    //   grad = out;               // adopt the returned ref for next layer
    // First call is special: grad_output is still owned by the caller,
    // softmax_backward adds a ref which backward_pass adopts.
    // Last call: unref both input and returned (conv1) gradient.
    boat_tensor_t* grad = grad_output;
    boat_tensor_t* out = NULL;

    // Layer 1: softmax (no unref — the caller still owns grad_output)
    out = boat_softmax_layer_backward(model->softmax, grad);
    if (!out) return;
    grad = out;

    out = boat_dense_layer_backward(model->fc2, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_relu_layer_backward(model->relu3, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_dense_layer_backward(model->fc1, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_flatten_layer_backward(model->flatten, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_pool_layer_backward(model->pool2, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_relu_layer_backward(model->relu2, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_conv_layer_backward(model->conv2, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_pool_layer_backward(model->pool1, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    out = boat_relu_layer_backward(model->relu1, grad);
    if (!out) { boat_tensor_unref(grad); return; }
    boat_tensor_unref(grad); grad = out;

    // Final layer: conv1. Unref both input and output gradients.
    out = boat_conv_layer_backward(model->conv1, grad);
    boat_tensor_unref(grad);
    if (out) boat_tensor_unref(out);
}

int main(int argc, char* argv[]) {
    printf("=== MNIST Digit Recognition with Boat Framework ===\n");

    // Disable stdout buffering so background/log output is visible in real time
    setvbuf(stdout, NULL, _IONBF, 0);

    // Check env var for quick test mode (small dataset, 1 epoch)
    const char* quick_test_env = getenv("MNIST_QUICK_TEST");
    int use_quick_test = quick_test_env && atoi(quick_test_env) == 1;

    // Check for data directory
    if (access("data", F_OK) == -1) {
        printf("Data directory not found. Please run 'python mnist_data.py' first.\n");
        return 1;
    }

    // Select data files based on test mode
    const char* train_images_file = use_quick_test
        ? "data/train_images_small.bin" : "data/train_images.bin";
    const char* train_labels_file = use_quick_test
        ? "data/train_labels_small.bin" : "data/train_labels.bin";
    const char* test_images_file = use_quick_test
        ? "data/test_images_small.bin" : "data/test_images.bin";
    const char* test_labels_file = use_quick_test
        ? "data/test_labels_small.bin" : "data/test_labels.bin";

    printf("Loading training data from %s...\n", train_images_file);
    boat_tensor_t* train_images = load_tensor_binary(train_images_file, BOAT_DTYPE_FLOAT32);
    boat_tensor_t* train_labels = load_tensor_binary(train_labels_file, BOAT_DTYPE_UINT8);

    printf("Loading test data from %s...\n", test_images_file);
    boat_tensor_t* test_images = load_tensor_binary(test_images_file, BOAT_DTYPE_FLOAT32);
    boat_tensor_t* test_labels = load_tensor_binary(test_labels_file, BOAT_DTYPE_UINT8);

    if (!train_images || !train_labels || !test_images || !test_labels) {
        fprintf(stderr, "Error loading data files\n");
        return 1;
    }

    const int64_t* train_shape = boat_tensor_shape(train_images);
    size_t train_samples = train_shape[0];
    printf("Training samples: %zu\n", train_samples);

    // Create model
    mnist_model_t* model = create_mnist_model();
    if (!model) {
        fprintf(stderr, "Failed to create model\n");
        return 1;
    }

    // Create Adam optimizer and register parameters
    boat_optimizer_t* optimizer = boat_adam_optimizer_create(0.001f, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) {
        fprintf(stderr, "Failed to create optimizer\n");
        free_mnist_model(model);
        return 1;
    }

    // Register all trainable parameters with their gradient tensors
    boat_optimizer_add_parameter(optimizer,
        boat_conv_layer_get_weight(model->conv1),
        boat_conv_layer_get_grad_weight(model->conv1));
    boat_optimizer_add_parameter(optimizer,
        boat_conv_layer_get_bias(model->conv1),
        boat_conv_layer_get_grad_bias(model->conv1));
    boat_optimizer_add_parameter(optimizer,
        boat_conv_layer_get_weight(model->conv2),
        boat_conv_layer_get_grad_weight(model->conv2));
    boat_optimizer_add_parameter(optimizer,
        boat_conv_layer_get_bias(model->conv2),
        boat_conv_layer_get_grad_bias(model->conv2));
    boat_optimizer_add_parameter(optimizer,
        boat_dense_layer_get_weight(model->fc1),
        boat_dense_layer_get_grad_weight(model->fc1));
    boat_optimizer_add_parameter(optimizer,
        boat_dense_layer_get_bias(model->fc1),
        boat_dense_layer_get_grad_bias(model->fc1));
    boat_optimizer_add_parameter(optimizer,
        boat_dense_layer_get_weight(model->fc2),
        boat_dense_layer_get_grad_weight(model->fc2));
    boat_optimizer_add_parameter(optimizer,
        boat_dense_layer_get_bias(model->fc2),
        boat_dense_layer_get_grad_bias(model->fc2));

    // Data standardization: compute mean and std from training set
    printf("Computing mean and std from training set...\n");
    float* train_data_ptr = (float*)boat_tensor_data(train_images);
    size_t train_total_pixels = train_shape[0] * train_shape[1] * train_shape[2] * train_shape[3];
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < train_total_pixels; i++) {
        sum += train_data_ptr[i];
        sum_sq += train_data_ptr[i] * train_data_ptr[i];
    }
    float mean = (float)(sum / train_total_pixels);
    float std = (float)sqrt(sum_sq / train_total_pixels - mean * mean);
    printf("Training set stats: mean=%.6f, std=%.6f\n", mean, std);
    for (size_t i = 0; i < train_total_pixels; i++) {
        train_data_ptr[i] = (train_data_ptr[i] - mean) / std;
    }
    // Also standardize test data
    const int64_t* test_shape_from_data = boat_tensor_shape(test_images);
    size_t test_total_pixels = test_shape_from_data[0] * test_shape_from_data[1] *
                               test_shape_from_data[2] * test_shape_from_data[3];
    float* test_data_ptr = (float*)boat_tensor_data(test_images);
    for (size_t i = 0; i < test_total_pixels; i++) {
        test_data_ptr[i] = (test_data_ptr[i] - mean) / std;
    }
    printf("Data standardization complete\n");

    // Training parameters
    int epochs = use_quick_test ? 1 : 10;
    float learning_rate = 0.001f;
    size_t batch_size = 32;
    size_t num_batches = train_samples / batch_size;

    // Create reusable batch input tensor
    int64_t batch_shape[] = {(int64_t)batch_size, 1, 28, 28};
    boat_tensor_t* batch_input = boat_tensor_create(batch_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!batch_input) {
        fprintf(stderr, "Failed to create batch input tensor\n");
        boat_optimizer_free(optimizer);
        free_mnist_model(model);
        return 1;
    }

    // Get direct pointers to training data for fast batch copy
    const float* train_images_data = (const float*)boat_tensor_const_data(train_images);
    const uint8_t* train_labels_data = (const uint8_t*)boat_tensor_const_data(train_labels);
    const float* test_images_data = (const float*)boat_tensor_const_data(test_images);
    const uint8_t* test_labels_data = (const uint8_t*)boat_tensor_const_data(test_labels);
    size_t sample_size = 28 * 28;

    printf("\nStarting training...\n");
    printf("Epochs: %d, Learning rate: %.4f, Batch size: %zu, Total batches per epoch: %zu\n",
           epochs, learning_rate, batch_size, num_batches);

    // Logging interval: report progress every ~10%
    size_t log_interval = num_batches / 10;
    if (log_interval < 1) log_interval = 1;

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        int epoch_correct = 0;
        int epoch_total = 0;

        clock_t start_time = clock();

        printf("  Epoch %d/%d: ", epoch + 1, epochs);
        fflush(stdout);

        // Batch iteration
        for (size_t batch = 0; batch < num_batches; batch++) {
            size_t start_idx = batch * batch_size;

            // Copy batch data into batch_input tensor (contiguous memcpy)
            float* batch_data = (float*)boat_tensor_data(batch_input);
            memcpy(batch_data, train_images_data + start_idx * sample_size,
                   batch_size * sample_size * sizeof(float));

            // Forward pass
            boat_tensor_t* output = forward_pass(model, batch_input);
            if (!output) {
                fprintf(stderr, "Forward pass failed at batch %zu\n", batch);
                continue;
            }

            // Compute accuracy for this batch
            const float* pred_data = (const float*)boat_tensor_const_data(output);
            for (size_t i = 0; i < batch_size; i++) {
                size_t base = i * 10;
                int pred_class = 0;
                float max_prob = pred_data[base];
                for (int j = 1; j < 10; j++) {
                    if (pred_data[base + j] > max_prob) {
                        max_prob = pred_data[base + j];
                        pred_class = j;
                    }
                }
                if (pred_class == train_labels_data[start_idx + i]) {
                    epoch_correct++;
                }
                epoch_total++;
            }

            // Compute gradient: grad = (pred - one_hot(label)) / batch_size
            // Cross-entropy with softmax output gives this simple gradient form
            int64_t grad_shape[] = {(int64_t)batch_size, 10};
            boat_tensor_t* grad_output = boat_tensor_create(grad_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
            if (grad_output) {
                float* grad_data = (float*)boat_tensor_data(grad_output);

                for (size_t i = 0; i < batch_size; i++) {
                    size_t base = i * 10;
                    for (int j = 0; j < 10; j++) {
                        grad_data[base + j] = pred_data[base + j];
                    }
                    grad_data[base + train_labels_data[start_idx + i]] -= 1.0f;
                }

                // Scale by 1/batch_size to compute mean gradient across batch
                float grad_scale = 1.0f / batch_size;
                size_t total_vals = batch_size * 10;
                for (size_t i = 0; i < total_vals; i++) {
                    grad_data[i] *= grad_scale;
                }

                backward_pass(model, grad_output);
                boat_tensor_unref(grad_output);
            }

            boat_tensor_unref(output);

            // Update model parameters
            boat_optimizer_step(optimizer);
            boat_optimizer_zero_grad(optimizer);

            // Progress logging
            if ((batch + 1) % log_interval == 0 || batch == num_batches - 1) {
                printf(".");
                fflush(stdout);
            }
        }

        clock_t end_time = clock();
        double epoch_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        float epoch_accuracy = epoch_total > 0 ? (float)epoch_correct / epoch_total : 0.0f;
        printf(" time=%.2fs, accuracy=%.2f%%\n",
               epoch_time, epoch_accuracy * 100.0f);
    }

    // Evaluate on test set in batches
    printf("\nEvaluating on test set...\n");
    const int64_t* test_images_shape = boat_tensor_shape(test_images);
    size_t test_samples = test_images_shape[0];

    int test_correct = 0;
    for (size_t start = 0; start < test_samples; start += batch_size) {
        size_t current_batch = batch_size;
        if (start + current_batch > test_samples) {
            current_batch = test_samples - start;
        }

        // Create input tensor for this batch
        int64_t eval_shape[] = {(int64_t)current_batch, 1, 28, 28};
        boat_tensor_t* eval_input = boat_tensor_create(eval_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!eval_input) continue;

        // Copy batch data
        float* eval_data = (float*)boat_tensor_data(eval_input);
        memcpy(eval_data, test_images_data + start * sample_size,
               current_batch * sample_size * sizeof(float));

        // Forward pass
        boat_tensor_t* output = forward_pass(model, eval_input);
        if (!output) {
            boat_tensor_unref(eval_input);
            continue;
        }

        // Compute accuracy
        const float* pred_data = (const float*)boat_tensor_const_data(output);
        for (size_t i = 0; i < current_batch; i++) {
            size_t base = i * 10;
            int pred_class = 0;
            float max_prob = pred_data[base];
            for (int j = 1; j < 10; j++) {
                if (pred_data[base + j] > max_prob) {
                    max_prob = pred_data[base + j];
                    pred_class = j;
                }
            }
            if (pred_class == test_labels_data[start + i]) {
                test_correct++;
            }
        }

        boat_tensor_unref(output);
        boat_tensor_unref(eval_input);
    }

    float test_accuracy = (float)test_correct / test_samples;
    printf("Test accuracy: %.2f%% (%d/%zu)\n", test_accuracy * 100.0f, test_correct, test_samples);

    // Cleanup
    boat_tensor_unref(batch_input);
    boat_optimizer_free(optimizer);
    free_mnist_model(model);
    boat_tensor_unref(train_images);
    boat_tensor_unref(train_labels);
    boat_tensor_unref(test_images);
    boat_tensor_unref(test_labels);

    printf("\nDone!\n");
    return 0;
}
