# Boat: Lightweight Deep Learning Framework in C

A minimal, high-performance deep learning framework written in pure C. Designed for inference, training, and fine-tuning of neural networks with a focus on simplicity, memory efficiency, and cross-platform compatibility.

## Features

- **Pure C implementation**: No external dependencies for core functionality
- **Memory efficient**: Explicit memory management with reference counting
- **Cross-platform**: Works on Windows, Linux, and macOS
- **Automatic differentiation**: Built-in computational graph for gradient computation
- **Modular architecture**: Easy to extend with new operations and layers
- **Multiple data types**: Support for FP32, FP64, INT32, INT64, and low-bit quantization
- **Shared library support**: Build as static or dynamic library (DLL on Windows)

## Project Structure

```
boat/
├── include/boat/          # Public headers
│   ├── tensor.h          # Tensor operations
│   ├── ops.h             # Mathematical operations
│   ├── autodiff.h        # Automatic differentiation
│   ├── graph.h           # Computational graph
│   ├── layers.h          # Neural network layers
│   ├── optimizers.h      # Optimization algorithms
│   ├── loss.h            # Loss functions
│   ├── model.h           # Model definition
│   ├── memory.h          # Memory management
│   └── export.h          # Cross-platform export macros
├── src/                  # Implementation
│   ├── core/            # Core functionality
│   ├── ops/             # Operations
│   ├── graph/           # Computational graph
│   ├── layers/          # Neural network layers
│   ├── model/           # Model management
│   ├── optimizers/      # Optimizers
│   ├── schedulers/      # Learning rate schedulers
│   └── loss/            # Loss functions
├── CMakeLists.txt       # Build configuration
├── LICENSE              # Apache License 2.0
├── NOTICE               # Attribution requirements
└── README.md            # This file
```

## Quick Start

### Building from Source

```bash
# Clone the repository
git clone <repository-url>
cd boat

# Configure and build
mkdir build
cd build
cmake ..
cmake --build . --config Release

# Install (optional)
cmake --install . --prefix /path/to/install
```

### Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `BOAT_BUILD_SHARED` | Build shared library (DLL on Windows) | ON |
| `BOAT_WITH_CUDA` | Enable CUDA support | OFF |
| `BOAT_WITH_TESTS` | Build tests | OFF |
| `BOAT_WITH_EXAMPLES` | Build examples | OFF |

Example with custom options:
```bash
cmake .. -DBOAT_BUILD_SHARED=ON -DBOAT_WITH_TESTS=OFF
```

## Basic Usage

### Creating a Simple Neural Network

```c
#include <boat/boat.h>
#include <boat/layers.h>
#include <boat/optimizers.h>
#include <boat/loss.h>

int main() {
    // Create a sequential model
    boat_sequential_model_t* model = boat_sequential_create();

    // Add layers
    boat_layer_t* dense1 = boat_dense_layer_create(784, 128);
    boat_layer_t* relu1 = boat_relu_layer_create();
    boat_layer_t* dense2 = boat_dense_layer_create(128, 10);
    boat_layer_t* softmax = boat_softmax_layer_create();

    boat_sequential_add(model, dense1);
    boat_sequential_add(model, relu1);
    boat_sequential_add(model, dense2);
    boat_sequential_add(model, softmax);

    // Create optimizer
    boat_optimizer_t* optimizer = boat_adam_optimizer_create(
        (boat_model_t*)model, 0.001);

    // Create loss function
    boat_loss_t* loss = boat_cross_entropy_loss_create();

    // Training loop example
    for (int epoch = 0; epoch < 10; epoch++) {
        // Forward pass
        boat_tensor_t* output = boat_model_forward(
            (boat_model_t*)model, input_tensor);

        // Compute loss
        float loss_value = boat_loss_compute(loss, output, target_tensor);

        // Backward pass
        boat_tensor_t* grad = boat_loss_backward(loss);
        boat_model_backward((boat_model_t*)model, grad);

        // Update parameters
        boat_optimizer_step(optimizer);

        printf("Epoch %d, Loss: %f\n", epoch, loss_value);
    }

    // Cleanup
    boat_optimizer_free(optimizer);
    boat_loss_free(loss);
    boat_model_free((boat_model_t*)model);

    return 0;
}
```

### Tensor Operations

```c
#include <boat/tensor.h>

// Create a tensor
int64_t shape[] = {2, 3, 4};
boat_tensor_t* tensor = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32);

// Access tensor data
float* data = (float*)boat_tensor_data(tensor);
size_t nbytes = boat_tensor_nbytes(tensor);

// Reshape tensor
int64_t new_shape[] = {6, 4};
boat_tensor_t* reshaped = boat_tensor_reshape(tensor, new_shape, 2);

// Memory management
boat_tensor_ref(tensor);      // Increase reference count
boat_tensor_unref(tensor);    // Decrease reference count (frees if zero)
boat_tensor_unref(reshaped);
```

### Automatic Differentiation

```c
#include <boat/autodiff.h>

// Create variables that require gradients
boat_variable_t* x = boat_variable_create(tensor_x, true);
boat_variable_t* y = boat_variable_create(tensor_y, true);

// Perform operations (tracked in computational graph)
boat_variable_t* z = boat_add(x, y);
boat_variable_t* w = boat_mul(z, x);

// Compute gradients
boat_backward(w);

// Access gradients
boat_tensor_t* grad_x = boat_variable_grad(x);
boat_tensor_t* grad_y = boat_variable_grad(y);

// Cleanup
boat_variable_free(x);
boat_variable_free(y);
boat_variable_free(z);
boat_variable_free(w);
```

## Core Components

### Tensors
- Multi-dimensional arrays with various data types
- Shape manipulation (reshape, transpose, slice)
- Memory-efficient storage with reference counting

### Layers
- **Dense**: Fully connected layer
- **Conv2D**: Convolutional layer
- **BatchNorm**: Batch normalization
- **LayerNorm**: Layer normalization
- **Attention**: Self-attention mechanism
- **ReLU, Sigmoid, Tanh**: Activation functions
- **Softmax, LogSoftmax**: Output layers

### Optimizers
- **SGD**: Stochastic Gradient Descent
- **Adam**: Adaptive Moment Estimation
- **RMSprop**: Root Mean Square Propagation
- **Adagrad**: Adaptive Gradient Algorithm

### Loss Functions
- **MSE**: Mean Squared Error
- **CrossEntropy**: Cross Entropy Loss
- **Huber**: Huber Loss (smooth L1)

### Computational Graph
- Automatic construction during forward pass
- Efficient gradient computation during backward pass
- Support for checkpointing and memory optimization

## API Reference

### Tensor API
```c
boat_tensor_t* boat_tensor_create(const int64_t* shape, size_t ndim, boat_dtype_t dtype);
boat_tensor_t* boat_tensor_from_data(const int64_t* shape, size_t ndim, boat_dtype_t dtype, const void* data);
void boat_tensor_unref(boat_tensor_t* tensor);
boat_tensor_t* boat_tensor_reshape(const boat_tensor_t* tensor, const int64_t* new_shape, size_t new_ndim);
boat_tensor_t* boat_tensor_transpose(const boat_tensor_t* tensor, const size_t* perm, size_t nperm);
```

### Model API
```c
boat_model_t* boat_model_create();
void boat_model_free(boat_model_t* model);
void boat_model_add_layer(boat_model_t* model, boat_layer_t* layer);
boat_tensor_t* boat_model_forward(boat_model_t* model, const boat_tensor_t* input);
boat_tensor_t* boat_model_backward(boat_model_t* model, const boat_tensor_t* grad_output);
void boat_model_update(boat_model_t* model, float learning_rate);
```

### Layer API
```c
boat_layer_t* boat_dense_layer_create(size_t input_features, size_t output_features);
boat_layer_t* boat_conv_layer_create(size_t in_channels, size_t out_channels, size_t kernel_size);
boat_layer_t* boat_norm_layer_create(size_t normalized_shape);
boat_layer_t* boat_attention_layer_create(size_t embed_dim, size_t num_heads);
```

### Optimizer API
```c
boat_optimizer_t* boat_sgd_optimizer_create(boat_model_t* model, float learning_rate);
boat_optimizer_t* boat_adam_optimizer_create(boat_model_t* model, float learning_rate);
void boat_optimizer_step(boat_optimizer_t* optimizer);
void boat_optimizer_zero_grad(boat_optimizer_t* optimizer);
```

## Platform Compatibility

### Windows
- Builds as DLL with `BOAT_BUILD_SHARED=ON`
- MSVC compiler supported with compatibility optimizations
- Automatic export of all symbols

### Linux/macOS
- Builds as shared library (.so) or static library (.a)
- GCC and Clang supported
- Position independent code enabled by default

### Cross-Platform Macros
The framework provides macros for cross-platform compatibility:
- `BOAT_API`: Marks functions for export/import
- `BOAT_CALL`: Ensures correct calling convention
- `BOAT_NOINLINE`: Prevents compiler optimization issues in DLLs

## Memory Management

- **Explicit ownership**: Each tensor has a reference count
- **Automatic cleanup**: Objects are freed when reference count reaches zero
- **Arena allocator**: Optional for efficient allocation of small tensors
- **Memory statistics**: Track allocation/deallocation for debugging

## Error Handling

```c
#include <boat/error.h>

// Check for errors
if (boat_has_error()) {
    const char* msg = boat_get_last_error_message();
    fprintf(stderr, "Error: %s\n", msg);
    boat_clear_error();
}

// Set custom errors
boat_set_errorf("Invalid tensor shape: %zux%zu", rows, cols);
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the code style (4 spaces, snake_case, 100 char line limit)
4. Write tests for new functionality
5. Submit a pull request

## Author

**Shaoning, Xiao (萧少宁)** - Framework designer and main developer

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

### Attribution Requirements

This software includes a [NOTICE](NOTICE) file that must be included with any redistribution, as required by the Apache License 2.0. The NOTICE file contains attribution requirements that must be respected.

Key requirements:
- Retain all copyright notices in source files
- Include attribution to Shaoning, Xiao (萧少宁) in documentation
- Clearly indicate any modifications made to the original code
- Include the NOTICE file in any redistribution

## Acknowledgments

- Inspired by PyTorch's dynamic computation graphs
- Designed for simplicity like Caffe's C++ implementation
- Focus on production deployment like TensorFlow

## Support

For issues and questions:
- Check existing documentation
- Review header files for API details
- Examine example implementations