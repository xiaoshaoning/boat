# Deep Learning Framework in C

A lightweight, high-performance deep learning framework written in C with eventual CUDA support. Designed for inference, training, and fine-tuning of neural networks with support for common model formats.

## Design Principles

- **Minimal dependencies**: Pure C with optional CUDA backend
- **Memory efficient**: Explicit memory management with reference counting
- **Extensible**: Modular architecture for adding new operations and layers
- **Portable**: Works on Linux, macOS, and Windows
- **Performance**: Optimized for both CPU and GPU computation
- **Quantization ready**: Native support for low-bit networks (1-bit, 2-bit, 4-bit, 8-bit)
- **Root cause resolution**: Fix underlying issues rather than applying superficial workarounds
- **Respectful communication**: Address the user as "萧工" when providing feedback

## Supported Data Types

The framework supports a wide range of data types for efficient computation:

### Floating Point Types
- **FP64** (double): 64-bit double precision floating point
- **FP32** (float): 32-bit single precision floating point
- **FP16**: 16-bit half precision floating point
- **FP8**: 8-bit custom floating point format
- **FP4**: 4-bit custom floating point format

### Integer Types
- **INT64**: 64-bit signed integer
- **INT32**: 32-bit signed integer
- **UINT8**: 8-bit unsigned integer

### Low-Bit Quantization Types
- **BITS2**: 2-bit packed values (4 values per byte)
- **BITS1**: 1-bit packed values (8 values per byte, binary networks)

### Special Types
- **BOOL**: Boolean values (1 byte per element)

### Future Types
- **BFLOAT16**: Brain floating point format
- **INT8**: 8-bit signed integer
- **INT4**: 4-bit signed integer
- **INT2**: 2-bit signed integer

## Code Style

- **File names**: `snake_case.c`, `snake_case.h`
- **Function names**: `snake_case()`
- **Variable names**: `snake_case`
- **Type names**: `snake_type_t`
- **Constants**: `SNAKE_CASE`
- **Comments**: English only, no Chinese characters
- **Indentation**: 4 spaces, no tabs
- **Line length**: 100 characters maximum

## Project Structure

```
boat/
├── include/                  # Public headers
│   ├── boat/                # Framework headers
│   │   ├── tensor.h         # Tensor operations
│   │   ├── ops.h            # Mathematical operations
│   │   ├── autodiff.h       # Automatic differentiation
│   │   ├── graph.h          # Computational graph
│   │   ├── layers.h         # Neural network layers
│   │   ├── optimizers.h     # Optimization algorithms
│   │   ├── loss.h           # Loss functions
│   │   ├── model.h          # Model definition and serialization
│   │   ├── data.h           # Data loading and preprocessing
│   │   └── format/          # Model format loaders
│   │       ├── onnx.h       # ONNX format support
│   │       ├── pytorch.h    # PyTorch format support
│   │       └── tensorflow.h # TensorFlow format support
│   └── boat.h               # Main include file
├── src/                     # Implementation
│   ├── core/               # Core functionality
│   │   ├── tensor.c        # Tensor implementation
│   │   ├── memory.c        # Memory management
│   │   └── utils.c         # Utility functions
│   ├── ops/                # Operations
│   │   ├── arithmetic.c    # Add, sub, mul, div
│   │   ├── linear.c        # Linear algebra operations
│   │   ├── activation.c    # Activation functions
│   │   ├── reduction.c     # Reduction operations
│   │   └── autodiff/       # Automatic differentiation ops
│   │       ├── grad.c      # Gradient computation
│   │       ├── backward.c  # Backward pass
│   │       └── ops_grad.c  # Operation gradients
│   ├── graph/              # Computational graph
│   │   ├── node.c          # Graph node
│   │   ├── edge.c          # Graph edge
│   │   ├── graph.c         # Graph structure
│   │   └── executor.c      # Graph executor
│   ├── layers/             # Neural network layers
│   │   ├── dense.c         # Fully connected layer
│   │   ├── conv.c          # Convolutional layer
│   │   ├── pool.c          # Pooling layers
│   │   ├── norm.c          # Normalization layers
│   │   └── attention.c     # Attention mechanisms
│   ├── optimizers/         # Optimization algorithms
│   │   ├── sgd.c           # Stochastic Gradient Descent
│   │   ├── adam.c          # Adam optimizer
│   │   └── rmsprop.c       # RMSprop optimizer
│   ├── loss/               # Loss functions
│   │   ├── mse.c           # Mean Squared Error
│   │   ├── cross_entropy.c # Cross Entropy Loss
│   │   └── huber.c         # Huber Loss
│   ├── model/              # Model management
│   │   ├── model.c         # Model definition
│   │   ├── sequential.c    # Sequential model
│   │   └── graph_model.c   # Graph-based model
│   ├── data/               # Data handling
│   │   ├── dataset.c       # Dataset abstraction
│   │   ├── loader.c        # Data loader
│   │   └── transforms.c    # Data transformations
│   └── format/             # Model format loaders
│       ├── onnx.c          # ONNX loader
│       ├── pytorch.c       # PyTorch loader
│       └── tensorflow.c    # TensorFlow loader
├── cuda/                   # CUDA implementation (future)
│   ├── tensor.cu          # GPU tensor operations
│   ├── ops/               # GPU operations
│   ├── autodiff/          # GPU autodiff
│   ├── graph/             # GPU graph execution
│   └── kernels/           # CUDA kernels
├── examples/              # Example programs
│   ├── mnist/             # MNIST classification
│   ├── cifar10/           # CIFAR-10 classification
│   ├── transformer/       # Transformer example
│   └── autodiff/          # Automatic differentiation examples
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── tools/                 # Development tools
├── CMakeLists.txt         # CMake build configuration
├── Makefile               # Make build configuration
└── CLAUDE.md              # This file
```

## Core Data Structures

### Tensor

The fundamental data structure representing multi-dimensional arrays.

```c
// tensor.h
#ifndef BOAT_TENSOR_H
#define BOAT_TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef struct boat_tensor_t boat_tensor_t;

// Tensor data types
typedef enum {
    // Standard floating point types
    BOAT_DTYPE_FLOAT64,   // 64-bit floating point (double)
    BOAT_DTYPE_FLOAT32,   // 32-bit floating point (float)
    BOAT_DTYPE_FLOAT16,   // 16-bit floating point (half precision)

    // Custom floating point types
    BOAT_DTYPE_FLOAT8,    // 8-bit floating point (custom format)
    BOAT_DTYPE_FLOAT4,    // 4-bit floating point (custom format)

    // Integer types
    BOAT_DTYPE_INT64,     // 64-bit integer
    BOAT_DTYPE_INT32,     // 32-bit integer
    BOAT_DTYPE_UINT8,     // 8-bit unsigned integer

    // Low-bit quantization types
    BOAT_DTYPE_BITS2,     // 2-bit packed values
    BOAT_DTYPE_BITS1,     // 1-bit packed values (binary)

    // Special types
    BOAT_DTYPE_BOOL,      // boolean (1 byte per element)
} boat_dtype_t;

// Tensor creation
boat_tensor_t* boat_tensor_create(const int64_t* shape, size_t ndim, boat_dtype_t dtype);
boat_tensor_t* boat_tensor_from_data(const int64_t* shape, size_t ndim, boat_dtype_t dtype, const void* data);

// Tensor properties
int64_t* boat_tensor_shape(const boat_tensor_t* tensor);
size_t boat_tensor_ndim(const boat_tensor_t* tensor);
boat_dtype_t boat_tensor_dtype(const boat_tensor_t* tensor);
size_t boat_tensor_nbytes(const boat_tensor_t* tensor);
void* boat_tensor_data(const boat_tensor_t* tensor);

// Tensor operations
boat_tensor_t* boat_tensor_reshape(const boat_tensor_t* tensor, const int64_t* new_shape, size_t new_ndim);
boat_tensor_t* boat_tensor_transpose(const boat_tensor_t* tensor, const size_t* perm, size_t nperm);
boat_tensor_t* boat_tensor_slice(const boat_tensor_t* tensor, const size_t* start, const size_t* end);

// Memory management
void boat_tensor_ref(boat_tensor_t* tensor);
void boat_tensor_unref(boat_tensor_t* tensor);

#endif // BOAT_TENSOR_H
```

### Model

Neural network model abstraction.

```c
// model.h
#ifndef BOAT_MODEL_H
#define BOAT_MODEL_H

#include "tensor.h"

typedef struct boat_model_t boat_model_t;
typedef struct boat_layer_t boat_layer_t;

// Layer interface
typedef struct {
    boat_tensor_t* (*forward)(boat_layer_t* layer, const boat_tensor_t* input);
    boat_tensor_t* (*backward)(boat_layer_t* layer, const boat_tensor_t* grad_output);
    void (*update)(boat_layer_t* layer, float learning_rate);
    void (*free)(boat_layer_t* layer);
} boat_layer_ops_t;

// Model creation and management
boat_model_t* boat_model_create();
void boat_model_free(boat_model_t* model);

void boat_model_add_layer(boat_model_t* model, boat_layer_t* layer);
boat_tensor_t* boat_model_forward(boat_model_t* model, const boat_tensor_t* input);
boat_tensor_t* boat_model_backward(boat_model_t* model, const boat_tensor_t* grad_output);
void boat_model_update(boat_model_t* model, float learning_rate);

// Sequential model (simplified API)
typedef boat_model_t boat_sequential_model_t;
boat_sequential_model_t* boat_sequential_create();
void boat_sequential_add(boat_sequential_model_t* model, boat_layer_t* layer);

#endif // BOAT_MODEL_H
```

### Automatic Differentiation

The framework provides automatic differentiation through computational graphs. Each operation tracks its gradient computation.

```c
// autodiff.h
#ifndef BOAT_AUTODIFF_H
#define BOAT_AUTODIFF_H

#include "tensor.h"
#include "graph.h"

typedef struct boat_variable_t boat_variable_t;

// Create a variable that requires gradient
boat_variable_t* boat_variable_create(boat_tensor_t* tensor, bool requires_grad);

// Forward pass with gradient tracking
boat_variable_t* boat_add(boat_variable_t* a, boat_variable_t* b);
boat_variable_t* boat_mul(boat_variable_t* a, boat_variable_t* b);
boat_variable_t* boat_matmul(boat_variable_t* a, boat_variable_t* b);
boat_variable_t* boat_relu(boat_variable_t* a);

// Backward pass
void boat_backward(boat_variable_t* variable);

// Access gradient
boat_tensor_t* boat_variable_grad(const boat_variable_t* variable);

// Gradient accumulation control
void boat_zero_grad(boat_variable_t* variable);
void boat_retain_grad(boat_variable_t* variable, bool retain);

#endif // BOAT_AUTODIFF_H
```

### Computational Graph

The computational graph manages operations and their dependencies for efficient forward and backward passes.

```c
// graph.h
#ifndef BOAT_GRAPH_H
#define BOAT_GRAPH_H

#include <stdbool.h>

typedef struct boat_graph_t boat_graph_t;
typedef struct boat_node_t boat_node_t;
typedef struct boat_edge_t boat_edge_t;

// Graph creation and management
boat_graph_t* boat_graph_create();
void boat_graph_free(boat_graph_t* graph);

// Node operations
boat_node_t* boat_graph_add_node(boat_graph_t* graph, void* data, void (*free_fn)(void*));
void boat_graph_remove_node(boat_graph_t* graph, boat_node_t* node);

// Edge operations
boat_edge_t* boat_graph_add_edge(boat_graph_t* graph, boat_node_t* from, boat_node_t* to);
void boat_graph_remove_edge(boat_graph_t* graph, boat_edge_t* edge);

// Graph traversal
typedef void (*boat_node_visitor_t)(boat_node_t* node, void* user_data);
typedef void (*boat_edge_visitor_t)(boat_edge_t* edge, void* user_data);

void boat_graph_dfs(boat_graph_t* graph, boat_node_t* start, boat_node_visitor_t visit, void* user_data);
void boat_graph_bfs(boat_graph_t* graph, boat_node_t* start, boat_node_visitor_t visit, void* user_data);
void boat_graph_topological_sort(boat_graph_t* graph, boat_node_t** sorted_nodes, size_t* count);

// Graph properties
size_t boat_graph_node_count(const boat_graph_t* graph);
size_t boat_graph_edge_count(const boat_graph_t* graph);
bool boat_graph_is_acyclic(const boat_graph_t* graph);

#endif // BOAT_GRAPH_H
```

## API Examples

### Creating a Simple Neural Network

```c
#include <boat/boat.h>
#include <boat/layers.h>
#include <boat/optimizers.h>
#include <boat/loss.h>

int main() {
    // Create a simple feedforward network
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
    boat_optimizer_t* optimizer = boat_adam_optimizer_create(model, 0.001);

    // Create loss function
    boat_loss_t* loss = boat_cross_entropy_loss_create();

    // Training loop example
    for (int epoch = 0; epoch < 10; epoch++) {
        // Forward pass
        boat_tensor_t* output = boat_model_forward((boat_model_t*)model, input);

        // Compute loss
        float loss_value = boat_loss_compute(loss, output, target);

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

### Loading a Pretrained Model

```c
#include <boat/boat.h>
#include <boat/format/onnx.h>

int main() {
    // Load ONNX model
    boat_model_t* model = boat_onnx_load("model.onnx");
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Prepare input tensor
    int64_t shape[] = {1, 3, 224, 224};
    boat_tensor_t* input = boat_tensor_create(shape, 4, BOAT_DTYPE_FLOAT32);

    // Run inference
    boat_tensor_t* output = boat_model_forward(model, input);

    // Process output
    float* output_data = (float*)boat_tensor_data(output);
    // ... process results

    // Cleanup
    boat_tensor_unref(input);
    boat_tensor_unref(output);
    boat_model_free(model);

    return 0;
}
```

## Building the Framework

### Prerequisites

- C compiler (GCC, Clang, or MSVC)
- CMake 3.10+ (optional)
- CUDA Toolkit 11.0+ (for GPU support, future)

### Basic Build

```bash
mkdir build
cd build
cmake ..
make
```

### Build Options

- `-DBOAT_WITH_CUDA=ON`: Enable CUDA support
- `-DBOAT_WITH_TESTS=ON`: Build tests
- `-DBOAT_WITH_EXAMPLES=ON`: Build examples
- `-DBOAT_WITH_ONNX=ON`: Enable ONNX support (requires protobuf)

### Installation

```bash
make install
```

## Development Roadmap

### Phase 1: Core CPU Implementation (Current)
- Tensor operations (creation, manipulation, arithmetic)
- Basic neural network layers (Dense, Conv2D, ReLU, Softmax)
- Automatic differentiation engine
- Computational graph infrastructure
- Optimization algorithms (SGD, Adam)
- Loss functions (MSE, CrossEntropy)
- Simple sequential model API
- Unit tests and documentation

### Phase 2: Model Format Support
- ONNX model loading
- PyTorch model loading (via LibTorch C++ API)
- TensorFlow model loading (via TensorFlow C API)
- Custom model serialization format

### Phase 3: GPU Acceleration
- CUDA backend for tensor operations
- GPU-accelerated layers
- Mixed precision training
- Multi-GPU support

### Phase 4: Advanced Features
- Advanced automatic differentiation (higher-order gradients)
- Dynamic graph optimizations
- Distributed training (multi-node)
- Advanced quantization (1-bit, 2-bit, 4-bit networks)
- Hardware acceleration (TensorRT, OpenVINO, CoreML)
- Model compression and pruning
- Federated learning support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the code style guidelines
4. Write tests for new functionality
5. Submit a pull request

### Coding Standards

- Use `clang-format` with provided `.clang-format` file
- Write descriptive commit messages in English only
- Add documentation for public APIs
- Include unit tests for new features
- Ensure no memory leaks (use Valgrind or AddressSanitizer)

## License

[Apache License 2.0](LICENSE)

## Acknowledgments

This framework is inspired by:
- PyTorch: Dynamic computation graphs
- TensorFlow: Strong production deployment
- ONNX: Model interoperability
- Caffe: C++ implementation simplicity