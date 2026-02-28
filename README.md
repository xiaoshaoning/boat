# Boat: Deep Learning Framework in C

Boat is a lightweight, high-performance deep learning framework written in pure C with eventual CUDA support. Designed for inference, training, and fine-tuning of neural networks with support for common model formats.

## Key Features

- **Pure C Implementation**: Minimal dependencies, easy integration into existing C/C++ projects
- **Automatic Differentiation**: Computational graph-based autodiff with gradient tracking
- **Comprehensive Data Type Support**:
  - Floating point: FP64, FP32, FP16, FP8, FP4
  - Integer: INT64, INT32, UINT8
  - Low-bit quantization: BITS2 (2-bit), BITS1 (1-bit binary networks)
  - Boolean: BOOL type
- **Model Format Support**: ONNX, PyTorch, TensorFlow, HuggingFace Safetensors
- **GPU Acceleration**: CUDA backend support (planned)
- **Quantization Ready**: Native support for low-bit networks (1-bit, 2-bit, 4-bit, 8-bit)
- **Memory Efficient**: Explicit memory management with reference counting
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **Extensible Architecture**: Modular design for adding new operations and layers

## Design Principles

- **Minimal Dependencies**: Pure C with optional CUDA backend
- **Memory Efficient**: Explicit memory management with reference counting
- **Extensible**: Modular architecture for adding new operations and layers
- **Portable**: Works on Linux, macOS, and Windows
- **Performance**: Optimized for both CPU and GPU computation
- **Quantization Ready**: Native support for low-bit networks

## Installation

### Prerequisites

- C compiler (GCC, Clang, or MSVC)
- CMake 3.10+ (recommended)
- Git

### Building from Source

```bash
# Clone the repository
git clone https://github.com/xiaoshaoning/boat.git
cd boat

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the library
make

# (Optional) Install system-wide
sudo make install
```

### Build Options

- `-DBOAT_WITH_TESTS=ON`: Build test suite
- `-DBOAT_WITH_EXAMPLES=ON`: Build example programs
- `-DBOAT_WITH_ONNX=ON`: Enable ONNX support (requires protobuf)

## Quick Start

### Basic Tensor Operations

```c
#include <boat/boat.h>
#include <boat/tensor.h>

int main() {
    boat_init();

    // Create a tensor
    int64_t shape[] = {2, 3};
    boat_tensor_t* tensor = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32);

    // Access tensor properties
    size_t ndim = boat_tensor_ndim(tensor);
    int64_t* tensor_shape = boat_tensor_shape(tensor);
    boat_dtype_t dtype = boat_tensor_dtype(tensor);

    // Perform operations
    boat_tensor_t* transposed = boat_tensor_transpose(tensor, NULL, 0);

    // Cleanup
    boat_tensor_unref(tensor);
    boat_tensor_unref(transposed);
    boat_cleanup();

    return 0;
}
```

### Neural Network Training Example

```c
#include <boat/boat.h>
#include <boat/layers.h>
#include <boat/optimizers.h>
#include <boat/loss.h>

int main() {
    boat_init();

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
    boat_optimizer_t* optimizer = boat_adam_optimizer_create((boat_model_t*)model, 0.001);

    // Create loss function
    boat_loss_t* loss = boat_cross_entropy_loss_create();

    // Training loop (simplified)
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
    boat_cleanup();

    return 0;
}
```

### Automatic Differentiation Example

```c
#include <boat/boat.h>
#include <boat/autodiff.h>

int main() {
    boat_init();

    // Create variables with gradient tracking
    boat_tensor_t* tensor_a = boat_tensor_from_data((int64_t[]){2, 2}, 2, BOAT_DTYPE_FLOAT32, data_a);
    boat_tensor_t* tensor_b = boat_tensor_from_data((int64_t[]){2, 2}, 2, BOAT_DTYPE_FLOAT32, data_b);

    boat_variable_t* a = boat_variable_create(tensor_a, true);
    boat_variable_t* b = boat_variable_create(tensor_b, true);

    // Perform operations with gradient tracking
    boat_variable_t* c = boat_add(a, b);
    boat_variable_t* d = boat_mul(c, a);
    boat_variable_t* e = boat_relu(d);

    // Compute gradients
    boat_backward(e);

    // Access gradients
    boat_tensor_t* grad_a = boat_variable_grad(a);
    boat_tensor_t* grad_b = boat_variable_grad(b);

    // Cleanup
    boat_variable_free(a);
    boat_variable_free(b);
    boat_variable_free(c);
    boat_variable_free(d);
    boat_variable_free(e);
    boat_cleanup();

    return 0;
}
```

## MNIST Example

Boat includes a complete MNIST digit recognition example that demonstrates the framework's capabilities for computer vision tasks.

### Model Architecture

A convolutional neural network (CNN) for MNIST classification:

```
Input: 1x28x28 (channels x height x width)
├── Conv2D(32, kernel_size=3x3, padding=1)
├── ReLU()
├── MaxPool2D(kernel_size=2x2, stride=2)
├── Conv2D(64, kernel_size=3x3, padding=1)
├── ReLU()
├── MaxPool2D(kernel_size=2x2, stride=2)
├── Flatten()
├── Dense(128)
├── ReLU()
├── Dense(10)
└── Softmax()
```

### Running the MNIST Example

```bash
# Navigate to the MNIST example directory
cd examples/mnist

# Prepare the data (requires Python 3.x)
python mnist_data.py

# Build the example
mkdir build
cd build
cmake ..
make

# Run the training and evaluation
./mnist
```

### Key Code Snippets

**Model Creation:**
```c
// Create a convolutional neural network for MNIST
boat_sequential_model_t* model = boat_sequential_create();

// Add convolutional layers
boat_layer_t* conv1 = boat_conv_layer_create(1, 32, 3, 3, 1, 1, 1, 1);
boat_layer_t* relu1 = boat_relu_layer_create();
boat_layer_t* pool1 = boat_pool_layer_create(BOAT_POOL_MAX, 2, 2, 2, 2, 0, 0);

boat_layer_t* conv2 = boat_conv_layer_create(32, 64, 3, 3, 1, 1, 1, 1);
boat_layer_t* relu2 = boat_relu_layer_create();
boat_layer_t* pool2 = boat_pool_layer_create(BOAT_POOL_MAX, 2, 2, 2, 2, 0, 0);

// Add fully connected layers
boat_layer_t* flatten = boat_flatten_layer_create();
boat_layer_t* fc1 = boat_dense_layer_create(7*7*64, 128);  // After two 2x2 poolings: 28/2/2 = 7
boat_layer_t* relu3 = boat_relu_layer_create();
boat_layer_t* fc2 = boat_dense_layer_create(128, 10);
boat_layer_t* softmax = boat_softmax_layer_create();

// Build the sequential model
boat_sequential_add(model, conv1);
boat_sequential_add(model, relu1);
boat_sequential_add(model, pool1);
boat_sequential_add(model, conv2);
boat_sequential_add(model, relu2);
boat_sequential_add(model, pool2);
boat_sequential_add(model, flatten);
boat_sequential_add(model, fc1);
boat_sequential_add(model, relu3);
boat_sequential_add(model, fc2);
boat_sequential_add(model, softmax);
```

**Training Loop:**
```c
// Create optimizer and loss function
boat_optimizer_t* optimizer = boat_adam_optimizer_create((boat_model_t*)model, 0.001);
boat_loss_t* loss = boat_cross_entropy_loss_create();

// Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    float epoch_loss = 0.0f;
    int correct = 0;

    for (int batch = 0; batch < num_batches; batch++) {
        // Get batch data
        boat_tensor_t* batch_images = get_batch_images(batch);
        boat_tensor_t* batch_labels = get_batch_labels(batch);

        // Forward pass
        boat_tensor_t* predictions = boat_model_forward((boat_model_t*)model, batch_images);

        // Compute loss
        float batch_loss = boat_loss_compute(loss, predictions, batch_labels);
        epoch_loss += batch_loss;

        // Compute accuracy
        correct += compute_correct_predictions(predictions, batch_labels);

        // Backward pass
        boat_tensor_t* grad = boat_loss_backward(loss);
        boat_model_backward((boat_model_t*)model, grad);

        // Update parameters
        boat_optimizer_step(optimizer);

        // Cleanup
        boat_tensor_unref(predictions);
        boat_tensor_unref(grad);
    }

    // Compute epoch statistics
    float accuracy = (float)correct / (num_batches * batch_size);
    printf("Epoch %d: Loss = %.4f, Accuracy = %.2f%%\n",
           epoch + 1, epoch_loss / num_batches, accuracy * 100.0f);
}
```

### Expected Results

With proper training, the MNIST example should achieve:
- Training accuracy: >95%
- Test accuracy: >95%
- Reasonable training time (minutes on CPU)

### Data Preparation

The `mnist_data.py` script downloads and preprocesses the MNIST dataset:

```python
import mnist
import numpy as np
import struct

# Load MNIST data
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize to [0, 1] range
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Reshape for Boat (N, C, H, W format)
train_images = train_images.reshape(-1, 1, 28, 28)
test_images = test_images.reshape(-1, 1, 28, 28)

# Save as binary files for C consumption
save_tensor_binary("train_images.bin", train_images)
save_tensor_binary("train_labels.bin", train_labels.reshape(-1, 1))
save_tensor_binary("test_images.bin", test_images)
save_tensor_binary("test_labels.bin", test_labels.reshape(-1, 1))
```

For more details, see the [MNIST example documentation](examples/mnist/CLAUDE.md).

## Core Components

### Tensor Operations
- Creation and manipulation of multi-dimensional arrays
- Reshape, transpose, slice operations
- Arithmetic operations (add, sub, mul, div)
- Linear algebra operations (matmul, dot product)
- Reduction operations (sum, mean, max, min)

### Neural Network Layers
- **Dense**: Fully connected layer
- **Conv2D**: 2D convolutional layer
- **Pooling**: MaxPool2D, AvgPool2D
- **Normalization**: BatchNorm, LayerNorm
- **Activation**: ReLU, Sigmoid, Tanh, Softmax
- **Attention**: Multi-head self-attention
- **RNN Layers**: LSTM, GRU

### Optimization Algorithms
- Stochastic Gradient Descent (SGD)
- Adam optimizer
- RMSprop optimizer
- Adagrad optimizer

### Loss Functions
- Mean Squared Error (MSE)
- Cross Entropy Loss
- Huber Loss

### Model Management
- Sequential model API
- Graph-based model definition
- Model serialization and loading

## Data Types

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

## Examples

The repository includes several comprehensive examples:

- **MNIST Classification**: Complete training pipeline for digit recognition
- **CIFAR-10**: Image classification example
- **Transformer**: Attention mechanism implementation
- **Automatic Differentiation**: Gradient computation examples
- **Scheduler Usage**: Learning rate scheduling examples

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
│   │       ├── tensorflow.h # TensorFlow format support
│   │       └── huggingface.h# HuggingFace format support
│   └── boat.h               # Main include file
├── src/                     # Implementation
│   ├── core/               # Core functionality
│   ├── ops/                # Operations
│   ├── graph/              # Computational graph
│   ├── layers/             # Neural network layers
│   ├── optimizers/         # Optimization algorithms
│   ├── loss/               # Loss functions
│   ├── model/              # Model management
│   ├── data/               # Data handling
│   └── format/             # Model format loaders
├── examples/               # Example programs
│   ├── mnist/             # MNIST classification
│   ├── cifar10/           # CIFAR-10 classification
│   ├── transformer/       # Transformer example
│   └── autodiff/          # Automatic differentiation examples
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── tools/                 # Development tools
├── benchmarks/            # Performance benchmarks
├── docs/                  # Documentation
└── scripts/               # Utility scripts
```

For detailed API documentation and development guidelines, see [CLAUDE.md](CLAUDE.md).

## Development Status

### Current Features (Implemented)
- Core tensor operations with multiple data types
- Automatic differentiation with computational graph
- Neural network layers (dense, conv, attention, etc.)
- Optimizers (Adam, RMSprop, SGD, Adagrad)
- Loss functions (MSE, cross-entropy, Huber)
- Sequential model API
- Model format loaders (ONNX, PyTorch, TensorFlow, HuggingFace)
- Cross-platform build with CMake
- Comprehensive test suite
- MNIST training example

### Planned Features
- CUDA backend for GPU acceleration
- Advanced quantization techniques
- Distributed training support
- Model compression and pruning
- Hardware acceleration (TensorRT, OpenVINO, CoreML)
- More model format support
- Advanced layers (transformer blocks, etc.)

## Testing

Run the test suite to verify the installation:

```bash
cd build
make test
```

Or run specific tests:

```bash
ctest -R test_tensor          # Run tensor tests
ctest -R test_autodiff        # Run autodiff tests
ctest -R test_layers          # Run layer tests
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Follow the code style guidelines
4. Write tests for new functionality
5. Submit a pull request

### Coding Standards
- Use `clang-format` with provided `.clang-format` file
- Write descriptive commit messages
- Add documentation for public APIs
- Include unit tests for new features
- Ensure no memory leaks (use Valgrind or AddressSanitizer)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This framework is inspired by:
- PyTorch: Dynamic computation graphs
- TensorFlow: Strong production deployment
- ONNX: Model interoperability
- Caffe: C++ implementation simplicity

## Contact

For questions, issues, or contributions, please use the [GitHub Issues](https://github.com/xiaoshaoning/boat/issues) page.