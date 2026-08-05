# Boat: Deep Learning Framework in C

Boat is a lightweight, high-performance deep learning framework written in pure C with CUDA GPU acceleration. Designed for inference, training, and fine-tuning of neural networks with support for common model formats.

## Key Features

- **Pure C Implementation**: Minimal dependencies, easy integration into existing C/C++ projects
- **Automatic Differentiation**: Computational graph-based autodiff with gradient tracking
- **Comprehensive Data Type Support**:
  - Floating point: FP64, FP32, FP16, FP8, FP4, BFLOAT16
  - Integer: INT64, INT32, INT8, UINT8
  - Low-bit quantization: BITS2 (2-bit packed), BITS1 (1-bit binary networks)
  - Boolean: BOOL type
- **Quantization Pipeline**: UINT8/INT8 affine quantization, BITS2 (2-bit), FLOAT4 (4-bit), per-channel, and QAT fake quantization
- **Model Format Support**: ONNX (load/export/runtime executor), PyTorch (via LibTorch), HuggingFace Safetensors, GGUF (Q4_0, Q4_1, Q5_0, Q8_0)
- **Data Pipeline**: Dataset/DataLoader abstraction with batching, shuffling, multi-threaded prefetch, and transforms
- **Performance Optimizations**: SIMD (AVX2/NEON), SGEMM micro-kernel (hand-tuned with packing), OpenBLAS backend for accelerated matrix multiplication, OpenMP parallelism, memory pooling
- **CUDA GPU Acceleration**: cuBLAS matmul, cuDNN conv/batchnorm, fused attention kernels (flash attention, GQA decode), FP8/BF16 inference and training kernels, custom CUDA kernels for element-wise ops, activations, pooling, normalization, and optimizers
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

#### Using CMake (Recommended)

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

#### Using Makefile

The project also includes a traditional Makefile for simpler builds:

```bash
# Clone the repository
git clone https://github.com/xiaoshaoning/boat.git
cd boat

# Build the library
make all

# Build with debug symbols
make dev

# Build optimized release version
make release

# Run tests
make test

# Clean build artifacts
make clean
```

The Makefile automatically compiles all source files and creates a shared library `libboat.so` (or `boat.dll` on Windows) in the `build/lib/` directory.

### Build Options

- `-DBOAT_WITH_TESTS=ON`: Build test suite
- `-DBOAT_WITH_EXAMPLES=ON`: Build example programs
- `-DBOAT_WITH_ONNX=ON`: Enable ONNX support (self-contained protobuf parser)
- `-DBOAT_WITH_CUDA=ON`: Enable CUDA GPU acceleration (requires CUDA Toolkit and NVIDIA GPU)
- `-DBOAT_WITH_CUDNN=ON`: Enable cuDNN integration (requires cuDNN)
- `-DBOAT_WITH_OPENBLAS=ON`: Enable OpenBLAS backend for accelerated matrix multiplication
  - Set `-DBOAT_OPENBLAS_ROOT=/path/to/openblas` if not in a standard location
- `-DBOAT_WITH_OPENMP=ON`: Enable OpenMP parallelism
- `-DBOAT_WITH_SIMD=ON`: Enable SIMD vectorization (AVX2/NEON)
- `-DBOAT_WITH_ONNXRUNTIME=ON`: Enable ONNX Runtime executor
- `-DBOAT_WITH_HUGGINGFACE=ON`: Enable Hugging Face safetensors support (vendored cJSON, no external dependency)
- `-DBOAT_WITH_GGUF=ON`: Enable GGUF format support
- `-DBOAT_WITH_PYTORCH=ON`: Enable PyTorch support (requires LibTorch)
- `-DBOAT_WITH_TENSORFLOW=ON`: Enable TensorFlow support
- `-DBOAT_WITH_COVERAGE=ON`: Enable code coverage instrumentation
- `-DBOAT_BUILD_SHARED=ON`: Build shared library (DLL on Windows)
- `-DBOAT_DEBUG=1`: Enable debug output

### Build Configurations

- **Debug**: Debug symbols and assertions (`-DCMAKE_BUILD_TYPE=Debug`)
- **Release**: Optimized build (`-O3 -DNDEBUG`)
- **MinSizeRel**: Size-optimized build
- **RelWithDebInfo**: Release with debug symbols

Tests and examples are disabled by default; enable them with `-DBOAT_WITH_TESTS=ON` and `-DBOAT_WITH_EXAMPLES=ON`.

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
    boat_layer_t* dense1 = boat_dense_layer_create(784, 128, true);
    boat_layer_t* relu1 = boat_relu_layer_create();
    boat_layer_t* dense2 = boat_dense_layer_create(128, 10, true);
    boat_layer_t* softmax = boat_softmax_layer_create();

    boat_sequential_add(model, dense1);
    boat_sequential_add(model, relu1);
    boat_sequential_add(model, dense2);
    boat_sequential_add(model, softmax);

    // Create optimizer
    boat_optimizer_t* optimizer = boat_adam_optimizer_create(0.001f, 0.9f, 0.999f, 1e-8f);

    // Create loss function
    boat_loss_t* loss = boat_cross_entropy_loss_create();

    // Training loop (simplified)
    for (int epoch = 0; epoch < 10; epoch++) {
        // Forward pass
        boat_tensor_t* output = boat_model_forward(model, input);

        // Compute loss
        float loss_value = boat_loss_compute(loss, output, target);

        // Backward pass
        boat_tensor_t* grad = boat_loss_backward(loss);
        boat_model_backward(model, grad);

        // Update parameters
        boat_optimizer_step(optimizer);
        boat_optimizer_zero_grad(optimizer);

        printf("Epoch %d, Loss: %f\n", epoch, loss_value);
    }

    // Cleanup
    boat_optimizer_free(optimizer);
    boat_loss_free(loss);
    boat_model_free(model);
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

# Build and run via CMake (from project root)
cd ../..
mkdir -p build && cd build
cmake .. -DBOAT_WITH_EXAMPLES=ON
make
./examples/mnist/mnist
```

### Automatic Differentiation Version

Boat also includes an advanced MNIST example using automatic differentiation (`mnist_autodiff.c`) that demonstrates:

- **Dynamic computation graph** with gradient tracking
- **Learning rate schedulers** (cosine annealing, step LR)
- **Gradient clipping** and monitoring
- **Memory optimization** with pooling strategies
- **Auto-tuning** of hyperparameters during training
- **Comprehensive logging** and progress tracking

The autodiff version is built automatically alongside the rest of the framework via CMake. Both `mnist` and `mnist_autodiff` are compiled when building with examples enabled:

```bash
mkdir build && cd build
cmake .. -DBOAT_WITH_EXAMPLES=ON
make
# Run either version:
./examples/mnist/mnist
./examples/mnist/mnist_autodiff
```

The autodiff version provides more detailed training metrics and automatic hyperparameter tuning capabilities.

### Key Code Snippets

**Model Creation:**
```c
// Create a convolutional neural network for MNIST
boat_sequential_model_t* model = boat_sequential_create();

// Add convolutional layers
boat_layer_t* conv1 = boat_conv_layer_create(1, 32, 3, 1, 1, 1);
boat_layer_t* relu1 = boat_relu_layer_create();
boat_layer_t* pool1 = boat_pool_layer_create(2, 2, 0);

boat_layer_t* conv2 = boat_conv_layer_create(32, 64, 3, 1, 1, 1);
boat_layer_t* relu2 = boat_relu_layer_create();
boat_layer_t* pool2 = boat_pool_layer_create(2, 2, 0);

// Add fully connected layers
boat_layer_t* flatten = boat_flatten_layer_create();
boat_layer_t* fc1 = boat_dense_layer_create(7*7*64, 128, true);
boat_layer_t* relu3 = boat_relu_layer_create();
boat_layer_t* fc2 = boat_dense_layer_create(128, 10, true);
boat_layer_t* softmax = boat_softmax_layer_create(-1);

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
boat_optimizer_t* optimizer = boat_adam_optimizer_create(0.001f, 0.9f, 0.999f, 1e-8f);
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
        boat_tensor_t* predictions = boat_model_forward(model, batch_images);

        // Compute loss
        float batch_loss = boat_loss_compute(loss, predictions, batch_labels);
        epoch_loss += batch_loss;

        // Compute accuracy
        correct += compute_correct_predictions(predictions, batch_labels);

        // Backward pass
        boat_tensor_t* grad = boat_loss_backward(loss);
        boat_model_backward(model, grad);

        // Update parameters
        boat_optimizer_step(optimizer);
        boat_optimizer_zero_grad(optimizer);

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

With the Adam optimizer and proper data standardization, the MNIST example achieves:
- Training accuracy: **>99%** (converges within 10 epochs with default settings)
- Test accuracy: **>96%** (verified on held-out test set)
- Training time: ~11 minutes on CPU (1000 samples, 10 epochs, batch size 32)

Both the manual gradient and automatic differentiation (`mnist_autodiff`) versions achieve comparable results.

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

## NanoChat Example

NanoChat is a GPT LLM example (d34 2.2B parameters) with CUDA-accelerated inference, training, and an OpenAI-compatible HTTP server.

### Chat CLI

```bash
# Build with CUDA enabled
mkdir build && cd build
cmake .. -DBOAT_WITH_CUDA=ON -DBOAT_WITH_EXAMPLES=ON
make

# Run interactive chat
./examples/nanochat/nanochat_cli <model_dir>
```

The chat CLI supports token-by-token streaming, markdown rendering (Windows console), and conversation history.

### HTTP Server

```bash
# Start the server
./examples/nanochat/server <model_dir>

# Query via curl (OpenAI-compatible API)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

### Training

```bash
# Run training (supports pretraining, SFT, and GRPO)
./examples/nanochat/nanochat_train <model_dir> <data_dir>
```

The training pipeline includes:
- Muon and AdamW optimizers
- FP8 dynamic tensorwise scaling
- BOS-aligned best-fit batching
- GRPO (Group Relative Policy Optimization) for RL fine-tuning

### Architecture

NanoChat implements the d34 architecture: 34 transformer layers, GQA (16 heads, 2 KV heads), RoPE, ReLU² activation, sliding window attention, value residual, and logit softcap. All attention and FFN operations are accelerated with custom fused CUDA kernels.

For detailed design, see [docs/nanochat_plan.md](docs/nanochat_plan.md).

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
- **Activation**: ReLU, PReLU, Sigmoid, Tanh, Softmax
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

The framework supports a comprehensive range of data types for efficient computation:

### Floating Point Types
- **FP64** (double): 64-bit double precision floating point
- **FP32** (float): 32-bit single precision floating point
- **FP16**: 16-bit half precision floating point
- **BFLOAT16**: 16-bit brain floating point (same exponent range as FP32)
- **FP8**: 8-bit custom floating point format
- **FP4**: 4-bit custom floating point format

### Integer Types
- **INT64**: 64-bit signed integer
- **INT32**: 32-bit signed integer
- **INT8**: 8-bit signed integer
- **UINT8**: 8-bit unsigned integer

### Low-Bit Quantization Types
- **BITS2**: 2-bit packed values (4 values per byte)
- **BITS1**: 1-bit packed values (8 values per byte, binary networks)

### Special Types
- **BOOL**: Boolean values (1 byte per element)

### Quantization Capabilities

| Feature | Bit-width | Type |
|---|---|---|
| Per-tensor affine | 8-bit | UINT8, INT8 |
| Per-channel affine | 8-bit | UINT8, INT8 |
| BITS2 packed | 2-bit | Asymmetric affine |
| FLOAT4 custom float | 4-bit | Direct (no affine) |
| QAT fake quantization | any | Simulates quantization noise during training |

## Examples

The repository includes several comprehensive examples:

- **MNIST Classification**: Complete training pipeline for digit recognition
- **CIFAR-10**: CNN image classification with data pipeline and transforms
- **Transformer**: End-to-end transformer with tokenization, training, and autoregressive decoding
- **Translator**: English-to-French MarianMT (Helsinki-NLP) inference engine using Safetensors weights
- **InsightFace**: Face recognition model (ResNet50-based) inference via ONNX runtime executor, producing 512-dim embeddings
- **Automatic Differentiation**: Gradient computation with dynamic computation graphs
- **Scheduler Usage**: Learning rate scheduling with cosine annealing, step LR, and lambda LR
- **ONNX Export**: Export trained boat models to ONNX format
- **NanoChat**: GPT LLM inference and training (d34 2.2B) with CUDA acceleration
  - Interactive chat CLI with token streaming
  - OpenAI-compatible HTTP server (JSON API)
  - Training loop with Muon/AdamW optimizers and FP8 support
  - Fused GQA attention kernels for fast decode

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
│   │   ├── prune.h          # Model pruning
│   │   ├── quantize.h       # Quantization
│   │   ├── sampling.h       # Token sampling utilities
│   │   ├── cuda_runtime.h   # CUDA runtime API
│   │   └── format/          # Model format loaders
│   │       ├── onnx.h       # ONNX format support
│   │       ├── onnxruntime.h# ONNX Runtime executor
│   │       ├── pytorch.h    # PyTorch format support
│   │       ├── tensorflow.h # TensorFlow format support
│   │       └── huggingface.h# HuggingFace format support
│   └── boat.h               # Main include file
├── src/                     # Implementation
│   ├── core/               # Core functionality
│   ├── ops/                # Operations (with device dispatch)
│   ├── graph/              # Computational graph
│   ├── layers/             # Neural network layers
│   ├── optimizers/         # Optimization algorithms (with CUDA paths)
│   ├── schedulers/         # Learning rate schedulers
│   ├── loss/               # Loss functions (with CUDA paths)
│   ├── model/              # Model management
│   └── format/             # Model format loaders
├── cuda/                   # CUDA backend
│   ├── kernels/            # CUDA kernels (basic, conv, dense, fused, norm, pool, optimizer, FP8, BF16)
│   ├── ops/                # CUDA ops (activation, arithmetic, linear)
│   ├── tensor.cu           # CUDA tensor copy
│   ├── cublas_handle.cu    # cuBLAS handle manager
│   ├── cudnn_handle.cu     # cuDNN handle manager
│   ├── graph/              # CUDA graph executor
│   └── autodiff/           # CUDA autodiff
├── bindings/js/            # Node.js N-API bindings
├── examples/               # Example programs
│   ├── mnist/             # MNIST classification
│   ├── cifar10/           # CIFAR-10 image classification
│   ├── common/            # Shared utilities (JSON, safetensors)
│   ├── nanochat/          # NanoChat GPT LLM (inference, training, server)
│   ├── transformer/       # Transformer end-to-end example
│   └── translator/        # English-French MarianMT translator
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── archive/           # Archived/legacy tests
├── benchmarks/            # Performance benchmarks
├── docs/                  # Documentation
└── scripts/               # Utility scripts
```

For detailed API documentation and development guidelines, see [CLAUDE.md](CLAUDE.md).

## Development Status

### Current Features (Implemented)
- Core tensor operations with multiple data types
- Automatic differentiation with computational graph
- Neural network layers (dense, conv, attention, LSTM, GRU, etc.)
- Optimizers (Adam, RMSprop, SGD, Adagrad) with CUDA update paths
- Learning rate schedulers (cosine annealing, step LR, lambda LR)
- Loss functions (MSE, cross-entropy, Huber) with CUDA backward paths
- Data pipeline (Dataset, DataLoader with multi-threaded prefetch)
- Post-training quantization (UINT8, INT8, BITS2, FLOAT4, per-channel)
- Quantization-aware training (QAT) with fake quantization
- Model pruning (magnitude-based, structured channel/filter pruning)
- Model format loaders (ONNX, PyTorch, TensorFlow, HuggingFace, GGUF)
- ONNX Runtime executor (graph-based direct inference for complex ONNX models)
- CUDA GPU acceleration (cuBLAS matmul, cuDNN conv/batchnorm, fused attention kernels, FP8/BF16 inference and training, custom kernels for element-wise ops, activations, pooling, normalization, and optimizers)
- Group/depthwise convolution with cuDNN acceleration
- PReLU activation layer (Parametric ReLU for modern CNN architectures)
- InsightFace face recognition model inference (ResNet50, 512-dim embeddings)
- Model serialization (custom binary format, v3 with per-channel metadata)
- Performance optimizations (SIMD, SGEMM with optional OpenBLAS backend, OpenMP, memory pool)
- Node.js N-API bindings (Tensor and Model operations)
- Cross-platform build with CMake
- Comprehensive test suite with CI (GitHub Actions: CPU matrix + CUDA build)
- MNIST training example (manual and autodiff, both >96% test accuracy)
- CIFAR-10 CNN training example
- Transformer end-to-end example
- English-French MarianMT translator (Safetensors-based inference)
- InsightFace face recognition (ONNX Runtime, 130-node graph executor)
- ONNX export (boat → ONNX serialization)
- **NanoChat GPT LLM (d34 2.2B)**:
  - Interactive chat CLI with token streaming
  - OpenAI-compatible HTTP server with JSON API
  - Training pipeline (pretraining, SFT, GRPO) with Muon/AdamW optimizers
  - FP8 dynamic tensorwise scaling for training
  - Fused GQA decode attention with KV cache
  - BF16 inference (avoids FP16 overflow)

### Planned Features
- WebAssembly backend for in-browser inference
- Distributed training support

## Code Quality

Boat follows strict code quality standards with comprehensive static analysis and const-correctness guidelines.

### Const Correctness
The framework enforces const correctness throughout its API to improve safety, readability, and compiler optimization. See the [Const Usage Guide](docs/const_usage_guide.md) for detailed guidelines on:
- Function parameter constness
- Return value constness
- Structure field constness
- Common patterns and examples

### Static Analysis
The project uses `cppcheck` for static analysis to detect potential issues. Run the analysis with:
```bash
cppcheck --enable=warning,style --suppress=missingInclude -I include src
```

**Recent Improvements**: The codebase has been extensively analyzed and refined to achieve **zero cppcheck warnings** across all source files. This includes fixes for:
- Const correctness issues (parameter and pointer constness)
- Unused variables and functions
- Variable shadowing
- Memory management patterns
- Type consistency and format strings

Static analysis reports are maintained in the repository (`cppcheck_*.txt`) to track code quality improvements over time.

### Automated Testing
All code changes are validated through comprehensive unit and integration tests.

## Testing

Run the test suite to verify the installation:

```bash
cd build
make test
```

Or run specific tests:

```bash
ctest -R test_tensor                   # Run tensor tests
ctest -R test_quantize                 # Run quantization tests
ctest -R test_serialization_integration # Run serialization roundtrip tests
ctest -R test_autodiff                 # Run autodiff tests
ctest -R test_layers                   # Run layer tests
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

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

This framework is inspired by:
- PyTorch: Dynamic computation graphs
- TensorFlow: Strong production deployment
- ONNX: Model interoperability
- Caffe: C++ implementation simplicity

## Contact

For questions, issues, or contributions, please use the [GitHub Issues](https://github.com/xiaoshaoning/boat/issues) page.