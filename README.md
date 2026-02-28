# Boat: Deep Learning Framework in C

Boat is a lightweight, high-performance deep learning framework written in C with eventual CUDA support. Designed for inference, training, and fine-tuning of neural networks with support for common model formats.

## Features

- **Pure C implementation**: Minimal dependencies, easy integration
- **Automatic differentiation**: Computational graph-based autodiff
- **Multiple data types**: Support for FP64, FP32, FP16, FP8, FP4, and low-bit networks (1-bit, 2-bit)
- **Model format support**: ONNX, PyTorch, TensorFlow model loading
- **GPU acceleration**: CUDA backend (future)
- **Quantization ready**: Native support for low-bit networks

## Quick Start

### Building

```bash
mkdir build
cd build
cmake ..
make
```

### Simple Example

```c
#include <boat/boat.h>
#include <boat/layers.h>
#include <boat/optimizers.h>

int main() {
    boat_init();

    // Create a simple neural network
    boat_sequential_model_t* model = boat_sequential_create();

    // Add layers
    boat_layer_t* dense1 = boat_dense_layer_create(784, 128);
    boat_layer_t* relu1 = boat_relu_layer_create();
    boat_layer_t* dense2 = boat_dense_layer_create(128, 10);

    boat_sequential_add(model, dense1);
    boat_sequential_add(model, relu1);
    boat_sequential_add(model, dense2);

    // Use the model...

    // Cleanup
    boat_model_free((boat_model_t*)model);
    boat_cleanup();

    return 0;
}
```

## Project Structure

See [CLAUDE.md](CLAUDE.md) for detailed project structure and API documentation.

## License

MIT License. See [LICENSE](LICENSE) for details.