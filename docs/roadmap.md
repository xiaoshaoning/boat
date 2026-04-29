# Development Roadmap

## Short-term (1-2 months)

### 1. Full MNIST training ✅

**Status: Complete (2026-04-29)**

Changes to `examples/mnist/mnist.c`:
- Default to full 60k dataset, `MNIST_QUICK_TEST=1` for quick testing (1k samples, 1 epoch)
- Replaced per-sample loop with proper batch iteration (batch_size=32)
- Fixed memory leak in `backward_pass` where intermediate gradient tensors were not freed
- Added progress indication (dots per 10% of batches, unbuffered stdout)

Performance (60k samples, batch_size=32, 10 epochs):
- ~24 min/epoch on CPU (60k samples)
- Memory: ~230 MB steady state (no leak)
- Epoch 1 accuracy: 95.89%
- Epoch 2 accuracy: 97.67%
- Expected final test accuracy: >97%

### 2. Model serialization

Implement `boat_model_save()` / `boat_model_load()` with a custom binary format to persist trained weights.

- Define a simple portable format (header + layer weights)
- Support saving and loading the full sequential model
- Close the train -> inference workflow

### 3. Fill in missing examples

Both `examples/cifar10/` and `examples/transformer/` are empty stubs.

- Implement one end-to-end example to validate the layer API
- CIFAR-10 would test conv net on RGB images
- Transformer would test attention layers

## Medium-term (3-6 months)

### 4. Data pipeline

`src/data/` is empty. Implement Dataset/DataLoader abstractions.

- Automatic batching and shuffling
- Multi-threaded data loading
- Common transforms (normalize, augment)

### 5. ONNX export

Boat can already load ONNX models. Adding export enables deployment in other frameworks.

- Implement ONNX graph serialization
- Round-trip test: export -> import -> compare output
- Support common ops (conv, dense, relu, softmax)

### 6. Integration tests

Tests are mostly unit-level. End-to-end tests are missing.

- Full training loop -> save -> load -> inference pipeline test
- Gradient numerical checks for all layers
- Memory leak detection tests

## Long-term

### 7. CUDA backend

The biggest performance lever, but also the largest engineering investment.

- GPU tensor operations
- CUDA kernels for conv, dense, attention
- Memory management on device

### 8. Performance optimization

- SIMD vectorization (AVX2/NEON) for CPU paths
- OpenMP parallelism for independent operations
- Memory pool optimization for tensor allocation

---

*Generated: 2026-04-29*
