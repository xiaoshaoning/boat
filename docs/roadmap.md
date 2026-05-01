# Development Roadmap

## Completed

### Phase 1: Core CPU Implementation ✅

| Feature | Status | Date |
|---|---|---|
| Tensor operations (create, reshape, slice, transpose) | Done | 2026-04 |
| Basic layers (Dense, Conv2D, ReLU, Softmax, Pool, Flatten, BatchNorm) | Done | 2026-04 |
| Automatic differentiation engine | Done | 2026-04 |
| Computational graph infrastructure | Done | 2026-04 |
| Optimizers (SGD, Adam, RMSprop, Adagrad) | Done | 2026-04 |
| Schedulers (StepLR, CosineAnnealing, LambdaLR) | Done | 2026-04 |
| Loss functions (MSE, CrossEntropy, Huber) | Done | 2026-04 |
| Sequential model API | Done | 2026-04 |
| Advanced layers (Attention, GRU, LSTM, LayerNorm) | Done | 2026-04 |
| Unit tests and gradient checks | Done | 2026-04 |

### Phase 2: Quantization ✅

| Feature | Status | Date |
|---|---|---|
| Post-training quantization (UINT8, asymmetric/symmetric) | Done | 2026-04 |
| Quantized weight save/load | Done | 2026-04 |
| On-the-fly dequant in dense/conv forward | Done | 2026-04 |
| INT8 signed quantization | Done | 2026-05 |

### Phase 3: Model Format Support ✅

| Feature | Status | Date |
|---|---|---|
| ONNX loader (Gemm + Relu) | Done | 2026-04 |
| ONNX loader extended (Conv, BN, MaxPool, Softmax, Flatten) | Done | 2026-05 |
| ONNX export (boat → ONNX serialization) | Done | 2026-05 |
| Custom binary serialization (boat_model_save/load) | Done | 2026-04 |
| PyTorch loader (via LibTorch C++ API) | Done | 2026-04 |
| HuggingFace Safetensors loader | Done | 2026-04 |

### Phase 4: Examples ✅

| Feature | Status | Date |
|---|---|---|
| MNIST training (full pipeline) | Done | 2026-04 |
| CIFAR-10 CNN training | Done | 2026-05 |
| Serialization roundtrip example | Done | 2026-04 |
| Optimizer benchmarks | Done | 2026-04 |
| Attention performance benchmark | Done | 2026-04 |

---

## Short-term (1-2 months)

### 1. BFLOAT16 data type

BFLOAT16 (brain floating point) is the standard ML training dtype, used by most modern models. It has the same exponent range as FLOAT32 but reduced mantissa precision, making it a drop-in replacement in many cases.

- Add `BOAT_DTYPE_BFLOAT16` to enum (at end for backward compat, before INT8)
- Implement FLOAT32 ↔ BFLOAT16 conversion routines
- No new arithmetic kernels needed — promote to FLOAT32 for compute, store as BFLOAT16
- Enables loading BF16 weights from HuggingFace models without F32 conversion

**Files:** `include/boat/tensor.h`, `src/core/tensor.c`, `src/format/huggingface.c`

### 2. Data pipeline (Dataset/DataLoader)

`src/data/` is empty. A proper data pipeline is essential for training at scale.

- `boat_dataset_t` abstraction with image/label loading
- `boat_dataloader_t` with automatic batching, shuffling, and iteration
- Common transforms (Normalize, RandomCrop, RandomHorizontalFlip)
- Multi-threaded prefetch via a simple thread pool

**Files:** `src/data/dataset.c`, `src/data/dataloader.c`, `include/boat/data.h`

### 3. Transformer end-to-end example

`examples/transformer/` exists but is empty. Implement a complete transformer training example.

- Text tokenization and vocabulary handling
- Embedding layer (token + positional)
- Transformer block: Self-Attention → LayerNorm → FFN → LayerNorm
- Training on a small translation or text generation task
- Inference with autoregressive decoding

**Files:** `examples/transformer/transformer.c`, `examples/transformer/CMakeLists.txt`

---

## Medium-term (3-6 months)

### 4. GGUF/GGML format support

GGUF is the standard format for running quantized LLMs locally (llama.cpp ecosystem). Supporting it opens boat to thousands of pre-trained community models.

- Implement GGUF container format parser (magic, metadata kv, tensors)
- Support common quantization types (Q4_0, Q4_1, Q5_0, Q8_0)
- Load and run inference with existing boat layers

**Files:** `src/format/gguf.c`, `src/format/gguf.h`, `include/boat/format/gguf.h`

### 5. Performance optimization

CPU performance is a bottleneck — single-threaded FP32 is too slow for anything beyond demos.

- SIMD vectorization (AVX2 on x86, NEON on ARM) for dense/conv inner loops
- SGEMM micro-kernel for matrix multiplication
- OpenMP parallelism for batch-independent operations
- Memory pool to reduce allocation overhead

**Files:** `src/ops/linear.c`, `src/layers/dense.c`, `src/layers/conv.c`, `src/core/memory.c`

### 6. INT4 / INT2 quantization

Extend PTQ to the more aggressive low-bit formats already defined in the enum.

- `BOAT_DTYPE_BITS2` (2-bit, 4 values per byte) quantization
- `BOAT_DTYPE_FLOAT4` (4-bit custom float) quantization
- Per-channel quantization for better accuracy at low bit-widths
- Quantization-aware training (QAT) support

**Files:** `src/core/quantize.c`, `tests/test_quantize.c`

---

## Long-term (6-12 months)

### 7. CUDA backend

GPU acceleration is the largest performance lever.

- CUDA tensor operations (allocate, copy, free on device)
- CUDA kernels for conv, dense, attention
- cuBLAS integration for matrix multiplication
- Automatic tensor migration between CPU and GPU

**Files:** `src/core/tensor.c`, `src/ops/*.cu`, `src/layers/*.cu`

### 8. Model pruning and compression

Reduce model size and compute for deployment.

- Weight pruning (magnitude-based, iterative)
- Structured pruning (channel, filter)
- Quantization-aware fine-tuning after pruning

### 9. ONNX Runtime backend

Use ONNX Runtime as an alternative execution provider for maximum inference performance.

- Replace manual layer-by-layer execution with ORT session
- Supports all ONNX operators without per-op implementation
- GPU acceleration via CUDA/TensorRT execution providers

---

## Proposals for consideration

### A. TensorFlow SavedModel format

Support loading TensorFlow 2.x SavedModel exports. Requires implementing the SavedModel protobuf structure and variable resolution. Lower priority than GGUF given the ML ecosystem shift toward ONNX and GGUF.

### B. WebAssembly backend

Compile boat to WebAssembly for in-browser inference. Would enable client-side ML applications (privacy-preserving, no server cost). Targets ONNX and GGUF model formats.

### C. Distributed training (multi-node)

Extend the optimizer and gradient sync for multi-node training. Requires a collective communication layer (NCCL for GPU, MPI for CPU). Ambitious — only justified if training at scale becomes a primary use case.

---

*Last updated: 2026-05-01*
