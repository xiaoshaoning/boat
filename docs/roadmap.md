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
| Advanced layers (Attention, GRU, LSTM, LayerNorm, PReLU) | Done | 2026-04 |
| Unit tests and gradient checks | Done | 2026-04 |

### Phase 2: Quantization ✅

| Feature | Status | Date |
|---|---|---|
| Post-training quantization (UINT8, asymmetric/symmetric) | Done | 2026-04 |
| Quantized weight save/load | Done | 2026-04 |
| On-the-fly dequant in dense/conv forward | Done | 2026-04 |
| INT8 signed quantization | Done | 2026-05 |
| BITS2 (2-bit) packed quantization | Done | 2026-05 |
| FLOAT4 (4-bit custom float) quantization | Done | 2026-05 |
| Per-channel quantization (per-axis scale/zero-point) | Done | 2026-05 |
| Quantization-aware training (QAT) fake quantization | Done | 2026-05 |
| Serialization v3 (per-channel metadata in model files) | Done | 2026-05 |

### Phase 3: Model Format Support ✅

| Feature | Status | Date |
|---|---|---|
| ONNX loader (Gemm + Relu) | Done | 2026-04 |
| ONNX loader extended (Conv, BN, MaxPool, Softmax, Flatten) | Done | 2026-05 |
| ONNX export (boat → ONNX serialization) | Done | 2026-05 |
| Custom binary serialization (boat_model_save/load) | Done | 2026-04 |
| PyTorch loader (via LibTorch C++ API) | Done | 2026-04 |
| HuggingFace Safetensors loader | Done | 2026-04 |
| GGUF format support (Q4_0, Q4_1, Q5_0, Q8_0) | Done | 2026-05 |

### Phase 4: Examples ✅

| Feature | Status | Date |
|---|---|---|
| MNIST training (full pipeline) | Done | 2026-04 |
| CIFAR-10 CNN training | Done | 2026-05 |
| Serialization roundtrip example | Done | 2026-04 |
| Optimizer benchmarks | Done | 2026-04 |
| Attention performance benchmark | Done | 2026-04 |
| Transformer end-to-end (tokenization, training, decoding) | Done | 2026-05 |
| InsightFace face recognition (ONNX ResNet50, 130-node graph) | Done | 2026-05 |

### Phase 5: Data Pipeline ✅

| Feature | Status | Date |
|---|---|---|
| Dataset abstraction with image/label loading | Done | 2026-05 |
| DataLoader with batching, shuffling, iteration | Done | 2026-05 |
| Common transforms (Normalize, RandomCrop, RandomHorizontalFlip) | Done | 2026-05 |
| Multi-threaded prefetch via thread pool | Done | 2026-05 |

### Phase 6: Performance Optimization ✅

| Feature | Status | Date |
|---|---|---|
| SIMD vectorization (AVX2 x86, NEON ARM) for dense/conv inner loops | Done | 2026-05 |
| SGEMM micro-kernel for matrix multiplication | Done | 2026-05 |
| OpenMP parallelism for batch-independent operations | Done | 2026-05 |
| Memory pool to reduce allocation overhead | Done | 2026-05 |

### Phase 7: Data Type Extensions ✅

| Feature | Status | Date |
|---|---|---|
| BFLOAT16 (brain floating point) — F32↔BF16 conversion | Done | 2026-05 |
| INT8 signed 8-bit integer data type | Done | 2026-05 |

---

## Short-term (1-2 months)

### 1. CUDA backend — tensor operations

GPU acceleration is the largest remaining performance lever. Start with the tensor foundation.

- CUDA memory management (cudaMalloc/cudaFree wrappers in memory.c)
- CUDA tensor copy between host and device
- CUDA kernel launch infrastructure (block/grid sizing, error checking)
- Basic element-wise CUDA kernels (add, mul, relu)
- Automatic tensor migration based on device field

**Files:** `src/core/tensor.c`, `src/core/memory.c`, `src/ops/*.cu`

### 2. Conv2D depthwise / group convolution

Extend Conv2D to support groups > 1 for depthwise separable convolutions (used in MobileNet, EfficientNet).

- Group-wise weight partitioning in forward/backward
- Depthwise (groups = in_channels) fast path
- unit tests with group > 1

**Files:** `src/layers/conv.c`, `tests/test_conv.c`

### 3. NanoChat LLM inference + training

Implement a full inference and training pipeline for nanochat GPT models, following the detailed plan at `docs/nanochat_plan.md`.

- Weight loader for nanochat checkpoint format (safetensors + meta)
- BPE tokenizer (tiktoken-compatible regex splitting)
- GPT model forward/backward: RoPE, GQA, RMSNorm, ReLU², sliding window attention, value residual, logit softcap
- KV cache + inference engine (prefill + decode loop)
- Sampling (top-k, temperature, repetition penalty)
- Muon + AdamW optimizer (Polar Express + NorMuon)
- BOS-aligned best-fit dataloader
- Pretraining / SFT with loss masking / RL (GRPO) training loops
- FP8 dynamic tensorwise scaling
- Interactive chat CLI with tool use
- Unit tests for all components

**Files:** `examples/nanochat/*.{c,h}` (20+ new files), `src/ops/` (RoPE, gather, top-k), `src/core/` (FP8 extend)

**Estimated:** 6 phases, ~12 weeks

---

## Medium-term (3-6 months)

### 3. CUDA backend — layer kernels

- CUDA kernels for dense layer (warp-level matrix multiply)
- CUDA kernels for conv2d (implicit GEMM or shared memory tiling)
- CUDA fused kernel: conv → batch norm → relu (reduce global memory roundtrips)
- cuBLAS integration for large matrix operations
- cuDNN integration (optional, for production conv/attention)

**Files:** `src/layers/*.cu`, `src/ops/*.cu`

### 4. Model pruning and compression

Reduce model size and compute for deployment.

- Weight pruning (magnitude-based, iterative)
- Structured pruning (channel, filter)
- Quantization-aware fine-tuning after pruning

### 5. ONNX Runtime backend

Use ONNX Runtime as an alternative execution provider for maximum inference performance.

- Replace manual layer-by-layer execution with ORT session
- Supports all ONNX operators without per-op implementation
- GPU acceleration via CUDA/TensorRT execution providers

---

## Long-term (6-12 months)

### 6. Distributed training (multi-node)

Extend the optimizer and gradient sync for multi-node training. Requires a collective communication layer (NCCL for GPU, MPI for CPU). Ambitious — only justified if training at scale becomes a primary use case.

### 7. WebAssembly backend

Compile boat to WebAssembly for in-browser inference. Would enable client-side ML applications (privacy-preserving, no server cost). Targets ONNX and GGUF model formats.

---

## Proposals for consideration

### A. TensorFlow SavedModel format

Support loading TensorFlow 2.x SavedModel exports. Requires implementing the SavedModel protobuf structure and variable resolution. Lower priority than GGUF given the ML ecosystem shift toward ONNX and GGUF.

### B. GGUF quantization format alignment

Align boat's internal quantization types (BITS2, FLOAT4) with llama.cpp Q-series formats for interop. Would allow loading GGUF Q-quantized weights directly into boat's native types.

### C. 1-bit (binary) network support

BOAT_DTYPE_BITS1 already exists in the enum. Implement BNN-style binary convolution with XNOR-popcount kernels, opening the door for extreme compression.

---

*Last updated: 2026-05-02*
