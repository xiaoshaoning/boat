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

### Phase 8: CUDA Backend ✅

| Feature | Status | Date |
|---|---|---|
| CUDA memory management (cudaMalloc/cudaFree wrappers) | Done | 2026-05 |
| CUDA tensor copy between host and device | Done | 2026-05 |
| Basic element-wise CUDA kernels (add, mul, relu, sigmoid) | Done | 2026-05 |
| cuBLAS handle manager with lazy initialization | Done | 2026-05 |
| cuBLAS matmul (single + strided batched) | Done | 2026-05 |
| boat_matmul dispatches to cuBLAS for CUDA tensors | Done | 2026-05 |
| Dense forward: element-per-thread and warp-level kernels | Done | 2026-05 |
| Conv2D via im2col + cuBLAS GEMM | Done | 2026-05 |
| Group convolution (groups > 1) | Done | 2026-05 |
| Depthwise convolution (groups = in_channels) | Done | 2026-05 |
| Batch norm forward (two-moment shared memory reduction) | Done | 2026-05 |
| Fused batch norm + ReLU kernel | Done | 2026-05 |

| cuDNN handle manager (lazy init + destroy) | Done | 2026-05 |
| cuDNN Conv2D forward with bias + groups | Done | 2026-05 |
| cuDNN BatchNorm forward training mode | Done | 2026-05 |
| cuDNN Conv2D backward (input, weight, bias gradient) | Done | 2026-05 |
| cuDNN BatchNorm backward | Done | 2026-05 |
| cuDNN dispatch in conv layer backward | Done | 2026-05 |
| cuDNN dispatch in batchnorm layer forward/backward | Done | 2026-05 |
| CMake integration (`BOAT_WITH_CUDNN`) | Done | 2026-05 |

**Files:** `cuda/kernels/basic.cu`, `cuda/kernels/dense.cu`, `cuda/kernels/conv.cu`, `cuda/kernels/fused.cu`, `cuda/cublas_handle.cu`, `cuda/cudnn_handle.cu`, `cuda/ops/*.cu`, `src/layers/conv.c`, `src/layers/batchnorm.c`

### Phase 9: ONNX Runtime Backend ✅

| Feature | Status | Date |
|---|---|---|
| ORT C API session wrapper (create, run, free) | Done | 2026-05 |
| ONNX model loading from file and memory buffer | Done | 2026-05 |
| Single-input/single-output inference (`boat_onnxruntime_run`) | Done | 2026-05 |
| Multi-input/multi-output inference (`boat_onnxruntime_run_multi`) | Done | 2026-05 |
| Model introspection (input/output count and names) | Done | 2026-05 |
| CPU execution provider | Done | 2026-05 |
| CUDA execution provider support (optional, auto-detect) | Done | 2026-05 |
| Graph-level optimizations (constant folding, node fusion) | Done | 2026-05 |
| Multi-threaded intra-op/inter-op parallelism | Done | 2026-05 |
| ONNX save → ORT load round-trip consistency test | Done | 2026-05 |
| CMake integration (`BOAT_WITH_ONNXRUNTIME`) | Done | 2026-05 |

**Files:** `include/boat/format/onnxruntime.h`, `src/format/onnxruntime.c`, `tests/test_onnxruntime.c`

---

## Short-term (1-2 months)

### 1. NanoChat LLM inference + training

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

### 2. Model pruning and compression

Reduce model size and compute for deployment.

- Weight pruning (magnitude-based, iterative)
- Structured pruning (channel, filter)
- Quantization-aware fine-tuning after pruning

---

## Long-term (6-12 months)

### 4. Distributed training (multi-node)

Extend the optimizer and gradient sync for multi-node training. Requires a collective communication layer (NCCL for GPU, MPI for CPU). Ambitious — only justified if training at scale becomes a primary use case.

### 5. WebAssembly backend

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

*Last updated: 2026-05-02* (cuDNN backward + layer dispatch completed)
