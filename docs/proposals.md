# Project Proposals — Boat Deep Learning Framework

Proposals for next-phase work, ordered by impact vs effort.

## 1. Continuous Integration (CI) ✅

**Why:** The `.github/workflows/` directory exists but there are 24+ tests that must be run manually. One broken commit can go unnoticed for days. With CUDA, C toolchains on Linux/Windows/Mac, and optional dependencies (cuDNN, ONNX Runtime, OpenBLAS), the configuration matrix is large enough that manual testing doesn't scale.

**What:**
- ✅ GitHub Actions workflow for CPU builds + `ctest` on push/PR (Ubuntu, Windows, macOS) — see `.github/workflows/ci.yml`
- ✅ CUDA build verification on push/PR (Ubuntu, CUDA 12.6, sm_86) — see `.github/workflows/ci-cuda.yml`. Tests run conditionally when GPU is available.
- ✅ Coverage reporting via gcovr + Codecov (Linux, Debug build) — see `.github/workflows/ci.yml` coverage job

**Effort:** Medium (2-3 days for basic, 1 week for full matrix)
**Impact:** Foundational — prevents regressions as the project grows
**Status:** Core matrix (CPU × 3 OS × 2 build types + CUDA build) complete. Coverage reporting remains as future work.

---

## 2. Language Bindings ⬜

**Why:** The framework is feature-complete enough (tensors, autodiff, optimizers, quantization, CUDA backend) that bindings to a high-level language would unlock real use. Without them, every user must write C. Two strong candidates:

### 2a. TypeScript/JavaScript Bindings (Node.js N-API) ✅

**Why charming:** TypeScript bindings open up unique possibilities that Python can't match — running inference directly in the browser via WebAssembly, or in Node.js/Deno servers without a Python runtime. The JS/TS ecosystem is the largest developer ecosystem, and a deep learning framework with first-class TS support would stand out. Modern N-API (node-addon-api) makes C↔JS interop clean without V8 dependency hell.

**What:**
- ✅ N-API native addon (`boat-napi.dll`) wrapping Tensor create/shape/dtype/ops and Model create/forward — see `bindings/js/src/boat_bridge.c`
- ✅ CMake build with `cmake-js` — see `bindings/js/CMakeLists.txt`
- ✅ TypeScript declarations — see `bindings/js/lib/types.d.ts`
- ✅ JS entry point — see `bindings/js/lib/index.js`
- ✅ Compiled and linked (Release build, `boat-napi.dll` + `boat-napi.lib`)
- WebAssembly backend, async inference, package distribution deferred

**Effort:** Medium-High (2-3 weeks for N-API core, +2 weeks for WASM, +1 week for TS types)
**Impact:** Highly differentiating — most ML frameworks have Python bindings, few have first-class TypeScript. Enables browser-side AI, Node.js inference servers, and taps into the JS/TS ecosystem.
**Status:** ✅ Completed (Phase 1: core addon). WASM, async, distribution remain.

### 2b. Python Bindings (Alternative) ⬜

**Why pragmatic:** Python is the de-facto language for ML/AI. A `ctypes` wrapper around the C API is low-effort and immediately useful for data scientists.

**What:**
- `ctypes`-based wrapper around `boat.h` public API
- NumPy bridge for data I/O
- Optional: `pybind11` for tighter integration

**Effort:** Low-Medium (1-2 weeks for ctypes, 3-4 weeks for pybind11)
**Impact:** High leverage for adoption, but less differentiating than TypeScript

---

## 3. Mixed Precision Training (BF16 Master Weights) ✅

**Why:** Currently the CUDA backend uses FP32 for training. With NanoChat-scale models (~2.2B params), BF16 master weights with FP32 accumulate would halve memory (~8.8 GB → ~4.4 GB for weights) and improve throughput. The cuBLAS `GemmEx` kernel already supports `CUDA_R_16BF` I/O with `CUDA_R_32F` compute (used in NanoChat inference). The optimizer needs a BF16-aware update path.

**What:**
- ✅ BF16 ↔ FP32 conversion CUDA kernels (library-level, promoted from NanoChat) — see `cuda/kernels/bf16_conversion.cu`
- ✅ BF16 fused optimizer kernels — SGD (vanilla + momentum) and Adam, with FP32 gradients + FP32 state → BF16 weight update — see `cuda/kernels/bf16_optimizer.cu`
- ✅ Adam optimizer: FP32 m/v state for BF16 params; BF16 CUDA dispatch + CPU fallback — see `src/optimizers/adam.c`
- ✅ SGD optimizer: FP32 velocity for BF16 params; BF16 CUDA dispatch + CPU fallback — see `src/optimizers/sgd.c`
- ✅ Verified: 15/15 tests pass (FP32↔BF16 round-trip, SGD, SGD+momentum, Adam, NaN/Inf), all within BF16 quantization tolerance

**Effort:** Medium (1-2 weeks)
**Impact:** Directly enables training larger models on available hardware. Feeds into the NanoChat training goal already on the roadmap.
**Status:** ✅ Completed (CUDA kernel + optimizer dispatch). NanoChat training pipeline integration remains.

---

## 4. FP8 Inference Kernels ⬜

**Why:** FP8 (8-bit floating point, `BOAT_DTYPE_FLOAT8`) is already a registered dtype in the framework but has no CUDA kernels. Hopper GPUs (H100, H200) have native FP8 tensor core support, offering 2× throughput vs FP16/BF16. Blackwell (RTX 5090) also supports FP8.

**What:**
- FP8 quantization: FP32 → FP8 (E4M3 for weights, E5M2 for activations)
- FP8 dequantization: FP8 → FP32
- cuBLAS FP8 matmul (`cublasGemmEx` with `CUDA_R_8F_E4M3`)
- FP8 element-wise kernels (add, mul, relu, residual)
- Integration with NanoChat for FP8 inference pass

**Effort:** Medium (1-2 weeks)
**Impact:** High for Hopper/Blackwell users. Further halves memory vs BF16 (~2.2 GB for 2.2B model weights).
**Status:** ⬜ Not started

---

## 5. Multi-GPU Tensor Parallelism ⬜

**Why:** Running NanoChat (2.2B) or larger models requires more memory than a single GPU provides. Tensor parallelism splits projections across GPUs with all-reduce synchronization.

**What:**
- Column-parallel and row-parallel linear layer sharding
- QKV/FC1 split across GPUs, all-reduce for results
- NCCL-based collective communication
- Drop-in for NanoChat inference (model-parallel decode)

**Effort:** High (3-4 weeks)
**Impact:** Enables models too large for one GPU. Significant complexity.
**Status:** ⬜ Not started

---

## 6. LLM Serving API ✅

**Why:** A lightweight HTTP server wrapping the NanoChat engine with an OpenAI-compatible `/v1/chat/completions` endpoint would make boat useful as a drop-in local inference server.

**What:**
- ✅ Custom C HTTP server (platform sockets: WinSock2/POSIX) — see `examples/nanochat/server.c`
- ✅ `/v1/chat/completions` endpoint (OpenAI-compatible request/response format)
- ✅ Token streaming via SSE (`stream: true`) and non-streaming modes
- ✅ `/v1/models` and `/health` endpoints
- ✅ CORS headers for cross-origin requests
- Public API in `examples/nanochat/server.h`
- Concurrent request queue with batching deferred

**Effort:** Medium (1-2 weeks)
**Impact:** High for demos and integration. Enables OpenAI-compatible local inference server.
**Status:** ✅ Completed (basic server). Concurrent batching remains.

---

## 7. Flash Attention CUDA Kernels ✅

**Why:** The current fused attention kernels work but use simple O(n²) shared memory patterns. Flash Attention-style tiling avoids materializing the full N×N attention matrix, reducing VRAM from O(N²) to O(N) for long sequences.

**What:**
- ✅ Fused prefill and decode attention kernels rewritten with warp shuffle (`__shfl_xor_sync`) dot product reduction — replaces tree reduction in shared memory, reducing `__syncthreads()` from ~7 per KV position to 2
- ✅ Online softmax — single fused loop with running max/sum, eliminating the O(seq_len) scores array and separate two-pass softmax
- ✅ Shared memory reduced from O(seq_len) to 20 bytes (static), enabling support for arbitrarily long sequences
- ✅ Verified against CPU reference across 12 test configurations (seq_len 1–2048, heads 1–17), max difference 1.95e-3 (within BF16 precision), no NaN/Inf — see `examples/nanochat/test_flash_attn.cu`
- Causal masking (built into prefill loop bound), GQA, ALiBi, non-power-of-2 fallback deferred

**Effort:** High (2-3 weeks)
**Impact:** Significant for long-context models (8K+ tokens). Completed for NanoChat (head_dim=128, MHA). GQA/ALiBi/extensions remain.

---

## Recommendation

Completed:
| # | Proposal | Status |
|---|----------|--------|
| 1 | CI | ✅ Core matrix done |
| 3 | Mixed Precision Training | ✅ CUDA kernels + optimizer dispatch done |
| 6 | LLM Serving API | ✅ Basic server done |
| 7 | Flash Attention CUDA Kernels | ✅ Warp shuffle + online softmax done |

Remaining priorities (next 1-2 months):
1. **Language bindings** ⬜ — TypeScript (differentiating, enables browser/Node.js) or Python (pragmatic)
2. **FP8 inference kernels** ⬜ — next step for Blackwell/Hopper optimization

FP8 and multi-GPU parallelism are worth pursuing once the single-GPU training pipeline is solid.
