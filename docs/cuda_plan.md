# CUDA/cuDNN Full Support

## Context ✅ ALL PHASES COMPLETE

All 5 phases of the CUDA backend plan have been implemented:
- **Phase 1**: All stub `.cu` files filled (arithmetic, activation, linear, tensor, pool, norm)
- **Phase 2**: Device dispatch added to all CPU ops (arithmetic, activation, linear)
- **Phase 3**: CUDA paths for all layers (attention refactored, pool/norm/relu/dense)
- **Phase 4**: CUDA update kernels for SGD/Adam optimizers and MSE/cross-entropy loss
- **Phase 5**: Build config fixed, all `.cu` files registered in CMakeLists.txt

Plus a CUDA-accelerated GLM-OCR inference example (`examples/ocr_cuda/`) with CogViT vision encoder and GLM decoder using M-RoPE, GQA attention, and custom CUDA kernels.

---

## Phase 1 — Fill stub .cu files (core wiring) ✅ DONE

### 1. `cuda/ops/arithmetic.cu` ✅
186 lines — all new kernels implemented (exp, log, sqrt, rsqrt, neg, abs, mod, sub_scalar, div_scalar, fill, scale).

### 2. `cuda/ops/activation.cu` ✅
177 lines — Softmax (shared memory reduction), LogSoftmax, GELU (tanh-approx) kernels implemented.

### 3. `cuda/ops/linear.cu` ✅
202 lines — Tiled 2D transpose (16×16 shared mem), general N-D fallback, dot product reduction.

### 4. `cuda/tensor.cu` ✅
98 lines — `boat_cuda_tensor_clone`, `boat_cuda_tensor_to_host`, `boat_cuda_tensor_to_device`.

### 5. `cuda/kernels/pool.cu` ✅
124 lines — MaxPool2d forward (one thread per output element) and backward (atomicAdd scatter).

### 6. `cuda/kernels/norm.cu` ✅
320 lines — LayerNorm forward/backward, RMSNorm forward/backward (shared memory reductions).

## Phase 2 — Device dispatch in CPU ops ✅ DONE

### 7. `src/ops/arithmetic.c` ✅
CUDA dispatch added to all element-wise ops (add/sub/mul/div, scalar variants, mod, exp/log/sqrt/rsqrt, abs/neg, clamp, sum/mean/var reductions).

### 8. `src/ops/activation.c` ✅
CUDA dispatch for softmax, relu, silu (via existing basic.cu kernels), and gelu.

### 9. `src/ops/linear.c` ✅
CUDA dispatch for transpose (tiled kernel) and dot (reduction kernel). `boat_matmul` already had cuBLAS path ✓.

## Phase 3 — Layer CUDA paths ✅ DONE

### 10. `src/layers/attention.c` ✅
Refactored from manual 6-nested-loop to op chain: `matmul(Q, K^T) × scale → softmax → matmul(weights, V)`. Each op gets CUDA via existing dispatch + cuBLAS matmul.

### 11. `src/layers/pool.c` ✅
MaxPool2d forward/backward CUDA dispatch (atomicAdd scatter for backward).

### 12. `src/layers/norm.c` ✅
LayerNorm/RMSNorm CUDA dispatch for forward and backward.

### 13. `src/layers/relu.c` ✅
Added `cache_input`, proper backward masking with `boat_cuda_relu_backward_f32`.

### 14. `src/layers/dense.c` ✅
Bias gradient CUDA path (sum-axis reduction).

## Phase 4 — Optimizer + Loss CUDA paths ✅ DONE

### 15. `src/optimizers/sgd.c` + `adam.c` ✅
- SGD update kernel (`param -= lr * grad`) with momentum/Nesterov.
- Adam update kernel (fused m/v/param update).
- Fixed: velocity/momentum buffers allocated on same device as params.

### 16. `src/loss/mse.c` + `cross_entropy.c` ✅
- MSE forward: device-side diff→square→sum, copy scalar to host.
- MSE backward: `grad = 2*(pred-target)/N` element-wise kernel.
- Cross-entropy backward: clip+div kernel.

## Phase 5 — Fix build config ✅ DONE

### 17. `CMakeLists.txt` ✅
CUDA architecture set, all new .cu files registered.

## Files to modify

| File | Change |
|---|---|
| `cuda/ops/arithmetic.cu` | Implement from stub |
| `cuda/ops/activation.cu` | Implement from stub |
| `cuda/ops/linear.cu` | Implement from stub |
| `cuda/tensor.cu` | Implement from stub |
| `cuda/kernels/pool.cu` | **New file** — MaxPool2d kernels |
| `cuda/kernels/norm.cu` | **New file** — LayerNorm/RMSNorm kernels |
| `include/boat/cuda_runtime.h` | Add ~30 new function declarations |
| `src/ops/arithmetic.c` | Add CUDA dispatch to all element-wise/math/reduction ops |
| `src/ops/activation.c` | Add CUDA dispatch to softmax/relu/silu/gelu |
| `src/ops/linear.c` | Add CUDA dispatch to transpose/dot |
| `src/layers/attention.c` | Decompose scaled_dot_product into ops |
| `src/layers/pool.c` | Add CUDA dispatch |
| `src/layers/norm.c` | Add CUDA dispatch for layernorm/rmsnorm |
| `src/layers/relu.c` | Fix backward with cache+mask |
| `src/layers/dense.c` | Add bias gradient CUDA path |
| `src/optimizers/sgd.c` | Add CUDA update kernel, fix device for velocity buffers |
| `src/optimizers/adam.c` | Add CUDA update kernel, fix device for m/v buffers |
| `src/loss/mse.c` | Add CUDA forward/backward |
| `src/loss/cross_entropy.c` | Add CUDA forward/backward |
| `CMakeLists.txt` | Add new .cu files, fix CUDA arch |

## Verification

```bash
cd build_cuda && cmake .. -DBOAT_WITH_CUDA=ON -DBOAT_WITH_CUDNN=ON -DBOAT_WITH_TESTS=ON
cmake --build . --config Release
# Run existing tests with CUDA tensors
ctest -C Release
# Run benchmark to verify cuBLAS works
Release/benchmark_sgemm.exe
```
