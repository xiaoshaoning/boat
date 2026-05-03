# CUDA/cuDNN Full Support

## Context

The boat framework has a well-designed CUDA infrastructure at the lower layers (memory copy/set, tensor allocation, kernel files in `cuda/kernels/`, cuBLAS/cuDNN handles), but the upper layers are entirely CPU. Six `.cu` stub files are empty, the ops layer lacks device dispatch for all but `boat_matmul`, and layers/optimizers/loss/autodiff have no CUDA paths.

"Full support" means: fill the stubs, add device dispatch to all ops, wire up layer CUDA paths, add optimizer/loss CUDA kernels, and fix the cubLAS/cuDNN configuration for the existing build.

---

## Phase 1 — Fill stub .cu files (core wiring)

### 1. `cuda/ops/arithmetic.cu` (~300 lines)
New kernels (following basic.cu grid-stride pattern):
- `boat_cuda_exp_f32`, `boat_cuda_log_f32`, `boat_cuda_sqrt_f32`, `boat_cuda_rsqrt_f32`
- `boat_cuda_neg_f32`, `boat_cuda_abs_f32`, `boat_cuda_mod_f32`
- `boat_cuda_sub_scalar_f32`, `boat_cuda_div_scalar_f32`
- `boat_cuda_fill_f32`, `boat_cuda_scale_f32`

Note: add/sub/mul/div/relu/sigmoid/tanh/silu/add_scalar/mul_scalar wrappers ALREADY exist in `kernels/basic.cu` — arithmetic.cu only needs the NEW ones above.

### 2. `cuda/ops/activation.cu` (~400 lines)
- Softmax kernel: one block per row, shared memory reduction for max→exp→sum→normalize
- LogSoftmax kernel: same pattern with log
- GELU kernel (tanh-approximation): element-wise

### 3. `cuda/ops/linear.cu` (~250 lines)
- Transpose: tiled 2D (16×16 shared mem) + general N-D fallback
- Dot product: 1D reduction

### 4. `cuda/tensor.cu` (~100 lines)
- `boat_cuda_tensor_clone`, `boat_cuda_tensor_to_host`, `boat_cuda_tensor_to_device`

### 5. New: `cuda/kernels/pool.cu` (~250 lines)
- MaxPool2d forward: one thread per output element
- MaxPool2d backward: atomicAdd scatter from output to input positions

### 6. New: `cuda/kernels/norm.cu` (~400 lines)
- LayerNorm forward: one block per row, shared-memory mean+var, normalize
- LayerNorm backward: three shared-memory reductions for gradient computation
- RMSNorm forward: simpler (no mean), one block per row
- RMSNorm backward

## Phase 2 — Device dispatch in CPU ops

### 7. `src/ops/arithmetic.c` — Element-wise ops CUDA dispatch
Add CUDA path to: add/sub/mul/div, all scalar variants, mod, exp/log/sqrt/rsqrt, abs/neg, clamp, sum/mean/var reductions. Pattern:
```c
#ifdef BOAT_WITH_CUDA
    if (boat_tensor_device(a) == BOAT_DEVICE_CUDA) {
        if (dtype == BOAT_DTYPE_FLOAT32) {
            boat_cuda_add_f32(in1, in2, out, n);
            return out;
        }
    }
#endif
```

### 8. `src/ops/activation.c` — Softmax + activations CUDA dispatch
- `boat_softmax`: dispatch to new CUDA kernel with (outer, axis_size, inner) signature
- `boat_relu/silu`: dispatch to existing kernels in basic.cu
- `boat_gelu`: dispatch to new kernel

### 9. `src/ops/linear.c` — Transpose + dot CUDA dispatch
- `boat_transpose`: dispatch to new tiled CUDA kernel
- `boat_dot`: dispatch to reduction kernel
- `boat_matmul`: already has cuBLAS path ✓

## Phase 3 — Layer CUDA paths

### 10. `src/layers/attention.c` — Decompose into ops (~100 lines refactor)
Replace manual 6-nested-loop scaled_dot_product_attention with op chain:
```
scores = matmul(Q, transpose(K)) × scale → softmax → matmul(weights, V)
```
Each op in this chain gets CUDA from Phases 1-2 + existing cuBLAS matmul.

### 11. `src/layers/pool.c` — MaxPool2d CUDA dispatch
Forward: launch new kernel. Backward: atomicAdd scatter.

### 12. `src/layers/norm.c` — LayerNorm/RMSNorm CUDA dispatch
Forward/backward: launch new kernels from phase 1.

### 13. `src/layers/relu.c` — Fix backward
- Add cache_input, proper backward masking with `boat_cuda_relu_backward_f32`

### 14. `src/layers/dense.c` — Bias gradient CUDA path
- Small CUDA sum-axis reduction for bias backward

## Phase 4 — Optimizer + Loss CUDA paths

### 15. `src/optimizers/sgd.c` + `adam.c`
- SGD update kernel: `param -= lr * grad`
- SGD momentum kernel: with Nesterov support
- Adam update kernel: fused m/v/param update
- **Fix:** velocity/momentum buffers should be created on same device as params (currently hardcoded CPU)

### 16. `src/loss/mse.c` + `cross_entropy.c`
- MSE forward: device-side diff→square→sum, copy scalar to host
- MSE backward: `grad = 2*(pred-target)/N` element-wise kernel
- Cross-entropy backward: clip+div kernel

## Phase 5 — Fix build config

### 17. `CMakeLists.txt` CUDA architecture fix
Current: `set(CMAKE_CUDA_ARCHITECTURES "100")` — Blackwell-only, should be `"89"` (Ada Lovelace / RTX 40xx) or `"75"` (Turing / RTX 20xx) for current hardware.

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
