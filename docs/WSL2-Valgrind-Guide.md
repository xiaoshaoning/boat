# WSL2 Compatibility & Valgrind Memory Check

Date: 2026-08-11
Scope: verify Boat builds and passes its test suite on WSL2, and run a full
Valgrind Memcheck pass over the library, tests, and examples.

## Summary

- Boat builds cleanly on WSL2 (Ubuntu 22.04, gcc 11.4) and passes **28/28
  tests** — same configuration the Linux CI uses, plus self-contained
  ONNX/GGUF.
- Valgrind Memcheck over all 28 test binaries: **0 errors, 0 bytes leaked**
  in every category.
- Four real memory bugs were found in the examples and one library defect in
  `src/layers/dense.c`; all were fixed and verified in a sandbox copy.
  The upstream repository was **not modified** — the exact patch is included
  at the end of this document.
- The `examples/regression` demo's high reported test loss is a data-split
  artifact (contiguous split = extrapolation), not a library bug; proven by
  re-running with a shuffled split.

## Environment

| Component | Version |
|---|---|
| OS | WSL2 (Ubuntu 22.04), kernel 6.18.33.2-microsoft-standard-WSL2 |
| CPU | 16 cores (12th Gen Intel i7-1260P) |
| gcc / g++ | 11.4.0 |
| cmake | 3.22.1 |
| make | 4.3 |
| valgrind | 3.18.1 |

## Build & test on WSL2

```bash
cmake -S . -B build-linux \
  -DBOAT_WITH_TESTS=ON \
  -DBOAT_WITH_EXAMPLES=ON \
  -DBOAT_WITH_HUGGINGFACE=ON \
  -DBOAT_WITH_ONNX=ON \
  -DBOAT_WITH_GGUF=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

cmake --build build-linux -j16
ctest --output-on-failure -j16
```

Result: build 100% clean; **28/28 tests passed** (re-verified after the
`dense.c` fix below). `examples/nanochat` and `examples/latex` are skipped as
expected (CUDA examples).

Windows-specific code is properly isolated:
- All `windows.h` / `__declspec` / Win32 API usage is inside `#ifdef _WIN32`.
- `tests/unit/test_export.c` (LoadLibrary) and `benchmarks/attention_performance.c`
  (Windows timing) are excluded by `if(WIN32)` in `tests/CMakeLists.txt`.

## Valgrind methodology

All binaries run with `OMP_NUM_THREADS=1` (Valgrind serializes threads anyway;
single-threaded also avoids OpenMP runtime noise):

```bash
OMP_NUM_THREADS=1 valgrind \
  --leak-check=full --show-leak-kinds=all --track-origins=yes \
  --num-callers=30 --error-exitcode=42 ./<binary>
```

`--error-exitcode=42` makes any error or definite/possible leak fail the run.

## Results

### Library tests: clean

All 28 test binaries: `ERROR SUMMARY: 0 errors from 0 contexts`, and
"All heap blocks were freed -- no leaks are possible" (or 0 bytes in every
leak category). No invalid reads/writes, no uninitialized-value uses, no
definite/indirect/possible leaks.

### Examples: 4 bugs found and fixed (verified in sandbox)

| # | File | Bug | Valgrind before | After fix |
|---|---|---|---|---|
| 1 | `examples/regression/regression.c` | Caller double-unrefs the gradient that `backward_reg`/`backward_ts` already consumed → use-after-free on `tensor->ref_count` (`tensor.c:196`) | 4000 errors | 0 errors |
| 2 | `examples/serialization/serialization.c` | `boat_model_free(model)` already frees layer data via `dense_free_op` (`model.c:525`); example then freed the same layers again (stale comment claimed `ops->free` is NULL) | 36 errors | 0 errors, all tests PASS |
| 3 | `src/layers/dense.c` (library) | `boat_dense_layer_create` left `grad_weight`/`grad_bias` uninitialized → getters returned garbage heap pointers → `adam_optimizer_add_parameter` accepted them → Adam applied garbage gradients (undefined behavior) | — | 0 errors; probe verifies correct `lr·grad` update |
| 4 | `examples/transformer/transformer.c` | `x` and `logits` never unref'd in the training loop → per-iteration leaks | 253 errors; 17,280 B definitely + 10,989,120 B indirectly lost | 0 errors, 0 bytes lost |

### Root-cause detail

**#1 (regression.c).** `backward_reg`/`backward_ts` consume the caller's
gradient reference on every path (error path `regression.c:151`, success path
`regression.c:152`), but the training loop unref'd `grad` again afterwards.
`boat_tensor_unref` decrements `ref_count` and frees at 0, so the second unref
reads freed memory at `tensor.c:196`. Fix: remove the caller's unref.

**#2 (serialization.c).** `boat_model_free` calls each layer's `ops->free`
(e.g. `dense_free_op` at `model.c:525`), which frees both the layer data and
the wrapper. The example's manual `boat_dense_layer_free(fc1)` etc. double-
freed. Note: the `ref_*` layers in that file are standalone (never added to a
model) and their manual frees are correct — only the `fc1/fc2/relu/sm` frees
after `boat_model_free` were wrong.

**#3 (dense.c).** `boat_dense_layer_create` never assigned
`layer->grad_weight = NULL` / `grad_bias = NULL`, and `boat_malloc` does not
zero memory. The unit tests never noticed because they pass explicitly created
grad tensors to the optimizer; the examples register the layer getters, which
returned garbage. Fix: zero-init in `create` and lazily create zero-filled
grad tensors in `boat_dense_layer_get_grad_weight` /
`boat_dense_layer_get_grad_bias` (matching the accumulation contract in
`boat_dense_layer_backward`, which already `boat_add_`s into a pre-existing
`grad_weight`). After the fix the regression demo's gradient flow is real.

**#4 (transformer.c).** The training loop created `x` (`transformer.c:600`)
and `logits` (`transformer.c:664`) each iteration and never unref'd them; all
other loop temporaries are cleaned up. Fix: add the two missing
`boat_tensor_unref` calls. The fixed example still trains (loss
0.076 → 0.049 over 60 epochs) and generates text.

## Investigation notes (not bugs)

### regression demo "test MSE = 2.27"

The demo normalizes x to `xs[i] = i/(n-1) ∈ [0,1]` and splits contiguously:

```
train: i = 0..1599   →  x ∈ [0, 0.80]
test:  i = 1600..1999 → x ∈ [0.80, 1.00]   (disjoint!)
```

The test set is therefore pure extrapolation; the model overfits the train
region and generalizes badly beyond it. The timeseries demo (same training
path) converges to test MSE 0.000002. Re-running the identical training with a
shuffled i.i.d. split converges to **0.086** — the library trains correctly.
Recommendation: shuffle the split before partitioning (or interleave).

### OpenMP thread-spawn overhead

With 16 threads, `test_loss_integration` and `test_optimizer_benchmark` take
~113 s each; with `OMP_NUM_THREADS=1` they take ~0.6 s. Correctness is
unaffected; the loops pay thread-creation/synchronization cost per iteration.
Worth profiling if test time matters.

### Data-dependent examples (MNIST / CIFAR / translator)

These require data files (`data/…`, a model dir) produced by the Python
scripts or downloads. In a network-restricted WSL they cannot be fully
exercised; the code paths they did run were valgrind-clean (0 errors before
the expected "data not found" exit).

## How to reproduce

```bash
# in WSL2
cp -r /mnt/<path-to-repo> ~/boat-wsl   # or git clone
cd ~/boat-wsl
cmake -S . -B build-linux -DBOAT_WITH_TESTS=ON -DBOAT_WITH_EXAMPLES=ON \
  -DBOAT_WITH_HUGGINGFACE=ON -DBOAT_WITH_ONNX=ON -DBOAT_WITH_GGUF=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-linux -j16
ctest --output-on-failure -j16

cd build-linux/tests
for t in test_*; do [ -x "$t" ] || continue
  OMP_NUM_THREADS=1 valgrind --leak-check=full --show-leak-kinds=all \
    --track-origins=yes --error-exitcode=42 ./"$t"; done
```

## Patch to apply upstream

The following changes were applied and verified in the sandbox
(`~/boat-wsl`); the upstream repository was left untouched.

```diff
--- a/src/layers/dense.c
@@ boat_dense_layer_create
     layer->use_bias = use_bias;
     layer->cache_input = NULL;
+    layer->grad_weight = NULL;
+    layer->grad_bias = NULL;

 // getters: lazily create zero-filled tensors instead of returning garbage
 BOAT_API boat_tensor_t* boat_dense_layer_get_grad_weight(const boat_dense_layer_t* layer) {
     if (!layer) return NULL;
-    return layer->grad_weight;
+    boat_dense_layer_t* l = (boat_dense_layer_t*)layer;
+    if (!l->grad_weight) {
+        l->grad_weight = boat_tensor_create_like(l->weight);
+        if (l->grad_weight) {
+            boat_memory_set(boat_tensor_data(l->grad_weight), 0,
+                            boat_tensor_nbytes(l->grad_weight),
+                            boat_tensor_device(l->grad_weight));
+        }
+    }
+    return l->grad_weight;
 }
 // (same pattern for boat_dense_layer_get_grad_bias with l->bias)

--- a/examples/regression/regression.c
@@ training loop (2 call sites: regression and timeseries demos)
     if (grad) {
-        backward_reg(model, grad);
-        boat_tensor_unref(grad);
+        backward_reg(model, grad);  // backward_reg consumes the grad reference
     }
```

```diff
--- a/examples/serialization/serialization.c
@@ step_create_and_save
-    // Clean up — model_free also frees the layer wrappers, but NOT the
-    // underlying layer data (fc1, fc2, relu, sm) since ops->free is NULL.
-    // We free the layers manually.
+    // Clean up — model_free frees the layer wrappers AND the underlying
+    // layer data via each layer type's ops->free (e.g. dense_free_op).
+    // The layer handles (fc1, fc2, relu, sm) must NOT be freed again.
     boat_model_free(model);
-    boat_dense_layer_free(fc1);
-    boat_relu_layer_free(relu);
-    boat_dense_layer_free(fc2);
-    boat_softmax_layer_free(sm);
```

```diff
--- a/examples/transformer/transformer.c
@@ training loop
         boat_tensor_unref(probs);
+        boat_tensor_unref(logits);
         // ---- Backward pass ----
...
         if (block_out) boat_tensor_unref(block_out);
+        boat_tensor_unref(x);
```

Verification status of the patch: all 28 tests still pass after the
`dense.c` change; regression, serialization, and transformer examples are
valgrind-clean (0 errors, 0 bytes lost) with the fixes applied.
