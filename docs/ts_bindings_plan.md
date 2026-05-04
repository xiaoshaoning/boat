# TypeScript/JavaScript Bindings — Node.js N-API Plan

## Executive Summary

Create first-class TypeScript/JavaScript bindings for the boat deep learning framework. The bindings target two runtime environments:

| Runtime | Approach | Capabilities |
|---|---|---|
| **Node.js** | N-API native addon | Full C API (tensors, models, training), CUDA GPU support, async non-blocking |
| **Browser** | WebAssembly (Emscripten) | CPU inference only, model load/forward, limited tensor ops |

The N-API addon is the primary deliverable — it unlocks the full framework from Node.js/Deno. The WASM build is a secondary target for browser-side inference demos.

---

## Architecture Overview

```
TypeScript / JavaScript user code
        │
        ├── @boat-ai/core (npm package)
        │       ├── lib/index.ts          # High-level JS API
        │       ├── lib/types.ts          # TypeScript type definitions
        │       ├── binding.gyp           # node-gyp build
        │       └── src/                  # N-API C++ addon source
        │               ├── addon.cpp     # N-API module entry + exports
        │               ├── tensor.ts.cpp # Tensor bindings
        │               ├── model.ts.cpp  # Model bindings
        │               ├── ops.ts.cpp    # Operation bindings
        │               └── utils.ts.cpp  # Version, dtype helpers
        │
        └── @boat-ai/wasm (npm package)
                ├── lib/index.ts          # WASM JS/TS API
                ├── build/                # Emscripten output
                └── src/                  # Emscripten glue
```

### Data Flow

```
JS TypedArray (Float32Array)
    │
    ▼
N-API Buffer / External
    │
    ▼
boat_tensor_t* (C, CPU or CUDA)
    │
    ▼
boat_model_forward() / ops
    │
    ▼
boat_tensor_t* result
    │
    ▼
Copy data → JS TypedArray
```

**Memory ownership**: Every `boat_tensor_t*` created from JS is tracked in a `napi_ref` registry. The JS `Tensor` wrapper's destructor (`napi_wrapper_finalize`) calls `boat_tensor_unref`. Tensors returned from C functions are wrapped in new JS objects with fresh references.

---

## Package Structure

```
bindings/js/
├── package.json            # @boat-ai/core package
├── binding.gyp             # node-gyp configuration
├── tsconfig.json
├── jest.config.ts
├── src/
│   ├── addon.cpp           # N-API entry point
│   ├── tensor_wrapper.cpp  # Tensor JS↔C bridge
│   ├── model_wrapper.cpp   # Model JS↔C bridge
│   ├── ops_wrapper.cpp     # Operation bindings
│   ├── quantize_wrapper.cpp# Quantization bindings
│   ├── dtype_utils.cpp     # DType enum mapping
│   └── utils.cpp           # Version, error handling
├── lib/
│   ├── index.ts            # Public API re-exports
│   ├── tensor.ts           # Tensor class
│   ├── model.ts            # Model class
│   ├── ops.ts              # Standalone ops
│   ├── types.ts            # TS type definitions
│   └── errors.ts           # Custom error classes
├── wasm/
│   ├── CMakeLists.txt      # Emscripten build
│   ├── wasm_glue.cpp       # WASM entry C API subset
│   └── wasm_index.ts       # WASM runtime loader
├── test/
│   ├── tensor.test.ts
│   ├── model.test.ts
│   └── ops.test.ts
└── examples/
    ├── inference.ts
    └── training.ts
```

---

## Phase 1 — N-API Core Addon (2-3 weeks)

### 1.1 Build Setup

- Create `bindings/js/binding.gyp` targeting `boat` static library
- Resolve `boat.h` include path from CMake build tree
- Conditionally link CUDA libraries when boat was built with CUDA
- Support both `Release` and `Debug` builds
- Platform: Linux (x64), macOS (arm64, x64), Windows (x64)

Dependencies: `node-addon-api` (C++ wrapper for N-API), `node-gyp` or `cmake-js`.

Use `cmake-js` rather than raw `node-gyp` to integrate cleanly with boat's existing CMake build:

```cmake
# bindings/js/CMakeLists.txt (alternative to binding.gyp)
find_package(Nodejs REQUIRED)
add_library(boat_napi SHARED
    src/addon.cpp
    src/tensor_wrapper.cpp
    src/model_wrapper.cpp
    ...
)
target_link_libraries(boat_napi PRIVATE boat ${NODEJS_LIB})
target_include_directories(boat_napi PRIVATE ${NODEJS_INCLUDE_DIR})
```

### 1.2 N-API Module Structure

One module initializer that exports all classes and functions:

```cpp
// src/addon.cpp
#include <napi.h>
#include "boat/boat.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Tensor class
    auto tensor_class = TensorWrapper::DefineClass(env, "Tensor", {
        InstanceMethod("shape", &TensorWrapper::Shape),
        InstanceMethod("dtype", &TensorWrapper::Dtype),
        InstanceMethod("toString", &TensorWrapper::ToString),
        // ...
    });
    exports.Set("Tensor", tensor_class);

    // Model class
    auto model_class = ModelWrapper::DefineClass(env, "Model", {
        InstanceMethod("forward", &ModelWrapper::Forward),
        InstanceMethod("load", &ModelWrapper::Load),
        InstanceMethod("save", &ModelWrapper::Save),
        // ...
    });
    exports.Set("Model", model_class);

    // Standalone exports
    exports.Set("version", Napi::String::New(env, boat_get_version_string()));
    exports.Set("dtypeSize", Napi::Function::New(env, DtypeSize));
    // ...
    return exports;
}

NODE_API_MODULE(boat_napi, Init)
```

### 1.3 Tensor Wrapper

The core bridge type. Wraps an opaque `boat_tensor_t*` and manages its lifetime.

```cpp
// src/tensor_wrapper.cpp
class TensorWrapper : public Napi::ObjectWrap<TensorWrapper> {
    boat_tensor_t* tensor_;
    napi_ref self_ref_; // prevent GC while C holds the ref

public:
    TensorWrapper(const Napi::CallbackInfo& info) : tensor_(nullptr) {
        // Constructor overloads:
        // new Tensor(shape, dtype)          → boat_tensor_create
        // new Tensor(data: Float32Array)    → boat_tensor_from_data
        // new Tensor(data, shape, dtype)    → boat_tensor_from_data with shape
    }

    // Property accessors
    Napi::Value Shape(const Napi::CallbackInfo& info);
    Napi::Value Dtype(const Napi::CallbackInfo& info);
    Napi::Value Device(const Napi::CallbackInfo& info);
    Napi::Value Size(const Napi::CallbackInfo& info);
    Napi::Value Ndims(const Napi::CallbackInfo& info);

    // Data access
    Napi::Value ToFloat32Array(const Napi::CallbackInfo& info); // copy to JS
    Napi::Value ToTypedArray(const Napi::CallbackInfo& info);   // copy in dtype

    // Operations (return new Tensor)
    Napi::Value Reshape(const Napi::CallbackInfo& info);
    Napi::Value Transpose(const Napi::CallbackInfo& info);
    Napi::Value Slice(const Napi::CallbackInfo& info);
    Napi::Value Clone(const Napi::CallbackInfo& info);
    Napi::Value ToDevice(const Napi::CallbackInfo& info);

    // Arithmetic (static methods)
    static Napi::Value Add(const Napi::CallbackInfo& info);
    static Napi::Value Sub(const Napi::CallbackInfo& info);
    static Napi::Value Mul(const Napi::CallbackInfo& info);
    static Napi::Value Div(const Napi::CallbackInfo& info);
    static Napi::Value MatMul(const Napi::CallbackInfo& info);

    // Comparisons / utilities
    Napi::Value Equals(const Napi::CallbackInfo& info);
    Napi::Value AllClose(const Napi::CallbackInfo& info);
    Napi::Value ToString(const Napi::CallbackInfo& info);

    ~TensorWrapper() {
        if (tensor_) boat_tensor_unref(tensor_);
    }

    static boat_tensor_t* Unwrap(Napi::Object obj);
    static Napi::Object Wrap(boat_tensor_t* tensor);
};
```

**Data transfer**: `Float32Array` / `Float64Array` ↔ `boat_tensor_t*`. Copy on creation and read-back. Future optimization: support external/zero-copy for typed arrays where the tensor is on CPU.

### 1.4 Model Wrapper

```cpp
// src/model_wrapper.cpp
class ModelWrapper : public Napi::ObjectWrap<ModelWrapper> {
    boat_model_t* model_;

public:
    ModelWrapper(const Napi::CallbackInfo& info) {
        // new Model() → boat_model_create()
    }

    Napi::Value Forward(const Napi::CallbackInfo& info) {
        // Takes Tensor, returns Tensor
        Napi::Object input_obj = info[0].As<Napi::Object>();
        boat_tensor_t* input = TensorWrapper::Unwrap(input_obj);
        boat_tensor_t* output = boat_model_forward(model_, input);
        return TensorWrapper::Wrap(output, env);
    }

    Napi::Value Backward(const Napi::CallbackInfo& info);
    Napi::Value AddLayer(const Napi::CallbackInfo& info);
    Napi::Value Load(const Napi::CallbackInfo& info);  // from file path
    Napi::Value Save(const Napi::CallbackInfo& info);

    ~ModelWrapper() {
        if (model_) boat_model_free(model_);
    }
};
```

**Async forward**: For inference with large models, wrap in `Napi::AsyncWorker` to avoid blocking the event loop:

```cpp
class AsyncForward : public Napi::AsyncWorker {
    boat_model_t* model_;
    boat_tensor_t* input_;
    boat_tensor_t* output_;
public:
    void Execute() override {
        output_ = boat_model_forward(model_, input_);
    }
    void OnOK() override {
        auto result = TensorWrapper::Wrap(output_);
        Callback().Call({env().Null(), result});
    }
};
```

### 1.5 Dtype Mapping

```cpp
// src/dtype_utils.cpp
constexpr const char* BOAT_DTYPE_NAMES[] = {
    "float64", "float32", "float16", "float8", "float4",
    "int64", "int32", "uint8", "int8",
    "bits2", "bits1",
    "bool", "bfloat16"
};

Napi::Number DtypeSize(const Napi::CallbackInfo& info) {
    int dtype = info[0].As<Napi::Number>().Int32Value();
    return Napi::Number::New(info.Env(), boat_dtype_size((boat_dtype_t)dtype));
}
```

### 1.6 Error Handling

Map C errors to JS exceptions:

```cpp
class ErrorConverter {
public:
    static void MaybeThrow(Napi::Env env, int error_code) {
        if (error_code != 0) {
            throw Napi::Error::New(env, boat_error_message(error_code));
        }
    }
};
```

---

## Phase 2 — TypeScript Type Definitions (1 week)

### 2.1 Types

```typescript
// lib/types.ts
export type DType =
    | 'float64' | 'float32' | 'float16' | 'float8' | 'float4'
    | 'int64' | 'int32' | 'uint8'
    | 'bfloat16' | 'bool';

export type Device = 'cpu' | 'cuda';

export type Shape = readonly number[];
```

### 2.2 Tensor Class

```typescript
// lib/tensor.ts
export class Tensor {
    /** Create tensor from shape + fill */
    constructor(shape: Shape, dtype?: DType, fill?: number);

    /** Create tensor from a flat TypedArray */
    constructor(data: TypedArray, shape?: Shape, dtype?: DType);

    /** Properties */
    get shape(): Shape;
    get dtype(): DType;
    get device(): Device;
    get size(): number;      // total elements
    get ndim(): number;
    get nbytes(): number;

    /** Data access — copies from C to JS */
    toFloat32Array(): Float32Array;
    toTypedArray(): TypedArray;
    toString(): string;

    /** Manipulation */
    reshape(...shape: number[]): Tensor;
    transpose(...axes: number[]): Tensor;
    slice(start: number[], end: number[]): Tensor;
    clone(): Tensor;
    toDevice(device: Device): Tensor;

    /** Arithmetic — returns new Tensor */
    add(other: Tensor | number): Tensor;
    sub(other: Tensor | number): Tensor;
    mul(other: Tensor | number): Tensor;
    div(other: Tensor | number): Tensor;
    matmul(other: Tensor): Tensor;

    /** Comparison */
    equals(other: Tensor, tolerance?: number): boolean;
}
```

### 2.3 Model Class

```typescript
// lib/model.ts
export class Model {
    constructor();

    forward(input: Tensor): Tensor;
    backward(grad: Tensor): Tensor;

    addLayer(layer: Layer): void;
    load(path: string): void;
    save(path: string): void;

    /** Async inference — runs on libuv thread pool */
    forwardAsync(input: Tensor): Promise<Tensor>;
}
```

### 2.4 Layer Definitions

```typescript
export interface LayerOptions {
    type: 'dense' | 'conv2d' | 'lstm' | 'gru' | 'flatten' | 'dropout';
    // type-specific options
    inFeatures?: number;
    outFeatures?: number;
    kernelSize?: number | [number, number];
    hiddenSize?: number;
    // ...
}
```

### 2.5 Standalone Ops

```typescript
// lib/ops.ts
export function matmul(a: Tensor, b: Tensor): Tensor;
export function add(a: Tensor, b: Tensor | number): Tensor;
export function relu(x: Tensor): Tensor;
export function softmax(x: Tensor, dim?: number): Tensor;
export function conv2d(input: Tensor, kernel: Tensor, options?: ConvOptions): Tensor;
// ...
```

---

## Phase 3 — High-Level JS API (1 week)

### 3.1 Convenience Construction

```typescript
// lib/index.ts
export function tensor(data: number[] | TypedArray, shape?: Shape): Tensor;
export function zeros(shape: Shape): Tensor;
export function ones(shape: Shape): Tensor;
export function randn(shape: Shape, mean?: number, std?: number): Tensor;
export function arange(start: number, end?: number, step?: number): Tensor;
```

### 3.2 Optimizer + Loss Helpers

```typescript
// lib/training.ts
export class Optimizer {
    constructor(model: Model, type: 'adam' | 'sgd', lr: number);
    step(): void;
    zeroGrad(): void;
}

export function crossEntropyLoss(predicted: Tensor, target: Tensor): number;
export function mseLoss(predicted: Tensor, target: Tensor): number;
```

### 3.3 Serialization

```typescript
// lib/serialization.ts
export function saveModel(model: Model, path: string): void;
export function loadModel(path: string): Model;
export function exportOnnx(model: Model, path: string): void;
```

### 3.4 Adapter for ONNX Runtime

Separate optional package `@boat-ai/ort` that bridges boat tensors to ONNX Runtime JS bindings for execution on ORT backends (DirectML, CoreML, etc.):

```typescript
// lib/ort_adapter.ts
export function boatTensorToOrt(t: Tensor): ort.Tensor;
export function ortTensorToBoat(t: ort.Tensor): Tensor;
```

---

## Phase 4 — WebAssembly Build (2 weeks)

### 4.1 Build Setup

Use Emscripten to compile a subset of boat to WASM:

```cmake
# bindings/js/wasm/CMakeLists.txt
set(CMAKE_TOOLCHAIN_FILE ${EMSCRIPTEN}/cmake/Modules/Platform/Emscripten.cmake)

add_executable(boat_wasm wasm_glue.cpp)
target_link_options(boat_wasm PRIVATE
    -s EXPORTED_FUNCTIONS='["_boat_wasm_init", "_boat_wasm_create_tensor", ...]'
    -s EXPORTED_RUNTIME_METHODS='["ccall", "getValue", "setValue"]'
    -s ALLOW_MEMORY_GROWTH=1
    --pre-js pre.js
)
```

### 4.2 WASM API Subset

WASM cannot support the full framework (no CUDA, no file system directly), so expose:

```c
// src/wasm_glue.c
// Tensor: create, from_data, shape, dtype, basic ops
// Model: load (from in-memory bytes), forward
// No CUDA, no GPU ops, no file I/O
```

### 4.3 WASM Loader

```typescript
// wasm/wasm_index.ts
export async function initBoatWasm(wasmUrl?: string): Promise<BoatWasmModule> {
    const module = await createWasmModule({
        wasmUrl: wasmUrl || defaultWasmUrl,
    });
    return new BoatWasmModule(module);
}
```

---

## Phase 5 — Distribution & Tooling (1 week)

### 5.1 npm Packages

| Package | Contents | Platform |
|---|---|---|
| `@boat-ai/core` | N-API addon + TS types | Node.js (prebuild binaries) |
| `@boat-ai/core-darwin-arm64` | macOS ARM64 binary | macOS |
| `@boat-ai/core-darwin-x64` | macOS x64 binary | macOS |
| `@boat-ai/core-linux-x64` | Linux x64 binary | Linux |
| `@boat-ai/core-win32-x64` | Windows x64 binary | Windows |
| `@boat-ai/wasm` | WASM module + TS types | Browser / Node.js |
| `@boat-ai/cli` | CLI tools (npx boat) | Cross-platform |

Use `prebuildify` to precompile binaries per platform so users `npm install` without needing C toolchain.

### 5.2 CI Integration

Extend existing GitHub Actions with a JS binding workflow:

```yaml
# .github/workflows/ci-js.yml
jobs:
  build-napi:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
    - uses: actions/setup-node@v4
    - run: npm install -g cmake-js
    - run: cmake-js build
    - run: npm test

  build-wasm:
    runs-on: ubuntu-latest
    steps:
    - uses: mymindstorm/setup-emsdk@v14
    - run: emcmake cmake ..
    - run: npm test

  publish:
    if: startsWith(github.ref, 'refs/tags/js-v')
    needs: [build-napi, build-wasm]
    runs-on: ubuntu-latest
    steps:
    - run: npx lerna publish from-git
```

### 5.3 Pre-built Binaries

```bash
# publish flow
npm run build:linux
npm run build:macos
npm run build:windows
npx prebuildify --napi
npx lerna publish
```

---

## API Design Principles

1. **Zero-copy typed arrays where possible**: On CPU, create tensors that reference existing TypedArray memory without copy (via `napi_create_external_arraybuffer`).
2. **Async by convention**: `model.forwardAsync()` for non-blocking inference; synchronous `forward()` for small models.
3. **Memory safety**: Every `boat_tensor_t*` is wrapped in a `napi_wrapper` with a finalizer that decrements the refcount. No manual `free()` from JS.
4. **Error conversion**: All C error codes → typed JS exceptions (e.g., `ShapeMismatchError`, `OutOfMemoryError`).
5. **TS-first types**: Every function has full TypeScript declarations with generics where useful (e.g., `Tensor<Shape>` in the future).
6. **Consistent naming**: camelCase in JS maps to snake_case in C (e.g., `tensor.reshape()` → `boat_tensor_reshape`).

---

## Effort Summary

| Phase | Duration | Deliverable |
|---|---|---|
| Phase 1: N-API Core | 2-3 weeks | `@boat-ai/core` native addon (Tensor + Model + ops) |
| Phase 2: TypeScript Types | 1 week | `.d.ts` files, type safety, JSDoc |
| Phase 3: High-Level JS API | 1 week | Convenience constructors, training helpers, serialization |
| Phase 4: WebAssembly | 2 weeks | `@boat-ai/wasm` CPU inference in browser |
| Phase 5: Distribution | 1 week | npm packages, prebuild binaries, CI |
| **Total** | **7-8 weeks** | |

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| **N-API ABI stability across Node versions** | Use `node-addon-api` C++ wrapper which handles ABI; test on Node 18/20/22 |
| **Large binary size** | Split into platform-specific packages; strip debug symbols |
| **MSVC + CUDA + N-API linking issues on Windows** | Dependency on CUDA is optional; native addon loads boat.dll at runtime |
| **WASM memory limit** | Use `ALLOW_MEMORY_GROWTH=1`; lazy weight loading for large models |
| **Thread safety** | Model forward is not thread-safe; add mutex lock in JS wrapper for concurrent calls |

---

## Future Extensions

- **Deno native plugin**: Reuse N-API addon via `Deno.dlopen`
- **Bun native plugin**: Microseconds-resolution FFI in Bun
- **TypeScript tensor generics**: `Tensor<[3, 224, 224]>` for compile-time shape checking
- **WebGPU backend**: Compute shader inference in browser (beyond WASM)
- **Vite/Next.js plugin**: Optimize WASM loading for framework bundlers
- **Jupyter Notebook kernel**: ijavascript kernel with boat tensor display
