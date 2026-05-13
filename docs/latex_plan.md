# Nougat-LaTeX OCR Plan — Framework-First Approach

## 1. Objectives

1. **Deploy Nougat-LaTeX**: Image → LaTeX OCR inference using boat framework.
2. **Enrich the framework**: Extract generically useful components into boat's main library so future vision and seq2seq models benefit without reinventing.

## 2. Model Overview

| Component | Architecture | Config |
|---|---|---|
| **Encoder** | Donut-Swin Transformer | embed_dim=128, depths=[2,2,14,2], num_heads=[4,8,16,32], window_size=7, image_size=[224,560] |
| **Decoder** | mBART (10-layer) | d_model=1024, heads=16, d_ff=4096, vocab=50000, max_pos=4096, GELU |
| **LM Head** | Linear(1024→50000) | weight-tied with embedding |
| **Tokenizer** | BPE | vocab=50000, special 22 tokens |

Weights: ~712MB F32, safetensors at `C:\llm\nougat-latex-base\`.

## 3. What Goes Into the Framework vs. the Example

```
boat/
├── include/boat/
│   ├── layers/
│   │   ├── swin.h           ← NEW: Swin Transformer (generic vision backbone)
│   │   └── transformer_decoder.h ← NEW: Cross-attention decoder block
│   ├── tokenizers/
│   │   └── bpe.h            ← NEW: BPE tokenizer subsystem
│   └── data/
│       └── ...              ← EXTEND: image loading transforms
├── src/
│   ├── layers/
│   │   ├── swin.c           ← NEW
│   │   └── transformer_decoder.c ← NEW
│   ├── tokenizers/
│   │   └── bpe.c            ← NEW
│   └── data/
│       └── ...              ← EXTEND
│
└── examples/latex/          ← Model-specific glue code only
    ├── CMakeLists.txt
    ├── latex.c              ← Main: load → encode → decode → output
    ├── nougat_model.c/h     ← Weight name mapping + container struct
    └── nougat_decoder.c/h   ← 10-layer decoder stack built on boat_transformer_decoder
```

### 3.1 Framework: `src/layers/swin.c` + `include/boat/layers/swin.h`

**Why in framework**: Swin Transformer is a production-grade vision backbone used by Donut, DINOv2, SwinIR, many more. Any future vision model in boat would benefit.

**API**:

```c
// Config
typedef struct {
    int32_t embed_dim;          // 128
    int32_t depths[4];          // [2,2,14,2] — blocks per stage
    int32_t num_heads[4];       // [4,8,16,32]
    int32_t window_size;        // 7
    int32_t num_channels;       // 3 (RGB)
    int32_t image_height, image_width;
    float   drop_path_rate;     // 0.1
} boat_swin_config_t;

// Forward: [N,C,H,W] → [N, num_patches, embed_dim_stage4]
boat_tensor_t* boat_swin_forward(
    const boat_swin_config_t* cfg,
    const boat_tensor_t* input,              // [N, C, H, W]
    const boat_swin_weights_t* weights,       // all learned params
    boat_swin_cache_t* cache                  // optional pre-computed attn masks
);
```

**Internals**:
- `boat_swin_patch_embed()` — Conv2d(3, embed_dim, 4, 4) + LayerNorm
- `boat_swin_block_forward()` — window attention + MLP with residual
- Window partition / reverse — generic helper for arbitrary feature map sizes
- Cyclic shift (`torch.roll`) for SW-MSA
- Relative position bias computation from learned table
- `boat_swin_patch_merging()` — 2×2 concat + LayerNorm + Linear(4d→2d)

### 3.2 Framework: `src/layers/transformer_decoder.c` + `include/boat/layers/transformer_decoder.h`

**Why in framework**: A transformer decoder with cross-attention is the backbone of every seq2seq model (T5, BART, mBART, MarianMT, Nougat). Boat has `attention.c` for the core MHA but lacks a full DecoderLayer. Making this a framework primitive means future seq2seq models require zero new code.

**API**:

```c
typedef struct {
    int32_t d_model;
    int32_t num_heads;
    int32_t d_ff;
    float   dropout;
    bool    pre_norm;            // true for mBART
    bool    use_learned_pos;     // true: learned embedding, false: sinusoidal
    int32_t max_position;
    const char* activation;      // "gelu" or "relu"
} boat_decoder_config_t;

// Single decoder layer forward
boat_tensor_t* boat_decoder_layer_forward(
    const boat_tensor_t* x,                    // [B, T, D]
    const boat_tensor_t* encoder_hidden,       // [B, S, D]
    boat_tensor_t* self_k_cache,               // KV cache (mutated)
    boat_tensor_t* self_v_cache,
    const boat_decoder_weights_t* w,
    const boat_decoder_config_t* cfg,
    boat_decoder_mask_t* mask                  // causal + padding masks
);
```

**KV Cache**: Built into the layer call — the cache tensors are passed in and mutated in-place, appended each step. No separate cache management needed at the example level.

### 3.3 Framework: `src/tokenizers/bpe.c` + `include/boat/tokenizers/bpe.h`

**Why in framework**: BPE is the most common tokenization algorithm for modern LLMs and seq2seq models. Boat needs a proper tokenizer abstraction. Starting with BPE (decode-only is sufficient for inference; encode can be added later).

**API**:

```c
typedef struct boat_bpe_tokenizer_t boat_bpe_tokenizer_t;

boat_bpe_tokenizer_t* boat_bpe_tokenizer_create(const char* tokenizer_json_path);
void boat_bpe_tokenizer_free(boat_bpe_tokenizer_t* tok);

// Decode: token IDs → string
char* boat_bpe_tokenizer_decode(boat_bpe_tokenizer_t* tok, 
                                 const int32_t* ids, size_t n_ids);

// Encode: string → token IDs (needed for evaluation)
int32_t* boat_bpe_tokenizer_encode(boat_bpe_tokenizer_t* tok,
                                    const char* text, size_t* out_len);

// Utilities
int32_t boat_bpe_tokenizer_bos_id(boat_bpe_tokenizer_t* tok);
int32_t boat_bpe_tokenizer_eos_id(boat_bpe_tokenizer_t* tok);
int32_t boat_bpe_tokenizer_pad_id(boat_bpe_tokenizer_t* tok);
size_t  boat_bpe_tokenizer_vocab_size(boat_bpe_tokenizer_t* tok);
```

**Decode-only Strategy for Phase 1**: Load `tokenizer.json` via cJSON, extract the `model.vocab` (id → token string mapping). Decode is a simple table lookup + special token handling. The full BPE merge logic for encoding can be added later when needed.

### 3.4 Framework (Extended): Image loading in `src/data/`

**What exists**: `src/data/` has tensor datasets, dataloaders, transforms (normalize, flip, crop).

**What to add**:
- `boat_image_load(const char* path, int* w, int* h, int* c)` — wraps stb_image (already available)
- `boat_image_resize(const boat_tensor_t* img, int new_h, int new_w)` — bilinear interpolation
- `boat_image_to_tensor(const uint8_t* pixels, int h, int w, int c)` — HWC→CHW + rescale

These are fundamentally general-purpose operations that any computer vision pipeline needs.

### 3.5 Example-Only: `examples/latex/nougat_model.c/h`

**Why NOT in framework**: This contains the Nougat-specific weight name mapping (safetensors tensor names → C struct fields), which is model-specific. The pattern (`nougat_model_weights_t` with named loading) is the right level of specificity for an example.

```c
typedef struct {
    // Pointers into boat_tensor_t* obtained from boat_hf_load_file()
    // Organized by component:
    // - Encoder weights (Swin): patch_embed, 4 stages, 3 downsamples
    // - Decoder weights (mBART): embedding, 10 layers, final LN, LM head
} nougat_model_t;

nougat_model_t* nougat_model_create(const char* model_dir);
void nougat_model_free(nougat_model_t* model);
```

### 3.6 Example-Only: `examples/latex/nougat_decoder.c/h`

**Why NOT in framework**: While `boat_transformer_decoder_layer_t` handles a single layer, the full 10-layer stack with embedding, layernorm_embedding, final_norm, LM head, and autoregressive loop is model-specific enough to live in the example.

The example decoder orchestrates:
```
nougat_decoder_forward():
  x = embed(input_ids) * sqrt(1024) + learned_pos[positions]
  x = layernorm_embedding(x)
  for each of 10 layers:
      x = boat_decoder_layer_forward(x, encoder_output, kv_cache[layer], ...)
  x = final_ln(x)
  return lm_head(x)  // + weight tying with embedding
```

The autoregressive loop:
```c
for (int step = 0; step < max_steps; step++) {
    boat_tensor_t* logits = nougat_decoder_forward(model, input_ids, step);
    next_id = boat_sampling_argmax(logits);  // or top-k
    if (next_id == eos_id) break;
    append_to_input(input_ids, next_id);
}
```

## 4. Data Flow (Updated)

```
┌─────────────────────────────────────────────┐
│              Input Image                     │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  boat_image_load()          [framework]    │
│  boat_image_to_tensor()     [framework]    │
│  boat_image_resize()        [framework]    │
│  Normalize (μ,σ) via ops    [framework]    │
└──────────────┬──────────────────────────────┘
               │ [1,3,224,560] F32
               ▼
┌─────────────────────────────────────────────┐
│  boat_swin_forward()         [framework]    │
│  (4 stages, window attn)                   │
└──────────────┬──────────────────────────────┘
               │ [1, 119, 1024]
               ▼
┌─────────────────────────────────────────────┐
│  nougat_decoder_forward()    [example]       │
│  (embed + 10× boat_decoder_layer + lm_head) │
│    └── boat_decoder_layer()   [framework]   │
│    └── boat_sampling_*()      [framework]   │
└──────────────┬──────────────────────────────┘
               │ token IDs
               ▼
┌─────────────────────────────────────────────┐
│  boat_bpe_tokenizer_decode()  [framework]   │
└──────────────┬──────────────────────────────┘
               │ LaTeX string
               ▼
            Output
```

## 5. Reuse Against Existing Boat Components

| Component | Where | New? |
|---|---|---|
| `boat_tensor_t` | `src/core/tensor.c` | Existing |
| `boat_matmul` | `src/ops/linear.c` | Existing |
| `boat_add`, `boat_mul` | `src/ops/arithmetic.c` | Existing |
| `boat_transpose` | `src/ops/linear.c` | Existing |
| `boat_gelu` | `src/ops/activation.c` | Existing |
| `boat_softmax` | `src/ops/activation.c` | Existing |
| `boat_layer_norm` | `src/layers/norm.c` | Existing |
| `boat_scaled_dot_product_attention` | `src/layers/attention.c` | Existing |
| `boat_conv2d` | `src/layers/conv.c` | Existing (for PatchEmbed) |
| `boat_hf_load_file` | `src/format/huggingface.c` | Existing |
| `boat_sampling_argmax` | `src/sampling.c` | Existing |
| **boat_swin_forward** | **`src/layers/swin.c`** | **NEW** |
| **boat_decoder_layer_forward** | **`src/layers/transformer_decoder.c`** | **NEW** |
| **boat_bpe_tokenizer_\*** | **`src/tokenizers/bpe.c`** | **NEW** |
| **boat_image_load/resize/to_tensor** | **`src/data/`** | **NEW (extend)** |

## 6. Implementation Phases

### Phase 1: Framework — Swin Transformer (`src/layers/swin.*`)

- [ ] Define `boat_swin_config_t` + `boat_swin_weights_t` structs
- [ ] `boat_swin_patch_embed()` — Conv2d + LayerNorm
- [ ] `boat_swin_block_forward()` — window partition, W-MSA/SW-MSA, relative pos bias, MLP
- [ ] `boat_swin_patch_merging()` — 2×2 concat + LN + Linear
- [ ] `boat_swin_forward()` — orchestrate all 4 stages
- [ ] Cyclic shift helper for SW-MSA (generic, reusable)
- [ ] Unit tests: window partition round-trip, shape checks, numerical correctness vs Python

### Phase 2: Framework — Transformer Decoder Layer (`src/layers/transformer_decoder.*`)

- [ ] Define `boat_decoder_config_t` + `boat_decoder_weights_t` struct
- [ ] `boat_decoder_layer_forward()` — self-attn → residual → cross-attn → residual → FFN → residual
- [ ] KV cache tensors (passed in, mutated in-place)
- [ ] Support both pre-norm and post-norm
- [ ] Support both GELU and ReLU activations
- [ ] Unit tests: single layer forward, KV cache growth, cross-attn masking

### Phase 3: Framework — BPE Tokenizer (`src/tokenizers/bpe.*`)

- [ ] `boat_bpe_tokenizer_create()` — parse tokenizer.json via cJSON
- [ ] `boat_bpe_tokenizer_decode()` — id→token lookup, special token skipping
- [ ] Helper: id queries (bos/eos/pad/vocab_size)
- [ ] Unit tests: decode known IDs, verify special tokens

### Phase 4: Framework — Image Loading (extend `src/data/`)

- [ ] Add stb_image as optional dependency to CMake
- [ ] `boat_image_load()` + `boat_image_to_tensor()`
- [ ] `boat_image_resize()` (bilinear)
- [ ] Unit tests: load a known image, verify tensor dims and values

### Phase 5: Example — Nougat Model (`examples/latex/`)

- [ ] `examples/latex/CMakeLists.txt`
- [ ] `nougat_model.c/h` — weight loading from safetensors
- [ ] `nougat_decoder.c/h` — 10-layer decoder stack + autoregressive loop
- [ ] `latex.c` — main entry point: parse args, load model, preprocess image, run encoder + decoder, output LaTeX
- [ ] Validation: run on sample equation images, compare output with Python Nougat

### Phase 6: Build System Integration

- [ ] Resolve image loading for Windows (stb_image or GDI+ fallback)
- [ ] Add `BOAT_WITH_VISION` CMake option to guard image loading + Swin
- [ ] Update `include/boat.h` to include new headers under appropriate guards

## 7. New Files Summary

| File | Type | Purpose |
|---|---|---|
| `include/boat/layers/swin.h` | Framework header | Swin Transformer API |
| `src/layers/swin.c` | Framework src | Swin implementation |
| `include/boat/layers/transformer_decoder.h` | Framework header | Cross-attn decoder layer API |
| `src/layers/transformer_decoder.c` | Framework src | Decoder layer implementation |
| `include/boat/tokenizers/bpe.h` | Framework header | BPE tokenizer API |
| `src/tokenizers/bpe.c` | Framework src | BPE tokenizer implementation |
| `include/boat/tokenizers.h` | Framework header | Aggregator for all tokenizer types |
| `include/boat/layers.h` update | Framework header | Add swin.h, transformer_decoder.h |
| `include/boat.h` update | Framework header | Add tokenizers/ subdirectory |
| `CMakeLists.txt` update | Build | Add new source files |
| `examples/latex/CMakeLists.txt` | Example build | Link boat |
| `examples/latex/latex.c` | Example | Main |
| `examples/latex/nougat_model.c/h` | Example | Weight container + loader |
| `examples/latex/nougat_decoder.c/h` | Example | 10-layer decoder + autoregressive loop |

## 8. What Stays Out of Framework

These are model-specific and belong in the example only:

- **Weight name mapping** (safetensors → C struct) — different per model
- **Autoregressive loop** — the generation strategy (when to stop, prompt format) is task-specific
- **Specific image preprocessing parameters** (224×560, ImageNet stats) — these belong in the example calling the framework's general image transforms
- **Nougat special tokens** (`[START_REF]`, `[IMAGE]`, `<fragments>`, etc.) — specific to Nougat vocabulary
