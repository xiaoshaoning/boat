# OCR App: GLM-OCR Inference with Boat Framework

## Overview

Implement an OCR inference app using the Boat framework, modeled after the
[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) architecture — a 0.9B
vision–language model for document understanding. The app loads a quantized
GLM-OCR model (GGUF format) and runs document parsing, table recognition,
and formula recognition on input images.

## References

- GLM-OCR model: `D:\huggingface\GLM-OCR\`
- Model config: `D:\huggingface\GLM-OCR\config.json`
- Tokenizer: `D:\huggingface\GLM-OCR\tokenizer.json` (BPE, vocab_size=59392)
- Preprocessor: `D:\huggingface\GLM-OCR\preprocessor_config.json`
- Existing boat example: `examples/transformer/transformer.c`

## Architecture

```
Image (RGB)
  │
  ▼
┌──────────────────────────────────────────────┐
│  Image Preprocessor                           │
│  - Resize to 336×336                          │
│  - Normalize (mean=[0.481,0.458,0.408],       │
│               std=[0.269,0.261,0.276])        │
│  - Patchify (14×14 patches → 24×24 = 576)     │
│  - Temporal merge of 2x patches               │
└──────────────┬───────────────────────────────┘
               │ 576 patch embeddings
               ▼
┌──────────────────────────────────────────────┐
│  CogViT Vision Encoder (24 layers)            │
│  - hidden_size=1024, 16 heads                 │
│  - SiLU activation in FFN                     │
│  - RMSNorm throughout                         │
│  - Output: 576 × 1024                          │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Connector (projection)                       │
│  - Linear: 1024 → 1536 (out_hidden_size)      │
│  - Spatial merge 2×2: 576 → 144 tokens        │
└──────────────┬───────────────────────────────┘
               │ 144 image tokens + text tokens
               ▼
┌──────────────────────────────────────────────┐
│  GLM Text Decoder (16 layers, GQA)            │
│  - hidden_size=1536, 16 heads, 8 KV heads     │
│  - RoPE with mRoPE (vision+text positions)    │
│  - RMSNorm, SiLU FFN                          │
│  - Multi-Token Prediction head                │
└──────────────┬──────────────────────────────┘
               │
               ▼
          Output text (OCR result)
```

## Model Components

### 1. Image Preprocessor

New file: `examples/ocr/image.c`, `examples/ocr/image.h`

- Load image file (BMP/PPM/PNG) → raw RGB pixels → FP32 tensor
- Resize to 336×336 (bilinear, nearest, or boxed)
- Normalize per-channel: `(pixel / 255 - mean) / std`
- Patchify: split into 14×14 patches → `[576, 3, 14, 14]` or flatten to `[576, 588]`
- Optional: temporal patch merge (for video support, merge pairs of patches)

```c
// Example API
boat_tensor_t* ocr_load_image(const char* path, int target_size);
boat_tensor_t* ocr_preprocess_image(boat_tensor_t* rgb, const ocr_image_config_t* cfg);
boat_tensor_t* ocr_patchify(const boat_tensor_t* rgb, int patch_size);
```

**Boat ops needed:** `boat_resize`, `boat_normalize`, `boat_patch_extract`
(likely new ops to add to boat framework, or doable with existing
reshape/transpose/slice primitives).

### 2. CogViT Vision Encoder

New file: `examples/ocr/cogvit.c`, `examples/ocr/cogvit.h`

A 24-layer ViT with:

- **Patch embedding**: Linear projection of patch features → `hidden_size=1024`
- **Position embedding**: Learnable 1D position embeddings (576 positions)
- **24 transformer blocks**, each:
  - RMSNorm → Multi-Head Self-Attention (16 heads, bias=true) → residual
  - RMSNorm → FFN (SiLU, hidden=4096) → residual
- **Output projection**: Linear 1024 → 1536

```c
typedef struct {
    // Patch embedding
    boat_tensor_t* patch_proj;     // [patch_dim, hidden_size]
    boat_tensor_t* pos_embed;      // [num_patches, hidden_size]
    // 24 transformer blocks
    cogvit_block_t blocks[24];
    // Output projection
    boat_tensor_t* out_proj;       // [1024, 1536]
} ocr_cogvit_t;

boat_tensor_t* ocr_cogvit_forward(ocr_cogvit_t* vit, const boat_tensor_t* patches);
```

**Boat ops needed:** RMSNorm (exists), SiLU (exists), matmul (exists),
multi-head attention helper.

### 3. Spatial Token Merger

New file: `examples/ocr/connector.c`, `examples/ocr/connector.h`

The vision output `[576, 1536]` is spatially merged 2×2 → `[144, 1536]`:

```
For a spatial grid of 24×24 = 576 tokens:
  Group each 2×2 block → 1 token
  24×24 / (2×2) = 12×12 = 144 tokens
  Per group: reshape [4, 1536], project → [1, 1536]
```

```c
typedef struct {
    boat_tensor_t* merge_weight;  // [4*1536, 1536] or [6144, 1536]
    boat_tensor_t* merge_bias;    // [1536]
} ocr_connector_t;

boat_tensor_t* ocr_spatial_merge(ocr_connector_t* conn, const boat_tensor_t* vit_output);
```

### 4. GLM Text Decoder

New file: `examples/ocr/glm_decoder.c`, `examples/ocr/glm_decoder.h`

16-layer transformer decoder with GQA (Grouped Query Attention):

- **Embedding**: Token embedding `[vocab_size=59392, hidden_size=1536]`
- **16 decoder blocks**, each:
  - RMSNorm → GQA Self-Attention (16 heads, 8 KV heads, RoPE) → residual
  - RMSNorm → FFN (SiLU, intermediate=4608) → residual
- **Output head**: RMSNorm → Linear `[1536, 59392]`
- **Multi-Token Prediction head**: Optional 17th layer for MTP loss

```c
typedef struct {
    int vocab_size;
    int hidden_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int intermediate_size;
    
    boat_tensor_t* token_embed;     // [vocab_size, hidden_size]
    glm_decoder_block_t blocks[16];
    boat_tensor_t* norm_weight;     // [hidden_size] final RMSNorm
    boat_tensor_t* output_weight;   // [hidden_size, vocab_size]
} ocr_glm_decoder_t;

// Forward with KV cache
boat_tensor_t* ocr_glm_forward(ocr_glm_decoder_t* dec,
                                const boat_tensor_t* tokens,
                                const boat_tensor_t* image_tokens,
                                ocr_kv_cache_t* kv_cache);
```

### 5. KV Cache for Autoregressive Decoding

New file: `examples/ocr/kv_cache.c`, `examples/ocr/kv_cache.h`

Cache K and V tensors for each layer across decode steps:

```c
typedef struct {
    boat_tensor_t* k_cache;  // [n_layers, max_seq_len, n_kv_heads, head_dim]
    boat_tensor_t* v_cache;  // [n_layers, max_seq_len, n_kv_heads, head_dim]
    int seq_len;             // Current sequence length
} ocr_kv_cache_t;

ocr_kv_cache_t* ocr_kv_cache_create(int n_layers, int n_kv_heads, int head_dim, int max_seq);
void ocr_kv_cache_append(ocr_kv_cache_t* cache, int layer, const boat_tensor_t* k, const boat_tensor_t* v);
```

### 6. BPE Tokenizer (Tiktoken-compatible)

New file: `examples/ocr/tokenizer.c`, `examples/ocr/tokenizer.h`

GLM-OCR uses a BPE tokenizer with 59392 tokens. Boat needs a minimal
tiktoken-compatible C tokenizer:

```c
typedef struct {
    // Rank-based BPE merge table
    int* merge_ranks;         // [num_merges * 2]
    int num_merges;
    uint8_t* vocab_bytes;     // Raw byte strings for each token id
    int* vocab_offsets;       // Offset into vocab_bytes for each token
    int vocab_size;
} ocr_tokenizer_t;

int* ocr_tokenizer_encode(ocr_tokenizer_t* tok, const char* text, int* out_len);
char* ocr_tokenizer_decode(ocr_tokenizer_t* tok, const int* ids, int len);
```

Store the tokenizer data alongside the GGUF model (export from
`tokenizer.json` + `tokenizer_config.json`).

### 7. Autoregressive Inference Engine

New file: `examples/ocr/inference.c`, `examples/ocr/inference.h`

Orchestrates the full image→text pipeline:

```
1. Load and preprocess image → patches
2. Encode prompt string → token ids
3. CogViT forward → visual features
4. Spatial merge → 144 image tokens
5. Prefill: run decoder on prompt + image tokens, build KV cache
6. Decode loop:
   a. Run decoder on last token
   b. Sample next token (greedy / temperature / top-k)
   c. Append to KV cache
   d. Stop at EOS or max_new_tokens
7. Decode token ids → text output
```

```c
typedef struct {
    ocr_cogvit_t* vit;
    ocr_connector_t* connector;
    ocr_glm_decoder_t* decoder;
    ocr_tokenizer_t* tokenizer;
} ocr_model_t;

typedef struct {
    int max_new_tokens;
    float temperature;
    int top_k;
    bool do_sample;
} ocr_gen_config_t;

char* ocr_generate(ocr_model_t* model,
                   const char* image_path,
                   const char* prompt,
                   const ocr_gen_config_t* cfg);
```

### 8. Sampling Strategies

New file: `examples/ocr/sampling.c`, `examples/ocr/sampling.h`

```c
// Greedy: pick argmax
int ocr_sample_greedy(const float* logits, int vocab_size);

// Temperature-scaled multinomial
int ocr_sample_temperature(const float* logits, int vocab_size, float temp);

// Top-k filtering + temperature
int ocr_sample_top_k(const float* logits, int vocab_size, int k, float temp);
```

## Data Flow Detail

### Prefill Phase

```
Prompt: "Text Recognition:"
→ Tokenize → [token_1, token_2, ...] (N tokens)
→ Embed → [N, 1536]
→ Concat image tokens: [144 + N, 1536]
→ Run decoder on all tokens, populate KV cache for all positions
```

### Decode Phase (one step at a time)

```
Input: last generated token id
→ Embed → [1, 1536]
→ Run decoder with KV cache (only compute on new token)
→ Output head → [1, 59392] logits
→ Sample → next token id
→ Stop if EOS (<|endoftext|> = 59246)
→ Max 8192 tokens as per original model
```

## File Structure

```
examples/ocr/
├── ocr.c                  # Main app: CLI, model loading, inference loop
├── image.c / image.h      # Image loading and preprocessing
├── cogvit.c / cogvit.h    # CogViT vision encoder
├── connector.c / connector.h  # Spatial token merger
├── glm_decoder.c / glm_decoder.h  # GLM text decoder
├── kv_cache.c / kv_cache.h  # KV cache for autoregressive decoding
├── tokenizer.c / tokenizer.h  # BPE tokenizer
├── inference.c / inference.h  # Orchestration + generation loop
├── sampling.c / sampling.h  # Sampling strategies
├── model_loader.c / model_loader.h  # GGUF weight loader
├── CMakeLists.txt         # Build configuration
└── README.md              # Usage instructions
```

## Model Loading (GGUF)

Export the GLM-OCR model safetensors to GGUF format using a Python
conversion script. The GGUF file stores all weight tensors in a packed
binary format that boat already supports (`BOAT_WITH_GGUF`).

Required GGUF tensors:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `vit.patch_proj.weight` | [588, 1024] | Patch embedding |
| `vit.patch_proj.bias` | [1024] | |
| `vit.pos_embed` | [576, 1024] | Position embedding |
| `vit.blocks.{i}.attn.qkv.weight` | [3072, 1024] | QKV projection (16 heads × 1024/16 = 64d, Q=64, K=64, V=64 → 3*64*16 = 3072) |
| `vit.blocks.{i}.attn.qkv.bias` | [3072] | |
| `vit.blocks.{i}.attn.proj.weight` | [1024, 1024] | Attention output |
| `vit.blocks.{i}.attn.proj.bias` | [1024] | |
| `vit.blocks.{i}.ffn.gate.weight` | [4096, 1024] | SiLU gate |
| `vit.blocks.{i}.ffn.down.weight` | [1024, 4096] | Down projection |
| `vit.blocks.{i}.ffn.up.weight` | [4096, 1024] | Up projection |
| `vit.blocks.{i}.norm1.weight` | [1024] | RMSNorm |
| `vit.blocks.{i}.norm2.weight` | [1024] | RMSNorm |
| `vit.out_proj.weight` | [1024, 1536] | Vision→text projection |
| `connector.merge.weight` | [6144, 1536] | Spatial merge |
| `connector.merge.bias` | [1536] | |
| `decoder.token_embed.weight` | [59392, 1536] | Token embedding |
| `decoder.blocks.{i}.attn.q.weight` | [1536, 1536] | Query projection (16 heads × 96d = 1536) |
| `decoder.blocks.{i}.attn.k.weight` | [1536, 768] | Key projection (8 KV heads × 96d = 768) |
| `decoder.blocks.{i}.attn.v.weight` | [1536, 768] | Value projection |
| `decoder.blocks.{i}.attn.o.weight` | [1536, 1536] | Attention output |
| `decoder.blocks.{i}.ffn.gate.weight` | [4608, 1536] | SiLU gate |
| `decoder.blocks.{i}.ffn.down.weight` | [1536, 4608] | Down projection |
| `decoder.blocks.{i}.ffn.up.weight` | [4608, 1536] | Up projection |
| `decoder.blocks.{i}.norm1.weight` | [1536] | Pre-attention RMSNorm |
| `decoder.blocks.{i}.norm2.weight` | [1536] | Pre-FFN RMSNorm |
| `decoder.norm.weight` | [1536] | Final RMSNorm |
| `decoder.output.weight` | [1536, 59392] | LM head (tied or untied) |

**Quantization:** All weights can be quantized to Q8_0 or Q4_0 in GGUF
for reduced memory and faster inference. Boat already supports these
GGUF quantization formats.

## Implementation Phases

### Phase 1: Infrastructure (week 1)

- **Tokenizer**: Implement BPE tokenizer reading from GGUF metadata or
  exported tok_data.bin. Encode/decode, special tokens.
- **Image loader**: Minimal PPM/BMP loader + resize (bilinear) +
  normalize + patchify. No external deps.
- **CMake build**: Add `examples/ocr/` subdirectory.

### Phase 2: Vision Encoder (week 2)

- **CogViT blocks**: RMSNorm, SiLU FFN, attention. Single-block test
  with random inputs.
- **24-layer stack**: Sequential forward through all blocks. Verify
  output shape matches `[576, 1024]`.
- **Output projection**: Linear `[1024, 1536]`. Spatial merge 2×2 to
  `[144, 1536]`.

### Phase 3: Text Decoder (week 2-3)

- **GQA attention**: Grouped query attention with separate Q/K/V
  projections. RoPE implementation (need mRoPE variant).
- **GLM decoder block**: RMSNorm → GQA → residual → RMSNorm → SiLU
  FFN → residual.
- **16-layer decoder**: Stack + KV cache integration.
- **Prefill + decode**: Full autoregressive loop with greedy sampling.

### Phase 4: Integration (week 3)

- **Model loader**: Read GGUF file and populate all model structs.
- **Full pipeline**: Image → preprocess → Vit → connector → decoder
  → generate text.
- **Prompt types**: Support "Text Recognition:", "Formula Recognition:",
  "Table Recognition:" prompts.
- **CLI interface**: `ocr --image img.jpg --prompt "Text Recognition:"`

### Phase 5: Quantization & Optimization (week 4)

- **GGUF Q8_0/Q4_0**: Use boat's existing GGUF quantized tensor support.
- **KV cache recycling**: Manage cache size for long documents.
- **Memory pool**: Reuse tensor allocations across decode steps.
- **OpenMP parallelism**: Parallelize attention heads during prefill.

## New Boat Ops Required

The following operations are not yet in boat and need implementation:

| Op | Description | Priority |
|----|-------------|----------|
| `boat_rotary_emb` | RoPE (Rotary Position Embedding) — needed for GLM decoder | High |
| `boat_multi_head_attn` | Multi-head attention with optional KV cache | High |
| `boat_gqa_attn` | Grouped Query Attention (generalizes MHA) | High |
| `boat_layer_norm` | LayerNorm (only RMSNorm exists currently) | Medium |
| `boat_top_k` | Top-k indexing for sampling | Medium |
| `boat_multinomial` | Categorical sampling from logits | Medium |
| `boat_pad` / `boat_gather` | For sequence operations | Medium |
| `mRope variant` | Multi-modal RoPE with separate position encoding for vision and text | Low (can use standard RoPE initially) |
| `SiLU forward` | SiLU activation exists but verify it works element-wise | Check |

## Testing

```c
// examples/ocr/test_ocr.c — per-component tests

// 1. Test tokenizer
void test_tokenizer();     // encode/decode roundtrip
void test_special_tokens(); // <|image|>, <|endoftext|>, etc.

// 2. Test vision encoder
void test_cogvit_forward();  // random input → verify output shape
void test_spatial_merge();   // 576 → 144 tokens

// 3. Test decoder
void test_gqa_attention();   // GQA forward shape check
void test_kv_cache();        // Cache append + retrieve
void test_prefill_decode();  // Full prefill → decode loop

// 4. Test end-to-end
void test_ocr_pipeline();    // Known image → expected output format
```

## Open Questions

1. **Weight export**: Safetensors → GGUF conversion script. Use Python
   to read the GLM-OCR safetensors and write a GGUF file that boat can
   load. Need to handle the CogViT and GLM-OCR layer naming.

2. **Image format support**: PPM is simplest for a C-only app (no deps).
   For real use, integrate stb_image.h (single-file, public domain) for
   PNG/JPEG support.

3. **RoPE implementation**: GLM-OCR uses mRoPE with sections [16, 24, 24]
   for multi-modal position encoding. Initial implementation can use
   standard RoPE (all heads get the same frequency), then extend to mRoPE
   for cross-modal position coordination.

4. **Vocabulary size**: 59392 tokens with hidden_size=1536 makes the LM
   head `[1536, 59392]` — about 365 MB in FP32 or 91 MB in Q8_0.
   Quantization is strongly recommended.

5. **Multi-Token Prediction (MTP)**: The config has `num_nextn_predict_layers=1`,
   meaning an extra output head predicts the next-next token during
   TRAINING. For inference, this is unused — standard single-token
   autoregressive decoding is sufficient.
