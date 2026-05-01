# NanoChat for Boat Framework

Implement a C inference and training engine for [nanochat](https://github.com/xiaoshaoning/nanochat)-trained GPT models using the Boat deep learning framework. This covers weight loading, tokenization, model forward/backward pass, pretraining, supervised fine-tuning (SFT), reinforcement learning (RL), inference (prefill + decode), sampling, and an interactive chat CLI.

## Architecture Overview

```
examples/nanochat/
├── CMakeLists.txt            # Build integration
├── nanochat.h                # Public API header
├── model.c                   # GPT model struct + forward/backward
├── model.h
├── config.h                  # GPTConfig struct
├── weights.c                 # Checkpoint loader/saver
├── weights.h
├── tokenizer.c               # BPE tokenizer (tiktoken-compatible)
├── tokenizer.h
├── sampling.c                # Top-k, temperature sampling
├── sampling.h
├── kv_cache.c                # KV cache for autoregressive decode
├── kv_cache.h
├── engine.c                  # Inference engine (prefill + decode loop)
├── engine.h
├── optim.c                   # Muon + AdamW combined optimizer
├── optim.h
├── dataloader.c              # BOS-aligned best-fit packing dataloader
├── dataloader.h
├── loss.c                    # Cross-entropy with masking
├── loss.h
├── train.c                   # Training loop (pretrain/SFT/RL)
├── train.h
├── nanochat_cli.c            # Interactive chat CLI (entry point)
├── nanochat_train.c          # Training entry point
└── README.md                 # Usage documentation
```

## NanoChat Model Architecture

The nanochat GPT model (`nanochat/gpt.py`) is a decoder-only Transformer with these features:

| Feature | Detail |
|---|---|
| **RoPE** | Rotary Position Embeddings, base=100000 |
| **RMSNorm** | No learnable parameters, applied after embedding, before attn/mlp, and at output |
| **QK Norm** | RMSNorm on Q and K after RoPE, scaled by 1.2 |
| **ReLU² MLP** | `c_fc` -> ReLU -> square -> `c_proj`, no bias |
| **GQA** | Group-Query Attention, configurable `n_kv_head` |
| **Untied weights** | Separate `wte` and `lm_head` |
| **Sliding window** | Per-layer via pattern string ("SSSL"), final layer always full |
| **Value Residual** | Per-token value embeddings added to V via learned gate (alternating layers) |
| **Smear** | Bigram-like: previous token embedding mixed into current via learned gate |
| **Backout** | Subtract mid-layer residual before final norm |
| **Per-layer scalars** | `resid_lambdas` (residual scale), `x0_lambdas` (initial embedding blend) |
| **Logit Softcap** | `15 * tanh(logits / 15)` |

### GPTConfig (default d12)

```c
typedef struct {
    int sequence_len;       // 2048
    int vocab_size;         // 32768
    int n_layer;            // 12
    int n_head;             // 6
    int n_kv_head;          // 6
    int n_embd;             // 768
    int head_dim;           // n_embd / n_head = 128
    int kv_dim;             // n_kv_head * head_dim = 768
    int padded_vocab_size;  // ceil(vocab_size / 64) * 64 = 32768
    char window_pattern[16];// "SSSL"
    int window_sizes[];     // per-layer window size
} nanochat_config_t;
```

### Weight tensors

| Key | Shape | Params |
|---|---|---|
| `transformer.wte.weight` | [32768, 768] | 25,165,824 |
| `lm_head.weight` | [32768, 768] | 25,165,824 |
| `h.{i}.attn.c_q.weight` | [768, 768] | 589,824 x 12 |
| `h.{i}.attn.c_k.weight` | [768, 768] | 589,824 x 12 |
| `h.{i}.attn.c_v.weight` | [768, 768] | 589,824 x 12 |
| `h.{i}.attn.c_proj.weight` | [768, 768] | 589,824 x 12 |
| `h.{i}.mlp.c_fc.weight` | [3072, 768] | 2,359,296 x 12 |
| `h.{i}.mlp.c_proj.weight` | [768, 3072] | 2,359,296 x 12 |
| `h.{i}.attn.ve_gate.weight` | [6, 12] | 72 x 6 |
| `value_embeds.{i}.weight` | [32768, 768] | 25,165,824 x 6 |
| `resid_lambdas` | [12] | 12 |
| `x0_lambdas` | [12] | 12 |
| `smear_gate.weight` | [1, 24] | 24 |
| `smear_lambda` | [1] | 1 |
| `backout_lambda` | [1] | 1 |

**Total for d12**: ~224M parameters (dominated by wte, lm_head, and value_embeds).

## Part 1: Inference Pipeline

### 1.1 Weight Loader (`weights.c` / `weights.h`)

PyTorch `.pt` (pickle) files are impractical to parse in C. A Python script converts checkpoints to a Boat-native format:

**`scripts/convert_nanochat.py`** (Python 3.10+):
- Load `.pt` with `torch.load()`, strip `_orig_mod.` prefix
- Extract `GPTConfig` from metadata
- Cast bf16/fp16 weights to fp32
- Transpose all linear weights from [out, in] to [in, out] (Boat convention)
- RoPE: precompute cos/sin arrays and include in output
- Serialize to Boat binary format (magic, version, config JSON, tensor count, per-tensor: name + shape + dtype + data)
- Also export tokenizer data: vocab merges, special tokens, pre-tokenization pattern

```c
nanochat_weights_t* nanochat_weights_load(const char* path);
void nanochat_weights_free(nanochat_weights_t* w);
```

### 1.2 Tokenizer (`tokenizer.c` / `tokenizer.h`)

BPE tokenizer compatible with nanochat's tiktoken-based tokenizer. Export format:

```
- vocab_size, special_token_count
- special_token_ids + strings (bos, user_start/end, assistant_start/end, python_start/end, output_start/end)
- For each BPE token: byte_length, bytes, rank
- GPT-4 split regex pattern (for pre-tokenization)
```

```c
nanochat_tokenizer_t* nanochat_tokenizer_load(const char* path);
int*  nanochat_tokenizer_encode(tok, text, &out_len);
char* nanochat_tokenizer_decode(tok, ids, n_ids);
int   nanochat_tokenizer_bos_id(tok);
int   nanochat_tokenizer_user_start_id(tok);
int   nanochat_tokenizer_assistant_end_id(tok);
```

**Pre-tokenization**: hand-coded DFA scanner for GPT-4 split pattern. UTF-8 support for Unicode categories. ASCII-only in initial version.

**BPE merge**: linked-list per chunk, iteratively merge lowest-rank pair. O(n²) naive, acceptable for inference sequence lengths.

### 1.3 New Boat ops needed

These are added to `src/ops/` and declared in `include/boat/ops.h`:

| Op | File | Purpose |
|---|---|---|
| RoPE precompute + apply | `src/ops/rope.c` | Position encoding for Q/K |
| Embedding lookup (gather) | `src/ops/gather.c` | `wte[tokens]` |
| Top-k | `src/ops/topk.c` | Sampling filter |
| ReLU² | `src/ops/activation.c` | MLP activation |
| Tanh | `src/ops/activation.c` | Logit softcap |
| Sigmoid | `src/ops/activation.c` | Smear gate, VE gate |

### 1.4 GPT Model forward pass (`model.c`)

The forward pass mirrors `GPT.forward()` from nanochat. Architecture:

```
Input tokens [1, T]
  → wte[tokens]                          # embedding lookup
  → rmsnorm                               # norm after embedding
  → smear                                 # bigram mixing (prefill: permute, decode: cached prev)
  → save x0                               # for x0 residual blend
  → for each layer i in 0..n_layer-1:
      x = resid_lambdas[i] * x + x0_lambdas[i] * x0    # per-layer scalar blend
      → rmsnorm                            # pre-attn norm
      → linear Q, K, V from x              # projections
      → RoPE(Q, K)                         # rotary embeddings
      → QK norm (rmsnorm, scale 1.2)       # qk normalization
      → attention (prefill or decode)       # with KV cache
      → value residual (if VE layer)       # add value_embeds[tokens] * gate
      → linear proj                        # c_proj
      → x = x + y                          # residual
      → rmsnorm                            # pre-mlp norm
      → MLP: c_fc → ReLU² → c_proj        # feed-forward
      → x = x + y                          # residual
      → if i == n_layer/2: save x_backout  # for backout
  → x = x - backout_lambda * x_backout    # backout
  → rmsnorm                                # final norm
  → lm_head(x)                             # logits [1, vocab_size]
  → 15 * tanh(logits / 15)                 # logit softcap
  → return logits
```

### 1.5 KV Cache

Per-layer key/value storage for autoregressive decode:

```c
boat_kv_cache_t* boat_kv_cache_create(n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
void boat_kv_cache_append(cache, layer, k_tensor, v_tensor);
void boat_kv_cache_prefill(dst, src, num_samples);  // expand batch=1 → batch=N
```

Cache layout: separate tensors per layer, `[batch, seq_len, n_kv_heads, head_dim]`.

**Prefill**: compute all K/V in one forward pass, fill cache.
**Decode**: compute single-token K/V, append to cache, attend to full cache.

### 1.6 Attention

Custom attention for nanochat (not reusing Boat's existing attention layer):

- **GQA**: expand K/V heads from `n_kv_head` to `n_head` via simple repetition
- **Sliding window**: during prefill, mask positions beyond `window_size`; during decode, attend to all cached tokens (capped at `window_size`)
- **Causal masking**: standard upper-triangular -inf mask
- **SDPA**: naive O(n²) implementation, float32

### 1.7 Inference Engine (`engine.c`)

```c
void nanochat_engine_generate(engine, prompt_tokens, prompt_len,
                               callback, userdata);
```

1. **Prefill**: one forward pass on all prompt tokens, populate KV cache
2. **Decode loop** (per step):
   - Sample next token (top-k + temperature + multinomial)
   - Tool use state machine: detect `<|python_start|>` / `<|python_end|>`, accumulate expression, evaluate calculator, inject result tokens
   - Stop on `<|assistant_end|>` or max_tokens
   - Single-token forward pass to get next logits

### 1.8 Sampling (`sampling.c`)

```c
int nanochat_sample_token(logits, vocab_size, temperature, top_k, &rng_state);
```

- temperature=0 → argmax
- top-k filtering → softmax → multinomial
- PCG random number generator

### 1.9 Tool Use (calculator)

Sandboxed expression evaluator matching nanochat's `use_calculator()`:
- Pure math: digits, `*`, `+`, `-`, `/`, `.`, `(`, `)`, space
- String operations: `.count()` on quoted strings
- Blocked: `__`, `import`, `exec`, `eval`, `open`, `**`
- Recursive-descent parser, no dependency on eval()

### 1.10 Chat CLI (`nanochat_cli.c`)

```
nanochat_cli --model model.boat --tokenizer tok.data [options]
```

Interactive REPL with conversation history, streaming output, `quit`/`clear` commands.

## Part 2: Training Pipeline

### 2.1 Training Architecture Overview

Training follows nanochat's three-stage pipeline:

```
Stage 1: Pretraining (base_train.py)
  Data: ClimbMix-400B parquet shards
  Task: Causal language modeling (next-token prediction)
  Loss: Cross-entropy on all tokens
  Optimizer: Muon (matrix params) + AdamW (embeddings, scalars)
  Scheduler: Warmup → constant → warmdown (LR, momentum, weight decay)

Stage 2: SFT (chat_sft.py)
  Data: SmolTalk + MMLU + GSM8K + SpellingBee conversations
  Task: Supervised fine-tuning on assistant responses
  Loss: Cross-entropy with masking (mask=1 for assistant tokens, 0 for prompts)
  Optimizer: Continued from pretrain with reset LR

Stage 3: RL (chat_rl.py)
  Data: GSM8K math problems
  Task: GRPO/REINFORCE with tool use rewards
  Optimizer: Same as SFT
```

### 2.2 Framework Requirements for Training

Training requires significant extensions to Boat beyond inference:

| Component | Current Boat Status | What's Needed |
|---|---|---|
| **Backward pass** | Partial (some layer stubs) | Full autograd through all GPT ops |
| **Cross-entropy loss** | Exists for classification | Causal CE with ignore_index masking |
| **AdamW optimizer** | Exists | Fused step, bias correction, weight decay |
| **Muon optimizer** | **Not implemented** | Polar Express + NorMuon (see optim.py) |
| **Gradient accumulation** | Not supported | Sum gradients across micro-batches |
| **LR scheduler** | Exists (cosine, step) | Warmup + constant + warmdown |
| **Momentum scheduler** | Not supported | Per-step momentum update |
| **Weight decay scheduler** | Not supported | Cosine decay |
| **Data pipeline** | Basic DataLoader | Parquet reader, best-fit packing |
| **Checkpointing** | Exists (model save/load) | Optimizer state, scheduler state, metadata |
| **SFT loss masking** | Not supported | Per-position ignore_index mask |
| **FP8 training** | **Not implemented** | Float8 matmul with dynamic scaling |
| **Distributed training** | Not applicable | CPU-only, no DDP planned |

### 2.3 Realistic Scope for Boat

Given Boat's current state (float32 CPU-only ops, limited autograd), training support is tiered:

**Tier 1 — Feasible now (CPU, small models)**:
- Pretrain a d4 model (4 layers, n_embd=256) on a small text dataset
- Simple cross-entropy loss, no masking
- AdamW optimizer only (no Muon initially)
- Basic LR schedule (linear warmup + cosine decay)
- Works for educational/experimental use (~hours per training run)

**Tier 2 — Requires autograd extension (CPU, medium models)**:
- Full backward pass through all GPT ops (RoPE, GQA attention, ReLU², smear, backout, etc.)
- Muon optimizer with Polar Express orthogonalization
- SFT with loss masking
- BOS-aligned best-fit dataloader
- Gradient accumulation

**Tier 3 — Requires CUDA (production models)**:
- Full d12/d20 model training
- FP8 training kernels
- Multi-GPU distributed training
- Realistic training times

### 2.4 Backward Pass Extension

Boat's current autograd (`src/autodiff.c`, `src/graph/`) uses a computational graph with variable nodes. For GPT training, every op in the forward pass needs a backward counterpart:

```c
// New backward ops needed:
void boat_relu2_backward(boat_tensor_t* grad_output, boat_tensor_t* input, boat_tensor_t* grad_input);
void boat_rmsnorm_backward(boat_tensor_t* grad_output, boat_tensor_t* input, boat_tensor_t* weight, boat_tensor_t* grad_input, boat_tensor_t* grad_weight);
void boat_rope_backward(boat_tensor_t* grad_q, boat_tensor_t* grad_k, boat_tensor_t* grad_output_q, boat_tensor_t* grad_output_k, ...);
void boat_attention_backward(...);  // full attention backward
void boat_linear_backward(boat_tensor_t* grad_output, boat_tensor_t* input, boat_tensor_t* weight, boat_tensor_t* grad_input, boat_tensor_t* grad_weight);
void boat_smear_backward(...);
void boat_softcap_backward(...);
```

**Approach:** Instead of wrapping every op in the variable/graph system (which adds overhead), build a custom `nanochat_model_backward()` function that manually chains backward calls in reverse order, directly computing gradients into pre-allocated buffers. This is simpler and more efficient than the generic graph approach, and mirrors nanochat's PyTorch autograd usage.

```c
typedef struct {
    // Gradient buffers (pre-allocated, same shapes as forward intermediates)
    boat_tensor_t* grad_wte;
    boat_tensor_t* grad_lm_head;
    boat_tensor_t** grad_c_q;    // [n_layer]
    boat_tensor_t** grad_c_k;
    boat_tensor_t** grad_c_v;
    boat_tensor_t** grad_c_proj;
    boat_tensor_t** grad_mlp_c_fc;
    boat_tensor_t** grad_mlp_c_proj;
    boat_tensor_t** grad_ve_gate;
    boat_tensor_t** grad_value_embeds;
    boat_tensor_t* grad_resid_lambdas;
    boat_tensor_t* grad_x0_lambdas;
    boat_tensor_t* grad_smear_gate;
    boat_tensor_t* grad_smear_lambda;
    boat_tensor_t* grad_backout_lambda;
} nanochat_gradients_t;

// Backward pass: computes gradients for all parameters
void nanochat_model_backward(nanochat_model_t* m,
                              boat_tensor_t* logits,
                              boat_tensor_t* targets,
                              nanochat_gradients_t* grads);
```

### 2.5 Loss Function (`loss.c`)

```c
// Compute cross-entropy loss with optional ignore_index masking
// logits: [B, T, vocab_size], targets: [B, T] with -1 for ignore
float nanochat_cross_entropy(const boat_tensor_t* logits,
                              const boat_tensor_t* targets,
                              boat_tensor_t* grad_output);  // [B, T, vocab_size]

// bits-per-byte conversion
float nanochat_bpb(float cross_entropy_loss, int vocab_size);
```

### 2.6 Optimizer (`optim.c`)

**AdamW** (matches nanochat's `adamw_step_fused`):

```c
typedef struct {
    float lr;
    float beta1, beta2;
    float eps;
    float weight_decay;
    int step;
    boat_tensor_t* exp_avg;    // first moment
    boat_tensor_t* exp_avg_sq; // second moment
} nanochat_adamw_state_t;

void nanochat_adamw_step(nanochat_adamw_state_t* state, boat_tensor_t* param, boat_tensor_t* grad);
```

**Muon** (matches nanochat's `muon_step_fused`):

```c
typedef struct {
    float lr;
    float momentum;
    float beta2;
    float weight_decay;
    int ns_steps;  // Polar Express iterations (default 5)
    boat_tensor_t* momentum_buffer;  // first moment
    boat_tensor_t* second_momentum_buffer;  // factored second moment (per-row or per-col)
} nanochat_muon_state_t;

void nanochat_muon_step(nanochat_muon_state_t* state, boat_tensor_t* param, boat_tensor_t* grad);
```

**Muon algorithm** (`src/ops/muon.c`):

1. **Nesterov momentum**: `buf.lerp_(grad, 1 - momentum); g = grad.lerp_(buf, momentum);`
2. **Polar Express orthogonalization** (5 iterations):
   - `X = g / (||g|| * 1.01 + 1e-6)`
   - For tall matrices (rows > cols): `A = XᵀX; B = b*A + c*A²; X = a*X + X*B`
   - For wide matrices (cols > rows): `A = XXᵀ; B = b*A + c*A²; X = a*X + B*X`
   - Coefficients from the Polar Express paper
3. **NorMuon variance reduction**:
   - Compute per-neuron variance of update
   - Normalize using EMA of second moment
   - Scale update inversely by normalized variance
4. **Cautious weight decay**: mask positive-correlation updates
5. **Parameter update**: `param -= lr * update + lr * wd * param * mask`

**Combined optimizer**:

```c
typedef struct {
    int num_groups;
    nanochat_optim_group_t* groups;  // array, each 'kind' = ADAMW or MUON
} nanochat_optimizer_t;

void nanochat_optim_step(nanochat_optimizer_t* optim);
void nanochat_optim_zero_grad(nanochat_optimizer_t* optim);

// Parameter grouping (matches nanochat's setup_optimizer):
// - lm_head: AdamW, lr=0.004, betas=(0.8,0.96), wd=0.01
// - wte: AdamW, lr=0.2, betas=(0.8,0.995), wd=0.001
// - value_embeds: AdamW, lr=0.1, betas=(0.8,0.995), wd=0.01
// - resid_lambdas: AdamW, lr=0.005, betas=(0.8,0.95), wd=0.05
// - x0_lambdas: AdamW, lr=0.5, betas=(0.96,0.95), wd=0.0
// - smear_params: AdamW, lr=0.2, betas=(0.8,0.95), wd=0.0
// - matrix params (grouped by shape): Muon, lr=0.02
```

### 2.7 Data Pipeline (`dataloader.c`)

**Pretraining data loader** (matches nanochat's `tokenizing_distributed_data_loader_bos_bestfit`):

```c
typedef struct {
    // Document buffer: list of tokenized documents
    int** doc_buffer;
    int* doc_lens;
    int doc_buffer_size;
    int doc_buffer_cap;
    // Source: parquet file reader (or pre-tokenized .bin shards)
    FILE* data_source;
    // Config
    int batch_size;
    int seq_len;
    int vocab_size;
} nanochat_dataloader_t;

// Get next batch: returns (inputs, targets) each [B, T]
bool nanochat_dataloader_next(dataloader, boat_tensor_t** inputs, boat_tensor_t** targets);
```

**BOS-aligned Best-Fit Packing Algorithm:**
1. Maintain a buffer of tokenized documents (each prefixed with `<|bos|>`)
2. For each row in batch:
   - While row has space:
     - Find largest document that fits entirely → use it
     - If none fits → crop shortest buffer document to fill exactly (pretrain) or pad with BOS (SFT)
3. Shift by 1: `inputs = row[0:T-1]`, `targets = row[1:T]`

**SFT data loader** adds loss masking:
```c
// Returns (inputs, targets, loss_mask)
// loss_mask[i,j] = 1 for assistant tokens (supervised), 0 for prompt/padding
```

### 2.8 Pretraining Loop (`train.c`)

```c
void nanochat_pretrain(nanochat_model_t* model,
                        nanochat_optimizer_t* optim,
                        nanochat_dataloader_t* loader,
                        nanochat_train_config_t* cfg);
```

Training loop (matches `base_train.py`):

```
for step in 0..num_iterations:
    # Forward + backward with gradient accumulation
    for micro_step in grad_accum_steps:
        logits = model.forward(inputs)           # [B, T, vocab_size]
        loss = cross_entropy(logits, targets)     # scalar
        loss.backward()                           # accumulate gradients
        fetch next batch (prefetch)

    # Learning rate schedule
    lr = get_lr_multiplier(step) * base_lr
    momentum = get_muon_momentum(step)
    wd = get_weight_decay(step)

    # Optimizer step
    optimizer.step()         # apply gradients with Muon + AdamW
    optimizer.zero_grad()    # clear gradients

    # Logging
    print loss, tokens/sec, MFU

    # Evaluation (every N steps)
    evaluate validation loss (BPB)
    sample from model

    # Checkpoint (every N steps)
    save checkpoint (model, optimizer, scheduler state)
```

**Schedulers** (matching nanochat):

```c
// LR: linear warmup → constant → linear warmdown to final_lr_frac
float nanochat_lr_multiplier(int step, int warmup_steps, int num_iterations,
                              float warmdown_ratio, float final_lr_frac);

// Momentum: warmup 0.85→0.97 over 400 steps, then constant, warmdown 0.97→0.90
float nanochat_muon_momentum(int step, int num_iterations, float warmdown_ratio);

// Weight decay: cosine decay from initial to 0
float nanochat_weight_decay(int step, int num_iterations, float initial_wd);
```

### 2.9 SFT Training

```c
void nanochat_sft(nanochat_model_t* model,
                   nanochat_optimizer_t* optim,
                   TaskMixture* tasks,
                   nanochat_train_config_t* cfg);
```

Key differences from pretraining:
- Loss masking: only compute loss on assistant tokens (ignore user prompts, special tokens, padding)
- Data: conversation datasets (SmolTalk, MMLU, GSM8K, SpellingBee) rendered via `render_conversation()`
- LR: warm-start from pretrained checkpoint with reset LR schedule
- Dataset-driven stopping (iterate through tasks, stop when consumed)

### 2.10 RL Training (`chat_rl.py` equivalent)

```c
void nanochat_rl(nanochat_model_t* model,
                  nanochat_optimizer_t* optim,
                  nanochat_engine_t* engine,
                  Task* task,
                  nanochat_rl_config_t* cfg);
```

Simplified GRPO/REINFORCE:
1. Sample multiple responses from current policy
2. Evaluate reward (e.g., correct answer for GSM8K)
3. Compute advantage (reward - baseline)
4. Policy gradient update: maximize log_prob * advantage
5. KL penalty to prevent policy from diverging too far

### 2.11 FP8 Training (`fp8.c`)

Float8 matmul wrapper matching nanochat's custom `fp8.py`:

```c
// FP8 matmul with dynamic tensorwise scaling
// input: [M, K], weight: [N, K] → output: [M, N]
void nanochat_fp8_matmul(const boat_tensor_t* input,
                          const boat_tensor_t* weight,
                          boat_tensor_t* output);

// Quantize to FP8 + back for gradient computation
void nanochat_fp8_quantize(const boat_tensor_t* x,
                            boat_tensor_t** x_fp8,
                            float* scale);
```

Note: FP8 requires CUDA hardware (`torch._scaled_mm` on H100+). On CPU, this is a no-op fallback to float32. FP8 support is documented for future CUDA enablement.

### 2.12 Checkpointing

Extend `weights.c` to handle training checkpoints:

```c
// Save training checkpoint (model + optimizer + metadata)
void nanochat_checkpoint_save(const char* path,
                               nanochat_model_t* model,
                               nanochat_optimizer_t* optim,
                               nanochat_train_state_t* state);

// Load training checkpoint
nanochat_train_state_t* nanochat_checkpoint_load(const char* path,
                                                   nanochat_model_t* model,
                                                   nanochat_optimizer_t* optim);
```

Checkpoint format (extending Boat's serialization):
```
- Magic + Version
- Model config (JSON)
- Model weights (per tensor)
- Optimizer state (per group: exp_avg, exp_avg_sq, momentum_buffer, second_momentum)
- Training state (step, dataloader position, LR parameters)
```

## Part 3: Implementation Plan

### Phase 1 — Inference Foundation (Week 1-2)

| Step | Description |
|---|---|
| 1 | Add new ops: RoPE, gather, top-k, ReLU², tanh, sigmoid |
| 2 | Python conversion script for checkpoints + tokenizer |
| 3 | `weights.c`: load Boat-native checkpoint |
| 4 | `tokenizer.c`: BPE tokenizer with pre-tokenizer |
| 5 | `kv_cache.c`: KV cache |
| 6 | `sampling.c`: top-k + temperature sampling |

### Phase 2 — Model + Inference Engine (Week 3-4)

| Step | Description |
|---|---|
| 7 | `model.c`: GPT forward pass (embed → smear → layers → backout → head) |
| 8 | Custom attention with GQA + sliding window |
| 9 | `engine.c`: prefill + decode pipeline |
| 10 | Tool use: calculator expression evaluator |
| 11 | `nanochat_cli.c`: interactive chat |
| 12 | End-to-end test: generate tokens matching Python reference |

### Phase 3 — Training Backward Pass (Week 5-6)

| Step | Description |
|---|---|
| 13 | Extend autograd: register backward functions for all GPT ops |
| 14 | `nanochat_model_backward()`: manual reverse-chain backward pass |
| 15 | `loss.c`: cross-entropy with mask, gradient computation |
| 16 | Gradient accumulation across micro-batches |
| 17 | RMSNorm + RoPE backward |
| 18 | Attention backward (causal, GQA, sliding window) |

### Phase 4 — Training Loop (Week 7-8)

| Step | Description |
|---|---|
| 19 | `optim.c`: AdamW implementation (fused step with bias correction) |
| 20 | `optim.c`: Muon (Polar Express + NorMuon) |
| 21 | `dataloader.c`: parquet reader + BOS-aligned best-fit packing |
| 22 | `train.c`: pretraining loop with gradient accumulation |
| 23 | LR/momentum/weight-decay schedulers |
| 24 | Gradient clipping and NaN detection |

### Phase 5 — SFT + RL (Week 9-10)

| Step | Description |
|---|---|
| 25 | `dataloader.c`: SFT data with loss masking |
| 26 | `train.c`: SFT training loop with conversation mixture |
| 27 | Task framework: SmolTalk, MMLU, GSM8K, SpellingBee datasets |
| 28 | `train.c`: RL training with GRPO |
| 29 | Calculator tool as reward function |
| 30 | `nanochat_train.c`: training CLI entry point |

### Phase 6 — Polish + Optimization (Week 11-12)

| Step | Description |
|---|---|
| 31 | Quantized inference (INT8, BITS2) for weights |
| 32 | Memory-mapped weight loading |
| 33 | fp8.c: CPU-noop fallback, CUDA stub |
| 34 | Performance profiling + SIMD optimization |
| 35 | Documentation + integration tests |
| 36 | Web server chat mode |

## Files Summary

### New files

```
scripts/convert_nanochat.py       # Weight + tokenizer conversion
examples/nanochat/CMakeLists.txt  # Build
examples/nanochat/nanochat.h      # Public API
examples/nanochat/config.h        # GPTConfig struct
examples/nanochat/weights.c       # Checkpoint load/save
examples/nanochat/weights.h
examples/nanochat/tokenizer.c     # BPE tokenizer
examples/nanochat/tokenizer.h
examples/nanochat/kv_cache.c      # KV cache
examples/nanochat/kv_cache.h
examples/nanochat/sampling.c      # Sampling + RNG
examples/nanochat/sampling.h
examples/nanochat/model.c         # GPT forward + backward
examples/nanochat/model.h
examples/nanochat/engine.c        # Inference engine
examples/nanochat/engine.h
examples/nanochat/optim.c         # Muon + AdamW optimizer
examples/nanochat/optim.h
examples/nanochat/loss.c          # Cross-entropy loss
examples/nanochat/loss.h
examples/nanochat/dataloader.c    # BOS-aligned best-fit dataloader
examples/nanochat/dataloader.h
examples/nanochat/train.c         # Training loop
examples/nanochat/train.h
examples/nanochat/nanochat_cli.c  # Chat CLI
examples/nanochat/nanochat_train.c # Training CLI
examples/nanochat/README.md       # Usage docs
```

### Files modified in Boat core

```
src/ops/rope.c           # NEW: RoPE precompute + apply
src/ops/gather.c         # NEW: Embedding lookup
src/ops/topk.c           # NEW: Top-k values + indices
src/ops/activation.c     # ADD: ReLU², tanh (fix stub), sigmoid (fix stub)
src/ops/muon.c           # NEW: Polar Express orthogonalization
include/boat/ops.h       # ADD: new op declarations
examples/CMakeLists.txt  # ADD: nanochat subdirectory
CMakeLists.txt           # ADD: optional BUILD_NANOCHAT option
```
