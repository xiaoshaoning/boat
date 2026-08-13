# MiniMind inference in C

A from-scratch C inference implementation of [MiniMind](https://github.com/jingyaogong/minimind),
a miniature Chinese/English language model, written against the Boat deep-learning
framework conventions (but self-contained — it does not depend on the Boat library).

## Model

The default configuration (`config.h`) targets MiniMind-3:

| Hyperparameter | Value |
|---|---|
| vocab size | 6400 |
| hidden size | 768 |
| layers | 8 |
| attention heads | 8 (GQA 2:1, 4 KV heads) |
| head dim | 96 |
| intermediate size | 2432 (SwiGLU) |
| max sequence length | 2048 |
| RoPE theta | 1e6 |
| RMS norm eps | 1e-6 |

Weights are FP32 and `model.embed_tokens.weight` is tied to `lm_head.weight`.

## Building

The example is standalone (no Boat dependency); it only needs `../common/json.c` for
JSON parsing. To build the CLI:

```bash
gcc -std=c11 -O2 -Iexamples/minimind -Iexamples/common \
    examples/minimind/engine.c examples/minimind/minimind_cli.c \
    examples/minimind/model.c examples/minimind/sampling.c \
    examples/minimind/tokenizer.c examples/minimind/weights.c \
    examples/common/json.c -lm -o minimind_cli
```

Or via CMake (standalone fallback path, or linked against `boat` when built inside
the Boat project):

```bash
cmake -S examples/minimind -B examples/minimind/build
cmake --build examples/minimind/build
```

## Running

```bash
minimind_cli <model_dir> [prompt] [max_tokens] [temp] [top_k]
```

`<model_dir>` must contain `model.bin`, `model_meta.json`, and `tokenizer.json`
(see `weights/`). With no `prompt`, it enters an interactive chat loop.

## Model weights

`weights/model.bin` (~263 MB FP32) is **not** checked in. Obtain it by either:

- exporting a MiniMind PyTorch checkpoint with `export_weights.py`, or
- downloading an existing MiniMind-3 checkpoint and running:

```bash
python export_weights.py path/to/minimind.pth weights/
```

`weights/tokenizer.json` (BPE vocab + merges) is checked in, so only `model.bin`
needs to be supplied.

## Files

| Path | Purpose |
|---|---|
| `config.h` | Model hyperparameters |
| `model.c` / `model.h` | Transformer forward pass (RMSNorm, GQA attention + RoPE, SwiGLU FFN, KV cache) |
| `weights.c` / `weights.h` | Loads `model.bin` + `model_meta.json` |
| `tokenizer.c` / `tokenizer.h` | BPE tokenizer (loads `tokenizer.json`) |
| `sampling.c` / `sampling.h` | Top-k / temperature sampling |
| `engine.c` / `engine.h` | Prefill + decode generation loop |
| `minimind_cli.c` | CLI chat demo |
| `minimind_gen.c` | Non-interactive generation from `gen_input.bin` (test harness) |
| `export_weights.py` | PyTorch → flat FP32 `model.bin` exporter |
| `test_forward.py`, `test_gen.py`, `debug_layers.py`, `trace_*.py` | Python-reference cross-checks |

## Verification

`test_forward.py` compares the C forward pass against PyTorch reference activations
(see `debug_data/` and `forward_test/`, which are regenerated and git-ignored).
