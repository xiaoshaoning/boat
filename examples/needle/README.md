# Needle 2 via Boat

Runs the **Needle 2** 45M-parameter Simple Attention Network (tool-calling /
structured-extraction model) from its self-contained `.cact` deployment blob,
using a from-scratch C implementation of the SAN decode path built on the Boat
example scaffold.

- `cact.[ch]` — parser + Cactus-Quants (CQ2/CQ3/CQ4/ternary) dequantizer for
  the `.cact` byte layout documented in `needle/model/export.py` (120-byte
  geometry header, 44-byte tensor directory records, LSB-packed codebook
  indices + FP16 per-group L2 norms, 64-byte aligned blobs).
- `tokenizer.[ch]` — the self-contained SentencePiece BPE tokenizer embedded
  in the blob (the `RefTokenizer` spec: merge-by-score, byte fallback, chat
  markers, dummy prefix).
- `san.[ch]` — the forward/generate path mirroring `needle/model/decode.py`:
  tied embedding, 27-layer MHC scan (4 lanes, Sinkhorn routing), gated GQA
  attention with RoPE + 256-token sliding window and a per-layer KV cache,
  Walsh-Hadamard MLP (fast transform), engram hashed n-gram KV memory at the
  configured sites, final ZCRMSNorm and tied logits.
- `main.c` — CLI (`needle2`), mirroring `needle run`.

Verified token-for-token against the JAX reference decode (`generate_cached`)
running the same dequantized weights: 24/24 generated tokens identical.

## Build

```sh
cmake -S . -B build -DBOAT_WITH_EXAMPLES=ON
cmake --build build --target needle2
```

## Run

The model weights are not in this repo; download `needle2.cact` from
`huggingface.co/Cactus-Compute/needle2` (or point at the local mirror):

```sh
./needle2 /path/to/needle2.cact --prompt "what is the weather in Lagos right now?" \
    --max-new-tokens 40
# ,"arguments":{"location":"Lagos"}}]</tool_call><|im_end|>

./needle2 /path/to/needle2.cact --prompt "The most surprising thing about" \
    --max-new-tokens 24 --print-tokens     # raw generated token ids
./needle2 --selftest                       # tokenizer self-test, no model file
```

On this machine the mirror lives at `D:\hugginface\needle2\needle2.cact`.

```sh
make needle2           # builds examples/needle/needle2(.exe) in place
```

See [VALGRIND.md](VALGRIND.md) for the recorded WSL2 valgrind run.

## Notes

- Pure CPU, no external dependencies beyond the boat library (used for the
  build/test harness and fp16 conversion; the SAN math is self-contained).
- The base model produces unconstrained text; the production engine additionally
  applies the tool-call grammar — this example exposes the raw autoregressive
  decode, so short prompts may emit tool-call fragments (as shown above).
- ~90 MB of fp32 weights are materialized at load time from the 13 MB blob.
