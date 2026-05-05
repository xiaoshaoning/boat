# Qwen3-ASR-0.6B Inference Example — Implementation Plan

## Overview

Build a Python ASR inference pipeline for Qwen3-ASR-0.6B (audio → text) as a self-contained example under `examples/qwen3-asr/`, following the same pattern as NanoChat.

## Model Architecture Summary

```
Audio (16kHz) → Mel [128,T] → 3× Conv2D + GELU (stride 2)
  → ConvOut [T',896] → + Sinusoidal PE → 18× EncoderLayer [T',896]
  → ln_post → proj1+GELU → proj2 → [T',1024]
  → merged into text embedding at placeholder positions
  → 28× DecoderLayer (GQA + MRoPE + SiLU-MLP) → norm → lm_head → vocab logits
```

## Implementation Phases

### Phase 1: Weight converter

Write `convert_weights.py`:
- Load `model.safetensors` via `safetensors` library
- Convert BF16 → FP32 numpy arrays
- Dump to `.npz` for efficient loading
- No framework changes — standalone script

### Phase 2: Mel Spectrogram

Write `mel.py`:
- Whisper-style log-mel spectrogram from numpy:
  - 16kHz input, pre-emphasis, Hann windowing
  - 400-pt FFT, 160-hop (10ms stride)
  - 128 mel bins (0-8000Hz), log1p compression
- WAV loading with 24kHz → 16kHz resampling

### Phase 3: Audio Encoder

Write `encoder.py`:
- Load weights from .npz
- Chunked 2D convolution: Conv2D(1→480) → Conv2D(480→480) → Conv2D(480→480) + GELU
- Reshape → Linear(7680→896) — the "conv_out" downsampling
- Sinusoidal position embedding (computed on the fly)
- 18 encoder layers: Pre-LN → MHA(14 heads, head_dim=64) → Residual → Pre-LN → GELU-FFN → Residual
- Post-encoder: LayerNorm → Linear+GELU → Linear → [T',1024]

### Phase 4: Text Decoder

Write `decoder.py`:
- Token embedding lookup [vocab_size, 1024]
- Audio merge: masked_scatter audio features into `<|audio_placeholder|>` positions
- 28 decoder layers:
  - Pre-LN (RMSNorm) → QKV (Q: 1024→2048, K/V: 1024→1024)
  - Per-head Q/K norm (RMSNorm, dim=128) → MRoPE (3D interleaved, [24,20,20], theta=1e6)
  - GQA attention (16Q/8KV, causal mask for text portion)
  - Residual → Post-LN → SiLU-gated MLP (gate+up→3072, down→1024) → Residual
- Final RMSNorm → lm_head (tied with embed)

### Phase 5: Generation Loop & Entry Point

Write `asr.py`:
- Prompt: `<|im_start|>transcribe\n<|audio_placeholder|>\n<|im_end|>\n<|im_start|>output\n`
- Encode audio once → features [T',1024]
- Autoregressive decode: merge features → decoder → argmax → append → repeat
- EOS at `<|im_end|>` (id=151645), max 256 tokens
- Token → text via BPE (tiktoken or manual decode from vocab.json)
- Resample WAV 24kHz→16kHz

## Verification

- Phase 2: Mel output matches `WhisperFeatureExtractor` (max error < 1e-4)
- Phase 3: Encoder output matches `qwen_asr` reference (cosine sim > 0.999)
- Phase 4: Decoder logits argmax matches reference for first 5 tokens
- Phase 5: `test.wav` transcribes to readable Chinese text

## Files

```
examples/qwen3-asr/
├── PLAN.md              — This document
├── README.md            — Usage instructions
├── convert_weights.py   — safetensors → .npz
├── mel.py               — Mel spectrogram
├── encoder.py           — Audio encoder
├── decoder.py           — Text decoder
└── asr.py               — Full pipeline runner
```
