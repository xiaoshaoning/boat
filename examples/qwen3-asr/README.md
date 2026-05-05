# Qwen3-ASR-0.6B — Self-Contained Speech-to-Text Example

A lightweight ASR inference pipeline for Qwen3-ASR-0.6B with minimal dependencies.

## Architecture

- **Audio encoder** (`encoder.py`): PyTorch backend. Conv frontend (3× stride-2) → 18 Pre-LN Transformer layers (d_model=896, 14 heads) → post-projection (1024-dim). Processes mel spectrograms in 100-frame chunks through conv layers, then full attention across all positions.

- **Text decoder** (`decoder.py`): Pure NumPy. 28-layer GQA transformer (16Q/8KV heads, head_dim=128) with KV-cached autoregressive generation. RMS Q/K pre-norm, SiLU-gated MLP, RoPE via rotate_half.

- **Utilities**:
  - `mel.py`: Mel spectrogram extraction (Whisper feature extractor)
  - `convert_weights.py`: Convert safetensors → npz for self-contained deployment
  - `asr.py`: Main pipeline script

## Usage

```bash
python asr.py <path-to-wav-file>
```

Example:
```bash
python asr.py test.wav
```

## Requirements

- Python 3.10+
- PyTorch (encoder)
- NumPy (decoder)
- `transformers` and `librosa` (mel extraction, weight conversion only)

The decoder runs purely on NumPy (no PyTorch needed for text generation).

## Files

| File | Description |
|------|-------------|
| `encoder.py` | PyTorch audio encoder (conv frontend + 18-layer Transformer) |
| `decoder.py` | NumPy text decoder with KV-cached generation |
| `asr.py` | Full inference pipeline (WAV → text) |
| `mel.py` | Mel spectrogram extraction |
| `convert_weights.py` | safetensors → npz weight converter |
| `qwen3_asr_weights.npz` | Model weights (1.7GB, not tracked in git) |
