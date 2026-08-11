# Suggested Future Directions for the boat Framework

Based on recent work (CUDA backend, NanoChat LLM, Qwen3-ASR, Nougat-LaTeX OCR, LLM Serving API), here are several directions worth pursuing:

## 1. Integrate OCR Models into LLM Serving

The existing OpenAI-compatible HTTP service (from NanoChat) can be extended to support OCR models. Wrap Nougat-LaTeX or GLM-OCR as API endpoints that accept an image and return LaTeX formulas, making it easy to integrate with other tools.

**Key files:** `examples/llm_server/`, `examples/latex/`

## 2. Generic Vision Encoder (ViT)

The current Swin implementation is specific to Nougat. Building a generic ViT encoder can unlock more models: CLIP, DINOv2, SigLIP, etc. The framework already has the necessary building blocks (Multi-Head Attention, LayerNorm, GELU); they just need to be composed into a standard ViT architecture.

**New files:** `include/boat/layers/vit.h`, `src/layers/vit.c`

## 3. VLM (Vision-Language Model)

Connect the vision encoder (Swin/ViT) to the NanoChat LLM decoder through a projection layer, similar to the LLaVA architecture. The vision and language capabilities are each already in place in the framework; combining them is a natural extension.

## 4. Model Export Pipeline (Python Tool)

Write a Python conversion tool for boat that converts any HuggingFace model (PyTorch `.bin` / `.safetensors`) into boat's native format. Currently every model requires a hand-written C loader — a unified format would greatly reduce the cost of porting new models.

**New files:** `scripts/export_to_boat.py`

## 5. Qwen3-ASR Improvements

The ASR pipeline is already functional and can be further enhanced:
- Non-greedy decoding (beam search)
- Punctuation restoration
- Voice activity detection (VAD) to support real-time processing

**Related files:** `examples/qwen3_asr/`, `src/cuda/`

## 6. Quantization Toolkit

The framework already supports BITS1/BITS2/BITS4/BITS8 data types. Develop a quantization tool that converts floating-point models to low-bit formats using calibration data, which can significantly reduce the memory footprint and inference latency of large models.

**New files:** `scripts/quantize.py`, `src/quantization/`
