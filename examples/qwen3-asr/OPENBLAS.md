OpenBLAS Integration for Qwen3-ASR
==================================

The qwen3asr binary is linked against OpenBLAS for accelerated matrix
multiplication. All linear projections (Q/K/V/O, MLP gate/up/down, LM head)
and attention score/weighted-sum computations use BLAS sgemm calls.

Architecture
------------
- Linear projections (NoTrans,NoTrans): boat_sgemm() from the boat framework,
  which delegates to cblas_sgemm when BOAT_USE_OPENBLAS is defined
- Attention Q@K^T (NoTrans,Trans): cblas_sgemm() called directly per head
- Attention weighted sum (NoTrans,NoTrans): cblas_sgemm() per head
- Fallback: original triple-loop code when OpenBLAS is not available

Building
--------
Requires BOAT_WITH_OPENBLAS=ON when configuring cmake:

  cmake -B build -G "Visual Studio 18 2026"       \
    -DBOAT_WITH_EXAMPLES=ON                        \
    -DBOAT_WITH_OPENBLAS=ON                        \
    -DBOAT_OPENBLAS_ROOT=D:/github/OpenBLAS/build_msvc/install

OpenBLAS location: D:/github/OpenBLAS/build_msvc/install/

Benchmark (Release, 3000-frame mel, 118 generated tokens)
----------------------------------------------------------
Before (hand-tuned SGEMM):  encoder=44.7s  prefix=32.3s  gen=19.7s  total=102s
After  (OpenBLAS):          encoder=67.0s  prefix=26.2s  gen=24.1s  total=123s

OpenBLAS helps the decoder prefix (~19% faster, larger matmuls with
1024x4096 dims) but hurts the encoder and generation (per-head attention
sgemm calls with T=390, HD=64 are too small to amortize BLAS call
overhead). The boat framework's hand-tuned SGEMM (AVX2 tiled micro-kernel)
handles small-to-medium matmuls well without BLAS dispatch overhead.

For optimal performance:
- Decoder-only or large-batch: OpenBLAS helps
- Encoder-heavy or small T: hand-tuned SGEMM is better
