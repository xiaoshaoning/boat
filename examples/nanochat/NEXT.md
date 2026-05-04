# Next Steps — NanoChat CUDA Inference

## Performance

- ~~**Fused attention kernels**: Current prefill attention extracts per-head Q/K and calls cuBLAS SGEMM per head (~17x sequential calls). A fused kernel or batched GEMM would be much faster.~~
  ✅ DONE — `fused_prefill_attention_cuda` (2D grid seq_len×num_heads) and `fused_decode_attention_cuda` (1D grid num_heads) replace per-head cuBLAS calls. Eliminated ~85 cuBLAS launches per decode step, ~100+ kernel calls per prefill.
- ~~**Decode temp buffer**: The decode function does `cudaMalloc` + `cudaFree` for tmp and ctx buffers every token — can pre-allocate once.~~
  ✅ DONE — `d_decode_tmp` and `d_decode_hidden` pre-allocated in `nanochat_cuda_model_t`, reused across all decode steps.
- ~~**BF16 inference**: Weights are BF16 in storage but loaded as FP32. Running in BF16 matches model training format and halves memory (~11GB → ~5.5GB). FP16 found to overflow (model hidden states RMS=365K+ at layer 33, exceeds FP16 max 65504).~~
  ✅ DONE — All weights/KV cache stored as BF16 on GPU, cuBLAS GemmEx (BF16 I/O `CUDA_R_16BF`, FP32 accumulate), BF16 fused attention, all element-wise ops have BF16 I/O with FP32 internal compute. lm_head outputs FP32 logits via `matmul_bt_bf16_out_f32_cuda` for softcap. Engine uses `embed_gather_bf16_cuda`.

## Usability

- ~~**Interactive chat mode + token streaming**: REPL loop for back-and-forth conversation, auto-wrapping prompts in `<|user_start|>...<|user_end|><|assistant_start|>` format, stripping special tokens from output.~~
  ✅ DONE — `nanochat_chat()` with colored REPL, `/reset` command, history management, context truncation. `nanochat_generate_stream()` with per-token callback. Special token IDs inserted directly into token array (bypasses BPE tokenization of special tokens). Streaming callback filters leaked `<|...|>` markers from output.

## Correctness / Diagnostics

- **Numerical diff against HF**: Run full 34-layer forward pass comparing CUDA vs PyTorch tensor-by-tensor to verify no remaining subtle bugs (attention masking, scale, softmax precision).


