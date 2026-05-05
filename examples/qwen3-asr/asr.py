#!/usr/bin/env python3
"""asr.py — Qwen3-ASR-0.6B full inference pipeline.

Usage:
    python asr.py <wav_path>
    python asr.py D:\\huggingface\\Qwen3-ASR-0.6B\\test.wav
"""
import sys, os, json, time, re
import numpy as np

# Add local modules
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

from mel import load_wav, log_mel_spectrogram, SAMPLE_RATE
from encoder import AudioEncoder
from decoder import (TextDecoder, build_prompt_tokens, merge_audio_embeddings,
                      AUDIO_PLACEHOLDER_ID, IM_END_ID, EOS_IDS, load_vocab, softmax)


# GPT-2-style bytes_to_unicode mapping (used by Qwen2Tokenizer)
def _build_byte_decoder():
    """Build reverse mapping from Unicode chars to bytes (GPT-2 style)."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs)}

_BYTE_DECODER = _build_byte_decoder()


def decode_tokens(ids, id_to_token):
    """Decode token IDs to text using GPT-2 BPE decoding."""
    text_parts = []
    for tid in ids:
        token = id_to_token.get(tid, f"<|{tid}|>")
        if token.startswith("<|"):
            continue  # skip special tokens
        text_parts.append(token)

    # Join, then convert bytes_to_unicode mapping back to UTF-8
    text = "".join(text_parts)
    try:
        text = bytearray(_BYTE_DECODER[c] for c in text).decode("utf-8", errors="replace")
    except (KeyError, ValueError):
        pass  # fallback to direct join

    text = text.replace("\n", " ").strip()
    return text


def main():
    if len(sys.argv) < 2:
        print("Usage: python asr.py <wav_path>", file=sys.stderr)
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading audio: {wav_path}")
    t0 = time.time()
    audio, sr = load_wav(wav_path)
    print(f"  {len(audio)} samples, {sr}Hz, duration={len(audio)/sr:.1f}s")

    # Limit to 30 seconds for faster testing (model's max context)
    # max_samples = 30 * SAMPLE_RATE
    # if len(audio) > max_samples:
    #     print(f"  Truncating to 30s")
    #     audio = audio[:max_samples]

    # Mel spectrogram
    mel = log_mel_spectrogram(audio)
    print(f"  Mel: {mel.shape} ({mel.shape[-1]} frames, {mel.shape[-1]*0.01:.1f}s)")

    # Load models
    print("Loading models ...")
    t1 = time.time()
    encoder = AudioEncoder()
    decoder = TextDecoder()
    print(f"  Models loaded in {time.time()-t1:.1f}s")

    # Audio encoder forward
    print("Running audio encoder ...")
    t2 = time.time()
    audio_features = encoder.forward(mel)
    T_audio = audio_features.shape[0]
    print(f"  Audio features: {audio_features.shape} ({time.time()-t2:.1f}s)")

    # Build prompt and merge audio
    prompt_ids = build_prompt_tokens()
    merged_embeds = merge_audio_embeddings(
        decoder.embed_weight, prompt_ids, audio_features)

    # Positions for merged sequence
    total_len = merged_embeds.shape[0]
    positions = np.arange(total_len, dtype=np.int32)

    # Load vocab for decoding
    print("Loading vocab ...")
    id_to_token = load_vocab()

    # Autoregressive generation
    print(f"Generating (max 256 tokens, {total_len} prefix) ...")
    t3 = time.time()

    # Use KV-cached generate
    output_ids = decoder.generate(merged_embeds, positions, max_new_tokens=256)

    gen_time = time.time() - t3
    final_text = decode_tokens(output_ids, id_to_token)
    print(f"\n=== Transcription ===")
    print(f"  Raw IDs: {output_ids}")
    print(f"  Text: {final_text}")
    print(f"  ({gen_time:.1f}s, {len(output_ids)} tokens)")
    print(f"  Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
