"""extract_mel.py — Extract mel spectrogram to binary for C consumption.

Usage:
    python extract_mel.py <wav_path> <output_bin>
    python extract_mel.py test.wav mel.bin

Output binary format:
    int32 T_mel           (number of mel frames)
    float[128 * T_mel]    (log-mel spectrogram values, row-major: [mel_bin, frame])
"""

import sys, os
import numpy as np

# Add local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mel import load_wav, log_mel_spectrogram


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_mel.py <wav_path> <output_bin>", file=sys.stderr)
        sys.exit(1)

    wav_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading audio: {wav_path}")
    audio, sr = load_wav(wav_path)
    print(f"  {len(audio)} samples, {sr}Hz")

    print("Extracting mel spectrogram ...")
    mel = log_mel_spectrogram(audio)
    print(f"  Mel shape: {mel.shape}")

    # Write binary: int32 T + float[128, T]
    T = mel.shape[1]
    with open(output_path, "wb") as f:
        f.write(np.int32(T).tobytes())
        f.write(mel.astype(np.float32).tobytes())

    print(f"Written to {output_path} ({T} frames, {os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    main()
