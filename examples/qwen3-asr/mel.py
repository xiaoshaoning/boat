"""mel.py — Mel spectrogram extraction for Qwen3-ASR (wraps WhisperFeatureExtractor)"""
import numpy as np
from scipy.io import wavfile
from scipy import signal as sp_signal
from transformers import WhisperFeatureExtractor

SAMPLE_RATE = 16000

# Singleton feature extractor
_FE = None
def get_feature_extractor():
    global _FE
    if _FE is None:
        _FE = WhisperFeatureExtractor.from_pretrained(
            "D:\\huggingface\\Qwen3-ASR-0.6B")
    return _FE


def log_mel_spectrogram(audio):
    """Compute log-mel spectrogram.

    Args:
        audio: 1D float32 array, 16kHz waveform

    Returns:
        mel: [128, T] float32 log-mel spectrogram (unpadded, actual frames)
    """
    fe = get_feature_extractor()
    out = fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
    # out.input_features is [1, 128, 3000] padded to nb_max_frames
    mel_padded = out.input_features[0]

    # Find actual length (non-padded region)
    # Pad value is the first column if audio is empty, or we can compute
    # from audio length: T = floor(audio_len / hop_length) + 1 for centered STFT
    hop = fe.hop_length
    T_actual = len(audio) // hop + 1
    T_actual = min(T_actual, mel_padded.shape[1])

    return mel_padded[:, :T_actual].astype(np.float32)


def load_wav(path, target_sr=SAMPLE_RATE):
    """Load WAV file, convert to mono, resample to target_sr."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = sp_signal.resample(audio, num_samples)

    return audio.astype(np.float32), target_sr
