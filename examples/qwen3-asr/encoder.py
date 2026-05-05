"""encoder.py — Qwen3-ASR audio encoder inference (PyTorch backend).

Architecture:
  Conv front-end: 3 × stride-2 conv1d (kernel=3, padding=1) → Linear(7680→896)
  18 × Pre-LN Transformer encoder layers (d_model=896, 14 heads, head_dim=64)
  Post-projection: LayerNorm → Linear(896→896) + GELU → Linear(896→1024)

Weights are loaded from the .npz file. Chunk-local attention matches HF reference.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(MODEL_DIR, "qwen3_asr_weights.npz")

_WEIGHTS = None
def _load(key):
    global _WEIGHTS
    if _WEIGHTS is None:
        _WEIGHTS = np.load(NPZ_PATH)
    return _WEIGHTS[key]


# ---------------------------------------------------------------------------
# Pure-NumPy reference helpers (kept for educational reference)
# ---------------------------------------------------------------------------

def gelu_np(x):
    from scipy.special import erf as _erf
    return x * 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))

def layer_norm_np(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias

def linear_np(x, weight, bias=None):
    return x @ weight.T + (bias if bias is not None else 0)


# ---------------------------------------------------------------------------
# Sinusoidal positional embedding (NumPy, used at chunk level)
# ---------------------------------------------------------------------------

def sinusoidal_embedding(T, D=896):
    log_timescale_increment = np.log(10000.0) / (D // 2 - 1)
    inv_timescales = np.exp(-log_timescale_increment * np.arange(D // 2, dtype=np.float32))
    scaled_time = np.arange(T, dtype=np.float32)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    pe = np.zeros((T, D), dtype=np.float32)
    pe[:, :D // 2] = np.sin(scaled_time)
    pe[:, D // 2:] = np.cos(scaled_time)
    return pe


# ---------------------------------------------------------------------------
# Grouped-query attention block
# ---------------------------------------------------------------------------

class EncoderSelfAttention(nn.Module):
    """Multi-head self-attention for audio encoder (14 heads, no GQA)."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        NH = self.num_heads
        HD = self.head_dim

        Q = self.q_proj(x).reshape(B, T, NH, HD).transpose(1, 2)  # [B, NH, T, HD]
        K = self.k_proj(x).reshape(B, T, NH, HD).transpose(1, 2)
        V = self.v_proj(x).reshape(B, T, NH, HD).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class EncoderFFN(nn.Module):
    """Feed-forward network for encoder layer (no GQA, standard FFN)."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """Pre-LN encoder transformer layer."""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = EncoderSelfAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.final_layer_norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x, attn_mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, attn_mask)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc2(F.gelu(self.fc1(x)))
        x = residual + x
        return x


# ---------------------------------------------------------------------------
# Audio encoder (PyTorch backend)
# ---------------------------------------------------------------------------

class AudioEncoder:
    """Qwen3-ASR audio encoder — PyTorch backend.

    Processes mel spectrogram through conv frontend + 18 transformer layers
    with chunk-local attention, producing audio features for the text decoder.
    """

    def __init__(self):
        self.cfg = {
            "d_model": 896,
            "num_heads": 14,
            "head_dim": 64,
            "ffn_dim": 3584,
            "num_layers": 18,
            "max_src_positions": 1500,
            "output_dim": 1024,
            "downsample_hidden_size": 480,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()
        self._load_weights()
        self.eval()

    def _build_model(self):
        cfg = self.cfg
        # Conv frontend
        self.conv1 = nn.Conv2d(1, cfg["downsample_hidden_size"], 3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(cfg["downsample_hidden_size"], cfg["downsample_hidden_size"], 3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(cfg["downsample_hidden_size"], cfg["downsample_hidden_size"], 3, stride=2, padding=1, bias=True)
        mel_bins = 128
        conv_flat = cfg["downsample_hidden_size"] * ((((mel_bins + 1) // 2 + 1) // 2 + 1) // 2)
        self.conv_out = nn.Linear(conv_flat, cfg["d_model"], bias=False)
        # Transformer layers
        self.layers = nn.ModuleList([
            EncoderLayer(cfg["d_model"], cfg["num_heads"], cfg["ffn_dim"])
            for _ in range(cfg["num_layers"])
        ])
        # Post-projection
        self.ln_post = nn.LayerNorm(cfg["d_model"], eps=1e-5)
        self.proj1 = nn.Linear(cfg["d_model"], cfg["d_model"], bias=True)
        self.proj2 = nn.Linear(cfg["d_model"], cfg["output_dim"], bias=True)

    def _load_weights(self):
        """Load weights from .npz into PyTorch modules."""
        def load_to(param, key):
            arr = _load(key)
            param.data = torch.from_numpy(arr).to(self.device)

        load_to(self.conv1.weight, "audio_tower.conv2d1.weight")
        load_to(self.conv1.bias, "audio_tower.conv2d1.bias")
        load_to(self.conv2.weight, "audio_tower.conv2d2.weight")
        load_to(self.conv2.bias, "audio_tower.conv2d2.bias")
        load_to(self.conv3.weight, "audio_tower.conv2d3.weight")
        load_to(self.conv3.bias, "audio_tower.conv2d3.bias")
        load_to(self.conv_out.weight, "audio_tower.conv_out.weight")

        for i, layer in enumerate(self.layers):
            load_to(layer.self_attn.q_proj.weight, f"audio_tower.layers.{i}.self_attn.q_proj.weight")
            load_to(layer.self_attn.q_proj.bias, f"audio_tower.layers.{i}.self_attn.q_proj.bias")
            load_to(layer.self_attn.k_proj.weight, f"audio_tower.layers.{i}.self_attn.k_proj.weight")
            load_to(layer.self_attn.k_proj.bias, f"audio_tower.layers.{i}.self_attn.k_proj.bias")
            load_to(layer.self_attn.v_proj.weight, f"audio_tower.layers.{i}.self_attn.v_proj.weight")
            load_to(layer.self_attn.v_proj.bias, f"audio_tower.layers.{i}.self_attn.v_proj.bias")
            load_to(layer.self_attn.out_proj.weight, f"audio_tower.layers.{i}.self_attn.out_proj.weight")
            load_to(layer.self_attn.out_proj.bias, f"audio_tower.layers.{i}.self_attn.out_proj.bias")
            load_to(layer.self_attn_layer_norm.weight, f"audio_tower.layers.{i}.self_attn_layer_norm.weight")
            load_to(layer.self_attn_layer_norm.bias, f"audio_tower.layers.{i}.self_attn_layer_norm.bias")
            load_to(layer.fc1.weight, f"audio_tower.layers.{i}.fc1.weight")
            load_to(layer.fc1.bias, f"audio_tower.layers.{i}.fc1.bias")
            load_to(layer.fc2.weight, f"audio_tower.layers.{i}.fc2.weight")
            load_to(layer.fc2.bias, f"audio_tower.layers.{i}.fc2.bias")
            load_to(layer.final_layer_norm.weight, f"audio_tower.layers.{i}.final_layer_norm.weight")
            load_to(layer.final_layer_norm.bias, f"audio_tower.layers.{i}.final_layer_norm.bias")

        load_to(self.ln_post.weight, "audio_tower.ln_post.weight")
        load_to(self.ln_post.bias, "audio_tower.ln_post.bias")
        load_to(self.proj1.weight, "audio_tower.proj1.weight")
        load_to(self.proj1.bias, "audio_tower.proj1.bias")
        load_to(self.proj2.weight, "audio_tower.proj2.weight")
        load_to(self.proj2.bias, "audio_tower.proj2.bias")

    def eval(self):
        for m in self.layers.modules() if hasattr(self, 'layers') else []:
            m.eval()
        # Set all modules to eval mode
        for module in [self.conv1, self.conv2, self.conv3, self.conv_out,
                        self.ln_post, self.proj1, self.proj2]:
            module.eval()

    def forward(self, mel):
        """Audio encoder forward pass.

        Args:
            mel: [128, T] log-mel spectrogram (numpy float32)

        Returns:
            audio_features: [T_out, 1024] encoder output (numpy float32)
        """
        T = mel.shape[-1]
        cfg = self.cfg
        conv_chunk_size = 50 * 2  # 100 frames per chunk

        # Process conv frontend with chunking (matching reference)
        chunk_pool = []
        for start in range(0, T, conv_chunk_size):
            chunk = mel[:, start:start + conv_chunk_size]
            input_len = chunk.shape[1]
            if input_len < conv_chunk_size:
                pad = conv_chunk_size - input_len
                chunk = np.pad(chunk, ((0, 0), (0, pad)), mode='constant',
                               constant_values=mel[:, :1].mean())

            # Conv frontend
            x = torch.from_numpy(chunk).float().to(self.device)
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 100]
            with torch.no_grad():
                x = F.gelu(self.conv1(x))
                x = F.gelu(self.conv2(x))
                x = F.gelu(self.conv3(x))
                b, c, f, t = x.shape
                x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
                x = self.conv_out(x)  # [1, T_conv, 896]
            x_np = x.squeeze(0).cpu().numpy()  # [T_conv, 896]

            # Trim to valid frames
            feat_len = (input_len - 1) // 2 + 1
            temp = (feat_len - 1) // 2 + 1
            valid = (temp - 1) // 2 + 1
            x_np = x_np[:valid]

            # Add sinusoidal positional embedding (chunk-local positions)
            pe = sinusoidal_embedding(x_np.shape[0], cfg["d_model"])
            chunk_pool.append(x_np + pe)

        # Concatenate all chunks and process through transformer layers.
        # With eager attention (matching the reference), this is FULL attention
        # across all positions — no chunk-local masking needed.
        x = np.concatenate(chunk_pool, axis=0)  # [T_total, 896]
        x_t = torch.from_numpy(x).float().to(self.device).unsqueeze(0)  # [1, T_total, D]
        for layer in self.layers:
            with torch.no_grad():
                x_t = layer(x_t)  # full attention, no mask

        # Post-projection
        with torch.no_grad():
            x_t = self.ln_post(x_t)
            x_t = F.gelu(self.proj1(x_t))
            x_t = self.proj2(x_t)

        return x_t.squeeze(0).cpu().numpy()  # [T_total, 1024]
