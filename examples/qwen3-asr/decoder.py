"""decoder.py — Qwen3-ASR text decoder inference with KV cache"""
import sys, os
import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(MODEL_DIR, "qwen3_asr_weights.npz")

_WEIGHTS = None
def _load(key):
    global _WEIGHTS
    if _WEIGHTS is None:
        _WEIGHTS = np.load(NPZ_PATH)
    return _WEIGHTS[key]


def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return x / rms * weight


def linear(x, weight, bias=None):
    return x @ weight.T + (bias if bias is not None else 0)


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# RoPE (standard, applied via rotate_half on all 128 dims)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(theta_base, head_dim, max_seq_len):
    inv_freq = 1.0 / (theta_base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    return np.cos(freqs), np.sin(freqs)


_ROPE_COS, _ROPE_SIN = None, None
def _get_rope(max_seq_len=65536):
    global _ROPE_COS, _ROPE_SIN
    if _ROPE_COS is None:
        _ROPE_COS, _ROPE_SIN = precompute_rope_freqs(1000000.0, 128, max_seq_len)
    return _ROPE_COS, _ROPE_SIN


def apply_rope(x, positions, cos, sin):
    """apply_rotary_pos_emb via rotate_half. x: [T, NH, HD=128]"""
    cos_val = cos[positions]  # [T, 64]
    sin_val = sin[positions]  # [T, 64]
    # Duplicate to 128 dims (reference: cat(freqs, freqs, dim=-1))
    cos_val = np.concatenate([cos_val, cos_val], axis=-1)  # [T, 128]
    sin_val = np.concatenate([sin_val, sin_val], axis=-1)  # [T, 128]
    # Unsqueeze for head dim
    cos_val = cos_val[:, np.newaxis, :]  # [T, 1, 128]
    sin_val = sin_val[:, np.newaxis, :]  # [T, 1, 128]
    # rotate_half: swap halves and negate first half
    x_half = np.concatenate([-x[..., 64:], x[..., :64]], axis=-1)
    return x * cos_val + x_half * sin_val


def load_vocab(path="D:\\huggingface\\Qwen3-ASR-0.6B\\vocab.json"):
    import json
    with open(path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return {v: k for k, v in vocab.items()}


AUDIO_PLACEHOLDER_ID = 151676
IM_START_ID = 151644
IM_END_ID = 151645
EOS_IDS = {IM_END_ID, 151643}


# ---------------------------------------------------------------------------
# TextDecoder
# ---------------------------------------------------------------------------

class TextDecoder:
    """Qwen3-ASR text decoder (28-layer GQA + MRoPE + SiLU-MLP)."""

    def __init__(self):
        self.cfg = {
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "num_layers": 28,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
        }
        self.load_weights()

    def load_weights(self):
        cfg = self.cfg
        self.embed_weight = _load("model.embed_tokens.weight")
        self.lm_head_weight = _load("lm_head.weight")
        self.norm_weight = _load("model.norm.weight")

        self.layers = []
        for i in range(cfg["num_layers"]):
            layer = {
                "input_norm_w": _load(f"model.layers.{i}.input_layernorm.weight"),
                "post_attn_norm_w": _load(f"model.layers.{i}.post_attention_layernorm.weight"),
                "q_w": _load(f"model.layers.{i}.self_attn.q_proj.weight"),
                "k_w": _load(f"model.layers.{i}.self_attn.k_proj.weight"),
                "v_w": _load(f"model.layers.{i}.self_attn.v_proj.weight"),
                "o_w": _load(f"model.layers.{i}.self_attn.o_proj.weight"),
                "q_norm_w": _load(f"model.layers.{i}.self_attn.q_norm.weight"),
                "k_norm_w": _load(f"model.layers.{i}.self_attn.k_norm.weight"),
                "gate_w": _load(f"model.layers.{i}.mlp.gate_proj.weight"),
                "up_w": _load(f"model.layers.{i}.mlp.up_proj.weight"),
                "down_w": _load(f"model.layers.{i}.mlp.down_proj.weight"),
            }
            self.layers.append(layer)

    def _gqa_attention(self, x, layer, positions, cos, sin,
                       k_cache=None, v_cache=None):
        """Single layer GQA attention.

        Returns: [T, H=1024] and optionally (updated_k_cache, updated_v_cache)
        """
        cfg = self.cfg
        T = x.shape[0]
        NH = cfg["num_heads"]
        NKV = cfg["num_kv_heads"]
        HD = cfg["head_dim"]
        G = NH // NKV

        Q = linear(x, layer["q_w"]).reshape(T, NH, HD)
        K = linear(x, layer["k_w"]).reshape(T, NKV, HD)
        V = linear(x, layer["v_w"]).reshape(T, NKV, HD)

        Q = rms_norm(Q, layer["q_norm_w"])
        K = rms_norm(K, layer["k_norm_w"])

        Q = apply_rope(Q, positions, cos, sin)
        K = apply_rope(K, positions, cos, sin)

        Q_t = Q.transpose(1, 0, 2)  # [NH, T, HD]

        if k_cache is not None:
            start = int(positions[0])
            k_cache[:, start:start + T, :] = K.transpose(1, 0, 2)
            v_cache[:, start:start + T, :] = V.transpose(1, 0, 2)
            K_use = k_cache[:, :start + T, :]
            V_use = v_cache[:, :start + T, :]
        else:
            K_use = K.transpose(1, 0, 2)
            V_use = V.transpose(1, 0, 2)

        K_exp = np.repeat(K_use, G, axis=0)
        V_exp = np.repeat(V_use, G, axis=0)

        attn = (Q_t @ K_exp.transpose(0, 2, 1)) * (HD ** -0.5)

        if T > 1:
            T_full = K_exp.shape[1]
            mask = np.triu(np.ones((T, T_full), dtype=np.float32) * -np.inf, k=1)
            attn = attn + mask[np.newaxis, :, :]

        attn = softmax(attn, axis=-1)

        out = attn @ V_exp
        out = out.transpose(1, 0, 2).reshape(T, NH * HD)
        result = linear(out, layer["o_w"])

        if k_cache is not None:
            return result, (k_cache, v_cache)
        return result

    def _mlp(self, x, layer):
        gate = silu(linear(x, layer["gate_w"]))
        up = linear(x, layer["up_w"])
        return linear(gate * up, layer["down_w"])

    def forward(self, hidden_states, positions):
        """Full forward pass (no KV cache). Returns [T, V] logits."""
        cos, sin = _get_rope()
        x = hidden_states
        for i in range(self.cfg["num_layers"]):
            layer = self.layers[i]
            residual = x
            x_norm = rms_norm(x, layer["input_norm_w"])
            x = residual + self._gqa_attention(x_norm, layer, positions, cos, sin)
            residual = x
            x_norm = rms_norm(x, layer["post_attn_norm_w"])
            x = residual + self._mlp(x_norm, layer)
        x = rms_norm(x, self.norm_weight)
        return linear(x, self.lm_head_weight)

    def generate(self, prefix_embeds, prefix_positions,
                 max_new_tokens=256, eos_ids=None):
        """Autoregressive generation with KV cache."""
        cfg = self.cfg
        if eos_ids is None:
            eos_ids = EOS_IDS
        cos, sin = _get_rope()

        T_prefix = prefix_embeds.shape[0]
        max_T = T_prefix + max_new_tokens

        k_caches = [np.zeros((cfg["num_kv_heads"], max_T, cfg["head_dim"]),
                             dtype=np.float32) for _ in range(cfg["num_layers"])]
        v_caches = [np.zeros((cfg["num_kv_heads"], max_T, cfg["head_dim"]),
                             dtype=np.float32) for _ in range(cfg["num_layers"])]

        # Prefix pass: process entire prompt+audio prefix, fill caches
        x = prefix_embeds
        for i in range(cfg["num_layers"]):
            layer = self.layers[i]
            residual = x
            x_norm = rms_norm(x, layer["input_norm_w"])
            out, _ = self._gqa_attention(x_norm, layer, prefix_positions, cos, sin,
                                          k_caches[i], v_caches[i])
            x = residual + out

            residual = x
            x_norm = rms_norm(x, layer["post_attn_norm_w"])
            x = residual + self._mlp(x_norm, layer)

        # First token from full prefix
        x_normed = rms_norm(x, self.norm_weight)
        logits = linear(x_normed, self.lm_head_weight)
        next_id = int(logits[-1].argmax())

        output_ids = [next_id]
        if next_id in eos_ids:
            return output_ids

        current_pos = T_prefix

        for step in range(max_new_tokens - 1):
            next_embed = self.embed_weight[next_id:next_id + 1]
            x = next_embed
            new_pos = np.array([current_pos], dtype=np.int32)

            for i in range(cfg["num_layers"]):
                layer = self.layers[i]

                residual = x
                x_norm = rms_norm(x, layer["input_norm_w"])
                out, _ = self._gqa_attention(x_norm, layer, new_pos, cos, sin,
                                              k_caches[i], v_caches[i])
                x = residual + out

                residual = x
                x_norm = rms_norm(x, layer["post_attn_norm_w"])
                x = residual + self._mlp(x_norm, layer)

            x_normed = rms_norm(x, self.norm_weight)
            logits = linear(x_normed, self.lm_head_weight)
            next_id = int(logits[0].argmax())

            output_ids.append(next_id)
            current_pos += 1

            if next_id in eos_ids:
                break

        return output_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt_tokens():
    """Build prompt token sequence with correct BPE token IDs.

    Prompt format: <|im_start|>transcribe\n<|audio_placeholder|>\n<|im_end|>\n<|im_start|>output\n
    BPE tokenization: 'transcribe' → [1458, 3114], 'output' → [3006]
    """
    tokens = []
    tokens.append(IM_START_ID)   # 151644
    tokens.append(1458)          # 'trans'
    tokens.append(3114)          # 'cribe'
    tokens.append(198)           # '\n'
    tokens.append(AUDIO_PLACEHOLDER_ID)  # 151676
    tokens.append(198)           # '\n'
    tokens.append(IM_END_ID)     # 151645
    tokens.append(198)           # '\n'
    tokens.append(IM_START_ID)   # 151644
    tokens.append(3006)          # 'output'
    tokens.append(198)           # '\n'
    return tokens


def merge_audio_embeddings(embed_weight, prompt_ids, audio_features):
    prompt_ids_arr = np.array(prompt_ids, dtype=np.int32)
    embeds = embed_weight[prompt_ids_arr]
    placeholder_pos = prompt_ids.index(AUDIO_PLACEHOLDER_ID)
    merged = np.concatenate([
        embeds[:placeholder_pos],
        audio_features,
        embeds[placeholder_pos + 1:],
    ], axis=0)
    return merged
