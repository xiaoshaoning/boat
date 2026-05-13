#!/usr/bin/env python3
"""
quantize.py - Quantize FP32 model weights for boat inference.

Quantizes HuggingFace-exported safetensors to UINT8/INT8/BITS2/BITS1,
storing scale and zero_point as companion tensors that a C loader can
read to reconstruct FP32 values on-the-fly.

The output is a quantized .safetensors file.  For each FP32 weight tensor
named "<name>", the output contains:

    <name>          — quantized data (U8, I8, or packed U8 for BITS2/BITS1)
    <name>.scale    — FP32 scale (scalar or 1-D for per-channel)
    <name>.zp       — int32 zero_point (scalar or 1-D for per-channel)
    <name>.qdtype   — uint8 quant-dtype marker (only for BITS2/BITS1)

Usage:
    # Quantize a single safetensors file to UINT8
    python scripts/quantize.py model.safetensors -o quantized.safetensors

    # Quantize a model directory (loads sharded safetensors)
    python scripts/quantize.py D:/models/my-model/ --dtype int8 --symmetric

    # Per-channel quantization to 2-bit
    python scripts/quantize.py model.safetensors --dtype bits2 --per-channel

    # Just inspect tensor sizes and estimated compression
    python scripts/quantize.py model.safetensors --dry-run
"""

import argparse
import json
import os
import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Quantization algorithms (mirrors boat/src/core/quantize.c logic)
# ---------------------------------------------------------------------------

def compute_quant_params(min_val: float, max_val: float,
                         quant_dtype: str, symmetric: bool):
    """Compute scale and zero_point, matching boat_compute_quant_params().

    Returns (scale, zero_point).
    """
    if quant_dtype == "float4" or quant_dtype == "f4":
        return 1.0, 0

    # Determine qmin/qmax
    ranges = {
        "uint8":   (0, 255),
        "int8":    (-128, 127),
        "bits2":   (0, 3),
        "bits1":   (0, 1),
    }
    qmin, qmax = ranges[quant_dtype]

    if min_val >= max_val:
        return 1.0, 0

    if symmetric:
        if quant_dtype == "uint8":
            scale = max(abs(min_val), abs(max_val)) / 127.0
            zp = 128
        elif quant_dtype == "int8":
            scale = max(abs(min_val), abs(max_val)) / 127.0
            zp = 0
        else:
            # bits1 / bits2 are always [0, qmax]; symmetric doesn't apply
            scale = (max_val - min_val) / float(qmax - qmin)
            if scale < 1e-8:
                scale = 1e-8
            zp = int(round(-min_val / scale))
    else:
        scale = (max_val - min_val) / float(qmax - qmin)
        if scale < 1e-8:
            scale = 1e-8
        zp = int(round(-min_val / scale))

    # Clamp zero_point
    if quant_dtype in ("bits1", "bits2"):
        zp = max(0, min(zp, qmax))
    elif quant_dtype == "uint8":
        zp = max(0, min(zp, 255))
    elif quant_dtype == "int8":
        zp = max(-128, min(zp, 127))

    return scale, zp


def quantize_array(arr: np.ndarray, quant_dtype: str,
                   symmetric: bool, per_channel: bool = False,
                   channel_dim: int = 0):
    """Quantize a FP32 numpy array.

    Args:
        arr: FP32 numpy array.
        quant_dtype: "uint8", "int8", "bits2", "bits1".
        symmetric: Use symmetric quantization.
        per_channel: Quantize each channel independently.
        channel_dim: Which dimension is the channel dimension.

    Returns:
        (quantized_data, scale, zero_point, actual_dtype)
        - actual_dtype is the safetensors dtype string for storage.
    """
    qdtype = quant_dtype
    dtype_safetensors_map = {
        "uint8": "U8",
        "int8": "I8",
        "bits2": "U8",   # packed, stored as U8
        "bits1": "U8",   # packed, stored as U8
        "float4": "U8",  # packed, stored as U8
    }
    storage_dtype = dtype_safetensors_map.get(qdtype, "U8")

    if per_channel:
        return _quantize_per_channel(arr, qdtype, symmetric, channel_dim, storage_dtype)
    else:
        return _quantize_per_tensor(arr, qdtype, symmetric, storage_dtype)


def _quantize_per_tensor(arr: np.ndarray, quant_dtype: str,
                         symmetric: bool, storage_dtype: str):
    """Per-tensor quantization."""
    min_val = float(arr.min())
    max_val = float(arr.max())
    scale, zp = compute_quant_params(min_val, max_val, quant_dtype, symmetric)

    if quant_dtype == "float4":
        qdata = pack_float4(arr)
        return qdata, np.float32(scale), np.int32(zp), storage_dtype, quant_dtype

    if quant_dtype == "bits2":
        qvals = np.round(arr / scale + zp).clip(0, 3).astype(np.uint8)
        qdata = pack_bits2(qvals)
        return qdata, np.float32(scale), np.int32(zp), storage_dtype, quant_dtype

    if quant_dtype == "bits1":
        qvals = np.round(arr / scale + zp).clip(0, 1).astype(np.uint8)
        qdata = pack_bits1(qvals)
        return qdata, np.float32(scale), np.int32(zp), storage_dtype, quant_dtype

    # UINT8 / INT8
    qmin, qmax = (0, 255) if quant_dtype == "uint8" else (-128, 127)
    qvals = np.round(arr / scale + zp).clip(qmin, qmax).astype(
        np.uint8 if quant_dtype == "uint8" else np.int8)
    return qvals, np.float32(scale), np.int32(zp), storage_dtype, quant_dtype


def _quantize_per_channel(arr: np.ndarray, quant_dtype: str,
                          symmetric: bool, channel_dim: int,
                          storage_dtype: str):
    """Per-channel quantization.

    Returns scales/zps as 1-D arrays of shape (n_channels,).
    """
    n_channels = arr.shape[channel_dim]
    # Permute so channel_dim -> 0
    perm = list(range(arr.ndim))
    perm[0], perm[channel_dim] = perm[channel_dim], perm[0]
    permuted = np.transpose(arr, perm)

    slices = []
    scales = np.zeros(n_channels, dtype=np.float32)
    zps = np.zeros(n_channels, dtype=np.int32)
    actual_dtype = quant_dtype

    for c in range(n_channels):
        channel_data = permuted[c]
        mv = float(channel_data.min())
        Mv = float(channel_data.max())
        s, z = compute_quant_params(mv, Mv, quant_dtype, symmetric)
        scales[c] = s
        zps[c] = z

        if quant_dtype == "bits2":
            qv = np.round(channel_data / s + z).clip(0, 3).astype(np.uint8)
            slices.append(pack_bits2(qv))
        elif quant_dtype == "bits1":
            qv = np.round(channel_data / s + z).clip(0, 1).astype(np.uint8)
            slices.append(pack_bits1(qv))
        elif quant_dtype == "float4":
            slices.append(pack_float4(channel_data))
        else:
            qmin, qmax = (0, 255) if quant_dtype == "uint8" else (-128, 127)
            qv = np.round(channel_data / s + z).clip(qmin, qmax).astype(
                np.uint8 if quant_dtype == "uint8" else np.int8)
            slices.append(qv)

    # We can't simply stack packed arrays (different sizes per channel after
    # packing).  For per-channel with packing, we store each channel
    # separately as a flat array and flatten them into one big array.
    if quant_dtype in ("bits2", "bits1", "float4"):
        # For packed formats, flatten per-channel packed bytes sequentially.
        # The loader must know n_channels and channel_dim to unpack.
        packed_sizes = np.array([len(s) for s in slices], dtype=np.int32)
        qdata = np.concatenate([s.ravel() for s in slices])
        # Prepend packed-sizes metadata so the C loader can split channels.
        # Packed sizes stored as int32 header at the start.
        meta_header = packed_sizes.tobytes()
        qdata = np.frombuffer(meta_header + qdata.tobytes(), dtype=np.uint8).copy()
        return qdata, scales, zps, storage_dtype, quant_dtype

    # For UINT8/INT8, concatenate along the channel dim (back-permuted)
    qarr = np.stack(slices, axis=0)  # (C, ...)
    # Invert permutation
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    qarr = np.transpose(qarr, inv_perm)
    return qarr, scales, zps, storage_dtype, quant_dtype


# ---------------------------------------------------------------------------
# Packing helpers for BITS2 / BITS1 (mirror boat/src/core/packed.c)
# ---------------------------------------------------------------------------

def pack_bits2(values: np.ndarray) -> np.ndarray:
    """Pack uint8 values in [0,3] into 4-per-byte layout.

    Byte layout: | v0 | v1 | v2 | v3 |  (v0 = byte & 3, v1 = (byte>>2)&3, ...)
    Same as boat_pack_bits2.
    """
    flat = values.ravel()
    n = len(flat)
    out_size = (n + 3) // 4
    out = np.zeros(out_size, dtype=np.uint8)
    for i in range(n):
        if i >= len(flat):
            break
        byte_idx = i // 4
        shift = (i % 4) * 2
        out[byte_idx] |= (flat[i] & 3) << shift
    return out


def pack_bits1(values: np.ndarray) -> np.ndarray:
    """Pack uint8 values in [0,1] into 8-per-byte layout.

    Same as boat_pack_bits1.
    """
    flat = values.ravel()
    n = len(flat)
    out_size = (n + 7) // 8
    out = np.zeros(out_size, dtype=np.uint8)
    for i in range(n):
        byte_idx = i // 8
        shift = i % 8
        out[byte_idx] |= (flat[i] & 1) << shift
    return out


def pack_float4(values: np.ndarray) -> np.ndarray:
    """Pack FP32 values into E2M1 4-bit floats, 2 per byte.

    Simple mapping: the 4-bit float format uses 1 sign + 2 exponent + 1 mantissa.
    This is a simplified version matching boat's FLOAT4 format.
    """
    flat = values.ravel()
    n = len(flat)
    out_size = (n + 1) // 2
    out = np.zeros(out_size, dtype=np.uint8)
    for i in range(0, n, 2):
        lo = _fp32_to_fp4_e2m1(flat[i])
        hi = _fp32_to_fp4_e2m1(flat[i + 1]) if i + 1 < n else 0
        out[i // 2] = lo | (hi << 4)
    return out


def _fp32_to_fp4_e2m1(val: float) -> int:
    """Convert float32 to 4-bit E2M1 format.

    E2M1: 1 sign bit, 2 exponent bits (bias=1), 1 mantissa bit.
    Representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, ...
    Actually a simplified mapping: clamp and quantize.
    """
    # Simplified: just clamp and quantize to nearest representable value
    v = min(max(val, -6.0), 6.0)
    sign = 0
    if v < 0:
        sign = 1
        v = -v
    # E2M1 representable positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    idx = min(range(len(table)), key=lambda i: abs(v - table[i]))
    return sign << 3 | idx


# ---------------------------------------------------------------------------
# Safetensors loader (reuses logic from export_to_boat.py)
# ---------------------------------------------------------------------------

def load_safetensors(path_or_dir: str) -> dict:
    """Load all tensors from a safetensors file or directory.

    Returns {name: numpy_array} with all dtypes converted to float32.
    """
    tensors = {}

    if os.path.isfile(path_or_dir):
        files = [path_or_dir]
    elif os.path.isdir(path_or_dir):
        files = sorted([
            os.path.join(path_or_dir, f)
            for f in os.listdir(path_or_dir)
            if f.endswith(".safetensors")
        ])
        if not files:
            # Fall back to .bin
            files = sorted([
                os.path.join(path_or_dir, f)
                for f in os.listdir(path_or_dir)
                if f.endswith(".bin") and not f.startswith(".") and f != "pytorch_model.bin.un~"
            ])
            if files:
                return _load_bin_files(files)
            print(f"ERROR: No weight files found in {path_or_dir}")
            sys.exit(1)
    else:
        print(f"ERROR: {path_or_dir} not found")
        sys.exit(1)

    for fpath in files:
        _load_safetensors_file(fpath, tensors)
    return tensors


def _load_safetensors_file(fpath: str, out: dict):
    """Load a .safetensors file, converting all float types to FP32."""
    with open(fpath, "rb") as f:
        data = f.read()

    header_len = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + header_len].decode("utf-8"))
    payload = data[8 + header_len:]

    dtype_map = {
        "F32": np.float32, "F64": np.float64, "F16": np.float16,
        "BF16": np.dtype("bfloat16") if hasattr(np, "dtype") and "bfloat16" in np.dtype.__dict__ else np.uint16,
        "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
        "U64": np.uint64, "U32": np.uint32, "U16": np.uint16, "U8": np.uint8,
        "BOOL": np.bool_,
    }

    for name, info in header.items():
        if name.startswith("__"):
            continue
        dtype_str = info["dtype"]
        shape = info["shape"]
        start, end = info["data_offsets"]
        np_dtype = dtype_map.get(dtype_str)
        if np_dtype is None:
            print(f"  WARNING: Unknown dtype {dtype_str} for {name}, skipping")
            continue
        raw = payload[start:end]
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()

        # Convert to float32
        if np.issubdtype(arr.dtype, np.floating) or arr.dtype == np.uint16:
            arr = arr.astype(np.float32)
        out[name] = arr

    print(f"  Loaded {fpath}")


def _load_bin_files(files: list) -> dict:
    """Load PyTorch .bin files."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch required to load .bin files.  pip install torch")
        sys.exit(1)

    tensors = {}
    for fpath in files:
        print(f"  Loading {os.path.basename(fpath)} ...")
        state = torch.load(fpath, map_location="cpu", weights_only=True)
        for key, val in state.items():
            tensors[key] = val.float().numpy()
    return tensors


# ---------------------------------------------------------------------------
# Safetensors writer
# ---------------------------------------------------------------------------

def _dtype_to_st_str(dt: np.dtype) -> str:
    mapping = {
        np.float64: "F64", np.float32: "F32", np.float16: "F16",
        np.int64: "I64", np.int32: "I32", np.int16: "I16",
        np.int8: "I8", np.uint64: "U64", np.uint32: "U32",
        np.uint16: "U16", np.uint8: "U8", np.bool_: "BOOL",
    }
    return mapping.get(dt, str(dt))


def safetensors_write(tensors: dict, path: str):
    """Write {name: numpy_array} to a safetensors file."""
    header = {}
    data_chunks = []
    offset = 0

    for name, arr in tensors.items():
        dtype_str = _dtype_to_st_str(arr.dtype)
        shape = list(arr.shape)
        nbytes = arr.nbytes
        header[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        data_chunks.append(arr.tobytes())
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * padding

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk in data_chunks:
            f.write(chunk)

    total_gb = offset / 1024**3
    print(f"  Wrote {len(tensors)} tensors ({total_gb:.2f} GB) to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def estimate_compression(tensors: dict, quant_dtype: str):
    """Print compression estimates without modifying data."""
    bytes_per_elem = {"uint8": 1, "int8": 1, "bits2": 0.25, "bits1": 0.125, "float4": 0.5}
    bpe = bytes_per_elem.get(quant_dtype, 1)

    total_fp32 = 0
    total_quant = 0
    print(f"\n  Compression estimate (quant_dtype={quant_dtype}):")
    print(f"  {'Tensor':<50} {'Shape':<30} {'FP32 MB':>8} {'Quant MB':>8} {'Ratio':>6}")
    print(f"  {'-'*50} {'-'*30} {'-'*8} {'-'*8} {'-'*6}")

    for name, arr in tensors.items():
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        fp32_mb = arr.nbytes / 1024**2
        n_elems = arr.size
        quant_bytes = int(np.ceil(n_elems * bpe))
        # Add scale/zp overhead (8 bytes per tensor or per channel)
        quant_bytes += 8
        quant_mb = quant_bytes / 1024**2
        ratio = fp32_mb / quant_mb if quant_mb > 0 else 0
        total_fp32 += fp32_mb
        total_quant += quant_mb
        print(f"  {name:<50} {str(list(arr.shape)):<30} {fp32_mb:>8.2f} {quant_mb:>8.2f} {ratio:>5.1f}x")

    overall = total_fp32 / total_quant if total_quant > 0 else 0
    print(f"  {'-'*50} {'-'*30} {'-'*8} {'-'*8} {'-'*6}")
    print(f"  {'TOTAL':<50} {'':<30} {total_fp32:>8.2f} {total_quant:>8.2f} {overall:>5.1f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize FP32 model weights for boat inference"
    )
    parser.add_argument("input", help="Path to .safetensors file or model directory")
    parser.add_argument("--dtype", default="uint8",
                        choices=["uint8", "int8", "bits2", "bits1", "float4"],
                        help="Quantization target dtype (default: uint8)")
    parser.add_argument("--symmetric", action="store_true",
                        help="Use symmetric quantization (zp=128 for UINT8, zp=0 for INT8)")
    parser.add_argument("--per-channel", action="store_true",
                        help="Quantize each output channel independently")
    parser.add_argument("--channel-dim", type=int, default=0,
                        help="Channel dimension for per-channel quant (default: 0)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .safetensors path (default: <input>.quantized.safetensors)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just estimate compression, don't write output")
    parser.add_argument("--include", default=None,
                        help="Only quantize tensors matching this glob/substring pattern")
    parser.add_argument("--exclude", default=None,
                        help="Skip tensors matching this glob/substring pattern")
    args = parser.parse_args()

    # Determine output path
    input_path = args.input
    if args.output:
        out_path = args.output
    elif os.path.isfile(input_path):
        base = input_path.rsplit(".", 1)[0]
        out_path = f"{base}.q{args.dtype}.safetensors"
    else:
        out_path = os.path.join(input_path, f"model.q{args.dtype}.safetensors")

    print(f"[boat-quantize] Loading: {input_path}")

    # Load all tensors
    all_tensors = load_safetensors(input_path)
    fp32_tensors = {
        k: v for k, v in all_tensors.items()
        if np.issubdtype(v.dtype, np.floating)
    }

    if args.dry_run:
        print(f"\n  Found {len(fp32_tensors)} float tensors out of {len(all_tensors)} total")
        estimate_compression(fp32_tensors, args.dtype)
        return

    # Filter tensors
    target_names = list(fp32_tensors.keys())
    if args.include:
        target_names = [n for n in target_names if args.include in n]
    if args.exclude:
        target_names = [n for n in target_names if args.exclude not in n]

    print(f"  Quantizing {len(target_names)}/{len(fp32_tensors)} float tensors to {args.dtype} "
          f"{'(symmetric)' if args.symmetric else '(asymmetric)'} "
          f"{'(per-channel)' if args.per_channel else '(per-tensor)'}")

    # Build output tensors dict: keep non-float tensors as-is
    output_tensors = {k: v for k, v in all_tensors.items()
                      if k not in fp32_tensors}
    output_tensors.update({
        k: v for k, v in all_tensors.items()
        if k in fp32_tensors and k not in target_names
    })

    # Quantize target tensors
    total_fp32_bytes = 0
    total_quant_bytes = 0
    for name in target_names:
        arr = fp32_tensors[name]
        total_fp32_bytes += arr.nbytes

        qdata, scale, zp, st_dtype, actual_dtype = quantize_array(
            arr, args.dtype, args.symmetric, args.per_channel, args.channel_dim)

        # Store quantized tensor
        output_tensors[name] = qdata

        # Store scale and zero_point
        output_tensors[f"{name}.scale"] = np.asarray(scale, dtype=np.float32)
        output_tensors[f"{name}.zp"] = np.asarray(zp, dtype=np.int32)

        # Store quant-dtype marker for packed formats
        if args.dtype in ("bits2", "bits1", "float4"):
            dtype_code = {"bits2": 2, "bits1": 1, "float4": 4}
            output_tensors[f"{name}.qdtype"] = np.array(
                dtype_code.get(args.dtype, 0), dtype=np.uint8)

        total_quant_bytes += qdata.nbytes + 4 + 4  # data + scale + zp

    # Report compression
    ratio = total_fp32_bytes / total_quant_bytes if total_quant_bytes > 0 else 0
    print(f"  FP32 size:  {total_fp32_bytes / 1024**2:.2f} MB")
    print(f"  Quant size: {total_quant_bytes / 1024**2:.2f} MB")
    print(f"  Ratio:      {ratio:.1f}x")

    # Write
    safetensors_write(output_tensors, out_path)
    print(f"[boat-quantize] Done.")


if __name__ == "__main__":
    main()
