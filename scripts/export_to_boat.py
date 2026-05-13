#!/usr/bin/env python3
"""
export_to_boat.py - Convert HuggingFace models to boat-compatible safetensors format.

Usage:
    python scripts/export_to_boat.py <model_dir> [--dtype f32] [--out <output_path>]

Converts pytorch_model.bin (pickle) to model.safetensors, and optionally converts
dtypes (e.g. BF16/FP16 -> F32) so boat can load them without runtime conversion.

Output is a .safetensors file that boat loads via examples/common/safetensors.c:
    safetensors_open() -> safetensors_find() -> safetensors_load_tensor()

Examples:
    # Convert a .bin model to .safetensors (in-place)
    python scripts/export_to_boat.py D:/huggingface/nanochat-d34-sft-hf

    # Convert and change dtype to F32, save to custom path
    python scripts/export_to_boat.py D:/huggingface/GLM-OCR --dtype f32 --out GLM-OCR-f32.safetensors

    # Just inspect tensor names (dry run)
    python scripts/export_to_boat.py D:/huggingface/bert-base-uncased --dry-run
"""

import argparse
import json
import os
import sys
import struct


# ---------------------------------------------------------------------------
# Safetensors writer (pure Python, no dependencies)
# ---------------------------------------------------------------------------

def _dtype_to_safetensors_str(dt: str) -> str:
    """Map numpy-style dtype names to safetensors dtype strings."""
    mapping = {
        "float64": "F64",
        "float32": "F32",
        "float16": "F16",
        "bfloat16": "BF16",
        "int64": "I64",
        "int32": "I32",
        "int16": "I16",
        "int8": "I8",
        "uint64": "U64",
        "uint32": "U32",
        "uint16": "U16",
        "uint8": "U8",
        "bool": "BOOL",
    }
    return mapping.get(dt, dt.upper())


def safetensors_write(tensors: dict, path: str):
    """Write a dictionary of {name: numpy_array} to a safetensors file.

    This is a minimal reimplementation to avoid the `safetensors` PyPI dependency.
    """
    # Build header
    header = {}
    data_chunks = []
    offset = 0

    for name, arr in tensors.items():
        dtype_str = _dtype_to_safetensors_str(str(arr.dtype))
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
    # Pad header to 8-byte alignment
    padding = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * padding

    # Write file: header_len(8 bytes LE) + header_json + data
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk in data_chunks:
            f.write(chunk)

    print(f"  Wrote {len(tensors)} tensors ({offset / 1024**3:.2f} GB) to {path}")


# ---------------------------------------------------------------------------
# Weight loader: tries both PyTorch .bin and existing .safetensors
# ---------------------------------------------------------------------------

def load_weights(model_dir: str) -> dict:
    import numpy as np
    """Load all tensors from a HuggingFace model directory.

    Returns {tensor_name: numpy_array}. Supports:
    - model.safetensors (and sharded: model-00001-of-NNNNN.safetensors)
    - pytorch_model.bin (and sharded: pytorch_model-00001-of-NNNNN.bin)
    """
    tensors = {}

    # Check for safetensors first (sharded or single)
    safetensors_files = sorted([
        f for f in os.listdir(model_dir)
        if f.endswith(".safetensors")
    ])
    if safetensors_files:
        # Try safetensors library with PyTorch (handles bfloat16) then convert to numpy
        try:
            import torch
            from safetensors import safe_open
            for fname in safetensors_files:
                fpath = os.path.join(model_dir, fname)
                print(f"  Loading {fname} ...")
                with safe_open(fpath, framework="pt") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key).cpu().float().numpy()
        except ImportError:
            # Fallback: try numpy framework (older safetensors version)
            try:
                from safetensors import safe_open
                for fname in safetensors_files:
                    fpath = os.path.join(model_dir, fname)
                    print(f"  Loading {fname} (via safe_open np) ...")
                    with safe_open(fpath, framework="np") as f:
                        for key in f.keys():
                            arr = f.get_tensor(key)
                            if hasattr(arr, 'dtype') and str(arr.dtype) == 'bfloat16':
                                arr = arr.astype(np.float32)
                            tensors[key] = arr
            except (ImportError, TypeError, RuntimeError):
                # Ultimate fallback: use our own safetensors reader
                for fname in safetensors_files:
                    fpath = os.path.join(model_dir, fname)
                    print(f"  Loading {fname} (via built-in reader) ...")
                    _load_safetensors_file(fpath, tensors)
    else:
        # Fall back to PyTorch .bin
        try:
            import torch
        except ImportError:
            print("ERROR: PyTorch is required to load .bin files.")
            print("Install it with: pip install torch")
            sys.exit(1)

        bin_files = sorted([
            f for f in os.listdir(model_dir)
            if f.endswith(".bin") and not f.startswith(".") and not f.endswith(".bin.un~")
        ])
        if not bin_files:
            bin_files = sorted([
                f for f in os.listdir(model_dir)
                if f.endswith(".bin")
            ])

        if not bin_files:
            print(f"ERROR: No weight files found in {model_dir}")
            print("Looked for: model.safetensors, *.safetensors, pytorch_model.bin, *.bin")
            sys.exit(1)

        for fname in bin_files:
            fpath = os.path.join(model_dir, fname)
            print(f"  Loading {fname} ...")
            state = torch.load(fpath, map_location="cpu", weights_only=True)
            for key, val in state.items():
                tensors[key] = val.numpy()

    print(f"  Loaded {len(tensors)} tensors total")
    return tensors


def _load_safetensors_file(fpath: str, out: dict):
    """Minimal safetensors reader (no safetensors PyPI package needed)."""
    import struct
    import json
    import numpy as np

    with open(fpath, "rb") as f:
        data = f.read()

    # Read header length (8 bytes LE)
    header_len = struct.unpack("<Q", data[:8])[0]
    header_data = data[8:8 + header_len]
    payload = data[8 + header_len:]

    header = json.loads(header_data.decode("utf-8"))

    dtype_map = {
        "F32": np.float32, "F64": np.float64, "F16": np.float16,
        "BF16": np.dtype("bfloat16") if hasattr(np, "dtype") else np.uint16,
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
            raise ValueError(f"Unknown dtype {dtype_str} for tensor {name}")
        raw = payload[start:end]
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
        out[name] = arr


# ---------------------------------------------------------------------------
# Dtype conversion
# ---------------------------------------------------------------------------

def convert_dtype(tensors: dict, target: str) -> dict:
    """Convert all tensors to the target dtype.

    Args:
        tensors: {name: numpy_array}
        target: "f32", "f16", "bf16"

    Returns:
        New dict with converted tensors.
    """
    import numpy as np

    target_map = {
        "f32": np.float32,
        "f16": np.float16,
        "bf16": np.dtype("bfloat16") if hasattr(np, "bfloat16") else np.uint16,
    }

    if target not in target_map:
        print(f"  Unknown target dtype '{target}', skipping conversion")
        return tensors

    target_dtype = target_map[target]
    converted = {}
    for name, arr in tensors.items():
        if arr.dtype != target_dtype:
            # Only convert float types (not int indices etc.)
            if np.issubdtype(arr.dtype, np.floating) or arr.dtype == np.dtype("bfloat16"):
                arr = arr.astype(target_dtype)
        converted[name] = arr
    return converted


def print_weight_c_table(tensors: dict, model_name: str = "model"):
    """Print tensor names and shapes as C comments, helpful for writing C loaders."""
    print(f"\n  // Tensor map for {model_name}")
    for name, arr in tensors.items():
        shape_str = "x".join(str(s) for s in arr.shape)
        print(f"  // {name}: [{shape_str}], dtype={arr.dtype}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to boat-compatible safetensors format"
    )
    parser.add_argument("model_dir", help="Path to HuggingFace model directory")
    parser.add_argument("--dtype", default=None,
                        help="Convert all float tensors to this dtype: f32, f16, bf16 (default: leave as-is)")
    parser.add_argument("--out", "-o", default=None,
                        help="Output path for the .safetensors file (default: <model_dir>/model.safetensors)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just list tensor names, don't write output")
    parser.add_argument("--print-c", action="store_true",
                        help="Print tensor info as C comments (useful when writing C loaders)")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        print(f"ERROR: {model_dir} is not a directory")
        sys.exit(1)

    print(f"[boat-export] Loading model from: {model_dir}")

    # Load weights
    tensors = load_weights(model_dir)

    if args.print_c or args.dry_run:
        print_weight_c_table(tensors, os.path.basename(model_dir.rstrip("/\\")))

    if args.dry_run:
        return

    # Dtype conversion
    if args.dtype:
        import numpy as np
        # Check current dtypes
        dtypes = set(str(t.dtype) for t in tensors.values())
        print(f"  Source dtypes: {', '.join(sorted(dtypes))}")
        print(f"  Converting to {args.dtype} ...")
        tensors = convert_dtype(tensors, args.dtype)

    # Determine output path
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(model_dir, "model.safetensors")

    # Check if output would overwrite a non-.bin file and warn
    if os.path.exists(out_path):
        print(f"  WARNING: {out_path} already exists, will overwrite")

    # Write safetensors
    safetensors_write(tensors, out_path)

    # Also write a .info file with tensor names for reference
    info_path = out_path.rsplit(".", 1)[0] + ".info.txt"
    with open(info_path, "w") as f:
        for name, arr in tensors.items():
            shape_str = "x".join(str(s) for s in arr.shape)
            f.write(f"{name}\t[{shape_str}]\t{arr.dtype}\n")
    print(f"  Wrote tensor info to {info_path}")

    print(f"[boat-export] Done.")


if __name__ == "__main__":
    main()
