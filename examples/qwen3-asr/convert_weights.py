"""convert_weights.py — Qwen3-ASR safetensors → .npz weight dump"""
import struct, json, sys, os, time
import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
HF_DIR = "D:\\huggingface\\Qwen3-ASR-0.6B"
SAFETENSORS_PATH = os.path.join(HF_DIR, "model.safetensors")
OUTPUT_PATH = os.path.join(MODEL_DIR, "qwen3_asr_weights.npz")

def bf16_to_f32_raw(raw_bytes):
    """Convert BF16 bytes to float32 numpy array."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.uint32) << 16
    return arr.view(np.float32)

def load_safetensors_header(path):
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header

def main():
    print(f"Loading weights from: {SAFETENSORS_PATH}")
    t0 = time.time()
    header = load_safetensors_header(SAFETENSORS_PATH)
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    print(f"Header parsed: {len(tensors)} tensors in {time.time()-t0:.1f}s")

    weight_groups = {}
    with open(SAFETENSORS_PATH, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        f.read(header_len)
        data_start = 8 + header_len

        for name, info in tensors.items():
            dtype = info['dtype']
            shape = info['shape']
            offset, end = info['data_offsets']
            f.seek(data_start + offset)
            raw = f.read(end - offset)

            if dtype == 'BF16':
                arr = bf16_to_f32_raw(raw).reshape(shape)
            elif dtype == 'F32':
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            else:
                print(f"  SKIP {name}: unsupported dtype {dtype}")
                continue

            short = name.replace('thinker.', '', 1)
            weight_groups[short] = arr
            sys.stdout.write(f"  {short}: {list(arr.shape)} {arr.nbytes/1024/1024:.0f}MB\n")
            sys.stdout.flush()

    np.savez_compressed(OUTPUT_PATH, **weight_groups)
    total_mb = sum(w.nbytes for w in weight_groups.values()) / 1024 / 1024
    out_size = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\nDone: {len(weight_groups)} tensors, {total_mb:.0f}MB FP32 → {out_size:.0f}MB .npz")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
