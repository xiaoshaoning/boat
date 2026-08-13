"""Export MiniMind PyTorch weights to flat FP32 binary + JSON metadata."""
import torch, json, os, sys, struct

def export_minimind(pth_path, output_dir):
    state = torch.load(pth_path, map_location='cpu', weights_only=True)
    os.makedirs(output_dir, exist_ok=True)
    meta = {}
    offset = 0
    total = 0

    with open(os.path.join(output_dir, 'model.bin'), 'wb') as f:
        for name in sorted(state.keys()):
            t = state[name].float().contiguous()
            data = t.numpy().tobytes()
            meta[name] = {'offset': offset, 'shape': list(t.shape)}
            f.write(data)
            offset += len(data)
            total += 1

    with open(os.path.join(output_dir, 'model_meta.json'), 'w') as f:
        json.dump({name: meta[name] for name in sorted(meta.keys())}, f, indent=2)

    print(f"Exported {total} tensors ({offset:,} bytes total) to {output_dir}")

def verify_export(pth_path, output_dir):
    """Verify exported weights match original."""
    state = torch.load(pth_path, map_location='cpu', weights_only=True)
    with open(os.path.join(output_dir, 'model_meta.json')) as f:
        meta = json.load(f)
    with open(os.path.join(output_dir, 'model.bin'), 'rb') as f:
        data = f.read()

    max_err = 0.0
    for name, info in meta.items():
        if name not in state:
            continue
        shape = info['shape']
        n_elems = 1
        for s in shape: n_elems *= s
        exported = torch.frombuffer(bytearray(data[info['offset']:info['offset'] + n_elems * 4]),
                                     dtype=torch.float32).reshape(shape)
        original = state[name].float()
        err = (exported - original).abs().max().item()
        if err > max_err:
            max_err = err
        if err > 1e-5:
            print(f"  MISMATCH {name}: max_err={err:.2e}")
    print(f"Verification complete. Max error: {max_err:.2e}")
    return max_err < 1e-5

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python export_weights.py <model.pth> <output_dir>")
        print("       python export_weights.py --verify <model.pth> <output_dir>")
        sys.exit(1)

    if sys.argv[1] == '--verify':
        verify_export(sys.argv[2], sys.argv[3])
    else:
        export_minimind(sys.argv[1], sys.argv[2])
