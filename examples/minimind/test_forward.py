"""Compare C forward pass against Python for correctness."""
import torch, json, sys, os, struct, subprocess

sys.path.insert(0, 'D:/github/minimind')
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

# Load Python model
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)
ckp = 'D:/github/minimind/out/full_sft_768.pth'
state = torch.load(ckp, map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=True)
model = model.float().eval()

# Test inputs: simple token sequences
test_inputs = [
    [1],  # just <|im_start|>
    [1, 832],  # <|im_start|> + first token of "user\n"
    [1, 832, 311, 234, 1968, 2, 234, 1, 1388, 570, 811, 234, 25, 234, 234, 26, 234, 234],
    # Above is the real chat template for "你好"
]

print("=" * 60)
print("Python Model Forward Pass Tests")
print("=" * 60)

for tokens in test_inputs:
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0, -1, :]  # last position

    # Save for C comparison
    test_dir = 'forward_test'
    os.makedirs(test_dir, exist_ok=True)
    name = f"test_{len(tokens)}tokens"

    # Save input tokens
    with open(os.path.join(test_dir, f'{name}_input.bin'), 'wb') as f:
        f.write(struct.pack('i', len(tokens)))
        for t in tokens:
            f.write(struct.pack('i', t))

    # Save expected logits (just top-10 and bottom-10 for quick check)
    with open(os.path.join(test_dir, f'{name}_logits.bin'), 'wb') as f:
        arr = logits.numpy().astype('float32')
        f.write(arr.tobytes())

    top5 = torch.topk(logits, 5)
    print(f"\n{len(tokens)} tokens: {tokens[:5]}...")
    print(f"  Top-5 token IDs: {top5.indices.tolist()}")
    print(f"  Top-5 logits:     {[f'{x:.4f}' for x in top5.values.tolist()]}")

    # Print the token text for top-5
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('D:/github/minimind/model')
    for i, (tid, val) in enumerate(zip(top5.indices.tolist(), top5.values.tolist())):
        token_str = tok.decode([tid])
        print(f"    #{i+1}: id={tid} '{token_str}' = {val:.4f}")

print("\nTest data saved to forward_test/ for C comparison.")
