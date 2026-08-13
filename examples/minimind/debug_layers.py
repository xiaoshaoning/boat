"""Save per-layer hidden states for comparison with C."""
import torch, json, os, sys, struct
sys.path.insert(0, 'D:/github/minimind')
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)
ckp = 'D:/github/minimind/out/full_sft_768.pth'
state = torch.load(ckp, map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=True)
model = model.float().eval()

# Register hooks to capture hidden states after each MiniMindBlock
hidden_states = []
outputs = {}

def hook_fn(module, input, output):
    # output is a tuple (hidden_states, present_kv)
    if isinstance(output, tuple):
        hidden_states.append(output[0].detach().clone())

hooks = []
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.register_forward_hook(hook_fn))

# Test with 2 tokens: [1, 832]
tokens = [1, 832]
input_ids = torch.tensor([tokens], dtype=torch.long)
with torch.no_grad():
    out = model(input_ids)
    logits = out.logits[0, -1, :]

for h in hooks: h.remove()

# Save hidden states
os.makedirs('debug_data', exist_ok=True)
print(f"Saved {len(hidden_states)} hidden states from layers")
for i, hs in enumerate(hidden_states):
    arr = hs[0].numpy().astype('float32')
    with open(f'debug_data/layer{i}_hidden.bin', 'wb') as f:
        f.write(arr.tobytes())

# Also save input tokens and final logits
with open('debug_data/input.bin', 'wb') as f:
    f.write(struct.pack('i', len(tokens)))
    for t in tokens: f.write(struct.pack('i', t))

arr = logits.numpy().astype('float32')
with open('debug_data/logits.bin', 'wb') as f:
    f.write(arr.tobytes())

# Print first few values of each layer
for i, hs in enumerate(hidden_states):
    print(f"Layer {i} hidden[0, 0, :5]: {hs[0, 0, :5].tolist()}")
print(f"Final logits top-5: {torch.topk(logits, 5).indices.tolist()}")
print(f"Final logits top-5 values: {[f'{x:.4f}' for x in torch.topk(logits, 5).values.tolist()]}")
