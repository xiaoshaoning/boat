"""Verify position ordering in CogViT."""
import torch
from PIL import Image
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

device = "cpu"
dtype = torch.bfloat16

model = GlmOcrForConditionalGeneration.from_pretrained(
    "D:/huggingface/GLM-OCR", torch_dtype=dtype, device_map=device,
)
model.eval()

processor = AutoProcessor.from_pretrained("D:/huggingface/GLM-OCR", torch_dtype=dtype)

image = Image.open("D:/huggingface/GLM-OCR/imgs/poem.jpg")

# Process and get vision model
vision = model.model.visual

# Get grid_thw
text = "test"
inputs = processor(text=[text], images=[image], return_tensors="pt")
grid_thw = inputs["image_grid_thw"]
print(f"grid_thw: {grid_thw}")  # [1, 76, 58]
h, w = grid_thw[0, 1].item(), grid_thw[0, 2].item()  # h=76, w=58

print(f"\nGrid: {h}x{w} = {h*w} patches")

# ===== Check rot_pos_emb position ordering =====
hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)

print(f"\nOriginal grid positions (first 16):")
for i in range(16):
    row = i // w
    col = i % w
    print(f"  idx {i}: (row={row}, col={col}) hpos={hpos_ids[row, col].item()} wpos={wpos_ids[row, col].item()}")

# Apply rot_pos_emb transformation
h_reshaped = hpos_ids.reshape(h//2, 2, w//2, 2).permute(0, 2, 1, 3).flatten()
w_reshaped = wpos_ids.reshape(h//2, 2, w//2, 2).permute(0, 2, 1, 3).flatten()

print(f"\nAfter rot_pos_emb reordering (first 16):")
for i in range(16):
    orig_row = i // w
    orig_col = i % w
    print(f"  pos_emb_idx {i}: hpos={h_reshaped[i].item()}, wpos={w_reshaped[i].item()} -> grid ({h_reshaped[i].item()},{w_reshaped[i].item()})")
    print(f"    hidden_states[{i}] = patch at (row={orig_row}, col={orig_col})")
    match = "MATCH" if (h_reshaped[i].item() == orig_row and w_reshaped[i].item() == orig_col) else "MISMATCH"
    print(f"    -> {match}")

# Check first 2x2 block
print(f"\n--- First 2x2 block analysis ---")
print(f"Block (0,0) covers original grid positions:")
for by in range(2):
    for bx in range(2):
        r, c = by, bx
        print(f"  pos ({r},{c}): hpos_orig={hpos_ids[r,c].item()}, wpos_orig={wpos_ids[r,c].item()}")
        emb_idx = by * 2 + bx  # within-block flatten order
        print(f"    pos_emb_idx={emb_idx}: hpos_reord={h_reshaped[emb_idx].item()}, wpos_reord={w_reshaped[emb_idx].item()}")

# Also check the downsample reshape
print(f"\n--- Downsample: view(-1, 2, 2, 1024) grouping ---")
print(f"Groups of 4 consecutive patches in hidden_states:")
for g in range(4):
    start = g * 4
    print(f"  Group {g}: patches at indices [{start},{start+1},{start+2},{start+3}]")
    for i in range(4):
        idx = start + i
        r = idx // w
        c = idx % w
        print(f"    [{idx}] = grid pos ({r},{c})")

print(f"\n=== KEY INSIGHT ===")
print(f"After downsample view(-1, 2, 2, 1024), each '2x2 block' in the 4D tensor")
print(f"consists of 4 CONSECUTIVE patches from hidden_states.")
print(f"For the first group [0,1,2,3]: these are patches (0,0),(0,1),(0,2),(0,3)")
print(f"which spans 4 columns in row 0, NOT a square 2x2 block!")
