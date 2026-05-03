"""Trace CogViT vision encoder intermediate values."""
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

device = "cpu"
dtype = torch.bfloat16

print("Loading model and processor...")
model = GlmOcrForConditionalGeneration.from_pretrained(
    "D:/huggingface/GLM-OCR",
    torch_dtype=dtype,
    device_map=device,
)
model.eval()

processor = AutoProcessor.from_pretrained(
    "D:/huggingface/GLM-OCR",
    torch_dtype=dtype,
)

# Load and process image
image = Image.open("D:/huggingface/GLM-OCR/imgs/poem.jpg")
print(f"Original image: {image.size}")

# Process using processor to get pixel_values and grid_thw
# We use a simple text prompt
text = "Please read the text in this image"
inputs = processor(text=[text], images=[image], return_tensors="pt")

pixel_values = inputs["pixel_values"].to(dtype=torch.float32)
image_grid_thw = inputs["image_grid_thw"]
print(f"pixel_values shape: {pixel_values.shape}")
print(f"image_grid_thw: {image_grid_thw}")

# Get the vision model
vision = model.model.visual
print(f"\nVision config: hidden={vision.config.hidden_size}, heads={vision.config.num_heads}, depth={vision.config.depth}")
head_dim = vision.config.hidden_size // vision.config.num_heads
print(f"head_dim: {head_dim}")
print(f"spatial_merge_size: {vision.spatial_merge_size}")
print(f"patch_size: {vision.patch_size}")

# ========== Step 1: Patch Embed ==========
print("\n=== PATCH EMBED ===")
with torch.no_grad():
    hidden = vision.patch_embed(pixel_values)
    print(f"patch_embed out shape: {hidden.shape}")
    norm_val = hidden.norm().item()
    per_token_norm = (hidden.norm(dim=-1).mean()).item()
    print(f"patch_embed norm: {norm_val:.4f} (per-token: {per_token_norm:.4f})")
    print(f"patch_embed[0]: {hidden[0, :4].tolist()}")
    print(f"hidden stats: mean={hidden.mean().item():.6f}, std={hidden.std().item():.6f}")

# ========== Step 1b: Rotary Position Embed ==========
print("\n=== ROTARY POSITION EMBED ===")
rotary_pos_emb, image_type_ids = vision.rot_pos_emb(image_grid_thw)
print(f"rotary_pos_emb shape: {rotary_pos_emb.shape}")
emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
print(f"emb shape: {emb.shape}")
print(f"emb[0]: {emb[0, :8].tolist()}")
position_embeddings = (emb.cos(), emb.sin())
print(f"cos shape: {position_embeddings[0].shape}, sin shape: {position_embeddings[1].shape}")

# ========== Step 2: Transformer Blocks ==========
print("\n=== TRANSFORMER BLOCKS ===")
# Compute cu_seqlens (for single image)
cu_seqlens = (image_grid_thw[:, 1] * image_grid_thw[:, 2]).cumsum(dim=0, dtype=torch.int32)
cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
print(f"cu_seqlens: {cu_seqlens}")

hidden_before = hidden.clone()
for i, blk in enumerate(vision.blocks):
    with torch.no_grad():
        hidden = blk(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
    if i % 4 == 0 or i == vision.config.depth - 1:
        nrm = hidden.norm().item()
        ptn = hidden.norm(dim=-1).mean().item()
        print(f"  block {i}: norm={nrm:.4f}, per-token={ptn:.4f}, mean={hidden.mean().item():.6f}")

# ========== Step 3: Post-LayerNorm ==========
print("\n=== POST LAYERNORM ===")
with torch.no_grad():
    hidden = vision.post_layernorm(hidden)
    nrm = hidden.norm().item()
    ptn = hidden.norm(dim=-1).mean().item()
    print(f"post_layernorm: norm={nrm:.4f}, per-token={ptn:.4f}")
    print(f"post_layernorm[0]: {hidden[0, :4].tolist()}")
    print(f"stats: mean={hidden.mean().item():.6f}, std={hidden.std().item():.6f}")

# ========== Step 4: Downsample ==========
print("\n=== DOWNSAMPLE ===")
with torch.no_grad():
    # reshape: [num_patches, dim] → [num_merged, 2, 2, dim] → permute → conv → view
    n_patches = hidden.shape[0]
    ds_hidden = hidden.view(-1, vision.spatial_merge_size, vision.spatial_merge_size, hidden.shape[-1])
    ds_hidden = ds_hidden.permute(0, 3, 1, 2)
    print(f"  before downsample: {ds_hidden.shape}")
    ds_hidden = vision.downsample(ds_hidden).view(-1, vision.config.out_hidden_size)
    print(f"  after downsample: {ds_hidden.shape}")
    nrm = ds_hidden.norm().item()
    ptn = ds_hidden.norm(dim=-1).mean().item()
    print(f"downsample: norm={nrm:.4f}, per-token={ptn:.4f}")
    print(f"downsample[0]: {ds_hidden[0, :4].tolist()}")

# ========== Step 5: Merger ==========
print("\n=== MERGER ===")
with torch.no_grad():
    merged = vision.merger(ds_hidden)
    print(f"merger out shape: {merged.shape}")
    nrm = merged.norm().item()
    ptn = merged.norm(dim=-1).mean().item()
    print(f"merger: norm={nrm:.4f}, per-token={ptn:.4f}")
    print(f"merger[0]: {merged[0, :4].tolist()}")
    print(f"stats: mean={merged.mean().item():.6f}, std={merged.std().item():.6f}")

print("\nDone!")
