"""Run Python GLM-OCR inference with correct chat template."""
import torch
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

# Load image
image = Image.open("D:/huggingface/GLM-OCR/imgs/poem.jpg")
print(f"Original image: {image.size}")

# Build messages with chat template
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Please read the text in this image"},
        ],
    },
]

# Apply chat template and process
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Chat template output: {text[:200]}...")

inputs = processor(text=[text], images=[image], return_tensors="pt")
print(f"Input keys: {inputs.keys()}")
print(f"input_ids shape: {inputs['input_ids'].shape}")
print(f"pixel_values shape: {inputs.get('pixel_values', 'N/A')}")
print(f"image_grid_thw: {inputs.get('image_grid_thw', 'N/A')}")

# Run generation
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
output_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nOutput: {output_text}")

# Now test with Chinese prompt
messages_zh = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "这是中文古籍的图片，请按照从上到下，从右到左的次序识别其中的汉语繁体文字"},
        ],
    },
]

text_zh = processor.apply_chat_template(messages_zh, tokenize=False, add_generation_prompt=True)
inputs_zh = processor(text=[text_zh], images=[image], return_tensors="pt")

with torch.no_grad():
    outputs_zh = model.generate(
        **inputs_zh,
        max_new_tokens=200,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
output_text_zh = processor.tokenizer.decode(outputs_zh[0], skip_special_tokens=True)
print(f"\nChinese output: {output_text_zh}")

print("\nDone!")
