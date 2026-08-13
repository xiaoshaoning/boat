"""Encode prompt with Python tokenizer, run C model, decode output."""
import subprocess, struct, sys, os

sys.path.insert(0, 'D:/github/minimind')
from transformers import AutoTokenizer

prompt = sys.argv[1] if len(sys.argv) > 1 else "你好"
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 32

tok = AutoTokenizer.from_pretrained('D:/github/minimind/model')

# Format chat prompt
conversation = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
tokens = tok.encode(text)

print(f"Prompt: {prompt}")
print(f"Chat template: {repr(text)}")
print(f"Token IDs ({len(tokens)}): {tokens}")

# Save tokens for C
with open('gen_input.bin', 'wb') as f:
    f.write(struct.pack('i', len(tokens)))
    for t in tokens: f.write(struct.pack('i', t))
    f.write(struct.pack('i', max_tokens))

# Run C model
exe = './minimind_gen.exe'
if os.path.exists(exe):
    result = subprocess.run([exe, './weights'], capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    if result.stderr: print("STDERR:", result.stderr)

    # Read generated tokens
    with open('gen_output.bin', 'rb') as f:
        n_out = struct.unpack('i', f.read(4))[0]
        out_tokens = []
        for _ in range(n_out):
            out_tokens.append(struct.unpack('i', f.read(4))[0])

    print(f"Generated {n_out} tokens: {out_tokens}")
    print(f"Decoded: {tok.decode(out_tokens)}")
else:
    print("C binary not found - compile first")
