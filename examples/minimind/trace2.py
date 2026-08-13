"""Write BPE trace to file (avoid terminal encoding issues)."""
import sys
sys.path.insert(0, 'D:/github/minimind')
from tokenizers import Tokenizer

out = open('trace_output.txt', 'w', encoding='utf-8')

tok = Tokenizer.from_file('D:/github/minimind/model/tokenizer.json')

# What is token 1968?
t = tok.id_to_token(1968)
out.write(f"Token 1968: {repr(t)}\n")
out.write(f"Token 1968 hex: {t.encode('utf-8').hex()}\n\n")

# Byte encoder
bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('\xa1'), ord('\xac')+1)) + list(range(ord('\xae'), ord('\xff')+1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
byte_encoder = dict(zip(bs, [chr(c) for c in cs]))

text = '你好'  # 你好
text_bytes = text.encode('utf-8')
out.write(f"Input: {text}\n")
out.write(f"Bytes: {list(text_bytes)}\n")

unicode_chars = ''.join(byte_encoder[b] for b in text_bytes)
out.write(f"Unicode chars ({len(unicode_chars)}): {[hex(ord(c)) for c in unicode_chars]}\n\n")

# Get merges
from tokenizers.models import BPE
model = tok._tokenizer.model  # This is the BPE model
state = model.__getstate__()
merges = state['merges']
out.write(f"Total merges: {len(merges)}\n")
out.write(f"First 5 merges: {[(a, b) for a,b in merges[:5]]}\n\n")

# Apply BPE merges
chars = list(unicode_chars)
out.write(f"Starting chars ({len(chars)}):\n")
for i, c in enumerate(chars):
    out.write(f"  [{i}] U+{ord(c):04X}\n")

merge_count = 0
for mi, (a, b) in enumerate(merges):
    new_chars = []
    j = 0
    while j < len(chars):
        if j < len(chars) - 1 and chars[j] == a and chars[j+1] == b:
            merged = a + b
            new_chars.append(merged)
            if mi < 20:
                out.write(f"  merge #{mi}: U+{ord(a):04X}+U+{ord(b):04X} => U+{' '.join(f'{ord(x):04X}' for x in merged)}\n")
            j += 2
        else:
            new_chars.append(chars[j])
            j += 1
    if new_chars != chars:
        merge_count += 1
    chars = new_chars

out.write(f"\nTotal merges applied: {merge_count}\n")
out.write(f"Final tokens ({len(chars)}):\n")
for c in chars:
    tid = tok.token_to_id(c)
    out.write(f"  {[hex(ord(x)) for x in c]} -> id={tid}\n")

out.write(f"\nExpected: {tok.encode(text).ids}\n")

# Also show: what are tokens 164, 195, 154, 163?
# (these are the tokens my C tokenizer produced for 你好)
out.write("\n=== Debug: C tokenizer output ===\n")
for tid in [164, 195, 154, 163]:
    t = tok.id_to_token(tid)
    out.write(f"  Token {tid}: {repr(t)} (hex: {t.encode('utf-8').hex()})\n")

out.close()
print("Output written to trace_output.txt")
