"""Trace BPE encoding of text, write to file."""
import json, sys
sys.path.insert(0, 'D:/github/minimind')

out = open('trace_bpe_output.txt', 'w', encoding='utf-8')

with open('D:/github/minimind/model/tokenizer.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
merges = data['model']['merges']
vocab = data['model']['vocab']

# GPT-2 byte-to-unicode mapping
bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('\xa1'), ord('\xac')+1)) + list(range(ord('\xae'), ord('\xff')+1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
byte_encoder = dict(zip(bs, [chr(c) for c in cs]))

# Test text
text = '你好'
text_bytes = text.encode('utf-8')
unicode_chars = [byte_encoder[b] for b in text_bytes]

out.write(f"Input: {text}\n")
out.write(f"Bytes: {[hex(b) for b in text_bytes]}\n")
out.write(f"Unicode chars: {[hex(ord(c)) for c in unicode_chars]}\n")
out.write(f"Num chars: {len(unicode_chars)}\n\n")

# Apply BPE step by step
chars = unicode_chars[:]
step = 0
while len(chars) > 1:
    best_rank = len(merges)
    best_i = -1
    for i in range(len(chars) - 1):
        for rank, (a, b) in enumerate(merges):
            if chars[i] == a and chars[i+1] == b:
                if rank < best_rank:
                    best_rank = rank
                    best_i = i
                break
    if best_i == -1:
        out.write(f"Step {step}: No more merges found\n")
        break
    a, b = merges[best_rank]
    merged = a + b
    out.write(f"Step {step}: merge[{best_rank}] ")
    out.write(f"a={[hex(ord(x)) for x in a]} b={[hex(ord(x)) for x in b]}")
    out.write(f" -> merged={[hex(ord(x)) for x in merged]}\n")
    chars[best_i] = merged
    chars.pop(best_i + 1)
    step += 1

out.write(f"\nFinal tokens ({len(chars)}):\n")
from tokenizers import Tokenizer
tok = Tokenizer.from_file('D:/github/minimind/model/tokenizer.json')
for c in chars:
    tid = tok.token_to_id(c)
    out.write(f"  {[hex(ord(x)) for x in c]} -> id={tid}\n")

out.write(f"\nExpected from tokenizer: {tok.encode(text).ids}\n")
out.close()
print("Output written to trace_bpe_output.txt")
