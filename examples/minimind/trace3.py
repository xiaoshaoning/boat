"""Quick compare: C vs Python byte mapping for '你好'."""
import json

with open('D:/github/minimind/model/tokenizer.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
merges = data['model']['merges']
vocab = data['model']['vocab']

# Python GPT-2 mapping
bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('\xa1'), ord('\xac')+1)) + list(range(ord('\xae'), ord('\xff')+1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
py_encoder = dict(zip(bs, [chr(c) for c in cs]))

# My C mapping
c_encoder = {}
c_n = 0
for b in range(256):
    if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255):
        c_encoder[b] = b
    else:
        c_encoder[b] = 256 + c_n
        c_n += 1

# Encode "你好" with both
text = '你好'
text_bytes = list(text.encode('utf-8'))
py_unicode = [py_encoder[b] for b in text_bytes]
c_unicode = [chr(c_encoder[b]) for b in text_bytes]

print(f"Input bytes: {[hex(b) for b in text_bytes]}")
print(f"Python unicode chars: {[hex(ord(c)) for c in py_unicode]}")
print(f"C      unicode chars: {[hex(ord(c)) for c in c_unicode]}")
print(f"Match: {py_unicode == c_unicode}")

if py_unicode != c_unicode:
    for i, (p, c) in enumerate(zip(py_unicode, c_unicode)):
        if p != c:
            print(f"  DIFF at pos {i}: Python U+{ord(p):04X}, C U+{ord(c):04X}")

# Apply BPE on Python's mapping
chars = py_unicode[:]
for mi, (a, b) in enumerate(merges):
    new_chars = []
    j = 0
    while j < len(chars):
        if j < len(chars) - 1 and chars[j] == a and chars[j+1] == b:
            new_chars.append(a + b)
            j += 2
        else:
            new_chars.append(chars[j])
            j += 1
    chars = new_chars

print(f"\nPython BPE final ({len(chars)} tokens):")
for ch in chars:
    tid = vocab.get(ch, '?')
    print(f"  {[hex(ord(x)) for x in ch]} -> id={tid}")

# Now with C's mapping
chars2 = c_unicode[:]
for mi, (a, b) in enumerate(merges):
    new_chars = []
    j = 0
    while j < len(chars2):
        if j < len(chars2) - 1 and chars2[j] == a and chars2[j+1] == b:
            new_chars.append(a + b)
            j += 2
        else:
            new_chars.append(chars2[j])
            j += 1
    chars2 = new_chars

print(f"\nC BPE final ({len(chars2)} tokens):")
for ch in chars2:
    tid = vocab.get(ch, '?')
    print(f"  {[hex(ord(x)) for x in ch]} -> id={tid}")
