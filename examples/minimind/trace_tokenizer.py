"""Trace BPE tokenizer encoding step-by-step for C parity."""
import sys, json
sys.path.insert(0, 'D:/github/minimind')
from tokenizers import Tokenizer

tok = Tokenizer.from_file('D:/github/minimind/model/tokenizer.json')

# Test inputs
tests = [
    ("你好", "Chinese hello"),
    ("你好", "Chinese hello raw, no template"),
    ("user\n", "ASCII with newline"),
    ("\n", "Just newline"),
]

# For now, focus on just "你好" to understand byte-to-unicode + BPE
text = "你好"
print(f"Input: {repr(text)}")
print(f"Input bytes: {text.encode('utf-8').hex()}")

# Use tokenizer's internal encode
encoded = tok.encode(text)
print(f"\nEncoded tokens: {encoded.ids}")
print(f"Encoded tokens text: {[tok.id_to_token(t) for t in encoded.ids]}")

# Now trace the byte-level pre-tokenizer
# GPT-2 byte-to-unicode mapping
# This is defined in tokenizers library: byte_level.py
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

# Apply byte-to-unicode to our text
text_bytes = text.encode('utf-8')
print(f"\nText bytes: {list(text_bytes)}")

unicode_chars = ''.join(byte_encoder[b] for b in text_bytes)
print(f"After byte-to-unicode: {repr(unicode_chars)}")
print(f"Unicode chars as codepoints: {[ord(c) for c in unicode_chars]}")

# Now split into individual characters for BPE
# The ByteLevel pre-tokenizer uses a regex to split: 's|.|\n|\r|\r\n|\s+'
# For simple ASCII: each char is separate
# For Chinese (multi-byte UTF-8): each byte maps to a unicode char
# After byte-to-unicode, each unicode char represents one byte

bpe_chars = list(unicode_chars)
print(f"\nBPE input chars ({len(bpe_chars)}): {[repr(c) for c in bpe_chars]}")

# Get merges from tokenizer
merges_raw = tok._tokenizer.model.merges
print(f"\nTotal merges: {len(merges_raw)}")
print(f"First 10 merges: {merges_raw[:10]}")

# Apply BPE merges to our chars
words = [bpe_chars]
for i, merge in enumerate(merges_raw):
    a, b = merge
    for w_idx, word in enumerate(words):
        new_word = []
        j = 0
        while j < len(word):
            if j < len(word) - 1 and word[j] == a and word[j+1] == b:
                new_word.append(a + b)  # merged token
                j += 2
            else:
                new_word.append(word[j])
                j += 1
        words[w_idx] = new_word

final_tokens = []
for word in words:
    final_tokens.extend(word)

print(f"After BPE ({len(final_tokens)} tokens): {[repr(t) for t in final_tokens]}")

# Map to token IDs
ids = []
for t in final_tokens:
    tid = tok.token_to_id(t)
    ids.append(tid)
    print(f"  {repr(t)} -> id={tid}" if tid else f"  {repr(t)} -> UNKNOWN")

print(f"\nFinal IDs: {ids}")
print(f"Expected:  {encoded.ids}")

# Also show what the tokenizer's pre-tokenizer split produces
print("\n--- Using tokenizer internals ---")
pre_tok = tok.pre_tokenizer
print(f"Pre-tokenizer type: {type(pre_tok).__name__}")
