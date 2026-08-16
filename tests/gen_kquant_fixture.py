"""Generate K-quant test fixtures: raw quantized bytes + reference dequantized
values, produced by the authoritative `gguf` (llama.cpp) dequantizer.

Deterministic (seeded RNG); the output tests/unit/kquant_fixture.bin is
committed so fresh checkouts can configure and run the test suite without the
gguf package (CI does not install it)."""
import os
import numpy as np
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize, quantize

TYPES = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]
N_BLOCKS = 3          # 3 super-blocks (256 values each)
N = 256 * N_BLOCKS

rng = np.random.RandomState(7)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unit", "kquant_fixture.bin")
with open(OUT, "wb") as f:
    f.write(np.uint32(len(TYPES)).tobytes())
    for name in TYPES:
        qtype = GGMLQuantizationType[name]
        # Random raw quantized bytes: the dequantizer is a pure function of
        # the block bytes, so any input validates the layout. Reference output
        # comes from the authoritative llama.cpp `gguf` dequantizer.
        _, type_size = 0, 0
        qbytes = rng.bytes(type_size)
        # type_size from the enum's quant size table
        from gguf.constants import GGML_QUANT_SIZES
        _, type_size = GGML_QUANT_SIZES[qtype]
        qbytes = rng.bytes(type_size * N_BLOCKS)
        expected = dequantize(np.frombuffer(qbytes, dtype=np.uint8), qtype).astype(np.float32)
        f.write(np.uint32(qtype.value).tobytes())
        f.write(np.uint64(N).tobytes())
        f.write(np.uint64(len(qbytes)).tobytes())
        f.write(qbytes)
        f.write(expected.tobytes())
        print(f"{name}: {len(qbytes)} bytes, first expected {expected[0]:.5f} {expected[1]:.5f}")
