=== needle2 verification (WSL2, recorded 2026-08-14) ===

Build: make needle2 (gcc -std=c11 -O2 -Wall -Wextra) -> examples/needle/needle2
         cmake --build build-linux --target needle2 (RelWithDebInfo)
Token-exactness: generated ids match the JAX reference decode
  (decode.py generate_cached over the same dequantized .cact weights)
  for all 24 tested tokens.
CTest: needle2_selftest passes (Windows 5/5, WSL 51/51).

--- valgrind run ---
=== needle2 valgrind (full model, WSL2) ===
Fri Aug 14 16:32:48 CST 2026
command: OMP_NUM_THREADS=1 valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --error-exitcode=42 ./needle2 /mnt/d/hugginface/needle2/needle2.cact --prompt "what is the weather in Lagos right now?" --max-new-tokens 32
==1514== Memcheck, a memory error detector
==1514== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==1514== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info
==1514== Command: ./needle2 /mnt/d/hugginface/needle2/needle2.cact --prompt what\ is\ the\ weather\ in\ Lagos\ right\ now? --max-new-tokens 32
==1514== 
model: needle2-san 0.1 (boat)
prompt: what is the weather in Lagos right now?
","arguments":{"location":"Lagos"}}]</tool_call><|im_end|>
---
","arguments":{"location":"Lagos"}}]</tool_call><|im_end|>
==1514== 
==1514== HEAP SUMMARY:
==1514==     in use at exit: 0 bytes in 0 blocks
==1514==   total heap usage: 9,055 allocs, 9,055 frees, 198,278,648 bytes allocated
==1514== 
==1514== All heap blocks were freed -- no leaks are possible
==1514== 
==1514== For lists of detected and suppressed errors, rerun with: -s
==1514== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
---
