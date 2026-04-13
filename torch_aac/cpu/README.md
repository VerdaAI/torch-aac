# `cpu/` — CPU Bitstream Assembly

Serial stages that produce the final AAC-LC ADTS byte stream. These run on CPU because bitstream packing is inherently sequential (each codeword's position depends on the cumulative length of all preceding codewords).

| Module | Role |
|--------|------|
| `bitstream.py` | **ADTS frame builder.** Assembles ICS headers, section data, scalefactor deltas, and spectral data into complete ADTS frames. Contains `BitWriter` (Python fallback) and `_CBitWriter` (deferred C packing). |
| `huffman.py` | Per-group Huffman encoding. Looks up codewords from the table, packs sign bits, emits escape codes for cb11 values > 15. Optimized with `tolist()` conversion and batched sign packing. |
| `scalefactor.py` | Scalefactor delta VLC encoding using the AAC scalefactor Huffman table. |
| `_bitwriter.c` | **C bit-packing extension.** JIT-compiled via `ctypes` on first import. Packs `(code, length)` arrays into bytes ~10× faster than Python. Falls back gracefully if no C compiler is available. |
| `_bitwriter_native.py` | Python loader for `_bitwriter.c`. Compiles the shared library, sets up ctypes argtypes, exposes `bitwriter_pack()`. |

## Performance

The C BitWriter was the single biggest throughput win: **2.3× speedup** over the Python accumulator. Combined with the batch GPU Huffman lookup, the full pipeline achieves ~97× realtime.

Before optimization, `write_bits()` consumed 55% of total encode time (1.1M Python function calls for 10s of audio). The C extension reduces this to a single bulk pack call per frame.

## Bitstream Structure

Each ADTS frame:
```
[ADTS Header — 7 bytes]
[ID_SCE (3 bits) + tag (4 bits)]
[ICS: global_gain + ics_info + section_data + scalefactor_data + pulse/tns/gain flags + spectral_data]
[ID_END (3 bits)]
[byte alignment padding]
```

PNS bands (cb=13) get noise scalefactor data instead of spectral data. See `_write_scalefactor_data()` for the dual delta-chain logic.
