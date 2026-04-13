# `tables/` — AAC Specification Tables

Static lookup tables extracted from the AAC-LC specification (ISO 14496-3) and FFmpeg's reference implementation.

| Module | Contents |
|--------|----------|
| `huffman_tables.py` | All 11 AAC Huffman codebooks (cb1-11) as Python dicts mapping value tuples → (codeword, bit_length). Scalefactor VLC table (121 entries). Index-to-values conversion with MSB-first ordering. ~3000 lines. |
| `sfb_tables.py` | Scalefactor band offset tables for each supported sample rate (16k, 44.1k, 48kHz). Defines band boundaries for long blocks. |
| `window_tables.py` | Sine window coefficients for MDCT windowing. Cached per window length + device. |

## Huffman Table Ordering

AAC Huffman codebooks use MSB-first ordering for mapping flat indices to value tuples. For a codebook with dimension `d` and base `b`:

```
index → values[0..d-1]
values[0] = (index // b^(d-1)) % b - offset    (MSB)
values[d-1] = index % b - offset                (LSB)
```

This was verified against FFmpeg's `ff_aac_codebook_vector_idx` tables. Getting this wrong produces structurally valid but amplitude-incorrect bitstreams (the decoder reads values in the wrong order within each group).

## Adding New Sample Rates

To add a sample rate: add its SFB offset table to `sfb_tables.py` and register it in the `SFB_LONG` dict. The offsets must match the AAC spec's Table 4.85 for the target sample rate. Cross-check against FFmpeg's `swb_offset_1024_*` arrays in `aactab.c`.
