# `gpu/` — GPU-Accelerated Stages

All compute-heavy encoder stages run here via PyTorch tensor operations. These modules work on CUDA, MPS (Apple Silicon), and CPU.

| Module | Role | Key Functions |
|--------|------|---------------|
| `filterbank.py` | MDCT / IMDCT via cosine basis matmul. Framing, windowing, overlap-add. | `mdct()`, `imdct()`, `frame_audio()`, `apply_window()` |
| `quantizer.py` | AAC non-uniform quantizer (`\|x\|^{3/4} / step`). Four modes: hard, STE (straight-through), noise injection, cubic soft-rounding. Density-adaptive bit cost estimator. | `quantize()`, `quantize_per_band()`, `estimate_bit_count()` |
| `rate_control.py` | GPU-parallel binary search for optimal global gain per frame. Also contains experimental per-band SF allocation. | `find_global_gain()`, `compute_scalefactors()` |
| `huffman_select.py` | Selects optimal Huffman codebook (cb1-11) per scalefactor band based on max quantized value. Supports `pairs_only` mode for GPU Huffman. | `select_codebooks()` |
| `huffman_encode.py` | **Batch-flattened GPU Huffman.** All groups from all frames are processed in one GPU gather. Returns `(codes, lengths)` numpy arrays for C packing. | `encode_spectral_batched()` |
| `psychoacoustic.py` | Bark-scale masking model: spreading function, ATH, per-band SMR. Used by experimental `find_scalefactors()`. | `compute_masking_thresholds()`, `compute_smr()` |
| `pns.py` | Perceptual Noise Substitution detection and energy computation. Calibrated (C=64) but disabled by default. | `detect_noise_bands()`, `compute_noise_energy_sf()` |
| `block_switch.py` | Transient detection via energy ratios + window sequence state machine (ONLY_LONG→LONG_START→EIGHT_SHORT→LONG_STOP). | `detect_transients()`, `get_window_sequence()` |
| `ms_stereo.py` | Mid/Side stereo per-band decision and transform. Energy-ratio criterion, disabled by default. | `compute_ms_mask()`, `apply_ms_transform()` |

## Key Design Decisions

- **Unnormalized MDCT**: The matmul `X = frames @ basis` skips the standard `2/N` normalization. This is compensated by `SF_OFFSET=164` in the quantizer. See [docs/technical.md](../../docs/technical.md).
- **Accurate bit estimator**: `estimate_bit_count()` uses a 8193-entry lookup table built from real Huffman code lengths, replacing a rough formula that overestimated by 1.2-2.5×.
- **MAX_QUANTIZED_VALUE=8191**: The full AAC escape code range (N≤8), not 4095 (which was only the max mantissa).
- **Pairs-only mode**: When the batch GPU Huffman path is active, codebook selection restricts to pair codebooks (cb5-11), making group dimension uniformly 2 for the fixed-layout tensor.
