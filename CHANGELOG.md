# Changelog

## v0.4.0 (2026-04-15)

Complete short block support for pre-echo reduction on transient audio.

### Added
- **Short block bitstream**: Full EIGHT_SHORT_SEQUENCE encoding with per-window quantization using tiled SFB offsets (8 × 14 = 112 bands mapped across 1024 coefficients), per-group codebook selection, and per-group scalefactor delta chains.
- **Transition windows**: LONG_START and LONG_STOP window functions (ISO 14496-3 Section 11.2.1.1) applied per-frame during MDCT for correct overlap-add at block boundaries.
- **Pre-echo elimination**: Short blocks reduce pre-echo by 170–186 dB on impulse signals — effectively zero leakage before transients. This is the primary perceptual benefit: a click that smeared across 21 ms (long block) is now confined to 2.67 ms (short block).
- **Tiled SFB offsets**: `get_sfb_offsets_short_tiled()` utility maps 8 short windows' scalefactor bands onto the flattened 1024-coefficient representation, enabling the existing GPU quantizer and rate control pipeline to process short blocks without reshaping.
- **20 new tests** in `test_short_blocks.py`: transient detection, block switching state machine, transition windows, tiled SFB offsets, mono/stereo FFmpeg decode, and pre-echo reduction measurement.

### Changed
- Encoder now splits batches into long and short subsets, processing each with appropriate SFB offsets. Short blocks use the CPU Huffman path; long blocks continue to use the fast GPU Huffman path when available.
- `apply_window()` accepts an optional `window_sequence` tensor and applies per-frame transition windows for LONG_START (ws=1) and LONG_STOP (ws=3).
- Bitstream writer uses per-group codebook indices for short blocks instead of tiling the same codebooks across all 8 groups.
- Stereo CPE now passes `window_sequence` through to both ICS writers.

### Fixed
- Removed the `win_seq = torch.zeros_like(win_seq)` override that forced all frames to LONG blocks regardless of transient detection.

---

## v0.3.0 (2026-04-14)

Short blocks, CUDA validation, and CI improvements.

### Added
- **Short block infrastructure** (partial): transient detection via energy ratios, sequential window sequence state machine (ONLY_LONG → LONG_START → EIGHT_SHORT → LONG_STOP), short MDCT (8 x 128-point), short SFB tables for all sample rates, bitstream writer scaffolding for 3-bit section lengths and scale_factor_grouping. **Note:** bitstream still writes all frames as LONG blocks because short block scalefactor layout (14 bands x 8 groups) is not yet integrated with the quantizer. Pre-echo reduction is not active yet — completing this is the v0.4.0 priority.
- **CUDA benchmarks**: Validated on RTX 3090 (212× realtime, 570 MB VRAM) and RTX 3080 Ti (206× realtime). Correctness verified (peak ratio ~1.0), differentiable gradients confirmed. Benchmark script at `benchmarks/cuda_validation.py`.

### Performance
- **212× realtime on RTX 3090** (60s mono 128 kbps) — up from ~97× on CPU.
- **1.2–3.4× faster than FFmpeg** native AAC on CUDA, depending on clip length.
- **203× aggregate** for 8-stream batch on RTX 3090.

### Fixed
- CI smoke tests folded into test job (avoids redundant ~2 GB PyTorch install, ~3 min faster).
- Notebook lint errors (import order, semicolons, one-line ifs) and formatting.
- PyTorch 2.10+ compatibility (`total_mem` → `total_memory` attribute).

---

## v0.2.0 (2026-04-14)

Integrations, tooling, and documentation release.

### Added
- **torchaudio integration**: `AACSimulation` differentiable transform for training pipelines, `AACEncode` for evaluation, `save_aac()` / `load_aac()` with torchaudio-like API. Compatible with `torch.nn.Sequential`.
- **Colab demo notebook**: `examples/demo.ipynb` with Open in Colab badge. Interactive demo of encoding, quality benchmarks, differentiable training, and speed measurement.
- **CI workflow**: Lint (ruff), test matrix (Python 3.10/3.11/3.12), encode smoke test (amplitude assertion), differentiable gradient check, batch encode verification. Concurrency control.
- **PyPI publish workflow**: Trusted publishing via OIDC on GitHub Release. Builds sdist + wheel with C source included.
- **`/release` command**: Slash command for version management — bumps version in `__init__.py` + `pyproject.toml`, generates changelog, tags, pushes.
- **Per-directory READMEs**: Every package directory (`gpu/`, `cpu/`, `tables/`, `benchmark/`, `examples/`, `tests/`, `docs/`) has a README explaining contents and architecture.
- **VERSIONS.md**: Quick-reference version history table.
- **CI badges**: CI status, Python version, license, PyTorch, Open in Colab.

### Changed
- PyPI package URLs updated to `github.com/VerdaAI/torch-aac`.
- Hatch build config: explicit sdist/wheel includes, C source force-included in wheel.

---

## v0.1.0 (2026-04-13)

First public release. GPU-accelerated, differentiable AAC-LC encoder.

### Highlights

- **Quality**: Outperforms Apple AudioToolbox and FFmpeg on SNR across all tested signal types (speech, music, noise, tones) at all bitrates (48k-320k).
- **Speed**: ~97× realtime on CPU, ~67× on Apple MPS.
- **Differentiable**: Bit-identical simulation of the encode path with gradient flow for training codec-robust models.

### Encode Pipeline
- MDCT via GPU matmul with pre-computed cosine basis
- AAC non-uniform quantizer (hard, STE, noise injection modes)
- GPU-parallel binary search rate control
- Batch-flattened GPU Huffman codebook lookup (cb1-11)
- JIT-compiled C BitWriter via ctypes (auto-compiles, Python fallback)
- ADTS bitstream assembly with full AAC-LC spec compliance
- Accurate Huffman bit cost estimator from real codebook code lengths

### Differentiable Mode
- `DifferentiableAAC` PyTorch module: MDCT → soft quantize → dequantize → IMDCT
- Bit-identical parity with encode path (correlation 1.0000)
- Differentiable `rate_loss` for rate-distortion training
- STE and noise injection quantization modes

### Multi-Platform
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon via Metal)
- CPU fallback
- `device="auto"` selection

### API
- `torch_aac.encode()` — one-shot encoding
- `torch_aac.encode_batch()` — parallel multi-stream encoding
- `torch_aac.encode_file()` — WAV file encoding
- `torch_aac.DifferentiableAAC` — differentiable codec module
- `torch_aac.AACEncoder` — encoder class with full control
- CLI: `python -m torch_aac -i input.wav -o output.aac`

### Quality
- 52 passing tests (filterbank, encoder, differentiable, psychoacoustic, batch)
- ruff lint + format clean
- Benchmarked against Apple AudioToolbox and FFmpeg native AAC

### Known Limitations
- Long blocks only (no short-block transient handling)
- ADTS container only (no M4A/MP4)
- PNS (Perceptual Noise Substitution) infrastructure complete but disabled by default
- Psychoacoustic per-band SF allocation experimental (not on default path)
