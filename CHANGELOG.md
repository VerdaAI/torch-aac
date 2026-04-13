# Changelog

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
