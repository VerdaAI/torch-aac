# torch-aac

GPU-accelerated and differentiable AAC-LC encoder using PyTorch.

> **Status**: v0.1.0-alpha — produces valid AAC-LC ADTS bitstreams (currently silent frames; spectral encoding in progress)

## Features

- **GPU-accelerated**: Batch-encode thousands of AAC frames simultaneously on CUDA
- **Differentiable**: Backpropagate through AAC encoding for codec-robust model training (STE + noise injection modes)
- **Standard output**: Produces valid AAC-LC ADTS bitstreams decodable by FFmpeg, VLC, and browsers
- **CPU fallback**: Works without a GPU via PyTorch CPU mode
- **Psychoacoustic model**: Bark-scale masking with spreading function for perceptual bit allocation

## Quick Start

```python
import torch_aac

# Encode audio to AAC
aac_bytes = torch_aac.encode(pcm_float32, sample_rate=48000, bitrate=128000)

# Differentiable mode for training
codec = torch_aac.DifferentiableAAC(sample_rate=48000, bitrate=128000)
decoded = codec(audio_tensor)  # gradients flow through
loss.backward()
```

## Install

```bash
pip install -e ".[dev]"
```

Requires PyTorch >= 2.0. Works on macOS (CPU), Linux (CPU/CUDA), with CUDA GPUs (RTX 3090, T4 tested).

## CLI

```bash
# File-based encoding
python -m torch_aac -i input.wav -o output.aac -b 128k

# Pipe mode (FFmpeg integration)
ffmpeg -i input.mp4 -f f32le -ar 48000 -ac 1 pipe:1 | python -m torch_aac -b 128k > output.aac
```

## Architecture

```
Audio → [GPU] Windowing → MDCT → Psychoacoustic Model → Quantization → Rate Control → Codebook Selection
      → [CPU] Huffman Packing → ADTS Bitstream → .aac file
```

Two modes:
- **Encode mode**: Full pipeline producing valid AAC-LC bytes
- **Differentiable mode**: GPU-only simulation with gradient flow (MDCT → soft quantize → dequantize → IMDCT)

## Current State (v0.1.0)

| Component | Status |
|-----------|--------|
| MDCT / IMDCT (GPU) | Done, verified against reference |
| Psychoacoustic model (GPU) | Done (Bark scale, spreading function, ATH) |
| Quantizer (GPU, 3 modes) | Done (hard, STE, noise injection) |
| Rate control (GPU parallel) | Done (binary search on global_gain) |
| Codebook selection (GPU) | Done |
| ADTS bitstream writer (CPU) | Done, FFmpeg-decodable |
| Differentiable mode | Done, gradients verified |
| Huffman tables | Stubs (V2 blocker for audible output) |
| Short blocks / block switching | V2 |
| M4A container | V2 |

### What works
- 52 tests passing (filterbank, encoder, differentiable, psychoacoustic, batch)
- Mono + stereo encoding at 16kHz, 44.1kHz, 48kHz
- Bitrates 48k–320k
- FFmpeg decodes output successfully
- Gradients flow through differentiable AAC simulation
- CLI with file and pipe modes

### Known limitations (v0.1.0)
- Output is valid but **silent** (max_sfb=0) — Huffman tables needed for spectral encoding
- MDCT uses matrix multiply (O(N²)) — FFT-based approach planned for V2
- No M4A/MP4 container support (ADTS only)
- Long blocks only (no short block transient handling)

## Supported Configurations

| Parameter | Values |
|-----------|--------|
| Sample rates | 16000, 44100, 48000 Hz |
| Channels | 1 (mono), 2 (stereo) |
| Bitrate | 48000–320000 bps |
| Profile | AAC-LC only |
| Container | ADTS (.aac) |
| Target GPUs | RTX 3090 (24GB), GCP T4 (16GB) |

## Development

```bash
# Setup
uv venv && uv pip install -e ".[dev]"

# Tests
pytest

# Lint
ruff check .

# Type check
mypy torch_aac/
```

## License

[Elastic License 2.0 (ELv2)](LICENSE) — free to use, modify, and distribute. You may not offer this software as a hosted/managed service.

## Acknowledgments

Built by [Verda AI](https://verda.ai).
