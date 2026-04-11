# torch-aac

GPU-accelerated and differentiable AAC-LC encoder using PyTorch.

> **Status**: v0.1.0 — produces valid AAC-LC ADTS bitstreams with correct audio content. Near-transparent reconstruction of tonal signals (60–73 dB SNR at 48–320 kbps), decodable by FFmpeg/VLC/browsers.

## Features

- **GPU-accelerated**: Batch-encode thousands of AAC frames simultaneously on CUDA
- **Differentiable**: Backpropagate through AAC encoding for codec-robust model training (STE + noise injection modes)
- **Standard output**: Produces valid AAC-LC ADTS bitstreams decodable by FFmpeg, VLC, and browsers
- **CPU fallback**: Works without a GPU via PyTorch CPU mode

## Quick Start

```python
import torch_aac

# Encode audio to AAC
aac_bytes = torch_aac.encode(pcm_float32, sample_rate=48000, bitrate=128000)

# Differentiable mode for training
codec = torch_aac.DifferentiableAAC(sample_rate=48000, bitrate=128000)
decoded = codec(audio_tensor)  # gradients flow through
loss.backward()

# Rate-aware training: penalize hard-to-compress outputs
decoded, rate_loss = codec(audio_tensor, return_rate_loss=True)
total_loss = reconstruction_loss + 0.1 * rate_loss
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
Audio → [GPU] Windowing → MDCT → Quantization → Rate Control → Codebook Selection
      → [CPU] Huffman Packing → ADTS Bitstream → .aac file
```

Two modes:
- **Encode mode**: Full pipeline producing valid AAC-LC bytes
- **Differentiable mode**: GPU-only simulation with gradient flow (MDCT → soft quantize → dequantize → IMDCT)

## Quality (v0.1.0)

Quality benchmarks vs FFmpeg decode roundtrip, 1 s mono 48 kHz:

| Signal      | 48 kbps          | 128 kbps         | 320 kbps         |
|-------------|------------------|------------------|------------------|
| sine 440 Hz | 63.9 dB SNR      | 73.0 dB SNR      | 73.2 dB SNR      |
| sine 1 kHz  | 59.8 dB SNR      | 72.0 dB SNR      | 72.8 dB SNR      |
| chord       | 54.8 dB SNR      | 70.4 dB SNR      | 72.1 dB SNR      |
| white noise |  1.9 dB SNR      |  7.0 dB SNR      | 26.7 dB SNR      |

Peak and RMS amplitude ratios are ~1.00 across all cases; correlation is 1.00 for tonal signals. White noise at low bitrate is the main limitation (see below).

## Current State (v0.1.0)

| Component                          | Status                                 |
|------------------------------------|----------------------------------------|
| MDCT / IMDCT (GPU, matmul basis)   | Done, verified against reference       |
| Quantizer (GPU, 3 modes)           | Done (hard, STE, noise injection)      |
| Rate control (GPU parallel)        | Done (binary search on global_gain)    |
| Codebook selection (GPU)           | Done (cb 1–11)                         |
| Full Huffman tables (cb 1–11)      | Done                                   |
| ADTS bitstream writer (CPU)        | Done, FFmpeg-decodable                 |
| Differentiable mode + rate loss    | Done, gradients verified, converges    |
| Psychoacoustic model               | Stub (uniform sf) — V2                 |
| Short blocks / block switching     | V2                                     |
| M4A container                      | V2                                     |

### What works
- 52 tests passing (filterbank, encoder, differentiable, psychoacoustic, batch)
- Mono + stereo encoding at 16 kHz, 44.1 kHz, 48 kHz
- Bitrates 48k–320k
- FFmpeg decodes output with correct amplitude and SNR
- Gradients flow through differentiable AAC simulation; training converges
- CLI with file and pipe modes

### Known limitations (v0.1.0)
- **Noise-like content at low bitrate** (< 64 kbps) reconstructs poorly. The encoder uses a uniform scalefactor per frame (no psychoacoustic model yet), so broadband noise gets under-allocated. A bark-scale masking model is planned for v0.2.
- MDCT uses matrix multiply (O(N²)) — FFT-based approach planned for V2
- No M4A/MP4 container support (ADTS only)
- Long blocks only (no short block transient handling)

## Supported Configurations

| Parameter     | Values                                 |
|---------------|----------------------------------------|
| Sample rates  | 16000, 44100, 48000 Hz                 |
| Channels      | 1 (mono), 2 (stereo)                   |
| Bitrate       | 48000–320000 bps                       |
| Profile       | AAC-LC only                            |
| Container     | ADTS (.aac)                            |
| Target GPUs   | RTX 3090 (24 GB), GCP T4 (16 GB)       |

## Development

```bash
# Setup
uv venv && uv pip install -e ".[dev]"

# Tests
pytest

# Quality bench
python benchmark/bench_quality.py

# Lint
ruff check .

# Type check
mypy torch_aac/
```

## License

[Elastic License 2.0 (ELv2)](LICENSE) — free to use, modify, and distribute. You may not offer this software as a hosted/managed service.

## Acknowledgments

Built by [Verda AI](https://verda.ai).
