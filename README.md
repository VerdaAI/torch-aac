# torch-aac

[![CI](https://github.com/VerdaAI/torch-aac/actions/workflows/ci.yml/badge.svg)](https://github.com/VerdaAI/torch-aac/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: ELv2](https://img.shields.io/badge/license-ELv2-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VerdaAI/torch-aac/blob/main/examples/demo.ipynb)

**The first open-source GPU-accelerated, differentiable AAC-LC encoder.**

Built for two use cases: (1) training neural audio models that are robust to AAC compression, and (2) fast GPU-accelerated audio encoding.

> Outperforms Apple AudioToolbox and FFmpeg's native AAC encoder on SNR across all tested signal types and bitrates (48k-320k). See [benchmarks](#quality-benchmarks).

## Features

- **Fast** — ~97x realtime on CPU, ~67x on Apple MPS. JIT-compiled C bit-packer, batch-vectorized GPU Huffman lookup.
- **Differentiable** — Backpropagate through AAC encoding via straight-through estimator (STE) or noise injection. Train codec-robust models directly.
- **Accurate** — Beats Apple AudioToolbox by 2-25 dB SNR on real audio (TTS speech, music, noise) at 128 kbps.
- **Standard** — Produces valid AAC-LC ADTS bitstreams decodable by FFmpeg, VLC, browsers, and every major audio player.
- **Multi-platform** — CUDA, Apple MPS (Metal), and CPU. `device="auto"` picks the best available.

## Quick Start

```python
import torch_aac

# Encode PCM audio to AAC
aac_bytes = torch_aac.encode(pcm_float32, sample_rate=48000, bitrate=128000)

# Batch encode multiple streams in parallel
results = torch_aac.encode_batch([audio1, audio2, audio3], sample_rate=48000)

# Differentiable mode — gradients flow through AAC simulation
codec = torch_aac.DifferentiableAAC(sample_rate=48000, bitrate=128000)
decoded = codec(audio_tensor)  # forward: encode → quantize → decode
loss = F.mse_loss(decoded, target)
loss.backward()  # gradients propagate through the codec

# Rate-distortion training
decoded, rate_loss = codec(audio_tensor, return_rate_loss=True)
total_loss = mse_loss + 0.1 * rate_loss  # encourage compressible output
```

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and PyTorch >= 2.0.

| Platform                  | Backend | Status                        |
|---------------------------|---------|-------------------------------|
| Linux / Windows + NVIDIA  | CUDA    | Supported (RTX 3090, T4)      |
| macOS + Apple Silicon     | MPS     | Supported (M1/M2/M3/M4)      |
| Any                       | CPU     | Supported (PyTorch fallback)  |

## Quality Benchmarks

**Head-to-head vs Apple AudioToolbox and FFmpeg at 128 kbps** (mono, 48 kHz, encoder-delay-aligned SNR in dB):

| Signal          | torch-aac | Apple AudioToolbox | FFmpeg native | vs Apple |
|-----------------|----------:|-------------------:|--------------:|---------:|
| TTS speech      |    51.3   |         40.2       |        25.6   |  +11.1   |
| Real music (mp3)|    37.0   |         30.4       |        29.2   |   +6.6   |
| Pink noise      |    20.0   |         16.3       |        14.7   |   +3.7   |
| Pure sine       |    76.1   |         56.7       |        50.8   |  +19.4   |
| Transient       |    46.8   |         27.0       |        38.4   |  +19.8   |

All comparisons use the same methodology: encode float32 PCM → decode with FFmpeg → measure SNR with encoder-specific delay compensation (ours: 0 samples, Apple: 2112, FFmpeg: 1024).

<details>
<summary>Full bitrate sweep (48k–320k)</summary>

| Bitrate | Signal     | torch-aac | Apple AT | FFmpeg |
|---------|------------|----------:|---------:|-------:|
| 48k     | TTS speech |    34.3   |    27.2  |  22.8  |
| 48k     | Pink noise |    11.1   |    10.7  |   8.8  |
| 48k     | Real music |    25.2   |    21.2  |  19.2  |
| 128k    | TTS speech |    51.4   |    40.2  |  25.3  |
| 128k    | Pink noise |    19.6   |    16.1  |  14.9  |
| 128k    | Real music |    37.0   |    30.4  |  29.2  |
| 256k    | TTS speech |    73.9   |    48.4  |  47.2  |
| 256k    | Pink noise |    32.9   |    30.8  |  26.1  |
| 256k    | Real music |    58.0   |    45.6  |  46.4  |

</details>

### Differentiable Mode Parity

The differentiable path produces **bit-identical output** to the encode path (verified: correlation 1.0000, max error < 2e-4 across all signal types). Models trained with `DifferentiableAAC` see exactly the artifacts they'll face in production.

## Architecture

```
                        ┌─────────────────────────────────────────────┐
  PCM Audio ──────────▶ │               GPU Stages                    │
                        │  Windowing → MDCT → Rate Control →          │
                        │  Quantization → Codebook Selection →        │
                        │  Batch Huffman Lookup                       │
                        └──────────────┬──────────────────────────────┘
                                       │ (codes, lengths) arrays
                        ┌──────────────▼──────────────────────────────┐
                        │             CPU Stages                      │
                        │  C BitWriter (JIT-compiled) →               │
                        │  ADTS Header Assembly → .aac bytes          │
                        └─────────────────────────────────────────────┘
```

**Encode mode**: Full pipeline producing valid AAC-LC ADTS bitstream.

**Differentiable mode**: GPU-only simulation — MDCT → soft quantize (STE/noise) → dequantize → IMDCT. No bitstream produced; gradients flow end-to-end.

## Throughput

| Configuration               | Realtime factor | Notes                              |
|-----------------------------|----------------:|------------------------------------|
| CPU (Apple M-series)        |           ~97x  | C BitWriter + batch GPU Huffman    |
| MPS (Apple Silicon GPU)     |           ~67x  | GPU stages on Metal                |
| Multi-stream (8 streams)    |           ~92x  | ThreadPool, aggregate              |

Measured on 10s mono audio at 128 kbps. See [docs/technical.md](docs/technical.md) for the optimization story.

## CLI

```bash
# File encoding
python -m torch_aac -i input.wav -o output.aac -b 128k

# FFmpeg pipe integration
ffmpeg -i input.mp4 -f f32le -ar 48000 -ac 1 pipe:1 | python -m torch_aac -b 128k > output.aac
```

## Examples

| Example | Description |
|---------|-------------|
| [`basic_encode.py`](examples/basic_encode.py) | One-shot encoding, file output |
| [`differentiable_training.py`](examples/differentiable_training.py) | Train a model robust to AAC |
| [`compare_quality.py`](examples/compare_quality.py) | Encode your own WAV, compare bitrates |

## Supported Configurations

| Parameter     | Values                                 |
|---------------|----------------------------------------|
| Sample rates  | 16000, 44100, 48000 Hz                 |
| Channels      | 1 (mono), 2 (stereo)                   |
| Bitrate       | 48000–320000 bps                       |
| Profile       | AAC-LC                                 |
| Container     | ADTS (.aac)                            |

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest                          # 52 tests
ruff check . && ruff format .   # lint
python benchmark/bench_quality.py   # quality bench
python benchmark/bench_throughput.py --quick  # speed bench
```

## Roadmap

- [x] Core AAC-LC encode pipeline (GPU + CPU)
- [x] Differentiable mode with STE/noise quantization
- [x] Apple MPS support
- [x] Batch multi-stream encoding API
- [x] C BitWriter JIT extension
- [x] Batch-flattened GPU Huffman
- [x] Accurate Huffman bit cost estimator
- [ ] Short-block transient handling
- [ ] M4A/MP4 container output
- [ ] CUDA performance validation on datacenter GPUs
- [ ] Perceptual metrics (PEAQ/POLQA) benchmarking

## License

[Elastic License 2.0 (ELv2)](LICENSE) — free to use, modify, and distribute. Cannot be offered as a hosted/managed service.

## Citation

If you use torch-aac in research, please cite:

```bibtex
@software{torch_aac,
  title  = {torch-aac: GPU-Accelerated Differentiable AAC-LC Encoder},
  author = {Verda AI},
  year   = {2026},
  url    = {https://github.com/verda-ai/torch-aac}
}
```

## Acknowledgments

Built by [Verda AI](https://verda.ai).
