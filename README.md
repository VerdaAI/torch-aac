# torch-aac

[![CI](https://github.com/VerdaAI/torch-aac/actions/workflows/ci.yml/badge.svg)](https://github.com/VerdaAI/torch-aac/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/torch-aac.svg)](https://pypi.org/project/torch-aac/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: ELv2](https://img.shields.io/badge/license-ELv2-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VerdaAI/torch-aac/blob/main/examples/demo.ipynb)

**The first open-source GPU-accelerated, differentiable AAC-LC encoder.**

Built for two use cases: (1) training neural audio models that are robust to AAC compression, and (2) fast GPU-accelerated audio encoding.

> Beats FFmpeg's native AAC encoder by +2 to +12 dB SNR on real audio (speech, music) across all tested bitrates. 200x realtime on RTX 3090.

## Install

```bash
pip install torch-aac
```

Or install from source:

```bash
git clone https://github.com/VerdaAI/torch-aac.git
cd torch-aac
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, PyTorch >= 2.0. GPU optional (CUDA or Apple MPS); falls back to CPU automatically.

| Platform                  | Backend | Status                        |
|---------------------------|---------|-------------------------------|
| Linux / Windows + NVIDIA  | CUDA    | Supported (RTX 3090, T4, etc) |
| macOS + Apple Silicon     | MPS     | Supported (M1/M2/M3/M4)      |
| Any                       | CPU     | Supported (PyTorch fallback)  |

## Quick Start

```python
import torch_aac

# Encode a WAV file to AAC
torch_aac.encode_file("input.wav", "output.aac", sample_rate=48000, bitrate=128000)

# Encode PCM audio to AAC bytes
aac_bytes = torch_aac.encode(pcm_float32, sample_rate=48000, bitrate=128000)

# Encode multiple streams in parallel
results = torch_aac.encode_batch([audio1, audio2, audio3], sample_rate=48000)
```

### Differentiable Mode

Train neural audio models robust to AAC compression:

```python
codec = torch_aac.DifferentiableAAC(sample_rate=48000, bitrate=128000)
decoded = codec(audio_tensor)  # forward: encode -> quantize -> decode
loss = F.mse_loss(decoded, target)
loss.backward()  # gradients flow through the AAC simulation

# Three quantization modes:
# "ste"   — faithful forward, surrogate backward (recommended)
# "noise" — stochastic approximation, real gradient
# "cubic" — smooth surrogate, real gradient (good for warm-up)
```

### Stereo Encoding

```python
# Stereo with per-channel rate control
enc = torch_aac.AACEncoder(sample_rate=48000, channels=2, bitrate=128000)
aac_bytes = enc.encode(stereo_pcm)  # shape: (samples, 2) or (2, samples)
```

## Features

- **Fast** — 200x realtime on RTX 3090, ~98x on Apple MPS. GPU/CPU split for mixed long+short blocks. JIT-compiled C bit-packer.
- **Differentiable** — Backpropagate through AAC via STE, noise injection, or cubic soft-rounding. Includes short-block simulation for transient-faithful training.
- **Accurate** — Beats FFmpeg on speech (+12 dB), music (+2 dB), and stereo content (22-32 dB on real music).
- **Standard** — Valid AAC-LC ADTS bitstreams. Passes FFmpeg strict decode (`-xerror`) at all sample rates (8-96 kHz).
- **Stereo** — Per-channel rate control with energy-proportional budget. M/S stereo infrastructure (opt-in).
- **Short blocks** — Transient detection, block switching (LONG_START/EIGHT_SHORT/LONG_STOP), transition windows per ISO 14496-3.
- **Multi-platform** — CUDA, Apple MPS, CPU. `device="auto"` picks the best available.

## Quality Benchmarks

### Real Audio (RTX 3090, cross-correlation aligned)

| Dataset | torch-aac vs FFmpeg | Wins |
|---|---|---|
| **Speech** (LibriSpeech) | **+12.4 dB** avg | 15/15 |
| **Music** (musdb18) | **+2.3 dB** avg | 23/30 |
| **Stereo music** (musdb18 refs) | 22-32 dB SNR | Strict decode OK |

### Synthetic Signals (128 kbps, mono, 48 kHz)

| Signal          | torch-aac | FFmpeg native | Delta |
|-----------------|----------:|--------------:|------:|
| 1 kHz sine      |    73.5   |        44.8   | +28.7 |
| Chord (440+880) |    71.5   |        42.7   | +28.8 |
| Speech formants |    72.0   |        45.9   | +26.1 |
| Noise           |    10.2   |         7.1   |  +3.1 |

### Differentiable Mode Parity

STE simulation matches real encode-decode with 78.6 dB parity on tones and 9.4 dB on impulses (short-block IMDCT reconstruction).

## Throughput

| Device                   | Realtime factor | Notes                              |
|--------------------------|---------------:|------------------------------------|
| **RTX 3090 (CUDA)**     |         200x   | GPU/CPU split for mixed blocks     |
| **RTX 3080 Ti (CUDA)**  |         197x   | Same pipeline                      |
| Apple MPS (M-series)     |          98x   | GPU stages on Metal                |
| CPU                      |       40-60x   | C BitWriter + CPU Huffman          |

Peak VRAM: 570 MB for 60s encode. Mixed long+short batches: 1.7% overhead vs pure-GPU.

## Architecture

```
                        +---------------------------------------------+
  PCM Audio ----------> |               GPU Stages                    |
                        |  Windowing -> MDCT -> Transient Detection ->|
                        |  Rate Control -> Quantization ->            |
                        |  Codebook Selection -> GPU Huffman Lookup    |
                        +--------------+------------------------------+
                                       | (codes, lengths) arrays
                        +--------------v------------------------------+
                        |             CPU Stages                      |
                        |  C BitWriter (JIT-compiled) ->              |
                        |  ADTS Header Assembly -> .aac bytes         |
                        +---------------------------------------------+
```

**Encode mode**: Full pipeline producing valid AAC-LC ADTS bitstream with short-block support and per-channel stereo.

**Differentiable mode**: GPU-only simulation — MDCT (long + short) -> soft quantize (STE/noise/cubic) -> dequantize -> IMDCT. No bitstream produced; gradients flow end-to-end.

## Supported Configurations

| Parameter     | Values                                                       |
|---------------|--------------------------------------------------------------|
| Sample rates  | 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000 Hz |
| Channels      | 1 (mono), 2 (stereo)                                         |
| Bitrate       | 48000-320000 bps                                             |
| Profile       | AAC-LC                                                       |
| Container     | ADTS (.aac)                                                  |
| Quant modes   | hard, ste, noise, cubic                                      |

## Known Limitations

- **No psychoacoustic per-band allocation** — uses uniform scalefactors. Efficient at low-mid bitrates; FFmpeg's per-band allocator is better at 192k+ on dense polyphonic content.
- **M/S stereo disabled by default** — infrastructure complete but the noise doubling from R=M-S reconstruction degrades wideband content. Enable via `encoder._enable_ms = True` for dual-mono content.
- **No TNS** (Temporal Noise Shaping) — short blocks handle most transient cases.
- **ADTS only** — no M4A/MP4 container output. Use FFmpeg to remux: `ffmpeg -i output.aac -c copy output.m4a`
- **Not thread-safe** — MDCT basis cache is global. Use separate processes for parallel encoding.

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

## Development

```bash
git clone https://github.com/VerdaAI/torch-aac.git
cd torch-aac
pip install -e ".[dev]"
pytest                          # 110 tests (strict decode, quality baselines, rate control)
ruff check . && ruff format .   # lint + format
```

## License

[Elastic License 2.0 (ELv2)](LICENSE) — free to use, modify, and distribute. Cannot be offered as a hosted/managed service.

## Citation

If you use torch-aac in research, please cite:

```bibtex
@software{torch_aac,
  title  = {torch-aac: GPU-Accelerated Differentiable AAC-LC Encoder},
  author = {Verda AI},
  year   = {2026},
  url    = {https://github.com/VerdaAI/torch-aac}
}
```

## Acknowledgments

Built by [Verda AI](https://verda.ai).
