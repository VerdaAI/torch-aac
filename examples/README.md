# `examples/` — Usage Examples

| Example | Description | Key API |
|---------|-------------|---------|
| [`basic_encode.py`](basic_encode.py) | Encode a sine wave to AAC, write to file. Shows both the one-shot `torch_aac.encode()` and the `AACEncoder` class API. | `encode()`, `AACEncoder` |
| [`differentiable_training.py`](differentiable_training.py) | Train a small MLP to produce audio that survives AAC compression. Demonstrates gradient flow through `DifferentiableAAC` and the optional `rate_loss`. | `DifferentiableAAC`, `return_rate_loss` |
| [`compare_quality.py`](compare_quality.py) | Encode **your own WAV file** at multiple bitrates, decode with FFmpeg, report SNR and write comparison WAVs for listening. The first thing to run when evaluating torch-aac on your content. | `encode()`, FFmpeg decode |

## Quick Start

```bash
# Basic encode
python examples/basic_encode.py

# Compare quality on your audio
python examples/compare_quality.py path/to/your/audio.wav

# Differentiable training demo
python examples/differentiable_training.py
```

## Requirements

- `ffmpeg` on PATH (for `compare_quality.py` decoding)
- PyTorch >= 2.0
- No GPU required (CPU works, MPS/CUDA faster)
