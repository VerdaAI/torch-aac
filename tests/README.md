# `tests/` — Test Suite

52 tests covering the full encoder pipeline. Run with:

```bash
pytest              # all tests
pytest -v           # verbose
pytest tests/test_encoder.py  # specific file
```

| Test File | What it covers | Tests |
|-----------|---------------|-------|
| `test_filterbank.py` | MDCT/IMDCT roundtrip, framing, windowing, overlap-add reconstruction accuracy. | 10 |
| `test_encoder.py` | End-to-end encode → FFmpeg decode → amplitude and error checks. Mono, stereo, multiple sample rates and bitrates. | 10 |
| `test_differentiable.py` | `DifferentiableAAC` forward/backward, gradient flow, STE vs noise mode, rate_loss, training convergence. | 12 |
| `test_batch.py` | Batch encoding, multi-stream `encode_batch()`, various input shapes and channel layouts. | 10 |
| `test_psychoacoustic.py` | Masking threshold computation, SMR, ATH, bark-scale conversion, spreading function. | 10 |
| `conftest.py` | Shared fixtures (sample rates, test audio generation). | — |

## Notes

- Tests run on whatever device `auto` selects (CPU, MPS, or CUDA).
- `test_encoder.py` requires `ffmpeg` on PATH for decode verification.
- Tests are designed to be fast (~2-3 seconds total on CPU).
- No external audio files needed — all test signals are synthesized.
