# Version History

| Version | Date       | Summary |
|---------|------------|---------|
| 0.6.0   | 2026-04-17 | DifferentiableAAC short-block parity, 32kHz fix, encode determinism, regression test suite |
| 0.5.1   | 2026-04-16 | Fix short block bitstream corruption (missing pulse_data_present bit), PyTorch 2.10+ compat |
| 0.5.0   | 2026-04-16 | Fix SFB tables (5 sample rates), tighten rate control, add cubic quantization, optimize state machine |
| 0.4.0   | 2026-04-15 | Complete short blocks: pre-echo elimination (170+ dB), transition windows, tiled SFB quantization |
| 0.3.0   | 2026-04-14 | Short block infra (partial), CUDA validated (212x on RTX 3090), CI improvements |
| 0.2.0   | 2026-04-14 | torchaudio integration, Colab demo, CI/PyPI workflows, per-directory docs |
| 0.1.0   | 2026-04-13 | First public release — GPU-accelerated differentiable AAC-LC encoder, beats Apple AudioToolbox on SNR, ~97x realtime, CUDA/MPS/CPU |

---

## Detailed Changelog

See [CHANGELOG.md](CHANGELOG.md) for full details per release.
