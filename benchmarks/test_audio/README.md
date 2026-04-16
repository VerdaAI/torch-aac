# Test Audio Files for Reproducing Benchmark Issues

## problematic/

Files where torch-aac produces degraded or corrupt output:

| File | Source | Issue |
|------|--------|-------|
| `1272-128104-0005.flac` | LibriSpeech (16kHz) | Corrupt bitstream at 192k (v1 and v2) |
| `4507-16021-0047.flac` | LibriSpeech (16kHz) | Negative SNR at all bitrates, corrupt at 192k |
| `1995-1836-0004.flac` | LibriSpeech (16kHz) | Negative SNR at all bitrates, corrupt at 192k |
| `cristina_vane_mix_30s.wav` | musdb18 (44.1kHz stereo, extracted as 48kHz mono) | Negative SNR at all bitrates in v1 |

Common pattern: 16kHz speech upsampled to 48kHz, and resampled music content.

## good/

Files where torch-aac performed well:

| File | Source | Result |
|------|--------|--------|
| `01-AchGottundHerr.wav` | Bach10 (44.1kHz mono) | +6 to +11 dB SNR advantage over FFmpeg |

## Reproducing

```bash
# Run benchmark on just the problematic files
python benchmarks/real_audio_benchmark.py --data-root benchmarks/test_audio/problematic --device cpu

# Run on the good files
python benchmarks/real_audio_benchmark.py --data-root benchmarks/test_audio/good --device cpu
```
