# `benchmark/` — Performance and Quality Benchmarks

| Script | What it measures | Usage |
|--------|-----------------|-------|
| `bench_quality.py` | SNR, peak/RMS ratio, correlation across 8 signal types (sines, chord, music, speech, transient, sweep, noise) at multiple bitrates. Encodes with torch-aac, decodes with FFmpeg. | `python benchmark/bench_quality.py` |
| `bench_throughput.py` | Encoding speed (frames/sec, realtime factor) on the current device. Compares against FFmpeg's native AAC encoder. | `python benchmark/bench_throughput.py --quick` |

## Running a Full Benchmark

```bash
# Quality across all signals and bitrates
python benchmark/bench_quality.py

# Quality for specific signals/bitrates
python benchmark/bench_quality.py --signals sine_1khz noise --bitrates 128000 320000

# Throughput
python benchmark/bench_throughput.py --durations 10 60

# Quick throughput (10s only)
python benchmark/bench_throughput.py --quick
```

## Available Test Signals

`bench_quality.py` generates these synthetically (no external audio files needed):

| Signal | Description |
|--------|-------------|
| `sine_440` | Pure 440 Hz sine, 0.5 amplitude |
| `sine_1khz` | Pure 1 kHz sine |
| `chord` | 440 + 880 + 1320 Hz (3 harmonics) |
| `music_like` | 10-partial harmonic series at 220 Hz with 4 Hz AM envelope |
| `speech_like` | 120 Hz pulse train × 3 formant resonances + breath noise |
| `transient` | Click train with 2 kHz decaying sine resonances |
| `sweep` | Log-frequency chirp 100 Hz → 10 kHz |
| `noise` | White Gaussian noise, 0.3 amplitude |

For real-audio comparison, use `examples/compare_quality.py` with your own WAV files.
